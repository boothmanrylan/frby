from __future__ import division
from __future__ import print_function

import argparse
import functools
import itertools
import os

import models
import utils
from dataset import Dataset
import numpy as np
import six
from six.moves import xrange
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def get_model_fn(num_gpus, variable_strategy, num_workers):
    def _resnet_model_fn(features, labels, mode, params):
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        weight_decay = params.weight_decay
        momentum = params.momentum

        tower_features = features
        tower_labels = labels
        tower_losses = []
        tower_gradvars = []
        tower_preds = []

        data_format = params.data_format
        if not data_format:
            if num_gpus == 0:
                data_format = 'channels_last'
            else:
                data_format = 'channels_first'

        if num_gpus == 0:
            num_devices = 1
            device_type = 'cpu'
        else:
            num_devices = num_gpus
            device_type = 'gpu'

        for i in range(num_devices):
            worker_device = '/{}:{}'.format(device_type, i)
            if variable_strategy == 'CPU':
                device_setter = utils.local_device_setter(
                        worker_device=worker_device)
            elif variable_strategy == 'GPU':
                device_setter = utils.local_device_setter(
                    ps_device_type='gpu',
                    worker_device=worker_device,
                    ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                        num_gpus, tf.contrib.training.byte_size_load_fn))
            with tf.variable_scope('resnet', reuse=bool(i != 0)):
                with tf.name_scope('tower_%d' % i) as name_scope:
                    with tf.device(device_setter):
                        loss, gradvars, preds = _tower_fn(
                                is_training, weight_decay, tower_features[i],
                                tower_labels[i], data_format,
                                params.num_layers, params.batch_norm_decay,
                                params.batch_norm_epsilon)
                        tower_losses.append(loss)
                        tower_gradvars.append(gradvars)
                        tower_preds.append(preds)
                        if i == 0:
                            update_ops = tf.get_collection(
                                    tf.GraphKeys.UPDATE_OPS, name_scope)
        gradvars = []
        with tf.name_scope('gradient_averaging'):
            all_grads = {}
            for grad, var in itertools.chain(*tower_gradvars):
                if grad is not None:
                    all_grads.setdefault(var, []).append(grad)
            for var, grads in six.iteritems(all_grads):
                with tf.device(var.device):
                    if len(grads) == 1:
                        avg_grad = grads[0]
                    else:
                        avg_grad = tf.multiply(tf.add_n(grads),
                                1. / len(grads))
                gradvars.append((avg_grad, var))

        if variable_strategy == 'GPU':
            consolidation_device = '/gpu:0'
        else:
            consolidation_device = '/cpu:0'
        with tf.device(consolidation_device):
            num_batches_per_epoch = (Dataset.examples_per_epoch(True) //
                (params.train_batch_size * num_workers))
            boundaries = list(num_batches_per_epoch *
                                np.array([82, 23, 300], dtype=np.int64))
            staged_lr = list(params.learning_rate *
                                np.array([1, .1, .01, .002], dtype=np.int64))

            learning_rate = tf.train.piecewise_constant(
                    tf.train.get_global_step(), boundaries, staged_lr)

            loss = tf.reduce_mean(tower_losses, name='loss')

            examples_sec_hook = utils.ExamplesPerSecondHook(
                    params.train_batch_size, every_n_steps=10)

            tensors_to_log = {'learning_rate': learning_rate, 'loss': loss}
            logging_hook = tf.train.LoggingTensorHook(
                    tensors=tensors_to_log, every_n_iter=100)

            train_hooks = [logging_hook, examples_sec_hook]

            optimizer = tf.train.MomentumOptimizer(
                    learning_rate=learning_rate, momentum=momentum)

            if params.sync:
                optimizer = tf.train.SyncReplicasOptimizer(optimizer,
                        replicas_to_aggregate=num_workers)
                sync_replicas_hook = optimizer.make_session_run_hook(
                        params.is_chief)
                train_hooks.append(sync_replicas_hook)

            train_op = [optimizer.apply_gradients(gradvars,
                global_step=tf.train.get_global_step())]
            train_op.extend(update_ops)
            train_op = tf.group(*train_op)

            predictions = {
                'classes':
                    tf.concat([p['classes'] for p in tower_preds], axis=0),
                'probabilities':
                    tf.concat([p['probabilities'] for p in tower_preds], axis=0)
                    }
            stacked_labels = tf.concat(labels, axis=0)
            metrics = {
                    'accuracy': tf.metrics.accuracy(stacked_labels,
                        predictions['classes'])
                    }
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op,
                training_hooks=train_hooks,
                eval_metric_ops=metrics)
    return _resnet_model_fn

def _tower_fn(is_training, weight_decay, feature, label, data_format,
              num_layers, batch_norm_decay, batch_norm_epsilon):
    model = models.ResNet(num_layers, batch_norm_decay=batch_norm_decay,
                          batch_norm_epsilon=batch_norm_epsilon,
                          is_training=is_training,
                          data_format=data_format)
    logits = model.forward_pass(feature, input_data_format='channels_last')
    tower_pred = {'classes': tf.argmax(input=logits, axis=1),
                  'probabilities': tf.nn.softmax(logits)}

    tower_loss = tf.losses.sparse_softmax_cross_entropy(logits=logits,
                                                        labels=label)
    tower_loss = tf.reduce_mean(tower_loss)

    model_params = tf.trainable_variables()
    tower_loss += weight_decay * tf.add_n(
            [tf.nn.l2_loss(v) for v in model_params])

    tower_grad = tf.gradients(tower_loss, model_params)

    return tower_loss, zip(tower_grad, model_params), tower_pred

def input_fn(file_pattern, training, num_shards, batch_size):
    with tf.device('/cpu:0'):
        dset = Dataset(file_pattern, training)
        data_batch, label_batch = dset.make_batch(batch_size)

        if num_shards <= 1:
            return [data_batch], [label_batch]

        data_batch = tf.unstack(data_batch, num=batch_size, axis=0)
        label_batch = tf.unstack(label_batch, num=batch_size, axis=0)
        feature_shards = [[] for i in range(num_shards)]
        label_shards = [[] for i in range(num_shards)]
        for i in xrange(batch_size):
            idx = i % num_shards
            feature_shards[idx].append(data_batch[i])
            label_shards[idx].append(label_batch[i])
        feature_shards = [tf.parallel_stack(x) for x in feature_shards]
        label_shads = [tf.parallel_stack(x) for x in label_shards]

        return feature_shards, label_shards

def get_experiment_fn(train_pattern, eval_pattern, num_gpus, variable_strategy):
    def _experiment_fn(run_config, hparams):
        train_input_fn = functools.partial(
                input_fn,
                train_pattern,
                training=True,
                num_shards=num_gpus,
                batch_size=hparams.train_batch_size)

        eval_input_fn = functools.partial(
                input_fn,
                eval_pattern,
                training=False,
                num_shards=num_gpus,
                batch_size=hparams.eval_batch_size)

        num_eval_examples = Dataset.examples_per_epoch(False)

        if num_eval_examples % hparams.eval_batch_size != 0:
            msg = 'validation set size must be multiple of eval_batch_size'
            raise ValueError(msg)

        train_steps = hparams.train_steps
        eval_steps = num_eval_examples // hparams.eval_batch_size

        classifier = tf.estimator.Estimator(
                model_fn=get_model_fn(num_gpus, variable_strategy,
                    run_config.num_worker_replicas or 1),
                config=run_config,
                params=hparams)

        return tf.contrib.learn.Experiment(
                classifier,
                train_input_fn=train_input_fn,
                eval_input_fn=eval_input_fn,
                train_steps=train_steps,
                eval_steps=eval_steps)
    return _experiment_fn

def main(job_dir, train_pattern, eval_pattern, num_gpus, variable_strategy,
         log_device_placement, num_intra_threads, **hparams):
    os.environ['TF_SYNC_FINISH'] = '0'
    os.environ['TF_ENABLE_WINOGRAD_NOFUSED'] = '1'

    sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=log_device_placement,
            intra_op_parallelism_threads=num_intra_threads,
            gpu_options=tf.GPUOptions(force_gpu_compatible=True))

    config = tf.contrib.learn.RunConfig(session_config=sess_config,
                                    model_dir=job_dir)
    tf.contrib.learn.learn_runner.run(
            get_experiment_fn(train_pattern, eval_pattern, num_gpus,
                              variable_strategy),
            run_config=config,
            hparams=tf.contrib.training.HParams(is_chief=config.is_chief,
                                                **hparams))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--train-pattern',
            type=str,
            required=True,
            help='File pattern matching files containing training data')
    parser.add_argument(
            '--eval-pattern',
            type=str,
            required=True,
            help='File pattern matching files containing evaluation data')
    parser.add_argument(
            '--job-dir',
            type=str,
            required=True,
            help='The directory where the model will be stored')
    parser.add_argument(
            '--variable-strategy',
            choices=['CPU', 'GPU'],
            type=str,
            default='CPU',
            help='Where to locate variable operations')
    parser.add_argument(
            '--num-gpus',
            type=int,
            default=1,
            help='The number of gpus used, Uses only CPU if set to 0.')
    parser.add_argument(
            '--num-layers',
            type=int,
            default=44,
            help='The number of layers in the model.')
    parser.add_argument(
            '--train-steps',
            type=int,
            default=80000,
            help='The number of steps to use for training.')
    parser.add_argument(
            '--train-batch-size',
            type=int,
            default=128,
            help='Batch size for training.')
    parser.add_argument(
            '--eval-batch-size',
            type=int,
            default=100,
            help='Batch size for validation.')
    parser.add_argument(
            '--momentum',
            type=float,
            default=0.9,
            help='Momentum for MomentumOptimizer.')
    parser.add_argument(
            '--weight-decay',
            type=float,
            default=2e-1,
            help='Weight decay for convolutions.')
    parser.add_argument(
            '--learning-rate',
            type=float,
            default=0.1,
            help="Initial learning rate")
    parser.add_argument(
            '--sync',
            action='store_true',
            default=False,
            help="""\
                 If present when running in a distributed environment will run
                 on sync mode.\
                 """)
    parser.add_argument(
        '--num_intra-threads',
        type=int,
        default=0,
        help="""\
             Number of threads to use for intra-op parallelism. When
             training on CPU set to 0 to have the system pick the
             appropriate number or alternatively set it to the number of
             physical CPU cores.\
             """)
    parser.add_argument(
        '--num-inter-threads',
        type=int,
        default=0,
        help="""\
             Number of theards to use for inter-op parallelism. If set to 0,
             the system will pick an appropriate number.\
             """)
    parser.add_argument(
        '--data-format',
        type=str,
        default=None,
        help="""\
             If not set, the data format best for the training device is
             used. Allowed values: channels_first (NCHW) channels_last
             (NHWC).\
             """)
    parser.add_argument(
        '--log-device-placement',
        action='store_true',
        default=False,
        help='Whether to log device placement.')
    parser.add_argument(
        '--batch-norm-decay',
        type=float,
        default=0.997,
        help='Decay for batch norm.')
    parser.add_argument(
        '--batch-norm-epsilon',
        type=float,
        default=1e-5,
        help='Epsilon for batch norm.')
    args = parser.parse_args()

    if args.num_gpus > 0:
        assert tf.test.is_gpu_available(), "Requested GPUs but none found."
    if args.num_gpus < 0:
        msg = 'Invalid GPU count: \"--num-gpus\" must be 0 or a positive int.'
        raise ValueError(msg)
    if args.num_gpus == 0 and args.variable_strategy == 'GPU':
        raise ValueError('num-gpus=0, CPU must be used as parameter server.'
                         '--variable-strategy=CPU')
    if (args.num_layers - 2) % 6 != 0:
        raise ValueError('(--num-layers - 2) % 6 must equal zero.')
    if args.num_gpus != 0 and args.train_batch_size % args.num_gpus != 0:
        raise ValueError('--train-batch-size must be multiple of --num-gpus.')
    if args.num_gpus !=0 and args.eval_batch_size % args.num_gpus != 0:
        raise ValueError('--eval-batch-size must be multiple of --num-gpus.')

    main(**vars(args))

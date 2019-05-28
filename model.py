import functools
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, MobileNet, VGG16
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow import contrib
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

autograph = contrib.autograph

tf.logging.set_verbosity(tf.logging.INFO)

SHAPE = (32, 38, 1)
CLASSES = 3

summary = ('accuracy', 'mean_absolute_error', 'Model Type', 'Slurm Job ID')

tf.app.flags.DEFINE_integer('buffer_size', 100,
                            'Size of buffer used for shuffling input data')
tf.app.flags.DEFINE_integer('num_gpus', 4,
                            'Number of GPUs used for training and testing')
tf.app.flags.DEFINE_integer('batch_size', 64,
                            'batch size, will be multiplied by NUM_GPUS')
tf.app.flags.DEFINE_integer('train_steps', 100,
                            'Number of steps used during training')
tf.app.flags.DEFINE_integer('eval_steps', 10000,
                            'Number of steps used during testing')
tf.app.flags.DEFINE_string('train_pattern',
                           '/scratch/r/rhlozek/rylan/tfrecords/train*',
                           'Unix file pattern pointing to training records')
tf.app.flags.DEFINE_string('eval_pattern',
                           '/scratch/r/rhlozek/rylan/tfrecords/evaluate*',
                           'Unix file pattern pointing to test records')
tf.app.flags.DEFINE_string('val_pattern',
                           '/scratch/r/rhlozek/rylan/tfrecords/validate*',
                           'Unix file pattern pointing to validation records')
tf.app.flags.DEFINE_string('checkpoint_path',
                           '/scratch/r/rhlozek/rylan/models/default',
                           'Directory where model checkpoints will be stored')
tf.app.flags.DEFINE_string('base_model', 'vgg',
                           'Keras application to use as the base model')
tf.app.flags.DEFINE_string('summary_file',
                           '/scratch/r/rhlozek/rylan/results_summary.csv',
                           'File to store model comparisons in after each run')
tf.app.flags.DEFINE_integer('identifier', 1, 'Slurm job submission number')
tf.app.flags.DEFINE_integer("seed", 12345, "Seed for reproducibility")

FLAGS = tf.app.flags.FLAGS


def parse_fn(example, classification):
    example_fmt = {
        'height':     tf.FixedLenFeature([], tf.int64),
        'width':      tf.FixedLenFeature([], tf.int64),
        'channels':   tf.FixedLenFeature([], tf.int64),
        'colorspace': tf.FixedLenFeature([], tf.string),
        'label':      tf.FixedLenFeature([], tf.int64),
        'text_label': tf.FixedLenFeature([], tf.string),
        'format':     tf.FixedLenFeature([], tf.string),
        'filename':   tf.FixedLenFeature([], tf.string),
        'image':      tf.FixedLenFeature([], tf.string),
        'dm':         tf.FixedLenFeature([], tf.float32)
    }

    parsed = tf.parse_single_example(example, example_fmt)
    image = tf.image.decode_image(parsed['image'], channels=1)
    image = tf.cast(image, tf.float32)
    image.set_shape(SHAPE)

    if classification:
        label = tf.one_hot(parsed['label'], CLASSES)
    else:
        label = tf.reshape(parsed['dm'], [1])

    return {'image': image, 'label': label}, label


def input_fn(pattern, classification, repeat=True):
    """
    Not 100% necessary to shuffle here because the samples were shuffled during
    the creation of the TFRecords, therefore we stop shuffling here so that the
    data output by input_fn is always in the same order allowing it to be
    passed to estimators and their results easily compared sample by sample.
    """
    records = tf.data.Dataset.list_files(pattern, shuffle=False)
    dataset = records.interleave(tf.data.TFRecordDataset, cycle_length=4)
    # dataset = dataset.shuffle(buffer_size=FLAGS.buffer_size)
    dataset = dataset.map(map_func=lambda x: parse_fn(x, classification))
    dataset = dataset.batch(FLAGS.batch_size * FLAGS.num_gpus)
    if repeat:
        dataset = dataset.repeat()
    return dataset


def get_base_model(model_name):
    model_name = model_name.lower()
    if model_name == "mobilenet":
        return MobileNet
    elif model_name == "vgg":
        return VGG16
    elif model_name == "densenet121":
        return DenseNet121
    elif model_name == "densenet169":
        return DenseNet169
    elif model_name == "densenet201":
        return DenseNet201
    else:
        return ResNet50


def summarize(dict): # TODO: proper ordering of columns and headers
    if not os.path.exists(FLAGS.summary_file):
        empty_results = {k:[] for k in sorted(summary)}
        empty_results = pd.DataFrame.from_dict(empty_results)
        empty_results.to_csv(FLAGS.summary_file, header=True)
    dict["Slurm Job ID"] = FLAGS.identifier
    dict["Model Type"] = FLAGS.base_model
    dict = {
        k:[v] for k,v in sorted(dict.items(), key=lambda x: x[0]) if k in summary
    }
    results = pd.DataFrame.from_dict(dict)
    with open(FLAGS.summary_file, 'a') as f:
        results.to_csv(f, header=False)


class TemperatureScaler(tf.train.SessionRunHook):
    def __init__(self, temp_var, model_fn, params, checkpoint_dir, input_fn):
        self.logits_tensor = tf.placeholder(tf.float32, name='logits_placeholder')
        self.labels_tensor = tf.placeholder(tf.float32, name='labels_placeholder')
        #TODO: logits_tensor is "logits_placeholder:0" feed_dict needs replica_x/logits_placeholder
        # this issue does not exist if the training is not distributed

        self.optim = tf.contrib.opt.ScipyOptimizerInterface(
            tf.losses.softmax_cross_entropy(
                onehot_labels=self.labels_tensor,
                logits=tf.divide(self.logits_tensor, temp_var)
            ),
            options={'maxiter': 100},
            var_list=[temp_var]
        )

        self.estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            params=params,
            model_dir=checkpoint_dir
        )

        self.input_fn = input_fn

    def end(self, session):
        preds = list(self.estimator.predict(self.input_fn, yield_single_examples=False))
        fd = {self.labels_tensor: np.vstack([p['labels'] for p in preds]),
              self.logits_tensor: np.vstack([p['logits'] for p in preds])}
        self.optim.minimize(session, feed_dict=fd)
        #TODO: temp gets updated but its value does not get passed to temp_var
        # inside the model


def model_fn(features, labels, mode, params):
    """
    features is now a dictionary with two keys: images, and labels. Therefore
    labels is passed to the function twice, the reason for this is that the
    estimator predict function will always toss labels out and we need labels
    in order to properly do temperature scaling
    see: https://github.com/tensorflow/tensorflow/issues/17824
    """
    base_model = get_base_model(params["model_name"])

    x = base_model(include_top=False, input_shape=SHAPE, weights=None,
                   pooling='max')(features['image'])
    logits = Dense(params['n_classes'], activation=None)(x)

    temp_var = tf.get_variable(
        "temp",
        shape=[1],
        initializer=tf.initializers.constant(1.5),
        aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        if params['n_classes'] > 1:
            loss = tf.losses.softmax_cross_entropy(labels, logits)
        else:
            loss = tf.losses.mean_squared_error(labels, logits)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        hook = TemperatureScaler(temp_var, model_fn, params,
                                 FLAGS.checkpoint_path + '/classifier',
                                 params["validation_input"])
        hooks = [hook] if params["temperature_scaling"] else None
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op,
                                          training_hooks=hooks)
    elif mode == tf.estimator.ModeKeys.PREDICT:
        if params['n_classes'] > 1:
            predicted_classes = tf.argmax(logits, 1)
            predictions = {
                'labels': features['label'],
                'predicted_class': predicted_classes,
                'logits': logits,
                'probs': tf.divide(logits, temp_var),
                'temp': temp_var.value()
            }
        else:
            predictions = {
                'true_dm': features['label'],
                'predicted_dm': logits
            }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    elif mode == tf.estimator.ModeKeys.EVAL:
        if params['n_classes'] > 1:
            loss = tf.losses.softmax_cross_entropy(labels, logits)
            predictions = tf.argmax(logits, axis=1)
            decoded_labels = tf.argmax(labels, axis=1)
            accuracy = tf.metrics.accuracy(labels=decoded_labels,
                                           predictions=predictions,
                                           name='acc_op')
            metrics = {'accuracy': accuracy}
            tf.summary.scalar('Accuracy', accuracy[1])
        else:
            loss = tf.losses.mean_squared_error(labels, logits)
            mse = tf.metrics.mean_squared_error(labels, logits)
            mae = tf.metrics.mean_absolute_error(labels, logits)
            metrics = {'mean_squared_error': mse,
                       'mean_absolute_error': mae}
            tf.summary.scalar('Mean Absolute Error', mae[1])
            tf.summary.scalar('Mean Squared Error', mse[1])
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)


def main(argv=None):
    # log model parameters:
    for flag in FLAGS.flag_values_dict():
        print("{}:\t{}".format(flag, FLAGS[flag].value))

    devices = ["/gpu:{}".format(x) for x in range(FLAGS.num_gpus)]
    mirror = tf.distribute.MirroredStrategy(devices)

    config = tf.estimator.RunConfig(train_distribute=mirror,
                                    eval_distribute=mirror,
                                    tf_random_seed=FLAGS.seed)

    val_input = functools.partial(input_fn, pattern=FLAGS.val_pattern,
                                  classification=True, repeat=False)
    params={'n_classes': CLASSES,
            'model_name': FLAGS.base_model,
            'validation_input': val_input,
            'temperature_scaling': False} # should be True once issues are fixed
    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.checkpoint_path + '/classifier',
        params=params,
        config=config
    )

    params['n_classes'] = 1
    params['temperature_scaling'] = False
    regressor = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.checkpoint_path + '/regressor',
        params=params,
        config=config
    )

    clf_train_input = functools.partial(
        input_fn,
        pattern=FLAGS.train_pattern,
        classification=True,
        repeat=True)
    classifier.train(clf_train_input, steps=FLAGS.train_steps)

    reg_train_input = functools.partial(
        input_fn,
        pattern=FLAGS.train_pattern,
        classification=False,
        repeat=True)
    regressor.train(reg_train_input, steps=FLAGS.train_steps)

    clf_eval_input = functools.partial(
        input_fn,
        pattern=FLAGS.eval_pattern,
        classification=True,
        repeat=False)
    clf_results = classifier.evaluate(clf_eval_input, steps=FLAGS.eval_steps)
    clf_preds = list(classifier.predict(clf_eval_input, yield_single_examples=False))

    reg_eval_input = functools.partial(
        input_fn,
        pattern=FLAGS.eval_pattern,
        classification=False,
        repeat=False)
    reg_results = regressor.evaluate(reg_eval_input, steps=FLAGS.eval_steps)
    reg_preds = list(regressor.predict(reg_eval_input, yield_single_examples=False))

    print('classification results: {}'.format(clf_results))
    print('regression results: {}'.format(reg_results))

    summarize({**clf_results, **reg_results})

    results = {
        'True Class': [np.argmax(p['labels']) for p in clf_preds],
        'Predicted Class': [p['predicted_class'][0] for p in clf_preds],
        'True DM': [p['true_dm'][0][0] for p in reg_preds],
        'Predicted DM': [p['predicted_dm'][0][0] for p in reg_preds]
    }
    results_df = pd.DataFrame(results)
    results_df.to_csv(FLAGS.checkpoint_path + '/results.csv', index=False)


if __name__ == '__main__':
    tf.app.run()

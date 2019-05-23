import functools
import numpy as np
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

tf.app.flags.DEFINE_integer('buffer_size', 100,
                            'Size of buffer used for shuffling input data')
tf.app.flags.DEFINE_integer('num_gpus', 4,
                            'Number of GPUs used for training and testing')
tf.app.flags.DEFINE_integer('batch_size', 64,
                            'batch size, will be multiplied by NUM_GPUS')
tf.app.flags.DEFINE_integer('train_steps', 10000,
                            'Number of steps used during training')
tf.app.flags.DEFINE_integer('eval_steps', 10000,
                            'Number of steps used during testing')
tf.app.flags.DEFINE_string('train_pattern',
                           '/scratch/r/rhlozek/rylan/tfrecords/train-*',
                           'Unix file pattern pointing to training records')
tf.app.flags.DEFINE_string('eval_pattern',
                           '/scratch/r/rhlozek/rylan/tfrecords/evaluate-*',
                           'Unix file pattern pointing to test records')
tf.app.flags.DEFINE_string('validation_pattern',
                           '/scratch/r/rhlozek/rylan/tfrecords/validate-*',
                           'Unix file pattern pointing to validation records')
tf.app.flags.DEFINE_string('checkpoint_path',
                           '/scratch/r/rhlozek/rylan/models/defualt',
                           'Directory where model checkpoints will be stored')
tf.app.flags.DEFINE_string('base_model', 'resnet',
                           'Keras application to use as the base model')
tf.app.flags.DEFINE_boolean('classification', True,
                            'Whether to classify samples or predict dm')
tf.app.flags.DEFINE_integer("seed", 12345, "Seed for reproducibility")

FLAGS = tf.app.flags.FLAGS


def temp_scaling(logits, labels, sess, maxiter=50):
    temp_var = tf.get_variable("temp", shape=[1], initializer=tf.initializers.constant(1.5))

    logits_tensor = tf.constant(logits, name='logits_valid', dtype=tf.float32)
    labels_tensor = tf.constant(labels, name='labels_valid', dtype=tf.float32)

    logits_w_temp = tf.divide(logits_tensor, temp_var)

    nll_loss_op = tf.losses.softmax_cross_entropy(onehot_labels=labels_tensor, logits=logits_w_temp)
    org_nll_loss_op = tf.identity(nll_loss_op)

    optim = tf.contrib.opt.ScipyOptimizerInterface(nll_loss_op, options={'maxiter': maxiter})

    sess.run(temp_var.initializer)
    sess.run(tf.local_variables_initializer())
    org_nll_loss = sess.run(org_nll_op)
    optim.minimize(sess)
    nll_loss = sess.run(nll_loss_op)
    temperature = sess.run(temp_var)

    return temperature


def get_prediction_probabilities(validation_dataset, eval_dataset, params, temp=None):
    val_images, val_labels = validation_dataset.make_one_shot_iterator().get_next()
    images, labels = eval_dataset.make_one_shot_iterator().get_next()

    val_predictions = model_fn(val_images, val_labels,
                               tf.estimator.ModeKeys.EVAL, params).predictions
    predictions = model_fn(images, labels,
                           tf.estimator.ModeKeys.EVAL, params).predictions

    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

        if temp is None: # use temp_scaling to find temperature
            val_logits = []
            val_labels = []
            while True:
                try:
                    preds, lbls = sess.run([val_predictions, val_labels])
                    val_logits += preds['logits']
                    val_labels += lbls
                except tf.errors.OutOfRangeError:
                    break

            temp = temp_scaling(val_logits, val_labels, sess)

        eval_logits = []
        eval_labels = []
        while True:
            try:
                preds, lbls = sess.run([predictions, labels])
                eval_logits += preds['logits']
                eval_labels += lbls
            except tf.errors.OutOfRangeError:
                break

        logits = np.asarray(logits) / temp
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        predictions = np.argmax(probs, axis=1)

        accuracy = tf.metrics.accuracy(labels, predictions)
        confusion_matrix = tf.confusion_matrix(np.argmax(labels, axis=1),
                                               np.argmax(predictions, axis=1))

        output = {
            'logits': logits,
            'probs': probs,
            'class_preds': predictions,
            'accuracy': accuracy,
            'confusion_matrix': confusion_matrix
        }
        return predictions, probs, eval_labels


def parse_fn(example):
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

    if FLAGS.classification:
        label = tf.one_hot(parsed['label'], CLASSES)
    else:
        label = parsed['dm']

    return image, label


def input_fn(pattern, repeat=True):
    records = tf.data.Dataset.list_files(pattern)
    dataset = records.interleave(tf.data.TFRecordDataset, cycle_length=4)
    dataset = dataset.shuffle(buffer_size=FLAGS.buffer_size)
    dataset = dataset.map(map_func=parse_fn)
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


def model_fn(features, labels, mode, params):
    base_model = get_base_model(params["model_name"])
    model = tf.keras.Sequential([ # this might need to be converted to the functional api
        base_model(include_top=False, input_shape=SHAPE, weights=None),
        Flatten(name='flatten'),
        Dense(4096, activation='relu', name='dense1'),
        Dense(4096, activation='relu', name='dense2'),
        Dense(params['n_classes'], activation=None, name='logits')
    ])

    if mode == tf.estimator.ModeKeys.TRAIN:
        logits = model(features, training=True)
        optimizer = tf.train.AdamOptimizer()
        if params['n_classes'] > 1:
            loss = tf.losses.softmax_cross_entropy(labels, logits)
        else:
            loss = tf.losses.mean_squared_error(labels, logits)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    elif mode == tf.estimator.ModeKeys.PREDICT:
        logits = model(features, training=False)
        if params['n_classes'] > 1:
            predicted_classes = tf.argmax(logits, 1)
            predictions = {
                'class_ids': predicted_classes[:, tf.newaxis],
                'logits': logits
            }
        else:
            predictions = logits
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    elif mode == tf.estimator.ModeKeys.EVAL:
        logits = model(features, training=False)
        if params['n_classes'] > 1:
            loss = tf.losses.softmax_cross_entropy(labels, logits)
        else:
            loss = tf.losses.mean_squared_error(labels, logits)
        if params['n_classes'] > 1:
            predictions = tf.argmax(logits, axis=1)
            decoded_labels = tf.argmax(labels, axis=1)
            accuracy = tf.metrics.accuracy(labels=decoded_labels,
                                           predictions=predictions,
                                           name='acc_op')
            # eval_metric_ops need to return an update_op
            # confusion_matrix = tf.confusion_matrix(decoded_labels,
            #                                        predictions)
            metrics = {'accuracy': accuracy}
                       #'confusion_matrix': confusion_matrix}
            tf.summary.scalar('accuracy', accuracy[1])
        else:
            mse = tf.keras.metrics.mean_squared_error(labels, logits)
            mae = tf.keras.metrics.mean_absolute_error(labels, logits)
            metrics = {'mean_squared_error': mse,
                       'mean_absolute_error': mae}
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)


def main(argv=None):
    # log model parameters:
    for flag in FLAGS.flag_values_dict():
        print("{}:\t{}".format(flag, FLAGS[flag].value))

    devices = ["/gpu:{}".format(x) for x in range(FLAGS.num_gpus)]
    mirror = tf.distribute.MirroredStrategy(devices)

    config = tf.estimator.RunConfig(train_distribute=mirror,
                                    eval_distribute=mirror,
                                    model_dir=FLAGS.checkpoint_path,
                                    tf_random_seed=FLAGS.seed)

    params={'n_classes': CLASSES if FLAGS.classification else 1,
            'model_name': FLAGS.base_model}
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        params=params,
        config=config
    )

    train_input = functools.partial(input_fn, pattern=FLAGS.train_pattern, repeat=True)
    estimator.train(train_input, steps=FLAGS.train_steps)

    # val_dataset = input_fn(FLAGS.validation_pattern, repeat=False)
    # eval_dataset = input_fn(FLAGS.eval_pattern, repeat=False)

    # results = get_prediction_probabilities(val_dataset, eval_dataset, params)
    # for key, value in results.items():
    #     print(key, value)

    eval_input = functools.partial(input_fn, pattern=FLAGS.eval_pattern, repeat=False)
    results = estimator.evaluate(eval_input, steps=FLAGS.eval_steps)
    print(results)

if __name__ == '__main__':
    tf.app.run()

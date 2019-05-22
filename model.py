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
tf.app.flags.DEFINE_integer('train_steps', 100,
                            'Number of steps used during training')
tf.app.flags.DEFINE_integer('test_steps', 10000,
                            'Number of steps used during testing')
tf.app.flags.DEFINE_string('train_pattern',
                           '/scratch/r/rhlozek/rylan/tfrecords/train-00010*',
                           'Unix file pattern pointing to training records')
tf.app.flags.DEFINE_string('test_pattern',
                           '/scratch/r/rhlozek/rylan/tfrecords/val-0000*',
                           'Unix file pattern pointing to test records')
tf.app.flags.DEFINE_string('validation_pattern',
                           '/scratch/r/rhlozek/rylan/tfrecords/VAL*',
                           'Unix file pattern pointing to validation records')
tf.app.flags.DEFINE_string('checkpoint_path',
                           '/scratch/r/rhlozek/rylan/models/defualt',
                           'Directory where model checkpoints will be stored')
tf.app.flags.DEFINE_string('base_model', 'resnet',
                           'Keras application to use as the base model')
tf.app.flags.DEFINE_boolean('classification', True,
                            'Whether to classify samples or predict dm')

FLAGS = tf.app.flags.FLAGS


class LogitModel(tf.keras.Model):
    """
    Wrapper for an already defined model, when training returns the softmax of
    the output logits, but returns the logits themselves when testing to allow
    for temperature scaling post training.
    """
    def __init__(self, model):
        """
        model should be an already defined tf.keras.Model whose final layer
        returns logits not a class prediction.
        """
        super(LogitModel, self).__init__()
        self.model = model

    def call(self, inputs, training=False):
        logits = self.model(inputs)
        if training:
            return tf.nn.softmax(logits)
        else:
            return logits

def extract_labels(dataset):
    dataset = data_fn()
    iter = dataset.make_one_shot_iterator()
    image, label = iter.get_next()

    labels = []

    with tf.Session() as sess:
        try:
            while True:
                step_label = sess.run(label)
                labels.append(step_label)
        except tf.errors.OutOfRangeError:
            pass

    return labels


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

    return temp_var


def inverse_softmax(x):
    """
    Given an array of outputs of the softmax function return an array of inputs
    """
    return np.log((-x * (np.sum(np.exp(x), axis=1, keepdims=True) - np.exp(x))) / (x - 1))


@autograph.convert(recursive=True)
def predict(estimator, data, temperature):
    iter = data.make_one_shot_iterator()
    probabilities = []
    predictions = []
    accuracies = []
    autograph.set_element_type(probabilities, tf.float32)
    autograph.set_element_type(predictions, tf.int)
    autograph.set_element_type(accuracies, tf.float32)
    try:
        while True:
            x, y = iter.get_next()
            y_p = estimator.predict(x)
            scaled_y_p = tf.divide(y_p, temperature)
            prob = tf.softmax(scaled_y_p)
            pred = tf.argmax(prob, axis=1)
            acc = tf.keras.metrics.categorical_accuracy(y, y_p)
            probabilities.append(prob)
            accuracies.append(acc)
            predictions.append(pred)
    except tf.errors.OutOfRangeError:
        pass
    return autograph.stack(probabilities), autograph.stack(predictions), tf.reduce_mean(autograph.stack(accuracies))


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
    x = tf.feature_column.input_layer(features, params["feature_columns"])
    x = tf.layers.conv2D(x, 64, (3, 3), activation='relu', padding='same', name='block1_conv1')
    x = tf.layers.conv2D(x, 64, (3, 3), activation='relu', padding='same', name='block1_conv2')
    x = tf.layers.MaxPooling2D(x, (2, 2), strides=(2, 2) name='block1_pool')
    x = tf.layers.conv2D(x, 128 (3, 3), activation='relu', padding='same', name='block2_conv1')
    x = tf.layers.conv2D(x, 128 (3, 3), activation='relu', padding='same', name='block2_conv2')
    x = tf.layers.MaxPooling2D(x, (2, 2), strides=(2, 2), name='block2_pool')
    x = tf.layers.conv2D(x, 256 (3, 3), activation='relu', padding='same', name='block3_conv1')
    x = tf.layers.conv2D(x, 256 (3, 3), activation='relu', padding='same', name='block3_conv2')
    x = tf.layers.conv2D(x, 256 (3, 3), activation='relu', padding='same', name='block3_conv3')
    x = tf.layers.conv2D(x, 256 (3, 3), activation='relu', padding='same', name='block3_conv4')
    x = tf.layers.MaxPooling2D(x, (2, 2), strides=(2, 2), name='block3_pool')
    x = tf.layers.conv2D(x, 512, (3, 3), activation='relu', padding='same', name='block4_conv1')
    x = tf.layers.conv2D(x, 512, (3, 3), activation='relu', padding='same', name='block4_conv2')
    x = tf.layers.conv2D(x, 512, (3, 3), activation='relu', padding='same', name='block4_conv3')
    x = tf.layers.conv2D(x, 512, (3, 3), activation='relu', padding='same', name='block4_conv4')
    x = tf.layers.MaxPooling2D(x, (2, 2), strides=(2, 2), name='block4_pool')



def build_model():
    if FLAGS.classification:
        output_neurons = CLASSES
        activation = tf.nn.softmax
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']
    else:
        output_neurons = 1
        activation = None
        loss = 'mean_squared_error'
        metrics = ['mean_squared_error', 'mean_absolute_error']

    base = get_base_model(FLAGS.base_model)
    model = tf.keras.model.Sequential([
        base(include_top=False, weights=None, input_shape=SHAPE, pooling='max'),
        tf.keras.layers.Dense(output_neurons, activation=activation)
    ])

    mode.compile(loss=loss, optimizer=tf.train.AdamOptimizer(), metrics=metrics)

    return model


def main(argv=None):
    model = build_model()

    # log model parameters:
    for flag in FLAGS.flag_values_dict():
        print("{}:\t{}".format(flag, FLAGS[flag].value))

    # log model overview
    model.summary()

    devices = ["/gpu:{}".format(x) for x in range(FLAGS.num_gpus)]
    mirror = tf.distribute.MirroredStrategy(devices)

    config = tf.estimator.RunConfig(train_distribute=mirror,
                                    eval_distribute=mirror,
                                    model_dir=FLAGS.checkpoint_path,
                                    tf_random_seed=FLAGS.seed)

    estimator = tf.keras.estimator.model_to_estimator(model, config=config)

    train_input = functools.partial(input_fn, pattern=FLAGS.train_pattern, repeat=True)
    estimator.train(train_input, steps=FLAGS.train_steps)

    ##########################################################################
    ##########################################################################
    val_dataset = input_fn(FLAGS.validation_pattern, repeat=False)
    val_images, val_labels = val_dataset.make_one_shot_iterator().get_next()

    val_predictions = estimator.predict(val_images, val_labels)

    dataset = input_fn(FLAGS.test_pattern, repeat=False)
    images, labels = dataset.make_one_shot_iterator().get_next()

    predictions = estimator.predict(images, labels)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

        val_prediction_values = []
        val_label_values = []
        while True:
            try:
                preds, lbls = sess.run([val_predictions, val_labels])
                prediction_values += preds
                label_values += lbls
            except tf.errors.OutOfRangeError:
                break

        temp = temp_scaling(val_prediction_values, val_label_values, sess)

        prediction_values = []
        label_values = []
        while True:
            try:
                preds, lbls = sess.run([predictions, labels])
                prediction_values += preds
                label_values += lbls
            except tf.errors.OutOfRangeError:
                break
    ##########################################################################
    ##########################################################################

    # eval_data = functools.partial(input_fn, pattern=FLAGS.test_pattern, repeat=False)
    # probs, preds, acc = evaluate(estimator, eval_data, temperature)

    # print(probs)
    # print(preds)
    # print(acc)


if __name__ == '__main__':
    tf.app.run()

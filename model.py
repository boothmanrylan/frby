import functools
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, MobileNet, VGG16
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201

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
tf.app.flags.DEFINE_integer('test_steps', 1000,
                            'Number of steps used during testing')
tf.app.flags.DEFINE_string('train_pattern',
                           '/scratch/r/rhlozek/rylan/tfrecords/train*',
                           'Unix file pattern pointing to training records')
tf.app.flags.DEFINE_string('test_pattern',
                           '/scratch/r/rhlozek/rylan/tfrecords/val*',
                           'Unix file pattern pointing to test records')
tf.app.flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate')
tf.app.flags.DEFINE_integer('decay_steps', 1000,
                            'Number of steps until learning rate decays')
tf.app.flags.DEFINE_float('decay_rate', 0.96,
                          'Rate at which the learning rate decays')
tf.app.flags.DEFINE_string('checkpoint_path',
                           '/scratch/r/rhlozek/rylan/models/defualt',
                           'Directory where model checkpoints will be stored')
tf.app.flags.DEFINE_integer('seed', 1234, 'Seed for reproducibility between reruns')
tf.app.flags.DEFINE_string('base_model', 'resnet',
                           'Keras application to use as the base model')

FLAGS = tf.app.flags.FLAGS

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
        'image':      tf.FixedLenFeature([], tf.string)
    }

    parsed = tf.parse_single_example(example, example_fmt)
    image = tf.image.decode_image(parsed['image'], channels=1)
    image.set_shape(SHAPE)
    label = tf.one_hot(parsed['label'], CLASSES)
    return image, label

def input_fn(pattern):
    records = tf.data.Dataset.list_files(pattern)
    dataset = records.interleave(tf.data.TFRecordDataset, cycle_length=4)
    dataset = dataset.shuffle(buffer_size=FLAGS.buffer_size)
    dataset = dataset.map(map_func=parse_fn)
    dataset = dataset.batch(FLAGS.batch_size * FLAGS.num_gpus)
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


def main(argv=None):
    base = get_base_model(FLAGS.base_model)
    model = tf.keras.models.Sequential([
        base(include_top=False, weights=None, input_shape=SHAPE, pooling='max'),
        tf.keras.layers.Dense(CLASSES, activation=tf.nn.softmax)
    ])

    # log model parameters:
    for flag in FLAGS.flag_values_dict():
        print("{}:\t{}".format(flag, FLAGS[flag].value))

    # log model overview
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.train.AdamOptimizer(),
                  metrics=['accuracy'])

    devices = ["/gpu:{}".format(x) for x in range(FLAGS.num_gpus)]
    mirror = tf.distribute.MirroredStrategy(devices)

    config = tf.estimator.RunConfig(train_distribute=mirror,
                                    eval_distribute=mirror,
                                    model_dir=FLAGS.checkpoint_path,
                                    tf_random_seed=FLAGS.seed)

    estimator = tf.keras.estimator.model_to_estimator(model, config=config)

    train_input = functools.partial(input_fn, FLAGS.train_pattern)
    estimator.train(train_input, steps=FLAGS.train_steps)

    eval_input = functools.partial(input_fn, FLAGS.test_pattern)
    results = estimator.evaluate(eval_input, steps=FLAGS.test_steps)

    print(results)

if __name__ == '__main__':
    tf.app.run()

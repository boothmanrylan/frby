import functools
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, MobileNet, VGG16
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201

tf.logging.set_verbosity(tf.logging.INFO)

SHAPE = (32, 38, 1)
CLASSES = 3

summary_keys = ('Slurm Job ID', 'Model Type', 'accuracy')

tf.app.flags.DEFINE_integer('buffer_size', 100,
                            'Size of buffer used for shuffling input data')
tf.app.flags.DEFINE_integer('num_gpus', 4,
                            'Number of GPUs used for training and testing')
tf.app.flags.DEFINE_integer('batch_size', 64,
                            'batch size, will be multiplied by NUM_GPUS')
tf.app.flags.DEFINE_integer('train_steps', 10000,
                            'Number of steps used during training')
tf.app.flags.DEFINE_integer('test_steps', 10000,
                            'Number of steps used during testing')
tf.app.flags.DEFINE_string('train_pattern',
                           '/scratch/r/rhlozek/rylan/tfrecords/train*',
                           'Unix file pattern pointing to training records')
tf.app.flags.DEFINE_string('test_pattern',
                           '/scratch/r/rhlozek/rylan/tfrecords/val*',
                           'Unix file pattern pointing to test records')
tf.app.flags.DEFINE_string('checkpoint_path',
                           '/scratch/r/rhlozek/rylan/models/default',
                           'Directory where model checkpoints will be stored')
tf.app.flags.DEFINE_integer('seed', 1234, 'Seed for reproducibility')
tf.app.flags.DEFINE_string('base_model', 'resnet',
                           'Keras application to use as the base model')
tf.app.flags.DEFINE_boolean('classification', True,
                            'Whether to classify samples or predict dm')
tf.app.flags.DEFINE_string('summary_file',
                           '/scratch/r/rhlozek/rylan/results_summary.csv',
                           'File to store model comparisons in after each run')
tf.app.flags.DEFINE_integer('identifier', 1, 'Slurm job submission number')

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

def append_to_csv(data, file):
    if not os.path.isfile(file):
        data.to_csv(file, mode='a', index=False, sep=',')
    else:
        data.to_csv(file, mode='a', index=False, sep=',', header=False)

def summarize(results_dict):
    if not os.path.exists(FLAGS.summary_file):
        empty_results = {k:[] for k in summary_keys}
        empty_results = pd.DataFrame(empty_results)
        empty_results.to_csv(FLAGS.summary_file, header=True)
    results_dict["Slurm Job ID"] = FLAGS.identifier
    results_dict["Model Type"] = FLAGS.base_model
    results_dict = {k:[v] for k,v in results_dict.items() if k in summary_keys}
    results = pd.DataFrame(results_dict)
    with open(FLAGS.summary_file, 'a') as f:
        results.to_csv(f, header=False)

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
    model = tf.keras.models.Sequential([
        base(include_top=False, weights=None, input_shape=SHAPE, pooling='max'),
        tf.keras.layers.Dense(output_neurons, activation=activation)
    ])

    model.compile(loss=loss, optimizer=tf.train.AdamOptimizer(), metrics=metrics)

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

    train_input = functools.partial(input_fn, FLAGS.train_pattern)
    estimator.train(train_input, steps=FLAGS.train_steps)

    eval_input = functools.partial(input_fn, FLAGS.test_pattern)
    results = estimator.evaluate(eval_input, steps=FLAGS.test_steps)

    print(results)

    summarize(results)

if __name__ == '__main__':
    tf.app.run()

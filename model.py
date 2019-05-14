import functools
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.applications import ResNet50

BUFFER_SIZE = 100
SHAPE = (32, 38, 1)
NUM_GPUS = 4
BATCH_SIZE = 64 * NUM_GPUS
EPOCHS = 10
CLASSES = 3
TRAIN_STEPS = 100000
EVAL_STEPS = 1000

train_file_pattern = "/scratch/r/rhlozek/rylan/tfrecords/train*"
test_file_pattern =  "/scratch/r/rhlozek/rylan/tfrecords/val*"

tf.logging.set_verbosity(tf.logging.INFO)

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
    example_format = {}
    parsed = tf.parse_single_example(example, example_fmt)
    image = tf.image.decode_image(parsed['image'], channels=1)
    image.set_shape(SHAPE)
    label = tf.one_hot(parsed['label'], CLASSES)
    return image, label

def input_fn(file_pattern):
    records = tf.data.Dataset.list_files(file_pattern)
    dataset = records.interleave(tf.data.TFRecordDataset, cycle_length=4)
    dataset = dataset.shuffle(buffer_size=BUFFER_SIZE)
    dataset = dataset.map(map_func=parse_fn)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.repeat()
    return dataset

train_input = functools.partial(input_fn, file_pattern=train_file_pattern)
eval_input = functools.partial(input_fn, file_pattern=test_file_pattern)

model = tf.keras.models.Sequential([
    ResNet50(include_top=False, weights=None, input_shape=SHAPE, pooling='max'),
    Dense(CLASSES, activation=tf.nn.softmax)
])

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=tf.train.AdamOptimizer(),
              metrics=['accuracy'])

devices = ["/gpu:{}".format(x) for x in range(NUM_GPUS)]
mirror = tf.distribute.MirroredStrategy(devices)
config = tf.estimator.RunConfig(train_distribute=mirror,
                                eval_distribute=mirror)

estimator = tf.keras.estimator.model_to_estimator(model, config=config)

estimator.train(train_input, steps=TRAIN_STEPS)

results = estimator.evaluate(eval_input, steps=EVAL_STEPS)

print(results)

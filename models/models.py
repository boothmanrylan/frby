import tensorflow as tf
from tensorflow import keras
import numpy as np

def parse_fn(example):
    example_fmt = {'height': tf.FixedLenFeature((), tf.int64),
                   'width': tf.FixedLenFeature((), tf.int64),
                   'data': tf.FixedLenFeature((), tf.string),
                   'label': tf.FixedLenFeature((), tf.int64)}
    parsed = tf.parse_single_example(example, example_fmt)

    label = tf.cast(parsed['label'], tf.int64)

    height = tf.cast(parsed['height'], tf.int64)
    width = tf.cast(parsed['width'], tf.int64)
    shape = tf.stack([height, width])

    data = tf.decode_raw(parsed['data'], tf.float64)
    #data = tf.reshape(data, shape)
    data.set_shape(shape)
    return data, label

def input_fn(paths, take, batch_size):
    train_datasets = []
    test_datasets = []
    for path in paths:
        dataset = tf.data.Dataset.list_files(path)
        dataset = tf.data.TFRecordDataset(dataset)
        train = dataset.take(take)
        test = dataset.skip(take).take(take//2)
        train_datasets.append(train.map(parse_fn))
        test_datasets.append(test.map(parse_fn))
    train = train_datasets[0]
    test = test_datasets[0]
    for d in train_datasets[1:]:
        train = train.concatenate(d)
    for d in test_datasets[1:]:
        test = test.concatenate(d)
    train = train.shuffle(100)
    train = train.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    test = test.shuffle(100)
    test = test.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    return train, test

if __name__ == '__main__':
    sess = tf.Session()

    path = '/scratch/r/rhlozek/rylan/tfrecords/'
    paths = [path + x for x in ['rfi*', 'psr*', 'frb*']]
    take = 10
    train_dataset, test_dataset = input_fn(paths, take, 3)
    shape = train_dataset.output_shapes[0]
    t1 = tf.Print(shape, [shape])
    t2 = t1 + 1
    sess.run(t2)
    exit()


    model = keras.Sequential()
    model.add(keras.layers.Dense(64, input_shape=shape, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(3, activation='softmax'))

    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_dataset, epochs=5, steps_per_epoch=30)

    output = model.evaluate(test_dataset, steps=30)
    print(output)

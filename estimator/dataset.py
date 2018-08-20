import tensorflow as tf
import numpy as np
import glob
import definitions
#definitions.height and definitions.width hold the height and width of the data

class Dataset(object):
    def __init__(self, pattern, training=True):
        self.pattern = pattern
        self.training = training

    def parser(self, example):
        features = tf.parse_single_example(
            example,
            features={
                'data': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)
            })

        # Make the data 'channels_first' with only one channel
        shape = tf.stack([1, definitions.height, definitions.width])

        data = tf.decode_raw(features['data'], tf.float32)

        data.set_shape(definitions.height * definitions.width)

        data = tf.cast(tf.reshape(data, shape), tf.float32)

        label = tf.cast(features['label'], tf.int64)

        return data, label

    def make_batch(self, batch_size):
        records = tf.data.Dataset.list_files(self.pattern)
        dataset = tf.data.TFRecordDataset(records).repeat()
        dataset = dataset.map(self.parser, num_parallel_calls=batch_size)

        if self.training:
            min_queue = int(Dataset.examples_per_epoch(self.training) * 0.4)
            dataset = dataset.shuffle(buffer_size=min_queue + (3 * batch_size))

        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        data_batch, label_batch = iterator.get_next()

        return data_batch, label_batch

    @staticmethod
    def examples_per_epoch(training=True):
        if training:
            return 40
        else:
            return 40

class NpDataset(object):
    def __init__(self, pattern, training=True):
        self.pattern = pattern
        self.training = training

    def get_dataset(self):
        paths = glob.glob(self.pattern)
        all_data = np.empty((len(paths), 1, definitions.height, definitions.width))
        all_labels = np.empty(len(paths))
        for idx, path in enumerate(paths):
            data = np.load(path)
            data = np.reshape(data, (1, data.shape[0], data.shape[1]))
            if 'frb' in path:
                label = 0
            elif 'psr' in path:
                label = 1
            else:
                label = 2
            all_data[idx, :, :, :] = data
            all_labels[idx] = label
        all_data = all_data.astype('float32')
        all_labels = all_labels.astype('int64')
        dataset = (all_data[:10], all_labels[:10])
        dataset = tf.data.Dataset.from_tensor_slices(dataset)
        return dataset

    def make_batch(self, batch_size):
        dataset = self.get_dataset().repeat()
        dataset = dataset.batch(batch_size)

        if self.training:
            min_queue = int(10 * 0.4)
            buffer_size = min_queue + (3 * batch_size)
            dataset = dataset.shuffle(buffer_size=buffer_size)

        iterator = dataset.make_one_shot_iterator()
        data_batch, label_batch = iterator.get_next()
        return data_batch, label_batch

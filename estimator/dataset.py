import tensorflow as tf
import definitions

class Dataset(object):
    def __init__(self, pattern, training=True):
        self.pattern = pattern
        self.training = training

    def get_tfrecords(self):
        records = tf.data.Dataset.list_files(self.pattern)
        return records

    def parser(self, example):
        features = tf.parse_single_example(
            example,
            features={
                'data': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
            })
        height = tf.cast(features['height'], tf.int64)
        width = tf.cast(features['width'], tf.int64)
        # Make the data 'channels_first' with only one channel
        shape = tf.stack([1, definitions.height, definitions.width])

        data = tf.decode_raw(features['data'], tf.float32)

        data.set_shape(definitions.height * definitions.width)
        print(data.get_shape())

        data = tf.cast(tf.reshape(data, shape), tf.float32)

        label = tf.cast(features['label'], tf.int64)

        return data, label

    def make_batch(self, batch_size):
        records = self.get_tfrecords()
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



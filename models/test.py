import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras

base_dir = '/scratch/r/rhlozek/rylan/training_data/'

def read_npy(filename, label):
    filename = filename.decode(sys.getdefaultencoding())
    data = np.load(filename)
    return data.astype(np.float32), np.cast[np.float32](label)

def get_files(base, key, types):
    if base[-1] != '/': base = base + '/'
    if key[-1] != '/': key = key + '/'
    ranks = [x + '/' for x in os.listdir(base)]
    types = [x + '/' for x in types if x[-1] != '/']
    paths = ['{}{}{}{}'.format(base, r, key, t)
             for r in ranks
             for t in types]
    files = [p + x for p in paths for x in os.listdir(p)]
    return files

def make_dataset(filenames, label, batch_size):
    dataset = tf.data.Dataset.from_tensor_slice((filenames))
    dataset = dataset.map(
            lambda x: tf.py_func(read_npy, [x, label], [tf.float32, tf.float32])
            )
    dataset = dataset.shuffle(buffer_size=20)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_intializable_iterator()
    next_element = iterator.get_next()
    return iterator, next_element

if __name__ == '__main__':
    files = get_files(base_dir, 'frb', ['normal', 'poisson', 'uniform'])
    files = files[:50]
    iterator, next_element = make_dataset(files, 1, 32)
    exit()

    sess = tf.Session()
    sess.run(iterator.initializer)
    coutn = 1
    while True:
        try:
            features, labels = sess.run(next_element)
            print(count)
            count += 1
        except tf.errors.OutOfRangeError:
            break

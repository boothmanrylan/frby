import tensorflow as tf

tf.enable_eager_execution()

import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

labels = {0: 'FRB', 1: 'Pulsar', 2: 'RFI'}

def parse_fn(example):
    example_fmt = {
        'data/0th_dim': tf.FixedLenFeature([], tf.int64),
        'data/1st_dim': tf.FixedLenFeature([], tf.int64),
        'data/data':    tf.VarLenFeature(tf.float32),

        'signal/0th_dim': tf.FixedLenFeature([], tf.int64),
        'signal/1st_dim': tf.FixedLenFeature([], tf.int64),
        'signal/data':    tf.VarLenFeature(tf.float32),

        'text_label':  tf.FixedLenFeature([], tf.string),
        'rfi_type':    tf.FixedLenFeature([], tf.string),

        'snr':         tf.FixedLenFeature([], tf.float32),
        'dm':          tf.FixedLenFeature([], tf.float32),
        'scat_factor': tf.FixedLenFeature([], tf.float32),
        'width':       tf.FixedLenFeature([], tf.float32),
        'spec_ind':    tf.FixedLenFeature([], tf.float32),
        'period':      tf.FixedLenFeature([], tf.float32),
        'fluence':     tf.FixedLenFeature([], tf.float32),
        't_ref':       tf.FixedLenFeature([], tf.float32),
        'f_ref':       tf.FixedLenFeature([], tf.float32),
        'rate':        tf.FixedLenFeature([], tf.float32),
        'delta_t':     tf.FixedLenFeature([], tf.float32),
        'max_freq':    tf.FixedLenFeature([], tf.float32),
        'min_freq':    tf.FixedLenFeature([], tf.float32),

        'label':       tf.FixedLenFeature([], tf.int64),
        'n_periods':   tf.FixedLenFeature([], tf.int64),
        'scintillate': tf.FixedLenFeature([], tf.int64),
        'window':      tf.FixedLenFeature([], tf.int64),
        'id':          tf.FixedLenFeature([], tf.int64),
    }
    parsed = tf.parse_single_example(example, example_fmt)
    x = parsed['data/0th_dim']
    y = parsed['data/1st_dim']
    data = tf.cast(parsed['data/data'], tf.float32)
    data = tf.sparse.to_dense(data)
    data = tf.reshape(data, tf.stack([x, y]))

    x = parsed['signal/0th_dim']
    y = parsed['signal/1st_dim']
    signal = tf.cast(parsed['signal/data'], tf.float32)
    signal = tf.sparse.to_dense(signal)
    signal = tf.reshape(signal, tf.stack([x, y]))

    snr = tf.cast(parsed['snr'], tf.float32)
    return data, signal, snr, parsed['label']
    # return image, parsed["label"], dm, filename

pattern = "/scratch/r/rhlozek/rylan/training_data/*.tfrecord"
records = tf.data.Dataset.list_files(pattern, shuffle=False)
dataset = records.interleave(tf.data.TFRecordDataset, cycle_length=4)
# dataset = dataset.shuffle(buffer_size=100)
dataset = dataset.map(map_func=parse_fn)

for i, elem in enumerate(dataset):
    if i > 100: break
    plt.subplot(121)
    plt.title('SNR: {:.3f}'.format(float(elem[2])))
    plt.imshow(elem[0], interpolation='nearest', aspect='auto')
    plt.subplot(122)
    plt.title(labels[int(elem[3])])
    plt.imshow(elem[1], interpolation='nearest', aspect='auto')
    plt.savefig('/scratch/r/rhlozek/rylan/tfrecord_test/{}.png'.format(i))

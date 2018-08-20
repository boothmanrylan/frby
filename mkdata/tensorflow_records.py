import os
import tensorflow as tf
import numpy as np

label_dict = {'frb': 0, 'psr': 1, 'rfi': 2}

def get_files(base, key, types, start=0, stop=-1):
    if base[-1] != '/': base = base + '/'
    if key[-1] != '/': key = key + '/'
    ranks = [x + '/' for x in os.listdir(base)]
    ranks = ranks[start:stop]
    types = [x + '/' for x in types if x[-1] != '/']
    paths = ['{}{}{}{}'.format(base, r, key, t)
             for r in ranks
             for t in types]
    files = [p + x for p in paths for x in os.listdir(p)]
    if key[-1] == '/': key = key[:-1]
    labels = [label_dict[key]] * len(files)
    return np.asarray(files), np.asarray(labels)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def write(path, data, labels):
    assert(data.shape[0] == labels.shape[0])
    n_samples = data.shape[0]
    count = 1
    with tf.python_io.TFRecordWriter(path) as tfw:
        for idx in range(n_samples):
            d = np.load(data[idx]).astype('float32')
            d = d.tostring()
            feature = {'label': _int64_feature(labels[idx]),
                       'data': _bytes_feature(d)}
            features = tf.train.Features(feature=feature)
            example = tf.train.Example(features=features)
            tfw.write(example.SerializeToString())
            print('wrote {}/{} to {}'.format(count, n_samples, path))
            count += 1

def read(file, N):
    iterator = tf.python_io.tf_record_iterator(path=file)

    count = 0
    for record in iterator:
        if count >= N: break
        example = tf.train.Example()
        example.ParseFromString(record)
        height = int(example.features.feature['height']
                .int64_list
                .value[0])
        width = int(example.features.feature['width']
                .int64_list
                .value[0])
        label = int(example.features.feature['label']
                .int64_list
                .value[0])
        data = (example.features.feature['data']
                .bytes_list
                .value[0])
        data = np.fromstring(data, dtype=np.float64)
        data = data.reshape((height, width))
        print(data.shape)
        print(label)

        count += 1

if __name__ == "__main__":
    directory = '/scratch/r/rhlozek/rylan/npy_data/'
    types = ['normal', 'poisson', 'uniform', 'telescope', 'solid']
    a = 0
    b = 40
    # path = '/scratch/r/rhlozek/rylan/tfrecords/frb{}-{}.tfrecords'.format(a,b)
    # data, labels = get_files(directory, 'frb/', types, start=a, stop=b)
    # write(path, data, labels)
    path = '/scratch/r/rhlozek/rylan/tfrecords/rfi{}-{}.tfrecords'.format(a,b)
    data, labels = get_files(directory, 'rfi/', types, start=a, stop=b)
    write(path, data, labels)
    exit()
    path = '/scratch/r/rhlozek/rylan/tfrecords/psr{}-{}.tfrecords'.format(a,b)
    data, labels = get_files(directory, 'psr/', types, start=a, stop=b)
    write(path, data, labels)

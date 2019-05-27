#!/usr/bin/python
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts numpy data to TFRecords file format with Example protos.

The numpy data set is expected to reside in .npy files located in the
following directory structure.

  data_dir/000/label_0/*/image0.npy
  data_dir/001/label_1/*/image1.npy
  ...
  data_dir/120/label_0/*/weird-image.npy
  data_dir/121/label_1/*/my-image.npy
  ...

where the second sub-directory is the unique label associated with these images.

This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of TFRecord files

  train_directory/train-00000-of-01024
  train_directory/train-00001-of-01024
  ...
  train_directory/train-01023-of-01024

and

  validation_directory/validation-00000-of-00128
  validation_directory/validation-00001-of-00128
  ...
  validation_directory/validation-00127-of-00128

The Example proto contains the following fields:

  image: string containing PNG encoded image in RGB colorspace
  height: integer, image height in pixels
  width: integer, image width in pixels
  colorspace: string, specifying the colorspace, always 'RGB'
  channels: integer, specifying the number of channels, always 3
  format: string, specifying the format, always 'PNG'
  filename: string containing the basename of the image file
            e.g. 'n01440764_10026.npy' or 'ILSVRC2012_val_00000293.npy'
  label: integer specifying the index in a classification layer.
    The label ranges from [0, num_labels]
  text_label: string specifying the human-readable version of the label
    e.g. 'dog'
  dm: the dispersion measure of the sample. 0 if the sample is rfi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading
import glob

from skimage.transform import downscale_local_mean
import imageio

import numpy as np
import tensorflow as tf
import pickle


# turn off unnecessary imageio warning
def silence(*args, **kwargs): pass
imageio.core.util._precision_warn = silence


tf.app.flags.DEFINE_string('data_dir', '/scratch/r/rhlozek/rylan/npy_data/',
                           'Input Data directory')
tf.app.flags.DEFINE_string('output_directory',
                           '/scratch/r/rhlozek/rylan/tfrecords/',
                           'Output data directory')
tf.app.flags.DEFINE_integer('train_shards', 20,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('val_shards', 20,
                            'Number of shards in validation TFRecord files.')
tf.app.flags.DEFINE_integer('eval_shards', 20,
                            'Number of shards in evaluation TFRecord files.')
tf.app.flags.DEFINE_integer('num_threads', 20,
                            'Number of threads to preprocess the images.')
tf.app.flags.DEFINE_float('percent_train', 0.5,
        'Percentage of data to use for training, rounding to nearest 10%')
tf.app.flags.DEFINE_float('percent_eval', 0.4,
        'Percentage of data to use for evaluation, rounded to nearest 10%')
tf.app.flags.DEFINE_float('percent_val', 0.1,
        'Percentage of data to use for validation, rounded to nearest 10%')

FLAGS = tf.app.flags.FLAGS


unique_labels = ["frb", "psr", "rfi"]

missing_metadata = 0


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def parse_filename(filename):
    """
    filenames no longer perfectly match their counterparts in the metadata
    files, in particuler /scratch/r/rhlozek/rylan/npy_data/000/frb/0.npy should
    be converted to /scratch/r/rhlozek/rylan/training_data/0/frb/0.npy

    Args:
        filename: string, path to the data sample of interest
    Returns:
        metadata_file: string path to the corresponding metadata pickle
        key: key to get this samples metadata from metedata_file
    """
    file = filename.split('/')
    root = '/'.join(file[:6])
    rank = int(file[6])
    metadata_file = root + '/{:03}/metadata{}.pkl'.format(rank, rank)
    key = '/'.join(file[:5]) + '/training_data/' + str(rank) + '/' + '/'.join(file[7:])
    return metadata_file, key


def get_dm(filename):
    """
    Returns the dispersion measure of the simulation contained in file. If the
    simulation is RFI, the returned dm is always 0.0

    Args:
        filename: string, path to the data sample
    Returns:
        dm: float
    """
    metadata_file, key = parse_filename(filename)
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    try:
        metadata = metadata[key]
    except KeyError:
        # metadata for this sample does not exist, don't save it
        return -1

    if 'dm' in metadata.keys():
        dm = metadata['dm'].value
    else: # sample is rfi 
        dm = 0.0
    return np.float32(dm)


def _convert_to_example(filename, image_buffer, label, text, height, width,
                        channels):
    """Build an Example proto for an example.

    Args:
        filename: string, path to an image file, e.g., '/path/to/example.JPG'
        image_buffer: string, PNG encoding of image
        label: integer, identifier for the ground truth for the network
        text: string, unique human-readable, e.g. 'dog'
        height: integer, image height in pixels
        width: integer, image width in pixels
        channels: integer, image depth in pixels
    Returns:
        Example proto or False is metadata for filename is missing
    """

    colorspace = 'RGB' # not true anymore, files are all b/w
    image_format = 'PNG'
    base_filename = os.path.basename(filename)

    dm = get_dm(filename)

    if dm < 0: # missing metadata
        return False

    example = tf.train.Example(features=tf.train.Features(feature={
        'height':     _int64_feature(height),
        'width':      _int64_feature(width),
        'channels':   _int64_feature(channels),
        'colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
        'label':      _int64_feature(label),
        'text_label': _bytes_feature(tf.compat.as_bytes(text)),
        'format':     _bytes_feature(tf.compat.as_bytes(image_format)),
        'filename':   _bytes_feature(tf.compat.as_bytes(base_filename)),
        'image':      _bytes_feature(tf.compat.as_bytes(image_buffer)),
        'dm':         _float_feature(dm)}))
    return example


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that decodes PNG data.
        self._decode_png_data = tf.placeholder(dtype=tf.string)
        self._decode_png = tf.image.decode_png(self._decode_png_data,
                                               channels=1)

    def decode_png(self, image_data):
        image = self._sess.run(self._decode_png,
                               feed_dict={self._decode_png_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 1
        return image


def _convert_to_image(filename):
    """Convert a single npy file into png bytes

    Args:
      filename: string, path to the npy file
    Returns:
      data: bytes, png representation of numpy ndarray
    """
    data = np.load(filename)
    data = downscale_local_mean(data, (32, 42))
    data = imageio.imwrite(uri=imageio.RETURN_BYTES, im=data, format='png')
    return data


def _process_image(filename, coder):
    """Process a single image file.

    Args:
        filename: string, path to an image file e.g., '/path/to/example.JPG'.
        coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
        image_buffer: string, PNG encoding of RGB image.
        height: integer, image height in pixels.
        width: integer, image width in pixels.
    """
    image_data = _convert_to_image(filename)

    # Decode the PNG.
    image = coder.decode_png(image_data)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]

    return image_data, height, width, channels


def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               texts, labels, num_shards):
    """Processes and saves list of images as TFRecord in 1 thread.

    Args:
        coder: instance of ImageCoder to provide TensorFlow image coding utils.
        thread_index: integer, unique batch to run index is within
                      [0, len(ranges)).
        ranges: list of pairs of integers specifying ranges of each batches to
                analyze in parallel.
        name: string, unique identifier specifying the data set
        filenames: list of strings; each string is a path to an image file
        texts: list of strings; each string is human readable, e.g. 'dog'
        labels: list of integer; each integer identifies the ground truth
        num_shards: integer number of shards for this data set.
    """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_directory, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        for i in np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int):
            filename = filenames[i]
            label = labels[i]
            text = texts[i]

            try:
                image, height, width, channels = _process_image(filename, coder)
            except Exception as e:
                print(e)
                print('SKIPPED: Unexpected error while decoding %s.' % filename)
                continue

            example = _convert_to_example(filename, image, label, text, height,
                                          width, channels)
            if not example:
                print('SKIPPED: metadata did not exist for {}.'.format(filename))
                global missing_metadata
                missing_metadata += 1
                continue

            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1

            if not counter % 1000:
                print('%s [thread %d]: Processed %d of %d images.' %
                      (datetime.now(), thread_index, counter, files_in_thread))
                sys.stdout.flush()

        writer.close()
        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, files_in_thread))
    sys.stdout.flush()


def _process_image_files(name, filenames, texts, labels, num_shards):
    """Process and save list of images as TFRecord of Example protos.

    Args:
        name: string, unique identifier specifying the data set
        filenames: list of strings; each string is a path to an image file
        texts: list of strings; each string is human readable, e.g. 'dog'
        labels: list of integer; each integer identifies the ground truth
        num_shards: integer number of shards for this data set.
    """
    assert len(filenames) == len(texts)
    assert len(filenames) == len(labels)

    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(labels), FLAGS.num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()

    threads = []
    for thread_index in range(len(ranges)):
        args = (coder, thread_index, ranges, name, filenames,
                texts, labels, num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(filenames)))
    sys.stdout.flush()


def _find_image_files(data_dir, unique_labels, start, stop):
    """Build a list of all images files and labels in the data set.

    Args:
        data_dir: string, path to the root directory of images.

          Assumes that the image data set resides in PNG files located in
          the following directory structure.

            data_dir/*/label1/*/image.jpeg
            data_dir/*/label2/*/image.jpeg

        unique_labels: list, all valid_labels.
        percent_train: float, the percentage of files to use for training

    Returns:
        filenames: list of strings; paths to a image files.
        texts: list of strings; each string is the class, e.g. 'dog'
        labels: list of integers; each integer is aclassification label.
    """
    labels = []
    texts = []
    filenames = []

    for idx, text in enumerate(unique_labels):
        path = '{}/[0-9][0-9][{}-{}]/{}/*/*'.format(data_dir, start, stop, text)
        matching_files = glob.glob(path)

        labels.extend([idx] * len(matching_files))
        texts.extend([text] * len(matching_files))
        filenames.extend(matching_files)

        print('found files in %d of %d classes' % (idx, len(unique_labels)))

    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(filenames)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    texts = [texts[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    print('Found %d npy files across %d labels inside %s.' %
          (len(filenames), len(unique_labels), data_dir))

    return filenames, texts, labels


def main(unused_argv):
    assert not FLAGS.train_shards % FLAGS.num_threads, (
        'Please make the num_threads commensurate with train_shards')
    assert not FLAGS.eval_shards % FLAGS.num_threads, (
        'Please make the num_threads commensurate with eval_shards')
    assert not FLAGS.val_shards % FLAGS.num_threads, (
        'Please make the num_threads commensurate with val_shards')
    assert FLAGS.percent_train + FLAGS.percent_eval + FLAGS.percent_val == 1.0, (
        'percent_train, precent_eval, percent_val should sum to 1.0')
    print('Saving results to {}'.format(FLAGS.output_directory))

    train_start = 0
    eval_start = int(10 * round(FLAGS.percent_train, 1))
    val_start = int(10 * round(FLAGS.percent_train + FLAGS.percent_eval, 1))
    train_stop = eval_start - 1
    eval_stop = val_start - 1

    train_data = _find_image_files(FLAGS.data_dir, unique_labels, 0, train_stop)
    _process_image_files(*(('train',) + train_data + (FLAGS.train_shards,)))
    print("Done making training data")

    eval_data = _find_image_files(FLAGS.data_dir, unique_labels, eval_start, eval_stop)
    _process_image_files(*(('evaluate',) + eval_data + (FLAGS.val_shards,)))
    print("Done making evaluation data")

    val_data = _find_image_files(FLAGS.data_dir, unique_labels, val_start, 9)
    _process_image_files(*(('validate',) + val_data + (FLAGS.eval_shards,)))
    print("Done making validation data")

    global missing_metadata
    if missing_metadata > 0:
        print('{} files did not have metadata.'.format(missing_metadata))

if __name__ == '__main__':
    tf.app.run()


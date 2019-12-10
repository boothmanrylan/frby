import os
import glob
import tensorflow as tf
import numpy as np
import imageio

dms = {
    '180725.J0613+67': 715.98,
    '180727.J1311+26': 642.07,
    '180729.J0558+56': 109.610,
    '180729.J1316+55': 317.32,
    '180730.J0353+87': 849.047,
    '180801.J2130+72': 656.20,
    '180806.J1515+75': 739.98,
    '180810.J0646+34': 414.95,
    '180810.J1159+83': 169.134,
    '180812.J0112+80': 802.57,
    '180814.J0422+73': 238.32,
    '180814.J1554+74': 189.38,
    '180817.J1533+42': 1006.840,
}

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
    dm = dms[base_filename.split('_')[0]]

    example = tf.train.Example(features=tf.train.Features(feature={
        'height':     _int64_feature(height),
        'width':      _int64_feature(width),
        'channels':   _int64_feature(channels),
        'colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
        'label':      _bytes_feature(tf.compat.as_bytes(base_filename)),
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

output = '/scratch/r/rhlozek/rylan/chime_data/chime_frbs.tfrecords'
files = glob.glob('/scratch/r/rhlozek/rylan/chime_data/npy/*.npy')

coder = ImageCoder()

writer = tf.python_io.TFRecordWriter(output)
for f in files:
    image, height, width, channels = _process_image(f, coder)
    example = _convert_to_example(f, image, None, '', height,
                                  width, channels)
    writer.write(example.SerializeToString())
writer.close()


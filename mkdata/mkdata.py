import os
import glob
import numpy as np
# from mpi4py import MPI
import astropy.units as u
from frb import FRB
from pulsar import Pulsar
from rfi import NormalRFI, UniformRFI, PoissonRFI, SolidRFI, TelescopeRFI
from baseband.helpers import sequentialfile as sf
from modifications import modifications
import tensorflow as tf
from skimage.measure import block_reduce

modifications = modifications[:2]

frbArgs = {'t_ref': None,
           'f_ref': None,
           'dm': (50, 2500) * (u.pc / u.cm**3),
           'fluence': (0.01, 150) * (u.Jy * u.ms),
           'freq': (800, 400) * u.MHz,
           'rate': (400 / 1024) * u.MHz,
           'scat_factor': (-5, -3),
           'width': (0.025, 40) * u.ms,
           'scintillate': None,
           'spec_ind': (-10, 15),
           'window': True,
           'normalize': True,
           'max_size': 2**15}

psrArgs = {'dm': (2, 25) * (u.pc / u.cm**3),
           'fluence': (0.01, 5) * (u.Jy * u.ms),
           'freq': (800, 400) * u.MHz,
           'rate': (400 / 1024) * u.MHz,
           'scat_factor': (-5, -4),
           'width': (0.025, 5) * u.ms,
           'scintillate': True,
           'spec_ind': (-1, 1),
           'max_size': 2**15,
           'period': (2, 5e2) * u.ms}

def _int64_feature(value):
    """Wrapper for inserting int64 featues into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    """Wrapper for inserting float feature into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _ndarray_feature(value):
    """Wrapper for inserting ndarray feature into Example proto."""
    # save the array dimensions
    for idx, x in enumerate(value.shape):
        f[idx] = _int64_feature(x)
    # flatten array before saving
    value = value.reshape(-1)
    f['array'] = tf.train.Feature(float_list=tf.train.FloatList(value=value))
    return f

def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_labels(label):
    if label == "frb":
        return 0
    elif label == "pulsar":
        return 1
    elif label == "rfi":
        return 2
    else:
        raise ValueError("Mislabelled data passed to convert_to_example")

def max_reduce(array, dim):
    """
    0 pads array so that 0th and 1st dimension are divisble by dim. Then
    applies max block reduction to reshape array to (dim, dim).
    """
    while array.shape[0] % dim != 0:
        array = np.vstack([np.zeros((1, array.shape[1])), array])
    while signal.shape[1] % dim != 0:
        array = np.hstack([np.zeros((array.shape[0], 1)), array])
    x = array.shape[0] // dim
    y = array.shape[1] // dim
    return block_reduce(array, (x, y), np.max)

def convert_to_example(event, id):
    try:
        data1 = _ndarray_feature(event.sample)
        data2 = _ndarray_feature(max_reduce(event.signal, 32))
        example = tf.train.Example(features=tf.train.Features(feature={
            'data/0th_dim': data[0],
            'data/1st_dim': data[1], # TODO: arrays with more than 2 dims?
            'data/data':    data['array'],

            'signal/0th_dim': data2[0],
            'signal/1st_dim': data2[1],
            'signal/data':    data2['array'],

            'text_label': _bytes_feature(tf.compat.as_bytes(event.label)),
            'rfi_type':   _bytes_feature(tf.compat.as_bytes(event.rfi_type)),

            'snr':         _float_feature(event.snr),
            'dm':          _float_feature(event.dm.value),
            'scat_factor': _float_feature(event.scat_factor),
            'width':       _float_feature(event.width),
            'spec_ind':    _float_feature(event.spec_ind),
            'period':      _float_feature(event.period.value),
            'fluence':     _float_feature(event.fluence.value),
            't_ref':       _float_feature(event.t_ref.value),
            'f_ref':       _float_feature(event.f_ref.value),
            'rate':        _float_feature(event.rate.value),
            'delta_t':     _float_feature(event.delta_t.value),
            'max_freq':    _float_feature(max(event.freq).value),
            'min_freq':    _float_feature(min(event.freq).value),

            'label':       _int64_feature(convert_labels(event.label)),
            'n_periods':   _int64_feature(event.n_periods),
            'scintillate': _int64_feature(int(event.scintillate)),
            'window':      _int64_feature(int(event.window)),
            'id':          _int64_feature(id)
        }))
        return example
    except AttributeError as E:
        m = "Event passed to convert_to_example must be FRB, Pulsar, or RFI"
        raise ValueError(m) from E

def inject_and_save(rfi, tfrecordwriter, sim_number, descriptor):
    """
    inject pulsars/frbs into rfi then save all into tfrecrod
    """
    rfi_type = rfi.rfi_type if descriptor is None else rfi.rfi_type + descriptor

    example = convert_to_example(rfi, sim_number)
    tfrecordwriter.write(example.SerializeToString())
    sim_number += 1

    psrArgs['background'] = rfi.sample
    psrArgs['rate'] = rfi.rate
    psrArgs['rfi_type'] = rfi_type
    try: # ensure DM and duration are compatabile
        psr = Pulsar(**psrArgs)
        if psr.snr > 0:
            example = convert_to_example(psr, sim_number)
            tfrecordwriter.write(example.SerializeToString())
            sim_number += 1
    except ValueError:
        pass

    frbArgs['background'] = rfi.sample
    frbArgs['rate'] = rfi.rate
    frbArgs['rfi_type'] = rfi_type
    frb = FRB(**frbArgs)
    if frb.snr > 0:
        example = convert_to_example(frb, sim_number)
        tfrecordwriter.write(example.SerializeToString())
        sim_number += 1

    return sim_number

def make_data(rfi, rank, writer):
    n_mods = len(modifications)

    sim_num = inject_and_save(rfi, writer, rank * (1 + n_mods), None)

    for idx, mods in enumerate(modifications):
        # Allow multiple modifications to be applied before saving
        for m in mods:
            rfi.apply_function(m['func'], name=m['name'], input=m['input'],
                               freq_range=m['freq_range'],
                               time_range=m['time_range'], **m['params'])
        sim_num = inject_and_save(rfi, writer, sim_num, 'modded')
        print('Done {}/{}'.format(idx + 1, n_mods))

        # Unapply the modifications to apply more to the same base
        rfi.reset()

    shape = rfi.sample.shape
    duration = rfi.attributes['duration']
    return shape, duration

if __name__ == "__main__":
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # size = comm.Get_size()
    rank = 0
    size = 1

    tfrecord = '/home/rylan/dunlap-2.0/data/data_{}.tfrecord'.format(rank)
    writer = tf.python_io.TFRecordWriter(tfrecord)

    aro_files = np.sort(glob.glob('/home/rylan/dunlap-2.0/data/vdif/*.vdif'))
    aro_files = aro_files[:1]
    group = 20

    # while aro_files.shape[0] - group <= size:
    #    aro_files = np.hstack([aro_files] * 2)

    rank_files = sf.open(aro_files)
    # rank_files = sf.open(aro_files[rank: rank + group])

    # Make the data with rfi from ARO as the background, use the shape and
    # duration of data made through this method as the shape and duration
    # of data made in the other methods
    rfi = TelescopeRFI(rank_files, rate_out=1*u.kHz)
    shape, duration = make_data(rfi, rank, writer)

    for rfiClass in [NormalRFI, UniformRFI, PoissonRFI, SolidRFI]:
        rfi = rfiClass(shape=shape, duration=duration)
        make_data(rfi, rank, writer)

    writer.close()

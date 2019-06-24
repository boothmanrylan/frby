import sys
import glob
import signal
import numpy as np
import astropy.units as u
import tensorflow as tf
from mpi4py import MPI
from skimage.measure import block_reduce
from baseband.helpers import sequentialfile as sf
from frb import FRB
from pulsar import Pulsar
from rfi import NormalRFI, UniformRFI, PoissonRFI, SolidRFI, TelescopeRFI
from modifications import modifications

frb_sims = 0
psr_sims = 0
rfi_sims = 0

interrupted = False
def signal_handler(signum, frame):
    global interrupted
    interrupted = True

summary_file = ""
def close_gracefully():
    comm = MPI.COMM_WORLD
    rank = comm.get_rank()

    sendbuf = np.array([frb_sims, psr_sims, rfi_sims], dtype=int)
    recvbuf = None

    if rank == 0:
        recvbuf = np.empty([size, 3], dtype=int)

    comm.Gather(sendbuf, recvbuf, root=0)

    if rank == 0:
        recvbuf = np.sum(recvbuf, axis=0)
        m = f",FRBs,Pulsars,RFI\nNsims,{recvbuf[0]},{recvbuf[1]},{recvbuf[2]}"
        if summary_file is "":
            print(m)
        else:
            with open(summary_file, 'w') as f:
            f.write(m))

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
    f = {}
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

def mean_reduce(signal, dim):
    """
    0 pads array so that 0th and 1st dimension are divisble by dim. Then
    applies max block reduction to reshape array to (dim, dim).
    """
    while signal.shape[0] % dim != 0:
        signal = np.vstack([np.zeros((1, signal.shape[1])), signal])
    x = signal.shape[0] // dim
    y = signal.shape[1] // dim
    return block_reduce(signal, (x, y), np.mean)

def convert_to_example(event, id):
    try:
        data1 = _ndarray_feature(event.sample)
        data2 = _ndarray_feature(mean_reduce(event.signal, 64))
        example = tf.train.Example(features=tf.train.Features(feature={
            'data/0th_dim': data1[0],
            'data/1st_dim': data1[1], # TODO: arrays with more than 2 dims?
            'data/data':    data1['array'],

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
    inject pulsars/frbs into rfi event then save with tfrecordwriter
    """
    rfi_type = rfi.rfi_type if descriptor is None else rfi.rfi_type + descriptor

    example = convert_to_example(rfi, sim_number)
    tfrecordwriter.write(example.SerializeToString())
    sim_number += 1
    global rfi_sims
    rfi_sims += 1

    try: # ensure DM and duration are compatabile
        psr = Pulsar(background=rfi.sample, rate=rfi.rate, rfi_type=rfi_type)
        if psr.snr > 0: # dont save sim with snr == 0
            example = convert_to_example(psr, sim_number)
            tfrecordwriter.write(example.SerializeToString())
            sim_number += 1
            global psr_sims
            psr_sims += 1
        else:
            print("Pulsar creation failed due to signal-to-noise ratio")
    except ValueError as E:
        print("Pulsar creation failed due to DM duration mismatch")

    frb = FRB(background=rfi.sample, rate=rfi.rate, rfi_type=rfi_type)
    if frb.snr > 0: # dont save sim with snr == 0
        example = convert_to_example(frb, sim_number)
        tfrecordwriter.write(example.SerializeToString())
        sim_number += 1
        global frb_sims
        frb_sims += 1
    else:
        print("Pulsar creation failed due to signal-to-noise-ratio")
    return sim_number

def make_data(rfi, writer, rank, sim_num, sims_per_rank):
    # make sims with unmodified background
    sim_num = inject_and_save(rfi, writer, sim_num, None)
    out = f"Done {sim_num - (rank * sims_per_rank)}/{sims_per_rank}\t\t\r"
    sys.stdout.write(out)
    sys.stdout.flush()

    # make sims with modified background
    for idx, mods in enumerate(modifications):
        if interrupted: close_gracefully()
        # Allow multiple modifications to be applied before saving
        for m in mods:
            rfi.apply_function(m['func'], name=m['name'], input=m['input'],
                               freq_range=m['freq_range'],
                               time_range=m['time_range'], **m['params'])

        sim_num = inject_and_save(rfi, writer, sim_num, 'modded')
        out = f"Done {sim_num - (rank * sims_per_rank)}/{sims_per_rank}\t\t\r"
        sys.stdout.write(out)
        sys.stdout.flush()

        # Unapply the modifications to apply more to the same base
        rfi.reset()

    shape = rfi.sample.shape
    duration = rfi.attributes['duration']
    return sim_num, shape, duration

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    group = 30 # 30 vdif files is ~2.5s
    aro_files = np.sort(glob.glob('/scratch/r/rhlozek/rylan/aro_rfi/*vdif'))
    while size * group >= aro_files.shape[0]:
       aro_files = np.hstack([aro_files, aro_files])

    rfi_types = [NormalRFI, UniformRFI, PoissonRFI]

    # +1 for Telescope
    n_rfi_types = len(rfi_types) + 1

    # 3 because each sim creates an FRB, a Pulsar, and an RFI sample
    sims_per_rank = 3 * n_rfi_types * (1 + len(modifications))

    init_sim_num = rank * sims_per_rank

    output_dir = "/scratch/r/rhlozek/rylan/training_data/"
    tfrecord = f"{output_dir}{rank:03d}.tfrecord"

    summary_file = f"{output_dir}dataset_summary.csv"

    with tf.io.TFRecordWriter(tfrecord) as writer:
        rank_files = sf.open(aro_files[(rank * group):(rank + 1)  * group])

        # Make the data with rfi from ARO as the background, use the shape and
        # duration of data made through this method as the shape and duration
        # of data made in the other methods
        rfi = TelescopeRFI(rank_files, rate_out=1*u.kHz)
        sim_num, shape, duration = make_data(rfi, writer, rank, init_sim_num, sims_per_rank)

        for rfiClass in rfi_types:
            rfi = rfiClass(shape=shape, duration=duration)
            sim_num, _, _ = make_data(rfi, writer, rank, sim_num, sims_per_rank)

    close_gracefully()


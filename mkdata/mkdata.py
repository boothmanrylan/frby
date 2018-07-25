import os
import h5py
import glob
import numpy as np
# from mpi4py import MPI
import astropy.units as u
from frb import FRB
from pulsar import Pulsar
from rfi import NormalRFI, UniformRFI, PoissonRFI, SolidRFI, TelescopeRFI
from baseband.helpers import sequentialfile as sf
from modifications import modifications

frb_args = {'t_ref': 0 * u.ms,
            'f_ref': 600 * u.MHz,
            'dm': (100, 1000) * (u.pc / u.cm**3),
            'fluence': (0.01, 150) * (u.Jy * u.ms),
            'freq': (800, 400) * u.MHz,
            'rate': (400 / 1024) * u.MHz,
            'scat_factor': (-5, -4),
            'width': (0.025, 30) * u.ms,
            'scintillate': True,
            'spec_ind': (-10, 15),
            'max_size': 2**15}

psr_args = {'t_ref': 0 * u.ms,
            'f_ref': 600 * u.MHz,
            'dm': (2, 25) * (u.pc / u.cm**3),
            'fluence': (0.01, 10) * (u.Jy * u.ms),
            'freq': (800, 400) * u.MHz,
            'rate': (400 / 1024) * u.MHz,
            'scat_factor': (-5, -4),
            'width': (0.025, 5) * u.ms,
            'scintillate': True,
            'spec_ind': (-1, 1),
            'max_size': 2**15,
            'period': (2, 5e2) * u.ms}

def save_to_hdf5(event, attributes, hdf5_file, path):
    dset = hdf5_file.create_dataset(path, data=event)
    for key, value in attributes.items():
        try:
            dset.attrs[key] = value
        except TypeError as E:
            dset.attrs[key] = np.string_(value)

def inject_and_save(rfi_event, hdf5_file, sim_number, path):
    rfi_path = '/rfi/{}/{}'.format(path, sim_number)
    save_to_hdf5(rfi_event.rfi, rfi_event.attributes, file, rfi_path)

    psr_path = '/psr/{}/{}'.format(path, sim_number)
    psr_args['background'] = rfi_event.rfi
    psr_args['rate'] = rfi_event.rfi.shape[1]/rfi_event.attributes['duration']
    psr = Pulsar(**psr_args)
    save_to_hdf5(psr.pulsar, psr.attributes, file, psr_path)
    return

    frb_path = '/frb/{}/{}'.format(path, sim_number)
    frb_args['background'] = rfi_event.rfi
    frb_args['rate'] = rfi_event.rfi.shape[1]/rfi_event.attributes['duration']
    frb = FRB(**frb_args)
    save_to_hdf5(frb.frb, frb.attributes, file, frb_path)

def make_data(rfi_class, files, kwargs, key, rank, hdf5_file):
    n_mods = len(modifications)

    rfi = rfi_class(files, **kwargs)
    path = '{}/unmodified'.format(key)
    inject_and_save(rfi, hdf5_file, rank * n_mods, path)

    for idx, mods in enumerate(modifications):
        # Allow multiple modifications to be applied before saving
        for m in mods:
            rfi.apply_function(m['func'], name=m['name'], input=m['input'],
                               freq_range=m['freq_range'],
                               time_range=m['time_range'], **m['params'])
        path = '{}/modified'.format(RFI['key'])
        inject_and_save(rfi, hdf5_file, (rank * n_mods) + idx, path)

        # Unapply the modification to be able to apply another to the same base
        rfi.reset()
    shape = rfi.rfi.shape
    duration = rfi.attributes['duration']
    return shape, duration

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()
rank = 0
size = 1

file_name = '/home/rylan/dunlap/data/training_data.hdf5'
file = h5py.File(file_name, 'w')

rfi_grp = file.create_group('/rfi')
psr_grp = file.create_group('/psr')
frb_grp = file.create_group('/frb')
for grp in [rfi_grp, psr_grp, frb_grp]:
    for sub_grp in ['telescope', 'solid', 'poisson', 'uniform', 'normal']:
        sub_sub_grp = grp.create_group(sub_grp)
        sub_sub_grp.create_group('unmodified')
        sub_sub_grp.create_group('modified')

aro_files = np.sort(glob.glob('/home/rylan/dunlap/data/vdif/*.vdif'))
group = 20

while aro_files.shape[0] - group < size:
    aro_files = np.hstack([aro_files] * 2)

rank_files = sf.open(aro_files[rank: rank + group])

# Make the data with rfi from ARO as the background, use the shape and duration
# of data made through this method as the shape and duration of data made in
# the other methods
shape, duration = make_data(TelescopeRFI, rank_files, {'rate_out': 1e-3*u.MHz},
                            rank, file)

normal_rfi = {'class': NormalRFI,
              'key': 'normal',
              'kwargs': {'shape': shape,
                         'duration': duration}}
uniform_rfi = {'class': UniformRFI,
               'key': 'uniform',
               'kwargs': {'shape': shape,
                         'duration': duration}}
poisson_rfi = {'class': PoissonRFI,
               'key': 'poisson',
               'kwargs': {'shape': shape,
                         'duration': duration}}
solid_rfi = {'class': SolidRFI,
             'key': 'solid',
             'kwargs': {'shape': shape,
                         'duration': duration}}

# Make the data with the random and solid backgrounds
rfi_types = [normal_rfi, uniform_rfi, poisson_rfi, solid_rfi]
for RFI in rfi_types:
    make_data(RFI['class'], rank_files, RFI['kwargs'], RFI['key'], rank, file)

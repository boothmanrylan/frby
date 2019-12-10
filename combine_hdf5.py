import h5py
import glob
import sys

file_pattern = '/scratch/r/rhlozek/rylan/split_training_data/*.hdf5'
h5files = sorted(glob.glob(file_pattern))

output_file = '/scratch/r/rhlozek/rylan/split_data/evaluation.hdf5'

def progress(names, counts, totals):
    assert len(names) == len(counts) and len(counts) == len(totals)
    msg = ""
    for i, n in enumerate(names):
        msg += f"{n}: {counts[i]}/{totals[i]}\t"
    msg += '\r'
    sys.stdout.write(msg)
    sys.stdout.flush()


with h5py.File(output_file, 'w') as output:
    for file_idx, h5file in enumerate(h5files):
        with h5py.File(h5file, 'r') as input:
            for group_idx, (g, group) in enumerate(input.items()):
                new_group = output.create_group(g)
                for name, attr in group.attrs.items():
                    new_group.attrs.create(name, attr)
                for data_idx, (d, dataset) in enumerate(group.items()):
                    new_dataset = new_group.create_dataset(d, data=dataset[:])
                    for name, attr in dataset.attrs.items():
                        new_dataset.attrs.create(name, attr)
                    progress(['File', 'Group', 'Dataset'],
                             [file_idx, group_idx, data_idx],
                             [len(h5files), len(input.keys()), len(group.keys())])

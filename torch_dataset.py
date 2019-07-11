import torch
import glob
import h5py
import pickle
import os
import sys
import random
import numpy as np


class CHIMEDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, frb, height, width):
        super().__init__()
        self.file_path = file_path
        self.h5file = h5py.File(self.file_path, 'r')
        self.frb = frb
        self.shape = self.h5file[self.frb].shape

        # Enforce shape to be evenly divisible by height, dont want to have to
        # remove frequency channels to make it fit like we do for time/width
        self.height = height
        if self.shape[0] % self.height != 0:
            m = f"Invalid height {height} for data of shape {self.shape}"
            raise ValueError(m)

        # if width is not evenly divisible by shape remove time steps at end to
        # make it fit
        self.width = width
        if self.shape[1] % self.width != 0:
            total_width = (self.shape[1] // self.width) * self.width
        else:
            total_width = self.shape[1]

        self.n_rows = (self.shape[0] // self.height)
        self.n_cols = (total_width // self.width)

    def __getitem__(self, index):
        dset = self.h5file[self.frb]
        i = index % self.n_rows
        j = index // self.n_rows
        islice = np.s_[i * self.height:(i + 1) * self.height]
        jslice = np.s_[j * self.width:(j + 1) * self.width]
        x = np.zeros((self.height, self.width), dtype=np.float32)
        dset.read_direct(x, (islice, jslice))
        x = x.reshape((1,) + x.shape)
        assert np.array_equal(x.shape, [1, self.height, self.width])
        return (x, index)

    def __len__(self):
        return self.n_rows * self.n_cols

    def close(self):
        self.h5file.close()


class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, directory, set='split', transform=None, training=True,
                 rebalance=10000, verbose=False, force_rebuild=False):
        """
        directory: str, path to directory training.hdf5 and evaluation.hdf5
        set: str, one of ['complete', 'split', 'overview']
        transform: transform applied to every sample
        rebalance: int, max number of samples per class
        verbose: bool, print status messages while building dataset
        force_rebuild: bool, if True the dataset is resampled

        The HDF5 dataset should be formatted so that each group represents one
        sample of RFI, pulsar, or FRB. Each group should contain multiple
        datasets representing each of the subsections of the overall image, and
        one additional dataset named 'signal' containing the ideal output
        of the first step of the model.  Each group should have an attribute
        named label identifying the collection of all the datasets in the group.
        Each non-signal dataset should contain an attribute named label
        identifying whether a burst is present in that dataset.

        If set is 'split' the (x, y) pairs will be all the non-signal datasets
        and their associated labels. If set is 'overview' the (x, y) pairs will
        be all the signal datasets and their associated group's labels. If set
        is 'complete' all the non-signal datasets will be returned with a
        single label from their group.
        """
        super().__init__()
        self.data_info = []
        self.transform = transform
        self.set = set
        self.verbose = verbose
        self.rebalance = rebalance

        assert self.set in ['complete', 'split', 'overview']

        training_path = 'training' if training else 'evaluation'
        directory = directory + '/' if directory[-1] != '/' else directory
        info_path = directory + self.set + '-' + training_path + '.pkl'
        self.file_path = directory + training_path + '.hdf5'

        if os.path.exists(info_path) and not force_rebuild:
            if self.verbose:
                print(f"{self.set} dataset for {self.file_path} already built")
            with open(info_path, 'rb') as f:
                self.data_info = pickle.load(f)
        else:
            self.build_dataset()
            with open(info_path, 'wb') as f:
                pickle.dump(self.data_info, f)

        self.h5file = h5py.File(self.file_path, 'r')

    def build_dataset(self):
        # iterate through all files all groups and all datasets
        if self.verbose:
            print(f"Setting up {self.set} dataset for {self.file_path}...")

        n_classes = 2 if self.set == 'split' else 3
        classes = np.zeros(n_classes)

        self.data_info = []
        with h5py.File(self.file_path, 'r') as h5file:
            groups = list(h5file.items())
            random.shuffle(groups)
            for g, group in groups:
                if self.set == 'split': # ignore signal datasets
                    datasets = list(group.items())
                    random.shuffle(datasets)
                    for d, dataset in datasets:
                        if d == 'signal':
                            continue
                        if classes[int(dataset.attrs['label'])] <= self.rebalance:
                            self.data_info.append({'group': g, 'dataset': d})
                            classes[int(dataset.attrs['label'])] += 1
                            if self.verbose:
                                sys.stdout.write(f"class counts: {classes}\t\r")
                                sys.stdout.flush()
                        if np.sum(classes) >= self.rebalance * n_classes:
                            if self.verbose:
                                print("\nDone building dataset")
                            return
                else: # ignore all non-signal datasets
                    if classes[int(group.attrs['label'])] <= self.rebalance:
                        self.data_info.append({'group': g, 'dataset': 'signal'})
                        classes[int(group.attrs['label'])] += 1
                        if self.verbose:
                            sys.stdout.write(f"class counts: {classes}\t\r")
                            sys.stdout.flush()
                    if np.sum(classes) >= self.rebalance * n_classes:
                        if self.verbose:
                            print("\nDone building dataset")
                        return
            if self.verbose:
                print("\nDone building dataset")


    def __getitem__(self, index):
        def prep_dataset(dset):
            x = dset[:].astype(np.float32)
            x = x.reshape((1, x.shape[0], x.shape[1]))
            if self.transform is not None:
                x = self.transform(x)
            else:
                x = torch.from_numpy(x)
            return x
        info = self.data_info[index]
        group = self.h5file[info['group']]
        if self.set != 'complete':
            dataset = group[info['dataset']]
            x = prep_dataset(dataset)
            y = torch.as_tensor(dataset.attrs['label'])
            return (x, y)
        else:
            y = torch.as_tensor(group.attrs['label'])
            for key in group.keys():
                if key != 'signal':
                    break
            x_shape = (group['signal'].shape[0], group['signal'].shape[1],
                       1, group[key].shape[0], group[key].shape[1])
            x = torch.zeros(x_shape)
            for d, dataset in group.items():
                if d == 'signal': continue
                i = dataset.attrs["i"]
                j = dataset.attrs["j"]
                x[i][j] = prep_dataset(dataset)
            return (x, y)

    def __len__(self):
        return len(self.data_info)

    def close(self):
        self.data_info = None
        self.h5file.close()

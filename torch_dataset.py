import torch
from torch.utils.data import Dataset
import itertools
import glob
import h5py
import pickle
import os
import sys
import random
import numpy as np
from unprocessed_dataset import ABDataset
from skimage.draw import line_aa
from skimage.util.shape import view_as_blocks


class CHIMEDataset(Dataset):
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

def normalize(X):
    X /= np.sqrt(np.einsum('ij,ij->', X, X))

def make_filter(height=16, width=16):
    start_points = [[x, 0] for x in range(height)]
    start_points += [[0, x] for x in range(1, width)]

    end_points = [[height - 1 - x, width - 1]
                  for x in range(height)]
    end_points += [[height - 1, width - 1 - x]
                   for x in range(1, width)]

    points = itertools.product(start_points, end_points)
    points = [list(itertools.chain.from_iterable(p)) for p in points]

    points = [x for x in points if x[0] <= x[2] and x[1] <= x[3]]

    filter = np.zeros((len(points), height, width))
    for i, p in enumerate(points):
        rr, cc, val = line_aa(*p)
        filter[i, rr, cc] = 1000 * val

    filter = filter.reshape(-1, height * width)

    return filter


class SimpleLines(Dataset):
    def __init__(self, height=64, width=64, scale=1.0, N=1000,
                 filter_height=16, filter_width=16, filter=False,
                 add_channels=True):
        super().__init__()
        self.height = height
        self.width = width
        self.scale = scale
        self.N = N
        self.add_channels = add_channels

        self.r0 = np.random.randint(0, self.height, self.N)
        self.r1 = [np.random.randint(x, self.height) for x in self.r0]
        self.c0 = np.random.randint(0, self.width, self.N)
        self.c1 = [np.random.randint(x, self.width) for x in self.c0]

        self.filter_shape = (filter_height, filter_width)
        self.filter = None
        if filter:
            self.filter = make_filter(*self.filter_shape)

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        Y = torch.as_tensor(i % 2)
        x = np.random.normal(0, 1, (self.height, self.width))
        if i % 2: # inject feature into 50% of samples
            r, c, v = line_aa(self.r0[i], self.c0[i], self.r1[i], self.c1[i])
            x[r, c] += v.astype(np.float32) * self.scale

        if self.filter is not None:
            # get each filter shaped block from x
            x = view_as_blocks(x, self.filter_shape)
            block_shape = x.shape
            x = x.reshape((-1, x.shape[-2] * x.shape[-1]))

            # Element wise sum of the product of each filter with the cube of the
            # product of itself and x for each block in x
            x = np.einsum('ij,kj->ik', x, self.filter)
            x = np.einsum('ij,jk->ik', np.power(x, 3), self.filter)

            # put x back together
            x = x.reshape(block_shape)
            x = np.block([list(t) for t in x])

            x[x < 0] = 0 # ReLU

        if self.add_channels:
            x = x.reshape((1,) + x.shape)
        else: # if not adding channels assume treating as time series
            x = x.T

        # convert to a Tensor
        x = torch.from_numpy(x.astype(np.float32))

        # subtract mean and divide by standard deviation per channel
        X = x - torch.mean(x, dim=1, keepdim=True)
        X /= torch.std(x, dim=1, keepdim=True)
        return (X, Y)


class RNNDataset(Dataset):
    def __init__(self, n=2, length=16, N=1000, prob=1, scale=10):
        self.n = n
        self.length = length
        self.N = N
        self.prob = prob
        self.scale = scale

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        x = np.random.normal(60, 1, (self.length, self.n)).astype(np.float32)
        y = index % 2
        if y:
            cut = self.length // self.n
            for i in range(self.n):
                if not np.random.randint(0, 1 / self.prob):
                    start = max(0, (i * cut) + np.random.randint(0, 10))
                    stop = min(self.length - 1, ((i + 1) * cut) + np.random.randint(0, 10))
                    x[start:stop, i] += self.scale
        return (x, y)


class UnprocessedDataset(ABDataset):
    def __getitem__(self, index):
        info = self.info[index]

        xmin, xmax = info['xmin'], info['xmax']
        ymin, ymax = info['ymin'], info['ymax']
        group = self.h5file[info['group']]

        # signal = group['signal'][xmin:xmax, ymin:ymax]
        # signal = torch.from_numpy(signal).float()
        # signal = signal.reshape((1,) + signal.shape)

        background = group['background'][xmin:xmax, ymin:ymax]
        background = torch.from_numpy(background).float()
        background = background.reshape((1,) + background.shape)

        # TODO: background isn't actually the bacground, it contains enough to
        # distinguish between burst and no burst

        y = info['label']
        if y == 0:
            x = background
        else:
            x = background
        return (x, y)

# class UnprocessedDataset(Dataset):
#     def __init__(self, path, height, width, training=True, size=100,
#                  verbose=True):
#         self.height = height
#         self.width = width
#         self.training = training
#         self.size = size
#         self.verbose = verbose
# 
#         training_path = path + 'training.hdf5'
#         evaluation_path = path + 'evaluation.hdf5'
# 
#         assert os.path.exists(training_path)
#         assert os.path.exists(evaluation_path)
# 
#         if self.training:
#             self.path = training_path
#             info_path = path + 'unprocessed-train-info.pkl'
#         else:
#             self.path = evaluation_path
#             info_path = path + 'unprocessed-eval-info.pkl'
# 
#         with h5py.File(self.path, 'r') as f:
#             for group in f.keys():
#                 shape = f[group]['signal'].shape
#                 break
# 
#         try:
#             assert shape[0] % height == 0
#             assert shape[1] % width == 0
#         except AssertionError:
#             raise ValueError(f"height: {self.height} and width: {self.width} "\
#                              f"not compatible with data shape: {shape}")
# 
#         self.nrows = shape[0] // self.height
#         self.ncols = shape[1] // self.width
# 
#         if os.path.exists(info_path):
#             with open(info_path, 'rb') as f:
#                 self.data_info = pickle.load(f)
#         else:
#             self.data_info = self._setup_dataset(info_path)
# 
#     def __len__(self):
#         return len(self.data_info)
# 
#     def __getitem__(self, index):
#         info = self.data_info[index]
#         xrange, yrange = self._getranges(info['index'])
#         with h5py.File(self.path, 'r') as f:
#             group = f[info['group']]
#             signal = torch.from_numpy(group['signal'][xrange, yrange])
#             background = torch.from_numpy(group['background'][xrange, yrange])
#         return (signal + background, info['label'])
# 
#     def _getranges(self, index):
#         row = index % self.ncols
#         col = index // self.ncols
#         xrange = np.s_[row * self.height:(row + 1) * self.height]
#         yrange = np.s_[col * self.width:(col + 1) * self.width]
#         return xrange, yrange
# 
#     def _setup_dataset(self, path):
#         info = []
#         class_counts = np.zeros(2) # burst vs no burst
#         with h5py.File(self.path, 'r') as f:
#             for g in f.keys():
#                 if f[g].attrs["label"] == 2:
#                     continue
#                 signal = f[g]['signal']
#                 burst_indices = f[g]['burst_indices'][:]
#                 for d in range(self.nrows * self.ncols):
#                     xrange, yrange = self._getranges(d)
#                     indices = np.indices(signal.shape)[:, xrange, yrange]
#                     burst = int(np.any(
#                         np.in1d(burst_indices[0], indices[0]) &
#                         np.in1d(burst_indices[1], indices[1])))
#                     if class_counts[burst] < self.size:
#                         info.append({'group': g, 'index': d, 'label': burst})
#                         class_counts[burst] += 1
#                     if self.verbose:
#                         sys.stdout.write(f"{class_counts}\r")
#                         sys.stdout.flush()
#                     if np.sum(class_counts) >= 2 * self.size:
#                         break
#                 if np.sum(class_counts) >= 2 * self.size:
#                     break
#         with open(path, 'wb') as f:
#             pickle.dump(info, f)
#         if self.verbose:
#             print("")
#         return info
# 
# 
# class UnsplitDataset(torch.utils.data.Dataset):
#     def __init__(self, dir, height, width, set, transform=None, training=True):
#         """
#         dir: str, directory containing training.hdf5 and evaluation.hdf5
#         height: int, the height in pixels of the image subsections
#         width: int, the width in pixels of the image subsections
#         transform: transform applied to each sample
#         training: bool, whether to use trainging.hdf5 or evaluation.hdf5
#         """
#         super().__init__()
#         self.dir = dir
#         self.height = height
#         self.width = width
#         self.transform = transform
#         self.training = training
#         self.set = set
# 
#         assert self.set in ['overview', 'split']
# 
#         assert os.path.exists(self.dir + 'training.hdf5')
#         assert os.path.exists(self.dir + 'evaluation.hdf5')
# 
#         with h5py.File(self.dir + 'training.hdf5', 'r') as f:
#             for k in f.keys():
#                 temp_data = f[k]
#                 break
#             assert temp_data.shape[0] % self.height == 0
#             assert temp_data.shape[1] % self.width == 0
# 
#         self.nrows = temp_data.shape[0] // self.height
#         self.ncols = temp_data.shape[1] // self.width
# 
#         if self.training:
#             self.h5file = h5py.File(self.dir + 'training.hdf5')
#         else:
#             self.h5file = h5py.File(self.dir + 'evaluation.hdf5')
# 
#         self.ndatasets = len(list(self.h5file.keys()))
# 
#     def __len__(self):
#         if self.set == 'split':
#             length = self.ndatasets * self.nrows * self.ncols
#         else:
#             length = self.ndatasets
#         return length
# 
#     def __getitem__(self, index):
#         if self.set == 'split':
#             (x, y) = self.get_split_item(self, index)
#         else:
#             (x, y) = self.get_overview_item(self, index)
# 
#         x.reshape((1,) + x.shape) # add channels to data
# 
#         if transform is not None:
#             x = transform(x)
#         else:
#             x = torch.from_numpy(x)
# 
#         return (x, y)
# 
#     def get_split_item(self, index):
#         dataset_index = index // self.ndatasets
#         image_index = index % self.ndatasets
# 
#         i = image_index % self.nrows
#         j = image_index // self.nrows
# 
#         xs = np.s_[i * self.height:(i + 1) * self.height]
#         ys = np.s_[j * self.width:(j + 1) * self.width]
# 
#         dset = self.h5file[dataset_index]
# 
#         y = self.contains_burst(dset.shape, xs, ys, dset.attrs['burst_indices'])
# 
#         x = np.zeros((self.height, self.width), dtype=np.float32)
#         dset.direct_read(x, (xs, ys))
#         return (x, y)
# 
#     def get_overview_item(self, index):
#         dset = self.h5file[index]
#         burst_indices = dset.attrs['burst_indices']
# 
#         all_xs = [np.s_[i * self.height:(i + 1) * self.height]
#                   for i in range(self.nrows)]
#         all_ys = [np.s_[j * self.width:(j + 1) * self.width]
#                   for j in range(self.ncols)]
# 
#         x = np.zeros(self.nrows * self.ncols)
#         for i, (xs, ys) in enumerate(itertools.product(all_xs, all_ys)):
#             x[i] = self.contains_burst(dst.shape, xs,ys, burst_indices)
#         x = x.reshape((self.nrows, self.ncols))
# 
#         y = dset.attrs['label']
#         return (x, y)
# 
#     def contains_burst(self, shape, xrange, yrange, burst_indices):
#         subset_indices = np.indices(shape)[:, xlow:xhigh, ylow:yhigh]
#         output = np.any(np.in1d(burst_indices[0], subset_indices[0]) &
#                         np.in1d(burst_indices[1], subset_indices[1]))
#         return int(output)
# 
#     def close(self):
#         self.h5file.close()


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

        n_classes = 2 #if self.set == 'split' else 2
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
                    if int(group.attrs['label']) >= 1: # hack for no pulsar dataset
                        label_index = 1
                    else:
                        label_index = 0
                    if classes[label_index] <= self.rebalance:
                        self.data_info.append({'group': g, 'dataset': 'signal'})
                        classes[label_index] += 1
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
            if (dataset.attrs['label']) >= 1:
                y = torch.as_tensor(1)
            else:
                y = torch.as_tensor(dataset.attrs['label'])
        else:
            if int(group.attrs['label']) >= 1:
                y = torch.as_tensor(1)
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


if __name__ == '__main__':
    dset = UnprocessedDataset(
            path='/scratch/r/rhlozek/rylan/unprocessed_data/',
            height=128,
            width=256,
            size=5000,
            verbose=True,
            training=True)

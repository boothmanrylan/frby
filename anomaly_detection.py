import argparse
import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import horovod.torch as hvd

hvd.init()

seed = 12345
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.set_device(hvd.local_rank())
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', 128)
parser.add_argument('--image-size', 64)
parser.add_argument('--nc', 1)
parser.add_argument('--nz', 100)
parser.add_argument('--ngf', 64)
parser.add_argument('--ndf', 64)
parser.add_argument('--num-epochs', 5)
parser.add_argument('--lr', 0.0002)
parser.add_argument('--beta1', 0.5)
args = parser.parse_args()

# TODO: need dataset and dataloader that will read in noise from ARO

class RFIDataset(Dataset):
    def __init__(self, path, size, training=True, N=100):
        super().__init__()
        self.path = path
        self.size = size
        self.training = training
        self.N = N

        self.training_path = path + 'training.hdf5'
        self.evaluation_path = path + 'evaluation.hdf5'

        if self.training:
            self.h5file = h5py.File(self.training_path, 'r')
        else:
            self.h5file = h5py.File(self.evaluation_path, 'r')

        for group in self.h5file.keys():
            self.shape = self.h5file[group]['signal'].shape
            break

        self.nrows = self.shape[0] // self.size
        self.ncols = self.shape[1] // self.size

        info = path + f"single_class_{N}_{size}.pkl"

        if os.path.exists(info):
            with open(info, 'rb') as f:
                self.info = pickle.read(f)
        else:
            self.info = self.setup_dataset()
            with open(info, 'wb') as f:
                pickle.dump(self.info, f)

    def __getitem__(self, index):
        return index

    def __len__(self):
        return self.N

    def close(self):
        self.h5file.close()

    def setup_dataset(self):
        info = []
        while len(info) < self.N:

class D(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(args.nc, args.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(args.ndf, args.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(args.ndf * 2, args.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(args.ndf * 4, args.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(args.ndf * 8, 1, 4, 1, 0, bias=False)
        )

        self.output = nn.Sequential(
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.output(self.block2(self.block1(x)))

    def intermediate(self, x):
        return self.block1(x)

class G(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(args.nz, args.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(args.ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(args.ngf * 8, args.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(args.ngf * 4, args.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(args.ngf * 2, args.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(args.ngf, args.nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

netD = D()
netD.cuda()
netD.apply(weights_init)

netG = G()
netG.cuda()
netG.apply(weigths_init)

criterion = nn.BCELoss()

latent_vectors = torch.randn(64, args.nz, 1, 1).cuda()

real_label = 1
fake_label = 0

optimD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimD = hvd.DistributedOptimizer(optimD, netD.named_parameters())

optimG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimG = hvd.DistributedOptimizer(optimG, netG.named_parameters())

for epoch in range(args.num_epochs):
    for i, data in enumerate(dataloader):
        data = data[0].cuda()

        netD.zero_grad()
        label = torch.full((data.size(0),), real_label).cuda()

        output = netD(data).view(-1)

        errD_real = criterion(output, label)
        errD_real.backward()

        noise = torch.randn(data.size(0), args.nz, 1, 1).cuda()
        fake = netG(noise)

        label.fill_(fake_label)

        output = netD(fake.detach()).view(-1)

        errD_fake = criterion(output, label)
        errD_fake.backward()

        optimD.step()

        netG.zero_grad()
        label.fill_(real_label)

        output = netD(fake).view(-1)

        errG = criterion(output, label)
        errG.backward()

        optimG.step()

for i, data in enumerate(eval_data_loader):
    loss = (1 -
    for _ in range(args.gamma):


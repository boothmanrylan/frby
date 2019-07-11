import argparse
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import torchvision.models as models
from torchvision.datasets import CIFAR10
from torchvision import transforms

import horovod.torch as hvd

from torch_dataset import HDF5Dataset, CHIMEDataset

vgg16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256,
         'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

SCRATCH = '/scratch/r/rhlozek/rylan/'

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--data', type=str, default=SCRATCH + 'split_data/')
parser.add_argument('-lr', type=float, default=0.0125)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--wd', type=float, default=5e-5)
parser.add_argument('--lscale', type=float, default=1)
parser.add_argument('--T', type=int, default=25)
parser.add_argument('--classes', type=int, default=10)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--checkpoint', type=str, default=SCRATCH+'models/default/')
args = parser.parse_args()


class AlwaysOnDropout(nn.Module):
    """
    Dropout layer that doesn't turn into the identity during evaluation
    """
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x):
        """
        Forcing training to always be true forces dropout to always be applied
        """
        x = F.dropout(x, p=self.p, training=True, inplace=self.inplace)
        return x


class Model(nn.Module):
    def __init__(self, nclasses, config, batch_norm=True, in_channels=1,
                 dropoutP=0.5, bayes=False):
        super().__init__()
        self.nclasses = nclasses
        self.p = dropoutP
        self.bayes = bayes
        self.N = None # the number of samples used to train the model
        layers = []

        dropout = AlwaysOnDropout(self.p) if self.bayes else nn.Dropout(self.p)

        for v in config:
            if v == 'M':
                if self.bayes: layers += [dropout]
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(True)]
                else:
                    layers += [conv2d, nn.ReLU(True)]
                in_channels = v

        self.vgg = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            dropout,
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            dropout,
            nn.Linear(4096, self.nclasses),
        )

    def forward(self, x):
        x = self.vgg(x)
        x = nn.AdaptiveAvgPool2d((7, 7))(x)
        x = x.view(x.size(0), -1) # flatten
        x = self.classifier(x)
        return x


def bayes(model, x, T):
    """
    See: https://arxiv.org/pdf/1506.02142.pdf
    and also: http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html

    model : neural network with AlwaysOnDropout before all pooling layers
    x : 1 batch of input data
    T : int, number of times to pass each sample through the model

    Returns: (predictions, likelihoods)
    """
    ys = torch.zeros((T, x.shape[0], model.nclasses)).cuda()
    for i in range(T):
        ys[i] = nn.Softmax(1)(model(x))
    return torch.mean(ys, 0)


def train(model, optimizer, data_loader, epochs, checkpoint_path=None):
    # load model from checkpoint if it exists
    if hvd.rank() == 0 and checkpoint_path is not None:
        try:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            model.cuda()
        except FileNotFoundError:
            start_epoch = 0
    else:
        start_epoch = 0

    # broadcast rank 0 model/optimizer state and starting epoch to other ranks
    hvd.broadcast(torch.tensor(start_epoch), root_rank=0, name='start_epoch')
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    if hvd.rank() == 0: print('Begin training loop...')

    # place model in training mode
    model.train()
    for epoch in range(start_epoch, epochs):
        for batch, (data, target) in enumerate(data_loader):
            data, target = data.cuda(), target.cuda() # move data to GPU
            optimizer.zero_grad() # clear the gradients of all tensors
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward() # compute gradients
            optimizer.step() # update parameters
            if hvd.rank() == 0:
                print(f"Epoch: {epoch+1}\tBatch: {batch+1}\tLoss: {loss.item()}")
        # save checkpoint after every 10th epoch and after the final epoch
        if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            if hvd.rank() == 0 and checkpoint_path is not None:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, checkpoint_path)


def gather(val, name):
    tensor = val.clone().detach()
    avg_tensor = hvd.allreduce(tensor, name='name')
    # check if there is a single element in avg_tensor
    if torch.tensor(avg_tensor.shape).eq(1).all():
        avg_tensor = avg_tensor.item()
    return avg_tensor


def evaluate(model, data_loader, length, checkpoint_path=None):
    # load model from checkpoint if it exists
    if hvd.rank() == 0 and checkpoint_path is not None:
        try:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.cuda()
        except FileNotFoundError:
            pass

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    if hvd.rank() == 0: print('Begin evaluation loop...')

    # place model in evaluation mode
    model.eval()
    with torch.no_grad():
        accuracy = 0.0
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.cuda(), target.cuda() # move data to GPU
            if model.bayes:
                out = bayes(model, data, args.T)
            else:
                out = model(data)
            pred = out.argmax(dim=1, keepdim=True)
            accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()
        accuracy /= length
        accuracy = gather(accuracy, 'avg_accuracy')
        if hvd.rank() == 0:
            print(f"Accuracy: {accuracy}")


def build_data_loader(dataset, dataset_kwargs, loader_kwargs):
    dset = dataset(**dataset_kwargs)
    sampler = DistributedSampler(dset, num_replicas=hvd.size(), rank=hvd.rank())
    loader = DataLoader(dset, sampler=sampler, **loader_kwargs)
    return dset, sampler, loader


def add_noise(x):
    """
    Adds gaussian noise to x then normalizes to [0-1]
    """
    x += np.random.normal(0, 0.3, x.shape)
    x = (x - x.min()) / (x.max() - x.min())
    return torch.from_numpy(x)


def two_step_model(inner_model, outer_model, data_loader, length, checkpoint_path):
    # load model from checkpoint if it exists
    if hvd.rank() == 0 and checkpoint_path is not None:
        try:
            checkpoint = torch.load(checkpoint_path + 'inner_model')
            inner_model.load_state_dict(checkpoint['model_state_dict'])
            inner_model.cuda()
            checkpoint = torch.load(checkpoint_path + 'outer_model')
            outer_model.load_state_dict(checkpoint['model_state_dict'])
            outer_model.cuda()
        except FileNotFoundError:
            pass

    # broadcast rank 0 model/optimizer state and starting epoch to other ranks
    hvd.broadcast_parameters(inner_model.state_dict(), root_rank=0)
    hvd.broadcast_parameters(outer_model.state_dict(), root_rank=0)

    inner_model.eval()
    outer_model.eval()
    with torch.no_grad():
        accuracy = 0.0
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.cuda(), target.cuda() # move data to GPU
            for D in data: # iterate across the batch
                d = D.reshape((D.shape[0] * D.shape[1],) + D.shape[2:])
                if inner_model.bayes:
                    outputA = bayes(inner_model, d, args.T)[:, 0]
                else:
                    outputA = inner_model(d).argmax(dim=1, keepdim=True)
                outputA = outputA.reshape((1, 1, D.shape[0], D.shape[1]))
                if outer_model.bayes:
                    predB = bayes(outer_model, outputA, args.T)
                else:
                    predB = outer_model(outputA.float().cuda())
                pred = predB.argmax(dim=1, keepdim=True)
                accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()
        accuracy /= length
        accuracy = gather(accuracy, 'avg_accuracy')
        if hvd.rank() == 0:
            print(f"Accuracy: {accuracy}")


def chime_frbs(model, checkpoint_path, chime_path, height, width):
    if hvd.rank() == 0:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.cuda()

    # broadcast rank 0 model state to other ranks
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    model.eval()

    frbs = os.listdir(f'{chime_path}/msgpacks')
    frbs.remove('repeater')
    for frb in frbs:
        if not hvd.rank(): print(f"working on {frb}...")
        dset, sampler, loader = build_data_loader(CHIMEDataset,
            {'file_path': f'{chime_path}/reduced_chime_frbs.hdf5',
             'frb': frb, 'height': height, 'width': width},
            {'batch_size': args.batch_size, 'num_workers': 0,
             'pin_memory': False}
        )

        with torch.no_grad():
            output = torch.zeros(len(dset)).cuda()
            for x, index in loader:
                if model.bayes:
                    model_out = bayes(model, x.cuda(), args.T)[:, 0]
                else:
                    model_out = model(x.cuda()).argmax(dim=1, keepdim=False)
                output[index] = model_out.float()

            # get the results from all gpus
            output = gather(output, 'output')

            # reshape and transpose to match original data pattern
            # columns and rows reversed because of transpose
            shape = (dset.n_cols, dset.n_rows)
            output = output.reshape(shape).transpose(0, 1)

            if not hvd.rank():
                np.save(f'{chime_path}/point_output/{frb}.npy', np.asarray(output.cpu()))

            dset.close()


if __name__ == '__main__':
    hvd.init()
    torch.manual_seed(args.seed)
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    #################################################################
    # Train the inner model
    #################################################################
    s_train_dset, s_train_sampler, s_train_loader = build_data_loader(HDF5Dataset,
        {'directory': args.data, 'set': 'split', 'training': True,
         'rebalance': 10000, 'verbose': not hvd.rank(), 'transform': None},
        {'batch_size': args.batch_size, 'num_workers': 0, 'pin_memory': True}
    )

    inner_model = Model(nclasses=2, config=vgg16, bayes=False)
    inner_model.cuda() # place model on GPU. Must be done before constructing optimizer.
    # inner_optimizer = optim.SGD(inner_model.parameters(),
    #                             lr=args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.wd)

    # # Distribute optimizer with horovod
    # inner_optimizer = hvd.DistributedOptimizer(
    #     inner_optimizer,
    #     named_parameters=inner_model.named_parameters()
    # )

    # train(inner_model, inner_optimizer, s_train_loader,
    #       args.epochs, args.checkpoint + 'inner_model')
    inner_model.N = len(s_train_sampler)

    data_shape = s_train_dset[0][0].shape

    s_train_dset.close()

    #################################################################
    # Evaluate the inner model
    #################################################################
    # s_eval_dset, s_eval_sampler, s_eval_loader = build_data_loader(HDF5Dataset,
    #     {'directory': args.data, 'set': 'split', 'training': False,
    #      'rebalance': 10000, 'verbose': not hvd.rank(), 'transform': None},
    #     {'batch_size': args.batch_size, 'num_workers': 0, 'pin_memory': True}
    # )

    # evaluate(inner_model, s_eval_loader, len(s_eval_sampler),
    #          args.checkpoint + 'inner_model')

    # s_eval_dset.close()

    #################################################################
    # Train the outer model
    #################################################################
    # o_train_dset, o_train_sampler, o_train_loader = build_data_loader(HDF5Dataset,
    #     {'directory': args.data, 'set': 'overview', 'training': True,
    #      'rebalance': 10000, 'verbose': not hvd.rank(), 'transform': add_noise},
    #     {'batch_size': args.batch_size, 'num_workers': 0, 'pin_memory': True}
    # )

    # outer_model = Model(nclasses=3, config=vgg16, bayes=True)
    # outer_model.cuda() # place model on GPU. Must be done before constructing optimizer.
    # outer_optimizer = optim.SGD(outer_model.parameters(),
    #                             lr=args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.wd)

    # # Distribute optimizer with horovod
    # outer_optimizer = hvd.DistributedOptimizer(
    #     outer_optimizer,
    #     named_parameters=outer_model.named_parameters()
    # )

    # train(outer_model, outer_optimizer, o_train_loader,
    #       args.epochs, args.checkpoint + 'outer_model')
    # outer_model.N = len(o_train_sampler)

    # o_train_dset.close()

    #################################################################
    # Evaluate the outer model
    #################################################################
    # o_eval_dset, o_eval_sampler, o_eval_loader = build_data_loader(HDF5Dataset,
    #     {'directory': args.data, 'set': 'overview', 'training': False,
    #      'rebalance': 10000, 'verbose': not hvd.rank(), 'transform': add_noise},
    #     {'batch_size': args.batch_size, 'num_workers': 0, 'pin_memory': True}
    # )

    # evaluate(outer_model, o_eval_loader, len(o_eval_sampler),
    #          args.checkpoint + 'outer_model')

    # o_eval_dset.close()

    #################################################################
    # Evaluate the two models combined
    #################################################################
    # eval_dset, eval_sampler, eval_loader = build_data_loader(HDF5Dataset,
    #     {'directory': args.data, 'set': 'complete', 'training': False,
    #      'rebalance': 10000, 'verbose': not hvd.rank(), 'transform': None},
    #     {'batch_size': 1, 'num_workers': 0, 'pin_memory': True}
    # )

    # two_step_model(inner_model, outer_model, eval_loader,
    #                len(eval_sampler), args.checkpoint)

    # eval_dset.close()

    #################################################################
    # Evaluate the first model on CHIME data
    #################################################################
    chime_path = '/scratch/r/rhlozek/rylan/chime_data'
    chime_frbs(inner_model, args.checkpoint + 'inner_model', chime_path,
               int(data_shape[1]), int(data_shape[2]))



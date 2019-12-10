import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line_aa
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class SimpleLines(Dataset):
    def __init__(self, height=64, width=64, scale=1.0, N=1000):
        super().__init__()
        self.height = height
        self.width = width
        self.scale = scale
        self.N = N

        self.r0 = np.random.randint(0, int(0.8*self.height), self.N)
        self.r1 = [np.random.randint(int(1.25*x), self.height) for x in self.r0]
        self.c0 = np.random.randint(0, int(0.8*self.width), self.N)
        self.c1 = [np.random.randint(int(1.25*x), self.width) for x in self.c0]

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        Y = torch.as_tensor(i % 2)
        x = np.random.normal(0, 1, (1, self.height, self.width))
        x = torch.from_numpy(x.astype(np.float32))
        if i % 2:
            r, c, v = line_aa(self.r0[i], self.c0[i], self.r1[i], self.c1[i])
            x[0, r, c] += torch.from_numpy(v.astype(np.float32) * self.scale)
        X = x - torch.mean(x, dim=1, keepdim=True)
        X /= torch.std(x, dim=1, keepdim=True)
        return (X, Y)


class ResidualBlock(nn.Module):
    def __init__(self, Cin, Cout, kernel_size=3, stride=1, dilation=1):
        super().__init__()

        # pad convs with kernel_size // 2 so kernel size doesn't affect shape
        self.conv1 = nn.Conv2d(Cin, Cout, kernel_size,
                               padding=(dilation * kernel_size) // 2,
                               stride=stride, dilation=dilation, bias=False)
        self.conv2 = nn.Conv2d(Cout, Cout, kernel_size,
                               padding=kernel_size // 2,
                               stride=1, dilation=1, bias=False)
        self.bn = nn.BatchNorm2d(Cout)

        # skip connection needs to be downsampled if convolution is strided or
        # if the convolution is dilated or if the number of channels changes
        self.skip = nn.Identity()
        if Cin != Cout or stride > 1 or dilation > 1:
            self.skip = nn.Sequential(
                nn.Conv2d(Cin, Cout, 1, stride=stride, bias=False),
                nn.BatchNorm2d(Cout)
            )

    def forward(self, x):
        identity = self.skip(x)

        x = self.conv1(x)
        x = self.bn(x)
        x = nn.ReLU(True)(x)

        x = nn.Dropout()(x)

        x = self.conv2(x)
        x = self.bn(x)

        x = nn.ReLU(True)(x)# + identity)

        return x

class ResidualNetwork(nn.Module):
    def __init__(self, input_shape, reduced_shape=(342, 256), verbose=False):
        super().__init__()

        self.input_shape = input_shape
        self.reduced_shape = reduced_shape
        self.verbose = verbose

        # block 0's purpose is to downsample input to reduced_shape if this
        # isn't necessary make block 0 the identity
        stride0 = max((self.input_shape[0] // (reduced_shape[0] - 1)), 1)
        stride1 = max((self.input_shape[1] // (reduced_shape[1] - 1)), 1)
        stride = (stride0, stride1)
        self.block0 = nn.Sequential(
            nn.Conv2d(1, 1, stride, stride=stride, dilation=3, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

        self.block1 = ResidualBlock(1,   32,  kernel_size=7, stride=2)

        self.block2 = ResidualBlock(32,  32,  kernel_size=3, stride=4)
        self.block3 = ResidualBlock(32,  32,  kernel_size=3)

        self.block4 = ResidualBlock(32,  64,  kernel_size=3, stride=4)
        self.block5 = ResidualBlock(64,  64,  kernel_size=3)
        self.block6 = ResidualBlock(64,  64,  kernel_size=3)

        self.block7 = ResidualBlock(64,  128, kernel_size=3, stride=2)
        self.block8 = ResidualBlock(128, 128, kernel_size=3)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        if self.verbose: self.show(x, 1)

        x = self.block0(x)
        if self.verbose: self.show(x, 2)

        x = self.block1(x)
        if self.verbose: self.show(x, 3)

        x = self.block2(x)
        x = self.block3(x)
        if self.verbose: self.show(x, 4)

        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        if self.verbose: self.show(x, 5)

        x = self.block7(x)
        x = self.block8(x)
        if self.verbose: self.show(x, 6)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)

        if self.verbose: plt.show()

        return x

    def show(self, x, fignum):
        im1 = x.detach().numpy()[0]
        im2 = x.detach().numpy()[1]
        if len(im1.shape) > 2:
            im1 = im1.mean(0)
        if len(im2.shape) > 2:
            im2 = im2.mean(0)
        f = plt.figure(fignum)
        plt.subplot(121)
        plt.imshow(im1, aspect='auto', interpolation='nearest')
        plt.subplot(122)
        plt.imshow(im2, aspect='auto', interpolation='nearest')

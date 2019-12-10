import torch.nn as nn

from utils import Flatten


class ResidualBlock(nn.Module):
    """
    Building block for ResidualNetwork, has the following layers:

    2d-convolution
    batch normalization
    ReLU
    dropout
    2d-convolution (with stride=1 and dilation=1)
    batch normalization
    add skip connection*
    ReLU

    *The skip connection is identical to the input in the case where the
    stride is zero, the dilation is 1, and the number of channels/features does
    not change. Otherwise the skip connection is downsampled using 2d
    convolution with a kernel_size of 1 and the given stride followed by batch
    normalization.
    """
    def __init__(self, Cin, Cout, kernel_size=3, stride=1, dilation=1):
        """
        See pytorch torch.nn.Conv2d for a more detailed description of
        parameters.

        stride and dilation are always set to 1 for the second convolution.
        """
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
        skip_connection = self.skip(x)

        x = self.conv1(x)
        x = self.bn(x)
        x = nn.ReLU(True)(x)

        x = nn.Dropout()(x)

        x = self.conv2(x)
        x = self.bn(x)

        x = nn.ReLU(True)(x + skip_connection)

        return x


class ResidualNetwork(nn.Module):
    """
    Residual neural network with 17 convolutional layers meant to be as close
    as possible to the architecture described in Zhang et al. arxiv:1809.03043.

    Note that in order to get the output sizes described in the paper the input
    sizes must also match: 10928 frequency channels by 256 time samples.
    """
    def __init__(self, input_shape, reduced_shape=(342, 256)):
        """
        input_shape: the shape of the data that will be passed to the model
        reduced_shape: the desired shape of the data after the first layer
        """
        super().__init__()

        self.input_shape = input_shape
        self.reduced_shape = reduced_shape

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

        self.fc = nn.Sequential(Flatten, nn.Linear(128, 2))

    def forward(self, x):
        x = self.block0(x)

        x = self.block1(x)

        x = self.block2(x)
        x = self.block3(x)

        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        x = self.block7(x)
        x = self.block8(x)

        x = self.avgpool(x)

        x = self.fc(x)

        return x

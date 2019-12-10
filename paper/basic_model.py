import torch.nn as nn

from utils import Flatten

class BasicModel(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings

        in_channels = 1
        layers = []
        for v in self.settings.config:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if self.settings.batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(True)]
                else:
                    layers += [conv2d, nn.ReLU(True)]
                in_channels = v
        layers.extend([
            nn.AdaptiveAvgPool2d((7, 7)),
            Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, self.settings.nclasses)
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

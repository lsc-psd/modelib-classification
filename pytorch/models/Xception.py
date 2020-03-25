import torch.nn as nn
import torch.nn.functional as F
import math

from torch.hub import load_state_dict_from_url

"""
Depse-width separable Conv Intro:
https://medium.com/@zurister/depth-wise-convolution-and-depth-wise-separable-convolution-37346565d4ec
"""

__all__ = ['Xception']
model_url = ''


class Xception(nn.Module):
    def __init__(self, num_classes=1000, init_weights=True, pretrain=False):
        super(Xception, self).__init__()
        self._build(num_classes)

        # automatically abandon init_weight if pretrain is True
        if pretrain:
            assert model_url is not '', f'Pretrained model for {self.__class__.__name__} not prepared yet.'
            state_dict = load_state_dict_from_url(model_url,
                                                  progress=True)
            self.load_state_dict(state_dict)
        elif init_weight:
            self._init_weights()

    def _build(self, num_classes):
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.block1 = ExceptionBlockA(64, 128, first_block=True)
        self.block2 = ExceptionBlockA(128, 256)
        self.block3 = ExceptionBlockA(256, 728)

        self.block4 = ExceptionBlockB(728)
        self.block5 = ExceptionBlockB(728)
        self.block6 = ExceptionBlockB(728)
        self.block7 = ExceptionBlockB(728)
        self.block8 = ExceptionBlockB(728)
        self.block9 = ExceptionBlockB(728)
        self.block10 = ExceptionBlockB(728)
        self.block11 = ExceptionBlockB(728)

        self.block12 = ExceptionBlockC(728,1024)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.relu4 = nn.ReLU(inplace=True)

        self.fc = nn.Linear(2048, num_classes)

    def _init_weights(self):
        for m in self.modules():
             if isinstance(m, nn.Conv2d):
                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                 m.weight.data.normal_(0, math.sqrt(2. / n))
             elif isinstance(m, nn.BatchNorm2d):
                 m.weight.data.fill_(1)
                 m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)

        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        # By using groups parameter, automatically added 1 conv per in_channel
        self.mult_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                                   padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1,
                                   0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.mult_conv(x)
        x = self.pointwise(x)
        return x


class ExceptionBlockA(nn.Module):
    """
    A grow first
    """
    def __init__(self, in_channels, out_channels, first_block=False):
        super(ExceptionBlockA, self).__init__()

        shortcut = []
        shortcut.append(nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False))
        shortcut.append(nn.BatchNorm2d(out_channels))
        self.shortcut = nn.Sequential(*shortcut)


        layers = []
        layers.append(nn.ReLU(inplace=True))
        layers.append(SeparableConv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.ReLU(inplace=True))
        layers.append(SeparableConv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.MaxPool2d(3, 2, 1))

        if first_block:
            layers = layers[1:]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        res = self.shortcut(x)
        x = self.layers(x)
        x += res
        return x


class ExceptionBlockB(nn.Module):
    def __init__(self, channels):
        super(ExceptionBlockB, self).__init__()

        layers = []
        layers.append(nn.ReLU(inplace=True))
        layers.append(SeparableConv2d(channels, channels, 3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(channels))

        layers.append(nn.ReLU(inplace=True))
        layers.append(SeparableConv2d(channels, channels, 3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(channels))

        layers.append(nn.ReLU(inplace=True))
        layers.append(SeparableConv2d(channels, channels, 3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(channels))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        res = x
        x = self.layers(x)
        x += res
        return x


class ExceptionBlockC(nn.Module):
    """
    C grow at last block
    """
    def __init__(self, in_channels, out_channels, first_block=False):
        super(ExceptionBlockC, self).__init__()

        shortcut = []
        shortcut.append(nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False))
        shortcut.append(nn.BatchNorm2d(out_channels))
        self.shortcut = nn.Sequential(*shortcut)

        layers = []
        layers.append(nn.ReLU(inplace=True))
        layers.append(SeparableConv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.ReLU(inplace=True))
        layers.append(SeparableConv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.MaxPool2d(3, 2, 1))

        if first_block:
            layers = layers[1:]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        res = self.shortcut(x)
        x = self.layers(x)
        x += res
        return x

import torch
import torch.nn as nn

__all__ = ['ResNet50']


class ResNet50(nn.Module):

    def __init__(self, num_classes=1000, init_weight=True):
        super(ResNet50, self).__init__()

        self.inplanes = 64
        self.dilation = 1
        self._build(num_classes)
        if init_weight:
            self._init_weights()

    def _build(self, num_classes):
        self.conv1 = Conv2d(3, 64, kernel_size=7, relu=True,
                            stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        r1_channel = 64
        self.ResBlock1_1 = Resblock.first(64, r1_channel)
        self.ResBlock1_2 = Resblock.follow(r1_channel)
        self.ResBlock1_3 = Resblock.follow(r1_channel)

        r2_channel = 128
        self.ResBlock2_1 = Resblock.first(r1_channel*4, r2_channel, stride=2)
        self.ResBlock2_2 = Resblock.follow(r2_channel)
        self.ResBlock2_3 = Resblock.follow(r2_channel)
        self.ResBlock2_4 = Resblock.follow(r2_channel)

        r3_channel = 256
        self.ResBlock3_1 = Resblock.first(r2_channel*4, r3_channel, stride=2)
        self.ResBlock3_2 = Resblock.follow(r3_channel)
        self.ResBlock3_3 = Resblock.follow(r3_channel)
        self.ResBlock3_4 = Resblock.follow(r3_channel)
        self.ResBlock3_5 = Resblock.follow(r3_channel)
        self.ResBlock3_6 = Resblock.follow(r3_channel)

        r4_channel = 512
        self.ResBlock4_1 = Resblock.first(r3_channel*4, r4_channel, stride=2)
        self.ResBlock4_2 = Resblock.follow(r4_channel)
        self.ResBlock4_3 = Resblock.follow(r4_channel)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.ResBlock1_1(x)
        x = self.ResBlock1_2(x)
        x = self.ResBlock1_3(x)

        x = self.ResBlock2_1(x)
        x = self.ResBlock2_2(x)
        x = self.ResBlock2_3(x)
        x = self.ResBlock2_4(x)

        x = self.ResBlock3_1(x)
        x = self.ResBlock3_2(x)
        x = self.ResBlock3_3(x)
        x = self.ResBlock3_4(x)
        x = self.ResBlock3_5(x)
        x = self.ResBlock3_6(x)

        x = self.ResBlock4_1(x)
        x = self.ResBlock4_2(x)
        x = self.ResBlock4_3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x


class Resblock(nn.Module):
    def __init__(self, in_channels, channels, stride, dilation):
        super(Resblock, self).__init__()

        self.conv1 = Conv2d(in_channels, channels, 1, bias=False, stride=1)
        self.conv2 = Conv2d(channels, channels,  3, bias=False, stride=stride,
                            dilation=dilation, padding=dilation)
        self.conv3 = Conv2d(channels, channels*4, 1,
                            bias=False, stride=1, relu=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

    @classmethod
    def first(cls, in_channels, channels, stride=1, dilation=1):
        """
        y[channel*4] = downsample(x[in_channel]) + conv1*1(conv3x3(conv1x1(x[in_channel])))
        """
        block = cls(in_channels, channels, stride, dilation)
        block.downsample = Downsample(in_channels, channels*4, stride)
        return block

    @classmethod
    def follow(cls, channels):
        """
        y[channel*4] = x[channel*4] + conv1*1(conv3x3(conv1x1(x[channel*4])))
        """
        block = cls(channels*4, channels, 1, 1)
        return block

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Downsample(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(Downsample, self).__init__()
        self.conv = Conv2d(inplanes, planes, 1, relu=False, bias=False, stride=stride)

    def forward(self, x):
        x = self.conv(x)
        return x

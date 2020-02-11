import torch
import torch.nn as nn

__all__ = ['ResNet50']


class ResNet50(nn.Module):

    def __init__(self, num_classes=1000, init_weight=True):
        super(ResNet50, self).__init__()

        self.inplanes = 64
        self.dilation = 1
        self.base_width = 64
        self._build(num_classes)
        if init_weight:
            self._init_weights()

    def _build(self, num_classes):
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        r1_pl = 64
        self.ResBlock1_1 = Resblock(self.inplanes, r1_pl,
                                    Downsample(self.inplanes, r1_pl*4))
        self.inplanes = r1_pl * 4
        self.ResBlock1_2 = Resblock(self.inplanes, r1_pl)
        self.ResBlock1_3 = Resblock(self.inplanes, r1_pl)

        r2_pl = 128
        self.ResBlock2_1 = Resblock(self.inplanes, r2_pl,
                                    Downsample(self.inplanes, r2_pl*4, stride=2),
                                    stride=2)
        self.inplanes = r2_pl * 4
        self.ResBlock2_2 = Resblock(self.inplanes, r2_pl)
        self.ResBlock2_3 = Resblock(self.inplanes, r2_pl)
        self.ResBlock2_4 = Resblock(self.inplanes, r2_pl)

        r3_pl = 256
        self.ResBlock3_1 = Resblock(self.inplanes, r3_pl,
                                    Downsample(self.inplanes, r3_pl*4, stride=2),
                                    stride=2)
        self.inplanes = r3_pl * 4
        self.ResBlock3_2 = Resblock(self.inplanes, r3_pl)
        self.ResBlock3_3 = Resblock(self.inplanes, r3_pl)
        self.ResBlock3_4 = Resblock(self.inplanes, r3_pl)
        self.ResBlock3_5 = Resblock(self.inplanes, r3_pl)
        self.ResBlock3_6 = Resblock(self.inplanes, r3_pl)

        r4_pl = 512
        self.ResBlock4_1 = Resblock(self.inplanes, r4_pl,
                                    Downsample(self.inplanes, r4_pl*4, stride=2),
                                    stride=2)
        self.inplanes = r4_pl * 4
        self.ResBlock4_2 = Resblock(self.inplanes, r4_pl)
        self.ResBlock4_3 = Resblock(self.inplanes, r4_pl)

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
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 relu=True,
                 batch_norm=False,
                 **kwargs):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              **kwargs)
        self.relu_ = relu
        if self.relu_:
            self.relu = nn.ReLU(inplace=True)

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        if self.relu_:
            x = self.relu(x)
        return x


class Resblock(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 downsample=None,
                 base_width=64,
                 stride=1,
                 dilation=1):
        super(Resblock, self).__init__()
        width = int(planes * (base_width / 64.))

        self.conv1 = Conv2d(inplanes, width, 1,
                            bias=False, stride=1, batch_norm=True)
        self.conv2 = Conv2d(width, width,  3,
                            bias=False, stride=stride, batch_norm=True,
                            dilation=dilation, padding=dilation)
        self.conv3 = Conv2d(width, planes * self.expansion, 1,
                            bias=False, stride=1, batch_norm=True, relu=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1):
        super(Downsample, self).__init__()
        self.conv = Conv2d(inplanes, planes, 1,
                           bias=False, stride=stride, batch_norm=True)

    def forward(self, x):
        x = self.conv(x)
        return x

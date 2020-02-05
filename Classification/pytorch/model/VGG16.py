import torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self,
                 input_channel=3,
                 num_classes=1000,
                 init_weights=True,
                 batch_norm=False,
                 dropout=True):
        super(VGG16, self).__init__()
        self._build(input_channel, num_classes, batch_norm, dropout)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _build(self, input_channel, num_classes, batch_norm, dropout):
        self._set_backbone(input_channel, batch_norm=batch_norm)
        self._set_avgpool((7, 7))  # enable for different size inputs
        self._set_classifier(num_classes, dropout=dropout)

    def _set_backbone(self, input_channels, batch_norm):
        n = [Conv2d(input_channels, 64, 3, batch_norm, padding=1),
             Conv2d(64, 64, 3, batch_norm, padding=1),
             MaxPool(2, 2),
             Conv2d(64, 128, 3, batch_norm, padding=1),
             Conv2d(128, 128, 3, batch_norm, padding=1),
             MaxPool(2, 2),
             Conv2d(128, 256, 3, batch_norm, padding=1),
             Conv2d(256, 256, 3, batch_norm, padding=1),
             Conv2d(256, 256, 3, batch_norm, padding=1),
             MaxPool(2, 2),
             Conv2d(256, 512, 3, batch_norm, padding=1),
             Conv2d(512, 512, 3, batch_norm, padding=1),
             Conv2d(512, 512, 3, batch_norm, padding=1),
             MaxPool(2, 2),
             Conv2d(512, 512, 3, batch_norm, padding=1),
             Conv2d(512, 512, 3, batch_norm, padding=1),
             Conv2d(512, 512, 3, batch_norm, padding=1),
             MaxPool(2, 2)
             ]
        self.backbone = nn.Sequential(*n)

    def _set_avgpool(self, x):
        self.avgpool = nn.AdaptiveAvgPool2d(x)

    def _set_classifier(self, num_classes, dropout):
        n = [Dense(512 * 7 * 7, 4096, dropout),
             Dense(4096, 4096, dropout),
             Linear(4096, num_classes)
             ]
        self.classifier = nn.Sequential(*n)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Conv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 batch_norm=False,
                 **kwargs):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              **kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.relu(x)
        return x


class Dense(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout=True):
        super(Dense, self).__init__()
        self.linear = nn.Linear(in_channels,
                                out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = dropout
        if self.dropout:
            self.do = nn.Dropout()

    def forward(self, x):
        x = self.linear(x)
        if self.dropout:
            x = self.do(x)
        x = self.relu(x)
        return x


class Linear(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_channels,
                                out_channels)

    def forward(self, x):
        x = self.linear(x)
        return x


class MaxPool(nn.Module):
    def __init__(self,
                 kernel_size,
                 stride):
        super(MaxPool, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size,
                                 stride=stride)

    def forward(self, x):
        x = self.pool(x)
        return x

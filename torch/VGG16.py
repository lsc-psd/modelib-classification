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

    def _set_backbone(self, input_channels, **kwargs):
        layers = []
        layers += conv2d(input_channels, 64, **kwargs)
        layers += conv2d(64, 64, **kwargs)
        layers += maxpool(2, 2)
        layers += conv2d(64, 128, **kwargs)
        layers += conv2d(128, 128, **kwargs)
        layers += maxpool(2, 2)
        layers += conv2d(128, 256, **kwargs)
        layers += conv2d(256, 256, **kwargs)
        layers += conv2d(256, 256, **kwargs)
        layers += maxpool(2, 2)
        layers += conv2d(256, 512, **kwargs)
        layers += conv2d(512, 512, **kwargs)
        layers += conv2d(512, 512, **kwargs)
        layers += maxpool(2, 2)
        layers += conv2d(256, 512, **kwargs)
        layers += conv2d(512, 512, **kwargs)
        layers += conv2d(512, 512, **kwargs)
        layers += maxpool(2, 2)
        self.backbone = nn.Sequential(*layers)

    def _set_avgpool(self, x):
        self.avgpool = nn.AdaptiveAvgPool2d(x)

    def _set_classifier(self, num_classes, **kwargs):
        layers = []
        layers += dense(512 * 7 * 7, 4096, **kwargs)
        layers += dense(4096, 4096, **kwargs)
        layers += linear(4096, num_classes)
        self.classifier = nn.Sequential(*layers)

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


def conv2d(in_channels,
           out_channels,
           kernel_size=3,
           padding=1,
           batch_norm=False):
    conv = nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=kernel_size,
                     padding=padding)
    relu = nn.ReLU(inplace=True)
    if batch_norm:
        b_norm = nn.BatchNorm2d(out_channels)
        return conv, b_norm, relu
    else:
        return conv, relu


def dense(in_channels,
          out_channels,
          dropout=True):
    linear_ = nn.Linear(in_channels,
                        out_channels)
    relu = nn.ReLU(inplace=True)
    if dropout:
        dropout_ = nn.Dropout()
        return [linear_, relu, dropout_]
    else:
        return [linear_, relu]


def linear(in_channels,
           out_channels):
    linear_ = nn.Linear(in_channels,
                        out_channels)
    return [linear_]


def maxpool(kernel_size,
            stride):
    return [nn.MaxPool2d(kernel_size=kernel_size,
                         stride=stride)]

import torch.nn as nn
import math


class MobileNetV3(nn.Module):
    def __init__(self, num_classes=1000, init_weight=True):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks

        self._build(num_classes)
        if init_weight:
            self._initialize_weights()

    def _build(self, num_classes):
        l = list()
        l.append(conv_3x3_bn(3, 16, 2))
        l.append(InvertedResidual(16, 3, 16, 16, 0, 0, 1))
        l.append(InvertedResidual(16, 3, 64, 24, 0, 0, 2))
        l.append(InvertedResidual(24, 3, 72, 24, 0, 0, 1))
        l.append(InvertedResidual(24, 5, 72, 40, 1, 0, 2))
        l.append(InvertedResidual(40, 5, 120, 40, 1, 0, 1))
        l.append(InvertedResidual(40, 5, 120, 40, 1, 0, 1))
        l.append(InvertedResidual(40, 3, 240, 80, 0, 1, 2))
        l.append(InvertedResidual(80, 3, 200, 80, 0, 1, 1))
        l.append(InvertedResidual(80, 3, 184, 80, 0, 1, 1))
        l.append(InvertedResidual(80, 3, 184, 80, 0, 1, 1))
        l.append(InvertedResidual(80, 3, 480, 112, 1, 1, 1))
        l.append(InvertedResidual(112, 3, 672, 112, 1, 1, 1))
        l.append(InvertedResidual(112, 5, 672, 160, 1, 1, 1))
        l.append(InvertedResidual(160, 5, 672, 160, 1, 1, 2))
        l.append(InvertedResidual(160, 5, 960, 160, 1, 1, 1))
        self.features = nn.Sequential(*l)

        self.conv = conv_1x1_bn(160, 960)
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            h_swish()
        )
        self.classifier = nn.Sequential(
            nn.Linear(960, 1280),
            h_swish(),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, kernel_size, hidden_dim, oup, use_se, use_hs, stride):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Sequential(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Sequential(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

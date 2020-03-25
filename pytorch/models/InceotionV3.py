import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.hub import load_state_dict_from_url

__all__ = ["Inception3"]
model_url = ''


class Inception3(nn.Module):
    def __init__(self,
                 input_channel,
                 num_classes=1000,
                 aux_logits=True,
                 transform_input=False,
                 batch_norm=True,
                 init_weight=True,
                 pretrain=False):
        super(Inception3, self).__init__()

        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self._build(input_channel,
                    num_classes,
                    batch_norm)

        # automatically abandon init_weight if pretrain is True
        if pretrain:
            assert model_url is not '', f'Pretrained model for {self.__class__.__name__} not prepared yet.'
            state_dict = load_state_dict_from_url(model_url,
                                                  progress=True)
            self.load_state_dict(state_dict)
        elif init_weight:
            self._init_weights()

    def _build(self,
               input_channel,
               num_classes,
               batch_norm):
        n = [Conv2d(input_channel, 32, 3, batch_norm, stride=1),
             Conv2d(32, 32, 3, batch_norm),
             Conv2d(32, 64, 3, batch_norm, padding=1),
             MaxPool(3, 2),
             Conv2d(64, 80, 1, batch_norm),
             Conv2d(80, 192, 3, batch_norm),
             MaxPool(3, 2)]
        self.convs = nn.Sequential(*n)

        n = [InceptionA(192, pool_features=32),
             InceptionA(256, pool_features=64),
             InceptionA(288, pool_features=64)]
        self.inceptions_a = nn.Sequential(*n)

        self.grid_reduction_1 = GridReduction1(288)

        n = [InceptionB(768, channels_7x7=128),
             InceptionB(768, channels_7x7=160),
             InceptionB(768, channels_7x7=160),
             InceptionB(768, channels_7x7=192)]
        self.inceptions_b = nn.Sequential(*n)

        if self.aux_logits:
            self.aux = InceptionAux(768, num_classes)

        self.grid_reduction_2 = GridReduction2(768)

        n = [InceptionC(1280),
             InceptionC(2048)]
        self.inceptions_c = nn.Sequential(*n)

        self.linear = Linear(2048, num_classes)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _transform_input(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def forward(self, x):
        x = self._transform_input(x)
        x = self.convs(x)
        x = self.inceptions_a(x)
        x = self.grid_reduction_1(x)
        x = self.inceptions_b(x)
        aux_defined = self.training and self.aux_logits
        if aux_defined:
            aux = self.aux(x)
        else:
            aux = None
        x = self.grid_reduction_2(x)
        x = self.inceptions_c(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = F.dropout(x, training=self.training)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x, aux


class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = Conv2d(in_channels, 64, 1)

        self.branch5x5_1 = Conv2d(in_channels, 48, 1)
        self.branch5x5_2 = Conv2d(48, 64, 5, padding=2)

        self.branch3x3dbl_1 = Conv2d(in_channels, 64, 1)
        self.branch3x3dbl_2 = Conv2d(64, 96, 3, padding=1)
        self.branch3x3dbl_3 = Conv2d(96, 96, 3, padding=1)

        self.branch_pool = Conv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        branches = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        outputs = torch.cat(branches, 1)
        return outputs


class GridReduction1(nn.Module):
    def __init__(self, in_channels):
        super(GridReduction1, self).__init__()
        self.branch3x3 = Conv2d(in_channels, 384, 3, stride=2)

        self.branch3x3dbl_1 = Conv2d(in_channels, 64, 1)
        self.branch3x3dbl_2 = Conv2d(64, 96, 3, padding=1)
        self.branch3x3dbl_3 = Conv2d(96, 96, 3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        branches = [branch3x3, branch3x3dbl, branch_pool]
        outputs = torch.cat(branches, 1)
        return outputs


class InceptionB(nn.Module):
    def __init__(self, in_channels, channels_7x7):
        super(InceptionB, self).__init__()
        self.branch1x1 = Conv2d(in_channels, 192, 1)

        c7 = channels_7x7
        self.branch7x7_1 = Conv2d(in_channels, c7, 1)
        self.branch7x7_2 = Conv2d(c7, c7, (1, 7), padding=(0, 3))
        self.branch7x7_3 = Conv2d(c7, 192, (7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = Conv2d(in_channels, c7, 1)
        self.branch7x7dbl_2 = Conv2d(c7, c7, (7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = Conv2d(c7, c7, (1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = Conv2d(c7, c7, (7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = Conv2d(c7, 192, (1, 7), padding=(0, 3))

        self.branch_pool = Conv2d(in_channels, 192, 1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        branches = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        outputs = torch.cat(branches, 1)
        return outputs


class GridReduction2(nn.Module):
    def __init__(self, in_channels):
        super(GridReduction2, self).__init__()
        self.branch3x3_1 = Conv2d(in_channels, 192, 1)
        self.branch3x3_2 = Conv2d(192, 320, 3, stride=2)

        self.branch7x7x3_1 = Conv2d(in_channels, 192, 1)
        self.branch7x7x3_2 = Conv2d(192, 192, (1, 7), padding=(0, 3))
        self.branch7x7x3_3 = Conv2d(192, 192, (7, 1), padding=(3, 0))
        self.branch7x7x3_4 = Conv2d(192, 192, 3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        branches = [branch3x3, branch7x7x3, branch_pool]
        outputs = torch.cat(branches, 1)
        return outputs


class InceptionC(nn.Module):

    def __init__(self, in_channels):
        super(InceptionC, self).__init__()
        self.branch1x1 = Conv2d(in_channels, 320, 1)

        self.branch3x3_1 = Conv2d(in_channels, 384, 1)
        self.branch3x3_2a = Conv2d(384, 384, (1, 3), padding=(0, 1))
        self.branch3x3_2b = Conv2d(384, 384, (3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = Conv2d(in_channels, 448, 1)
        self.branch3x3dbl_2 = Conv2d(448, 384, 3, padding=1)
        self.branch3x3dbl_3a = Conv2d(384, 384, (1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = Conv2d(384, 384, (3, 1), padding=(1, 0))

        self.branch_pool = Conv2d(in_channels, 192, 1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        branches = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        outputs = torch.cat(branches, 1)
        return outputs


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = Conv2d(in_channels, 128, kernel_size=1)
        self.conv1 = Conv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        x = self.conv0(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


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

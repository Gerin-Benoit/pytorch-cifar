'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from .spectral_norm_conv_inplace import *
from .spectral_norm_fc import *


class ActNormLP2D(nn.Module):
    def __init__(self, num_channels):
        super(ActNormLP2D, self).__init__()
        self.num_channels = num_channels
        self._log_scale = Parameter(torch.zeros(num_channels))  # Tensor not zeros
        self._shift = Parameter(torch.zeros(num_channels))
        self._init = False
        self.eps = 1e-6

    def log_scale(self):
        return self._log_scale[None, :, None, None]

    def shift(self):
        return self._shift[None, :, None, None]

    def forward(self, x):
        if not self._init:
            with torch.no_grad():
                # initialize params to input stats
                assert self.num_channels == x.size(1)

                mean = torch.transpose(x, 0, 1).contiguous().view(self.num_channels, -1).mean(dim=1, keepdim=False)
                zero_mean = x - mean[None, :, None, None]
                var = torch.transpose(zero_mean ** 2, 0, 1).contiguous().view(self.num_channels, -1).mean(dim=1)
                std = (var + self.eps) ** .5
                log_scale = torch.log(1. / std)
                log_scale = torch.clamp(log_scale, None, 0.0)
                self._log_scale.data = log_scale
                self._shift.data = - mean * torch.exp(log_scale)
                self._init = True

        log_scale = self.log_scale()
        return x * torch.exp(log_scale) + self.shift()


class ActNormLP2D_else(nn.Module):
    def __init__(self, num_channels):
        super(ActNormLP2D_else, self).__init__()
        self.num_channels = num_channels
        self._scale = Parameter(torch.zeros(num_channels))  # Tensor not zeros
        self._shift = Parameter(torch.zeros(num_channels))
        self._init = False
        self.eps = 1e-6

    def scale(self):
        return self._scale[None, :, None, None]

    def shift(self):
        return self._shift[None, :, None, None]

    def forward(self, x):
        if not self._init:
            with torch.no_grad():
                # initialize params to input stats
                assert self.num_channels == x.size(1)

                mean = torch.transpose(x, 0, 1).contiguous().view(self.num_channels, -1).mean(dim=1, keepdim=False)
                zero_mean = x - mean[None, :, None, None]
                var = torch.transpose(zero_mean ** 2, 0, 1).contiguous().view(self.num_channels, -1).mean(dim=1)
                std = (var + self.eps) ** .5
                scale = 1. / std
                scale = torch.clamp(scale, -1, 1)
                self._scale.data = scale
                self._shift.data = - mean * scale
                self._init = True

        scale = self.scale()
        return x * scale + self.shift()


"""This block is from the DDU github, which uses average pooling shortcut and leaky relu to improve sensitivity"""


class AvgPoolShortCut(nn.Module):
    def __init__(self, stride, out_c, in_c):
        super(AvgPoolShortCut, self).__init__()
        self.stride = stride
        self.out_c = out_c
        self.in_c = in_c

    def forward(self, x):
        if x.shape[2] % 2 != 0:
            x = F.avg_pool2d(x, 1, self.stride)
        else:
            x = F.avg_pool2d(x, self.stride, self.stride)
        pad = torch.zeros(x.shape[0], self.out_c - self.in_c, x.shape[2], x.shape[3], device=x.device, )
        x = torch.cat((x, pad), dim=1)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm=nn.BatchNorm2d, c=0, shape=None, mod=True):
        super(BasicBlock, self).__init__()
        self.conv1 = wrapper_spectral_norm(nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False), kernel_size=3, c=c, shape=shape)
        self.bn1 = norm(planes)
        self.conv2 = wrapper_spectral_norm(nn.Conv2d(planes, planes, kernel_size=3,
                                                     stride=1, padding=1, bias=False), kernel_size=3, c=c, shape=shape)
        self.bn2 = norm(planes)
        self.mod = mod
        self.activation = F.leaky_relu if self.mod else F.relu

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if self.mod:
                self.shortcut = nn.Sequential(AvgPoolShortCut(stride, self.expansion * planes, in_planes))
            else:
                self.shortcut = nn.Sequential(
                    wrapper_spectral_norm(nn.Conv2d(in_planes, self.expansion * planes,
                                                    kernel_size=1, stride=stride, bias=False), kernel_size=1, c=c,
                                          shape=shape),
                    norm(self.expansion * planes)
                )

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out, inplace=True)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm=nn.BatchNorm2d, c=0, shape=None, mod=True):
        super(Bottleneck, self).__init__()
        self.conv1 = wrapper_spectral_norm(nn.Conv2d(in_planes, planes, kernel_size=1, bias=False), kernel_size=1, c=c,
                                           shape=shape)
        self.bn1 = norm(planes)
        if stride == 2:
            shape = (shape[0], shape[1] // 2, shape[2] // 2)
        self.conv2 = wrapper_spectral_norm(nn.Conv2d(planes, planes, kernel_size=3,
                                                     stride=stride, padding=1, bias=False), kernel_size=3, c=c,
                                           shape=shape)
        self.bn2 = norm(planes)
        self.conv3 = wrapper_spectral_norm(nn.Conv2d(planes, self.expansion *
                                                     planes, kernel_size=1, bias=False), kernel_size=1, c=c,
                                           shape=shape)
        self.bn3 = norm(self.expansion * planes)

        self.mod = mod
        self.activation = F.leaky_relu if self.mod else F.relu

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if self.mod:
                self.shortcut = nn.Sequential(AvgPoolShortCut(stride, self.expansion * planes, in_planes))
            else:
                self.shortcut = nn.Sequential(
                    wrapper_spectral_norm(nn.Conv2d(in_planes, self.expansion * planes,
                                                    kernel_size=1, stride=stride, bias=False), kernel_size=1, c=c,
                                          shape=shape),
                    norm(self.expansion * planes)
                )

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)), inplace=True)
        # print('1', out.shape)
        out = self.activation(self.bn2(self.conv2(out)), inplace=True)
        # print('1', out.shape)
        out = self.bn3(self.conv3(out))
        # print('1', out.shape)
        out += self.shortcut(x)
        # print('1', out.shape)
        out = self.activation(out, inplace=True)
        # print('1', out.shape)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, norm=nn.BatchNorm2d, c=0, device='cpu', mod=False,
                 fc_sn=False):
        super(ResNet, self).__init__()
        img_size = (3, 32, 32)
        self.in_planes = 64
        self.mod = mod
        self.fc_sn = fc_sn
        self.activation = F.leaky_relu if self.mod else F.relu

        self.conv1 = wrapper_spectral_norm(nn.Conv2d(3, 64, kernel_size=3,
                                                     stride=1, padding=1, bias=False), kernel_size=3, c=c,
                                           shape=img_size)
        self.bn1 = norm(64)
        shape = (64, 32, 32)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, norm=norm, c=c, shape=shape)
        shape = (128, 32, 32)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, norm=norm, c=c, shape=shape)
        shape = (256, 16, 16)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, norm=norm, c=c, shape=shape)
        shape = (512, 8, 8)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, norm=norm, c=c, shape=shape)
        if self.fc_sn:
            self.linear = spectral_norm_fc(nn.Linear(512 * block.expansion, num_classes), 1,
                                           n_power_iterations=1)
        else:
            self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.device = device
        self.smoothness = torch.tensor(c).to(self.device)

    def _make_layer(self, block, planes, num_blocks, stride, norm=nn.BatchNorm2d, c=0, shape=None):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, norm=norm, c=c, shape=shape, mod=self.mod))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)), inplace=True)
        # print(out.shape)
        out = self.layer1(out)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out = self.layer3(out)
        # print(out.shape)
        out = self.layer4(out)
        # print(out.shape)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def get_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        # print(out.shape)
        out = self.layer1(out)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out = self.layer3(out)
        # print(out.shape)
        out = self.layer4(out)
        # print(out.shape)
        out = F.avg_pool2d(out, 4)
        f = out.view(out.size(0), -1)
        # print("feature map shape:", f.shape)
        out = self.linear(f)
        return out, f

    def clamp_norm_layers(self):
        if self.smoothness != 0:
            c = self.smoothness if self.smoothness > 0 else -self.smoothness
            for name, p in self.named_parameters():
                if "_log_scale" in name:
                    p.data.clamp_(None, torch.log(c))
                elif "_scale" in name:
                    p.data.clamp_(-c, c)


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50(c=0, num_classes=10, norm_layer='batchnorm', device='cpu', mod=False, fc_sn=False):
    norm = nn.BatchNorm2d if norm_layer == 'batchnorm' else ActNormLP2D_else if norm_layer == 'actnorm_2' else ActNormLP2D
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes,
                  norm=norm, c=c, device=device, mod=mod, fc_sn=fc_sn)


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


def wrapper_spectral_norm(layer, kernel_size, c=0, shape=0):
    if c == 0:
        return layer
    if c > 0:
        return spectral_norm_fc(layer, c,
                                n_power_iterations=1)

    if c < 0:
        if kernel_size == 1:
            # use spectral norm fc, because bound are tight for 1x convolutions
            return spectral_norm_fc(layer, -c,
                                    n_power_iterations=1)
        else:
            # use spectral norm based on conv, because bound not tight
            return spectral_norm_conv(layer, -c, shape,
                                      n_power_iterations=1)


def fix_st(state_dict):
    # Create a new state_dict with the desired key changes
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith('weight_orig'):
            # Rename keys ending with 'weight_orig' to end with 'weight'
            new_key = key[:-5]
            new_state_dict[new_key] = value
        elif 'weight_' not in key:
            # Keep other keys that do not contain 'weight_'
            new_state_dict[key] = value

    return new_state_dict

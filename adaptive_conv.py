import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AdaptiveConv2D(nn.Module):
    def __init__(self, ni, no, kernel_size=3, stride=1, dilation=1, size=None, lite=False, pixel_aware=False):
        super(AdaptiveConv2D, self).__init__()
        assert size is not None or lite is True or pixel_aware is True, 'Either pass expected feature map size or use lite or use pixel aware'
        pad = kernel_size // 2
        self.pixel_aware = pixel_aware
        self.lite = lite
        self.size = size
        # Mimic the convolutional operation (local spatial features)
        self.conv1 = nn.Conv2d(ni, no, kernel_size, stride, pad, dilation)
        # Mimic the self transformation
        self.conv2 = nn.Conv2d(ni, no, 1, stride, 0, dilation)
        # Mimic the fully conected layer (global information)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(ni, no, 1)
        self.fc2 = nn.Conv2d(no, no, 1)
        # Element wise weighted sum of the above 3 operations (these are learnt)
        if lite:
            self.w = nn.Parameter(torch.ones(3, 1, 16, 16))
        elif pixel_aware:
            self.fusion_conv1 = nn.Conv2d(no * 3, no, 1)
            self.fusion_non_linear = nn.Sequential(nn.ReLU(inplace=True), nn.BatchNorm2d(no))
            self.fusion_conv2 = nn.Conv2d(no, no, 1)
        else:
            assert isinstance(self.size, tuple), 'Tuple of size expected'
            self.w = nn.Parameter(torch.ones(3, 1, self.size[0], self.size[1]))
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        fc_x = F.interpolate(self.fc2(self.relu(self.fc1(self.gap(x)))), size=x1.shape[2:], mode='nearest')
        # Weighted sum is 1
        if not self.pixel_aware:
            if self.lite:
                params = F.softmax(F.interpolate(self.w, size=x1.shape[2:]), dim=1)
            else:
                params = F.softmax(self.w, dim=1)
            alpha, beta, gamma = torch.chunk(params, 3, dim=0)
            return alpha * x1 + beta * x2 + gamma * fc_x
        else:
            concat_ops = torch.cat([x1, x2, fc_x], dim=1)
            out = self.fusion_conv2(self.fusion_non_linear(self.fusion_conv1(concat_ops)))
            return out
        
class AdaptiveResidualLayer(nn.Module):
    def __init__(self, num_channels, kernel_size=3, stride=1, bottlneck=False, weight=1.0, size=None, lite=False, act='relu', use_bn=True, pixel_aware=False):
        super(AdaptiveResidualLayer, self).__init__()
        assert size is not None or lite is True or pixel_aware is True, 'Either pass expected feature map size or use lite or use pixel aware'
        self.pixel_aware = pixel_aware
        self.size = size
        self.lite = lite
        self.weight = weight
        self.num_channels = num_channels
        self.use_bn = use_bn
        out_channels = num_channels // 2 if bottlneck else num_channels
        self.adap_conv1 = AdaptiveConv2D(num_channels, out_channels, kernel_size, stride, 1, self.size, self.lite, self.pixel_aware)
        second_sz = (self.size[0] // stride, self.size[1] // stride) if self.size is not None else None
        self.adap_conv2 = AdaptiveConv2D(out_channels, num_channels, kernel_size, stride, 1, second_sz, self.lite, self.pixel_aware)
        self.act = nn.ModuleList([nn.ReLU(), nn.ReLU()]) if act == 'relu' else nn.ModuleList([act, act])
        self.bn = nn.BatchNorm2d(num_channels) if use_bn else None
    def forward(self, x):
        conv_out = self.act[1](self.adap_conv2(self.act[0](self.adap_conv1(x))) * self.weight + x)
        out = self.bn(conv_out) if self.use_bn else conv_out
        return out

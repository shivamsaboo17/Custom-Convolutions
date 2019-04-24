import torch
import torch.nn as nn
import torch.nn.functional as F

class OctaveConv2D(nn.Module):
    def __init__(self, ni, no, ks=3, alpha_in=0.5, alpha_out=0.5, first=False, last=False):
        super(OctaveConv2D, self).__init__()
        assert not (first and last), 'Layer cannot be both first and last'
        if first:
            alpha_in = 0
        if last:
            alpha_out = 0
        self.first = first
        self.last = last
        self.input_high_channels = int((1 - alpha_in) * ni)
        self.input_low_channels = ni - self.input_high_channels
        output_high_channels = int((1 - alpha_out) * no)
        output_low_channels = no - output_high_channels
        self.conv_high2high = nn.Conv2d(self.input_high_channels, output_high_channels, ks, padding=1)
        if not last:
            self.conv_high2low = nn.Conv2d(self.input_high_channels, output_low_channels, ks, padding=1)
        if not (last or first):
            self.conv_low2low = nn.Conv2d(self.input_low_channels, output_low_channels, ks, padding=1)
        if not first:
            self.conv_low2high = nn.Conv2d(self.input_low_channels, output_high_channels, ks, padding=1)
    def forward(self, x):
        out_high, out_low = 0, 0
        if not self.first:
            x_high, x_low = x
            if not self.last:
                l2l = self.conv_low2low(x_low)
                out_low += l2l
            l2h = F.interpolate(self.conv_low2high(x_low), scale_factor=2)
            out_high += l2h
        else:
            x_high = x[0]
            h2h = self.conv_high2high(x_high)
            if not self.last:
                h2l = self.conv_high2low(F.interpolate(x_high, scale_factor=0.5))
                out_low += h2l
            out_high += h2h
        return out_high, out_low

class OctaveResidualLayer(nn.Module):
    def __init__(self, channels, kernel_size=3, alpha=0.5, act='relu', bottleneck=False, use_bn=True, weight=1.0):
        super(OctaveResidualLayer, self).__init__()
        output_channels = channels if not bottleneck else channels // 2
        high_channels = int((1 - alpha) * channels)
        low_channels = channels - high_channels
        self.weight = weight
        self.conv_1 = OctaveConv2D(channels, output_channels, kernel_size, alpha, alpha)
        self.conv_2 = OctaveConv2D(output_channels, channels, kernel_size, alpha, alpha)
        self.act = nn.ModuleList([nn.ReLU()] * 4)
        self.bn = nn.ModuleList([nn.BatchNorm2d(ch) for ch in [high_channels, low_channels]]) if use_bn else None
    def forward(self, x):
        h, l = self.conv_1(x)
        h, l = self.act[0](h), self.act[1](l)
        h, l = self.conv_2([h, l])
        h, l = self.act[2](h), self.act[3](l)
        h, l = [self.bn[0](h), self.bn[1](l)] if self.bn is not None else [h, l] 
        return [h * self.weight + x[0], l * self.weight + x[1]]
# -*-coding:utf-8-*-
import torch.nn as nn
from torch.nn import functional as F
import torch

class SuperBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(SuperBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        # self._check_input_dim(input)
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean[:input.shape[1]] = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean[:input.shape[1]]
                # update running_var with unbiased var
                self.running_var[:input.shape[1]] = exponential_average_factor * var * n / (n - 1) \
                    + (1 - exponential_average_factor) * self.running_var[:input.shape[1]]
        else:
            mean = self.running_mean[:input.shape[1]]
            var = self.running_var[:input.shape[1]]
        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :input.shape[1], None, None] + self.bias[None, :input.shape[1], None, None]
        return input


class SuperCon(nn.Module):
    def __init__(self, in_chanel, out_channel):
        super(SuperCon, self).__init__()
        self.con = nn.Sequential(
            nn.ReflectionPad2d(1),
            SuperConv2d(in_channels=in_chanel, out_channels=out_channel, kernel_size=4, stride=2, padding=0),
            SuperBatchNorm2d(out_channel, affine=True),
            nn.LeakyReLU(0.2))

    def forward(self, input):
        y = self.con(input)
        return y


class SuperDecon(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SuperDecon, self).__init__()
        self.decon = nn.Sequential(
            SuperConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            SuperConv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=0),
            SuperBatchNorm2d(out_channel, affine=True),
            nn.ReLU(inplace=True))

    def forward(self, input):  # configs是一个列表
        y = self.decon(input)
        return y
        


## Channel Attention (CA) Layer
class SuperCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SuperCALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            SuperConv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            SuperConv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(x)
        return x * y
        


class SuperRCAB_bn(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction=2, bias=True, bn=True, act=nn.ReLU(True), res_scale=1):
        super(SuperRCAB_bn, self).__init__()
        modules_body = []
        # modules_CA = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, padding=1, bias=bias))
            if bn: modules_body.append(SuperBatchNorm2d(n_feat, affine=True))
            if i == 0: modules_body.append(act)
        self.body = nn.Sequential(*modules_body)
        self.CA = SuperCALayer(n_feat, reduction)
        self.res_scale = res_scale

    def forward(self, input):
        res = self.body(input)
        res = self.CA(res)
        res += input
        return res


# Super
class SuperConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(SuperConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias, padding_mode)
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation,
                        self.groups)


class SuperConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros'):
        super(SuperConvTranspose2d, self).__init__(in_channels, out_channels,
                                                   kernel_size, stride, padding,
                                                   output_padding, groups, bias,
                                                   dilation, padding_mode)
        # self.conTran = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x, output_size=None):
        output_padding = self._output_padding(x, output_size, self.stride, self.padding, self.kernel_size, )
        return F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding,
                                  output_padding, self.groups,
                                  self.dilation)

class SuperSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, norm_layer=SuperBatchNorm2d,
                 bias=True, scale_factor=1):
        super(SuperSeparableConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels * scale_factor, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=in_channels, bias=bias),
            norm_layer(in_channels * scale_factor, affine=True),
            nn.Conv2d(in_channels=in_channels * scale_factor, out_channels=out_channels,
                      kernel_size=1, stride=1, bias=bias),
        )

    def forward(self, x):
        y = self.conv(x)
        return y

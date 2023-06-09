# -*-coding:utf-8-*-
import torch.nn as nn
# from MPNCOV.python import MPNCOV


def default_conv(in_channels, out_channels, kernel_size, padding=0, bias=True):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=padding,
                      groups=in_channels),
            nn.BatchNorm2d(in_channels, affine=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1),
        )


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, config, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, config, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(config, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        # print('ex', x.shape)
        y = self.avg_pool(x)
        # print('ex', y.shape)
        y = self.conv_du(y)
        # print('ex', y.shape)
        return x * y

# 带有IN的RCAB
class RCAB_bn(nn.Module):
    def __init__(self, config, conv, n_feat, kernel_size, reduction=2, bias=True, bn=True, act=nn.ReLU(True), res_scale=1):
        super(RCAB_bn, self).__init__()
        modules_body = []
        for i in range(2):
            if i == 0:
                modules_body.append(conv(n_feat, config[0], kernel_size, padding=1, bias=bias))
                modules_body.append(nn.BatchNorm2d(config[0]))
                modules_body.append(act)
            if i == 1:
                modules_body.append(conv(config[0], n_feat, kernel_size, padding=1, bias=bias))
                modules_body.append(nn.BatchNorm2d(n_feat))
        modules_body.append(CALayer(config[1], n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

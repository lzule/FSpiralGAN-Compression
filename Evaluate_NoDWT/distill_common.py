# -*-coding:utf-8-*-
import math
import numpy as np
import torch
import torch.nn as nn

# from MPNCOV.python import MPNCOV
# {'con1': 6, 'con2': 6, 'con3': 4, 'con4': 4, 'con5': 6,



def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


def Separate_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=(kernel_size // 2), bias=bias, groups=in_channels),
                         nn.BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=False,
                                        track_running_stats=False),
                         nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1))
                         )


## Channel Attention (CA) Layer
class CALayer_1(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer_1, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# 带有bn的RCAB
class RCAB_bn_1(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction=2, bias=True, bn=True, act=nn.ReLU(True), res_scale=1):
        super(RCAB_bn_1, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, 6, kernel_size, bias=bias))
        modules_body.append(nn.BatchNorm2d(6))
        modules_body.append(act)
        modules_body.append(conv(6, n_feat, kernel_size, bias=bias))
        modules_body.append(nn.BatchNorm2d(n_feat))
        modules_body.append(CALayer_1(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class CALayer_2(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer_2, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, 4, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# 带有bn的RCAB
class RCAB_bn_2(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction=2, bias=True, bn=True, act=nn.ReLU(True), res_scale=1):
        super(RCAB_bn_2, self).__init__()
        modules_body = list()
        modules_body.append(conv(n_feat, 12, kernel_size, bias=bias))
        modules_body.append(nn.BatchNorm2d(12))
        modules_body.append(act)
        modules_body.append(conv(12, n_feat, kernel_size, bias=bias))
        modules_body.append(nn.BatchNorm2d(n_feat))
        modules_body.append(CALayer_2(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class CALayer_3(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer_3, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, 4, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# 带有bn的RCAB
class RCAB_bn_3(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction=2, bias=True, bn=True, act=nn.ReLU(True), res_scale=1):
        super(RCAB_bn_3, self).__init__()
        modules_body = list()
        modules_body.append(conv(n_feat, 6, kernel_size, bias=bias))
        modules_body.append(nn.BatchNorm2d(6))
        modules_body.append(act)
        modules_body.append(conv(6, n_feat, kernel_size, bias=bias))
        modules_body.append(nn.BatchNorm2d(n_feat))
        modules_body.append(CALayer_3(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class CALayer_4(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer_4, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, 12, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# 带有bn的RCAB
class RCAB_bn_4(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction=2, bias=True, bn=True, act=nn.ReLU(True), res_scale=1):
        super(RCAB_bn_4, self).__init__()
        modules_body = list()
        modules_body.append(conv(n_feat, 12, kernel_size, bias=bias))
        modules_body.append(nn.BatchNorm2d(12))
        modules_body.append(act)
        modules_body.append(conv(12, n_feat, kernel_size, bias=bias))
        modules_body.append(nn.BatchNorm2d(n_feat))
        modules_body.append(CALayer_4(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

# class CALayer_5(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(CALayer_5, self).__init__()
#         # global average pooling: feature --> point
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         # feature channel downscale and upscale --> channel weight
#         self.conv_du = nn.Sequential(
#                 nn.Conv2d(channel, 4, 1, padding=0, bias=True),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(4, channel, 1, padding=0, bias=True),
#                 nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.conv_du(y)
#         return x * y
#
# # 带有bn的RCAB
# class RCAB_bn_5(nn.Module):
#     def __init__(self, conv, n_feat, kernel_size, reduction=2, bias=True, bn=True, act=nn.ReLU(True), res_scale=1):
#         super(RCAB_bn_5, self).__init__()
#         modules_body = []
#         modules_body.append(conv(n_feat, 12, kernel_size, bias=bias))
#         modules_body.append(nn.BatchNorm2d(12))
#         modules_body.append(act)
#         modules_body.append(conv(12, n_feat, kernel_size, bias=bias))
#         modules_body.append(nn.BatchNorm2d(n_feat))
#         modules_body.append(CALayer_5(n_feat, reduction))
#         self.body = nn.Sequential(*modules_body)
#         self.res_scale = res_scale
#
#     def forward(self, x):
#         res = self.body(x)
#         res += x
#         return res

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

    def forward(self, input, configs):
        x = input
        for module in self.con:
            if isinstance(module, SuperConv2d):
                config = {'channel': configs}
                x = module(x, config)
            else:
                x = module(x)
        return x  # 存储数据con1、con2 ...


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

    def forward(self, input, configs):  # configs是一个列表
        x = input
        for module in self.decon:
            if isinstance(module, SuperConvTranspose2d):
                config = {'channel': configs[0]}
                x = module(x, config)
            elif isinstance(module, SuperConv2d):
                config = {'channel': configs[1]}
                x = module(x, config)
            else:
                x = module(x)
        return x


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

    def forward(self, x, channel):
        y = self.avg_pool(x)
        input_channel = x.shape[1]
        # print('y', y.shape)
        SuCon_num = 0
        for module in self.conv_du:
            if isinstance(module, SuperConv2d) and SuCon_num == 0:
                config = {'channel': channel}
                y = module(y, config)
                SuCon_num += 1
                # print(y.shape)
            elif isinstance(module, SuperConv2d) and SuCon_num == 1:
                config = {'channel': input_channel}
                y = module(y, config)
                # print(y.shape)
            else:
                y = module(y)
                # print(y.shape)
        # print('x', x.shape)
        # print('y', y.shape)
        return y * x


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

    def forward(self, input, configs):
        x = input
        input_channel = input.shape[1]
        Separ_num = 0
        for module in self.body:
            if isinstance(module, SuperSeparableConv2d) and Separ_num == 0:
                config = {'channel': configs[0]}
                x = module(x, config)
                Separ_num += 1
            elif isinstance(module, SuperSeparableConv2d) and Separ_num == 1:
                config = {'channel': input_channel}
                x = module(x, config)
            else:
                x = module(x)
        x = self.CA(x, configs[1])
        # print('AB:A', x.shape)
        return torch.add(x, input)


# Super
class SuperConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(SuperConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias, padding_mode)

    def forward(self, x, config):
        in_nc = x.size(1)  # 输入通道数
        out_nc = config['channel']
        weight = self.weight[:out_nc, :in_nc]  # 权值分享 [oc, ic, H, W]
        if self.bias is not None:
            bias = self.bias[:out_nc]
        else:
            bias = None
        return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation,
                        self.groups)  # Pytorch里一般小写的都是函数式的接口，相应的大写的是类式接口


class SuperConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros'):
        super(SuperConvTranspose2d, self).__init__(in_channels, out_channels,
                                                   kernel_size, stride, padding,
                                                   output_padding, groups, bias,
                                                   dilation, padding_mode)

    def forward(self, x, config, output_size=None):
        output_padding = self._output_padding(x, output_size, self.stride, self.padding, self.kernel_size, )
        in_nc = x.size(1)
        out_nc = config['channel']
        weight = self.weight[:in_nc,
                 :out_nc]  # [ic, oc, H, W] 这一步相当于取self.weight的第0维的0：第in_nc个元素，第1维的0：第out_nc个元素，所以得到的矩阵尺寸只有第0、1维的长度变化了，其余还是没有变化的
        if self.bias is not None:
            bias = self.bias[:out_nc]
        else:
            bias = None
        return F.conv_transpose2d(x, weight, bias, self.stride, self.padding,
                                  output_padding, self.groups,
                                  self.dilation)  # 使用上面的部分weight以及bias权重来进行卷积运算，conv_transpose2d需要好好研究一下


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

    def forward(self, x, config):
        in_nc = x.size(1)
        out_nc = config['channel']

        conv = self.conv[0]  # 这个卷积的权重weight有点奇怪，size为[oc,1,H,W]
        assert isinstance(conv, nn.Conv2d)  # x = [_, in_nc, _, _]
        weight = conv.weight[:in_nc]  # [oc, 1, H, W]  # 这里应该是标注错误了，索引为[1]的部分数值是不会改变的，正确应为[ic_nc, in_feature, H, W]
        # print(weight.shape)
        if conv.bias is not None:
            bias = conv.bias[:in_nc]
        else:
            bias = None
        x = F.conv2d(x, weight, bias, conv.stride, conv.padding, conv.dilation, in_nc)

        x = self.conv[1](x)

        conv = self.conv[2]  # 此时这里的conv是1x1conv2d
        assert isinstance(conv, nn.Conv2d)
        weight = conv.weight[:out_nc, :in_nc]  # [oc, ic, H, W]  # 在这里，参数减少了
        # print(weight.shape)
        if conv.bias is not None:
            bias = conv.bias[:out_nc]
        else:
            bias = None
        x = F.conv2d(x, weight, bias, conv.stride, conv.padding, conv.dilation, conv.groups)
        return x

# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi


class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, images):
        bach_size = images.size()[0]

        x_tv = images[:, :, :, 1:] - images[:, :, :, :-1]
        y_tv = images[:, :, 1:, :] - images[:, :, :-1, :]

        x_tv_size = self._tensor_size(x_tv)
        y_tv_size = self._tensor_size(y_tv)

        x_tv = torch.sum(torch.pow(x_tv, 2))
        y_tv = torch.sum(torch.pow(y_tv, 2))

        return (x_tv/x_tv_size + y_tv/y_tv_size)/bach_size

    def _tensor_size(self, inputs):
        size = inputs.size()
        return size[1] * size[2] * size[-1]


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        # torch.nn.L1Loss

    def forward(self, images, targets):
        return F.l1_loss(images, targets)
        # return torch.mean(torch.abs(images - targets))

# 和cycleGAN里的GANLoss一样
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))  # 缓存注册register_buffer(),函数注释上面说buffer表示那些不是parameter但是需要存储的变量
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = torch.nn.MSELoss()
        else:
            self.loss = torch.nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)  # tensor.expand_as()这个函数就是把一个tensor变成和函数括号内一样形状的tensor，用法与expand（）类似。

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)  # 在这里做 1-DG()


class MSELoss(nn.MSELoss):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, inputs, targets):
        return F.mse_loss(inputs, targets)


class AngularLoss(torch.nn.Module):

    def __init__(self):
        super(AngularLoss, self).__init__()

    def forward(self, illum_gt, illum_pred):
        # img_gt = img_input / illum_gt
        # illum_gt = img_input / img_gt
        # illum_pred = img_input / img_output

        # ACOS
        cos_between = torch.nn.CosineSimilarity(dim=1)
        cos = cos_between(illum_gt, illum_pred)
        cos = torch.clamp(cos, -0.99999, 0.99999)
        loss = torch.mean(torch.acos(cos)) * 180 / pi

        # MSE
        # loss = torch.mean((illum_gt - illum_pred)**2)

        # 1 - COS
        # loss = 1 - torch.mean(cos)

        # 1 - COS^2
        # loss = 1 - torch.mean(cos**2)
        return loss


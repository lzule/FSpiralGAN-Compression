# -*-coding:utf-8-*-
import sys

sys.path.insert(0, 'model/')
import torch.nn as nn

import torch
from model.StuCommon import SuperRCAB_bn, SuperConv2d, SuperSeparableConv2d, SuperConvTranspose2d, SuperCon, SuperDecon, SuperBatchNorm2d


class Student_G(nn.Module):
    def __init__(self, ngf=32):
        super(Student_G, self).__init__()
        # Encoder code-------------------------------------start
        self.configs = None
        self.model = list()
        self.__setattr__('con1', SuperCon(3, ngf))
        self.__setattr__('con2', SuperCon(ngf, ngf))
        self.__setattr__('con3', SuperCon(ngf, ngf))
        self.__setattr__('con4', SuperCon(ngf, ngf))
        self.__setattr__('con5', SuperCon(ngf, ngf))
        self.__setattr__('RCAB1', SuperRCAB_bn(conv=SuperSeparableConv2d, n_feat=ngf, kernel_size=3, reduction=1, act=nn.ReLU(True)))
        self.__setattr__('decon4', SuperDecon(ngf, ngf))
        self.__setattr__('RCAB2', SuperRCAB_bn(conv=SuperSeparableConv2d, n_feat=ngf * 2, kernel_size=3, reduction=1, act=nn.ReLU(True)))
        self.__setattr__('decon5', SuperDecon(ngf * 2, ngf))
        self.__setattr__('RCAB3', SuperRCAB_bn(conv=SuperSeparableConv2d, n_feat=ngf * 2, kernel_size=3, reduction=1, act=nn.ReLU(True)))
        self.__setattr__('decon6', SuperDecon(ngf * 2, ngf))
        self.__setattr__('RCAB4', SuperRCAB_bn(conv=SuperSeparableConv2d, n_feat=ngf * 2, kernel_size=3, reduction=1, act=nn.ReLU(True)))
        self.__setattr__('decon7', SuperDecon(ngf * 2, ngf))
        self.__setattr__('decon8', nn.Sequential(
            SuperConvTranspose2d(in_channels=ngf * 2, out_channels=ngf, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            SuperConv2d(in_channels=ngf, out_channels=ngf, kernel_size=3, stride=1, padding=0),
            SuperBatchNorm2d(ngf, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            SuperConv2d(in_channels=ngf, out_channels=3, kernel_size=3, stride=1, padding=0),
            nn.Tanh()))

    def forward(self, x):
        con1 = self.__getattr__('con1')(x)
        con2 = self.__getattr__('con2')(con1)
        con3 = self.__getattr__('con3')(con2)
        con4 = self.__getattr__('con4')(con3)
        con5 = self.__getattr__('con5')(con4)

        decon4 = self.__getattr__('RCAB1')(con5)
        decon4 = self.__getattr__('decon4')(decon4)
        decon4 = torch.cat([decon4, con4], dim=1)

        decon5 = self.__getattr__('RCAB2')(decon4)
        decon5 = self.__getattr__('decon5')(decon5)
        decon5 = torch.cat([decon5, con3], dim=1)

        decon6 = self.__getattr__('RCAB3')(decon5)
        decon6 = self.__getattr__('decon6')(decon6)
        decon6 = torch.cat([decon6, con2], dim=1)

        decon7 = self.__getattr__('RCAB4')(decon6)
        decon7 = self.__getattr__('decon7')(decon7)
        decon7 = torch.cat([decon7, con1], dim=1)

        decon8 = self.__getattr__('decon8')(decon7)
        return decon8

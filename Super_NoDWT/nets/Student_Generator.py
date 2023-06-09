# -*-coding:utf-8-*-
import sys

sys.path.insert(0, 'model/')
import torch.nn as nn

import torch
from model.SuperCommon import SuperRCAB_bn, SuperConv2d, SuperSeparableConv2d, SuperConvTranspose2d, SuperCon, SuperDecon, SuperBatchNorm2d


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

    def forward(self, input):
        configs = self.configs
        input = input.clamp(-1, 1)  # 将输入限定在[-1, 1]范围内
        x = input
        encoder = dict()
        # SuperTrain-------------------------Encoder Start
        for i in range(5):
            Con_name = 'con' + str(i + 1)
            x = self.__getattr__(Con_name)(x, configs[Con_name])
            encoder[Con_name] = x
            # SuperTrain#########################Encoder Ended
        # SuperTrain-------------------------RCAB_Decon Start
        for i in range(1, 5):
            RCAB_name = 'RCAB' + str(i)
            Decon_name = 'decon' + str(i + 3)
            if i != 1:
                x = torch.cat([x, encoder['con' + str(6 - i)]], dim=1)
            x = self.__getattr__(RCAB_name)(x, configs[RCAB_name])
            x = self.__getattr__(Decon_name)(x, configs[Decon_name])
        # SuperTrain#########################RCAB_Decon Ended

        # SuperTrain-------------------------Final_Decon Start
        for j, module in enumerate(self.__getattr__('decon8')):
            if isinstance(module, SuperConvTranspose2d):
                config = {'channel': configs['decon8'][0]}
                x = module(x, config)
            elif isinstance(module, SuperConv2d) and j == 3:
                config = {'channel': configs['decon8'][1]}
                x = module(x, config)
            elif isinstance(module, SuperConv2d) and j == 7:
                config = {'channel': module.out_channels}
                x = module(x, config)
            else:
                x = module(x)
        # SuperTrain-------------------------Final_Decon Ended
        return x

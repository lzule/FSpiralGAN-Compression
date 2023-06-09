# -*-coding:utf-8-*-
import torch.nn as nn
import torch
from excom import RCAB_bn, default_conv


class Expot(nn.Module):
    def __init__(self, config):
        super(Expot, self).__init__()
        self.con1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=3, out_channels=config['con1'], kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(config['con1'], affine=True),
            nn.LeakyReLU(0.2))

        self.con2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=config['con1'], out_channels=config['con2'], kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(config['con2'], affine=True),
            nn.LeakyReLU(0.2))

        self.con3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=config['con2'], out_channels=config['con3'], kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(config['con3'], affine=True),
            nn.LeakyReLU(0.2))

        self.con4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=config['con3'], out_channels=config['con4'], kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(config['con4'], affine=True),
            nn.LeakyReLU(0.2))

        self.con5 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=config['con4'], out_channels=config['con5'], kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(config['con5'], affine=True),
            nn.LeakyReLU(0.2))

        self.RCAB1 = RCAB_bn(config=config['RCAB1'], conv=default_conv, n_feat=config['con5'], kernel_size=3, reduction=1, act=nn.ReLU(True))
        self.decon4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=config['con5'], out_channels=config['decon4'][0], kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=config['decon4'][0], out_channels=config['decon4'][1], kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(config['decon4'][1], affine=True),
            nn.ReLU(inplace=True))

        self.RCAB2 = RCAB_bn(config=config['RCAB2'], conv=default_conv, n_feat=config['con4'] + config['decon4'][1], kernel_size=3, reduction=1, act=nn.ReLU(True))
        self.decon5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=config['con4'] + config['decon4'][1], out_channels=config['decon5'][0], kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=config['decon5'][0], out_channels=config['decon5'][1], kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(config['decon5'][1], affine=True),
            nn.ReLU(inplace=True))

        self.RCAB3 = RCAB_bn(config=config['RCAB3'], conv=default_conv, n_feat=config['con3'] + config['decon5'][1], kernel_size=3, reduction=1, act=nn.ReLU(True))
        self.decon6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=config['con3'] + config['decon5'][1], out_channels=config['decon6'][0], kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=config['decon6'][0], out_channels=config['decon6'][1], kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(config['decon6'][1], affine=True),
            nn.ReLU(inplace=True))

        self.RCAB4 = RCAB_bn(config=config['RCAB4'], conv=default_conv, n_feat=config['con2'] + config['decon6'][1], kernel_size=3, reduction=1, act=nn.ReLU(True))
        self.decon7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=config['con2'] + config['decon6'][1], out_channels=config['decon7'][0], kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=config['decon7'][0], out_channels=config['decon7'][1], kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(config['decon7'][1], affine=True),
            nn.ReLU(inplace=True))

        # self.RCAB5 = RCAB_bn(config=config, conv=default_conv, n_feat=ngf * 2, kernel_size=3, reduction=1, act=nn.ReLU(True))
        self.decon8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=config['con1'] + config['decon7'][1], out_channels=config['decon8'][0], kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=config['decon8'][0], out_channels=config['decon8'][1], kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(config['decon8'][1], affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=config['decon8'][1], out_channels=3, kernel_size=3, stride=1, padding=0),
            nn.Tanh())

    def forward(self, x):
        con1 = self.con1(x)
        con2 = self.con2(con1)
        con3 = self.con3(con2)
        con4 = self.con4(con3)
        con5 = self.con5(con4)

        decon4 = self.RCAB1(con5)
        decon4 = self.decon4(decon4)
        decon4 = torch.cat([decon4, con4], dim=1)

        decon5 = self.RCAB2(decon4)
        decon5 = self.decon5(decon5)
        decon5 = torch.cat([decon5, con3], dim=1)

        decon6 = self.RCAB3(decon5)
        decon6 = self.decon6(decon6)
        decon6 = torch.cat([decon6, con2], dim=1)

        decon7 = self.RCAB4(decon6)
        decon7 = self.decon7(decon7)
        decon7 = torch.cat([decon7, con1], dim=1)

        decon8 = self.decon8(decon7)
        return decon8

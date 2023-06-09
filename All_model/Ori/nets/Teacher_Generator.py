# -*-coding:utf-8-*-
import torch.nn as nn
import torch
from model.Common import RCAB_bn, default_conv


class Teacher_G(nn.Module):
    def __init__(self, ngf=32):
        super(Teacher_G, self).__init__()
        self.con1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=3, out_channels=ngf, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(ngf, affine=True),
            nn.LeakyReLU(0.2))

        self.con2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=ngf, out_channels=ngf, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(ngf, affine=True),
            nn.LeakyReLU(0.2))

        self.con3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=ngf, out_channels=ngf, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(ngf, affine=True),
            nn.LeakyReLU(0.2))

        self.con4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=ngf, out_channels=ngf, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(ngf, affine=True),
            nn.LeakyReLU(0.2))

        self.con5 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=ngf, out_channels=ngf, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(ngf, affine=True),
            nn.LeakyReLU(0.2))

        self.RCAB1 = RCAB_bn(conv=default_conv, n_feat=ngf, kernel_size=3, reduction=1, act=nn.ReLU(True))
        self.decon4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ngf, out_channels=ngf, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=ngf, out_channels=ngf, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(ngf, affine=True),
            nn.ReLU(inplace=True))

        self.RCAB2 = RCAB_bn(conv=default_conv, n_feat=ngf * 2, kernel_size=3, reduction=1, act=nn.ReLU(True))
        self.decon5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ngf * 2, out_channels=ngf, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=ngf, out_channels=ngf, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(ngf, affine=True),
            nn.ReLU(inplace=True))

        self.RCAB3 = RCAB_bn(conv=default_conv, n_feat=ngf * 2, kernel_size=3, reduction=1, act=nn.ReLU(True))
        self.decon6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ngf * 2, out_channels=ngf, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=ngf, out_channels=ngf, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(ngf, affine=True),
            nn.ReLU(inplace=True))

        self.RCAB4 = RCAB_bn(conv=default_conv, n_feat=ngf * 2, kernel_size=3, reduction=1, act=nn.ReLU(True))
        self.decon7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ngf * 2, out_channels=ngf, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=ngf, out_channels=ngf, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(ngf, affine=True),
            nn.ReLU(inplace=True))

        self.RCAB5 = RCAB_bn(conv=default_conv, n_feat=ngf * 2, kernel_size=3, reduction=1, act=nn.ReLU(True))
        self.decon8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ngf * 2, out_channels=ngf, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=ngf, out_channels=ngf, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(ngf, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=ngf, out_channels=3, kernel_size=3, stride=1, padding=0),
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

# -*-coding:utf-8-*-
import sys
sys.path.insert(0, 'model/')
import torch.nn as nn
import torch
# from common import RCAB_bn, default_conv
from distill_common import Separate_conv, RCAB_bn_1, RCAB_bn_2, RCAB_bn_3, RCAB_bn_4   #Li 修改了引入模式

class Encoder(nn.Module):  #Li 编码器
    def __init__(self, ngf=64):
        super(Encoder, self).__init__()
        self.con1 = nn.Sequential(  #Li EDB网络结构
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(6, affine=True),
            nn.LeakyReLU(0.2))

        self.con2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=6, out_channels=6, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(6, affine=True),
            nn.LeakyReLU(0.2))

        self.con3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=6, out_channels=4, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(4, affine=True),
            nn.LeakyReLU(0.2))

        self.con4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(4, affine=True),
            nn.LeakyReLU(0.2))

        self.con5 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=4, out_channels=6, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(6, affine=True),
            nn.LeakyReLU(0.2))

    def forward(self, x):
        con1 = self.con1(x)
        con2 = self.con2(con1)
        con3 = self.con3(con2)
        con4 = self.con4(con3)
        con5 = self.con5(con4)
        return [con1, con2, con3, con4, con5]

#  'RCAB1': [6, 8], 'decon4': [6, 4], 'RCAB2': [12, 4], 'decon5': [4, 4], 'RCAB3': [6, 4], 'decon6': [4, 6],
#  'RCAB4': [12, 12], 'decon7': [8, 8], 'decon8': [8, 8]}
class Decoder(nn.Module):  #Li 译码器
    def __init__(self, ngf=32):
        super(Decoder, self).__init__()
        self.RCAB1 = RCAB_bn_1(conv=Separate_conv, n_feat=6, kernel_size=3, reduction=1, act=nn.ReLU(True))
        self.decon4 = nn.Sequential(  #Li EUB网络架构
            nn.ConvTranspose2d(in_channels=6, out_channels=6, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=6, out_channels=4, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(4, affine=True),
            nn.ReLU(inplace=True))

        self.RCAB2 = RCAB_bn_2(conv=Separate_conv, n_feat=8, kernel_size=3, reduction=1,act=nn.ReLU(True))
        self.decon5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(4, affine=True),
            nn.ReLU(inplace=True))

        self.RCAB3 = RCAB_bn_3(conv=Separate_conv, n_feat=8, kernel_size=3, reduction=1,act=nn.ReLU(True))
        self.decon6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=4, out_channels=6, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(6, affine=True),
            nn.ReLU(inplace=True))

        self.RCAB4 = RCAB_bn_4(conv=Separate_conv, n_feat=12, kernel_size=3, reduction=1,act=nn.ReLU(True))
        self.decon7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=12, out_channels=8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(8, affine=True),
            nn.ReLU(inplace=True))

        self.decon8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=14, out_channels=8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(8, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, stride=1, padding=0),
            nn.Tanh())
        self.encoder = Encoder(ngf=ngf)  #Li 加上了编码器 但是不很懂为什么？后续观察

    def forward(self, x):
        con1, con2, con3, con4, con5 = self.encoder(x)  #Li 先编码，剩余是译码步骤 #LH 得到等上采样块的网络
        decon4 = self.RCAB1(con5)
        decon4 = self.decon4(decon4)
        decon4 = torch.cat([decon4, con4], dim=1)

        decon5 = self.RCAB2(decon4)
        decon5 = self.decon5(decon5)
        decon5 = torch.cat([decon5, con3], dim=1)

        decon6= self.RCAB3(decon5)
        decon6 = self.decon6(decon6)
        decon6 = torch.cat([decon6, con2], dim=1)

        decon7 = self.RCAB4(decon6)
        decon7 = self.decon7(decon7)
        decon7 = torch.cat([decon7, con1], dim=1)

        decon8 = self.decon8(decon7)
        return decon8
        # return decon8, decon7,decon6,decon5, decon4,con1, con2, con3, con4, con5



class Discriminator(nn.Module):  #Li 这一部分是属于网络架构的那一模块呢？ #lh 鉴别器
    def __init__(self, ndf=64):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=3*2, out_channels=ndf, kernel_size=4, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(ndf, affine=True))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(ndf * 2, affine=True))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=4, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(ndf * 4, affine=True))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=4, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(ndf * 8, affine=True))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Conv2d(in_channels=ndf * 8, out_channels=1, kernel_size=4, stride=1, padding=1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)  #Li 这个*号不懂 #Lh 和指针作用相似，解引用

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    decoder = Decoder(64)


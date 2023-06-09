from torch import nn


class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super(Discriminator, self).__init__()
        layers = list()
        layers.append(nn.Conv2d(in_channels=3 * 2, out_channels=ndf, kernel_size=4, stride=2, padding=1))
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
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

from utils import *

class Discriminator(nn.Module):
    def __init__(self, nc, nd):
        super(Discriminator, self).__init__()
        self.nc=nc
        self.nd=nd
        self.main = nn.Sequential(
            nn.Conv2d(nc, nd, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nd, nd * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nd * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nd * 2, nd * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nd * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nd * 4, nd * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nd * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nd * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
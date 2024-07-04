from utils import *

## nz: Size of latent vector
## nf: Size of feature maps
## nc: No. of Channels

class Generator(nn.Module):
    def __init__(self, nz, nf, nc):
        super(Generator, self).__init__()
        self.nz=nz
        self.nf=nf
        self.nc=nc

        self.l1=nn.ConvTranspose2d(self.nz, 8*self.nf, 4, 1, 0, bias=False)
        self.l2=nn.BatchNorm2d(self.nf*8)
        self.l3=nn.ReLU()
        self.l4=nn.ConvTranspose2d(8*self.nf, 4*self.nf, 4, 2, 1, bias=False)
        self.l5=nn.BatchNorm2d(self.nf*4)
        self.l6=nn.ReLU()
        self.l7=nn.ConvTranspose2d(4*self.nf, 2*self.nf, 4, 2, 1, bias=False)
        self.l8=nn.BatchNorm2d(2*self.nf)
        self.l9=nn.ReLU()
        self.l10=nn.ConvTranspose2d(2*self.nf, self.nf, 4, 2, 1, bias=False)
        self.l11=nn.BatchNorm2d(self.nf)
        self.l12=nn.ReLU()
        self.l13=nn.ConvTranspose2d(self.nf, self.nc, 4, 2, 1, bias=False)
        self.l14=nn.Tanh()
        self.model=nn.Sequential(self.l1, self.l2, self.l3, self.l4, self.l5, self.l6,
                                 self.l7, self.l8, self.l9, self.l10, self.l11, self.l12, self.l13, self.l14)
        
    def forward(self, x):
        return self.model(x)


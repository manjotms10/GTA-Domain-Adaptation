from torch import nn
from torch.nn import functional as F

class ResNetBlock(nn.Module):    
    def __init__(self, n):
        super(ResNetBlock, self).__init__()
        self.nf = n
        self.model = self.build_block(n)
        
    def build_block(self, n):
        model = []
        model += 2 * [
            nn.ReflectionPad2d(1),
            nn.Conv2d(n, n, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(n),
            nn.ReLU(True)
        ]        
        return nn.Sequential(*model)
    
    def forward(self, x):
        return x + self.model(x)
    

class Generator(nn.Module):
    
    def __init__(self, n=128):
        super(Generator, self).__init__()
        self.n = n
        self.block = self.model(n)
        
    
    def model(self, n):
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(3, 32, kernel_size=7, padding=0,
                 bias=True),
                 nn.InstanceNorm2d(32),
                 nn.ReLU(True)]
        model += [nn.Conv2d(32, 64, kernel_size=3,
                        stride=2, padding=1, bias=True),
                              nn.InstanceNorm2d(64),
                              nn.ReLU(True)]

        model += [nn.Conv2d(64, 128, kernel_size=3,
                        stride=2, padding=1, bias=True),
                              nn.InstanceNorm2d(128),
                              nn.ReLU(True)]

        model += 6 * [ResNetBlock(128)]

        model += [nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                  nn.InstanceNorm2d(64),
                  nn.ReLU(True)]

        model += [nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                  nn.InstanceNorm2d(32),
                  nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(32, 3, kernel_size=7, stride=1, padding=0),
                 nn.ReLU(True)]

        return nn.Sequential(*model)
        
    
    def forward(self, x):
        return self.block(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        layers.extend(discriminator_block(in_channels, 64, 2, False))
        layers.extend(discriminator_block(64, 128, 2, True))
        layers.extend(discriminator_block(128, 256, 2, True))
        layers.extend(discriminator_block(256, 512 , 2, True))
        layers.append(nn.Conv2d(512, 1, 3, 1, 1))
        self.model = nn.Sequential(*layers)
        self.flat = nn.Linear(16*16, 1)
        self.out = nn.Sigmoid()

    def forward(self, img_A):
        y = self.model(img_A)
        y = y.view(y.size(0), -1)
        y = self.flat(y)
        y = self.out(y)
        return y
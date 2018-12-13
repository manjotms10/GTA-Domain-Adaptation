from torch import nn
import torch


class GeneratorUNet(nn.Module):
    def __init__(self):
        super(GeneratorUNet, self).__init__()
        
        # Convolution layers
        nc = 3
        ngf = 64
        
        # input is (nc) x 256 x 256
        self.conv1 = nn.Conv2d(nc, ngf, 4, 2, 1, bias=True)
        self.lr1 = nn.LeakyReLU(inplace=True)
        # state size. (ngf) x 128 x 128
        self.conv2 = nn.Conv2d(ngf, ngf*2, 4, 2, 1, bias=True)
        self.bn2 = nn.BatchNorm2d(ngf*2)
        self.lr2 = nn.LeakyReLU(inplace=True)
        # state size. (ngf*2) x 64 x 64
        self.conv3 = nn.Conv2d(ngf*2, ngf*4, 4, 2, 1, bias=True)
        self.bn3 = nn.BatchNorm2d(ngf*4)
        self.lr3 = nn.LeakyReLU(inplace=True)
        # state size. (ngf*4) x 32 x 32
        self.conv4 = nn.Conv2d(ngf*4, ngf*8, 4, 2, 1, bias=True)
        self.bn4 = nn.BatchNorm2d(ngf*8)
        self.lr4 = nn.LeakyReLU(inplace=True)
        # state size. (ngf*8) x 16 x 16
        self.conv5 = nn.Conv2d(ngf*8, ngf*8, 4, 2, 1, bias=True)
        self.bn5 = nn.BatchNorm2d(ngf*8)
        self.lr5 = nn.LeakyReLU(inplace=True)
        # state size. (ngf*8) x 8 x 8
        self.conv6 = nn.Conv2d(ngf*8, ngf*8, 4, 2, 1, bias=True)
        self.bn6 = nn.BatchNorm2d(ngf*8)
        self.lr6 = nn.LeakyReLU(inplace=True)
        # state size. (ngf*8) x 4 x 4
        self.conv7 = nn.Conv2d(ngf*8, ngf*8, 4, 2, 1, bias=True)
        self.bn7 = nn.BatchNorm2d(ngf*8)
        self.lr7 = nn.LeakyReLU(inplace=True)
        # state size. (ngf*8) x 2 x 2
        self.conv8 = nn.Conv2d(ngf*8, ngf*8, 4, 2, 1, bias=True)
        self.bn8 = nn.BatchNorm2d(ngf*8)
        self.r8 = nn.ReLU(inplace=True)
        
        # Transpose Convolutional Layers
        
        # input is (ngf*8) x 1 x 1
        self.tr_conv1 = nn.ConvTranspose2d(ngf*8, ngf*8, 4, 2, 1, bias=True)
        self.tr_bn1 = nn.BatchNorm2d(ngf*8)
        self.tr_r1 = nn.ReLU(inplace=True)
        # state size. (ngf*8)*2 x 2 x 2
        self.tr_conv2 = nn.ConvTranspose2d((ngf*8)*2, ngf*8, 4, 2, 1, bias=True)
        self.tr_bn2 = nn.BatchNorm2d(ngf*8)
        self.tr_r2 = nn.ReLU(inplace=True)
        # state size. (ngf*8)*2 x 4 x 4
        self.tr_conv3 = nn.ConvTranspose2d((ngf*8)*2, ngf*8, 4, 2, 1, bias=True)
        self.tr_bn3 = nn.BatchNorm2d(ngf*8)
        self.tr_r3 = nn.ReLU(inplace=True)
        # state size. (ngf*8)*2 x 8 x 8
        self.tr_conv4 = nn.ConvTranspose2d((ngf*8)*2, ngf*8, 4, 2, 1, bias=True)
        self.tr_bn4 = nn.BatchNorm2d(ngf*8)
        self.tr_r4 = nn.ReLU(inplace=True)
        # state size. (ngf*8)*2 x 16 x 16
        self.tr_conv5 = nn.ConvTranspose2d((ngf*8)*2, ngf*4, 4, 2, 1, bias=True)
        self.tr_bn5 = nn.BatchNorm2d(ngf*4)
        self.tr_r5 = nn.ReLU(inplace=True)
        # state size. (ngf*4)*2 x 32 x 32
        self.tr_conv6 = nn.ConvTranspose2d((ngf*4)*2, ngf*2, 4, 2, 1, bias=True)
        self.tr_bn6 = nn.BatchNorm2d(ngf*2)
        self.tr_r6 = nn.ReLU(inplace=True)
        # state size. (ngf*2)*2 x 64 x 64
        self.tr_conv7 = nn.ConvTranspose2d((ngf*2)*2, ngf, 4, 2, 1, bias=True)
        self.tr_bn7 = nn.BatchNorm2d(ngf)
        self.tr_r7 = nn.ReLU(inplace=True)
        # state size. (ngf)*2 x 128 x 128
        self.tr_conv8 = nn.ConvTranspose2d((ngf)*2, nc, 4, 2, 1, bias=True)
        self.out = nn.Tanh()
        # state size. (nc) x 256 x 256
    
    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.bn2(self.conv2(self.lr1(c1)))
        c3 = self.bn3(self.conv3(self.lr2(c2)))
        c4 = self.bn4(self.conv4(self.lr3(c3)))
        c5 = self.bn5(self.conv5(self.lr4(c4)))
        c6 = self.bn6(self.conv6(self.lr5(c5)))
        c7 = self.bn7(self.conv7(self.lr6(c6)))
        c8 = self.bn8(self.conv8(self.lr7(c7)))
        
        t1 = self.tr_bn1(self.tr_conv1(self.r8(c8)))
        t1 = torch.cat((t1, c7), dim=1)
        t2 = self.tr_bn2(self.tr_conv2(self.tr_r1(t1)))
        t2 = torch.cat((t2, c6), dim=1)
        t3 = self.tr_bn3(self.tr_conv3(self.tr_r2(t2)))
        t3 = torch.cat((t3, c5), dim=1)
        t4 = self.tr_bn4(self.tr_conv4(self.tr_r3(t3)))
        t4 = torch.cat((t4, c4), dim=1)
        t5 = self.tr_bn5(self.tr_conv5(self.tr_r4(t4)))
        t5 = torch.cat((t5, c3), dim=1)
        t6 = self.tr_bn6(self.tr_conv6(self.tr_r5(t5)))
        t6 = torch.cat((t6, c2), dim=1)
        t7 = self.tr_bn7(self.tr_conv7(self.tr_r6(t6)))
        t7 = torch.cat((t7, c1), dim=1)
        t8 = self.tr_conv8(self.tr_r7(t7))
        t8 = self.out(t8)
        return t8

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
    

class GeneratorResNet(nn.Module):
    
    def __init__(self, n=128):
        super(Generator, self).__init__()
        self.n = n
        self.block = self.model(n)
        
    
    def model(self, n):
        model = [nn.ReflectionPad2d(1),
                 nn.Conv2d(3, 32, kernel_size=3, padding=0,
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

        model += [nn.ReflectionPad2d(1),
                  nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=0),
                 nn.Sigmoid()]

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
        self.flat = nn.Linear(16*16*4, 1)
        self.out = nn.Sigmoid()

    def forward(self, img_A):
        y = self.model(img_A)
        y = y.view(y.size(0), -1)
        y = self.flat(y)
        y = self.out(y)
        return y


# Defining Generator and Discriminator for CycleGANS

class CycleGanResnetGenerator(nn.Module):
    def __init__(self, ngf=32, use_dropout=True):
        super(CycleGanResnetGenerator, self).__init__()

        self.in_channels = 3
        self.out_channels = 3
        self.num_resnet_blocks = 9

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(self.in_channels, ngf, kernel_size=7, padding=0,
                           bias=nn.InstanceNorm2d),
                 nn.BatchNorm2d(ngf),
                 nn.ReLU(True)]

        # we down-sample for 2 layers
        for i in range(2):
            in_ch = 2**i * ngf
            out_ch = 2 * in_ch
            model += [nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2,
                                padding=1, bias=nn.InstanceNorm2d),
                      nn.BatchNorm2d(out_ch),
                      nn.ReLU(True)]

        # Add Resnet Blocks
        in_ch = 4 * ngf
        for i in range(self.num_resnet_blocks):
            model += [CycleGanResnetBlock(in_ch, use_dropout)]

        # We up-sample for 2 layers
        for i in range(2):
            in_ch = 2**(2 - i) * ngf
            out_ch = int(in_ch / 2.0)
            model += [nn.ConvTranspose2d(in_ch, out_ch,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=nn.InstanceNorm2d),
                      nn.BatchNorm2d(out_ch),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, self.out_channels, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# Resnet Module to be used in Generator
class CycleGanResnetBlock(nn.Module):

    def __init__(self, dim, use_dropout=True):
        super(CycleGanResnetBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(dim, dim, kernel_size=3, padding=0,
                                bias=nn.InstanceNorm2d),
                      nn.BatchNorm2d(dim),
                      nn.ReLU(True)]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, padding=0,
                                 bias=nn.InstanceNorm2d),
                       nn.BatchNorm2d(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class CycleGanDiscriminator(nn.Module):
    def __init__(self, ndf=32, n_layers=3):
        super(CycleGanDiscriminator, self).__init__()

        self.input_channels = 3

        model = [nn.Conv2d(self.input_channels, ndf,
                      kernel_size=4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, True)]

        out_ch = ndf

        for n in range(1, n_layers):
            in_ch = out_ch
            out_ch = min(2**n, 8) * ndf
            model += [
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2,
                          padding=1, bias=nn.InstanceNorm2d),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, True)
            ]

        in_ch = out_ch
        out_ch = min(2**n_layers, 8) * ndf
        model += [
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=1, padding=1,
                      bias=nn.InstanceNorm2d),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, True)
        ]

        model += [nn.Conv2d(out_ch, 1, kernel_size=4, stride=1, padding=1)]

        self.model = nn.Sequential(*model)
        self.fc = nn.Linear(900,1)

    def forward(self, input):
        x = self.model(input)
        x = x.view(x.size(0), -1)
        return nn.functional.sigmoid(self.fc(x))
    
    
class DualGansGenerator(nn.Module):
    def __init__(self):
        super(DualGansGenerator, self).__init__()
        nc = 3
        ngf = 64
        
        # Convolution layers
        # input is (nc) x 256 x 256
        self.conv1 = nn.Conv2d(nc, ngf, 4, 2, 1, bias=True)
        self.lr1 = nn.LeakyReLU(inplace=True)
        # state size. (ngf) x 128 x 128
        self.conv2 = nn.Conv2d(ngf, ngf*2, 4, 2, 1, bias=True)
        self.bn2 = nn.BatchNorm2d(ngf*2)
        self.lr2 = nn.LeakyReLU(inplace=True)
        # state size. (ngf*2) x 64 x 64
        self.conv3 = nn.Conv2d(ngf*2, ngf*4, 4, 2, 1, bias=True)
        self.bn3 = nn.BatchNorm2d(ngf*4)
        self.lr3 = nn.LeakyReLU(inplace=True)
        # state size. (ngf*4) x 32 x 32
        self.conv4 = nn.Conv2d(ngf*4, ngf*8, 4, 2, 1, bias=True)
        self.bn4 = nn.BatchNorm2d(ngf*8)
        self.lr4 = nn.LeakyReLU(inplace=True)
        # state size. (ngf*8) x 16 x 16
        self.conv5 = nn.Conv2d(ngf*8, ngf*8, 4, 2, 1, bias=True)
        self.bn5 = nn.BatchNorm2d(ngf*8)
        self.lr5 = nn.LeakyReLU(inplace=True)
        # state size. (ngf*8) x 8 x 8
        self.conv6 = nn.Conv2d(ngf*8, ngf*8, 4, 2, 1, bias=True)
        self.bn6 = nn.BatchNorm2d(ngf*8)
        self.lr6 = nn.LeakyReLU(inplace=True)
        # state size. (ngf*8) x 4 x 4
        self.conv7 = nn.Conv2d(ngf*8, ngf*8, 4, 2, 1, bias=True)
        self.bn7 = nn.BatchNorm2d(ngf*8)
        self.lr7 = nn.LeakyReLU(inplace=True)
        # state size. (ngf*8) x 2 x 2
        self.conv8 = nn.Conv2d(ngf*8, ngf*8, 4, 2, 1, bias=True)
        self.bn8 = nn.BatchNorm2d(ngf*8)
        self.r8 = nn.ReLU(inplace=True)
        
        # Transposed Convolutional Layers
        
        # input is (ngf*8) x 1 x 1
        self.tr_conv1 = nn.ConvTranspose2d(ngf*8, ngf*8, 4, 2, 1, bias=True)
        self.tr_bn1 = nn.BatchNorm2d(ngf*8)
        self.tr_d1 = nn.Dropout2d(p=0.5, inplace=True)
        self.tr_r1 = nn.ReLU(inplace=True)
        # state size. (ngf*8)*2 x 2 x 2
        self.tr_conv2 = nn.ConvTranspose2d((ngf*8)*2, ngf*8, 4, 2, 1, bias=True)
        self.tr_bn2 = nn.BatchNorm2d(ngf*8)
        self.tr_d2 = nn.Dropout2d(p=0.5, inplace=True)
        self.tr_r2 = nn.ReLU(inplace=True)
        # state size. (ngf*8)*2 x 4 x 4
        self.tr_conv3 = nn.ConvTranspose2d((ngf*8)*2, ngf*8, 4, 2, 1, bias=True)
        self.tr_bn3 = nn.BatchNorm2d(ngf*8)
        self.tr_d3 = nn.Dropout2d(p=0.5, inplace=True)
        self.tr_r3 = nn.ReLU(inplace=True)
        # state size. (ngf*8)*2 x 8 x 8
        self.tr_conv4 = nn.ConvTranspose2d((ngf*8)*2, ngf*8, 4, 2, 1, bias=True)
        self.tr_bn4 = nn.BatchNorm2d(ngf*8)
        self.tr_r4 = nn.ReLU(inplace=True)
        # state size. (ngf*8)*2 x 16 x 16
        self.tr_conv5 = nn.ConvTranspose2d((ngf*8)*2, ngf*4, 4, 2, 1, bias=True)
        self.tr_bn5 = nn.BatchNorm2d(ngf*4)
        self.tr_r5 = nn.ReLU(inplace=True)
        # state size. (ngf*4)*2 x 32 x 32
        self.tr_conv6 = nn.ConvTranspose2d((ngf*4)*2, ngf*2, 4, 2, 1, bias=True)
        self.tr_bn6 = nn.BatchNorm2d(ngf*2)
        self.tr_r6 = nn.ReLU(inplace=True)
        # state size. (ngf*2)*2 x 64 x 64
        self.tr_conv7 = nn.ConvTranspose2d((ngf*2)*2, ngf, 4, 2, 1, bias=True)
        self.tr_bn7 = nn.BatchNorm2d(ngf)
        self.tr_r7 = nn.ReLU(inplace=True)
        # state size. (ngf)*2 x 128 x 128
        self.tr_conv8 = nn.ConvTranspose2d((ngf)*2, nc, 4, 2, 1, bias=True)
        self.out = nn.Tanh()
        # state size. (nc) x 256 x 256
    
    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.bn2(self.conv2(self.lr1(c1)))
        c3 = self.bn3(self.conv3(self.lr2(c2)))
        c4 = self.bn4(self.conv4(self.lr3(c3)))
        c5 = self.bn5(self.conv5(self.lr4(c4)))
        c6 = self.bn6(self.conv6(self.lr5(c5)))
        c7 = self.bn7(self.conv7(self.lr6(c6)))
        c8 = self.bn8(self.conv8(self.lr7(c7)))
        
        t1 = self.tr_d1(self.tr_bn1(self.tr_conv1(self.r8(c8))))
        t1 = torch.cat((t1, c7), dim=1)
        t2 = self.tr_d2(self.tr_bn2(self.tr_conv2(self.tr_r1(t1))))
        t2 = torch.cat((t2, c6), dim=1)
        t3 = self.tr_d3(self.tr_bn3(self.tr_conv3(self.tr_r2(t2))))
        t3 = torch.cat((t3, c5), dim=1)
        t4 = self.tr_bn4(self.tr_conv4(self.tr_r3(t3)))
        t4 = torch.cat((t4, c4), dim=1)
        t5 = self.tr_bn5(self.tr_conv5(self.tr_r4(t4)))
        t5 = torch.cat((t5, c3), dim=1)
        t6 = self.tr_bn6(self.tr_conv6(self.tr_r5(t5)))
        t6 = torch.cat((t6, c2), dim=1)
        t7 = self.tr_bn7(self.tr_conv7(self.tr_r6(t6)))
        t7 = torch.cat((t7, c1), dim=1)
        t8 = self.tr_conv8(self.tr_r7(t7))
        t8 = self.out(t8)
        return t8
    
    
class DualGansDiscriminator(nn.Module):
    def __init__(self):
        super(DualGansDiscriminator, self).__init__()
        nc = 3
        ndf = 64
        
        self.net = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 128 x 128
            nn.MaxPool2d((2, 2)), 
            
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 32 x 32
            nn.MaxPool2d((2, 2)),
    
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            )
        
        self.flat = nn.Linear(ndf * 8 * 2 * 2, 1)
        self.out = nn.Sigmoid()
        
    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size()[0], -1)
        x = self.flat(x)
        x = self.out(x)
        return x

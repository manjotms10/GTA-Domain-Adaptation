from torch import nn
from torch.nn import functional as F

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=3, kernel_size=3, out_channels=32, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer2 = nn.Conv2d(in_channels=32, kernel_size=3, out_channels=64, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.layer3 = nn.Conv2d(in_channels=64, kernel_size=3, out_channels=128, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.layer4 = nn.Conv2d(in_channels=128, kernel_size=3, out_channels=256, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.layer5 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.bn5 = nn.Upsample(scale_factor=2)
        self.layer6 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.bn6 = nn.Upsample(scale_factor=2)
        self.layer7 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False)
        self.bn7 = nn.Upsample(scale_factor=2)
        self.layer8 = nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False)
        self.bn8 = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.layer1(x))), kernel_size=2)
        x = F.max_pool2d(F.relu(self.bn2(self.layer2(x))), kernel_size=2)
        x = F.max_pool2d(F.relu(self.bn3(self.layer3(x))), kernel_size=2)
        x = F.max_pool2d(F.relu(self.bn4(self.layer4(x))), kernel_size=2)
        x = F.relu(x)
        x = self.bn5(self.layer5(x))
        x = self.bn6(self.layer6(x))
        x = self.bn7(self.layer7(x))
        x = self.bn8(self.layer8(x))
        return x


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
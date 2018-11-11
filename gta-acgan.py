import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import glob
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import os
nn = torch.nn
F = torch.nn.functional
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DataLoader:
    def __init__(self, data_root, image_size, batch_size):
        '''
        Parameters:
        
        '''
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.data_path = data_root
        self.image_size = image_size
        self.batch_size = batch_size
        self.train_names = glob.glob(self.data_path + 'real_A/*')
        self.names = [self.train_names[i].split('/')[-1] for i in range(len(self.train_names))]
        self.data_transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(image_size),
                torchvision.transforms.CenterCrop(image_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    
    def image_loader(self, image_name):
        """load image, returns cuda tensor"""
        image = Image.open(image_name)
        image = self.data_transforms(image).float()
        image = torch.autograd.Variable(image, requires_grad=True)
        image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
        return image[0].to(self.device)  #assumes that you're using GPU

    def show(self, img):
        npimg = img.cpu().detach().numpy()
        npimg = np.transpose(npimg, (1,2,0))
        if npimg.shape[2] == 3:
            plt.imshow(npimg)
        else:
            plt.imshow(npimg[:,:,0], cmap='gray')
            
    def imshow(self, img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.cpu().detach().numpy()
        plt.figure(figsize = (10,2))
        plt.imshow(np.transpose(npimg, (1, 2, 0)), aspect='auto')

    def data_generator(self):
        root = self.data_path
        batch_size = self.batch_size
        
        images_dir = root + 'real_A/'
        labels_dir = root + 'fake_B/'

        while True:
            x, y = [], []
            idx = np.random.choice(self.names, batch_size)
            for i in range(idx.shape[0]):
                x.append(self.image_loader(images_dir + idx[i]))
                y.append(self.image_loader(labels_dir + idx[i]))
            yield torch.stack(x), torch.stack(y)
            
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.01)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
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
        self.bn5 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.layer6 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.bn6 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.layer7 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False)
        self.bn7 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.layer8 = nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False)
        self.bn8 = nn.UpsamplingBilinear2d(scale_factor=2)
        
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
            
data = DataLoader(data_root= './gta/', image_size=(256, 256), batch_size=64)
x, y = next(data.data_generator())
x, y = x.to(device), y.to(device)

opt = dict()

opt["n_epochs"] = 100
opt["dataset_name"] = 'GTA'
opt["batch_size"] = 64
opt["lr"] = 0.0002
opt["b1"] = 0.5
opt["b2"] = 0.99
opt["decay_epoch"] = 100
opt["n_cpu"] = 4
opt["img_height"] = 256
opt["img_width"] = 256
opt["channels"] = 3
opt["sample_interval"] = 25
opt["checkpoint_interval"] = 1

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


ensure_dir('saved_images/%s' % 'GTA')
ensure_dir('saved_models/%s' % 'GTA')


criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

lambda_pixel = 100

generator = Generator()
discriminator = Discriminator()

if torch.cuda.is_available():
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()
    
generator = nn.DataParallel(generator, list(range(torch.cuda.device_count())))
discriminator = nn.DataParallel(discriminator, list(range(torch.cuda.device_count())))

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt["lr"], betas=(opt["b1"], opt["b2"]))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt["lr"], betas=(opt["b1"], opt["b2"]))

def sample_images(batches_done):
    x, y = next(data.data_generator())
    real_A = Variable(x.type(Tensor))
    real_B = Variable(y.type(Tensor))
    fake_B = generator(real_A)
    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    save_image(img_sample, 'saved_images/%s.png' % (epoch + i), nrow=5, normalize=True)
    

    
for epoch in range(opt['n_epochs']):
    for i in range(2500 // opt['batch_size']):

        x, y = next(data.data_generator())
        
        real_A = Variable(x.type(Tensor))
        real_B = Variable(y.type(Tensor))

        valid = Variable(Tensor(np.ones((real_A.size(0), 1))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), 1))), requires_grad=False)

        optimizer_G.zero_grad()

        fake_B = generator(real_A)
        pred_fake = discriminator(fake_B)
        loss_GAN = criterion_GAN(pred_fake, valid)
        loss_pixel = criterion_pixelwise(fake_B, real_B)

        loss_G = loss_GAN + lambda_pixel * loss_pixel

        loss_G.backward()

        optimizer_G.step()

        optimizer_D.zero_grad()

        pred_real = discriminator(real_A)
        loss_real = criterion_GAN(pred_real, valid)

        pred_fake = discriminator(fake_B.detach())
        loss_fake = criterion_GAN(pred_fake, fake)

        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        optimizer_D.step()

        print("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f]" %
                                                        (epoch, opt["n_epochs"],
                                                        i, 2500//opt["batch_size"],
                                                        loss_D.item(), loss_G.item(),
                                                        loss_pixel.item(), loss_GAN.item()))

        if i % opt["sample_interval"] == 0:
            sample_images(i)


    if opt['checkpoint_interval'] != -1 and epoch % opt['checkpoint_interval'] == 0:
        torch.save(generator.state_dict(), 'saved_models/generator_%d.pth' % (epoch))
        torch.save(discriminator.state_dict(), 'saved_models/discriminator_%d.pth' % (epoch))
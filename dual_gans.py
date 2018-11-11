
# coding: utf-8

# In[1]:


from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import glob
from PIL import Image
import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


# In[2]:


# Root directory for dataset
data_root = "/datasets/home/32/232/tdobhal/Project/6_train/images/"

# Number of images in the directory
num_images = 23418

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this size using a transformer.
image_size = 256

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 100

# Learning rate for optimizers
lr = 0.00005

# Alpha hyperparam for RMS optimizers
alpha = 0.9

# Number of GPUs available. Use 0 for CPU mode.
ngpu = torch.cuda.device_count()

# Device to run on
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[3]:


class DataLoader:
    def __init__(self):
        '''
        Parameters:
        
        '''
        self.device = device
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
        image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
        return image[0].to(self.device)  # assumes that you're using GPU

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
            yield torch.stack(torch.from_numpy(x)).to(self.device), torch.stack(torch.from_numpy(y)).to(self.device)


# In[4]:


data = DataLoader()


# In[5]:


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.01)
        torch.nn.init.constant_(m.bias.data, 0.0)


# In[6]:


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        
        # Convolution layers
        
        # input is (nc) x 256 x 256
        self.conv1 = nn.Conv2d(nc, ngf, 4, 2, 1, bias=True)
        # state size. (ngf) x 128 x 128
        self.conv2 = nn.Conv2d(ngf, ngf*2, 4, 2, 1, bias=True)
        self.bn2 = nn.BatchNorm2d(ngf*2)
        # state size. (ngf*2) x 64 x 64
        self.conv3 = nn.Conv2d(ngf*2, ngf*4, 4, 2, 1, bias=True)
        self.bn3 = nn.BatchNorm2d(ngf*4)
        # state size. (ngf*4) x 32 x 32
        self.conv4 = nn.Conv2d(ngf*4, ngf*8, 4, 2, 1, bias=True)
        self.bn4 = nn.BatchNorm2d(ngf*8)
        # state size. (ngf*8) x 16 x 16
        self.conv5 = nn.Conv2d(ngf*8, ngf*8, 4, 2, 1, bias=True)
        self.bn5 = nn.BatchNorm2d(ngf*8)
        # state size. (ngf*8) x 8 x 8
        self.conv6 = nn.Conv2d(ngf*8, ngf*8, 4, 2, 1, bias=True)
        self.bn6 = nn.BatchNorm2d(ngf*8)
        # state size. (ngf*8) x 4 x 4
        self.conv7 = nn.Conv2d(ngf*8, ngf*8, 4, 2, 1, bias=True)
        self.bn7 = nn.BatchNorm2d(ngf*8)
        # state size. (ngf*8) x 2 x 2
        self.conv8 = nn.Conv2d(ngf*8, ngf*8, 4, 2, 1, bias=True)
        self.bn8 = nn.BatchNorm2d(ngf*8)
        
        # Transpose Convolutional Layers
        
        # input is (ngf*8) x 1 x 1
        self.tr_conv1 = nn.ConvTranspose2d(ngf*8, ngf*8, 4, 2, 1, bias=True)
        self.tr_bn1 = nn.BatchNorm2d(ngf*8)
        # state size. (ngf*8)*2 x 2 x 2
        self.tr_conv2 = nn.ConvTranspose2d((ngf*8)*2, ngf*8, 4, 2, 1, bias=True)
        self.tr_bn2 = nn.BatchNorm2d(ngf*8)
        # state size. (ngf*8)*2 x 4 x 4
        self.tr_conv3 = nn.ConvTranspose2d((ngf*8)*2, ngf*8, 4, 2, 1, bias=True)
        self.tr_bn3 = nn.BatchNorm2d(ngf*8)
        # state size. (ngf*8)*2 x 8 x 8
        self.tr_conv4 = nn.ConvTranspose2d((ngf*8)*2, ngf*8, 4, 2, 1, bias=True)
        self.tr_bn4 = nn.BatchNorm2d(ngf*8)
        # state size. (ngf*8)*2 x 16 x 16
        self.tr_conv5 = nn.ConvTranspose2d((ngf*8)*2, ngf*4, 4, 2, 1, bias=True)
        self.tr_bn5 = nn.BatchNorm2d(ngf*4)
        # state size. (ngf*4)*2 x 32 x 32
        self.tr_conv6 = nn.ConvTranspose2d((ngf*4)*2, ngf*2, 4, 2, 1, bias=True)
        self.tr_bn6 = nn.BatchNorm2d(ngf*2)
        # state size. (ngf*2)*2 x 64 x 64
        self.tr_conv7 = nn.ConvTranspose2d((ngf*2)*2, ngf, 4, 2, 1, bias=True)
        self.tr_bn7 = nn.BatchNorm2d(ngf)
        # state size. (ngf)*2 x 128 x 128
        self.tr_conv8 = nn.ConvTranspose2d((ngf)*2, nc, 4, 2, 1, bias=True)
        # state size. (nc) x 256 x 256
    
    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.bn2(self.conv2(nn.LeakyReLU(c1)))
        c3 = self.bn3(self.conv3(nn.LeakyReLU(c2)))
        c4 = self.bn4(self.conv4(nn.LeakyReLU(c3)))
        c5 = self.bn5(self.conv5(nn.LeakyReLU(c4)))
        c6 = self.bn6(self.conv6(nn.LeakyReLU(c5)))
        c7 = self.bn7(self.conv7(nn.LeakyReLU(c6)))
        c8 = self.bn8(self.conv8(nn.LeakyReLU(c7)))
        
        t1 = self.tr_bn1(self.tr_conv1(nn.ReLU(c8)))
        t1 = torch.cat(t1, c7, axis=1)
        t2 = self.tr_bn2(self.tr_conv2(nn.ReLU(t1)))
        t2 = torch.cat(t2, c6, axis=1)
        t3 = self.tr_bn3(self.tr_conv3(nn.ReLU(t2)))
        t3 = torch.cat(t3, c5, axis=1)
        t4 = self.tr_bn4(self.tr_conv4(nn.ReLU(t3)))
        t4 = torch.cat(t4, c4, axis=1)
        t5 = self.tr_bn5(self.tr_conv5(nn.ReLU(t4)))
        t5 = torch.cat(t5, c3, axis=1)
        t6 = self.tr_bn6(self.tr_conv6(nn.ReLU(t5)))
        t6 = torch.cat(t6, c2, axis=1)
        t7 = self.tr_bn7(self.tr_conv7(nn.ReLU(t6)))
        t7 = torch.cat(t7, c1, axis=1)
        t8 = self.tr_conv8(nn.ReLU(t7))
        t8 = nn.Tanh(t8)
        return t8
    


# In[7]:


gen_a = Generator()
gen_b = Generator()


# In[8]:


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
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
            
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=True),
            nn.Sigmoid()
            # state size. (1) x 1 x 1
        )

    def forward(self, input):
        return self.main(input)


# In[9]:


dis_a = Discriminator()
dis_b = Discriminator()


# In[10]:


gen_a = nn.DataParallel(gen_a, list(range(ngpu)))
dis_a = nn.DataParallel(dis_a, list(range(ngpu)))
gen_b = nn.DataParallel(gen_b, list(range(ngpu)))
dis_b = nn.DataParallel(dis_b, list(range(ngpu)))

gen_a.apply(weights_init_normal)
dis_a.apply(weights_init_normal)
gen_b.apply(weights_init_normal)
dis_b.apply(weights_init_normal)


# In[14]:


criterion = torch.nn.BCELoss()

optim_gen_a = torch.optim.RMSprop(gen_a.parameters(), lr, alpha)
optim_gen_b = torch.optim.RMSprop(gen_b.parameters(), lr, alpha)
optim_dis_a = torch.optim.RMSprop(dis_a.parameters(), lr, alpha)
optim_dis_b = torch.optim.RMSprop(dis_b.parameters(), lr, alpha)


# ### Train Loop

# In[ ]:


sample_interval = 25
checkpoint_interval = 1

for epoch in range(num_epochs):
    for i in range(num_images // batch_size):
        x, y = next(data.data_generator())
        real_a = Variable(x)
        real_b = Variable(y)      
        valid = Variable(torch.ones((real_A.size(0), 1)).to(device), requires_grad=False)
        fake = Variable(torch.zeros((real_A.size(0), 1)).to(device), requires_grad=False)
        
        # Training Discriminator A with real_A batch
        optim_dis_a.zero_grad();
        pred_real_dis_a = dis_a(real_a).view(-1)
        err_real_dis_a = criterion(pred_real_dis_a, valid)
        err_real_dis_a.backward()
        
        # Training Discriminator B with real_B batch
        optim_dis_b.zero_grad();
        pred_real_dis_b = dis_b(real_b).view(-1)
        err_real_dis_b = criterion(pred_real_dis_b, valid)
        err_real_dis_b.backward()
        
        # Training Discriminator B with fake_B batch of of Generator A
        fake_b = gen_a(real_a)
        pred_fake_dis_b = dis_b(fake_b.detach()).view(-1)
        err_fake_dis_b = criterion(pred_fake_dis_b, fake)
        err_fake_dis_b.backward()
        
        # Training Discriminator A with fake_A batch of of Generator B
        fake_a = gen_b(real_b)
        pred_fake_dis_a = dis_a(fake_a.detach()).view(-1)
        err_fake_dis_a = criterion(pred_fake_dis_a, fake)
        err_fake_dis_a.backward()
        
        # Update params of Discriminator A and B
        err_dis_a = err_real_dis_a + err_fake_dis_a
        optim_dis_a.step()
        err_dis_b = err_real_dis_b + err_fake_dis_b
        optim_dis_b.step()
        
        # Train and update Generator A based on Discriminator B's prediction
        optim_gen_a.zero_grad()
        pred_out_dis_b = dis_b(fake_b).view(-1)
        err_gen_a = criterion(pred_out_dis_b, valid)
        err_gen_a.backward()
        optim_gen_a.step()
        
        # Train and update Generator B based on Discriminator A's prediction
        optim_gen_b.zero_grad()
        pred_out_dis_a = dis_a(fake_a).view(-1)
        err_gen_b = criterion(pred_out_dis_a, valid)
        err_gen_b.backward()
        optim_gen_b.step()
        
        # Print statistics and save checkpoints
        print("\r[Epoch %d/%d] [Batch %d/%d] [D_A loss: %f] [D_B loss: %f] [G_A loss: %f, G_B loss: %f]" %
                                                        (epoch, num_epochs,
                                                        i, num_images//batch_size,
                                                        err_dis_a.item(), err_dis_b.item(), 
                                                        err_gen_a.item(), err_gen_b.item()))

        if i % sample_interval == 0:
            img_sample = torch.cat((real_a.data, fake_a.data, real_b.data, fake_b.data), -2)
            save_image(img_sample, 'saved_images/%s.png' % (epoch + i), nrow=5, normalize=True)


    if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
        torch.save(gen_a.state_dict(), 'saved_models/generator_a_%d.pth' % (epoch))
        torch.save(gen_b.state_dict(), 'saved_models/generator_b_%d.pth' % (epoch))
        torch.save(dis_a.state_dict(), 'saved_models/discriminator_a_%d.pth' % (epoch))
        torch.save(dis_b.state_dict(), 'saved_models/discriminator_b_%d.pth' % (epoch))


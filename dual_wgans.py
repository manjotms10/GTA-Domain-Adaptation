
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
from torch.autograd import grad
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


# In[2]:


# Root directory for project
proj_root = "/datasets/home/32/232/tdobhal/Project/"

# Root directory for dataset
data_root = "/datasets/home/32/232/tdobhal/Project/6_train/images/"

# Number of images in the directory
num_images = 23418

# Batch size during training
batch_size = 16

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

# Number of training loops for critic/discriminator
num_critic = 2

# Learning rate for optimizers
lr = 0.00005

# Alpha hyperparam for RMS optimizers
alpha = 0.9

# Lambda hyperparam for gradient penalty
lambda_gradient = 0.1

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
            yield torch.stack(x), torch.stack(y)


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
    


# In[7]:


gen_a = Generator().to(device)
gen_b = Generator().to(device)


# In[8]:


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 128 x 128
            nn.MaxPool2d((2, 2)), 
            
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 32 x 32
            nn.MaxPool2d((2, 2)),
    
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            )
        
        self.flat = nn.Linear(ndf * 8 * 2 * 2, 1)
        
    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size()[0], -1)
        x = self.flat(x)
        return x


# In[9]:


dis_a = Discriminator().to(device)
dis_b = Discriminator().to(device)


# In[10]:


# DataParallel for more than 1 gpu
# gen_a = nn.DataParallel(gen_a, list(range(ngpu)))
# dis_a = nn.DataParallel(dis_a, list(range(ngpu)))
# gen_b = nn.DataParallel(gen_b, list(range(ngpu)))
# dis_b = nn.DataParallel(dis_b, list(range(ngpu)))

gen_a.apply(weights_init_normal)
dis_a.apply(weights_init_normal)
gen_b.apply(weights_init_normal)
dis_b.apply(weights_init_normal)


# In[11]:


# Gradient penalty for Wasserstein Loss
def gradient_penalty(real_dis, fake_dis, discriminator):
    shape = [real_dis.size(0)] + [1] * (real_dis.dim() - 1)
    alpha = torch.rand(shape).to(device)
    z = real_dis + alpha * (fake_dis - real_dis)

    # gradient penalty
    z = Variable(z, requires_grad=True).to(device)
    output = discriminator(z)
    g = grad(output, z, grad_outputs=torch.ones(output.size()).to(device), create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1)**2).mean()

    return lambda_gradient * gp


# In[12]:


optim_gen_a = torch.optim.RMSprop(gen_a.parameters(), lr, alpha)
optim_gen_b = torch.optim.RMSprop(gen_b.parameters(), lr, alpha)
optim_dis_a = torch.optim.RMSprop(dis_a.parameters(), lr, alpha)
optim_dis_b = torch.optim.RMSprop(dis_b.parameters(), lr, alpha)


# ### Train Loop

# In[ ]:


sample_interval = 25
checkpoint_interval = 500

prev_load = 0
if(prev_load):
    gen_a.load_state_dict(torch.load(proj_root + 'saved_models/dual_wgans/generator_a_0_1000.pth'))
    dis_a.load_state_dict(torch.load(proj_root + 'saved_models/dual_wgans/discriminator_a_0_1000.pth'))
    gen_b.load_state_dict(torch.load(proj_root + 'saved_models/dual_wgans/generator_b_0_1000.pth'))
    dis_b.load_state_dict(torch.load(proj_root + 'saved_models/dual_wgans/discriminator_b_0_1000.pth'))

    
for epoch in range(num_epochs):
    for i in range(num_images // batch_size):
        # Adjust training loop
        if(prev_load and i <= 1000 and epoch <= 0):
            continue     
        
        for j in range(num_critic):
            x, y = next(data.data_generator())
            real_a = Variable(x).to(device)
            real_b = Variable(y).to(device)
        
            # Training Discriminator A with real_A batch
            optim_dis_a.zero_grad();
            pred_real_dis_a = dis_a(real_a).view(-1)
            
            # Training Discriminator B with real_B batch
            optim_dis_b.zero_grad();
            pred_real_dis_b = dis_b(real_b).view(-1)
            
            # Training Discriminator B with fake_B batch of Generator A
            fake_b = gen_a(real_a)
            pred_fake_dis_b = dis_b(fake_b.detach()).view(-1)
            
            # Training Discriminator A with fake_A batch of Generator B
            fake_a = gen_b(real_b)
            pred_fake_dis_a = dis_a(fake_a.detach()).view(-1)
            
            # Update params of Discriminator A and B
            err_dis_a = pred_fake_dis_a.mean() - pred_real_dis_a.mean() + gradient_penalty(real_a.data, fake_a.data, dis_a)
            err_dis_a.backward()
            optim_dis_a.step()
            err_dis_b = pred_fake_dis_b.mean() - pred_real_dis_b.mean() + gradient_penalty(real_b.data, fake_b.data, dis_b)
            err_dis_b.backward()
            optim_dis_b.step()
            
        # Train and update Generator A based on Discriminator B's prediction
        x, y = next(data.data_generator())
        real_a = Variable(x).to(device)
        real_b = Variable(y).to(device)
        
        optim_gen_a.zero_grad()
        fake_b_gen = gen_a(fake_a)
        pred_out_dis_b = dis_b(fake_b_gen).view(-1)
        err_gen_a = -pred_out_dis_b.mean()
        err_gen_a.backward()
        optim_gen_a.step()
        
        # Train and update Generator B based on Discriminator A's prediction
        optim_gen_b.zero_grad()
        fake_a_gen = gen_b(fake_b)
        pred_out_dis_a = dis_a(fake_a_gen).view(-1)
        err_gen_b = -pred_out_dis_a.mean()
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
            save_image(img_sample, proj_root + 'saved_images/dual_wgans/%d_%d.png' % (epoch, i), nrow=5, normalize=True)


        if i % checkpoint_interval == 0:
            torch.save(gen_a.state_dict(), proj_root + 'saved_models/dual_wgans/generator_a_%d_%d.pth' % (epoch, i))
            torch.save(gen_b.state_dict(), proj_root + 'saved_models/dual_wgans/generator_b_%d_%d.pth' % (epoch, i))
            torch.save(dis_a.state_dict(), proj_root + 'saved_models/dual_wgans/discriminator_a_%d_%d.pth' % (epoch, i))
            torch.save(dis_b.state_dict(), proj_root + 'saved_models/dual_wgans/discriminator_b_%d_%d.pth' % (epoch, i))


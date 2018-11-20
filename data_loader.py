import glob

import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt


class DataLoader:
    def __init__(self, data_root, image_size, batch_size, paired=True):
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
        self.paired = paired

    def image_loader(self, image_name):
        """load image, returns cuda tensor"""
        image = Image.open(image_name)
        image = self.data_transforms(image).float()
        image = torch.autograd.Variable(image, requires_grad=False)
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
            x_idx = np.random.choice(self.names, batch_size)

            if self.paired:
                y_idx = x_idx
            else:
                y_idx = np.random.choice(self.names, batch_size)

            for i in range(x_idx.shape[0]):
                x.append(self.image_loader(images_dir + x_idx[i]))
                y.append(self.image_loader(labels_dir + y_idx[i]))

            yield torch.stack(x), torch.stack(y)

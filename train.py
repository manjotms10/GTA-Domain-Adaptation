import os
import numpy as np
import torch
from torch.autograd import Variable
from data_loader import DataLoader
from networks import GeneratorUNet, GeneratorResNet, Discriminator, ResNetBlock
from utils import ensure_dir, get_opts, weights_init_normal, sample_images
from logger import logger

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

data = DataLoader(data_root='../gta/', image_size=(512, 512), batch_size=16)
opt = get_opts()

ensure_dir('saved_images/%s' % 'GTA')
ensure_dir('saved_models/%s' % 'GTA')

criterion_GAN = torch.nn.MSELoss().to(device)
criterion_pixelwise = torch.nn.L1Loss().to(device)

lambda_pixel = 10

generator = GeneratorUNet().to(device)
discriminator = Discriminator().to(device)

generator = torch.nn.DataParallel(generator, list(range(torch.cuda.device_count())))
discriminator = torch.nn.DataParallel(discriminator, list(range(torch.cuda.device_count())))

if opt['load_model']:
    if os.path.isfile("saved_models/generator.pth"):
        generator.load_state_dict(torch.load("saved_models/generator.pth"))
    if os.path.isfile("saved_models/discriminator.pth"):
        discriminator.load_state_dict(torch.load("saved_models/discriminator.pth"))
else:
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt["lr"], betas=(opt["b1"], opt["b2"]))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt["lr"], betas=(opt["b1"], opt["b2"]))

for epoch in range(opt['n_epochs']):
    for i in range(25000 // opt['batch_size']):

        y, x = next(data.data_generator())

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

        pred_real = discriminator(real_B)
        loss_real = criterion_GAN(pred_real, valid)
        pred_fake = discriminator(fake_B.detach())
        loss_fake = criterion_GAN(pred_fake, fake)

        loss_D = 0.5 * (loss_real + loss_fake)
        loss_D.backward()
        optimizer_D.step()

        message = ("\r[Epoch {}/{}] [Batch {}/{}] [D loss: {}] [G loss: {}, pixel: {}, adv: {}]"
                .format(epoch, opt["n_epochs"], i, 25000//opt["batch_size"],
                        loss_D.item(), loss_G.item(), loss_pixel.item(), loss_GAN.item()))
        print(message)
        logger.info(message)

        if i % opt["sample_interval"] == 0:
            sample_images(data, i, generator, "{}-{}".format(epoch,i))

    if opt['checkpoint_interval'] != -1 and epoch % opt['checkpoint_interval'] == 0:
        torch.save(generator.state_dict(), 'saved_models/generator.pth')
        torch.save(discriminator.state_dict(), 'saved_models/discriminator.pth')

import itertools

import numpy as np
import torch
from torch import nn, Tensor
from torch.autograd import Variable
from torchvision.utils import save_image

from utils import EpochTracker, weights_init_normal
from networks import DualGansDiscriminator, DualGansGenerator


class DualGANs:

    def __init__(self, device, file_prefix, learning_rate=0.0005, alpha=0.9,
                 train=False, semi_supervised=False):
        
        print("Starting Dual Gans with Train = {} and Semi Supervised = {}".format(train, semi_supervised))
        
        if semi_supervised is True:
            self.architecture = 'dual_gans_semi_'
        else:
            self.architecture = 'dual_gans_un_'
            
        self.is_train = train
        self.is_semi_supervised = semi_supervised
        self.device = device
        self.file_prefix = file_prefix

        self.epoch_tracker = EpochTracker(file_prefix + self.architecture + "epoch.txt")

        self.gen_a_file = file_prefix + self.architecture + 'generator_a.pth'
        self.gen_b_file = file_prefix + self.architecture + 'generator_b.pth'
        self.dis_a_file = file_prefix + self.architecture + 'discriminator_a.pth'
        self.dis_b_file = file_prefix + self.architecture + 'discriminator_b.pth'

        if self.epoch_tracker.file_exists or not self.is_train:
            self.GenA = self.init_net(DualGansGenerator(), self.gen_a_file)
            self.GenB = self.init_net(DualGansGenerator(), self.gen_b_file)
        else:
            self.GenA = self.init_net(DualGansGenerator())
            self.GenB = self.init_net(DualGansGenerator())

        self.real_A = self.real_B = self.fake_A = self.fake_B = self.new_A = self.new_B = None

        if train:
            if self.epoch_tracker.file_exists:
                self.DisA = self.init_net(DualGansDiscriminator(), self.dis_a_file)
                self.DisB = self.init_net(DualGansDiscriminator(), self.dis_b_file)
            else:
                self.DisA = self.init_net(DualGansDiscriminator())
                self.DisB = self.init_net(DualGansDiscriminator())

            # define loss functions
            self.criterionGAN = nn.BCELoss()
            self.criterionWasserstein = nn.L1Loss()
            self.criterionSupervised = nn.L1Loss()

            # initialize optimizers
            self.optimizer_g = torch.optim.RMSprop(itertools.chain(self.GenA.parameters(), self.GenB.parameters()),                                                                                learning_rate, alpha)
            self.optimizer_d = torch.optim.RMSprop(itertools.chain(self.DisA.parameters(), self.DisB.parameters()),                                                                                learning_rate, alpha)
            self.optimizers = [self.optimizer_g, self.optimizer_d]

            self.loss_disA = self.loss_disB = 0
            self.loss_genA = self.loss_genB = 0
            self.supervised_A = self.supervised_B = 0
            self.loss_G = 0
        else:
            self.pixelLoss = nn.L1Loss()
            self.test_A = self.test_B = 0

    def set_input(self, real_A, real_B):
        self.real_A = real_A.to(self.device)
        self.real_B = real_B.to(self.device)

    def forward(self):
        self.fake_B = self.GenA(self.real_A).to(self.device)
        self.new_A = self.GenB(self.fake_B).to(self.device)

        self.fake_A = self.GenB(self.real_B).to(self.device)
        self.new_B = self.GenA(self.fake_A).to(self.device)

    def backward_d(self, netD, real, fake):
        true = Variable(Tensor(np.ones((real.size(0), 1))), requires_grad=False).to(self.device)
        false = Variable(Tensor(np.zeros((real.size(0), 1))), requires_grad=False).to(self.device)

        predict_real = netD(real)
        loss_d_real = self.criterionGAN(predict_real, true)

        predict_fake = netD(fake.detach())
        loss_d_fake = self.criterionGAN(predict_fake, false)

        loss_d = (loss_d_real + loss_d_fake) * 0.5
        loss_d.backward()

        return loss_d

    def backward_g(self):
        valid = Variable(Tensor(np.ones((self.real_A.size(0), 1))), requires_grad=False).to(self.device)
        self.loss_genA = self.criterionGAN(self.DisA(self.fake_B), valid)
        self.loss_genB = self.criterionGAN(self.DisB(self.fake_A), valid)

        # Forward cycle loss
        self.loss_wasserstein_A = self.criterionWasserstein(self.new_A, self.real_A)
        # Backward cycle loss
        self.loss_wasserstein_B = self.criterionWasserstein(self.new_B, self.real_B)

        if self.is_semi_supervised:
            self.supervised_A = self.criterionSupervised(self.fake_B[:2,:,:,:], self.real_B[:2,:,:,:])
            self.supervised_B = self.criterionSupervised(self.fake_A[:2,:,:,:], self.real_A[:2,:,:,:])

        # combined loss
        self.loss_G = (self.loss_genA + self.loss_genB + self.loss_wasserstein_A + self.loss_wasserstein_B)

        if self.is_semi_supervised:
            self.loss_G += self.supervised_A + self.supervised_B

        self.loss_G.backward()

    def train(self):
        # forward
        self.forward()

        # GenA and GenB
        self.set_requires_grad([self.DisA, self.DisB], False)
        self.optimizer_g.zero_grad()
        self.backward_g()
        self.optimizer_g.step()

        # DisA and DisB
        self.set_requires_grad([self.DisA, self.DisB], True)
        self.optimizer_d.zero_grad()

        # backward Dis A
        self.loss_disA = self.backward_d(self.DisA, self.real_B, self.fake_B)

        # backward Dis B
        self.loss_disB = self.backward_d(self.DisB, self.real_A, self.fake_A)

        self.optimizer_d.step()

    def test(self):
        with torch.no_grad():
            self.forward()
            self.test_A = self.pixelLoss(self.fake_B, self.real_B)
            self.test_B = self.pixelLoss(self.fake_A, self.real_A)


    def save_progress(self, path, epoch, iteration, save_epoch=False):
        path +=  self.architecture 
            
        img_sample = torch.cat((self.real_A.data, self.fake_A.data, self.real_B.data, self.fake_B.data), -2)
        save_image(img_sample, path + "{}_{}.png".format(epoch, iteration), nrow=4, normalize=True)

        nets = {self.GenA:self.gen_a_file,
                self.GenB:self.gen_b_file,
                self.DisA:self.dis_a_file,
                self.DisB:self.dis_b_file}

        for net, file in nets.items():
            if save_epoch:
                file = "{}_{}".format(file, epoch)
            if torch.cuda.is_available():
                torch.save(net.module.state_dict(), file)
                net.to(self.device)
            else:
                torch.save(net.cpu().state_dict(), file)

        self.epoch_tracker.write(epoch, iteration)

    def save_image(self, path, name):
        save_image(self.real_A.data, path + "{}_fakeA.png".format(name), normalize=True)
        save_image(self.real_B.data, path + "{}_fakeB.png".format(name), normalize=True)

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    @staticmethod
    def init_net(net, file=None):
        gpu_ids = list(range(torch.cuda.device_count()))

        if file is not None:
            net.load_state_dict(torch.load(file))
        else:
            net.apply(weights_init_normal)

        if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())
            net.to(gpu_ids[0])
            net = torch.nn.DataParallel(net, gpu_ids)

        return net


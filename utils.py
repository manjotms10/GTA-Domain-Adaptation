import os

import torch
from torch.autograd import Variable
from torch.utils import data
from torchvision.utils import save_image

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def weights_init_normal(m):

    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.01)
        torch.nn.init.constant_(m.bias.data, 0.0)


def get_opts():
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

    return opt


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def sample_images(batches_done, generator, number):
    x, y = next(data.data_generator())
    real_A = Variable(x.type(Tensor))
    real_B = Variable(y.type(Tensor))
    fake_B = generator(real_A)
    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    save_image(img_sample, 'saved_images/%s.png' % (number), nrow=5, normalize=True)

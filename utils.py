import os

import torch
from torch.autograd import Variable
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
    opt = {
        "n_epochs": 100,
        "dataset_name": 'GTA',
        "batch_size": 64,
        "lr": 0.0002,
        "b1": 0.5,
        "b2": 0.99,
        "decay_epoch": 100,
        "n_cpu": 4,
        "img_height": 256,
        "img_width": 256,
        "channels": 3,
        "sample_interval": 25,
        "checkpoint_interval": 1,
        "load_model": True
    }

    return opt


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def sample_images(data, batches_done, generator, number):
    x, y = next(data.data_generator())
    real_A = Variable(x.type(Tensor))
    real_B = Variable(y.type(Tensor))
    fake_B = generator(real_A)
    img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    save_image(img_sample, 'saved_images/%s.png' % (number), nrow=5, normalize=True)
    return x, y

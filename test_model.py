from math import ceil

import torch
from torch import Tensor
from torch.autograd import Variable

from cycle_gan import CycleGAN
from data_loader import DataLoader
from logger import logger
from utils import ensure_dir, get_opts

project_root = "./"
data_root = "./gta/"
models_prefix = project_root + "saved_models/"
images_prefix = project_root + "saved_test_images/"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def test_cycle_gan():
    opt = get_opts()

    ensure_dir(models_prefix)
    ensure_dir(images_prefix)

    cycle_gan = CycleGAN(device, models_prefix, opt["lr"], opt["b1"], train=False)
    data = DataLoader(data_root=data_root,
                      image_size=(opt['img_height'], opt['img_width']),
                      batch_size=1, train=False)

    total_images = len(data.names)
    print("Total Testing Images", total_images)

    loss_A = 0.0
    loss_B = 0.0

    for i in range(total_images):

        print(i, "/", total_images)

        y, x = next(data.data_generator(i))
        name = data.names[i]

        real_A = Variable(x.type(Tensor))
        real_B = Variable(y.type(Tensor))

        cycle_gan.set_input(real_A, real_B)
        cycle_gan.test()
        cycle_gan.save_image(images_prefix, name)
        loss_A += cycle_gan.test_A
        loss_B += cycle_gan.test_B

    info = "Average Loss A:{} B :{}".format(loss_A/(1.0 * total_images), loss_B/(1.0 * total_images))
    print(info)
    logger.info(info)


if __name__ == "__main__":
    test_cycle_gan()

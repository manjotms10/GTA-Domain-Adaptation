from math import ceil

import torch
from torch import Tensor
from torch.autograd import Variable

from dual_gans import DualGANs
from data_loader import DataLoader
from logger import logger
from utils import ensure_dir, get_opts

project_root = "./"
data_root = "./gta/images/"
models_prefix = project_root + "saved_models/"
images_prefix = project_root + "saved_images/"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def train_dual_gans(data_root, semi_supervised=False):
    opt = get_opts()

    ensure_dir(models_prefix)
    ensure_dir(images_prefix)

    dual_gans = DualGANs(device, models_prefix, opt["lr"], opt["alpha"],
                         train=True, semi_supervised=semi_supervised)
    data = DataLoader(data_root=data_root,
                      image_size=(opt['img_height'], opt['img_width']),
                      batch_size=opt['batch_size'])

    total_images = len(data.names)
    print("Total Training Images", total_images)

    total_batches = int(ceil(total_images / opt['batch_size']))

    for epoch in range(dual_gans.epoch_tracker.epoch, opt['n_epochs']):
        for iteration in range(total_batches):

            if (epoch == dual_gans.epoch_tracker.epoch and
                        iteration < dual_gans.epoch_tracker.iter):
                continue

            x, y = next(data.data_generator(iteration))

            real_A = Variable(x.type(Tensor))
            real_B = Variable(y.type(Tensor))

            dual_gans.set_input(real_A, real_B)
            dual_gans.train()

            message = (
                "\r[Epoch {}/{}] [Batch {}/{}] [DA:{}, DB:{}] [GA:{}, GB:{}, wassersteinA:{}, wassersteinB:{}, G:{}]"
                    .format(epoch, opt["n_epochs"], iteration, total_batches,
                            dual_gans.loss_disA.item(),
                            dual_gans.loss_disB.item(),
                            dual_gans.loss_genA.item(),
                            dual_gans.loss_genB.item(),
                            dual_gans.loss_wasserstein_A.item(),
                            dual_gans.loss_wasserstein_B.item(),
                            dual_gans.loss_G))
            print(message)
            logger.info(message)

            if iteration % opt['sample_interval'] == 0:
                dual_gans.save_progress(images_prefix, epoch, iteration)
        dual_gans.save_progress(images_prefix, epoch, total_batches, save_epoch=True)


if __name__ == "__main__":
    train_dual_gans(data_root)

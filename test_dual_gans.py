import torch
from torch import Tensor
from torch.autograd import Variable

from dual_gans import DualGANs
from data_loader import DataLoader
from logger import logger
from utils import ensure_dir, get_opts

project_root = "./"
data_root = "./gta/images/"
models_prefix = project_root + "saved_models/test_"
images_prefix = project_root + "saved_images/"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def test_dual_gans(data_root, semi_supervised=True):
    opt = get_opts()

    ensure_dir(models_prefix)
    ensure_dir(images_prefix)
    
    dual_gans = DualGANs(device, models_prefix, opt["lr"], opt["alpha"], train=False, 
                         semi_supervised=semi_supervised)
    data = DataLoader(data_root=data_root,
                      image_size=(opt['img_height'], opt['img_width']),
                      batch_size=opt["test_batch_size"], train=False)

    total_images = len(data.names)
    print("Total Testing Images", total_images)

    loss_A = 0.0
    loss_B = 0.0
    name_loss_A = []
    name_loss_B = []

    for i in range(total_images//opt["test_batch_size"]):
        print(i, "/", total_images//opt["test_batch_size"])
        x, y = next(data.data_generator(i))
        name = data.names[i]
        real_A = Variable(x.type(Tensor))
        real_B = Variable(y.type(Tensor))

        dual_gans.set_input(real_A, real_B)
        dual_gans.test()
        dual_gans.save_image(images_prefix, name)
        loss_A += dual_gans.test_A
        loss_B += dual_gans.test_B
        name_loss_A.append((dual_gans.test_A, name))
        name_loss_B.append((dual_gans.test_B, name))

    info = "Average Loss A:{} B :{}".format(loss_A/(1.0 * total_images), loss_B/(1.0 * total_images))
    print(info)
    logger.info(info)
    name_loss_A = sorted(name_loss_A)
    name_loss_B = sorted(name_loss_B)
    print("top 10 images")
    print(name_loss_A[:10])
    print(name_loss_B[:10])


if __name__ == "__main__":
    test_dual_gans(data_root)

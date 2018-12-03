import torch

from utils import EpochTracker, weights_init_normal


class GAN:

    def __init__(self, device, file_prefix):
        self.device = device
        self.file_prefix = file_prefix

        self.epoch_tracker = EpochTracker(file_prefix + "epoch.txt")

    def set_input(self, real_A, real_B):
        self.real_A = real_A.to(self.device)
        self.real_B = real_B.to(self.device)

    def forward(self):
        pass

    def train(self):
        pass

    def test(self):
        with torch.no_grad():
            self.forward()

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


import os
import torchvision

from tqdm import tqdm
from torch import Tensor


class AverageMeter(object):
    """Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger:
    def __init__(self, path: str, config: str) -> None:
        self.path = path
        if not config is None:
            with open(os.path.join(self.path, "log.txt"), "a") as f:
                f.write(config + "\n")

    def info(self, msg: str, print_msg: bool = False) -> None:
        if print_msg:
            print(msg)
        with open(os.path.join(self.path, "log.txt"), "a") as f:
            f.write(msg + "\n")

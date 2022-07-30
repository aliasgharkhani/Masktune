import torch

def select_device(use_cuda=True) -> str:
    if torch.cuda.is_available() and use_cuda:
        return "cuda"
    else:
        return "cpu"
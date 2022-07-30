import torch


def calculate_accuracy(labels: torch.tensor, outputs: torch.tensor):
    equals = labels.eq(outputs)
    return torch.sum(equals).item() / len(labels)

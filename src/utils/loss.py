import torch
import torch.nn.functional as F


def sd_loss(yhat, y, sp):
    per_sample_losses = torch.log(
        1.0 + torch.exp(-yhat[:, 0] * (2.0 * y - 1.0)))
    actual_loss = per_sample_losses.mean()
    actual_loss += sp * ((yhat[torch.where(y == 1)] - 2.5) ** 2).mean()
    actual_loss += sp * ((yhat[torch.where(y == 0)] - 0.44) ** 2).mean()
    return actual_loss


def gce(outputs, targets, q=0.5):
    output_probabilities = F.softmax(outputs, dim=1)
    return torch.mean((1-torch.pow(torch.gather(output_probabilities, 1, torch.unsqueeze(targets, dim=-1)), q))/q)

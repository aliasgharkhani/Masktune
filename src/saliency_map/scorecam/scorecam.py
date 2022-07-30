import torch
import torch.nn.functional as F
from src.saliency_map.scorecam.basecam import BaseCAM

import numpy as np

def PCA(data, dims_rescaled_data=2):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    import numpy as NP
    from scipy import linalg as LA
    m, n = data.shape
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = NP.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric, 
    # the performance gain is substantial
    evals, evecs = LA.eigh(R)
    # sort eigenvalue in decreasing order
    idx = NP.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return NP.dot(evecs.T, data.T).T, evals, evecs


class ScoreCAM(BaseCAM):

    """
    ScoreCAM, inherit from BaseCAM

    """

    def __init__(self, model, target_layer, use_cuda):
        super().__init__(model, target_layer, use_cuda)
    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()
        if self.use_cuda:
            input = input.cuda()
        # predication on raw input
        logit = self.model(input)

        if class_idx is None:
            # predicted_class = logit.max(1)[-1]
            predicted_class = torch.unsqueeze(logit.max(1)[-1], dim=-1)
            # score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            predicted_class = torch.LongTensor([class_idx])
            # score = logit[:, class_idx].squeeze()

        # logit = F.softmax(logit)

        if self.use_cuda:
            predicted_class = predicted_class.cuda()
        #   score = score.cuda()
        #   logit = logit.cuda()

        self.model.zero_grad()
        # score.backward(retain_graph=retain_graph)
        gpu_ids = sorted(list(self.activations.keys()))
        activations = []
        for gpu_id in gpu_ids:
            activations.append(self.activations[gpu_id])
        activations = torch.cat(activations)
        # activations = self.activations["value"]
        b, k, u, v = activations.size()

        # binarized_activations = torch.where(activations < 1e-5, 0, 1)
        # sparsity = torch.sum(binarized_activations, dim=(2, 3)) / (u*v)
        # values, indices = torch.sort(sparsity, dim=1)
        # sorted_activations = torch.stack(list(map(lambda x, y: y[x], indices, activations)))
        # activations = sorted_activations
        # b, k, u, v = activations.size()


        # score_saliency_map = torch.zeros((1, 1, h, w))
        score_saliency_map = torch.zeros((b, 1, h, w))

        if self.use_cuda:
            activations = activations.cuda()
            score_saliency_map = score_saliency_map.cuda()

        with torch.no_grad():
            for i in range(k):

                # upsampling
                saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
                saliency_map = F.interpolate(
                    saliency_map, size=(h, w), mode="bilinear", align_corners=False
                )

                # if saliency_map.max() == saliency_map.min():
                # continue

                # normalize to 0-1
                saliency_map_min, saliency_map_max = torch.unsqueeze(
                    torch.unsqueeze(torch.amin(saliency_map, dim=(2, 3)), dim=-1),
                    dim=-1,
                ), torch.unsqueeze(
                    torch.unsqueeze(torch.amax(saliency_map, dim=(2, 3)), dim=-1),
                    dim=-1,
                )
                norm_saliency_map = (saliency_map - saliency_map_min) / (
                    saliency_map_max - saliency_map_min + 1e-8
                )
                # norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

                # how much increase if keeping the highlighted region
                # predication on masked input
                output = self.model(input * norm_saliency_map)
                output = F.softmax(output, dim=1)
                # score = output[0][predicted_class]
                score = torch.unsqueeze(
                    torch.unsqueeze(torch.gather(output, 1, predicted_class), dim=-1),
                    dim=-1,
                )

                # score_saliency_map +=  score * saliency_map
                score_saliency_map += (
                    torch.where((saliency_map_max - saliency_map_min) == 0, 0, 1)
                    * score
                ) * saliency_map

        score_saliency_map = F.relu(score_saliency_map)
        # score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()
        score_saliency_map_min, score_saliency_map_max = torch.unsqueeze(
            torch.unsqueeze(torch.amin(score_saliency_map, dim=(2, 3)), dim=-1), dim=-1
        ), torch.unsqueeze(
            torch.unsqueeze(torch.amax(score_saliency_map, dim=(2, 3)), dim=-1), dim=-1
        )

        # if score_saliency_map_min == score_saliency_map_max:
        #     return None

        score_saliency_map = (score_saliency_map - score_saliency_map_min).div(
            score_saliency_map_max - score_saliency_map_min
        )
        # score_saliency_map = (score_saliency_map - score_saliency_map_min).div(score_saliency_map_max - score_saliency_map_min).data

        # return score_saliency_map
        return torch.squeeze(score_saliency_map).detach().cpu().numpy()
        

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)

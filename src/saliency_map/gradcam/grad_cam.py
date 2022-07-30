import numpy as np
from src.saliency_map.gradcam.base_cam import BaseCAM


class GradCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None, grad_cam_weight='mean'):
        self.grad_cam_weight = grad_cam_weight
        super(
            GradCAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        
        if self.grad_cam_weight == 'mean':
            return np.mean(grads, axis=(2, 3))[:, :, None, None]
        elif self.grad_cam_weight == 'raw':
            grads[grads<0] = 0
            return grads

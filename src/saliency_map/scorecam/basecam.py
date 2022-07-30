"""
Part of code borrows from https://github.com/1Konny/gradcam_plus_plus-pytorch
"""

class BaseCAM(object):
    """Base class for Class activation mapping.

    : Args
        - **model, target_layer** : 

    """

    def __init__(self, model, target_layer, use_cuda):

        self.model = model
        self.target_layer = target_layer
        self.use_cuda = use_cuda
        self.model.eval()
        
        if self.use_cuda:
            self.model.cuda()
        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients["value"] = grad_output[0].cpu().detach()
            
        def forward_hook(module, input, output):
            self.activations[input[0].device.index] = output.cpu().detach()


        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def forward(self, input, class_idx=None, retain_graph=False):
        return None

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)

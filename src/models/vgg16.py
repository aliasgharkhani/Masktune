import torch.nn as nn
import math
import torch

DEPTH_ENTROPY = False
__all__ = [
    'VGG', 'vgg16', 'vgg16_bn'
]

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, input_size = 32, activation='relu'):
        super(VGG, self).__init__()
        self.features = features
        self.act = activation
        if self.act == 'relu':
            self.activ = nn.ReLU
        elif self.act == 'selu':
            self.activ = nn.SELU

        if input_size == 32:
            self.classifier = nn.Sequential(nn.Linear(512,512), self.activ(inplace=True), \
                                        nn.BatchNorm1d(512),nn.Dropout2d(0.5),nn.Linear(512, num_classes))
        elif input_size == 64:
            self.classifier = nn.Sequential(nn.Linear(2048,512), self.activ(inplace=True), \
                                        nn.BatchNorm1d(512),nn.Dropout2d(0.5),nn.Linear(512, num_classes))
        self._initialize_weights()

    def get_grad_cam_target_layer(self):
        return self.features[-3]
    
    def forward(self, x, return_entropy=False, return_l2=False):
        if return_entropy and not return_l2:
            ents = []
            for i in self.features:
                x = i(x)
                # if isinstance(i, torch.nn.modules.ReLU):
                #     ents.append(get_feature_map_entropy(x))
            
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x, torch.tensor(ents, requires_grad=True).type_as(x).mean(), torch.tensor([0], requires_grad=False).type_as(x).mean()

        elif not return_entropy and return_l2:
            l2s = []
            for i in self.features:
                x = i(x)
                # if isinstance(i, torch.nn.modules.ReLU):
                #     l2s.append(get_feature_map_l2(x))
            
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x, torch.tensor([0], requires_grad=False).type_as(x).mean(), torch.tensor(l2s, requires_grad=True).type_as(x).mean()

        elif return_entropy and return_l2:
            ents = []
            l2s = []
            for i in self.features:
                x = i(x)
                # if isinstance(i, torch.nn.modules.ReLU):
                #     l2s.append(get_feature_map_l2(x))
                #     ents.append(get_feature_map_entropy(x))
            
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x, torch.tensor(ents, requires_grad=True).type_as(x).mean(), torch.tensor(l2s, requires_grad=True).type_as(x).mean()
        else:
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False, activation='relu'):
    if activation == 'relu':
        activ = nn.ReLU
    elif activation== 'selu':
        activ = nn.SELU
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif type(v)==int:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, activ(inplace=True), nn.BatchNorm2d(v)]
                # the order is modified to match the model of the baseline that we compare to
            else:
                layers += [conv2d, activ(inplace=True)]
            in_channels = v
        elif type(v)==float:
            layers += [nn.Dropout2d(v)]
    return nn.Sequential(*layers)


cfg = {
    'D': [64,0.3, 64, 'M', 128,0.4, 128, 'M', 256,0.4, 256,0.4, 256, 'M', 512,0.4, 512,0.4, 512, 'M', 512,0.4, 512,0.4, 512, 'M',0.5]
}

def vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D']), **kwargs)
    return model


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(make_layers(cfg['D'], batch_norm=True, activation=kwargs['activation']), **kwargs)
    return model


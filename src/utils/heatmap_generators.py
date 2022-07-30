from src.saliency_map.scorecam.scorecam import ScoreCAM
from src.saliency_map.gradcam.grad_cam import GradCAM
from pytorch_grad_cam import AblationCAM, EigenCAM, XGradCAM, GradCAMPlusPlus, FullGrad, LayerCAM
from src.utils import SparsityHeatmap

def get_heatmap_generator(generator_name, model, target_layer, use_cuda=False, **kwargs):
    if generator_name == 'grad_cam':
        heat_map_generator = GradCAM(
            model=model,
            target_layers=[target_layer],
            use_cuda=use_cuda,
            grad_cam_weight=kwargs['grad_cam_weight']
        )
    elif generator_name == 'score_cam':
        heat_map_generator = ScoreCAM(
            model=model,
            target_layer=target_layer,
            use_cuda=use_cuda,
        )
    elif generator_name == 'sparsity':
        heat_map_generator = SparsityHeatmap(
            model=model,
            target_layer=target_layer
        )
    elif generator_name == 'xgrad_cam':
        heat_map_generator = XGradCAM(
            model=model,
            target_layers=[target_layer],
            use_cuda=use_cuda,
        )
    elif generator_name == 'ablation_cam':
        heat_map_generator = AblationCAM(
            model=model,
            target_layers=[target_layer],
            use_cuda=use_cuda,
        )
    elif generator_name == 'eigen_cam':
        heat_map_generator = EigenCAM(
            model=model,
            target_layers=[target_layer],
            use_cuda=use_cuda,
        )
    elif generator_name == 'gradcam_plusplus':
        heat_map_generator = GradCAMPlusPlus(
            model=model,
            target_layers=[target_layer],
            use_cuda=use_cuda,
        )
    elif generator_name == 'full_grad':
        heat_map_generator = FullGrad(
            model=model,
            target_layers=[target_layer],
            use_cuda=use_cuda,
        )
    elif generator_name == 'layer_cam':
        heat_map_generator = LayerCAM(
            model=model,
            target_layers=[target_layer],
            use_cuda=use_cuda,
        )
    return heat_map_generator
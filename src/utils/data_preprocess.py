import torch

from torch import Tensor
from typing import List, Tuple

import numpy as np

def filter_data_by_label(data, targets, class_labels_to_filter):
    """
    extract indices of data that have labels that exist in the desired_class_labels
    """
    filtered_target_idx = torch.cat(
        [torch.where(targets == label)[0] for label in class_labels_to_filter]
    )
    return data[filtered_target_idx], targets[filtered_target_idx]


def group_labels(targets, old_to_new_label_mapping):
    """
    assign new labels to data based on the label_grouping
    """
    new_labels = list(old_to_new_label_mapping.keys())
    old_label_groupings = list(old_to_new_label_mapping.values())

    for i, target in enumerate(targets):
        for idx, old_label_grouping in enumerate(old_label_groupings):
            if target in old_label_grouping:
                target = new_labels[idx]

        targets[i] = torch.tensor(int(target))
    return targets


def add_color_bias_to_images(
    class_number: int,
    data: Tensor,
    targets: Tensor,
    bias_conflicting_data_ratio: float,
    bias_colors: List[list]=None,
    bias_type: str="background",
    **kwargs,
) -> Tuple[Tensor, List[list]]:    
    colors = []
    if class_number == 2:
        bias_colors = [[255, 0, 0], [255, 0, 0]]
    for i in range(class_number):
        data_number_to_add_bias = round(len(data[torch.where(targets==i)[0]])*(1-bias_conflicting_data_ratio))
        if class_number == 2 and i == 1:
            data_number_to_add_bias = round(len(data[torch.where(targets==i)[0]])*bias_conflicting_data_ratio)
        target_i_data = data[torch.where(targets==i)[0][:data_number_to_add_bias]]
        if bias_colors is None:
            color = torch.randint(0, 256, (3,), dtype=torch.uint8)
            colors.append(color.numpy())
        else:
            color = torch.tensor(bias_colors[i], dtype=torch.uint8)
            colors.append(bias_colors[i])
        for j in range(3):
            if bias_type == "background":
                target_i_data[:, :, :, j] = torch.where(target_i_data[:, :, :, j]==0, color[j], target_i_data[:, :, :, j])
            elif bias_type == "foreground":
                target_i_data[:, :, :, j] = torch.where(target_i_data[:, :, :, j]>0, color[j], target_i_data[:, :, :, j])
            elif bias_type == "square":
                target_i_data[:, :kwargs["square_size"], :kwargs["square_size"], j] = torch.ones((len(target_i_data), kwargs["square_size"], kwargs["square_size"])) * color[j]
                if kwargs["square_number"] == 2:
                    target_i_data[:, :kwargs["square_size"], -kwargs["square_size"]:, j] = torch.ones((len(target_i_data), kwargs["square_size"], kwargs["square_size"])) * color[2-j]
        data[torch.where(targets==i)[0][:data_number_to_add_bias]] = target_i_data
    return data, np.array(colors)

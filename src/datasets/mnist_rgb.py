from torchvision.datasets import MNIST
from src.utils import add_color_bias_to_images, filter_data_by_label, group_labels
from PIL import Image
from tqdm import tqdm
from glob import glob

import os
import torch

import numpy as np

from torchvision import transforms

from typing import Tuple

class BiasedMNIST(MNIST):
    def __init__(
        self,
        class_labels_to_filter=[i for i in range(0, 10)],
        new_to_old_label_mapping={0: [0, 1, 2, 3, 4], 1: [5, 6, 7, 8, 9]},
        bias_conflicting_data_ratio=0.1,
        bias_type="square",
        square_size=4,
        bias_colors=None,
        square_number=1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.train = kwargs["train"]
        self.return_masked = False
        if kwargs["train"]:
            self.split = "train"
        else:
            self.split = "test"
        img_data_dir = os.path.join(
            kwargs["root"],
            "BiasedMNIST",
            "images",
            self.split,
            bias_type,
        )
        if bias_type == "square":
            img_data_dir = os.path.join(img_data_dir, square_number)
        img_data_dir = os.path.join(img_data_dir, f"{round((1-bias_conflicting_data_ratio)*100)}")
        bias_colors_file_path = os.path.join(img_data_dir, "bias_colors.npy")
        if not (os.path.isdir(img_data_dir) and len(os.listdir(img_data_dir)) > 0):
            print(
                f"\n\nstart creating and saving {self.split} dataset of BiasedMnist\n\n"
            )
            os.makedirs(img_data_dir, exist_ok=True)
            self.data, self.targets = filter_data_by_label(
                self.data, self.targets, class_labels_to_filter
            )
            self.targets = group_labels(self.targets, new_to_old_label_mapping)
            self.data = torch.unsqueeze(self.data, dim=-1).repeat((1, 1, 1, 3))
            # permute_indices = torch.randperm(len(self.data))
            # self.data = self.data[permute_indices]
            # self.targets = self.targets[permute_indices]
            if bias_type != "none":
                self.data, self.bias_colors = add_color_bias_to_images(
                    len(new_to_old_label_mapping),
                    self.data.clone(),
                    self.targets,
                    bias_conflicting_data_ratio,
                    bias_colors=bias_colors,
                    bias_type=bias_type,
                    square_size=square_size,
                    square_number=square_number,
                )
                np.save(bias_colors_file_path, self.bias_colors)
            for target in list(new_to_old_label_mapping.keys()):
                os.makedirs(os.path.join(img_data_dir, str(target)), exist_ok=True)
            for id, (data, target) in enumerate(zip(self.data, self.targets)):
                Image.fromarray(data.numpy().astype(np.uint8)).save(
                    os.path.join(img_data_dir, str(target.item()), f"{id}.png")
                )
            self.data = []
            self.targets = []
            print(
                f"\n\nfinished creating and saving {self.split} dataset of BiasedMnist\n\n"
            )
        elif bias_conflicting_data_ratio < 1.0:
            if bias_colors is None:
                self.bias_colors = np.load(bias_colors_file_path)
            else:
                self.bias_colors = bias_colors
                np.save(bias_colors_file_path, self.bias_colors)

        self.update_data(img_data_dir)
            
    def update_data(self, data_file_directory, masked_data_file_directory=None):
        self.data_path = []
        self.data = []
        self.targets = []
        self.masked_data_path = []
        
        data_classes = sorted(os.listdir(data_file_directory))
        print("-" * 10, f"indexing {self.split} data", "-" * 10)
        for data_class in tqdm(data_classes):
            try:
                target = int(data_class)
            except:
                continue
            class_image_file_paths = glob(
                os.path.join(data_file_directory, data_class, "*")
            )
            for image_file_path in class_image_file_paths:
                self.data.append(Image.open(image_file_path))
            self.data_path += class_image_file_paths
            if masked_data_file_directory is not None:
                self.return_masked = True
                masked_class_image_file_paths = sorted(glob(
                    os.path.join(masked_data_file_directory, data_class, '*')))
                self.masked_data_path += masked_class_image_file_paths
            self.targets += [target] * len(class_image_file_paths)


    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, img_file_path, target) where target is index of the target class.
        """
        img, img_file_path, target = self.data[index], self.data_path[index], self.targets[index]
        if self.transform is not None:
            transform = transforms.Compose([transforms.ToTensor()])
            img = transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_masked:
            masked_img_file_path = self.masked_data_path[index]
            masked_img = Image.open(masked_img_file_path)
            if self.transform is not None:
                masked_img = self.transform(masked_img)
            return img, img_file_path, target, masked_img
        return img, img_file_path, target

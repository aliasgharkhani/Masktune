from torchvision.datasets import CIFAR10
from PIL import Image
from tqdm import tqdm
from glob import glob

import os

import numpy as np


class CIFAR10Dataset(CIFAR10):
    def __init__(
            self,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.return_masked = False
        head, last_directory = os.path.split(kwargs['root'])
        img_data_dir = os.path.join(
            head, 'images', 'train' if self.train else 'test')
        targets = np.unique(self.targets)
        if not (os.path.isdir(img_data_dir) and len(os.listdir(img_data_dir)) > 0):
            dataset_to_create = 'train' if self.train else 'test'
            print(
                f"\n\nstart creating and saving {dataset_to_create} dataset of Cifar10\n\n")
            os.makedirs(img_data_dir, exist_ok=True)
            for target in targets:
                os.makedirs(os.path.join(
                    img_data_dir, str(target)), exist_ok=True)
            for id, (data, target) in enumerate(zip(self.data, self.targets)):
                Image.fromarray(data).save(os.path.join(
                    img_data_dir, str(target), f'{id}.png'))
            self.data = []
            self.targets = []
            print(
                f"\n\nfinished creating and saving {dataset_to_create} dataset of Cifar10\n\n")
        self.update_data(img_data_dir)

    def __len__(self):
        return len(self.data_path)

    def update_data(self, data_file_directory, masked_data_file_path=None):
        self.data_path = []
        self.masked_data_path = []
        self.targets = []
        data_classes = sorted(os.listdir(data_file_directory))
        print("-"*10, f"indexing {'train' if self.train else 'test'} data", "-"*10)
        for data_class in tqdm(data_classes):
            try:
                target = int(data_class)
            except:
                continue
            class_image_file_paths = glob(
                os.path.join(data_file_directory, data_class, '*'))
            self.data_path += class_image_file_paths
            if masked_data_file_path is not None:
                self.return_masked = True
                masked_class_image_file_paths = sorted(glob(
                    os.path.join(masked_data_file_path, data_class, '*')))
                self.masked_data_path += masked_class_image_file_paths
            self.targets += [target] * len(class_image_file_paths)


    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, img_file_path, target) where target is index of the target class.
        """
        img_file_path, target = self.data_path[index], self.targets[index]
        img = Image.open(img_file_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_masked:
            masked_img_file_path = self.masked_data_path[index]
            masked_img = Image.open(masked_img_file_path)
            if self.transform is not None:
                masked_img = self.transform(masked_img)
            return img, img_file_path, target, masked_img
        return img, img_file_path, target

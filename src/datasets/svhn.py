from torchvision.datasets import SVHN
from PIL import Image
from tqdm import tqdm
from glob import glob

import os

import numpy as np


class SVHNDataset(SVHN):
    def __init__(
            self,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.split = kwargs['split']
        self.return_masked = False
        head, tail = os.path.split(kwargs['root'])
        img_data_dir = os.path.join(
            head, 'images', kwargs['split'])
        labels = np.unique(self.labels)
        if not (os.path.isdir(img_data_dir) and len(os.listdir(img_data_dir)) > 0):
            print(
                f"\n\nstart creating and saving {kwargs['split']} dataset of SVHN\n\n")
            os.makedirs(img_data_dir, exist_ok=True)
            for label in labels:
                os.makedirs(os.path.join(
                    img_data_dir, str(label)), exist_ok=True)
            for id, (data, label) in enumerate(zip(self.data, self.labels)):
                Image.fromarray(np.rollaxis(data, 0, 3).astype(np.uint8)).save(
                    os.path.join(img_data_dir, str(label), f'{id}.png'))
            self.data = []
            self.labels = []
            print(
                f"\n\finished creating and saving {kwargs['split']} dataset of SVHN\n\n")
        self.update_data(img_data_dir)

    def update_data(self, data_file_directory, masked_data_file_path=None):
        self.data_path = []
        self.masked_data_path = []
        self.labels = []
        data_classes = sorted(os.listdir(data_file_directory))
        print("-"*10, f"indexing {self.split} data", "-"*10)
        for data_class in tqdm(data_classes):
            try:
                label = int(data_class)
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
            self.labels += [label] * len(class_image_file_paths)


    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, img_file_path, label) where label is index of the label class.
        """
        img_file_path, label = self.data_path[index], self.labels[index]
        img = Image.open(img_file_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        if self.return_masked:
            masked_img_file_path = self.masked_data_path[index]
            masked_img = Image.open(masked_img_file_path)
            if self.transform is not None:
                masked_img = self.transform(masked_img)
            return img, img_file_path, label, masked_img
        return img, img_file_path, label

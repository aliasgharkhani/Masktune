from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from glob import glob

import os
import shutil

import pandas as pd
import numpy as np


data_split = {
    0: 'train',
    1: 'val',
    2: 'test'
}


class CelebADataset(Dataset):
    def __init__(
            self,
            root,
            raw_data_path,
            split='train',
            transform=None,
            target_transform=None,
            attr_name='Blond_Hair',
            confounder_names="Male",
            return_confounder=False,
    ) -> None:
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.return_confounder = return_confounder
        self.return_masked = False
        self.data_path = []
        self.masked_data_path = []
        self.labels = []
        self.confounders = {}
        img_data_dir = os.path.join(
            root, 'images', split)
        if not (os.path.isdir(img_data_dir) and len(os.listdir(img_data_dir)) > 0):
            print(
                f"\n\nstart creating and saving {split} dataset of CelebA\n\n")
            attrs_df = pd.read_csv(os.path.join(
                raw_data_path, 'list_attr_celeba.csv'))
            image_ids = attrs_df['image_id'].values
            attrs_df = attrs_df.drop(labels='image_id', axis='columns')
            attr_names = attrs_df.columns.copy()
            attrs_df = attrs_df.values
            attrs_df[attrs_df == -1] = 0
            target_idx = attr_names.get_loc(attr_name)
            labels = attrs_df[:, target_idx]
            confounder_idx = attr_names.get_loc(confounder_names)
            confounders = attrs_df[:, confounder_idx]
            partition_df = pd.read_csv(os.path.join(
                raw_data_path, 'list_eval_partition.csv'))
            partitions = partition_df['partition']
            os.makedirs(img_data_dir, exist_ok=True)
            for label in np.unique(labels):
                os.makedirs(os.path.join(
                    img_data_dir, str(label)), exist_ok=True)
            for image_id, label, confounder, partition in tqdm(zip(image_ids, labels, confounders, partitions), total=len(image_ids)):
                if data_split[partition] == split:
                    shutil.copy(os.path.join(raw_data_path, 'img_align_celeba', image_id), os.path.join(
                        img_data_dir, str(label), image_id))
                    self.data_path.append(os.path.join(
                        img_data_dir, str(label), image_id))
                    self.labels.append(label)
                    self.confounders[image_id] = confounder
            print(
                f"\n\nfinished creating and saving {split} dataset of CelebA\n\n")
            return
        attrs_df = pd.read_csv(os.path.join(
            raw_data_path, 'list_attr_celeba.csv'))
        image_ids = attrs_df['image_id'].values
        attrs_df = attrs_df.drop(labels='image_id', axis='columns')
        attr_names = attrs_df.columns.copy()
        attrs_df = attrs_df.values
        attrs_df[attrs_df == -1] = 0
        confounder_idx = attr_names.get_loc(confounder_names)
        confounders = attrs_df[:, confounder_idx]
        partition_df = pd.read_csv(os.path.join(
            raw_data_path, 'list_eval_partition.csv'))
        partitions = partition_df['partition']
        for image_id, confounder, partition in tqdm(zip(image_ids, confounders, partitions), total=len(image_ids)):
            if data_split[partition] == split:
                self.confounders[image_id] = int(confounder)

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
            if self.return_confounder:
                return img, img_file_path, label, masked_img, self.confounders[img_file_path.split('/')[-1]]
            return img, img_file_path, label, masked_img
        if self.return_confounder:
            return img, img_file_path, label, self.confounders[img_file_path.split('/')[-1]]
        return img, img_file_path, label

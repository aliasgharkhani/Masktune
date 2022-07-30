from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from glob import glob

import os


class CatsVsDogsDataset(Dataset):
    def __init__(
            self,
            raw_data_path,
            root,
            train=True,
            transform=None,
            target_transform=None,
    ) -> None:
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.return_masked = False
        img_data_dir = os.path.join(
            root, 'images', 'train' if train else 'test')
        if not (os.path.isdir(img_data_dir) and len(os.listdir(img_data_dir)) > 0):
            train_img_data_dir = os.path.join(root, 'images', 'train')
            test_img_data_dir = os.path.join(root, 'images', 'test')
            dataset_to_create = 'train' if train else 'test'
            print(
                f"\n\nstart creating and saving {dataset_to_create} dataset of CatsVsDogs\n\n")
            os.makedirs(train_img_data_dir, exist_ok=True)
            os.makedirs(test_img_data_dir, exist_ok=True)
            os.makedirs(os.path.join(train_img_data_dir, '0'), exist_ok=True)
            os.makedirs(os.path.join(train_img_data_dir, '1'), exist_ok=True)
            os.makedirs(os.path.join(test_img_data_dir, '0'), exist_ok=True)
            os.makedirs(os.path.join(test_img_data_dir, '1'), exist_ok=True)
            with open(os.path.join(root, 'raw', 'cat_idx.txt')) as cat_test_idx_file:
                cat_test_indices = cat_test_idx_file.readlines()
            with open(os.path.join(root, 'raw', 'dog_idx.txt')) as dog_test_idx_file:
                dog_test_indices = dog_test_idx_file.readlines()
            for cat_test_id, dog_test_id in zip(cat_test_indices, dog_test_indices):
                cat_test_id, cat_suffix = cat_test_id.strip().split('.')
                dog_test_id, dog_suffix = dog_test_id.strip().split('.')
                cat_test_file_path = os.path.join(
                    raw_data_path, f'cat.{cat_test_id}.{cat_suffix}')
                cat_test_new_file_path = os.path.join(
                    test_img_data_dir, '0', f'{cat_test_id}_0.{cat_suffix}')
                os.replace(cat_test_file_path, cat_test_new_file_path)
                dog_test_file_path = os.path.join(
                    raw_data_path, f'dog.{dog_test_id}.{dog_suffix}')
                dog_test_new_file_path = os.path.join(
                    test_img_data_dir, '1', f'{dog_test_id}_1.{dog_suffix}')
                os.replace(dog_test_file_path, dog_test_new_file_path)

            train_cat_dog_files = os.listdir(raw_data_path)
            for train_file in train_cat_dog_files:
                train_file_label, train_file_id, train_file_suffix = train_file.split(
                    '.')
                label = 0 if train_file_label == 'cat' else 1
                train_file_path = os.path.join(raw_data_path, train_file)
                train_file_new_path = os.path.join(
                    train_img_data_dir, str(label), f'{train_file_id}_{label}.{train_file_suffix}')
                os.replace(train_file_path, train_file_new_path)
            print(
                f"\n\nfinished creating and saving {dataset_to_create} dataset of CatVsDogs\n\n")

        self.update_data(img_data_dir)

    def __len__(self):
        return len(self.data_path)

    def update_data(self, data_file_directory, masked_data_file_path=None):
        self.data_path = []
        self.masked_data_path = []
        self.labels = []
        data_classes = sorted(os.listdir(data_file_directory))
        print("-"*10, f"indexing {'train' if self.train else 'test'} data", "-"*10)
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

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, img_file_path, label) where label is index of the label class.
        """
        img_file_path, label = self.data_path[index], self.labels[index]
        img = Image.open(img_file_path).resize(size=(64, 64))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        if self.return_masked:
            masked_img_file_path = self.masked_data_path[index]
            masked_img = Image.open(masked_img_file_path).resize(size=(64, 64))
            if self.transform is not None:
                masked_img = self.transform(masked_img)
            return img, img_file_path, label, masked_img
        return img, img_file_path, label

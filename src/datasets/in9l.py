from torch.utils.data import Dataset
from PIL import Image
from glob import glob
from tqdm import tqdm

import os

class IN9L(Dataset):
    def __init__(
            self,
            root,
            split,
            transform=None,
    ) -> None:
        super().__init__()
        self.split = split
        self.data_path = []
        self.masked_data_path = []
        self.targets = []
        self.return_masked = False
        self.transform = transform
        if split == 'train' or split == 'val':
            raw_img_data_dir = os.path.join(root, split)
        else:
            raw_img_data_dir = os.path.join(
                root, 'test', split, 'val')

        self.update_data(raw_img_data_dir)

    def __len__(self):
        return len(self.data_path)

    def update_data(self, data_file_directory, masked_data_file_path=None):
        self.data_path = []
        self.masked_data_path = []
        self.targets = []
        data_class_names = sorted(os.listdir(data_file_directory))
        print("-"*10, f"indexing {self.split} data", "-"*10)
        for data_class_name in tqdm(data_class_names):
            try:
                target = int(data_class_name.split('_')[0])
            except:
                continue
            class_image_file_paths = glob(
                os.path.join(data_file_directory, data_class_name, '*'))
            self.data_path += class_image_file_paths
            if masked_data_file_path is not None:
                self.return_masked = True
                masked_class_image_file_paths = sorted(glob(
                    os.path.join(masked_data_file_path, str(target), '*')))
                self.masked_data_path += masked_class_image_file_paths
            self.targets += [target] * len(class_image_file_paths)


    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, img_file_path, target) where target is index of the target class.
        """
        target = self.targets[index]
        img = Image.open(self.data_path[index])
        if self.transform is not None:
            img = self.transform(img)
        if self.return_masked:
            masked_img_file_path = self.masked_data_path[index]
            masked_img = Image.open(masked_img_file_path)
            if self.transform is not None:
                masked_img = self.transform(masked_img)
            return img, self.data_path[index], target, masked_img
        return img, self.data_path[index], target

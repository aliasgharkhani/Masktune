from PIL import Image
from tqdm import tqdm
from glob import glob

from torch.utils.data import Dataset
import os

class Imagenet1(Dataset):
    def __init__(
            self,
            data_dir,
            class_mapping_file_dir,
            transform=None,
            target_transform=None,
    ) -> None:
        self.split = data_dir.split('/')[-1]
        self.return_masked = False
        self.transform = transform
        self.target_transform = target_transform
        self.class_mapping_dict = {}
        with open(class_mapping_file_dir) as mapping_file:
            for row in mapping_file:
                imagenet_class_name, class_id, _, _ = row.split('|')
                self.class_mapping_dict[imagenet_class_name.strip()] = int(class_id.strip())
        # with open(class_mapping_file_dir) as mapping_file:
        #     for row in mapping_file:
        #         imagenet_class_name, _, human_readable_label = row.split()
        #         self.class_mapping_dict[imagenet_class_name] = human_readable_label.replace('_', ' ')
        # self.human_readable_label_to_id = {}
        # with open(pytorch_class_mapping) as mapping_file:
        #     for row in mapping_file:
        #         human_readable_label, id = row.split(':')
        #         for name in human_readable_label.split(','):
        #             self.human_readable_label_to_id[name.strip()] = int(id)
        self.update_data(data_dir)

    def update_data(self, data_file_directory, masked_data_file_path=None):
        self.data_path = []
        self.masked_data_path = []
        self.targets = []
        data_classes = sorted(os.listdir(data_file_directory))
        print("-"*10, f"indexing {self.split} data", "-"*10)
        for data_class in tqdm(data_classes):
            if data_class.startswith('.'):
                continue
            if data_class.isdigit():
                target = int(data_class)
            else:
                target = self.class_mapping_dict[data_class]
            class_image_file_paths = glob(
                os.path.join(data_file_directory, data_class, '*'))
            self.data_path += class_image_file_paths
            if masked_data_file_path is not None:
                self.return_masked = True
                masked_class_image_file_paths = sorted(glob(
                    os.path.join(masked_data_file_path, data_class, '*')))
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
        img_file_path, target = self.data_path[index], self.targets[index]
        img = Image.open(img_file_path).convert('RGB')
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

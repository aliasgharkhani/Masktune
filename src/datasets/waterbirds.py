import csv
import shutil
import os


from torch.utils.data import Dataset
from glob import glob
from PIL import Image
from tqdm import tqdm


data_split = {
    0: 'train',
    1: 'val',
    2: 'test'
}


class WaterbirdsDataset(Dataset):
    def __init__(
            self,
            raw_data_path,
            root,
            split='train',
            transform=None,
            target_transform=None,
            return_places=False,
    ) -> None:
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.return_places = return_places
        self.return_masked = False
        img_data_dir = os.path.join(
            root, 'images', split)
        self.places = {}
        if not (os.path.isdir(img_data_dir) and len(os.listdir(img_data_dir)) > 0):
            print(
                f"\n\nstart creating {split} dataset of Waterbirds\n\n")
            self.data_path = []
            self.masked_data_path = []
            self.targets = []

            os.makedirs(img_data_dir, exist_ok=True)
            with open(os.path.join(raw_data_path, 'metadata.csv')) as meta_file:
                csv_reader = csv.reader(meta_file)
                shutil.copy(os.path.join(raw_data_path, 'metadata.csv'), os.path.join(root, 'metadata.csv'))
                for idx, row in enumerate(csv_reader):
                    if idx == 0:
                        continue
                    img_id,	img_filename, y, split_index, place, place_filename = row
                    if data_split[int(split_index)] == split:
                        os.makedirs(os.path.join(
                            img_data_dir, y), exist_ok=True)
                        shutil.copy(os.path.join(raw_data_path, img_filename), os.path.join(
                            img_data_dir, y, img_filename.split('/')[-1]))
                        self.data_path.append(os.path.join(
                            img_data_dir, y, img_filename.split('/')[-1]))
                        self.targets.append(int(y))
                        self.places[img_filename.split('/')[-1]] = place
            print(
                f"\n\nfinished creating {split} dataset of Waterbirds\n\n")
            return
        with open(os.path.join(root, 'metadata.csv')) as meta_file:
            csv_reader = csv.reader(meta_file)
            for idx, row in enumerate(csv_reader):
                if idx == 0:
                    continue
                img_id,	img_filename, y, split_index, place, place_filename = row
                if data_split[int(split_index)] == split:
                    self.places[img_filename.split('/')[-1]] = int(place)
        self.update_data(img_data_dir)

    def update_data(self, data_file_directory, masked_data_file_path=None):
        self.data_path = []
        self.masked_data_path = []
        self.targets = []
        data_classes = sorted(os.listdir(data_file_directory))
        print("-"*10, f"indexing {self.split} data", "-"*10)
        for data_class in tqdm(data_classes):
            target = int(data_class)
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
            tuple: (image, img_file_path, target)
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
            if self.return_places:
                return img, img_file_path, target, masked_img, self.places[img_file_path.split('/')[-1]]
            return img, img_file_path, target, masked_img
        if self.return_places:
            return img, img_file_path, target, self.places[img_file_path.split('/')[-1]]
        return img, img_file_path, target

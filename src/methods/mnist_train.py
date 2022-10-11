from copy import deepcopy
from src.methods.base_method import TrainBaseMethod
from src.datasets import BiasedMNIST
from src.utils import Cutout
from numpy.random import default_rng

import torchvision.transforms as transforms

import torch
import os


class MnistTrain(TrainBaseMethod):
    def __init__(self, args) -> None:
        super().__init__(args)

    def prepare_data_loaders(self) -> None:
        self.transform_test = transforms.Compose([transforms.ToTensor()])
        self.transform_train = transforms.Compose(
            [transforms.ToTensor(), ])

        self.train_dataset = BiasedMNIST(
            root=os.path.join(self.args.base_dir, "datasets"),
            train=True,
            transform=self.transform_train,
            download=True,
            class_labels_to_filter=[i for i in range(0, 10)],
            new_to_old_label_mapping={
                0: [0, 1, 2, 3, 4], 1: [5, 6, 7, 8, 9]},
            bias_conflicting_data_ratio=self.args.train_bias_conflicting_data_ratio,
            bias_type=self.args.bias_type,
            square_number=self.args.square_number,
        )

        self.val_dataset = deepcopy(self.train_dataset)
        val_data_dir = self.train_dataset.img_data_dir.replace("train", "val")
        if not (os.path.isdir(val_data_dir) and len(os.listdir(val_data_dir)) > 0):
            
            os.makedirs(val_data_dir, exist_ok=True)
            for target in [0, 1]:
                os.makedirs(os.path.join(val_data_dir, str(target)), exist_ok=True)
            rng = default_rng()
            val_indices = rng.choice(len(self.train_dataset), size=12000, replace=False)
            for val_index in val_indices:
                file_path = self.train_dataset.data_path[val_index]
                target = self.train_dataset.targets[val_index]
                new_file_path = os.path.join(
                    val_data_dir, str(target), file_path.split("/")[-1])
                os.replace(file_path, new_file_path)
            self.train_dataset.update_data(self.train_dataset.img_data_dir)
            self.val_dataset.update_data(val_data_dir)
        else:
            self.val_dataset.update_data(val_data_dir)
        if self.args.use_random_masking:
            transform_data_to_mask = transforms.Compose(
            [transforms.ToTensor(), Cutout(1, [i for i in range(2, 15)])])
        else:
            transform_data_to_mask = self.transform_train
        self.data_to_mask_dataset = BiasedMNIST(
            root=os.path.join(self.args.base_dir, "datasets"),
            train=True,
            transform=transform_data_to_mask,
            download=True,
            class_labels_to_filter=[i for i in range(0, 10)],
            new_to_old_label_mapping={
                0: [0, 1, 2, 3, 4], 1: [5, 6, 7, 8, 9]},
            bias_conflicting_data_ratio=self.args.train_bias_conflicting_data_ratio,
            bias_type=self.args.bias_type,
            square_number=self.args.square_number,
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch,
            shuffle=True,
            num_workers=self.args.workers,
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.args.test_batch,
            shuffle=False,
            num_workers=self.args.workers,
        )
        self.data_to_mask_loader = torch.utils.data.DataLoader(
            self.data_to_mask_dataset,
            batch_size=self.args.masking_batch_size,
            shuffle=True,
            num_workers=self.args.workers,
        )
        self.test_dataset = BiasedMNIST(
            root=os.path.join(self.args.base_dir, "datasets"),
            train=False,
            transform=self.transform_test,
            download=True,
            class_labels_to_filter=[i for i in range(0, 10)],
            new_to_old_label_mapping={
                0: [0, 1, 2, 3, 4], 1: [5, 6, 7, 8, 9]},
            bias_conflicting_data_ratio=self.args.test_bias_conflicting_data_ratio,
            bias_type=self.args.test_data_types[0],
            square_number=self.args.square_number,
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.args.test_batch,
            shuffle=False,
            num_workers=self.args.workers,
        )
        print("-" * 10, "datasets and dataloaders are ready.", "-" * 10)

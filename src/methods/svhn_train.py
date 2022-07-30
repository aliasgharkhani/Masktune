from src.methods.base_method import TrainBaseMethod
from src.datasets import SVHNDataset
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision.transforms as transforms
import numpy as np

import torch
import os


class SVHNTrain(TrainBaseMethod):
    def __init__(self, args) -> None:
        super().__init__(args)

    def prepare_data_loaders(self) -> None:
        self.transform_test = transforms.Compose([
                transforms.ToTensor()])
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor()
        ])
        self.transform_data_to_mask = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.train_dataset = SVHNDataset(root=os.path.join(
            self.args.base_dir, 'datasets', 'SVHN', 'raw'), split='train', transform=self.transform_train, download=True)
        self.val_dataset = SVHNDataset(root=os.path.join(
            self.args.base_dir, 'datasets', 'SVHN', 'raw'), split='train', transform=self.transform_test, download=True)
        train_val_indices_file = os.path.join(self.args.base_dir, "datasets", "SVHN", "raw", "train_val_indices.npy")
        if os.path.isfile(train_val_indices_file):
            train_idx, val_idx = np.load(train_val_indices_file, allow_pickle=True)
        else:
            num_train = len(self.train_dataset)
            indices = list(range(num_train))
            split = int(np.floor(0.2 * num_train))

            # if shuffle:
            np.random.seed(50)
            np.random.shuffle(indices)

            train_idx, val_idx = indices[split:], indices[:split]
            np.save(train_val_indices_file, np.array([train_idx, val_idx]))
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        self.data_to_mask_dataset = SVHNDataset(root=os.path.join(
            self.args.base_dir, 'datasets', 'SVHN', 'raw'), split='train', transform=self.transform_data_to_mask, download=True)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.args.train_batch, sampler=train_sampler, shuffle=False, num_workers=self.args.workers)
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.args.test_batch, sampler=val_sampler, shuffle=False, num_workers=self.args.workers)
        self.data_to_mask_loader = torch.utils.data.DataLoader(
            self.data_to_mask_dataset, batch_size=self.args.masking_batch_size, sampler=train_sampler, shuffle=False, num_workers=self.args.workers)
        self.test_dataset = SVHNDataset(root=os.path.join(
            self.args.base_dir, 'datasets', 'SVHN', 'raw'), split='test', transform=self.transform_test, download=True)

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.args.test_batch, shuffle=False, num_workers=self.args.workers)

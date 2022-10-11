from src.methods.base_method import TrainBaseMethod
from src.datasets import IN9L
from src.utils import load_checkpoint, change_column_value_of_existing_row
from src.methods.base_method import Mode

import torchvision.transforms as transforms

import torch


IMAGENET_PCA = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}


class Lighting(object):
    """
    Lighting noise (see https://git.io/fhBOc)
    """

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


# Special transforms for ImageNet(s)
TRAIN_TRANSFORMS_IMAGENET = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1
    ),
    transforms.ToTensor(),
    Lighting(0.05, IMAGENET_PCA['eigval'],
             IMAGENET_PCA['eigvec'])
])


class IN9lTrain(TrainBaseMethod):
    def __init__(self, args) -> None:
        super().__init__(args)

    def prepare_data_loaders(self) -> None:
        self.transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.4717, 0.4499, 0.3837], [
                    0.2600, 0.2516, 0.2575]),
            ])
        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1
            ),
            transforms.ToTensor(),
            Lighting(0.05, IMAGENET_PCA['eigval'],
                        IMAGENET_PCA['eigvec']),
            transforms.Normalize([0.4717, 0.4499, 0.3837], [
                0.2600, 0.2516, 0.2575]),
        ])
        transform_data_to_mask = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.4717, 0.4499, 0.3837], [
                0.2600, 0.2516, 0.2575])
        ])
        self.train_dataset = IN9L(
            root=self.args.dataset_dir, split='train', transform=self.transform_train)
        self.val_dataset = IN9L(
            root=self.args.dataset_dir, split='val', transform=self.transform_test)
        self.data_to_mask_dataset = IN9L(
            root=self.args.dataset_dir, split='train', transform=transform_data_to_mask)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.args.train_batch, shuffle=True, num_workers=self.args.workers)
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.args.test_batch, shuffle=False, num_workers=self.args.workers)
        self.data_to_mask_loader = torch.utils.data.DataLoader(
            self.data_to_mask_dataset, batch_size=self.args.masking_batch_size, shuffle=True, num_workers=self.args.workers)

        self.test_loader = {}
        for test_data_type in self.args.test_data_types:
            self.test_dataset = IN9L(root=self.args.dataset_dir, split=test_data_type,
                                        transform=self.transform_test)
            self.test_loader[test_data_type] = torch.utils.data.DataLoader(
                self.test_dataset, batch_size=self.args.test_batch, shuffle=False, num_workers=self.args.workers)

    def test(self, checkpoint_path=None):
        assert checkpoint_path is not None, "checkpoint path should be passed to the test function to test on that"
        self.logger.info("-" * 10 + "testing the model" +
                         "-" * 10, print_msg=True)
        (
            self.model,
            _,
            _,
            self.current_epoch,
            _,
        ) = load_checkpoint(
            model=self.model,
            optimizer=None,
            lr_scheduler=None,
            checkpoint_path=checkpoint_path,
        )
        for test_data_type in self.test_loader:
            accuracy = self.run_an_epoch(
                data_loader=self.test_loader[test_data_type], epoch=0, mode=Mode.test
            )
            self.logger.info(f"test data type: {test_data_type}", print_msg=True)
            self.logger.info(f"accuracy: {accuracy}", print_msg=True)
        change_column_value_of_existing_row(
            "accuracy",
            accuracy,
            self.run_configs_file_path,
            self.run_id,
        )

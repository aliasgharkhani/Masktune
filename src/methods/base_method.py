import copy
import shutil
from src.models import SmallCNN, ResNet50, resnet32, vgg16_bn
from src.utils import (
    AverageMeter,
    calculate_accuracy,
    change_column_value_of_existing_row,
    load_checkpoint,
    save_checkpoint,
    apply_mask_and_save_images,
    select_device,
    Logger,
    write_config_to_csv,
     update_dataset_and_dataloader,
    mask_heatmap_using_threshold
)

from pytorch_grad_cam import XGradCAM

from torch import Tensor
from tqdm import tqdm
from abc import ABC, abstractmethod

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

import torch
import os
import math
import time
import enum

model_size_configuration = {
    'cifar10': {
        'num_classes': 10,
        'input_size': 32
    },
    'svhn': {
        'num_classes': 10,
        'input_size': 32
    },
    'catsvsdogs': {
        'num_classes': 2,
        'input_size': 64
    },
    'in9l': {
        'num_classes': 9,
        'input_size': 224
    },
    'mnist': {
        'num_classes': 2,
        'input_size': 28
    },
    "celeba": {
        'num_classes': 2,
        'input_size': 224
    }
}


class Mode(enum.Enum):
    train = 0
    val = 1
    test = 2

class TrainBaseMethod(ABC):
    def __init__(self, args) -> None:
        self.args = args
        self.current_epoch = 0
        os.makedirs(os.path.join(
            self.args.base_dir, "runs"
        ), exist_ok=True)
        self.run_configs_file_path = os.path.join(
            self.args.base_dir, "runs", "run_configs.csv"
        )
        self.run_id = write_config_to_csv(args, self.run_configs_file_path)
        os.makedirs(os.path.join(
            self.args.base_dir, "runs"
        ), exist_ok=True)
        self.run_dir = os.path.join(
            args.base_dir, "runs", str(self.run_id))
        os.makedirs(self.run_dir, exist_ok=True)
        log_dir = os.path.join(self.run_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        model_save_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(model_save_dir, exist_ok=True)
        self.model_save_dir = model_save_dir
        masked_data_save_dir = os.path.join(self.run_dir, "masked_data", "1")
        os.makedirs(masked_data_save_dir, exist_ok=True)
        self.masked_data_save_dir = masked_data_save_dir
        self.best_erm_model_checkpoint_path = os.path.join(self.model_save_dir, "best_erm_model_checkpoint.pt")
        self.last_erm_model_checkpoint_path = os.path.join(self.model_save_dir, "last_erm_model_checkpoint.pt")
        self.finetuned_model_checkpoint_path = os.path.join(self.model_save_dir, "finetuned_model.pt")
        self.device = select_device(self.args.use_cuda)
        self.loss_function = nn.CrossEntropyLoss()
        if os.path.isfile(os.path.join(log_dir, "log.txt")):
            self.logger = Logger(log_dir, None)
        else:
            self.logger = Logger(log_dir, str(self.args))
        self.prepare_data_loaders()
        self.prepare_model(arch=self.args.arch)
        self.model = self.model.to(self.device)
        self.prepare_optimizer()
        self.prepare_lr_scheduler()

    @abstractmethod
    def prepare_data_loaders(self) -> None:
        pass

    def prepare_lr_scheduler(self) -> None:
        if self.args.lr_scheduler_name == "multi_step":
            lr_scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=self.optimizer,
                milestones=self.args.schedule,
                gamma=self.args.gamma,
                last_epoch=-1,
            )
        else:
            raise NotImplementedError(
                f"{self.args.lr_scheduler_name} scheduler not implemented!"
            )
        self.lr_scheduler = lr_scheduler

    def prepare_optimizer(self) -> None:
        if self.args.optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
                nesterov=self.args.use_nesterov
            )
        elif self.args.optimizer == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
        else:
            raise NotImplementedError(
                f"{self.args.lr_scheduler_name} optimizer not implemented!"
            )
    
    
    def prepare_model(self, arch, multi_gpu=True) -> None:
        if arch == "small_cnn":
            self.model = SmallCNN(
                num_classes=model_size_configuration[self.args.dataset]['num_classes'])
        elif arch == "resnet50":
            self.model = ResNet50(
                pretrained=self.args.use_pretrained_weights, num_classes=model_size_configuration[self.args.dataset]['num_classes'])
        elif arch == 'resnet32':
            self.model = resnet32(pretrained=self.args.use_pretrained_weights, num_classes=model_size_configuration[self.args.dataset]['num_classes'])
        elif arch == "vgg16_bn":
            self.model = vgg16_bn(num_classes=model_size_configuration[self.args.dataset]['num_classes'],
                                                     input_size=model_size_configuration[self.args.dataset]['input_size'], activation='relu')
        else:
            raise NotImplementedError()
        if multi_gpu:
            self.model = nn.DataParallel(
                self.model, device_ids=self.args.gpu_ids)
        self.model = self.model.to(self.device)

    
    def run_an_epoch(self, data_loader, epoch, mode: Mode = Mode.train):
        if mode == Mode.train:
            self.model.train()
        else:
            self.model.eval()
        losses = AverageMeter()
        accuracies = AverageMeter()
        with torch.set_grad_enabled(mode == Mode.train):
            progress_bar = tqdm(data_loader)
            self.logger.info(
                f"{mode.name} epoch: {epoch}"
            )
            for data in progress_bar:
                progress_bar.set_description(f'{mode.name} epoch {epoch}')
                inputs, targets = data[0], data[2]
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, targets)
                losses.update(loss.item(), inputs.size(0))
                output_probabilities = F.softmax(outputs, dim=1)
                probabilities, predictions = output_probabilities.data.max(1)
                accuracies.update(calculate_accuracy(targets, predictions), 1)
                if mode == Mode.train:
                    self.optimize(loss=loss)

                progress_bar.set_postfix(
                    {
                        "loss": losses.avg,
                        "accuracy": accuracies.avg,
                    }
                )
            self.logger.info(
                f"loss: {losses.avg}"
            )
            self.logger.info(
                f"accuracy: {accuracies.avg}"
            )
        return accuracies.avg

    def optimize(self, loss: Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



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
        accuracy = self.run_an_epoch(
            data_loader=self.test_loader, epoch=0, mode=Mode.test
        )
        change_column_value_of_existing_row(
            "accuracy",
            accuracy,
            self.run_configs_file_path,
            self.run_id,
        )

    def train_erm(self, use_lr_scheduler: bool=False, best_resume_checkpoint_path: str=None, last_resume_checkpoint_path: str=None) -> None:
        resume_epoch = 0
        best_accuracy = -math.inf
        if best_resume_checkpoint_path is not None and last_resume_checkpoint_path is not None:
            shutil.copyfile(best_resume_checkpoint_path, self.best_erm_model_checkpoint_path)
            shutil.copyfile(last_resume_checkpoint_path, self.last_erm_model_checkpoint_path)
            if self.args.resume:
                (
                    self.model,
                    self.optimizer,
                    self.lr_scheduler,
                    last_epoch,
                    best_accuracy,
                ) = load_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    lr_scheduler=self.lr_scheduler,
                    checkpoint_path=last_resume_checkpoint_path
                )
                resume_epoch = last_epoch + 1
                self.logger.info(
                    "-" *
                    10 +
                    f"model checkpoint loaded and resuming from epoch {resume_epoch}" + "-" * 10,
                    print_msg=True
                )
            else:
                (
                    self.model,
                    self.optimizer,
                    self.lr_scheduler,
                    last_epoch,
                    best_accuracy,
                ) = load_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    lr_scheduler=self.lr_scheduler,
                    checkpoint_path=best_resume_checkpoint_path
                )
                self.logger.info(
                    "-" *
                    10 +
                    f"model checkpoint of epoch {last_epoch} loaded" + "-" * 10,
                    print_msg=True
                )
                return
        else:
            self.logger.info(
                "-" *
                10 +
                "training the model from scratch" + "-" * 10,
                print_msg=True
            )
        for current_epoch in range(resume_epoch, self.args.epochs):
            self.current_epoch = current_epoch
            _ = self.run_an_epoch(
                data_loader=self.train_loader, epoch=current_epoch, mode=Mode.train)
            val_accuracy = self.run_an_epoch(
                data_loader=self.val_loader, epoch=0, mode=Mode.val
            )
            if use_lr_scheduler:
                self.lr_scheduler.step()
            self.logger.info(
                f"lr: {self.lr_scheduler.get_last_lr()[0]}",
                print_msg=True
            )
            if val_accuracy > best_accuracy:
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    lr_scheduler=self.lr_scheduler,
                    checkpoint_path=self.best_erm_model_checkpoint_path,
                    current_epoch=self.current_epoch,
                    accuracy=val_accuracy,
                )
                best_accuracy = val_accuracy
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                checkpoint_path=self.last_erm_model_checkpoint_path,
                current_epoch=self.current_epoch,
                accuracy=val_accuracy,
            )


    def finetune(self, use_lr_scheduler: bool=True, erm_checkpoint_path: str=None, use_random_masking: bool=False) -> None:
        assert erm_checkpoint_path is not None, "erm checkpoint should be passed to the finetune function!"        
        self.logger.info(
            "-" *
            10 +
            "finetuning the erm model for one epoch" + "-" * 10,
            print_msg=True
        )
        (
            self.model,
            self.optimizer,
            self.lr_scheduler,
            _,
            _,
        ) = load_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            checkpoint_path=erm_checkpoint_path
        )
        data_loader=self.train_loader
        if use_random_masking:
            data_loader = self.data_to_mask_loader
        self.run_an_epoch(
            data_loader=data_loader, epoch=0, mode=Mode.train)
        val_accuracy = self.run_an_epoch(
            data_loader=self.val_loader, epoch=0, mode=Mode.val
        )
        if use_lr_scheduler:
            self.lr_scheduler.step()
        self.logger.info(
            f"lr: {self.lr_scheduler.get_last_lr()[0]}",
            print_msg=True
        )
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            checkpoint_path=self.finetuned_model_checkpoint_path,
            current_epoch=0,
            accuracy=val_accuracy,
        )
    
    def check_and_load_saved_masks(self, saved_mask_dir):
        data_is_complete = False
        if saved_mask_dir is not None and os.path.isdir(saved_mask_dir):
            masked_data_count = 0
            for label in os.listdir(saved_mask_dir):
                masked_data_count += len(os.listdir(
                    os.path.join(saved_mask_dir, label)))
            if masked_data_count == len(self.train_dataset):
                self.logger.info(
                    "-"*10 + f"using masked data from saved data dir" + "-"*10, print_msg=True)
                self.masked_data_save_dir = self.args.saved_mask_dir
                shutil.copyfile(self.args.best_erm_model_checkpoint_path, self.best_erm_model_checkpoint_path)
                shutil.copyfile(self.args.last_erm_model_checkpoint_path, self.last_erm_model_checkpoint_path)
                data_is_complete = True

        return data_is_complete
    
    def random_mask(self):
        self.train_erm(best_resume_checkpoint_path=self.args.best_erm_model_checkpoint_path, last_resume_checkpoint_path=self.args.last_erm_model_checkpoint_path)
        self.test(checkpoint_path=self.best_erm_model_checkpoint_path)
        self.finetune(use_lr_scheduler=False, erm_checkpoint_path=self.last_erm_model_checkpoint_path, use_random_masking=True)

    def masktune(self) -> None:
        if self.args.masktune_iterations == 1:
            masked_data_is_ready = self.check_and_load_saved_masks(self.args.saved_mask_dir)
        else:
            masked_data_is_ready = False
        if masked_data_is_ready:
            self.test(checkpoint_path=self.best_erm_model_checkpoint_path)
        else:
            self.train_erm(best_resume_checkpoint_path=self.args.best_erm_model_checkpoint_path, last_resume_checkpoint_path=self.args.last_erm_model_checkpoint_path)
            self.test(checkpoint_path=self.best_erm_model_checkpoint_path)
            if not self.args.use_random_masking:
                self.mask_data(erm_checkpoint_path=self.best_erm_model_checkpoint_path)

        if not self.args.use_random_masking:
            self.train_dataset, self.train_loader = update_dataset_and_dataloader(
                self.train_dataset, data_dir=self.masked_data_save_dir, batch_size=self.args.train_batch, workers=self.args.workers)
        self.finetune(use_lr_scheduler=False, erm_checkpoint_path=self.last_erm_model_checkpoint_path, use_random_masking=self.args.use_random_masking)
        for i in range(self.args.masktune_iterations-1):
            if self.args.accumulative_masking:
                self.data_to_mask_dataset, self.data_to_mask_loader = update_dataset_and_dataloader(
                    self.data_to_mask_dataset, data_dir=self.masked_data_save_dir, batch_size=self.args.masking_batch_size, workers=self.args.workers)
            masked_data_save_dir = os.path.join(self.run_dir, "masked_data", str(i+2))
            os.makedirs(masked_data_save_dir, exist_ok=True)
            self.masked_data_save_dir = masked_data_save_dir
            self.mask_data(erm_checkpoint_path=self.finetuned_model_checkpoint_path)
            self.train_dataset, self.train_loader = update_dataset_and_dataloader(
                self.train_dataset, data_dir=self.masked_data_save_dir, batch_size=self.args.train_batch, workers=self.args.workers)
            self.finetune(use_lr_scheduler=False, erm_checkpoint_path=self.finetuned_model_checkpoint_path)


    def mask_data(self, erm_checkpoint_path: str=None):
        assert erm_checkpoint_path is not None, "erm checkpoint should be passed to be used for masking"
        self.logger.info(
            "-" * 10 + f"masking {len(self.data_to_mask_dataset)} data" + "-" * 10, print_msg=True)
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
            checkpoint_path=erm_checkpoint_path,
        )
        heat_map_generator = XGradCAM(
            model=self.model,
            target_layers=[self.model.module.get_grad_cam_target_layer()],
            use_cuda=self.args.use_cuda,
        )
        masking_start_time = time.time()
        for data in tqdm(self.data_to_mask_loader):
            images, images_pathes, targets = data[0], data[1], data[2]
            images = images.to(self.device)
            heat_maps = heat_map_generator(images)
            image_masks = mask_heatmap_using_threshold(heat_maps=heat_maps)
            apply_mask_and_save_images(
                image_masks=image_masks, masked_data_save_dir=self.masked_data_save_dir, images_pathes=images_pathes, targets=targets
            )
        tqdm.write("Masking Time: {}".format(
            time.time() - masking_start_time))


    def test_selective_classification(self, erm_model_checkpoint_path: str=None, finetuned_model_checkpoint_path: str=None):
        assert erm_model_checkpoint_path is not None, "erm model's checkpoint path should be passed to the test function to test on that"
        assert finetuned_model_checkpoint_path is not None, "finetuned model's checkpoint path should be passed to the test function to test on that"
        finetuned_model = copy.deepcopy(self.model)
        finetuned_model = finetuned_model.to(self.device)
        self.logger.info(
            "-" * 10 + "testing with selective classification" + "-" * 10, print_msg=True)
        (
            finetuned_model,
            _,
            _,
            self.current_epoch,
            _,
        ) = load_checkpoint(
            model=finetuned_model,
            optimizer=None,
            lr_scheduler=None,
            checkpoint_path=finetuned_model_checkpoint_path
        )
        finetuned_model.eval()
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
            checkpoint_path=erm_model_checkpoint_path
        )
        self.model.eval()

        val_eqs = []
        val_prob = []
        test_eqs = []
        test_prob = []
        with torch.no_grad():
            for inputs, _, targets in tqdm(self.val_loader):
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)
                outputs = self.model(inputs)
                output_probabilities = F.softmax(outputs, dim=1)
                final_outputs = finetuned_model(inputs)
                final_output_probabilities = F.softmax(
                    final_outputs, dim=1)
                probabilities = F.softmax(
                    output_probabilities * final_output_probabilities, dim=1)
                class_probs, class_preds = probabilities.data.max(1)
                val_prob.append(class_probs)
                equals = class_preds.cpu().eq(targets.data.cpu())
                val_eqs.append(equals)

            val_eqs = torch.cat(val_eqs, 0).cpu()
            val_prob = torch.cat(val_prob, 0).cpu()
            val_indices = torch.sort(val_prob, descending=True)[1]
            val_eqs = torch.gather(val_eqs, dim=0, index=val_indices)
            val_prob = torch.gather(val_prob, dim=0, index=val_indices)

            for inputs, _, targets in tqdm(self.test_loader):
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)
                outputs = self.model(inputs)
                output_probabilities = F.softmax(outputs, dim=1)
                final_outputs = finetuned_model(inputs)
                final_output_probabilities = F.softmax(
                    final_outputs, dim=1)
                probabilities = F.softmax(
                    output_probabilities * final_output_probabilities, dim=1)
                class_probs, class_preds = probabilities.data.max(1)
                test_prob.append(class_probs)
                equals = class_preds.cpu().eq(targets.data.cpu())
                test_eqs.append(equals)
            
            test_eqs = torch.cat(test_eqs, 0).cpu()
            test_prob = torch.cat(test_prob, 0).cpu()
            test_indices = torch.sort(test_prob, descending=True)[1]
            test_eqs = torch.gather(test_eqs, dim=0, index=test_indices)
            test_prob = torch.gather(test_prob, dim=0, index=test_indices)
            
            for e_cov in self.args.coverage:
                thresholded_index = round((e_cov / 100) * len(val_indices))
                threshold = val_prob[min(thresholded_index, len(val_prob)-1)]
                predicted_samples = test_prob>=threshold
                error = 1 - (torch.sum(predicted_samples * test_eqs) / torch.sum(predicted_samples))
                self.logger.info(
                    "EXP COV {}, COV {}, ERROR: {}".format(
                        e_cov, round((torch.sum(predicted_samples).item() / len(predicted_samples)) * 100.0, 2), 100 * round(error.item(), 4)
                    ),
                    print_msg=True
                )
                
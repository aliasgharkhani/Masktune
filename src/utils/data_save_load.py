from PIL import Image
import os
import torch
import cv2

from multiprocessing import Pool
from itertools import cycle

import numpy as np
import torch.nn as nn

def apply_mask_and_save_individual_image_in_png(image_mask, masked_data_save_dir, target, image_path):
    target_dir = os.path.join(masked_data_save_dir, str(target.item()))
    os.makedirs(target_dir, exist_ok=True)
    original_image = Image.open(image_path).convert('RGB')
    image_mask = np.expand_dims(cv2.resize(image_mask, dsize=original_image.size, interpolation=cv2.INTER_NEAREST), axis=-1)
    original_image = np.array(original_image) * image_mask
    im = Image.fromarray(original_image.astype(np.uint8))
    im.save(os.path.join(target_dir, image_path.split("/")[-1]))



def apply_mask_and_save_images(image_masks, masked_data_save_dir, images_pathes, targets):
    pool = Pool()
    pool.starmap(apply_mask_and_save_individual_image_in_png, zip(
        image_masks, cycle([masked_data_save_dir]), targets, images_pathes))
    pool.close()


def save_checkpoint(
    model, optimizer, lr_scheduler, checkpoint_path: str, current_epoch, accuracy
):
    state = {
        "optimizer": optimizer.state_dict(),
        "scheduler": lr_scheduler.state_dict(),
        "epoch": current_epoch,
        "accuracy": accuracy,
    }
    if isinstance(model, nn.DataParallel):
        state["model"] = model.module.state_dict()
    else:
        state["model"] = model.state_dict()
    torch.save(state, checkpoint_path)
    del state
    torch.cuda.empty_cache()


def load_checkpoint(model, optimizer, lr_scheduler, checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    else:
        state = torch.load(checkpoint_path)
        i = 0
        if isinstance(model, nn.DataParallel):
            model_dict = model.module.state_dict()
        else:
            model_dict = model.state_dict()
        model_keys = list(model_dict.keys())
        for key in list(state['model'].keys()):
            if i < len(model_keys) and model_keys[i] in key:
                model_dict[model_keys[i]] = state['model'][key]
                i += 1
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(model_dict)
        else:
            model.load_state_dict(model_dict)
        if optimizer is not None:
            optimizer.load_state_dict(state["optimizer"])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(state["scheduler"])
        current_epoch = state["epoch"] + 1
        # accuracy = state["accuracy"]
        accuracy = 0
        del state
        torch.cuda.empty_cache()
        return model, optimizer, lr_scheduler, current_epoch, accuracy

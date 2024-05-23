import glob
from pathlib import Path
import os

import torch
from torchvision.io import read_image
import torchvision
import torchvision.transforms.functional as tf
import torch.nn.functional as F
from einops import rearrange
import cv2
import albumentations as alb


# Resize and pad image
def process_image(image, input_h, input_w):
    image_h, image_w = tuple(image.shape[1:])

    # Resize to a square of input_h x image_w and maintain aspect ratio with padding
    new_h, new_w = input_h, input_w
    if image_w > image_h:
        new_h = int((image_h / image_w) * new_w)
        if new_h % 2 == 1:
            new_h -= 1
    elif image_w < image_h:
        new_w = int((image_w / image_h) * new_h)
        if new_w % 2 == 1:
            new_w -= 1

    #print(new_w, new_h)

    pad_w, pad_h = int(input_w - new_w), int(input_h - new_h)
    #print(pad_w, pad_h)
    image = tf.resize(image, (new_h, new_w))
    image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0).float()
    image *= (1. / 255.)
    return image, (new_h, new_w), (pad_h, pad_w)


def read_gt_mask(file_path, image_pad):
    pad_h, pad_w = image_pad
    gt_mask = read_image(file_path)
    gt_mask = torchvision.transforms.Grayscale()(gt_mask).float()
    # gt_mask = tf.resize(gt_mask, image_size, interpolation=tf.InterpolationMode.NEAREST)
    # gt_mask = F.pad(gt_mask, (0, pad_w, 0, pad_h), mode='constant', value=0).float()
    # gt_mask = tf.resize(gt_mask, (20, 20), interpolation=tf.InterpolationMode.NEAREST)
    gt_mask = rearrange(gt_mask, 'c h w -> h w c')
    gt_mask = torch.flatten(gt_mask, start_dim=1)

    gt_mask_h, gt_mask_w = tuple(gt_mask.shape[:])
    target = torch.zeros((gt_mask_h, gt_mask_w, 1), dtype=torch.float32)

    for j in range(gt_mask_h):
        for i in range(gt_mask_w):
            if gt_mask[j, i] >= 128:
                target[j, i, 0] = 1

    return target


class Dataset:
    def __init__(self, images_dir, gt_masks_dir, gt_ext, input_h, input_w, batch_size, hflip=False, load_one=False, load_one_batch=False):
        if hflip:
            batch_size = max(1, batch_size // 2)

        self.batch_cache = []
        self.image_paths = glob.glob(images_dir + "\\*")
        cursor = 0

        # aspect_ratios = []

        print("-- LOADING DATASET INTO RAM --")
        while True:
            image_batch = []
            gt_batch = []

            # Create a batch
            for i in range(cursor, min(len(self.image_paths), cursor + batch_size)):
                img_path = self.image_paths[i]
                img_name = Path(img_path).stem
                gt_path = gt_masks_dir + "\\" + img_name + gt_ext

                if not os.path.exists(gt_path):
                    continue

                # Read and process the image and add it to the batch
                image = read_image(img_path)
                # image_h, image_w = tuple(image.shape[1:])
                # ar = (image_w/image_h)
                #
                # if ar not in aspect_ratios:
                #     aspect_ratios.append(ar)

                proc_image, new_size, padding = process_image(image, input_h, input_w)
                image_batch.append(proc_image)

                if hflip:
                    image_flip = tf.hflip(image)
                    proc_image_flip, _, _ = process_image(image_flip, input_h, input_w)
                    image_batch.append(proc_image_flip)

                # Get the corresponding ground truth mask for the image
                gt_mask = read_gt_mask(gt_path, padding)
                gt_batch.append(gt_mask)

                if hflip:
                    gt_mask_flip = tf.hflip(gt_mask)
                    gt_batch.append(gt_mask_flip)

                if load_one:
                    break

            image_batch_tensor = torch.stack(image_batch)
            gt_batch_tensor = torch.stack(gt_batch)
            self.batch_cache.append({'images': image_batch_tensor, 'gt_masks': gt_batch_tensor})

            cursor += batch_size
            if cursor >= len(self.image_paths) or load_one or load_one_batch:
                break

        # print(aspect_ratios)

    def __iter__(self):
        for v in self.batch_cache:
            yield v['images'], v['gt_masks']

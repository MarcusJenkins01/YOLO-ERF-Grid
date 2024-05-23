import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import os
import json
import torchvision.transforms.functional as tf
from einops import rearrange

from yoloerf import YOLO_ERF
from dataset import Dataset


# Validate configuration
WEIGHTS_PATH = "polynomial-lr focal sgd lr 0.01 wd 1e-4 best.pt"
IMAGES_DIR_VAL = "C:\\Users\\865le\Documents\\Drone autonomous SLZ detection\\yolov8\\datasets\\VisDrone\\VisDrone2019-DET-val\\images"
GT_MASKS_DIR_VAL = "C:\\Users\\865le\\Documents\\Drone autonomous SLZ detection\\yolov8\\datasets\\VisDrone\\VisDrone2019-DET-val\\masks"
GT_EXTENSION_VAL = ".jpg"
GT_EXTENSION = ".jpg"
IMAGE_W = 640
IMAGE_H = 480
BATCH_SIZE = 1
EPOCHS_PER_VAL = 2

THRESHOLD = 0.5


# Prepare model for eval
model = YOLO_ERF(
    init_biases=False)

model.cuda()
model.eval()
print("NO. OF PARAMETERS:", sum(p.numel() for p in model.parameters()))

# Load val set into RAM
val_set = Dataset(images_dir=IMAGES_DIR_VAL,
                  gt_masks_dir=GT_MASKS_DIR_VAL,
                  gt_ext=GT_EXTENSION_VAL,
                  input_w=IMAGE_W,
                  input_h=IMAGE_H,
                  batch_size=BATCH_SIZE, load_one=False)


def file_exists(file_path):
    return len(file_path) > 0 and \
        os.path.exists(file_path) and \
        os.path.isfile(file_path)


def val():
    # Loads specified weights if they exist
    if file_exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH))

    tn_count = fp_count = fn_count = tp_count = 0

    for image_batch, target_batch in val_set:
        image_batch = image_batch.cuda()
        target_batch = target_batch.cuda()

        pred = model(image_batch)

        # Test display of the output
        pred[pred[:, :, :, 0] < THRESHOLD] = 0
        pred[pred[:, :, :, 0] >= THRESHOLD] = 1

        for d in range(len(pred)):
            s_pred = pred[d]
            pred_flat = torch.flatten(s_pred, start_dim=0)
            target_flat = torch.flatten(target_batch[d], start_dim=0)

            for i in range(0, len(pred_flat)):
                p = int(pred_flat[i])
                t = int(target_flat[i])
                if p == t:
                    if p == 1:
                        tp_count += 1
                    else:
                        tn_count += 1
                else:
                    if p == 1:
                        fp_count += 1
                    else:
                        fn_count += 1

            input_image = image_batch[d]
            # input_image = UnNormalize(input_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) * (1. / 255.)
            input_image = rearrange(input_image, 'c h w -> h w c')

            # Display the output
            s_pred = rearrange(s_pred, 'h w c -> c h w')
            s_pred = tf.resize(s_pred, (IMAGE_H, IMAGE_W),
                               interpolation=tf.InterpolationMode.NEAREST)
            s_pred = rearrange(s_pred, 'c h w -> h w c')
            s_pred *= 5.
            input_image[..., 0] += s_pred[..., 0]
            input_image = input_image.detach().cpu().numpy()
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            cv2.imshow("output", input_image)
            cv2.waitKey(0)

    ##        print(tp_count)
    ##        print(tn_count)
    ##        print(fp_count)
    ##        print(fn_count)
    ##        print("Precision:", precision)
    ##        print()

    print("True positives:", tp_count)
    print("False positives:", fp_count)
    print("True negatives:", tn_count)
    print("False negatives:", fn_count)

    average_precision = tp_count / max(1, tp_count + fp_count)
    average_recall = tp_count / max(1, tp_count + fn_count)

    f1_score = (2 * (average_precision * average_recall)) / (average_precision + average_recall)
    print("Average precision:", average_precision)
    print("Average recall:", average_recall)
    print("F1 score:", f1_score)

val()

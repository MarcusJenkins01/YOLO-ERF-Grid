import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import cv2
import os
import json
import torchvision.transforms.functional as tf
from einops import rearrange
import warnings

from yoloerf import YOLO_ERF
from dataset import Dataset

# Training configuration
MAX_EPOCHS = 1000
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
FOCAL_LOSS_GAMMA = 1
FOCAL_LOSS_ALPHA = 0.65
TOTAL_ITERS = 30e3
WEIGHTS_PATH = "polynomial-lr focal sgd lr 0.01 wd 1e-4.pt"
WEIGHTS_PATH_BEST = "polynomial-lr focal sgd lr 0.01 wd 1e-4 best.pt"
WEIGHTS_PATH_BEST_IOU = "polynomial-lr focal sgd lr 0.01 wd 1e-4 best iou.pt"

# Train dataset configuration
IMAGES_DIR = "VisDrone\\VisDrone2019-DET-train\\images"
GT_MASKS_DIR = "VisDrone\\VisDrone2019-DET-train\\masks"
GT_EXTENSION = ".jpg"
IMAGE_H = 480
IMAGE_W = 640
BATCH_SIZE = 3

# Validate configuration
IMAGES_DIR_VAL = "C:\\Users\\865le\Documents\\Drone autonomous SLZ detection\\yolov8\\datasets\\VisDrone\\VisDrone2019-DET-val\\images"
GT_MASKS_DIR_VAL = "C:\\Users\\865le\\Documents\\Drone autonomous SLZ detection\\yolov8\\datasets\\VisDrone\\VisDrone2019-DET-val\\masks"
GT_EXTENSION_VAL = ".jpg"
EPOCHS_PER_VAL = 1
THRESHOLD = 0.5


# Prepare model for training
model = YOLO_ERF()

model.cuda()
print("NO. OF PARAMETERS:", sum(p.numel() for p in model.parameters()))

# Load training set into RAM
train_set = Dataset(images_dir=IMAGES_DIR,
                    gt_masks_dir=GT_MASKS_DIR,
                    gt_ext=GT_EXTENSION,
                    input_h=IMAGE_H,
                    input_w=IMAGE_W,
                    batch_size=BATCH_SIZE, load_one_batch=False)

val_set = Dataset(images_dir=IMAGES_DIR_VAL,
                  gt_masks_dir=GT_MASKS_DIR_VAL,
                  gt_ext=GT_EXTENSION_VAL,
                  input_h=IMAGE_H,
                  input_w=IMAGE_W,
                  batch_size=BATCH_SIZE, load_one_batch=False)


def file_exists(file_path):
    return len(file_path) > 0 and \
        os.path.exists(file_path) and \
        os.path.isfile(file_path)


def save_train_info(dict):
    train_info_str = json.dumps(dict)
    with open('train_info.json', 'w') as o:
        o.write(train_info_str)


class BinaryFocalLoss(torch.nn.modules.loss._Loss):
    def __init__(self, gamma=2, alpha=0.75):
        super().__init__(None, None)
        self.gamma = gamma
        self.alpha = alpha
        self.eps = torch.finfo(torch.float32).eps

    def forward(self, input_tensor, target):
        # Warn if sizes don't match
        if not target.size() == input_tensor.size():
            warnings.warn(
                f"Using a target size ({target.size()}) that is different to the input size" \
                "({input_tensor.size()}). \n This will likely lead to incorrect results" \
                "due to broadcasting.\n Please ensure they have the same size.",
                stacklevel=2,
            )

        # Broadcast to get sizes/shapes to match
        input_tensor, target = torch.broadcast_tensors(input_tensor, target)
        assert input_tensor.shape == target.shape, "Input and target tensor shapes don't match"

        bg_preds = input_tensor[target == 0]
        obj_preds = input_tensor[target == 1]

        fl_obj = -self.alpha * (1 - obj_preds) ** self.gamma * torch.log(torch.clamp(obj_preds, min=self.eps, max=1.0))
        fl_bg = -(1 - self.alpha) * bg_preds ** self.gamma * torch.log(torch.clamp(1 - bg_preds, min=self.eps, max=1.0))

        fl_all = torch.cat((fl_obj, fl_bg))
        fl_mean = torch.mean(fl_all)

        return fl_mean


# class PolyLR(_LRScheduler):
#     def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
#         self.power = power
#         self.max_iters = max_iters  # avoid zero lr
#         self.min_lr = min_lr
#         super(PolyLR, self).__init__(optimizer, last_epoch)
#
#     def get_lr(self):
#         return [max(base_lr * (1 - self.last_epoch / self.max_iters) ** self.power, self.min_lr)
#                 for base_lr in self.base_lrs]


class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, total_iters=5, power=1.0, last_epoch=-1, verbose=False, min_lr=1e-6):
        self.total_iters = total_iters
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        decay_factor = ((1.0 - self.last_epoch / self.total_iters) /
                        (1.0 - (self.last_epoch - 1) / self.total_iters)) ** self.power
        return [max(group["lr"], self.min_lr * decay_factor) for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [
            (
                max(base_lr * (1.0 - min(self.total_iters, self.last_epoch) / self.total_iters) ** self.power,
                    self.min_lr)
            )
            for base_lr in self.base_lrs
        ]


def train():
    # Load train data info and add current weights if they don't exist
    train_info = {}
    if file_exists("train_info.json"):
        with open('train_info.json', 'r') as f:
            train_info = json.load(f)

    if train_info.get(WEIGHTS_PATH) is None or not file_exists(WEIGHTS_PATH):
        train_info[WEIGHTS_PATH] = {
            'total_epochs': 0,
            'loss_data': [],
            'val_loss_data': [],
            'f1_data': []
        }

    start_epoch = train_info[WEIGHTS_PATH]['total_epochs']
    best_val_loss = train_info[WEIGHTS_PATH].get("best_val_loss")
    best_val_iou = train_info[WEIGHTS_PATH].get("best_val_iou")

    # Loads specified weights if they exist
    if file_exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH))

    # Set up optimiser and loss function
    optimiser = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    scheduler = PolynomialLR(optimiser, total_iters=TOTAL_ITERS, power=0.9)
    if start_epoch > 0:
        scheduler.step(start_epoch)

    # Set up optimiser and loss function
    # optimiser = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    # loss_fn = nn.BCELoss()
    # loss_fn = BinaryFocalLoss(gamma=FOCAL_LOSS_GAMMA, alpha=FOCAL_LOSS_ALPHA)
    # optimiser = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    # optimiser = optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)
    loss_fn = BinaryFocalLoss(gamma=FOCAL_LOSS_GAMMA, alpha=FOCAL_LOSS_ALPHA)

    # Test model runs before loading in dataset
    test_input = torch.rand((BATCH_SIZE, 3, IMAGE_H, IMAGE_W), dtype=torch.float32).cuda()
    test_target = torch.rand((BATCH_SIZE, 30, 40, 1), dtype=torch.float32).cuda()
    test_pred = model(test_input)
    test_loss = loss_fn(test_pred, test_target)

    for epoch in range(start_epoch, MAX_EPOCHS):
        model.train()

        total_loss = 0
        run_count = 0

        for image_batch, target_batch in train_set:
            image_batch = image_batch.cuda()
            target_batch = target_batch.cuda()

            pred = model(image_batch)
            optimiser.zero_grad()
            loss = loss_fn(pred, target_batch)
            loss.backward()
            optimiser.step()
            total_loss += float(loss.detach().cpu())
            run_count += 1

##            pred_1 = pred[0]
##            pred_1 = rearrange(pred_1, 'h w c -> c h w')
##            pred_1 = tf.resize(pred_1, (640, 640),
##                            interpolation=tf.InterpolationMode.NEAREST)
##            pred_1 = rearrange(pred_1, 'c h w -> h w c')
##            pred_1 = pred_1.detach().cpu().squeeze(0).numpy()
##            cv2.imshow("output mask", pred_1)
##            cv2.waitKey(100)

        mean_loss = total_loss / run_count
        print("Epoch:", epoch, "completed")
        print("Mean loss:", mean_loss)
        torch.save(model.state_dict(), WEIGHTS_PATH)  # save weights after each epoch

        # Check performance on validation dataset to save best weights
        if epoch % EPOCHS_PER_VAL == 0:
            model.eval()

            total_loss_val = 0
            run_count_val = 0
            tn_count = fp_count = fn_count = tp_count = 0

            for image_batch_val, target_batch_val in val_set:
                image_batch_val = image_batch_val.cuda()
                target_batch_val = target_batch_val.cuda()

                pred_val = model(image_batch_val)
                loss_val = loss_fn(pred_val, target_batch_val)
                total_loss_val += float(loss_val.detach().cpu())

                # Calculate F1 score
                pred_val[pred_val[:, :, :, 0] < THRESHOLD] = 0
                pred_val[pred_val[:, :, :, 0] >= THRESHOLD] = 1

                for d in range(len(pred_val)):
                    pred_flat = torch.flatten(pred_val[d], start_dim=0)
                    target_flat = torch.flatten(target_batch_val[d], start_dim=0)

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

                run_count_val += 1

            precision = tp_count / max(1, tp_count + fp_count)
            recall = tp_count / max(1, tp_count + fn_count)
            if precision > 0 or recall > 0:
                f1_score = (2 * (precision * recall)) / (precision + recall)
                print("F1 score:", f1_score)

            iou_score = tp_count / max(tp_count + fp_count + fn_count, 1)
            print("IoU:", iou_score)

            print("True positives:", tp_count)
            print("False positives:", fp_count)
            print("True negatives:", tn_count)
            print("False negatives:", fn_count)

            mean_val_loss = total_loss_val / run_count_val
            print("Mean val loss:", mean_val_loss)

            if best_val_loss is None or mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                torch.save(model.state_dict(), WEIGHTS_PATH_BEST)
                train_info[WEIGHTS_PATH]['best_val_loss'] = mean_val_loss
                save_train_info(train_info)
                print("*New best val loss*")

            if best_val_iou is None or iou_score > best_val_iou:
                best_val_iou = iou_score
                torch.save(model.state_dict(), WEIGHTS_PATH_BEST_IOU)
                train_info[WEIGHTS_PATH]['best_val_iou'] = iou_score
                print("*New best val IoU*")

        print()

        # Save train info
        train_info[WEIGHTS_PATH]['total_epochs'] += 1
        train_info[WEIGHTS_PATH]['loss_data'].append(mean_loss)
        train_info[WEIGHTS_PATH]['val_loss_data'].append(mean_val_loss)
        train_info[WEIGHTS_PATH]['f1_data'].append(f1_score)
        save_train_info(train_info)


train()


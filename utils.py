import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import measure as skim
from scipy import ndimage as ndi


def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def calculate_iou(inputs, targets, smooth=1):
    inputs = F.sigmoid(inputs)       
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection 
    IoU = (intersection + smooth)/(union + smooth)
    
    return IoU

def calculate_iou_multiclass(inputs, targets, smooth=1):
    inputs = F.sigmoid(inputs)

    transform_targets = torch.zeros((inputs.shape))

    for i in range(transform_targets.size(1)):
        transform_targets[:, i, :, :] = torch.where(targets.float() == torch.tensor(i+1).float().cuda(), targets.float(), torch.tensor(0).float().cuda()).squeeze(1)
    
    inputs = inputs.view(-1).cuda()
    transform_targets = transform_targets.view(-1).cuda()
    intersection = (inputs * transform_targets).sum()
    total = (inputs + transform_targets).sum()
    union = total - intersection 
    IoU = (intersection + smooth)/(union + smooth)
    
    return IoU


def make_weight_map(mask, w0=3, sigma=2):
    mask = mask.astype(bool)
    shape = mask.shape[:2]
    ly, lx = shape

    # В w_dt будут финальные веса.
    w_c = np.zeros(shape, dtype=np.float32)
    w_dt = np.zeros(shape, dtype=np.float32)

    # Сначала считаем w_c.
    mask_fg_num = np.sum(mask)
    mask_bg_num = ly * lx - mask_fg_num
    weight_bg = 1 / mask_bg_num if mask_bg_num else 1
    weight_fg = 1 / mask_fg_num if mask_fg_num else 1
    weight_max = max(weight_bg, weight_fg)

    w_c[mask == 0] = weight_bg / weight_max
    w_c[mask != 0] = weight_fg / weight_max

    # Затем создаем массив (buffer_size, ly, lx), где каждый срез (ly, lx) содержит расстояния до одного полигона.
    if mask_fg_num:
        buffer_size = 3
        distances = np.zeros((buffer_size,) + shape, dtype=np.float32)
        labels, labels_num = skim.label(mask, connectivity=2, return_num=True)
        for i in range(labels_num):
            dt = ndi.distance_transform_edt(np.logical_not(labels == i + 1)).astype(np.float32)
            distances[i if i < buffer_size - 1 else -1, :, :] = dt
            if i >= buffer_size - 1:
                distances = np.partition(distances, 1, axis=0)

        # Расстояния до двух ближайших полигонов.
        d1, d2 = distances[:2]

        w_dt = w0 * np.exp(-((d1 ** 2 + d2 ** 2) / (2 * sigma ** 2))) * (mask == 0)

    return w_c + w_dt


def make_mask_with_borders(mask):
    im_border = make_weight_map(mask)
    im_border = np.where((im_border > 1), 2, im_border)
    im_border = np.where((im_border > 0) & (im_border < 1), 0, im_border)
    im_border = np.where(im_border == 2, im_border, 0)
    im_final = mask + im_border

    return im_final

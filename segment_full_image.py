import os
import numpy as np
from tqdm import tqdm
from skimage import io
import segmentation_models_pytorch as smp
import torch
from models.incfcn import Inc_FCN


def get_one_patch_coords(y, x, ly, lx, patch_size):
    if y + patch_size > ly or x + patch_size > lx:
        if y + patch_size > ly:
            y_stop = ly
            y_start = ly - patch_size
            y = y_start
        else:
            y_stop = y + patch_size
            y_start = y
        if x + patch_size > lx:
            x_stop = lx
            x_start = lx - patch_size
            x = x_start
        else:
            x_stop = x + patch_size
            x_start = x
    else:
        y_start = y
        y_stop = y + patch_size
        x_start = x
        x_stop = x + patch_size

    return y, x, y_start, y_stop, x_start, x_stop


def get_patches_coords(ly, lx, patch_size, step_size):
    y_steps = []

    for t in range(0, ly, step_size):
        y_steps.append(t)
        if t + patch_size > ly:
            break

    x_steps = []

    for t in range(0, lx, step_size):
        x_steps.append(t)
        if t + patch_size > lx:
            break

    for y in y_steps:
        for x in x_steps:
            yield get_one_patch_coords(y, x, ly, lx, patch_size)


def segment(
    image,
    model,
    classes_num=1,
    threshold=0.5,
    patch_size=512,
    patch_overlap_size=64,
    batch_size=32,
    show_progress=False
):
    
    ly, lx, lc = image.shape[:3]
    step = patch_size - patch_overlap_size

    mask = torch.zeros(image.shape[:2] + (classes_num,))
    averaging_coeffs = torch.zeros(image.shape[:2] + (1,))

    patches = list(get_patches_coords(ly, lx, patch_size, step))
    patches_num = len(patches)

    batch = []
    patch_coords = []

    if show_progress:
        pbar = tqdm(total=patches_num)

    patch_num = 0

    for y, x, y_start, y_stop, x_start, x_stop in patches:
        if show_progress:
            pbar.update(1)
        patch_num += 1
        patch = image[y_start:y_stop, x_start:x_stop, :]

        if torch.any(torch.tensor(patch).float()):
            batch.append(patch)
            patch_coords.append((y_start, y_stop, x_start, x_stop))

        if batch and (len(batch) == batch_size or patch_num == patches_num):
            pred_batch = []
            with torch.no_grad():
                for prb in batch:
                    prb = np.transpose(prb, (2, 0, 1))
                    prb = np.expand_dims(prb, axis=0)
                    prb = prb.astype(np.float32)
                    prb = torch.from_numpy(prb)
                    prb = prb.cuda()
                    _pred = model(prb)
                    _pred = torch.sigmoid(_pred)
                    _pred = _pred[0].cpu().numpy()
                    _pred = np.squeeze(_pred, axis=0)
                    _pred = _pred > 0.01
                    _pred = np.array(_pred, dtype=np.uint8)
                    pred_batch.append(_pred)
                
                for i in range(len(batch)):
                    pred = pred_batch[i]
                    y_start, y_stop, x_start, x_stop = patch_coords[i]
                    pred = np.expand_dims(pred, axis=-1)

                    mask[y_start:y_stop, x_start:x_stop] += pred
                    averaging_coeffs[y_start:y_stop, x_start:x_stop, 0] += 1
                
                batch = []
                patch_coords = []

    if show_progress:
        pbar.close()

    averaging_coeffs[averaging_coeffs == 0] = 1
    mask /= averaging_coeffs

    if threshold:
        mask = mask > threshold
    
    if classes_num == 1:
        mask = mask[..., 0]

    return mask


def segment_full_image(model, image, *, patch_size, patch_overlap_size, classes_num, threshold=None):
    ly, lx = image.shape[:2]

    unpad = False
    if ly < patch_size or lx < patch_size:
        unpad = True
        print('Image is too small for segmentation. Padding image.')
        image = np.pad(
            image,
            (
                (0, max(0, patch_size - ly)),
                (0, max(0, patch_size - lx)),
                (0, 0)
            ),
            constant_values=0.
        )
        print(f'Shape after padding: {image.shape}')

    print('Segmenting')

    pred_prob = segment(
        image,
        model,
        classes_num=classes_num,
        threshold=None,
        patch_size=patch_size,
        patch_overlap_size=patch_overlap_size,
        show_progress=True
    )

    if unpad:
        pred_prob = pred_prob[:ly, :lx, ...]

    if classes_num > 1:
        channels = list(range(classes_num))
        pred_prob = pred_prob[..., [channels[-1]] + channels[:-1]]

    mask = np.zeros_like(pred_prob, dtype=bool)

    if classes_num > 1:
        idx = np.argmax(pred_prob, axis=-1)

        for cls_idx in range(classes_num):
            mask[..., cls_idx] = idx == cls_idx
        
        mask = mask[..., [channels[1:] + [channels[0]]]]
    elif threshold:
        mask = pred_prob > threshold
    else:
        mask = pred_prob

    return mask


def image_preprocessing(path):
    image = io.imread(path)
    image = image / (2 ** 8 - 1)
    image = torch.tensor(image).float()

    return image


def segment_one_image():
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    torch.cuda.empty_cache()
    device = torch.device("cuda")
    classes_num = 1

    path_to_full_image = "/home/n.kotov1/sar_data/Sentinel-1_full_6.tif"
    image = image_preprocessing(path_to_full_image)
    
    # model = Inc_FCN(3, num_classes=classes_num)
    # model = torch.nn.DataParallel(model)
    # model = model.to(device)

    model = smp.DeepLabV3Plus(encoder_name='resnet50', 
                   encoder_weights="imagenet", 
                   in_channels=3, 
                   classes=classes_num)
    model = model.to(device)

    weights = torch.load("/home/n.kotov1/sar_segmentation/weight/Germany_Africa_field/Deeplabv3plus_ResNet50_rmsprop_dice/Deeplabv3plus_ResNet50_rmsprop_dice_best.pth", map_location=device)
    # weights = torch.load("/home/n.kotov1/sar_segmentation/weight/Germany_Africa_field/IncFCN_rmsprop_dice/IncFCN_rmsprop_dice_best.pth", map_location=device)
    model.load_state_dict(weights, strict = False)
    model = model.cuda()
    model.eval()
    print('Model loaded')

    mask = segment_full_image(
        model,
        image,
        patch_size=2048,
        patch_overlap_size=1024,
        classes_num=classes_num,
        threshold=0.01
    )

    mask = mask.numpy()
    mask = mask.astype(np.uint8) * (2 ** 8 - 1)

    from skimage.io import imsave
    imsave('Sentinel-1_full_6.tif', mask)


def main():
    segment_one_image()


if __name__ == "__main__":
    main()

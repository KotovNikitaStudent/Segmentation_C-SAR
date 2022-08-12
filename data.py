import numpy as np
from skimage import io
import torch
from torch.utils.data import Dataset


class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path, transform=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)
        self.transform = transform

    def __getitem__(self, index):
        image = io.imread(self.images_path[index])
        mask = io.imread(self.masks_path[index])
        
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)

        mask = mask.astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        return image, mask

    def __len__(self):
        return self.n_samples
import os
from skimage import io
import findpeaks
import numpy as np
from glob import glob
from tqdm import tqdm


def main():
    winsize = 7
    cu_value = 0.3
    
    DATA_ROOT = "/raid/n.kotov1/sar_data/set0_p256_s128_3ch_fsplit_sat_resc_lee"
    folders = ["train", "test", "val"]

    for fl in folders:
        local_folder = sorted(glob(os.path.join(DATA_ROOT, fl, "images", "*.tif")))
        
        for it in tqdm(local_folder, total=len(local_folder)):
            img = io.imread(it)
            new_img = np.zeros((img.shape))
    
            for ch in range(img.shape[-1]):
                im = img[:, :, ch]
                im = np.array(im * (2 ** 8 - 1), dtype=np.uint8)
                image_lee = findpeaks.lee_filter(im, win_size=winsize, cu=cu_value)
                new_img[:, :, ch] = image_lee
                image_kuan = findpeaks.kuan_filter(im, win_size=winsize, cu=cu_value)
                new_img[:, :, ch] = image_kuan
                image_median = findpeaks.median_filter(im, win_size=winsize)
                new_img[:, :, ch] = image_median
            
            new_img = new_img / (2 ** 8 - 1)
            io.imsave(it, new_img)

    
if __name__ == "__main__":
    main()

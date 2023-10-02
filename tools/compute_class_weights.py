from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2

from dan.utils.data import get_filenames_list
from Front2BEV.utils import get_test_dirs
import Front2BEV.utils.bev as bev

DATASET_PATH = Path("/media/aisyslab/BICHO/Datasets/Dan-2023-Front2BEV/")
N_CLASSES = 3

msk = bev.mask64.copy()
print("\nMASK64:", np.unique(msk, return_counts=True)[1], '\n')

def main():

    pixel_count = {key:value for (key,value) in enumerate([0 for i in range(N_CLASSES + 1)])}
    pixel_count_fov = {key:value for (key,value) in enumerate([0 for i in range(N_CLASSES)])}

    test_paths = get_test_dirs(DATASET_PATH)

    total_pixels = 0
    total_fov_pixels = 0

    for test_path in tqdm(test_paths):
        bev_raw_path = test_path / "bev" / f"{N_CLASSES}k"
        bev_imgs = get_filenames_list(bev_raw_path, ".png")

        for bev_img_name in bev_imgs:
            bev_img = cv2.imread(str(bev_raw_path / bev_img_name), cv2.IMREAD_GRAYSCALE)

            pixel_count, n_pixels = bev.count_pixels(bev_img, pixel_count, N_CLASSES,False)
            total_pixels += n_pixels

            pixel_count_fov, n_fov_pixels = bev.count_pixels(bev_img, pixel_count_fov, N_CLASSES, True)
            total_fov_pixels += n_fov_pixels
            
    weights = {key:(total_pixels/value) for (key,value) in pixel_count.items()}
    weights_fov = {key:(total_fov_pixels/value) for (key,value) in pixel_count_fov.items()}


    print("\nTotal_pixels:", total_pixels) 
    print('Pixel_count:', pixel_count)
    print('Class weights:', weights, '\n')

    print("\nTotal_FOV_pixels:", total_fov_pixels) 
    print('FOV_Pixel_count:', pixel_count_fov)
    print('Class weights (FOV):', weights_fov, '\n')

if __name__ == '__main__':
  main()




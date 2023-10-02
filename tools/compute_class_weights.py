import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')

from pathlib import Path
from tqdm import tqdm
import numpy as np

import cv2

from dan.utils.data import get_filenames_list

from utils import get_test_dirs

DATASET_PATH = Path("/media/dan/BICHO/Datasets/Dan-2023-Front2BEV/")

N_CLASSES = 3

from utils.bev import *
def get_valid_pixels(img):
    FOVmsk = mask64.copy()
    print("\n FOVMASK", np.unique(FOVmsk, return_counts=True))

    valid_FOV_index = FOVmsk.reshape(-1) != 0

    valid_index = img.reshape(-1) != N_CLASSES
    valid_index = valid_index * valid_FOV_index

    img = img.reshape(-1)[valid_index]
    img = img.reshape(1, -1)
    print("\n", len(FOVmsk.flatten()), len(valid_index),len(img.flatten()))

def debug():
    print("\n FOVMASK", np.unique(mask64, return_counts=True))

    bev_img = cv2.imread("__assets/1911.jpg", cv2.IMREAD_GRAYSCALE)

    segmented = remap_seg(bev_img, bev_color2class, N_CLASSES)
    resized = resize_img(segmented)

    plt.imshow(resized)
    plt.show()




def main():

    pixel_count = {key:value for (key,value) in enumerate([0 for i in range(N_CLASSES + 1)])}

    test_paths = get_test_dirs(DATASET_PATH)

    total_pixels = 0

    for test_path in tqdm(test_paths):
        bev_raw_path = test_path / "bev" / f"{N_CLASSES}k"
        bev_imgs = get_filenames_list(bev_raw_path, ".png")

        for bev_img_name in tqdm(bev_imgs):
            bev_img = cv2.imread(str(bev_raw_path / bev_img_name), cv2.IMREAD_GRAYSCALE)

            valid_pixels = get_valid_pixels(bev_img)

            clss, counts = np.unique(bev_img, return_counts=True)
            for cls, count in zip(clss, counts):
                pixel_count[cls] += count

            total_pixels += len(bev_img.flatten())
            
            break
        break
           

    weights = {key:(total_pixels/value) for (key,value) in pixel_count.items()}

    print("\nTotal_pixels:", total_pixels)      


    print(pixel_count)
    print()
    print(weights)

    
if __name__ == '__main__':
  #  main()
  debug()


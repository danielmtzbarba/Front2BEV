from pathlib import Path
from tqdm import tqdm
import numpy as np

import cv2

from dan.utils import make_folder
from dan.utils.data import get_filenames_list

from utils import get_test_dirs
from utils.bev import postprocess, bev_color2class

MASK = cv2.imread(str(Path("utils") / "_mask1024.png"), 0).astype(np.uint8)
DATASET_PATH = Path("D:/Datasets/Dan-2023-Front2BEV/")

def main():

    test_paths = get_test_dirs(DATASET_PATH)
    for test_path in tqdm(test_paths):
        bev_raw_path = test_path / "bev"
        save_bev = make_folder(test_path, "bev2")
        print("Ssaving in: ", save_bev)

        bev_imgs = get_filenames_list(bev_raw_path, ".jpg")
        for bev_img_name in tqdm(bev_imgs):
            bev_img = cv2.imread(str(bev_raw_path / bev_img_name), cv2.IMREAD_GRAYSCALE)
            bev_post = postprocess(bev_img, bev_color2class, MASK)
            cv2.imwrite(str(save_bev / bev_img_name).replace('.jpg', '.png'), bev_post)

if __name__ == '__main__':
    main()
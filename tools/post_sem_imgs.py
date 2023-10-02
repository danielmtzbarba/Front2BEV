import sys
sys.path.append("/home/aircv1/Data/Luis/aisyslab/Daniel/Projects/")

from pathlib import Path
from tqdm import tqdm

import cv2

from dan.utils import make_folder
from dan.utils.data import get_filenames_list

from Front2BEV.utils import get_test_dirs
from Front2BEV.utils.bev import postprocess, bev_color2class

#DATASET_PATH = Path("/media/aisyslab/BICHO/Datasets/Dan-2023-Front2BEV/")
DATASET_PATH = Path("/home/aircv1/Data/Luis/aisyslab/Daniel/Dan-2023-Front2BEV/")

N_CLASSES = 3
def main():

    test_paths = get_test_dirs(DATASET_PATH)
    for test_path in tqdm(test_paths):
        bev_raw_path = test_path / "bev"
        save_bev = make_folder(bev_raw_path, f"{N_CLASSES}k")
        print("Saving in: ", save_bev)

        bev_raw_path = test_path / "bev" / "sem"
        
        bev_imgs = get_filenames_list(bev_raw_path, ".jpg")
        for bev_img_name in tqdm(bev_imgs):
            bev_img = cv2.imread(str(bev_raw_path / bev_img_name), cv2.IMREAD_GRAYSCALE)
            bev_post = postprocess(bev_img, bev_color2class, n_classes=N_CLASSES)
            cv2.imwrite(str(save_bev / bev_img_name).replace('.jpg', '.png'), bev_post)

if __name__ == '__main__':
    main()
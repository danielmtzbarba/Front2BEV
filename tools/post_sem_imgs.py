import sys
sys.path.append("/home/aircv1/Data/Luis/aisyslab/Daniel/Projects/")

from pathlib import Path
from tqdm import tqdm

import cv2

from dan.utils import make_folder
from dan.utils.data import get_filenames_list

from Front2BEV.utils import get_test_dirs
import Front2BEV.utils.bev as bev
from Front2BEV.utils.bev_classes import bev_cls

def post_proc_bev(test_paths, n_classes):
    for test_path in tqdm(test_paths):
        bev_raw_path = test_path / "bev"
        save_bev = make_folder(bev_raw_path, f"{n_classes}k")
        print("Saving in: ", save_bev)

        bev_raw_path = test_path / "bev" / "sem"
        
        bev_imgs = get_filenames_list(bev_raw_path, ".jpg")
        for bev_img_name in tqdm(bev_imgs):
            bev_img = cv2.imread(str(bev_raw_path / bev_img_name), cv2.IMREAD_GRAYSCALE)
            bev_post = bev.postprocess(bev_img, bev_cls[n_classes],
                                        n_classes, morph=True)
            cv2.imwrite(str(save_bev / bev_img_name).replace('.jpg', '.png'), bev_post)


#DATASET_PATH = Path("/media/aisyslab/BICHO/Datasets/Dan-2023-Front2BEV/")
DATASET_PATH = Path("/home/aircv1/Data/Luis/aisyslab/Daniel/Dan-2023-Front2BEV/")

def main():
    test_paths = get_test_dirs(DATASET_PATH)
    for n_cls in range(2, 7):
        print('\n Class:', n_cls)
        post_proc_bev(test_paths, n_cls)
    
if __name__ == '__main__':
    main()
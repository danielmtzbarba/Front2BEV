from pathlib import Path
from tqdm import tqdm

import cv2
from PIL import Image

from dan.utils import make_folder
from dan.utils.data import get_filenames_list

from src.utils import get_test_dirs
import src.data.front2bev.bev as bev
from src.data.front2bev.bev_classes import bev_cls
from src.data.front2bev.utils import mask64
from src.utils.visualize import *


def proc_dir(path, num_class, size):
    save_bev = make_folder(path, f"{num_class}k")
    print("Saving in: ", save_bev)
    bev_raw_path = path / "sem"
    bev_imgs = get_filenames_list(bev_raw_path, ".jpg")

    for bev_img_name in tqdm(bev_imgs):

        bev_img = cv2.imread(str(bev_raw_path / bev_img_name), cv2.IMREAD_GRAYSCALE)
        bev_post = bev.postprocess(bev_img, bev_cls[num_class], size, 
                    bev.resize_img(mask64, size), num_class, morph=True)
        
        encoded_masks = encode_binary_labels(bev_post)
        Image.fromarray(encoded_masks.astype(np.int32), mode='I').save(str(save_bev / bev_img_name).replace('.jpg', '.png'))

def post_proc_bev(test_paths, num_class, size):
    # Create BEV gt for each dire
    for test_path in tqdm(test_paths):
        bev_raw_path = test_path / "bev"     
        proc_dir(bev_raw_path, num_class, size)
        

SIZE = (64, 64)
DATASET_PATH = Path("/media/danielmtz/data/datasets/Dan-2023-Front2BEV/")

#DATASET_PATH = Path("/home/aircv1/Data/Luis/aisyslab/Daniel/Dan-2023-Front2BEV/")

if __name__ == '__main__':
    test_paths = get_test_dirs(DATASET_PATH)
    for n_cls in range(2, 7):
        print('\n Class:', n_cls)
        post_proc_bev(test_paths, n_cls, SIZE)
        
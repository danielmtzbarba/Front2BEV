import cv2
import numpy as np
from pathlib import Path

from tqdm import tqdm

from dan.utils import make_folder

from dan.utils.data import get_filenames_list
from utils import get_test_dirs
from utils.bev import postprocess

bev_map = {
    0:   0, # Espacio Ocupado
    90:  1, # Espacio Libre
    120: 2, # Banqueta prro
    16:  2, # Vehiculos
    190: 3, # Lineas carril
     1:  4,
}


def remap_segmentation(img):
    h = img.shape[0]
    w = img.shape[1]

    # loop over the image
    for y in range(0, h):
        for x in range(0, w):
            if img.item(y, x) == 0:
                continue
            elif img.item(y, x) == 90:
                img[y, x] = 1
            elif img.item(y, x) == 16:
                img[y, x] =  2
            elif img.item(y, x) == 190:
                img[y, x] = 3
            else:
                img[y, x] = 4

    img[30:35, 32] = 5
    
    return img

MASK = cv2.imread(str(Path("utils") / "binary_bev_mask.jpg"), 0).astype(np.uint8)


def main(dataset_path):

    test_paths = get_test_dirs(dataset_path)
    for test_path in tqdm(test_paths):

        bev_raw_path = test_path / "bev"
        save_bev = make_folder(test_path, "bev2")

        bev_imgs = get_filenames_list(bev_raw_path, ".jpg")
        for bev_img_name in tqdm(bev_imgs):
            bev_img = cv2.imread(str(bev_raw_path / bev_img_name), cv2.IMREAD_GRAYSCALE)
            bev_post = postprocess(bev_img)
            cv2.imwrite(str(save_bev / bev_img_name).replace('.jpg', '.png'), bev_post)

if __name__ == '__main__':
    root_path = Path("/media/aisyslab/ADATA HD710M PRO/DATASETS/Front2BEV/")
    main(root_path)
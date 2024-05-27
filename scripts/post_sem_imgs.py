from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image

from dan.utils import make_folder
from dan.utils.data import get_filenames_list

from src.utils import get_test_dirs
import src.data.front2bev.bev as bev
from src.data.front2bev.bev_classes import bev_cls
from src.data.front2bev.utils import mask64
from src.data.utils import encode_binary_labels
from src.utils.visualize import plot_class_masks, plot_encoded_masks
import matplotlib.pyplot as plt

def process_img(bev_img, num_class, save_path):
    bev_post = bev.postprocess(bev_img, bev_cls[num_class], size, 
                bev.resize_img(mask64, size), num_class, morph=True)
    
    encoded_masks = encode_binary_labels(bev_post.transpose((2, 1, 0))).transpose()
    Image.fromarray(encoded_masks.astype(np.int32), mode='I').save(str(save_bev / bev_img_name).replace('.jpg', '.png'))

    if False:
        plot_encoded_masks(encoded_masks)
        plt.show()
        break

def process_dir(path, num_class, size):

    bev_imgs = get_filenames_list(bev_raw_path, ".jpg")

    for bev_img_name in tqdm(bev_imgs):

        bev_img = cv2.imread(str(bev_raw_path / bev_img_name), cv2.IMREAD_GRAYSCALE)




SIZE = (200, 196)
DATASET_PATH = Path("/media/dan/data/datasets/Dan-2024-Front2BEV/")

#DATASET_PATH = Path("/home/aircv1/Data/Luis/aisyslab/Daniel/Datasets/Dan-2023-Front2BEV/")

if __name__ == '__main__':
    test_paths = get_test_dirs(DATASET_PATH)
    for num_class in range(2, 7):
        print('\n Class:', num_class)

        for test_path in tqdm(test_paths):
            bev_raw_path = test_path / "bev"     
            # Create ground truth dir
            save_bev = make_folder(bev_raw_path, f"{num_class}k")
            print("Saving in: ", save_bev)
            bev_raw_path = save_bev / "sem"

#            process_dir(bev_raw_path, num_class, SIZE)

        

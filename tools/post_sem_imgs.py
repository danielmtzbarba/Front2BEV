from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image

from dan.utils import make_folder
from dan.utils.data import get_filenames_list

from src.utils import get_test_dirs
import src.data.front2bev.bev as bev
from src.data.front2bev.bev_classes import bev_cls, bev_cls_real
from src.data.front2bev.utils import mask64
from src.data.utils import encode_binary_labels
from src.utils.visualize import plot_class_masks, plot_encoded_masks
import matplotlib.pyplot as plt


def proc_dir(path, num_class, size, display=False):
    save_bev = make_folder(path, f"{num_class}k")
    print("Saving in: ", save_bev)
    bev_raw_path = path / "sem"
    bev_imgs = get_filenames_list(bev_raw_path, ".jpg")

    pixel_count_total = {
        key: value for (key, value) in enumerate([0 for _ in range(num_class)])
    }
    pixel_count_total["N"] = 0

    for bev_img_name in tqdm(bev_imgs):
        bev_sem = cv2.imread(str(bev_raw_path / bev_img_name), cv2.IMREAD_GRAYSCALE)

        _, decoded_masks, pixel_count = bev.postprocess(
            bev_sem,
            bev_cls[num_class],
            size,
            np.ones_like(bev.resize_img(mask64, size)),
            num_class,
            morph=False,
        )

        for key, value in pixel_count.items():
            pixel_count_total[key] += value

        encoded_masks = encode_binary_labels(
            decoded_masks.transpose((2, 1, 0))
        ).transpose()
        Image.fromarray(encoded_masks.astype(np.int32), mode="I").save(
            str(save_bev / bev_img_name).replace(".jpg", ".png")
        )

        if display:
            plot_encoded_masks(encoded_masks)
            plt.show()
            break

    return pixel_count_total


def post_proc_bev(test_paths, num_class, size):
    # Create BEV gt for each dire

    pixel_count_total = {
        key: value for (key, value) in enumerate([0 for _ in range(num_class)])
    }
    pixel_count_total["N"] = 0

    for test_path in tqdm(test_paths):
        bev_raw_path = test_path / "bev"
        pixel_count = proc_dir(bev_raw_path, num_class, size)

        for key, value in pixel_count.items():
            pixel_count_total[key] += value

    return pixel_count_total


SIZE = (200, 196)

#DATASET_PATH = "/run/media/dan/dan/datasets/Front2BEV-RGBD"
DATASET_PATH = "/media/aisyslab/dan/datasets/TEST-F2B-RGBD/"
if __name__ == "__main__":
    test_paths = get_test_dirs(DATASET_PATH)
    n_cls = 5
    print("\n Class:", n_cls)
    pixel_count_total = post_proc_bev(test_paths, n_cls, SIZE)
    print("\n Class:", n_cls)
    print(pixel_count_total)

import cv2
import numpy as np
from pathlib import Path
from time import sleep
import matplotlib.pyplot as plt

from dan.utils.data import get_dataset_from_path
from utils.bev import postprocess, vis_bev_img, bev_class2color, bev_color2class

MASK = cv2.imread(str(Path("utils") / "binary_bev_mask.jpg"), 0).astype(np.uint8)
from utils import mask64


def plot_im(im):
    plt.imshow(im, cmap="binary")
    plt.show()

ROOT_PATH = "E:/Datasets/Dan-2023-Front2BEV/"

MAP = 'Town10HD/'
LAYERS = 'layers_none/'

x_dir = ROOT_PATH + MAP + LAYERS + "rgb"
y_dir = ROOT_PATH + MAP + LAYERS + "bev"

_, bev_img_paths = get_dataset_from_path(x_dir, y_dir, '.jpg', '.jpg')

for img_path in bev_img_paths:
    bev_sem_img = cv2.imread(img_path, 0)
    bev_gt = postprocess(bev_sem_img, bev_color2class, MASK)
    vis = vis_bev_img(bev_gt, bev_class2color, mask64)

    plot_im(bev_gt)
    plot_im(vis)

    cv2.imshow('frame', vis)
    break
    sleep(0.01)
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        break
cv2.destroyAllWindows()


im = cv2.imread('124.jpg', 0)

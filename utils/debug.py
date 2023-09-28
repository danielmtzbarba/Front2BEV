import cv2
import numpy as np
from pathlib import Path
from time import sleep

from dan.utils.data import get_dataset_from_path
from utils.bev import *

MASK = cv2.imread(str(Path("utils") / "_mask1024.png"), 0).astype(np.uint8)
from utils import out_of_fov

ROOT_PATH = "D:/Datasets/Dan-2023-Front2BEV/"

MAP = 'Town10HD/'
LAYERS = 'layers_none/'

x_dir = ROOT_PATH + MAP + LAYERS + "rgb"
y_dir = ROOT_PATH + MAP + LAYERS + "bev"

_, bev_img_paths = get_dataset_from_path(x_dir, y_dir, '.jpg', '.jpg')


for img_path in bev_img_paths:
    bev_sem_img = cv2.imread(img_path, 0)
    bev_gt = postprocess(bev_sem_img, bev_color2class, MASK)
    vis = vis_bev_img(bev_gt, bev_class2color, out_of_fov)

    cv2.imshow('vis', vis)
    cv2.imshow('bev', bev_sem_img)

    sleep(0.01)
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        break
cv2.destroyAllWindows()



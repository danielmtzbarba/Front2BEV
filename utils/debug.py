import cv2
from dan.utils.data import get_dataset_from_path
from utils.bev import *

from tqdm import tqdm

ROOT_PATH = "D:/Datasets/Dan-2023-Front2BEV/"

MAP = 'Town10HD/'
LAYERS = 'layers_all/'

x_dir = ROOT_PATH + MAP + LAYERS + "rgb"
y_dir = ROOT_PATH + MAP + LAYERS + "bev"

_, bev_img_paths = get_dataset_from_path(x_dir, y_dir, '.jpg', '.jpg')

n_classes = 3

import matplotlib.pyplot as plt
from time import sleep

plt.imshow(cv2.imread('utils/1461.jpg', 0))
plt.show()

for img_path in tqdm(bev_img_paths):
    bev_sem_img = cv2.imread(img_path, 0)
    bev_gt = postprocess(bev_sem_img, bev_color2class, n_classes)
    vis = vis_bev_img(bev_gt, bev_class2color)

    cv2.imshow('vis', vis)
    cv2.imshow('bev', bev_sem_img)
    sleep(0.05)

    plt.imshow(bev_gt)
    plt.show()
    break
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        break
cv2.destroyAllWindows()



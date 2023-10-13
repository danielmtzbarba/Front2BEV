import cv2
from time import sleep

import Front2BEV.utils.bev as bev 
from Front2BEV.utils.bev_classes import bev_cls

from dan.utils.data import get_dataset_from_path

def show_gif(img_paths_list, n_classes, transform=False):
    for im_path in img_paths_list:
        frame = cv2.imread(im_path, 0)

        if transform:
            frame = bev.vis_bev_img(frame, bev_cls[n_classes])

        cv2.imshow('video',frame)
        sleep(0.05)
        if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
            break
    cv2.destroyAllWindows()

# *****************************************************
ROOT_PATH = "/media/aisyslab/BICHO/Datasets/Dan-2023-Front2BEV/"

MAP = 'Town01/'
LAYERS = 'layers_none/'
N_CLASSES = 3

x_dir = ROOT_PATH + MAP + LAYERS + "rgb"
y_dir = ROOT_PATH + MAP + LAYERS + "bev/3k"

_, bev_img_paths = get_dataset_from_path(x_dir, y_dir, '.jpg', '.png')

show_gif(bev_img_paths, N_CLASSES, True)

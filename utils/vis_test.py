import cv2
from time import sleep

from utils import out_of_fov
from utils.bev import vis_bev_img, bev_class2color

from dan.utils.data import get_dataset_from_path

def show_gif(img_paths_list, transform=False):
    for im_path in img_paths_list:
        frame = cv2.imread(im_path, 0)

        if transform:
            frame = vis_bev_img(frame, bev_class2color, out_of_fov)

        cv2.imshow('video',frame)
        sleep(0.05)
        if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
            break
    cv2.destroyAllWindows()

# *****************************************************
ROOT_PATH = "D:/Datasets/Dan-2023-Front2BEV/"

MAP = 'Town01/'
LAYERS = 'layers_none/'

x_dir = ROOT_PATH + MAP + LAYERS + "rgb"
y_dir = ROOT_PATH + MAP + LAYERS + "bev2"

_, bev_img_paths = get_dataset_from_path(x_dir, y_dir, '.jpg', '.png')

show_gif(bev_img_paths, True)

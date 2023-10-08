import matplotlib.pyplot as plt
import numpy as np
import random
import cv2

# --------------------------------------------------------------
import dan.utils.graph as graph
import Front2BEV.utils.bev as bev

from Front2BEV.utils.bev_classes import bev_cls, bev_class2color
from dan.utils.data import get_dataset_from_path
# --------------------------------------------------------------
def compare_morph(sem_img):
    bev_gt = bev.postprocess(sem_img, bev_cls[N_CLASSES],
                          N_CLASSES, morph=False)
    bev_morph = bev.postprocess(sem_img, bev_cls[N_CLASSES],
                          N_CLASSES, morph=True)
    
    bev_rgb = bev.bevAsRGB(bev_gt, N_CLASSES, bev_class2color)
    bev_morph_rgb = bev.bevAsRGB(bev_morph, N_CLASSES, bev_class2color)


    graph.compare_images([sem_img, bev_rgb, bev_morph_rgb],
                          title='BEV Morphology')
# --------------------------------------------------------------
ROOT_PATH = "/media/aisyslab/dan/Datasets/Dan-2023-Front2BEV/"

MAP = 'Town10HD/'
LAYERS = 'traffic/'

x_dir = ROOT_PATH + MAP + LAYERS + "rgb"
y_dir = ROOT_PATH + MAP + LAYERS + "bev/sem"

_, bev_img_paths = get_dataset_from_path(x_dir, y_dir, '.jpg', '.jpg')

N_CLASSES = 5
# --------------------------------------------------------------
rand_img_path = random.choice(bev_img_paths)

img_path = y_dir + '/201.jpg'
bev_sem_img = cv2.imread(rand_img_path, 0)

compare_morph(bev_sem_img)

plt.show()

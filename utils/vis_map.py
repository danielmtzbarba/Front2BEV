import numpy as np
import cv2

color_list = np.array([
    [0],
    [255],
    [100],
    [200],  
    [128],
    [75]
    ], dtype=np.uint8)


def vis_bev_img(bev_map):
    MASK = cv2.imread('binary_bev_mask.jpg', 0)
    masked = cv2.bitwise_and(bev_map,bev_map, mask = MASK)
    for i, color in enumerate(color_list):
        masked[masked == i] = color
    return cv2.resize(bev_map, (512,512), interpolation=cv2.INTER_AREA)

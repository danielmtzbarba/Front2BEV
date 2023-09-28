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

DIM_BEV_OUT = (64, 64)

bev_color2class = {
    0:   0, # Non-Free
    50: 0, # Vehicle
    90:  1, # Free space
    190: 1, # Road Lines
    120: 2, # Banqueta prro
}

bev_class2color = {
    0:  50, # Non-Free
    1:  255, # Free space
    2: 128, # Road Lines
}


def mask_img(img, mask):
    return cv2.bitwise_and(img, img, mask=mask)

def resize_img(img):
    return cv2.resize(img, DIM_BEV_OUT, interpolation = cv2.INTER_NEAREST)

def remap_seg(bev_img, bev_map):
    for val, klass in bev_map.items():
        bev_img[bev_img == val] = klass
    bev_img[bev_img > 10] = 0
    return bev_img

def postprocess(img, bev_map, mask):
    masked = mask_img(img, mask)
    resized = resize_img(masked)
    segmented = remap_seg(resized, bev_map)
    return segmented

def vis_bev_img(bev_im, bev_map, mask):
    masked = bev_im.copy()
    for klass, color in bev_map.items():
        masked[bev_im == klass] = color
    masked = mask_img(masked, mask)

  #  bev_vis = masked[OUT_OF_FOV] = 0
    return cv2.resize(masked, (512,512), interpolation=cv2.INTER_AREA)

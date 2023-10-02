from pathlib import Path
import numpy as np
import cv2

DIM_BEV_OUT = (64, 64)

bev_color2class = {
    0:   0, # Non-Free
    50: 0, # Vehicle
    90:  1, # Free space
    190: 1, # Road Lines
    127: 1, #idk
    119: 1, # greens
    78: 1, # Cable
    153: 1, # Post
    178: 1, #TrafficLight
    120: 2, # Banqueta prro
    33: 2, # islands

}

bev_class2color = {
    0:  50, # Non-Free
    1:  255, # Free space
    2: 128, # Road Lines
}

def setup_mask():
    mask_1024 = cv2.imread(str(Path("__assets") / "_mask1024.png"), 0)
    mask64 = cv2.imread(str(Path("__assets") / "_mask64.png"), 0)
    mask = np.zeros_like(mask64)
    fov = mask64 > 128
    mask[fov] = 1
    out_of_fov = mask == 0
    return mask_1024, mask, fov, out_of_fov

mask_1024, mask64, fov, out_of_fov = setup_mask()

def resize_img(img):
    return cv2.resize(img, DIM_BEV_OUT, interpolation = cv2.INTER_NEAREST)

def remap_seg(bev_img, bev_mapping, n_classes):
    for val, klass in bev_mapping.items():
        bev_img[bev_img == val] = klass
    bev_img[bev_img > n_classes] = 0
    return bev_img

def postprocess(img, bev_map, n_classes):
    resized = resize_img(img)
    segmented = remap_seg(resized, bev_map, n_classes)
    segmented[30:33,32] = 1
    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    morph = cv2.morphologyEx(segmented, cv2.MORPH_CLOSE, kernel)
    return morph

def vis_bev_img(bev_im, bev_map):
    bev_im = bev_im.copy()
    bev_im[out_of_fov] = 10
    for klass, color in bev_map.items():
        bev_im[bev_im == klass] = color
    bev_im[out_of_fov] = 0

    return cv2.resize(bev_im, (512,512), interpolation=cv2.INTER_AREA)
from pathlib import Path
import numpy as np
import cv2

# --------------------------------------------------------------------------
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
# --------------------------------------------------------------------------
def setup_mask():
    mask64 = cv2.imread(str(Path("__assets") / "_mask64.png"), 0)
    mask = np.zeros_like(mask64)
    fov = mask64 > 128
    mask[fov] = 1
    return mask

mask64 = setup_mask()

# --------------------------------------------------------------------------
def get_FOV_pixels(img, mask, n_classes):
    masked_image = mask_img(img, mask64, n_classes)
    fov_mask = mask.reshape(-1) != 0
    fov_index = masked_image.reshape(-1) != n_classes
    fov_index = fov_index * fov_mask
    fov_pixels = img.reshape(-1)[fov_index]
    fov_pixels = fov_pixels.reshape(1, -1)
    return fov_pixels

def resize_img(img):
    return cv2.resize(img, (64, 64),
                       interpolation = cv2.INTER_NEAREST)

def remap_seg(bev_img, bev_mapping, n_classes):
    bev_img = bev_img.copy()
    for val, klass in bev_mapping.items():
        bev_img[bev_img == val] = klass
    bev_img[bev_img > n_classes] = 0
    return bev_img

def mask_img(img, mask64, n_classes):
    img = img.copy()
    img[30:33,32] = 1
    out_of_fov = (mask64 == 0)
    img *=  mask64
    img[out_of_fov] = n_classes
    return img

def apply_morph(img, kernel_size = 2):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                        (kernel_size,kernel_size))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def postprocess(img, bev_map, n_classes):
    segmented = remap_seg(img, bev_map, n_classes)
    resized = resize_img(segmented)
   # morphed = apply_morph(resized, 2)
    return mask_img(resized, mask64, n_classes)

def vis_bev_img(bev_im, bev_map):
    bev_im = bev_im.copy()
    out_of_fov_mask = (mask64 == 0)
    bev_im[out_of_fov_mask] = 10
    for klass, color in bev_map.items():
        bev_im[bev_im == klass] = color
    bev_im[out_of_fov_mask] = 0

    return cv2.resize(bev_im, (512,512), interpolation=cv2.INTER_AREA)

# --------------------------------------------------------------------------
def count_pixels(img, count_dict, n_classes, only_fov=False):
    if only_fov:
        img = get_FOV_pixels(img, mask64, n_classes)

    clss, counts = np.unique(img, return_counts=True)
    for cls, count in zip(clss, counts):
        count_dict[cls] += count

    n_pixels = len(img.flatten())
    return count_dict, n_pixels
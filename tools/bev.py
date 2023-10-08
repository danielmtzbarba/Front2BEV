from pathlib import Path
import numpy as np
import cv2
# --------------------------------------------------------------------------
def setup_mask():
    mask64 = cv2.imread(str(Path("_assets") / "_mask64.png"), 0)
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

def dilated_class(sem_img, bev_img, cmap,
                   k= 3, i=3, morph=False):
    bev_img = bev_img.copy()
    cls_mask = sem_img.copy()
    cls_mask = (sem_img == cmap[0]).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                        (k, k))
    cls_mask = cv2.resize(cv2.dilate(cls_mask, kernel,
                          iterations = i), (64,64))
    if morph:
        cls_mask = apply_morph(cls_mask, 2)

    bev_img[cls_mask == 1] = cmap[1]
    return bev_img

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

def postprocess(sem_img, bev_map, n_classes, morph=False):
    remapped = remap_seg(sem_img, bev_map, n_classes)
    bev_img = resize_img(remapped)

    if morph:
        bev_img = apply_morph(bev_img, 2)

        if n_classes > 5:
            bev_img = dilated_class(sem_img, bev_img,
                                     [190, 5], k=5, i=3,
                                     morph=True)
        if n_classes > 4:
            bev_img = dilated_class(sem_img, bev_img,
                                     [84, 4], k=3, i=3)
            return mask_img(bev_img, mask64, n_classes)

    return mask_img(bev_img, mask64, n_classes)

def bevAsRGB(bev_img, n_classes, cmap):
    bev_img = bev_img.copy()
    bev_rgb = np.stack((bev_img,)*3, axis=-1)
    for cl in range(n_classes):
        bev_rgb[bev_img == cl, :] = cmap[cl]
    bev_rgb[bev_img == n_classes, :] = (0, 0, 0)
    bev_rgb[31, 32, :] = (25, 126, 0)
    return bev_rgb

# --------------------------------------------------------------------------
def count_pixels(img, count_dict, n_classes, only_fov=False):
    if only_fov:
        img = get_FOV_pixels(img, mask64, n_classes)

    clss, counts = np.unique(img, return_counts=True)
    for cls, count in zip(clss, counts):
        count_dict[cls] += count

    n_pixels = len(img.flatten())
    return count_dict, n_pixels
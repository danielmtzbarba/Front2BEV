# --------------------------------------------------------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt
import src.utils.visualize as vis

from src.data.utils import encode_binary_labels
# --------------------------------------------------------------------------

def resize_img(img, size):
    return cv2.resize(img, size,
                       interpolation = cv2.INTER_NEAREST)

def dilated_class(sem_img, bev_img, cmap, size,
                   k= 3, i=3, morph=True):
    bev_img = bev_img.copy()
    cls_mask = sem_img.copy()
    cls_mask = (sem_img == cmap[0]).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                        (k, k))
    cls_mask = cv2.resize(cv2.dilate(cls_mask, kernel,
                          iterations = i), size)
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
    out_of_fov = (mask64 == 0)
    img *=  mask64
    img[out_of_fov] = n_classes
    return img

def apply_morph(img, kernel_size = 2):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                        (kernel_size,kernel_size))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def decode_masks(encoded_bev, n_classes, fov_mask):
    w, h = encoded_bev.shape
    pixel_count = {'N': w*h}
    masks = np.zeros((w, h, n_classes + 1))
    for k in range(n_classes):
        k_mask = encoded_bev == k
        masks[:,:, k] = k_mask 
        pixel_count[k] = np.count_nonzero(k_mask) 
    masks[:, :, n_classes] = fov_mask
    return masks, pixel_count

def postprocess(sem_img, bev_map, size, fov_mask, n_classes, morph=True, display=False):
    remapped = remap_seg(sem_img, bev_map, n_classes)
    resized = resize_img(remapped, size)

    if morph:
        eroded = apply_morph(resized, 2)

        if n_classes > 5:
            bev_img = dilated_class(sem_img, eroded, [190, 5],
                                    size, k=5, i=3,  morph=True)
        if n_classes > 4:
            bev_img = dilated_class(sem_img, eroded,  [84, 4],
                                    size, k=3, i=3, morph=True)
        else:
            bev_img = eroded

        masks, pixel_count  =  decode_masks(bev_img, n_classes, fov_mask)

    else:
        bev_img = resized
        masks, pixel_count =  decode_masks(bev_img, n_classes, fov_mask)
    
    if display:
       encoded_masks = encode_binary_labels(masks.transpose((2, 1, 0))).transpose()
       vis.plot_post_pipeline([sem_img, remapped, resized, eroded,  bev_img], fov_mask)
       vis.plot_class_masks(masks, fov_mask) 
       vis.plot_img_list(masks)
       vis.plot_encoded_masks(encoded_masks)
       plt.show()
         
    return bev_img, masks, pixel_count 
# --------------------------------------------------------------------------


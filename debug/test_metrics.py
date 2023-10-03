from Front2BEV.utils.bev import postprocess, bev_color2class
from Front2BEV.utils.eval import metric_eval_bev

import cv2

N_CLASSES = 3

bev_nn = postprocess(cv2.imread('__assets/1461.jpg', 0), bev_color2class, N_CLASSES)
bev_gt = cv2.imread('/media/aisyslab/BICHO/Datasets/Dan-2023-Front2BEV/Town10HD/layers_all/bev/3k/1461.png', 0)

print(metric_eval_bev(bev_nn, bev_gt, N_CLASSES))
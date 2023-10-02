import numpy as np


import models.VAE.py_img_seg_eval.eval_segm as eval_segm

from utils.bev import mask64


def metric_eval(current_nn, current_gt, n_classes):

    current_gt = current_gt.cpu().numpy().squeeze()
    current_nn = np.reshape(np.argmax(current_nn.cpu().numpy().transpose((0, 2, 3, 1)), axis=3), [64, 64])

    FOVmsk = mask64.copy()

    valid_FOV_index = FOVmsk.reshape(-1) != 0

    valid_index = current_gt.reshape(-1) != n_classes
    valid_index = valid_index * valid_FOV_index

    current_gt = current_gt.reshape(-1)[valid_index]
    current_nn = current_nn.reshape(-1)[valid_index]

    current_gt = current_gt.reshape(1, -1)
    current_nn = current_nn.reshape(1, -1)

    # eval_segm.pixel_accuracy(current_nn, current_gt)
    acc = eval_segm.mean_accuracy(current_nn, current_gt)
    iou = eval_segm.mean_IU(current_nn, current_gt)
    # eval_segm.frequency_weighted_IU(current_nn, current_gt)
    return acc, iou
import Front2BEV.utils.segm_metrics as eval_segm

import Front2BEV.tools.bev as bev

def metric_eval_bev(bev_nn, bev_gt, n_classes):

    fov_pixels_nn = bev.get_FOV_pixels(bev_nn, bev.mask64, n_classes)
    fov_pixels_gt = bev.get_FOV_pixels(bev_gt, bev.mask64, n_classes)

    acc = eval_segm.mean_accuracy(fov_pixels_nn, fov_pixels_gt)
    iou = eval_segm.mean_IU(fov_pixels_nn, fov_pixels_gt)
    
    return acc, iou
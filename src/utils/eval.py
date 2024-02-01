import src.utils.metrics as metrics

import src.data.front2bev.bev as bev

def metric_eval_bev(bev_nn, bev_gt, n_classes):

    fov_pixels_nn = bev.get_FOV_pixels(bev_nn, bev.mask64, n_classes)
    fov_pixels_gt = bev.get_FOV_pixels(bev_gt, bev.mask64, n_classes)

    acc = metrics.mean_accuracy(fov_pixels_nn, fov_pixels_gt)
    iou = metrics.mean_IU(fov_pixels_nn, fov_pixels_gt)
    
    return acc, iou

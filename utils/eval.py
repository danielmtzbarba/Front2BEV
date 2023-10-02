import Front2BEV.utils.segm_metrics as eval_segm

from Front2BEV.utils.bev import mask64, get_FOV_pixels

def metric_eval_bev(bev_nn, bev_gt, n_classes):

    fov_pixels_nn = get_FOV_pixels(bev_nn, mask64, n_classes)
    fov_pixels_gt = get_FOV_pixels(bev_gt, mask64, n_classes)

    acc = eval_segm.mean_accuracy(fov_pixels_nn, fov_pixels_gt)
    iou = eval_segm.mean_IU(fov_pixels_nn, fov_pixels_gt)
    # eval_segm.frequency_weighted_IU(current_nn, current_gt)
    return acc, iou
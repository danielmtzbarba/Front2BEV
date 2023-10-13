from tqdm import tqdm
import numpy as np
import torch

from models.VAE import VAE

from dan.utils.torch import load_model
from Front2BEV.utils.eval import metric_eval_bev

def test_model(args):
    model = VAE(k_classes=args.n_classes)
    model = load_model(model, args.ckpt_path)
    model = model.to(0)

    # Set model to evaluate mode
    model.eval()  

    acc = 0.0
    iou = 0.0

    # Iterate over data.
    for temp_batch in tqdm(args.test_loader):
        batch_rgb = temp_batch['rgb'].float().to(0)
        batch_map_gt = temp_batch['map'].long().to(0)

        # forward
        with torch.set_grad_enabled(False):
            pred_map, _, _ = model(batch_rgb, False)
            bev_gt = batch_map_gt.cpu().numpy().squeeze()
            bev_nn = np.reshape(
                        np.argmax(pred_map.cpu().numpy().transpose((0, 2, 3, 1)),
                                    axis=3), [64, 64])
            
            temp_acc, temp_iou = metric_eval_bev(bev_nn, bev_gt, args.n_classes)
            acc += temp_acc
            iou += temp_iou
    
    print("\nTest acc: ", acc / len(args.test_loader))
    print("\nTest mIoU: ", iou / len(args.test_loader))
    
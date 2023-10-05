from tqdm import tqdm
import numpy as np
import torch

from models.VAE.data_loader import *
from models.VAE.vae_nets import vae_mapping

from dan.utils.torch import load_model
from Front2BEV.utils.eval import metric_eval_bev

def test_model(args):
    model = vae_mapping(k_classes=args.n_classes)
    model = load_model(model, args.ckpt_path)
    model = model.to(args.device)

    # Set model to evaluate mode
    model.eval()  

    acc = 0.0
    iou = 0.0

    # Iterate over data.
    for temp_batch in tqdm(args.dataloaders['test']):
        batch_rgb = temp_batch['rgb'].float().to(args.device)
        batch_map_gt = temp_batch['map'].long().to(args.device)

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
    
    print("\nTest acc: ", acc / len(args.dataloaders["test"]))
    print("\nTest mIoU: ", iou / len(args.dataloaders["test"]))
    
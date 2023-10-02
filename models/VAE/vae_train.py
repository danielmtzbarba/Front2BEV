import os
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from models.VAE.data_loader import *
from models.VAE.vae_nets import *

from utils.eval import metric_eval_bev
from dan.utils import save_pkl_file

def loss_function_map(pred_map, map, mu, logvar, args):
    if args.class_weights is not None:
        args.class_weights = torch.Tensor(args.class_weights).to(args.device)

    if args.ignore_class:
        CE = F.cross_entropy(pred_map, map.view(-1, 64, 64), weight=args.class_weights, ignore_index=args.n_classes)
    else:
        CE = F.cross_entropy(pred_map, map.view(-1, 64, 64), weight=args.class_weights)

    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return 0.9*CE + 0.1*KLD, CE, KLD

def train_model(args):

    log_batch = {
        'loss': [],
        'CE_loss': [],
        'KLD_loss': [],
    }

    log_epoch = {
        'epoch': [],
        'mean_train_loss': [],
        'mean_val_loss': [],
        'val_acc': [],
        'val_iou': [],
    }

    model = vae_mapping(k_classes=args.n_classes)
    model = model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)

    if args.restore_ckpt:
        if os.path.isfile(args.ckpt_path):
            checkpoint = torch.load(args.ckpt_path)
            epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            epoch = 0
    else:
        epoch = 0

    while epoch < args.n_epochs:

        log_epoch['epoch'].append(epoch)

        print('\nEpoch {}/{}'.format(epoch, args.n_epochs - 1))
        print('-' * 50)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            acc = 0.0
            iou = 0.0

            # Iterate over data.
            for batch in tqdm(args.dataloaders[phase]):
                batch_rgb = batch['rgb'].float().to(args.device)
                batch_map_gt = batch['map'].long().to(args.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward: Track history only if training
                with torch.set_grad_enabled(phase == 'train'):
                    pred_map, mu, logvar = model(batch_rgb, phase == 'train')

                    loss, CE, KLD = loss_function_map(pred_map, batch_map_gt, mu, logvar, args)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    else:
                        # Validation
                        bev_gt = batch_map_gt.cpu().numpy().squeeze()
                        bev_nn = np.reshape(
                                    np.argmax(pred_map.cpu().numpy().transpose((0, 2, 3, 1)),
                                                axis=3), [64, 64])
                    
                        temp_acc, temp_iou = metric_eval_bev(bev_nn, bev_gt, args.n_classes)
                        acc += temp_acc
                        iou += temp_iou

                running_loss += loss.item()

                # ------------------------------
                # Logging per batch
                # ------------------------------
                if phase == 'train':
                    log_batch['loss'].append(loss.item())
                    log_batch['CE_loss'].append(CE.item())
                    log_batch['KLD_loss'].append(KLD.item())
                # ------------------------------

            # ------------------------------
            # Logging per epoch
            # ------------------------------
            if phase == 'train':
                running_loss = running_loss / len(args.dataloaders["train"])
                log_epoch['mean_train_loss'].append(running_loss)
                print("\nEpoch:", epoch, "Train loss (mean):", running_loss)
                print('-' * 50)

            else:
                running_loss = running_loss / len(args.dataloaders["val"])
                log_epoch['mean_val_loss'].append(running_loss)
                print("\nEpoch:", epoch, "Val loss (mean):", running_loss)
                print('-' * 50)

        # ------------------------------
        # Logging metrics and save model
        # ------------------------------

        log_epoch['val_acc'].append(acc / len(args.dataloaders["val"]))
        print("Val acc: ", acc / len(args.dataloaders["val"]))

        log_epoch['val_iou'].append(iou / len(args.dataloaders["val"])) 
        print("Val mIoU: ", iou / len(args.dataloaders["val"]))
        print('-' * 50)


        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
            }, args.ckpt_path)
        
        print('\nModel saved at epoch', (epoch+1))
        print('-' * 50)
        epoch += 1

        log_dict = {
            'batches':log_batch,
            'epochs': log_epoch
        }    

        save_pkl_file(log_dict, args.log_path)
        
    # --------------------------
    print('\nTraining ended')
    # --------------------------
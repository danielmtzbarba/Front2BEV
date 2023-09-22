import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms

from VAE.data_loader import *
from VAE.vae_nets import *
from VAE.util import metric_eval

from dan.utils.torch import get_torch_device

seed = 8964
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def train_model(device, dataloaders, n_epochs, n_classes,
                ckpt_path = None, restore_ckpt=False):
    
    model = vae_mapping(k_classes=n_classes)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)

    if restore_ckpt:
        if os.path.isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            epoch = 0
    else:
        epoch = 0


    while epoch < n_epochs:
        print('Epoch {}/{}'.format(epoch, n_epochs - 1))
        print('-' * 10)

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
            for temp_batch in tqdm(dataloaders[phase]):
                temp_rgb = temp_batch['rgb'].float().to(device)
                temp_map = temp_batch['map'].long().to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    pred_map, mu, logvar = model(temp_rgb, phase == 'train')
                    loss, CE, KLD = loss_function_map(pred_map, temp_map, mu, logvar)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        temp_acc, temp_iou = metric_eval(pred_map, temp_map)
                        acc += temp_acc
                        iou += temp_iou

                running_loss += loss.item()

                # tensorboardX logging
            if phase == 'train':
                 pass
                 #   writer.add_scalar(phase+'_loss', loss.item(), epoch * len(train_set) / batch_size + i)
                 #   writer.add_scalar(phase+'_loss_CE', CE.item(), epoch * len(train_set) / batch_size + i)
                 #   writer.add_scalar(phase+'_loss_KLD', KLD.item(), epoch * len(train_set) / batch_size + i)
                

                # statistics
            if phase == 'train':
                running_loss = running_loss / len(dataloaders["train"])
                print(phase, running_loss)
            else:
                running_loss = running_loss / len(dataloaders["val"])
                print(phase, running_loss, acc / len(dataloaders["val"]), iou / len(dataloaders["val"]))
            #    writer.add_scalar(phase+'_acc', acc.item()/len(val_set), (epoch + 1) * len(train_set) / batch_size)
            #    writer.add_scalar(phase+'_iou', iou.item()/len(val_set), (epoch + 1) * len(train_set) / batch_size)



        # save model per epoch
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
            }, ckpt_path)
        print('model after %d epoch saved...' % (epoch+1))
        epoch += 1

    # writer.close()

if __name__ == '__main__':
    n_epochs = 5
    batch_size = 1
    n_workers = 1

    # Use train set for choosing hyper-parameters, and use train+val for final traning and testing
    train_csv_path = 'dataset/Cityscapes/CS_train_64.csv'
    train_plus_val_csv_path = 'dataset/Cityscapes/CS_trainplusval_64.csv'
    val_csv_path = 'dataset/Cityscapes/CS_val_64.csv'

    restore_ckpt = False
    ckpt_path = '__checkpoints/vae_checkpoint_2.pth.tar'

    device = get_torch_device()

    train_model(device, batch_size, n_workers, n_epochs, train_csv_path, val_csv_path, ckpt_path, restore_ckpt=False)
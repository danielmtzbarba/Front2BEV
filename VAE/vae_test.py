import io
import torch

import numpy as np
from torchvision import transforms

from VAE.data_loader import *
from VAE.vae_nets import vae_mapping
from VAE.util import vis_with_FOVmsk

from dan.utils import make_folder
from dan.torch_utils import get_torch_device, load_model

def test_model(device, ckpt_path, datset_csv_path, batch_size, output_path):

    # Define dataloaders
    test_set = OccMapDataset(datset_csv_path, transform=transforms.Compose([Rescale((256, 512)), ToTensor()]))
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = vae_mapping()
    model = load_model(model, ckpt_path)
    model = model.to(device)

    # Set model to evaluate mode
    model.eval()  

    # Iterate over data.
    for i, temp_batch in enumerate(test_loader):
        print('Test Sample no. ', i)
        temp_rgb = temp_batch['rgb'].float().to(device)

        # forward
        with torch.set_grad_enabled(False):
            pred_map, mu, logvar = model(temp_rgb, False)

            output_pred = np.reshape(np.argmax(pred_map.cpu().numpy().transpose((0, 2, 3, 1)), axis=3), [64, 64]).astype(np.uint8)
            io.imsave(output_path / f'{i}_pred.png', output_pred)
            io.imsave(output_path / "vis" / f'{i}_pred_vis.png', vis_with_FOVmsk(output_pred))
    
    
if __name__ == '__main__':
    #
    device = get_torch_device()

    CKPT_PATH = 'VAE/__checkpoints/vae_checkpoint.pth.tar'
    DATASET_CSV_PATH = 'dataset/Cityscapes/test_vae.csv'
    BATCH_SIZE = 1

    TEST_DIR = "TEST"
    OUTPUT_PATH = make_folder("__results", TEST_DIR)
    make_folder(OUTPUT_PATH / "vis")

    test_model(device, CKPT_PATH, DATASET_CSV_PATH, BATCH_SIZE, OUTPUT_PATH)
    #
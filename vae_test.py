import torch

import numpy as np
from torchvision import transforms
import os

from VAE.data_loader import *
from VAE.vae_nets import *
from VAE.util import vis_with_FOVmsk


from dan.utils import make_folder

TEST_DIR = "TEST"

RESULTS_DIR = make_folder("__results",TEST_DIR)

checkpoint_path = 'VAE/__checkpoints/vae_checkpoint.pth.tar'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define dataloaders
test_set = OccMapDataset('dataset/test_vae.csv', transform=transforms.Compose([Rescale((256, 512)), ToTensor()]))
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

model = vae_mapping()
model = model.to(device)

if os.path.isfile(checkpoint_path):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    print('trained model loaded...')
else:
    print('cannot load trained model...')
    exit()

model.eval()  # Set model to evaluate mode



# Iterate over data.
for i, temp_batch in enumerate(test_loader):
    print('example no. ', i)
    temp_rgb = temp_batch['rgb'].float().to(device)

    # forward
    # track history if only in train
    with torch.set_grad_enabled(False):
        pred_map, mu, logvar = model(temp_rgb, False)

        map_to_save = np.reshape(np.argmax(pred_map.cpu().numpy().transpose((0, 2, 3, 1)), axis=3), [64, 64]).astype(np.uint8)
        io.imsave(RESULTS_DIR / f'{i}_nn_pred_c.png', vis_with_FOVmsk(map_to_save))
      #  io.imsave(map_list[i][:-4] + '_nn_pred.png', map_to_save)
      # io.imsave(map_list[i][:-4] + '_nn_pred_c.png', vis_with_FOVmsk(map_to_save))

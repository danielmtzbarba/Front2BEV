import warnings
warnings.filterwarnings("ignore")

import cv2
from time import sleep

from utils.front2bev import Front2BEV
from utils.bev import vis_bev_img, bev_class2color
from models.VAE.vae_nets import vae_mapping

from dan.utils.data import get_dataset_from_path

from torchvision.transforms import ToTensor
from dan.utils.torch.transforms import Transforms, Normalize, Rescale

# -------------------------------------------
TEST_NAME = "F2B_3K_VAE"
ckpt_path = f'E:/Checkpoints/Dan-2023-Front2BEV/{TEST_NAME}.pth.tar'
dataset_root_path = "E:/Datasets/Dan-2023-Front2BEV/"
# -------------------------------------------

# -------------------------------------------
MAP = 'Town10HD/'
LAYERS = 'layers_all/'
# -------------------------------------------
x_dir = dataset_root_path + MAP + LAYERS + "rgb"
y_dir = dataset_root_path + MAP + LAYERS + "bev/3k"
# -------------------------------------------

front_img_paths, _ = get_dataset_from_path(x_dir, y_dir, '.jpg', '.png')

model = vae_mapping(k_classes=3)
transforms = Transforms([Rescale((256, 512)), ToTensor(), Normalize()])

m = Front2BEV(model, ckpt_path)

for img_path in front_img_paths:
    front_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    front_img = transforms(front_img)
    bev = m.transform(front_img)
    vis = vis_bev_img(bev, bev_class2color)
    cv2.imshow('frame', vis)

    sleep(0.01)
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        break
cv2.destroyAllWindows()

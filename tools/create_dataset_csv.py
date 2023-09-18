from dan.tools.dataset2CSV import dataset2CSV

"""
ROOT_PATH = "C:\\Users\\Danie\\OneDrive\\dan\\RESEARCH\\DATASETS\\"
Y_DIR = ROOT_PATH + "Lu-2019-Monocular2BEV\\Cityscapes\\val\\frankfurt\\"
X_DIR = ROOT_PATH + "Cityscapes\\leftImg8bit\\val\\frankfurt\\"
dataset2CSV("test.csv", X_DIR, Y_DIR, "_leftImg8bit.png", "_occ_map.png")
"""

#X_DIR = "C:\\Users\\Danie\\OneDrive\\dan\\RESEARCH\\DATASETS\\Dan-2023-CarlaBEV\\TOWN01\\rgb"
#Y_DIR = "C:\\Users\\Danie\\OneDrive\\dan\\RESEARCH\\DATASETS\\Dan-2023-CarlaBEV\\TOWN01\\bev"

ROOT_PATH = "/home/aircv1/Data/Luis/aisyslab/Daniel/Datasets/Dan-2023-CarlaBEV/TOWN01"
X_DIR = ROOT_PATH + "rgb"
Y_DIR = ROOT_PATH + "bev"

dataset2CSV("bev-vae-test.csv", X_DIR, Y_DIR, ".jpg", ".png")
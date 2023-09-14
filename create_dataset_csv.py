from dan.tools.dataset2CSV import dataset2CSV

ROOT_PATH = "C:\\Users\\Danie\\OneDrive\\dan\\RESEARCH\\DATASETS\\"
Y_DIR = ROOT_PATH + "Lu-2019-Monocular2BEV\\Cityscapes\\val\\frankfurt\\"
X_DIR = ROOT_PATH + "Cityscapes\\leftImg8bit\\val\\frankfurt\\"


dataset2CSV("test.csv", X_DIR, Y_DIR, "_leftImg8bit.png", "_occ_map.png")
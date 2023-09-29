from pathlib import Path
import os

from dan.tools.dataset2CSV import get_csv_datasets
from utils import replace_abs_path

dataset_data_path =  "D:/Datasets/Dan-2023-Front2BEV/"

output_csv_path = "__datasets/Dan-2023-Front2bev/front2bev.csv"

def get_test_dirs(dataset_path):
    maps = [ Path(f.path) for f in os.scandir(dataset_path) if f.is_dir() ]
    tests = []
    for map in maps:
        tests.extend([ Path(f.path) for f in os.scandir(map) if f.is_dir() ])
    
    x_dirs, y_dirs = [], []
    for test in tests:
        x_dirs.append(test / "rgb")
        y_dirs.append(test / "bev2")

    return x_dirs, y_dirs

x_dirs, y_dirs = get_test_dirs(dataset_data_path)

train_csv_path, val_csv_path, test_csv_path = get_csv_datasets(output_csv_path, x_dirs, y_dirs)

old_path = r"D:\\Datasets\\"

replace_abs_path(train_csv_path, old_path, "")
replace_abs_path(val_csv_path, old_path, "")
replace_abs_path(test_csv_path, old_path, "")


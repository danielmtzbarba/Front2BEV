from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import os


def get_test_dirs(dataset_path):
    maps = [ Path(f.path) for f in os.scandir(dataset_path) if f.is_dir() ]
    tests = []
    for map in maps:
        tests.extend([ Path(f.path) for f in os.scandir(map) if f.is_dir() ])
    return tests

def replace_abs_path(csv_path, old_path, new_path):
    df = pd.read_csv(csv_path, header=None)
    new_df = df.replace(regex=[old_path],value=new_path)
    new_df.to_csv(csv_path, header=False, index=False)
    return new_df

def setup_mask():
    mask = cv2.imread(str(Path("utils") / "mask64.png"), 0)
    mask64 = np.zeros_like(mask)

    fov = mask64 > 128
    mask64[fov] = 1

    out_of_fov = mask64 == 0

    return mask64, fov, out_of_fov

mask64, fov, out_of_fov = setup_mask()
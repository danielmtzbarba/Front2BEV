import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_img(path):
    print(path)
    return cv2.imread(path, cv2.IMREAD_COLOR)

def preprocess(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv_img

def plot_hist(img):
    plt.hist(img.ravel(),256,[0,256])
    plt.show()

def show_img(img):
    cv2.imshow('rawbev', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def proc_img(img):
    lower = np.array([75, 0, 180])
    upper = np.array([156, 37, 225])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(img, img, mask= mask)
    return output 

IMG_DIR = '/media/dan/data/datasets/Dan-2024-F2B-Autominy'
TRACKS = ['Track01']
SCENES = ['scene-1']

if __name__ == '__main__':
    for track in TRACKS:
        for scene in SCENES:
            scene_dir = os.path.join(IMG_DIR, track, scene, "rawbev")
            imgs = os.listdir(scene_dir)
            for img_name in imgs:
                img = load_img(os.path.join(scene_dir, img_name))
                mask = proc_img(img)
                show_img(mask)

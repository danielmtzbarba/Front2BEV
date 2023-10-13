from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd


import utils.bev as bev

msk = bev.mask64.copy()
print("\nMASK64:", np.unique(msk, return_counts=True)[1], '\n')

def get_class_weights(img_dataset, n_classes):
    pixel_count = {key:value for (key,value) in enumerate([0 for i in range(n_classes + 1)])}
    pixel_count_fov = {key:value for (key,value) in enumerate([0 for i in range(n_classes)])}

    total_pixels = 0
    total_fov_pixels = 0

    for i in tqdm(range(len(img_dataset))):
        bev_img = img_dataset[i]

        pixel_count, n_pixels = bev.count_pixels(bev_img, pixel_count, n_classes, False)
        total_pixels += n_pixels

        pixel_count_fov, n_fov_pixels = bev.count_pixels(bev_img, pixel_count_fov, n_classes, True)
        total_fov_pixels += n_fov_pixels
            
    weights = {key:(total_pixels/value) for (key,value) in pixel_count.items()}
    weights_fov = {key:(total_fov_pixels/value) for (key,value) in pixel_count_fov.items()}

    print("\nTotal_pixels:", total_pixels) 
    print('Pixel_count:', pixel_count)
    print('Class weights:', weights, '\n')

    print("\nTotal_FOV_pixels:", total_fov_pixels) 
    print('FOV_Pixel_count:', pixel_count_fov)
    print('Class weights (FOV):', weights_fov, '\n')

    return [total_pixels, str(pixel_count), str(weights), total_fov_pixels,
             str(pixel_count_fov), str(weights_fov)]


from dan.utils.torch.datasets import ImageDataset

csv_path = Path('__datasets/Dan-2023-Front2bev/')
filename = 'front2bev_'
root_path = '/media/dan/dan/Datasets/'

def main():

    for config in ['layers_all', 'layers_none']:
        weight_data = []

        for n in [2, 3]:
            path = csv_path / config / f"{n}k" / f"{filename}{n}k.csv"
            df = pd.read_csv(path, header=None, usecols=[3]).drop(0)
            df = df.apply(lambda path: (root_path + path))
            bev_dataset = ImageDataset(df.values)
            wdata = get_class_weights(bev_dataset, n)
            weight_data.append([config, n, *wdata])
            weights_df = pd.DataFrame(weight_data, columns=['config', 'n_classes', 'total_pix', 'pix_count',
                                        'weights', 'total_fov_pix','fov_pix_count','fov_weights'])
            weights_df.to_csv(str(csv_path / config / f'{n}k' / f'weights_{n}k.csv'), index=False)

    config = 'traffic'
    for n in [2, 3, 4, 5, 6]:
        weight_data = []
        path = csv_path / config / f"{n}k" / f"{filename}{n}k.csv"
        df = pd.read_csv(path, header=None, usecols=[3]).drop(0)
        df = df.apply(lambda path: (root_path + path))
        bev_dataset = ImageDataset(df.values)
        wdata = get_class_weights(bev_dataset, n)
        weight_data.append([config, n, *wdata])

        weights_df = pd.DataFrame(weight_data, columns=['config', 'n_classes', 'total_pix', 'pix_count',
                                            'weights', 'total_fov_pix','fov_pix_count','fov_weights'])
        weights_df.to_csv(str(csv_path / config / f'{n}k' / f'weights_{n}k.csv'), index=False)
   

if __name__ == '__main__':
  main()




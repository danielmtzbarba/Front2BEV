from dan.tools.dataset2CSV import get_csv_datasets

# Use train set for choosing hyper-parameters, and use train+val for final traning and testing
# train_plus_val_csv_path = 'dataset/Cityscapes/CS_trainplusval_64.csv'
ROOT_PATH = "C:/Users/Danie/OneDrive/dan/RESEARCH/DATASETS/Dan-2023-CarlaBEV/TOWN01/"
x_dir = ROOT_PATH + "rgb"
y_dir = ROOT_PATH + "map"

csv_dataset_path = "dataset/Front2BEV/bev-vae.csv"

train_csv_path, val_csv_path, _ = get_csv_datasets(csv_dataset_path, x_dir, y_dir)

"""
print(f"Dataset size: {len(dataset_arrays[0])}")
print(f"Train size: {len(train_samples[0])}")
print(f"Val size: {len(val_samples[0])}")
print(f"Test size: {len(test_samples[0])}")
"""

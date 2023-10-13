# Use train set for tuning hyper-parameters.
# Then use train+val for final traning and testing.

TEST_NAME = "TEST_NAME"
CONFIG = "CONFIG"
N_CLASSES = "N_CLASSES"

args = {
    'test_name': TEST_NAME,
    'res_path': f'__results/{TEST_NAME}/',
    'seed': 1596,

    'restore_ckpt': False,
    'ckpt_path': f'/home/aircv1/Data/Luis/aisyslab/Daniel/Checkpoints/{TEST_NAME}.pth.tar',
    'log_path': f"/home/aircv1/Data/Luis/aisyslab/Daniel/Logs/{TEST_NAME}.pkl",
    'save_every': 1,

    'distributed': True,
    'n_gpus': 4,

    'n_epochs':  10,
    'batch_size': 32,
    'n_workers': 16,

    'n_classes': 0,
    'class_weights': None,
    'ignore_class': True,

    # Dataset absolute path
    'dataset_root_path': "/home/aircv1/Data/Luis/aisyslab/Daniel/Datasets/",
    # Relative paths to csv datasets
    'train_csv_path':  f"_datasets/Dan-2023-Front2bev/{CONFIG}/{N_CLASSES}k/front2bev_{N_CLASSES}k-train.csv",
    'val_csv_path': f"_datasets/Dan-2023-Front2bev/{CONFIG}/{N_CLASSES}k/front2bev_{N_CLASSES}k-val.csv",
    'test_csv_path': f"_datasets/Dan-2023-Front2bev/{CONFIG}/{N_CLASSES}k/front2bev_{N_CLASSES}k-test.csv",
}
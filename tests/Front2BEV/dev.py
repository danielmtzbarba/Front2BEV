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
    'ckpt_path': f"/media/dan/dan/Checkpoints/Dan-2023-Front2BEV/BEV_VAE_05oct/{TEST_NAME}.pth.tar",
    'log_path': f"/media/dan/dan/Logs/Dan-2023-Front2BEV/BEV_VAE_05oct/{TEST_NAME}.pkl",
    'save_every': 1,


    'distributed': True,
    'n_gpus': 1,

    'n_epochs':  3,
    'batch_size': 4,
    'n_workers': 4,

    'n_classes': 3,
    
    'class_weights': None,
    'ignore_class': True,

    'num_class': 3,
    'map_config': "layers_all",
    "num_workers": 1,

    # Dataset absolute path
    'dataset_root_path': "/media/dan/dan/Datasets/Dan-2023-Front2BEV",
    'dataset_root': "/media/dan/dan/Datasets/Dan-2023-Front2BEV/",

    # Relative paths to csv datasets
    #'train_csv_path':  f"_datasets/Dan-2023-Front2bev/{CONFIG}/{N_CLASSES}k/front2bev_{N_CLASSES}k-train.csv",
    #'val_csv_path': f"_datasets/Dan-2023-Front2bev/{CONFIG}/{N_CLASSES}k/front2bev_{N_CLASSES}k-val.csv",
    #'test_csv_path': f"_datasets/Dan-2023-Front2bev/{CONFIG}/{N_CLASSES}k/front2bev_{N_CLASSES}k-test.csv",

    'csv_path': 'src/datasets/Front2BEV-debug'
}

import os, ast
import pandas as pd
from yacs.config import CfgNode
from argparse import ArgumentParser


def load_config(config_path):
    with open(config_path) as f:
        return CfgNode.load_cfg(f)

def get_default_configuration():
    root = os.path.abspath(os.path.join(__file__, '..', '..', '..'))
    defaults_path = os.path.join(root, 'configs/config.yml')
    return load_config(defaults_path)

def get_console_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', choices=['front2bev', 'nuscenes'],
                        default='front2bev', help='dataset to train on')
    parser.add_argument('--model', choices=['ved', 'pyramid'],
                        default='ved', help='model to train')
    parser.add_argument('--experiment', default='test', 
                        help='name of experiment config to load')
    parser.add_argument('--resume', default=None, 
                        help='path to an experiment to resume')
    parser.add_argument('--pc', default='home', 
                        help='machine config')
    parser.add_argument('--options', nargs='*', default=[],
                        help='list of addition config options as key-val pairs')
    return parser.parse_args()

def get_configuration():

    args = get_console_args()

    # Load config defaults
    config = get_default_configuration()

    # Load experiment options
    config.merge_from_file(f'configs/experiments/{args.experiment}.yml')

    # Override with command line options
    #config.merge_from_list(args.options)

    # Load pc options
    config.merge_from_file(f'configs/pc/{args.pc}.yml')

    # Load dataset options
    config.merge_from_file(f'configs/datasets/{config.train_dataset}.yml')

    # Load model options
    config.merge_from_file(f'configs/models/{config.model}.yml')

    # Restore config from an existing experiment
    if args.resume is not None:
        config.merge_from_file(os.path.join(args.resume, 'config.yml'))

    if config.train_dataset == "front2bev":
        config.class_weights = get_dataset_weights(config)

    # Finalise config
    config.freeze()

    return config

def create_experiment(config, resume):
    # Restore an existing experiment if a directory is specified
    if resume is not None:
        print("\n==> Restoring experiment from directory:\n" + resume)
        logdir = resume
    else:
        # Otherwise, generate a run directory based on the current time
        logdir = os.path.join(os.path.expandvars(config.logdir), config.name, f"{config.num_class}k-{config.map_config}")
        print("\n==> Creating new experiment in directory:\n" + logdir)
        try:
            os.makedirs(logdir)
        except:
            # Directory exists
            pass
    # Save the current config
    with open(os.path.join(logdir, 'config.yml'), 'w') as f:
        f.write(config.dump())

    print(config.name, config.map_config, config.num_class)
    
    return logdir

def get_dataset_weights(config):
    df_weights = pd.read_csv(os.path.join(config.csv_path, "weights.csv"))
    weights_dict = df_weights.loc[(df_weights['config']==config.map_config) & (df_weights['n_classes']== config.num_class)].reset_index()
    print(weights_dict['fov_weights'][0])
    weights_fov_dict = ast.literal_eval(weights_dict['fov_weights'][0])
    
    no_weights=[1 for i in range(config.num_class)]
    [weights_fov_dict[i] for i in range(config.num_class)]
    return no_weights 

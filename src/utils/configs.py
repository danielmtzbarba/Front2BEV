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

    # Load pc options
    config.merge_from_file(f'configs/pc/{args.pc}.yml')

    # Load model options
    config.merge_from_file(f'configs/models/{args.model}.yml')

    # Load dataset options
    config.merge_from_file(f'configs/datasets/{args.dataset}.yml')

    # Load experiment options
    config.merge_from_file(f'configs/experiments/{args.experiment}.yml')

    # Override with command line options
    #config.merge_from_list(args.options)


    # Restore config from an existing experiment
    if args.resume is not None:
        config.merge_from_file(os.path.join(args.resume, 'config.yml'))
    
    config.name = f'{config.name}-{config.map_config}-{config.weight_mode}'
    config.logdir = os.path.join(config.logdir, config.name)
    create_experiment(config, args.resume)

    # Finalize config
    config.freeze()

    return config

def create_experiment(config, resume):
    # Restore an existing experiment if a directory is specified
    if resume is not None:
        print("\n==> Restoring experiment from directory:\n" + resume)
        logdir = resume
    else:
        # Otherwise, generate a run directory based on the current time
        print("\n==> Creating new experiment in directory:\n" + config.logdir)
        try:
            os.makedirs(config.logdir)
        except:
            # Directory exists
            pass
    # Save the current config
    with open(os.path.join(config.logdir, 'config.yml'), 'w') as f:
        f.write(config.dump())

    print(config.name)
    return config.logdir

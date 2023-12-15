import warnings
warnings.filterwarnings("ignore")

from dan.utils import load_pkl_file
from dan.utils.torch import set_deterministic

from src.utils.dataloader import get_f2b_dataloaders
import scripts.plot_train_res as graph

# -----------------------------------------------------------------------------
import argparse

def set_console_args(name):
    
    from configs.experiments.dev import args

    argparser = argparse.ArgumentParser(description='Front2BEV export results')
    
    argparser.add_argument('-c','--mapconfig', help='Map Config')

    argparser.add_argument('-k','--kclasses', help='K classes')

    console_args = argparser.parse_args()

    config = console_args.mapconfig
    n = console_args.kclasses

    test_name = f"{name}-{config}-{n}k"
    args["name"] = test_name
    args["num_class"] = int(n)
    args["map_config"] = config
    return args
# -----------------------------------------------------------------------------

from dan.utils import dict2obj

def main():
    name = "F2B-VED"
    args = set_console_args(name)
    set_deterministic(args["seed"])

    args = dict2obj(args)
    args.distributed = False

    dataloaders = get_f2b_dataloaders(args)

    args.test_loader = dataloaders['test']

    print("\n", args.name)
  
    log_file = load_pkl_file(args.logdir + f'/{args.name}.pkl')

    
    ax = graph.plot_train_loss(log_file['batches']['loss'],
                                    ['Variational Encoder Decoder', args.map_config, args.num_class],
                                    save_path=f'_results/{args.name}-train.png')
    

    ax = graph.plot_val_metrics(log_file['epochs'],
                                ['Variational Encoder Decoder', args.map_config, args.num_class],
                                save_path=f'_results/{args.name}-val.png')
   
    
if __name__ == '__main__':
    main()
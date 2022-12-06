import os, random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import wandb

from models.handler import train, test

from utils.data_utils import get_full_df
from utils.graph_utils import get_graph_dataset

from utils.math_utils import WandbLogger


from models.gnn_models import A3TGCN_Temporal

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
SEED = 117 # John-117 :>
TOTAL_DAYS = 400

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Available datasets:
#     Houston
#     Chicago
#     Los Angeles
#     New York
#     San Antonio

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--evaluate', type=bool, default=True)
parser.add_argument('--dataset', type=str, default='Houston')
parser.add_argument('--window_size', type=int, default=24)
parser.add_argument('--horizon', type=int, default=6)
parser.add_argument('--train_ratio', type=float, default=0.7)
parser.add_argument('--valid_ratio', type=float, default=0.2)
parser.add_argument('--test_length', type=float, default=0.1)
parser.add_argument('--epoch', type=int, default=40)
parser.add_argument('--lr', type=float, default=3e-4)
# parser.add_argument('--multi_layer', type=int, default=5)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--validate_freq', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--norm_method', type=str, default='z_score') #TODO: change to z-score
parser.add_argument('--optimizer', type=str, default='RMSProp')
parser.add_argument('--early_stop', type=bool, default=False)
parser.add_argument('--exponential_decay_step', type=int, default=5)
parser.add_argument('--decay_rate', type=float, default=0.5)
# parser.add_argument('--dropout_rate', type=float, default=0.5)
# parser.add_argument('--leakyrelu_rate', type=int, default=0.2)
parser.add_argument('--is_wandb_used', type=bool, default=False)
# parser.add_argument("--gpu_devices", type=int, nargs='+', default=0, help="")
parser.add_argument('--start_poi', type=int, default=0)
parser.add_argument('--end_poi', type=int, default=5)
parser.add_argument('--model', type=str, default='A3TGCN')


args = parser.parse_args()

print(f'Training configs: {args}')
result_train_file = os.path.join('output/baselines', args.dataset, f'train_{args.start_poi}_{args.end_poi}')
result_test_file = os.path.join('output/baselines', args.dataset, f'test_{args.start_poi}_{args.end_poi}')
if not os.path.exists(result_train_file):
    os.makedirs(result_train_file)
if not os.path.exists(result_test_file):
    os.makedirs(result_test_file)




def get_datasets(city):
    csv_path = f'/home/users/arash/datasets/safegraph/weekly_patterns_2019-01-07_2020-06-08_{city}.csv'
    poi_info_csv_path = '/home/users/arash/datasets/safegraph/core_poi_info_2019-01-07_2020-06-08.csv'
    
    df = get_full_df(csv_path_weekly=csv_path, 
                         poi_info_csv_path=poi_info_csv_path, 
                         start_row=args.start_poi, end_row=args.end_poi, 
                         total_days=TOTAL_DAYS,
                         city=city)
    
    train_dataset, valid_dataset, test_datset = get_graph_dataset(
        df=df,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        window_size=args.window_size,
        horizon=args.horizon
    )
    return train_dataset, valid_dataset, test_datset



def main():
    train_dataset, valid_dataset, test_dataset = get_datasets(args.dataset)
    run_name = f'{args.dataset}-{args.model}-{str(datetime.now().strftime("%Y-%m-%d %H:%M"))}'
    
    wandb_logger = WandbLogger("POI_forecast_baselines", args.is_wandb_used, run_name)
    wandb_logger.log_hyperparams(args)
    
    if args.train:
        try:
            before_train = datetime.now().timestamp()
            _, normalize_statistic = train(wandb_logger, train_dataset, valid_dataset,
                                           args, result_train_file)
            after_train = datetime.now().timestamp()
            print(f'Training took {(after_train - before_train) / 60} minutes')
        except KeyboardInterrupt:
            print('-' * 99)
            print('Exiting from training early')
    if args.evaluate:
        before_evaluation = datetime.now().timestamp()
        test(wandb_logger, test_dataset, args, result_train_file, result_test_file)
        after_evaluation = datetime.now().timestamp()
        print(f'Evaluation took {(after_evaluation - before_evaluation) / 60} minutes')
    print('done')
            





def main2():
    CITY = 'Houston'
    OUTPUT_DIRECTORY = f'./output/baselines/{CITY}'
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    train_dataset, valid_dataset, test_dataset = get_datasets(CITY)
    train_iter, valid_iter, test_iter = [
        iter(train_dataset), iter(valid_dataset), iter(test_dataset)
    ]
    
    
    # GPU support
    device = torch.device('cuda') # cuda


    # Create model and optimizers
    model = A3TGCN_Temporal(node_features=1, periods=args.horizon)
    
    model.to('cuda')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    model.train()

    print("Running training...")
    for epoch in range(20): 
        loss = 0
        step = 0
        for time, snapshot in tqdm(enumerate(train_dataset)):
            snapshot = snapshot.to(device)
            # Get model predictions
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            # Mean squared error
            loss = loss + torch.mean((y_hat-snapshot.y)**2) 
            step += 1


        loss = loss / (step + 1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("Epoch {} train MSE: {:.4f}".format(epoch, loss.item()))

    
if __name__ == '__main__':
    main()
from email import header
from operator import index
import os
import torch
import wandb
import numpy as np
import random
from datetime import datetime
from models.handler import train, test
from models.handler import get_cat_static_features
import argparse
import pandas as pd
import ast
from utils.math_utils import evaluate
from utils.math_utils import WandbLogger
from utils.data_utils import get_merged_df, get_full_df, load_poi_db, smooth_poi
from utils.data_utils import get_cat_codes, get_cat_codes_df, get_semantic_embs
from utils.data_utils import get_globs_types_df, get_dist_adj_mat
import time, pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
SEED = 117 # John-117 :>

TOTAL_DAYS = 400
CAT_COLS = ['top_category', 'sub_category']

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
parser.add_argument('--window_size', type=int, default=168)
parser.add_argument('--horizon', type=int, default=6)
parser.add_argument('--train_ratio', type=float, default=0.7)
parser.add_argument('--valid_ratio', type=float, default=0.2)
parser.add_argument('--test_length', type=float, default=0.1)
parser.add_argument('--epoch', type=int, default=40)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--multi_layer', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--validate_freq', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--norm_method', type=str, default='z_score') #TODO: change to z-score
parser.add_argument('--optimizer', type=str, default='RMSProp')
parser.add_argument('--early_stop', type=bool, default=False)
parser.add_argument('--exponential_decay_step', type=int, default=5)
parser.add_argument('--decay_rate', type=float, default=0.5)
parser.add_argument('--dropout_rate', type=float, default=0.5)
parser.add_argument('--leakyrelu_rate', type=int, default=0.2)
parser.add_argument('--is_wandb_used', type=bool, default=True)
parser.add_argument("--gpu_devices", type=int, nargs='+', default=0, help="")
parser.add_argument("--cache_data", type=bool, default=True)
parser.add_argument("--run_identity", type=str, default='BysGNN - Attention Adj')
parser.add_argument('--start_poi', type=int, default=0)
parser.add_argument('--end_poi', type=int, default=400)


args = parser.parse_args()
# gpu_devices = ','.join([str(id) for id in args.gpu_devices])
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

print(f'Training configs: {args}')
#data_file = os.path.join('dataset', args.dataset + '.csv')
result_train_file = os.path.join('output', args.dataset, f'train_{args.start_poi}_{args.end_poi}')
result_test_file = os.path.join('output', args.dataset, f'test_{args.start_poi}_{args.end_poi}')
if not os.path.exists(result_train_file):
    os.makedirs(result_train_file)
if not os.path.exists(result_test_file):
    os.makedirs(result_test_file)
#data = pd.read_csv(data_file).values

# csv_path = '/storage/dataset/poi_haowen/Houston/Housron_2019-12-30_2020-06-08.csv'
# csv_path  ='/root/dataset_all/poi_forecast/Houston/Houston_2019-12-30_2020-06-08.csv'
csv_path = f'/home/users/arash/datasets/safegraph/weekly_patterns_2019-01-07_2020-06-08_{args.dataset}.csv'
poi_info_csv_path = '/home/users/arash/datasets/safegraph/core_poi_info_2019-01-07_2020-06-08.csv'


# def get_good_poi(df):
#     bads = np.array([ 14,  15,  32,  47, 221, 291, 339, 358, 398, 412, 417])
#     df = df.reset_index().drop(bads, axis='index')
#     return df

    

    
    
    
#data_frame = get_merged_df(csv_path=csv_path, num=2000)
# data_frame = get_merged_df(csv_path=csv_path, start_row=args.start_poi,end_row=args.end_poi)
data_pkl_dir = f'./cache_data/'
data_pkl_path = os.path.join(data_pkl_dir, f'data-{args.start_poi}-{args.end_poi}-{TOTAL_DAYS}-{args.dataset}.pkl')
if not os.path.exists(data_pkl_dir):
    os.makedirs(data_pkl_dir)
if args.cache_data and os.path.exists(data_pkl_path):
    print('Data already exists...')
    data_frame = pd.read_pickle(data_pkl_path)
    print('Data loaded')
else:
    data_frame = get_full_df(csv_path_weekly=csv_path, 
                        poi_info_csv_path=poi_info_csv_path, 
                        start_row=args.start_poi, end_row=args.end_poi, 
                        total_days=TOTAL_DAYS,
                        city=args.dataset)

    #TODO: Remove this line?
    # data_frame = get_globals_df(data_frame)

    data_frame = get_globs_types_df(data_frame, 'top_category')

    # data_frame.to_csv('full_dataset.csv', index=False)

    # data_frame = get_good_poi(data_frame)
    # data_frame = smooth_poi(data_frame)

    pd.to_pickle(data_frame, data_pkl_path)
    print('Data cached...')

data = pd.DataFrame(data_frame["visits_by_each_hour"].to_list()).T
args.n_route = data.shape[-1]

# # shuffle the data
# data = data.sample(frac=1).reset_index(drop=True)

# split data
days = int(data.shape[0] / 24)

train_ratio = args.train_ratio
valid_ratio = args.valid_ratio


train_days = int(train_ratio * days)
valid_days = int(days*valid_ratio)
test_days = days-train_days-valid_days

train_data = data[:train_days*24]
valid_data = data[train_days*24:(train_days + valid_days)*24]
test_data = data[(train_days + valid_days)*24:(train_days + valid_days+test_days)*24]


semantic_embs = get_semantic_embs(data_frame, args.end_poi, args.start_poi)

print(f"train {train_data.shape} valid {valid_data.shape} test {test_data.shape}")

run_name = f'{args.dataset}-{args.start_poi}-{args.end_poi}-w:{args.window_size}-h:{args.horizon}-{args.run_identity}-{str(datetime.now().strftime("%Y-%m-%d %H:%M"))}'

dist_adj_mat = get_dist_adj_mat(data_frame, nodes_num=args.end_poi-args.start_poi)

wandb_logger = WandbLogger("POI_forecast", args.is_wandb_used, run_name)
wandb_logger.log_hyperparams(args)
if __name__ == '__main__':
    if args.train:
        try:
            before_train = datetime.now().timestamp()
            _, normalize_statistic = train(wandb_logger,train_data, valid_data,
                                           args, result_train_file,
                                           nodes_num=args.end_poi-args.start_poi,
                                           dist_adj_mat=dist_adj_mat,
                                           semantic_embs=semantic_embs)
            after_train = datetime.now().timestamp()
            print(f'Training took {(after_train - before_train) / 60} minutes')
        except KeyboardInterrupt:
            print('-' * 99)
            print('Exiting from training early')
    if args.evaluate:
        before_evaluation = datetime.now().timestamp()
        test(wandb_logger, test_data, args, result_train_file, result_test_file,
             nodes_num=args.end_poi-args.start_poi)
        after_evaluation = datetime.now().timestamp()
        print(f'Evaluation took {(after_evaluation - before_evaluation) / 60} minutes')
    print('done')
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torch_data
import numpy as np
import time
import os

from models.gnn_models import A3TGCN_Temporal

def get_model(model_name):
    if model_name == 'A3TGCN':
        return A3TGCN_Temporal


def train(wandb_logger, train_dataset, valid_dataset, args, result_file, model_name):
    print('training...')
    
    model_class = get_model(model_name)
    model = model_class(args.node_features, args.horizon)
    
    model = nn.DataParallel(model)
    model.to(args.device)
    wandb_logger.watch_model(model)

    if len(train_dataset) == 0:
        raise Exception('Cannot organize enough training data')
    if len(valid_dataset) == 0:
        raise Exception('Cannot organize enough validation data')
    
    
    
    return 'no', 'data'


def test(wandb_logger,test_data, args, result_train_file, result_test_file):
    print('testing...')
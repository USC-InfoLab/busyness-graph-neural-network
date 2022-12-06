import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torch_data
import numpy as np
import time
import os


def train(wandb_logger, train_dataset, valid_dataset, args, result_file):
    print('training...')
    return 'no', 'data'


def test(wandb_logger,test_data, args, result_train_file, result_test_file):
    print('testing...')
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torch_data
import numpy as np
import time
import os

from models.gnn_models import A3TGCN_Temporal, DCRNN_Temporal, GConvGRU_Temporal
from utils.graph_utils import get_graph_dataset, de_normalize
from utils.math_utils import evaluate


def get_model(model_name):
    if model_name == 'A3TGCN':
        return A3TGCN_Temporal
    if model_name == 'DCRNN':
        return DCRNN_Temporal
    if model_name == 'ConvGRU':
        return GConvGRU_Temporal
        
    
    
def save_model(model, model_dir, model_name, epoch=None):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + f'{model_name}.pt')
    with open(file_name, 'wb') as f:
        torch.save(model, f)
        

def load_model(model_dir, model_name, epoch=None):
    if not model_dir:
        return
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + f'{model_name}.pt')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(file_name):
        return
    with open(file_name, 'rb') as f:
        model = torch.load(f)
    return model


def inference(model, dataset, device, window_size, horizon):
    forecast_set = []
    target_set = []
    model.eval()
    with torch.no_grad():
        for i, snapshot in enumerate(dataset):
            snapshot = snapshot.to(device)
            inputs = snapshot.x
            target = snapshot.y
            step = 0
            forecast_steps = np.zeros([inputs.shape[1], horizon, inputs.shape[0]], dtype=np.float)
            while step < horizon:
                forecast_result = model(inputs, snapshot.edge_index, snapshot.edge_attr)
                len_model_output = forecast_result.size()[1]
                if len_model_output == 0:
                    raise Exception('Get blank inference result')
                inputs[:, :, :window_size - len_model_output] = inputs[:, :,len_model_output:window_size].clone()
                inputs[:, :, window_size - len_model_output:] = forecast_result.clone().unsqueeze(1)
                forecast_steps[:, step:min(horizon - step, len_model_output) + step, :] = \
                    forecast_result.detach().reshape(-1, inputs.shape[0]).unsqueeze(0).cpu().numpy()
                step += min(horizon - step, len_model_output)
            forecast_set.append(forecast_steps)
            target_set.append(target.detach().reshape(-1, inputs.shape[0]).unsqueeze(0).cpu().numpy())
    return np.concatenate(forecast_set, axis=0), np.concatenate(target_set, axis=0)



def validate(model, dataset, device, normalize_method, statistic,
             window_size, horizon, result_file=None):
    start = datetime.now()
    forecast_norm, target_norm = inference(model, dataset, device, window_size, horizon)
    if normalize_method and statistic:
        forecast = de_normalize(forecast_norm, normalize_method, statistic)
        target = de_normalize(target_norm, normalize_method, statistic)
    else:
        forecast, target = forecast_norm, target_norm
        
    
    score = evaluate(target, forecast)
    score_by_node = evaluate(target, forecast, by_node=True)
    end = datetime.now()

    score_norm = evaluate(target_norm, forecast_norm)
    print(f'NORM: MAPE {score_norm[0]:7.9%}; MAE {score_norm[1]:7.9f}; RMSE {score_norm[2]:7.9f}.')
    print(f'RAW : MAPE {score[0]:7.9%}; MAE {score[1]:7.9f}; RMSE {score[2]:7.9f}.')
    if result_file:
        if not os.path.exists(result_file):
            os.makedirs(result_file)
        step_to_print = 0
        forcasting_2d = forecast[:, step_to_print, :]
        forcasting_2d_target = target[:, step_to_print, :]

        np.savetxt(f'{result_file}/target.csv', forcasting_2d_target, delimiter=",")
        np.savetxt(f'{result_file}/predict.csv', forcasting_2d, delimiter=",")
        np.savetxt(f'{result_file}/predict_abs_error.csv',
                   np.abs(forcasting_2d - forcasting_2d_target), delimiter=",")
        np.savetxt(f'{result_file}/predict_ape.csv',
                   np.abs((forcasting_2d - forcasting_2d_target) / forcasting_2d_target), delimiter=",")
        
    return dict(mae=score[1], mae_node=score_by_node[1], mape=score[0], mape_node=score_by_node[0],
                rmse=score[2], rmse_node=score_by_node[2])


def train(wandb_logger, train_data, valid_data, adj_mat, args, result_file, model_name):
    print('training...')
    
    model_class = get_model(model_name)
    model = model_class(args.node_features, args.horizon)
    
    # model = nn.DataParallel(model)
    model.to(args.device)
    wandb_logger.watch_model(model)

    if len(train_data) == 0:
        raise Exception('Cannot organize enough training data')
    if len(valid_data) == 0:
        raise Exception('Cannot organize enough validation data')
    
    
    if args.norm_method == 'z_score':
        train_mean = np.mean(train_data, axis=0)
        train_std = np.std(train_data, axis=0)
        normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
    elif args.norm_method == 'min_max':
        train_min = np.min(train_data, axis=0)
        train_max = np.max(train_data, axis=0)
        normalize_statistic = {"min": train_min.tolist(), "max": train_max.tolist()}
    else:
        normalize_statistic = None
    if normalize_statistic is not None:
        with open(os.path.join(result_file, 'norm_stat.json'), 'w') as f:
            json.dump(normalize_statistic, f)
            
    
    if args.optimizer == 'RMSProp':
        my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.lr, eps=1e-08)
    else:
        my_optim = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)
    
    train_set = get_graph_dataset(train_data, adj_mat, 
                                  args.window_size, args.horizon,
                                  args.norm_method, normalize_statistic)
    
    valid_set = get_graph_dataset(valid_data, adj_mat, 
                                args.window_size, args.horizon,
                                args.norm_method, normalize_statistic)
    

    forecast_loss = nn.MSELoss(reduction='mean').to(args.device)

    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params += param
    print(f"Total Trainable Params: {total_params}")
    
    best_validate_mae = np.inf
    validate_score_non_decrease_count = 0
    performance_metrics = {}
    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        model.train()
        loss_total = 0
        cnt = 0
        
        
        cur_batch_cnt = 0
        loss = 0
        for i, snapshot in enumerate(train_set):
            snapshot = snapshot.to(args.device)
            forecast = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            loss += forecast_loss(forecast, snapshot.y)
            cur_batch_cnt += 1
            if cur_batch_cnt % args.batch_size == 0:     
                cnt += 1
                model.zero_grad()
                loss.backward()
                my_optim.step()
                loss_total += float(loss) / args.batch_size
                loss = 0
                    
            
            
        print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f}'.format(epoch, (
        time.time() - epoch_start_time), loss_total / cnt))
        wandb_logger.log("train_total_loss", loss_total / cnt,epoch)
        
        save_model(model, result_file, model_name, epoch)
        
        if (epoch+1) % args.exponential_decay_step == 0:
            my_lr_scheduler.step()
        if (epoch + 1) % args.validate_freq == 0:
            is_best_for_now = False
            print('------ validate on data: VALIDATE ------')
            performance_metrics = \
                            validate(model, valid_set, args.device, args.norm_method, normalize_statistic, args.window_size, args.horizon,
                            result_file=result_file)
            
            wandb_logger.log("val_mae", performance_metrics['mae'],epoch)
            wandb_logger.log("val_mape",  performance_metrics['mape'], epoch)
            wandb_logger.log("val_rmse", performance_metrics['rmse'], epoch)
            if best_validate_mae > performance_metrics['mae']:
                best_validate_mae = performance_metrics['mae']
                is_best_for_now = True
                validate_score_non_decrease_count = 0
            else:
                validate_score_non_decrease_count += 1
            # save model
            if is_best_for_now:
                save_model(model, result_file, model_name)
        # early stop
        if args.early_stop and validate_score_non_decrease_count >= args.early_stop_step:
            break
    return performance_metrics, normalize_statistic


def test(wandb_logger, test_data, adj_mat, args, result_train_file, result_test_file, model_name):
    print('testing...')
    with open(os.path.join(result_train_file, 'norm_stat.json'),'r') as f:
        normalize_statistic = json.load(f)
    model = load_model(result_train_file, model_name)
    test_set = get_graph_dataset(test_data, adj_mat, 
                                  args.window_size, args.horizon,
                                  args.norm_method, normalize_statistic)
    performance_metrics = validate(model, test_set, args.device,
                                args.norm_method, normalize_statistic, 
                                args.window_size, args.horizon,
                                result_file=result_test_file)
    mae, mape, rmse = performance_metrics['mae'], performance_metrics['mape'], performance_metrics['rmse']
    wandb_logger.log("test_mae", mae, 0)
    wandb_logger.log("test_mape", mape, 0)
    wandb_logger.log("test_rmse", rmse, 0)
    print('Performance on test set: MAPE: {:5.2f} | MAE: {:5.2f} | RMSE: {:5.4f}'.format(mape, mae, rmse))

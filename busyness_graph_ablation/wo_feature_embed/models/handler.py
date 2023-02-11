import json
from datetime import datetime

from data_loader.forecast_dataloader import ForecastDataset, de_normalized
from models.base_model import Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torch_data
import numpy as np
import time
import os

from utils.math_utils import evaluate
from utils.data_utils import emb_sz_rule


def get_cat_static_features(cat_codes_data, device):
    cat_vecs = []
    for cat_col in cat_codes_data.columns:
        cat_vec = F.one_hot(torch.tensor(cat_codes_data[cat_col].to_numpy())).to(device)
        cat_vecs.append(cat_vec)
    cat_static_features = torch.cat(cat_vecs, dim=1).to(device)
    return cat_static_features


def save_model(model, model_dir, epoch=None):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_stemgnn.pt')
    with open(file_name, 'wb') as f:
        torch.save(model, f)


def load_model(model_dir, epoch=None):
    if not model_dir:
        return
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, epoch + '_stemgnn.pt')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(file_name):
        return
    with open(file_name, 'rb') as f:
        model = torch.load(f)
    return model


def inference(model, dataloader, device, node_cnt, window_size, horizon, static_features):
    forecast_set = []
    target_set = []
    model.eval()
    with torch.no_grad():
        for i, (inputs, target) in enumerate(dataloader):
            inputs = inputs.to(device)
            target = target.to(device)
            step = 0
            forecast_steps = np.zeros([inputs.size()[0], horizon, node_cnt], dtype=np.float)
            while step < horizon:
                forecast_result, attention_scores = model(inputs)
                len_model_output = forecast_result.size()[1]
                if len_model_output == 0:
                    raise Exception('Get blank inference result')
                inputs[:, :window_size - len_model_output, :] = inputs[:, len_model_output:window_size,
                                                                   :].clone()
                inputs[:, window_size - len_model_output:, :] = forecast_result.clone()
                forecast_steps[:, step:min(horizon - step, len_model_output) + step, :] = \
                    forecast_result[:, :min(horizon - step, len_model_output), :].detach().cpu().numpy()
                step += min(horizon - step, len_model_output)
            forecast_set.append(forecast_steps)
            target_set.append(target.detach().cpu().numpy())
            attention_scores_out = attention_scores.detach().cpu().numpy().mean(axis=0)
    return np.concatenate(forecast_set, axis=0), np.concatenate(target_set, axis=0), attention_scores_out


def validate(model, dataloader, device, normalize_method, statistic,
             node_cnt, window_size, horizon,
             static_features, result_file=None, nodes_num=None):
    start = datetime.now()
    forecast_norm, target_norm, attention_scores = inference(model, dataloader, device,
                                           node_cnt, window_size, horizon,
                                           static_features=static_features)
    if normalize_method and statistic:
        forecast = de_normalized(forecast_norm, normalize_method, statistic)
        target = de_normalized(target_norm, normalize_method, statistic)
    else:
        forecast, target = forecast_norm, target_norm
        
    # TODO: this is for removing global features from evaluation
    if nodes_num is not None:
        forecast = forecast[:, :, :nodes_num]
        forecast_norm = forecast_norm[:, :, :nodes_num]
        target = target[:, :, :nodes_num]
        target_norm = target_norm[:, :, :nodes_num]
    
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
        
        np.savetxt(f'{result_file}/attention_scores.csv', attention_scores, delimiter=",")

    return dict(mae=score[1], mae_node=score_by_node[1], mape=score[0], mape_node=score_by_node[0],
                rmse=score[2], rmse_node=score_by_node[2])


def train(wandb_logger, train_data, valid_data, args, result_file, static_features=None, cat_codes_dict=None, nodes_num=None):
    node_cnt = train_data.shape[1]
    
    #TODO: adding categorical varialbes
    if static_features is not None:
        # Note: '+1' in len(code) + 1 is due to the fact that we also take an index number for unknown labels, which is
        # not included in the cat_codes_dict
        embedding_size_dict = {col: len(code) + 1 for col, code in cat_codes_dict.items()}
        embedding_dim_dict = {col: emb_sz_rule(embedding_size) for col, embedding_size in embedding_size_dict.items()}
    else:
        embedding_size_dict = None
        embedding_dim_dict = None
    
    
    model = Model(wandb_logger, node_cnt, 2, 
                  args.window_size, args.multi_layer, horizon=args.horizon,
                  embedding_size_dict=embedding_size_dict,
                  embedding_dim_dict=embedding_dim_dict)
    model = nn.DataParallel(model)
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

    train_set = ForecastDataset(train_data, window_size=args.window_size, horizon=args.horizon,
                                normalize_method=args.norm_method, norm_statistic=normalize_statistic)
    valid_set = ForecastDataset(valid_data, window_size=args.window_size, horizon=args.horizon,
                                normalize_method=args.norm_method, norm_statistic=normalize_statistic)
    train_loader = torch_data.DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=True,
                                         num_workers=0)
    valid_loader = torch_data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

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
        for i, (inputs, target) in enumerate(train_loader):
            inputs = inputs.to(args.device)
            target = target.to(args.device)
            model.zero_grad()
            #TODO: change this static_features?
            forecast, attention_scores = model(inputs)
            loss = forecast_loss(forecast, target)
            cnt += 1
            loss.backward()
            my_optim.step()
            loss_total += float(loss)
        print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f}'.format(epoch, (
                time.time() - epoch_start_time), loss_total / cnt))
        wandb_logger.log("train_total_loss", loss_total / cnt,epoch)

        save_model(model, result_file, epoch)
        if (epoch+1) % args.exponential_decay_step == 0:
            my_lr_scheduler.step()
        if (epoch + 1) % args.validate_freq == 0:
            is_best_for_now = False
            print('------ validate on data: VALIDATE ------')
            performance_metrics = \
                validate(model, valid_loader, args.device, args.norm_method, normalize_statistic,
                         node_cnt, args.window_size, args.horizon,
                         static_features=static_features, result_file=result_file, nodes_num=nodes_num)

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
                save_model(model, result_file)
        # early stop
        if args.early_stop and validate_score_non_decrease_count >= args.early_stop_step:
            break
    return performance_metrics, normalize_statistic


def test(wandb_logger,test_data, args, result_train_file, result_test_file, static_features, nodes_num=None):
    with open(os.path.join(result_train_file, 'norm_stat.json'),'r') as f:
        normalize_statistic = json.load(f)
    model = load_model(result_train_file)
    node_cnt = test_data.shape[1]
    test_set = ForecastDataset(test_data, window_size=args.window_size, 
                               horizon=args.horizon,
                               normalize_method=args.norm_method, 
                               norm_statistic=normalize_statistic,
                               df_static_features=static_features)
    test_loader = torch_data.DataLoader(test_set, batch_size=args.batch_size, drop_last=False,
                                        shuffle=False, num_workers=0)
    performance_metrics = validate(model, test_loader, args.device,
                                   args.norm_method, normalize_statistic, 
                                   node_cnt, args.window_size, args.horizon, 
                                   static_features=static_features, 
                                   result_file=result_test_file,
                                   nodes_num=nodes_num)
    mae, mape, rmse = performance_metrics['mae'], performance_metrics['mape'], performance_metrics['rmse']
    wandb_logger.log("test_mae", mae, 0)
    wandb_logger.log("test_mape", mape, 0)
    wandb_logger.log("test_rmse", rmse, 0)
    # wandb_logger.log("test_mae_by_node", performance_metrics['mae_node'], 0)
    # wandb_logger.log("test_mape_by_node", performance_metrics['mae_node'], 0)
    # wandb_logger.log("test_rmse_by_node", performance_metrics['mae_node'], 0)
    print('Performance on test set: MAPE: {:5.2f} | MAE: {:5.2f} | RMSE: {:5.4f}'.format(mape, mae, rmse))
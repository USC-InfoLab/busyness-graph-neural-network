import numpy as np
import torch
import pandas as pd
import urllib.request
import zipfile

from torch_geometric.utils import dense_to_sparse
from torch_geometric_temporal.signal import StaticGraphTemporalSignal, StaticGraphTemporalSignalBatch


def get_distances(coords):
    num_points = coords.shape[0]
    distances = np.array([[np.linalg.norm(i-j) for j in coords] for i in coords])
    return distances


def gaussian_kern(arr, thres=1):
    res = arr.copy()
    res[res<=thres] = np.exp(-(res[res<=thres]**2)/(np.nanstd(arr)**2))
    res[np.isnan(arr)] = 0
    res[arr>thres] = 0
    return res


def get_unsqueezed(arr):
    time_points = arr.shape[0]
    nodes_num = arr.shape[-1]
    res = arr.copy()
    res = np.expand_dims(arr, 2)
    res = res.reshape((time_points, 1, nodes_num))
    return res


def z_normalize(arr):
    means = np.mean(arr, axis=(0, 2))
    res = arr - means.reshape(1, -1, 1)
    stds = np.std(res, axis=(0, 2))
    res = res / stds.reshape(1, -1, 1)
    return res


def normalize(data, norm_method, norm_statistic):
    if norm_method == 'min_max':
        if not norm_statistic:
            norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
        scale = np.array(['max']) - norm_statistic['min'] + 1e-5
        data = (data - norm_statistic['min']) / scale
        data = np.clip(data, 0.0, 1.0)
    elif norm_method == 'z_score':
        if not norm_statistic:
            norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
        mean = norm_statistic['mean']
        std = norm_statistic['std']
        std = [1 if i == 0 else i for i in std]
        data = (data - mean) / std
        norm_statistic['std'] = std
    return data, norm_statistic


def de_normalize(data, normalize_method, norm_statistic):
    if normalize_method == 'min_max':
        if not norm_statistic:
            norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
        scale = np.array(norm_statistic['max']) - norm_statistic['min'] + 1e-5
        data = data * scale + norm_statistic['min']
    elif normalize_method == 'z_score':
        if not norm_statistic:
            norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
        mean = norm_statistic['mean']
        std = norm_statistic['std']
        std = [1 if i == 0 else i for i in std]
        data = data * std + mean
    return data


def to_graph_dataset(arr, adj_mat, num_timesteps_in, num_timesteps_out):
    norm_arr = z_normalize(get_unsqueezed(arr))
    A = torch.from_numpy(adj_mat)
    X = torch.from_numpy(norm_arr)
    edge_indices, values = dense_to_sparse(A)
    edges = edge_indices.numpy()
    edge_weights = values.numpy()
    
    indices = [
    (i, i + (num_timesteps_in + num_timesteps_out))
    for i in range(X.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
    ]
    
    features, target = [], []
    for i, j in indices:
        features.append((X[:, :, i : i + num_timesteps_in]).numpy())
        target.append((X[:, 0, i + num_timesteps_in : j]).numpy())
    
    dataset = StaticGraphTemporalSignal(
    edges, edge_weights, features, target
    )
    
    return dataset




def get_graph_dataset(df, train_ratio, valid_ratio, window_size, horizon):
    # Building the adjacency matrix based on distance
    coords = df[['latitude', 'longitude']].to_numpy()
    distances = get_distances(coords)
    thres = np.nanmean(distances)
    adj_mat = gaussian_kern(distances, thres=thres)
    
    # Getting the node features
    data= pd.DataFrame(df["visits_by_each_hour"].to_list()).T
    days = int(data.shape[0] / 24)
    train_days = int(train_ratio * days)
    valid_days = int(days*valid_ratio)
    test_days = days-train_days-valid_days

    train_data = data[:train_days*24].to_numpy()
    valid_data = data[train_days*24:(train_days + valid_days)*24].to_numpy()
    test_data = data[(train_days + valid_days)*24:(train_days + valid_days+test_days)*24].to_numpy()
    
    train_graph_dataset = to_graph_dataset(train_data, adj_mat, window_size, horizon)
    valid_graph_dataset = to_graph_dataset(valid_data, adj_mat, window_size, horizon)
    test_graph_dataset = to_graph_dataset(test_data, adj_mat, window_size, horizon)
    
    return train_graph_dataset, valid_graph_dataset, test_graph_dataset


def get_adj_mat(df):
    coords = df[['latitude', 'longitude']].to_numpy()
    distances = get_distances(coords)
    thres = np.nanmean(distances)
    adj_mat = gaussian_kern(distances, thres=thres)
    return adj_mat

        
def get_graph_dataset(arr, adj_mat, num_timesteps_in, num_timesteps_out,
                      norm_method=None, norm_stats=None):
    # norm_arr = get_unsqueezed(arr)
    if norm_method:
        norm_arr, _ = normalize(arr, norm_method, norm_stats)
    A = torch.from_numpy(adj_mat)
    X = torch.from_numpy(get_unsqueezed(norm_arr))
    X = X.permute(2, 1, 0)
    edge_indices, values = dense_to_sparse(A)
    edges = edge_indices.numpy()
    edge_weights = values.numpy()
    
    indices = [
    (i, i + (num_timesteps_in + num_timesteps_out))
    for i in range(X.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
    ]
    
    features, target = [], []
    for i, j in indices:
        features.append((X[:, :, i : i + num_timesteps_in]).numpy())
        target.append((X[:, 0, i + num_timesteps_in : j]).numpy())
    
    dataset = StaticGraphTemporalSignal(
    edges, edge_weights, features, target
    )
    
    return dataset
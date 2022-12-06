import numpy as np
import torch
import pandas as pd
import urllib.request
import zipfile

from torch_geometric.utils import dense_to_sparse
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

from utils.data_utils import get_full_df

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
    res = res.reshape((nodes_num, 1, time_points))
    return res


def z_normalize(arr):
    means = np.mean(arr, axis=(0, 2))
    res = arr - means.reshape(1, -1, 1)
    stds = np.std(res, axis=(0, 2))
    res = res / stds.reshape(1, -1, 1)
    return res


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
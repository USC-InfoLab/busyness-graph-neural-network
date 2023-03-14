import numpy as np
import pandas as pd
import time, os, math
from scipy.signal import savgol_filter
import ast



class bounding_box:
    def __init__(self, _lat_min, _lon_min, _lat_max, _lon_max):
        self.lat_min = _lat_min
        self.lon_min = _lon_min
        self.lat_max = _lat_max
        self.lon_max = _lon_max


class stat_collector:
    def __init__(self):
        self.parquet_file_count=0
        self.data_record_count = 0
        self.memory_usage_in_GB = 0		#gives an estimate of the total RAM usage if all files were read into memory at the same time.
        self.unique_device_count = 0
        self.avg_pos_acc = 0
        self.starting_time = time.process_time()
        self.elapsed_time = time.process_time()
        self.unique_geohash_count = 0

def get_merged_df(csv_path, start_row, end_row, total_days):

    #start = time.time()
    merge_df = pd.read_csv(csv_path)

    merge_df = merge_df.sort_values(by=['raw_visit_counts'], ascending=False)
    merge_df = merge_df.iloc[start_row:end_row]
    #print(merge_df)
    merge_df["visits_by_each_hour"] = merge_df["visits_by_each_hour"].apply(lambda x: ast.literal_eval(x))
    merge_df["visits_by_day"] = merge_df["visits_by_day"].apply(lambda x: ast.literal_eval(x))
    merge_df["visits_by_each_hour"] = merge_df["visits_by_each_hour"].apply(lambda x: x[:total_days*24])
    merge_df["visits_by_day"] = merge_df["visits_by_day"].apply(lambda x: x[:total_days])
    return merge_df


def load_poi_db(city):
    poi_folder = "/storage/dataset/poi_haowen/CoreRecords-CORE_POI-2019_03-2020-03-25/"
    poi_columns = ["safegraph_place_id", "parent_safegraph_place_id", "location_name", "safegraph_brand_ids", "brands",
                   "top_category", "sub_category", "naics_code", "latitude", "longitude", "street_address", "city",
                   "region", "postal_code", "iso_country_code", "phone_number", "open_hours", "category_tags"]
    files = os.listdir(poi_folder)


    poi_s = stat_collector()
    poi_db = pd.DataFrame(columns=poi_columns)
    for f in files:
        if f[-3:] == 'csv' and 'brand' not in f:
            print(f)
            df = pd.read_csv(poi_folder + f)
            df = df.loc[df['city']==city]
            poi_db = poi_db.append(df, ignore_index=True, sort=False)
            poi_s.memory_usage_in_GB += df.memory_usage(deep=True).sum() / 1000000000
            poi_s.data_record_count += df.shape[0]
            poi_s.parquet_file_count += 1
    return poi_db, poi_s


def get_full_df(csv_path_weekly, poi_info_csv_path, start_row, end_row, total_days, city):
    weekly_patterns = get_merged_df(csv_path_weekly, start_row, end_row, total_days)
    poi_info = pd.read_csv(poi_info_csv_path)
    poi_df = pd.merge(weekly_patterns, poi_info, on='safegraph_place_id', how='inner')
    poi_db, poi_s = load_poi_db(city=city)
    poi_df = poi_df.merge(poi_db, how='left', on='safegraph_place_id', suffixes=('', '_y'))
    poi_df.drop(poi_df.filter(regex='_y$').columns, axis=1, inplace=True)
    del poi_db
    return poi_df


def get_globals_df(df, glob_num_cols=5):
    res_df = df.copy()
    res_df['visits_by_each_hour'] = df['visits_by_each_hour'].apply(lambda x: np.array(x))
    res_df['visits_by_day'] = df['visits_by_day'].apply(lambda x: np.array(x))
    glob_df = res_df.agg({'visits_by_day': 'sum',
    'visits_by_each_hour': 'sum',
    'raw_visit_counts': 'sum',
    })
    glob_df = pd.DataFrame(glob_df).T
    new_df = pd.DataFrame(np.repeat(glob_df.values, glob_num_cols, axis=0))
    new_df.columns = glob_df.columns
    new_df['safegraph_place_id'] = [f'Glob_{i}' for i in range(len(new_df))]
    new_df['top_category'] = [f'Global' for i in range(len(new_df))]
    new_df['sub_category'] = [f'Global' for i in range(len(new_df))]
    res_df = res_df.append(new_df, ignore_index=True)
    return res_df


def get_globs_types_df(df, cat_col):
    new_df = df.copy()
    new_df['visits_by_each_hour'] = df['visits_by_each_hour'].apply(lambda x: np.array(x))
    new_df['visits_by_day'] = df['visits_by_day'].apply(lambda x: np.array(x))
    glob_df = new_df.groupby('top_category').agg({'visits_by_day': 'sum',
    'visits_by_each_hour': 'sum',
    'raw_visit_counts': 'sum',
    })
    glob_df = pd.DataFrame(glob_df)
    glob_df['top_category'] = glob_df.index
    glob_df['sub_category'] = ['Global' for i in range(len(glob_df))]
    glob_df['safegraph_place_id'] = [f'Category_Global_{i}' for i in range(len(glob_df))]
    res_df = get_globals_df(df, 5*math.ceil(glob_df.shape[0]/5) - glob_df.shape[0])
    res_df = res_df.append(glob_df, ignore_index=True)
    return res_df
    


def smooth_poi(df, window_size=12, poly_deg=3):
    df['visits_by_each_hour'] = df['visits_by_each_hour'].apply(lambda x: savgol_filter(x, window_size, poly_deg))
    return df


def get_cat_codes(df, cat_cols):
    cat_code_dict = {}
    for col in cat_cols:
        category_col = df[col].astype('category')
        cat_code_dict[col] = {value: idx for idx, value in enumerate(category_col.cat.categories)} 
    return cat_code_dict


def get_cat_codes_df(df, cat_code_dict):
    cat_df = pd.DataFrame()
    for col, code_dict in cat_code_dict.items():
        code_fillna_value = len(code_dict)
        cat_df[col] = df[col].map(code_dict).fillna(code_fillna_value).astype(np.int64)
    return cat_df


def emb_sz_rule(n_cat):
    "Rule of thumb to pick embedding size corresponding to `n_cat`"
    return min(600, round(1.6 * n_cat ** 0.56))
    # return min(600, round(5* int(n_cat ** 0.3)))
    # return round(math.sqrt(n_cat))


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


def get_dist_adj_mat(df, nodes_num):
    coords = df[['latitude', 'longitude']].to_numpy()
    distances = get_distances(coords)
    thres = np.nanstd(distances) * 2
    adj_mat = gaussian_kern(distances, thres=thres)
    adj_mat[:, nodes_num:] = 1
    adj_mat[nodes_num:, :] = 1
    return adj_mat
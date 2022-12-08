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


def get_split_data(city, args):
    csv_path = f'/home/users/arash/datasets/safegraph/weekly_patterns_2019-01-07_2020-06-08_{city}.csv'
    poi_info_csv_path = '/home/users/arash/datasets/safegraph/core_poi_info_2019-01-07_2020-06-08.csv'
    
    df = get_full_df(csv_path_weekly=csv_path, 
                         poi_info_csv_path=poi_info_csv_path, 
                         start_row=args.start_poi, end_row=args.end_poi, 
                         total_days=args.total_days,
                         city=city)
    
    
    data= pd.DataFrame(df["visits_by_each_hour"].to_list()).T
    days = int(data.shape[0] / 24)
    train_days = int(args.train_ratio * days)
    valid_days = int(days*args.valid_ratio)
    test_days = days-train_days-valid_days
    
    train_data = data[:train_days*24].to_numpy()
    valid_data = data[train_days*24:(train_days + valid_days)*24].to_numpy()
    test_data = data[(train_days + valid_days)*24:(train_days + valid_days+test_days)*24].to_numpy()
    
    return train_data, valid_data, test_data, df

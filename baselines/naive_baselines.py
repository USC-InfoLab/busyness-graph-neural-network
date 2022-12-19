import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from utils.data_utils import get_full_df

SEED = 117
TOTAL_DAYS = 400
START_POI = 0
END_POI = 400
TRAIN_RATIO = 0.7
VALID_RATIO = 0.2
TEST_RATIO = 1 - (TRAIN_RATIO + VALID_RATIO)
START_DATE = '2018-12-31'
WINDOW_SIZE = 24
HORIZON = 6

CITY = 'Houston'
OUTPUT_DIRECTORY = f'./output/baselines/{CITY}'


def get_data(city):
    csv_path = f'/home/users/arash/datasets/safegraph/weekly_patterns_2019-01-07_2020-06-08_{city}.csv'
    poi_info_csv_path = '/home/users/arash/datasets/safegraph/core_poi_info_2019-01-07_2020-06-08.csv'
    
    df = get_full_df(csv_path_weekly=csv_path, 
                         poi_info_csv_path=poi_info_csv_path, 
                         start_row=START_POI, end_row=END_POI, 
                         total_days=TOTAL_DAYS,
                         city=city)
    
    data = pd.DataFrame(df["visits_by_each_hour"].to_list()).T
    date_format = r'%Y-%m-%d'
    start_time = datetime.strptime(START_DATE, date_format)
    end_time = start_time + timedelta(days=TOTAL_DAYS)
    time_span = pd.date_range(start_time, end_time, freq='1H', closed='left').to_numpy()
    data = data.set_index(time_span)
    train_days = int(TRAIN_RATIO * TOTAL_DAYS)
    valid_days = int(TOTAL_DAYS * VALID_RATIO)
    test_days = TOTAL_DAYS - train_days - valid_days
    valid_dates = time_span[train_days*24 + WINDOW_SIZE:(train_days + valid_days)*24 - HORIZON + 1]
    
    return data, valid_dates


def get_target(data, valid_dates, save=False):
    target = data.loc[valid_dates]
    save_path = 'output/target.csv'
    if save:
        target.to_csv(save_path, index=False)
    return target
    
    
    
def naive_seasonal_fcast(data, valid_dates, shift_days=7, save=False):
    output_name = 'naive_seasonal_forecasts.csv'
    save_path = os.path.join(OUTPUT_DIRECTORY, output_name)   
    forecasts = data.shift(shift_days, freq="1D")
    valid_fcasts = forecasts.loc[valid_dates]
    if save:
        valid_fcasts.to_csv(save_path, index=False)
    return valid_fcasts


def historical_avg_fcast(data, valid_dates, num_weeks = 4, save=False):
    output_name = 'historical_avg_forecasts.csv'
    save_path = os.path.join(OUTPUT_DIRECTORY, output_name)   
    forecasts = data.shift(7, freq='1D')
    for timestamp in valid_dates[:1]:
        for week in range(1, num_weeks):
            forecasts.loc[timestamp] += forecasts.loc[timestamp - np.timedelta64(7*week, 'D')]
    forecasts = (forecasts/num_weeks).round().astype('int')
    valid_fcasts = forecasts.loc[valid_dates]
    if save:
        valid_fcasts.to_csv(save_path, index=False)
    return valid_fcasts


def main():
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    data, valid_dates = get_data(CITY)
    target = get_target(data, valid_dates, save=True)
    # naive_forecasts = naive_seasonal_fcast(data, valid_dates, save=True)
    # hist_avg_fcasts = historical_avg_fcast(data, valid_dates, save=True)
    
    
    
if __name__ == '__main__':
    main()
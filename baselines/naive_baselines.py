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

def masked_MAPE(v, v_, axis=None):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAPE averages on all elements of input.
    '''
    # TODO: Remove this line
    v_ = np.where(v_<0, 0,v_)
    
    mask = (v < 1E-5)
    percentage = np.abs(v_ - v) / np.abs(v)
    percentage = np.where(percentage > 5, 5, percentage) # TODO remove this
    if np.any(mask):
        masked_array = np.ma.masked_array(percentage, mask=mask)  # mask the dividing-zero as invalid
        result = masked_array.mean(axis=axis)
        if isinstance(result, np.ma.MaskedArray):
            return result.filled(np.nan)
        else:
            return result
    return np.mean(percentage, axis).astype(np.float64)


def MAPE(v, v_, axis=None):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAPE averages on all elements of input.
    '''
    # TODO: Remove this line
    v_ = np.where(v_<0, 0,v_)
    mape = (np.abs(v_ - v) / np.abs(v)+1e-5).astype(np.float64)
    mape = np.where(mape > 5, 5, mape)
    return np.mean(mape, axis)


def RMSE(v, v_, axis=None):
    '''
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, RMSE averages on all elements of input.
    '''
    return np.sqrt(np.mean((v_ - v) ** 2, axis)).astype(np.float64)




def MAE(v, v_, axis=None):
    '''
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAE averages on all elements of input.
    '''

    return np.mean(np.abs(v_ - v), axis).astype(np.float64)


def evaluate(y, y_hat, by_step=False, by_node=False):
    '''
    :param y: array in shape of [count, time_step, node].
    :param y_hat: in same shape with y.
    :param by_step: evaluate by time_step dim.
    :param by_node: evaluate by node dim.
    :return: array of mape, mae and rmse.
    '''
    if not by_step and not by_node:
        return masked_MAPE(y, y_hat), MAE(y, y_hat), RMSE(y, y_hat)
    if by_step and by_node:
        return masked_MAPE(y, y_hat, axis=0), MAE(y, y_hat, axis=0), RMSE(y, y_hat, axis=0)
    if by_step:
        return masked_MAPE(y, y_hat, axis=(0, 2)), MAE(y, y_hat, axis=(0, 2)), RMSE(y, y_hat, axis=(0, 2))
    if by_node:
        return masked_MAPE(y, y_hat, axis=(0, 1)), MAE(y, y_hat, axis=(0, 1)), RMSE(y, y_hat, axis=(0, 1))


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


def get_target(data, city, valid_dates, save=False):
    target = data.loc[valid_dates]
    save_path = f'./output/baselines/{CITY}/target.csv'
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


def eval(city, start_poi, end_poi):
    hist_avg_val_prediction_path = f'output/baselines/{city}/historical_avg_forecasts.csv'
    naive_seas_val_prediction_path = f'output/baselines/{city}/naive_seasonal_forecasts.csv'
    val_target_path = f'output/baselines/{city}/target.csv'
    hist_avg_val_predict_df = pd.read_csv(hist_avg_val_prediction_path, header=None).to_numpy()[:, start_poi:end_poi]
    naive_seas_val_predict_df = pd.read_csv(naive_seas_val_prediction_path, header=None).to_numpy()[:, start_poi:end_poi]
    val_target_df = pd.read_csv(val_target_path, header=None).to_numpy()[:, start_poi:end_poi]
    # mape_hist_avg, mae_hist_avg, rmse_hist_avg = evaluate(val_target_df, hist_avg_val_predict_df)
    # mape_naive_seas, mae_naive_seas, rmse_naive_seas = evaluate(val_target_df, naive_seas_val_predict_df)
    # print(f'Eval {city:13s}: MAPE: {mape:.3f} | MAE: {mae} | RMSE: {rmse}')
    hist_avg_res = evaluate(val_target_df, hist_avg_val_predict_df)
    naive_seas_res = evaluate(val_target_df, naive_seas_val_predict_df)
    return hist_avg_res, naive_seas_res


def eval_all_cities(start_poi=0, end_poi=400):
    print(f'START POI: {start_poi} | END POI: {end_poi}')
    cities = ['Houston', 'Los Angeles', 'Chicago', 'New York', 'San Antonio']
    hist_avg_all_res = []
    naive_seas_all_res = []
    for city in cities:
        hist_avg_res, naive_seas_res = eval(city, start_poi, end_poi)
        hist_avg_all_res.append(hist_avg_res)
        naive_seas_all_res.append(naive_seas_res)
    print('Historical Average Results:')
    for i, (mape, mae, rmse) in enumerate(hist_avg_all_res):
        print(f'{cities[i]:13s}: MAPE: {mape:.3f} | MAE: {mae:.3f} | RMSE: {rmse:.3f}')
    print('-----------------------')
    print('Naive Seasonal Results:')
    for i, (mape, mae, rmse) in enumerate(naive_seas_all_res):
        print(f'{cities[i]:13s}: MAPE: {mape:.3f} | MAE: {mae:.3f} | RMSE: {rmse:.3f}')


        


def main():
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    # data, valid_dates = get_data(CITY)
    # target = get_target(data, CITY, valid_dates, save=True)
    # naive_forecasts = naive_seasonal_fcast(data, valid_dates, save=True)
    # hist_avg_fcasts = historical_avg_fcast(data, valid_dates, save=True)
    eval_all_cities()
    print('\n\n\n\n')
    eval_all_cities(0, 20)
    
    
    
if __name__ == '__main__':
    main()
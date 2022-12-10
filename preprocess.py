import os
import json
import numpy as np
from scipy import stats

SCRIPT_DIR_PATH = os.path.dirname(__file__)
DATA_DIR_REL_PATH = 'data/'
RAW_DATA_DIR_REL_PATH = 'data/raw/'
DATA_DIR_ABS_PATH = os.path.join(SCRIPT_DIR_PATH, DATA_DIR_REL_PATH)
RAW_DATA_DIR_ABS_PATH = os.path.join(SCRIPT_DIR_PATH, RAW_DATA_DIR_REL_PATH)

FFT_WINDOW_SIZE = 14
SEQUENCE_LENGTH = 14
PREDICTION_WINDOW_SIZE = 7
TRAIN_SPLIT = 0.95


def load_raw_data():
    f_daily_avg_gas_limit = open(os.path.join(RAW_DATA_DIR_ABS_PATH, 'dailyavggaslimit.json'), 'r')
    f_daily_avg_gas_price = open(os.path.join(RAW_DATA_DIR_ABS_PATH, 'dailyavggasprice.json'), 'r')
    f_daily_gas_used = open(os.path.join(RAW_DATA_DIR_ABS_PATH, 'dailygasused.json'), 'r')
    f_daily_txn_fee = open(os.path.join(RAW_DATA_DIR_ABS_PATH, 'dailytxnfee.json'), 'r')
    f_eth_daily_market_cap = open(os.path.join(RAW_DATA_DIR_ABS_PATH, 'ethdailymarketcap.json'), 'r')
    f_eth_daily_price = open(os.path.join(RAW_DATA_DIR_ABS_PATH, 'ethdailyprice.json'), 'r')

    json_daily_avg_gas_limit = json.load(f_daily_avg_gas_limit)
    json_daily_avg_gas_price = json.load(f_daily_avg_gas_price)
    json_daily_gas_used = json.load(f_daily_gas_used)
    json_daily_txn_fee = json.load(f_daily_txn_fee)
    json_eth_daily_market_cap = json.load(f_eth_daily_market_cap)
    json_eth_daily_price = json.load(f_eth_daily_price)

    f_daily_avg_gas_limit.close()
    f_daily_avg_gas_price.close()
    f_daily_gas_used.close()
    f_daily_txn_fee.close()
    f_eth_daily_market_cap.close()
    f_eth_daily_price.close()

    daily_avg_gas_limit_data = np.asarray([[d['unixTimeStamp'], d['gasLimit']] for d in json_daily_avg_gas_limit['result']][8:], dtype=np.float64)
    daily_avg_gas_price_data = np.asarray([[d['unixTimeStamp'], d['avgGasPrice_Wei']] for d in json_daily_avg_gas_price['result']][8:], dtype=np.float64)
    daily_gas_used_data = np.asarray([[d['unixTimeStamp'], d['gasUsed']] for d in json_daily_gas_used['result']][8:], dtype=np.float64)
    daily_txn_fee_data = np.asarray([[d['unixTimeStamp'], d['transactionFee_Eth']] for d in json_daily_txn_fee['result']][8:], dtype=np.float64)
    eth_daily_market_cap_data = np.asarray([[d['unixTimeStamp'], d['marketCap']] for d in json_eth_daily_market_cap['result']][8:], dtype=np.float64)
    eth_daily_price_data = np.asarray([[d['unixTimeStamp'], d['value']] for d in json_eth_daily_price['result']][8:], dtype=np.float64)

    daily_avg_gas_limit_data.sort(axis=0)
    daily_avg_gas_price_data.sort(axis=0)
    daily_gas_used_data.sort(axis=0)
    daily_txn_fee_data.sort(axis=0)
    eth_daily_market_cap_data.sort(axis=0)
    eth_daily_price_data.sort(axis=0)

    daily_avg_gas_limit_data = stats.zscore(daily_avg_gas_limit_data)
    daily_avg_gas_price_data = stats.zscore(daily_avg_gas_price_data)
    daily_gas_used_data = stats.zscore(daily_gas_used_data)
    daily_txn_fee_data = stats.zscore(daily_txn_fee_data)
    eth_daily_market_cap_data = stats.zscore(eth_daily_market_cap_data)
    # eth_daily_price_data = stats.zscore(eth_daily_price_data) # DO NOT DO THIS

    return (daily_avg_gas_limit_data[:, 1], daily_avg_gas_price_data[:, 1], daily_gas_used_data[:, 1], daily_txn_fee_data[:, 1], eth_daily_market_cap_data[:, 1], eth_daily_price_data[:, 1])


def get_price_ffts(eth_daily_price_data):
    windows = []
    for i in range(0, eth_daily_price_data.shape[0] - FFT_WINDOW_SIZE + 1):
        window = eth_daily_price_data[i:i + FFT_WINDOW_SIZE]
        windows += [stats.zscore(window)]
    windows = np.vstack(windows)
    return np.abs(np.fft.fft(windows))


def get_preprocessed_data(full_sequence):
    windows, y = [], []
    for i in range(0, full_sequence.shape[0] - SEQUENCE_LENGTH + 1 - PREDICTION_WINDOW_SIZE):
        window = full_sequence[i:i + SEQUENCE_LENGTH, :]
        windows += [window]
        prediction_window = full_sequence[i + SEQUENCE_LENGTH:i + SEQUENCE_LENGTH + PREDICTION_WINDOW_SIZE, 5]
        y += [[np.amin(prediction_window), np.mean(prediction_window), np.amax(prediction_window)]]
    return (np.stack(windows), np.array(y))


def split_data(data, y):
    train_index = int(TRAIN_SPLIT * data.shape[0])
    # dev_index = train_index + int((data.shape[0] - train_index) / 2) + 1
    # return (data[:train_index], data[train_index:dev_index], data[dev_index:], y[:train_index], y[train_index:dev_index], y[dev_index:])
    return (data[:train_index], data[train_index:], y[:train_index], y[train_index:])


if __name__ == '__main__':
    raw_data = load_raw_data()
    eth_daily_price_data = raw_data[-1]
    price_ffts = get_price_ffts(eth_daily_price_data)
    full_sequence = np.concatenate((np.stack(raw_data, axis=1)[FFT_WINDOW_SIZE - 1:, :], price_ffts), axis=1)
    preprocessed_data, y = get_preprocessed_data(full_sequence)
    # X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(preprocessed_data, y)
    X_train, X_dev, y_train, y_dev = split_data(preprocessed_data, y)

    print('X_train.shape = ', X_train.shape)
    print('y_train.shape = ', y_train.shape)
    print('X_dev.shape   = ', X_dev.shape)
    print('y_dev.shape   = ', y_dev.shape)
    # print('X_test.shape  = ', X_test.shape)
    # print('y_test.shape  = ', y_test.shape)

    np.save(open(os.path.join(DATA_DIR_ABS_PATH, 'X_train.npy'), 'wb'), X_train)
    np.save(open(os.path.join(DATA_DIR_ABS_PATH, 'y_train.npy'), 'wb'), y_train)
    np.save(open(os.path.join(DATA_DIR_ABS_PATH, 'X_dev.npy'), 'wb'), X_dev)
    np.save(open(os.path.join(DATA_DIR_ABS_PATH, 'y_dev.npy'), 'wb'), y_dev)
    # np.save(open(os.path.join(DATA_DIR_ABS_PATH, 'X_test.npy'), 'wb'), X_test)
    # np.save(open(os.path.join(DATA_DIR_ABS_PATH, 'y_test.npy'), 'wb'), y_test)

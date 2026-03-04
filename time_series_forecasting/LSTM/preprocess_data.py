import math
import numpy as np
from load_data import load_data
from sklearn.preprocessing import MinMaxScaler
import torch

def preprocess_data():
    df = load_data()

    dataset = df[["temperature", "humidity", "wind_speed", "pressure", "cloudiness", "solar_radiance"]].values  

    training_data_len = math.ceil(len(dataset) * .8)
    print("Number of data training points: ", training_data_len)

    dataset_train = dataset[:training_data_len]
    dataset_test  = dataset[training_data_len:]
    print("Training data:", dataset_train.shape, "| Test data:", dataset_test.shape)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(dataset_train)
    scaled_test  = scaler.transform(dataset_test)

    print("Scaled training dataset (first 5 rows):\n", scaled_train[:5])
    print("Scaled test dataset (first 5 rows):\n",     scaled_test[:5])
    print()

    total_len      = len(dataset)
    steps_per_day  = 8          # 24hrs / 3hr steps
    steps_per_year = 8 * 365    # full annual cycle

    hour_indices = np.arange(total_len)

    # Hour-of-day cycle — captures daily temperature rise/fall pattern
    hour_sin = np.sin(2 * np.pi * hour_indices / steps_per_day).reshape(-1, 1)
    hour_cos = np.cos(2 * np.pi * hour_indices / steps_per_day).reshape(-1, 1)

    # Day-of-year cycle — captures seasonal/annual temperature pattern
    day_sin = np.sin(2 * np.pi * hour_indices / steps_per_year).reshape(-1, 1)
    day_cos = np.cos(2 * np.pi * hour_indices / steps_per_year).reshape(-1, 1)

    # Split all time features at the same training boundary
    hour_sin_train, hour_sin_test = hour_sin[:training_data_len], hour_sin[training_data_len:]
    hour_cos_train, hour_cos_test = hour_cos[:training_data_len], hour_cos[training_data_len:]
    day_sin_train,  day_sin_test  = day_sin[:training_data_len],  day_sin[training_data_len:]
    day_cos_train,  day_cos_test  = day_cos[:training_data_len],  day_cos[training_data_len:]

    
    scaled_train = np.concatenate([
        scaled_train, hour_sin_train, hour_cos_train, day_sin_train, day_cos_train
    ], axis=1)
    scaled_test = np.concatenate([
        scaled_test, hour_sin_test, hour_cos_test, day_sin_test, day_cos_test
    ], axis=1)

    sequence_length = 72

    x_train, y_train = [], []
    for i in range(len(scaled_train) - sequence_length):
        x_train.append(scaled_train[i:i + sequence_length])
        y_train.append(scaled_train[i + sequence_length, 0])  
    x_train = torch.tensor(np.array(x_train), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train), dtype=torch.float32).unsqueeze(1)
    print(x_train.shape, y_train.shape)  

    x_test, y_test = [], []
    for i in range(len(scaled_test) - sequence_length):
        x_test.append(scaled_test[i:i + sequence_length])
        y_test.append(scaled_test[i + sequence_length, 0])
    x_test = torch.tensor(np.array(x_test), dtype=torch.float32)
    y_test = torch.tensor(np.array(y_test),  dtype=torch.float32).unsqueeze(1)
    print(x_test.shape, y_test.shape)   

    return x_train, y_train, x_test, y_test, scaler
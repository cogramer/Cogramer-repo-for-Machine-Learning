import math
import numpy as np
import pandas as pd
from load_data import load_data
from sklearn.preprocessing import MinMaxScaler
import torch

def preprocess_data():
    df = load_data()

    # ── MUST happen before reindexing ────────────────────────────────────────────
    # After asfreq every row is exactly 3h apart, so diff() becomes constant
    # Compute original gaps here while the raw timestamps still reflect reality
    df = df.set_index("datetime").sort_index()

    original_index = df.index  # save original timestamps before reindexing
    original_solar = df["solar_radiance"].copy() # save original solar for later use in nighttime masking

    # ── Reindex to fixed 3h grid ─────────────────────────────────────────────────
    full_index = pd.date_range(
        start=original_index.min(),
        end=original_index.max(),
        freq="3h"
    )
    df = df.reindex(full_index)  # missing rows become NaN

    # ── Missingness mask — BEFORE interpolation ──────────────────────────────────
    # 0 = originally observed, 1 = imputed
    # Computed now while NaN rows still exist
    missingness_mask = (
        df["temperature"]
        .isna()
        .to_numpy(dtype=np.float32)
        .reshape(-1, 1)
    )

    # ── Delta-time feature — computed from original observation gaps ─────────────
    # Map each row in the full 3h grid back to whether it existed in the raw data.
    # Observed rows start at the normal 3h interval.
    # Missing rows accumulate elapsed time since the last observed point.
    full_series = pd.Series(0, index=full_index)

    gap_hours = np.where(
        full_series.index.isin(original_index),
        3.0,    # observed row — normal sampling interval
        np.nan  # missing row — will accumulate from previous observation
    ).astype(float)

    # Forward-fill cumulative gap length:
    # each consecutive missing row adds another 3h.
    gap_series = pd.Series(gap_hours, index=full_index)

    for i in range(1, len(gap_series)):
        if np.isnan(gap_series.iloc[i]):
            gap_series.iloc[i] = gap_series.iloc[i - 1] + 3.0

    delta_hours = gap_series.to_numpy(dtype=np.float32).reshape(-1, 1)

    # ── Physically aware interpolation ───────────────────────────────────────────
    # Linear for most variables — valid for temperature, humidity, wind, etc...
    df["temperature"]   = df["temperature"].interpolate(method="time")
    df["humidity"]      = df["humidity"].interpolate(method="time")
    df["wind_speed"]    = df["wind_speed"].interpolate(method="time")
    df["pressure"]      = df["pressure"].interpolate(method="time")
    df["cloudiness"]    = df["cloudiness"].interpolate(method="time")

    # Nightime must be 0, Linear interpolation across a night gap would produce wrong positive values
    solar = df["solar_radiance"].copy()
    solar = solar.interpolate(method="time")

    # Learn which hours are "day" based on when solar radiance was actually observed in the raw data
    positive_solar = original_solar[original_solar > 0.0]

    # Daylight hours are those where solar radiance was observed to be positive at least once in the raw data
    daylight_hours = (
        positive_solar.groupby(
            pd.to_datetime(positive_solar.index).hour
        )
        .size()
        .index
    )

    # Everything else is forced to zero
    night_mask = ~full_index.hour.isin(daylight_hours)

    solar[night_mask] = 0.0
    df["solar_radiance"] = solar

    df = df.reset_index().rename(columns={"index": "datetime"})

    total_rows = len(df)
    imputed_count = int(missingness_mask.sum())
    print(f"Total timesteps after reindex: {total_rows}")
    print(f"Rows imputed: {imputed_count} ({imputed_count/total_rows*100:.1f}%)")

    # ── Timestamp-based cyclic features ──────────────────────────────────────────
    hours     = df["datetime"].dt.hour.to_numpy(dtype=np.float32)
    dayofyear = df["datetime"].dt.dayofyear.to_numpy(dtype=np.float32)

    hour_sin = np.sin(2 * np.pi * hours     / 24 ).reshape(-1, 1)
    hour_cos = np.cos(2 * np.pi * hours     / 24 ).reshape(-1, 1)
    day_sin  = np.sin(2 * np.pi * dayofyear / 365).reshape(-1, 1)
    day_cos  = np.cos(2 * np.pi * dayofyear / 365).reshape(-1, 1)

    # ── Scale delta separately ────────────────────────────────────────────────────
    delta_scaler = MinMaxScaler(feature_range=(0, 1))
    delta_scaled = delta_scaler.fit_transform(delta_hours)

    # ── Weather variables ─────────────────────────────────────────────────────────
    dataset = df[["temperature", "humidity", "wind_speed",
                  "pressure", "cloudiness", "solar_radiance"]].to_numpy(dtype=np.float32)  # (N, 6)

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

    # ── Split all auxiliary features at training boundary ────────────────────────
    def split(arr):
        return arr[:training_data_len], arr[training_data_len:]

    hour_sin_train,    hour_sin_test    = split(hour_sin)
    hour_cos_train,    hour_cos_test    = split(hour_cos)
    day_sin_train,     day_sin_test     = split(day_sin)
    day_cos_train,     day_cos_test     = split(day_cos)
    delta_train,       delta_test       = split(delta_scaled)
    mask_train,        mask_test        = split(missingness_mask)

    # ── Concatenate everything ────────────────────────────────────────────────────
    # Columns: N weather features, 4 cyclic features, delta, mask
    scaled_train = np.concatenate([
        scaled_train, hour_sin_train, hour_cos_train,
        day_sin_train, day_cos_train, delta_train, mask_train
    ], axis=1)
    scaled_test = np.concatenate([
        scaled_test, hour_sin_test, hour_cos_test,
        day_sin_test, day_cos_test, delta_test, mask_test
    ], axis=1)

    # ── Build sequences, skipping windows with large gaps ────────────────────────
    sequence_length = 72
    max_gap_hours   = 12  # skip any sequence containing a gap larger than this

    x_train, y_train = [], []
    for i in range(len(scaled_train) - sequence_length):
        window_delta = delta_hours[i:i + sequence_length]  # raw hours, not scaled
        if float(window_delta.max()) > max_gap_hours:
            continue  
        x_train.append(scaled_train[i:i + sequence_length])
        y_train.append(scaled_train[i + sequence_length, 0])

    x_train = torch.tensor(np.array(x_train), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train), dtype=torch.float32).unsqueeze(1)
    print(x_train.shape, y_train.shape)  # expect (N, sequence_length, feature_count), (N, output_count)

    x_test, y_test = [], []
    for i in range(len(scaled_test) - sequence_length):
        window_delta = delta_hours[training_data_len + i:
                                   training_data_len + i + sequence_length]
        if float(window_delta.max()) > max_gap_hours:
            continue
        x_test.append(scaled_test[i:i + sequence_length])
        y_test.append(scaled_test[i + sequence_length, 0])

    x_test = torch.tensor(np.array(x_test), dtype=torch.float32)
    y_test = torch.tensor(np.array(y_test),  dtype=torch.float32).unsqueeze(1)
    print(x_test.shape, y_test.shape)    # expect (N, sequence_length, feature_count), (N, output_count)

    return x_train, y_train, x_test, y_test, scaler
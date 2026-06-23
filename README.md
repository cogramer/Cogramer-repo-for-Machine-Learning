# Weather Forecasting with LSTM

This is a multi-feature time series forecasting project using LSTM (Long Short-Term Memory) networks for weather prediction.

The models predicts temperature using historical meteorological variables and cyclical time encoding, trained on OpenWeatherMap forecast archives.

![OpenWeather Logo](docs\pictures\logo_white.png)

Link to [OpenWeatherMap](https://openweathermap.org/)

Link to [model training results.](https://docs.google.com/spreadsheets/d/17tpbZEZOtY72469nGBVYa6CG-Byimf8oFetg2LYOqYw/edit?usp=sharing)

---

> [!IMPORTANT]
> To use the models:
> 1. Install [Python 3.11.14](https://www.python.org/downloads/release/python-31114/)
> 2. Install dependencies from requirements.txt
> 3. Run main.py

> [!TIP]
> - Install [SQLite](https://sqlite.org/index.html) to access database files
> - You can change the database and tables used in load_data.py
> - CONFIG.py contains the configurations for training the model, except for sequence_length and max_gap_hours (both found in preprocess_data.py)

---

## Dataset

Weather data is sourced from OpenWeatherMap using the 5-day / 3-hour forecast API.

Link to the website: https://openweathermap.org/

Forecast target location: Ho Chi Minh City

Current archived forecast data covers: 2024-09-17 → 2026-06-25 = **647 days**.

Sampling interval is **3 hours** with an expected frequency of **8 timesteps per day**.

## Input Features

Base meteorological features:

- Temperature
- Humidity
- Wind Speed
- Pressure
- Cloudiness
- Solar Radiance

Auxiliary temporal features:

- Hour sine
- Hour cosine
- Day-of-year sine
- Day-of-year cosine

Additional robustness features (Alpha only):

- Delta-time feature
- Missingness mask

---

## Model Architecture

Both versions use the same LSTM backbone:

```sh
LSTM(
    input_size,
    hidden_size=128,
    num_layers=2,
    dropout=0.2
)
```

**Output:** Single-step temperature forecast

**Sequence length:** 72 timesteps (9 days)

**Train/test split:** 80/20

## Training Configuration

| Feature                      | Purpose                      |
| -----------------------------|------------------------------|
| Huber Loss                   | Robust against outliers      |
| Adam optimizer               | Adaptive learning            |
| LR scheduler                 | Automatic fine-tuning        |
| Gradient clipping            | Stabilizes LSTM training     |
| Early stopping               | Prevents overfitting         |
| Checkpoint saving            | Preserves best model         |
| Mini-batch training          | Efficient learning           |

## Model Variants

---

### `LSTMv0.0.9-beta`

Baseline preprocessing pipeline.

Characteristics:

```sh

```

- Assumes perfect 3-hour timestep continuity
- Uses row-position based cyclical encoding
- No missing timestep handling
- No interpolation
- No missingness awareness

Uses synthetic timestep indices for time encoding:

```sh
hour_indices = np.arange(total_len)
```

Cyclical features:

- Daily cycle
- Seasonal cycle

Advantages:

- Simpler preprocessing
- Faster pipeline
- Works well on complete datasets

Limitations:

- Breaks temporal correctness when gaps exist
- Cannot detect irregular timestep spacing

---

### `LSTMv0.0.9-alpha`

Gap-aware preprocessing pipeline. Built to handle missing timesteps and irregular forecast archives.

#### Added Improvements:

1. **Fixed-grid reindexing:** forces a strict 3-hour interval grid. Missing timestamps are explicitly inserted.
2. **Missingness mask:** tracks whether a timestep was observed (0) or imputed (1). This allows the model to distinguish synthetic values.
3. **Delta-time feature:** encodes elapsed time since last observation, allowing temporal gap awareness. Example: 3h, 3h, 6h, 9h...
4. **Physically-aware interpolation:**

- Standard linear interpolation:

   - temperature
   - humidity
   - wind speed
   - pressure
   - cloudiness

- Solar radiance: interpolated separately and forced to zero during non-daylight hours. This prevents physically impossible nighttime solar values.

5. **Large-gap window filtering:** any sequence containing a gap of over **12 hours** is skipped. Best to avoid training on heavily reconstructed segments.

#### Advantages:

- Robust to missing rows
- Preserves temporal integrity
- Better suited for long-running datasets

#### Limitations:

- More preprocessing complexity
- Slightly higher variance between runs

---

## Evaluation Datasets

### Dataset (1): Complete

2025-01-12 → 2025-12-23 = **346 days**

**Total:** 2760 timesteps

**Missing rows:** 0

### Dataset (2): Sparse

2024-09-18 → 2026-06-24 = **645 days**

**Total:** 5152 timesteps

**Missing rows:** 86

Missing rows are scattered throughout the archive.

## Results

### Dataset (1): Complete Data

| Model | Test Loss |  MAPE |   MAE |
| ----- | --------: | ----: | ----: |
| Beta  |    0.0018 | 2.90% | 56.7% |
| Alpha |    0.0018 | 2.92% | 56.3% |

#### Notes:

- Performance is nearly identical.
- Alpha occasionally achieves slightly better MAE.
- Complete datasets reduce the value of gap-aware preprocessing.

Dataset 1 - Beta Loss

![Dataset 1 - Beta Loss](docs\pictures\test_result_02.png)

Dataset 1 - Alpha Loss

![Dataset 1 - Alpha Loss](docs\pictures\test_result_07.png)

### Dataset (2): Sparse Data

| Model | Test Loss |  MAPE |   MAE |
| ----- | --------: | ----: | ----: |
| Beta  |    0.0014 | 2.59% | 63.5% |
| Alpha |    0.0012 | 2.33% | 67.0% |

#### Notes:

- Both models improve with larger training volume.
- Alpha achieves lower test loss and lower MAPE.
- Alpha shows stronger sensitivity to preprocessing quality.
- Variance between repeated runs is noticeably higher than Beta.

***=> Alpha extracts more signal from incomplete temporal data, but is more dependent on preprocessing consistency.***

Dataset 2 - Beta Loss

![Dataset 1 - Beta Loss](docs\pictures\test_result_09.png)

Dataset 2 - Alpha Loss

![Dataset 1 - Alpha Loss](docs\pictures\test_result_08.png)

---

## In conclusion

**LSTMv0.0.9-beta** is a strong baseline for complete and clean archives.

**LSTMv0.0.9-alpha** is more robust for long-term, imperfect real-world forecast archives.

In environments with:

- missing timesteps
- irregular sampling
- forecast collection failures

***Alpha provides better temporal fidelity and stronger generalization potential.***

***=> Future updates will be based on model Alpha.***

---

## Future work:

- [ ] Multi-target forecasting (temperature + humidity + wind)
- [ ] Attention-based sequence models
- [ ] Transformer architectures
- [ ] Forecast horizon extension (3h → 24h+)
- [ ] Better physically-informed solar modeling
- [ ] Confidence interval estimation

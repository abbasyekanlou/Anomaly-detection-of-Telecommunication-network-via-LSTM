# Cellular Network Traffic Forecasting and Anomaly Detection Using LSTM

A TensorFlow portfolio project for forecasting cellular Internet activity and identifying unusual traffic patterns using a Long Short-Term Memory (LSTM) neural network.

The model uses the previous **24 hours** of cellular Internet activity to predict the next **60 minutes**:

- Sampling interval: 10 minutes
- Input sequence: 144 time steps
- Forecast horizon: 6 time steps
- Input tensor shape: `(samples, 144, 1)`
- Output tensor shape: `(samples, 6)`

## Project overview

Accurate cellular traffic forecasting can support network capacity planning, proactive resource allocation, congestion management, load balancing, and energy-efficient network operation.

This project studies a univariate forecasting problem for one Milan grid square:

> Given the previous 24 hours of Internet activity, predict the next six 10-minute activity values.

Candidate anomalies are then identified when the absolute difference between the observed and predicted activity exceeds a threshold estimated from validation data.

## Dataset

This project uses a one-week subset of the **Telecom Italia Big Data Challenge** telecommunications dataset for Milan.

Kaggle subset:

[Italian Telecom Data 2013 — One Week](https://www.kaggle.com/datasets/ocanaydin/italian-telecom-data-2013-1week)

Original dataset publication:

> G. Barlacchi, M. De Nadai, R. Larcher, et al., “A multi-source dataset of urban life in the city of Milan and the Province of Trentino,” *Scientific Data*, vol. 2, article 150055, 2015.  
> DOI: [10.1038/sdata.2015.55](https://doi.org/10.1038/sdata.2015.55)

The one-week subset contains daily CSV files from November 4 to November 10, 2013. The original records include:

| Column | Description |
|---|---|
| `square_id` | Milan geographical grid-square identifier |
| `timestamp` | Start of the 10-minute interval in Unix milliseconds |
| `country_code` | Country associated with the communication |
| `sms_in` | Incoming SMS activity |
| `sms_out` | Outgoing SMS activity |
| `call_in` | Incoming call activity |
| `call_out` | Outgoing call activity |
| `internet` | Internet connection activity |

This project:

- Selects grid square `5161`
- Uses the `internet` activity field
- Aggregates records across country codes
- Builds a continuous 10-minute time series
- Interpolates missing intervals when necessary

The Internet field represents anonymized CDR-based activity, not measured throughput in Mbps or transferred traffic volume in GB.

## Dataset setup

Download and extract the seven CSV files into:

```text
data/
└── raw/
    ├── sms-call-internet-mi-2013-11-04.csv
    ├── sms-call-internet-mi-2013-11-05.csv
    ├── sms-call-internet-mi-2013-11-06.csv
    ├── sms-call-internet-mi-2013-11-07.csv
    ├── sms-call-internet-mi-2013-11-08.csv
    ├── sms-call-internet-mi-2013-11-09.csv
    └── sms-call-internet-mi-2013-11-10.csv
```

The raw dataset is not included in this repository because of its size and separate data license.

## Methodology

### Data preparation

1. Load the daily CSV files in chunks.
2. Select one Milan grid square.
3. Aggregate Internet activity by timestamp.
4. Convert timestamps to Milan local time.
5. Create a continuous 10-minute time series.
6. Divide the observations chronologically:
   - 70% training
   - 15% validation
   - 15% testing
7. Fit `MinMaxScaler` using only the training period.
8. Generate sliding windows.

The data is not randomly shuffled before splitting, preventing future observations from leaking into the training set.

### Forecasting model

```text
Input: previous 144 observations
          ↓
LSTM layer: 64 units
          ↓
Dense layer: 6 outputs
          ↓
Forecast: next 60 minutes
```

TensorFlow model:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(144, 1)),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(6),
])
```

Training configuration:

| Parameter | Value |
|---|---:|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | Mean squared error |
| Training metric | Mean absolute error |
| Batch size | 32 |
| Maximum epochs | 100 |
| Early stopping patience | 10 epochs |

### Baseline

The LSTM is compared with a daily seasonal baseline:

> Predict the next hour using the activity observed at the same time on the previous day.

A forecasting model should be compared against a simple baseline because strong daily seasonality can make the previous-day forecast difficult to outperform.

## Initial results

Results from the initial seven-day experiment:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Previous-day baseline | 251.09 | 356.42 | 0.9467 |
| LSTM | 369.01 | 540.85 | 0.8773 |

The LSTM did not outperform the previous-day baseline in the initial experiment. This is an important experimental result rather than a software error.

The primary limitation is the small dataset: only seven days are available, leaving approximately five days for training after chronological splitting. Daily cellular activity is strongly seasonal, which gives the previous-day baseline a major advantage.

Results can vary slightly between runs and TensorFlow environments.

### Forecast error by horizon

The initial LSTM error increased as the forecast horizon became longer:

| Forecast horizon | MAE |
|---:|---:|
| 10 minutes | 326.12 |
| 20 minutes | 338.75 |
| 30 minutes | 351.24 |
| 40 minutes | 380.35 |
| 50 minutes | 401.21 |
| 60 minutes | 416.37 |

## Anomaly detection

The project uses forecast residuals for unsupervised candidate anomaly detection.

For each first-step prediction:

```text
absolute error = |actual activity - predicted activity|
```

The anomaly threshold is the 99th percentile of the validation absolute errors:

```python
threshold = np.percentile(validation_error, 99)
```

A test point is marked as a candidate anomaly when its absolute error exceeds this threshold.

The dataset does not provide verified anomaly labels. Therefore, detected observations are described as **candidate anomalies**, not confirmed outages, attacks, or network failures.

## Visualizations

The script generates the following figures:

```text
outputs/
└── figures/
    ├── 01_data_split.png
    ├── 02_training_loss.png
    ├── 03_forecast_example.png
    ├── 04_horizon_mae.png
    └── 05_test_anomalies.png
```


## Citation

When using the dataset, cite the original publication:

```bibtex
@article{barlacchi2015multisource,
  title={A multi-source dataset of urban life in the city of Milan and the Province of Trentino},
  author={Barlacchi, Gianni and De Nadai, Marco and Larcher, Roberto and Casella, Antonio and Chitic, Cristiana and Torrisi, Giovanni and Antonelli, Fabrizio and Vespignani, Alessandro and Pentland, Alex and Lepri, Bruno},
  journal={Scientific Data},
  volume={2},
  pages={150055},
  year={2015},
  doi={10.1038/sdata.2015.55}
}
```

## Licenses

### Project source code

The source code in this repository is licensed under the [MIT License](LICENSE).

The MIT License applies only to the original project code. It does not replace or modify the license of the Telecom Italia dataset or any third-party dependency.

### Dataset

The Telecom Italia Big Data Challenge data is made available under the **Open Data Commons Open Database License 1.0 (ODbL-1.0)**.

Under the ODbL, public use requires attribution, and redistribution of an adapted database may require making that database available under the same license. Review the complete license before redistributing the dataset or an adapted version:

[Open Data Commons ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/)

The raw CSV files are not relicensed under the MIT License and are not included in this repository.

### Dependencies

TensorFlow, pandas, NumPy, Matplotlib, scikit-learn, Joblib, and other dependencies remain subject to their own licenses.

## Acknowledgments

The data originates from the Telecom Italia Big Data Challenge and was produced with contributions from Telecom Italia and collaborating research institutions. The downloadable one-week subset used in this project is hosted on Kaggle by its dataset uploader.

---

This repository is intended as a machine-learning and telecommunications portfolio project.

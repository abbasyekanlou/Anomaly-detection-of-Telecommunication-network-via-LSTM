import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("outputs")
FIGURE_DIR = OUTPUT_DIR / "figures"

SQUARE_ID = 5161
INPUT_STEPS = 144
OUTPUT_STEPS = 6
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
EPOCHS = 100
BATCH_SIZE = 32

np.random.seed(42)
tf.random.set_seed(42)


def load_traffic():
    files = sorted(DATA_DIR.rglob("sms-call-internet-mi-*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR.resolve()}")

    parts = []

    for i, file in enumerate(files, 1):
        print(f"Loading {i}/{len(files)}: {file.name}")

        first_value = str(pd.read_csv(file, nrows=1, header=None).iloc[0, 0])
        has_header = not first_value.replace(".", "", 1).isdigit()

        for chunk in pd.read_csv(
            file,
            header=0 if has_header else None,
            usecols=[0, 1, 7],
            chunksize=500_000,
        ):
            chunk.columns = ["square_id", "timestamp", "internet"]
            chunk = chunk.loc[
                chunk["square_id"] == SQUARE_ID,
                ["timestamp", "internet"],
            ].copy()

            if chunk.empty:
                continue

            chunk["internet"] = pd.to_numeric(
                chunk["internet"], errors="coerce"
            ).fillna(0)

            parts.append(
                chunk.groupby("timestamp", as_index=False)["internet"].sum()
            )

    if not parts:
        raise ValueError(f"Grid square {SQUARE_ID} was not found")

    data = pd.concat(parts).groupby("timestamp", as_index=False)["internet"].sum()
    data["datetime"] = (
        pd.to_datetime(data["timestamp"], unit="ms", utc=True)
        .dt.tz_convert("Europe/Rome")
        .dt.tz_localize(None)
    )

    data = (
        data.drop(columns="timestamp")
        .set_index("datetime")
        .sort_index()
        .asfreq("10min")
    )

    if data["internet"].isna().any():
        data["internet"] = data["internet"].interpolate(
            method="time", limit_direction="both"
        )

    return data.astype("float32")


def make_windows(values, train_end, val_end):
    sets = {
        "train": [[], [], []],
        "val": [[], [], []],
        "test": [[], [], []],
    }

    last_start = len(values) - INPUT_STEPS - OUTPUT_STEPS + 1

    for start in range(last_start):
        target_start = start + INPUT_STEPS
        target_end = target_start + OUTPUT_STEPS

        if target_end <= train_end:
            split = "train"
        elif target_start >= train_end and target_end <= val_end:
            split = "val"
        elif target_start >= val_end:
            split = "test"
        else:
            continue

        sets[split][0].append(values[start:target_start])
        sets[split][1].append(values[target_start:target_end, 0])
        sets[split][2].append(target_start)

    result = {}
    for split, (X, y, indices) in sets.items():
        if not X:
            raise ValueError(f"No {split} windows were created")
        result[split] = (
            np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32),
            np.array(indices),
        )

    return result


def inverse_scale(values, scaler):
    shape = values.shape
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(shape)


def metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true.ravel(), y_pred.ravel()),
        "RMSE": np.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel())),
    }


def save_plot(name):
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / name, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    data = load_traffic()
    train_end = int(len(data) * TRAIN_RATIO)
    val_end = int(len(data) * (TRAIN_RATIO + VAL_RATIO))

    plt.figure(figsize=(14, 5))
    plt.plot(data.index, data["internet"])
    plt.axvline(data.index[train_end], linestyle="--", label="Validation")
    plt.axvline(data.index[val_end], linestyle="--", label="Test")
    plt.title(f"Internet Activity — Grid Square {SQUARE_ID}")
    plt.xlabel("Time")
    plt.ylabel("Internet activity")
    plt.legend()
    plt.grid(alpha=0.3)
    save_plot("01_data_split.png")

    scaler = MinMaxScaler()
    scaler.fit(data[["internet"]].iloc[:train_end])
    scaled = scaler.transform(data[["internet"]]).astype(np.float32)

    windows = make_windows(scaled, train_end, val_end)
    X_train, y_train, _ = windows["train"]
    X_val, y_val, val_indices = windows["val"]
    X_test, y_test, test_indices = windows["test"]

    print("\nShapes")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:  ", X_val.shape, "y_val:  ", y_val.shape)
    print("X_test: ", X_test.shape, "y_test: ", y_test.shape)

    model = tf.keras.Sequential([
        tf.keras.layers.Input((INPUT_STEPS, 1)),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(OUTPUT_STEPS),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss="mse",
        metrics=["mae"],
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
            )
        ],
    )

    val_pred = inverse_scale(model.predict(X_val, verbose=0), scaler)
    test_pred = inverse_scale(model.predict(X_test, verbose=0), scaler)
    val_true = inverse_scale(y_val, scaler)
    test_true = inverse_scale(y_test, scaler)

    previous_day = inverse_scale(X_test[:, :OUTPUT_STEPS, 0], scaler)

    results = pd.DataFrame({
        "Previous day": metrics(test_true, previous_day),
        "LSTM": metrics(test_true, test_pred),
    }).T

    print("\nModel comparison")
    print(results.round(4))

    val_error = np.abs(val_true[:, 0] - val_pred[:, 0])
    threshold = np.percentile(val_error, 99)
    test_error = np.abs(test_true[:, 0] - test_pred[:, 0])

    anomalies = pd.DataFrame({
        "datetime": data.index[test_indices],
        "actual": test_true[:, 0],
        "predicted": test_pred[:, 0],
        "error": test_error,
        "anomaly": test_error > threshold,
    })

    print(f"\nAnomaly threshold: {threshold:.2f}")
    print("Candidate anomalies:", anomalies["anomaly"].sum())

    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Training")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(alpha=0.3)
    save_plot("02_training_loss.png")

    first = int(test_indices[0])
    history_times = data.index[first - INPUT_STEPS:first]
    future_times = data.index[first:first + OUTPUT_STEPS]

    plt.figure(figsize=(14, 5))
    plt.plot(
        history_times,
        data["internet"].iloc[first - INPUT_STEPS:first],
        label="Previous 24 hours",
    )
    plt.plot(future_times, test_true[0], marker="o", label="Actual")
    plt.plot(
        future_times,
        test_pred[0],
        marker="o",
        linestyle="--",
        label="LSTM",
    )
    plt.title("Example Next-Hour Forecast")
    plt.xlabel("Time")
    plt.ylabel("Internet activity")
    plt.legend()
    plt.grid(alpha=0.3)
    save_plot("03_forecast_example.png")

    horizon_mae = [
        mean_absolute_error(test_true[:, i], test_pred[:, i])
        for i in range(OUTPUT_STEPS)
    ]

    plt.figure(figsize=(9, 5))
    plt.plot(range(10, 70, 10), horizon_mae, marker="o")
    plt.title("MAE by Forecast Horizon")
    plt.xlabel("Forecast horizon (minutes)")
    plt.ylabel("MAE")
    plt.grid(alpha=0.3)
    save_plot("04_horizon_mae.png")

    flagged = anomalies[anomalies["anomaly"]]

    plt.figure(figsize=(14, 5))
    plt.plot(anomalies["datetime"], anomalies["actual"], label="Actual")
    plt.plot(anomalies["datetime"], anomalies["predicted"], label="Predicted")

    if not flagged.empty:
        plt.scatter(
            flagged["datetime"],
            flagged["actual"],
            marker="x",
            s=60,
            label="Candidate anomaly",
        )

    plt.title("Test Forecast and Candidate Anomalies")
    plt.xlabel("Time")
    plt.ylabel("Internet activity")
    plt.legend()
    plt.grid(alpha=0.3)
    save_plot("05_test_anomalies.png")

    results.to_csv(OUTPUT_DIR / "metrics.csv")
    anomalies.to_csv(OUTPUT_DIR / "anomalies.csv", index=False)

    print(f"\nFinished. Results saved in {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

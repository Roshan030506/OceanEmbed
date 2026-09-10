"""Train OceanEmbed, print validation metrics by depth, and write dashboard files."""

import inspect
import json
import time
from datetime import datetime, timezone

import numpy as np

from model import DEPTHS, load_or_generate_dummy_data, train_model


def evaluate(model, valid_x, valid_y, target_mean, target_std):
    if hasattr(model, "regressor"):
        prediction = model.predict(valid_x)
    else:
        torch = __import__("torch")
        model.eval()
        with torch.no_grad():
            prediction, _ = model(valid_x)
        prediction = prediction.numpy()
    actual = valid_y * target_std + target_mean
    predicted = prediction * target_std + target_mean
    actual = actual.numpy() if hasattr(actual, "numpy") else actual
    predicted = predicted.numpy() if hasattr(predicted, "numpy") else predicted
    rows = []
    for index, depth in enumerate(DEPTHS):
        error = predicted[:, index] - actual[:, index]
        actual_flat = actual[:, index].ravel()
        predicted_flat = predicted[:, index].ravel()
        ss_res = np.sum((actual_flat - predicted_flat) ** 2)
        ss_tot = np.sum((actual_flat - actual_flat.mean()) ** 2)
        rows.append((depth, np.sqrt(np.mean(error ** 2)),
                     1.0 - ss_res / max(ss_tot, 1e-12), error.mean()))
    return predicted, actual, rows


def main():
    data = load_or_generate_dummy_data("ocean_data.npz")
    n_days = len(data["sst"])
    n_lat, n_lon = data["lat"].size, data["lon"].size
    n_train = int(n_days * 0.8)

    t0 = time.perf_counter()
    model, (valid_x, valid_y), stats = train_model(data)
    train_time = time.perf_counter() - t0

    predicted, actual, rows = evaluate(model, valid_x, valid_y, stats[2], stats[3])

    print("\nOceanEmbed validation metrics")
    print("Depth (m) | RMSE (deg C) | R2       | Bias (deg C)")
    print("----------+--------------+----------+-------------")
    for depth, rmse, r2, bias in rows:
        print(f"{depth:9.0f} | {rmse:12.4f} | {r2:8.4f} | {bias:11.4f}")

    # --- write dashboard output files ---
    n_parameters = sum(p.numel() for p in model.parameters())

    lat_ids = np.array([2, n_lat // 2, n_lat - 3])
    lon_ids = np.array([2, n_lon // 2, n_lon - 3])
    valid_days = np.arange(n_train, n_days)
    day_pos = np.linspace(0, len(valid_days) - 1, 3).astype(int)
    sample_days = valid_days[day_pos]

    pred_profiles = np.empty((3, len(DEPTHS)))
    actual_profiles = np.empty((3, len(DEPTHS)))
    for s in range(3):
        pred_profiles[s] = predicted[day_pos[s], :, lat_ids[s], lon_ids[s]]
        actual_profiles[s] = actual[day_pos[s], :, lat_ids[s], lon_ids[s]]

    sample_points = [{
        "label": f"P{s + 1}",
        "day_index": int(sample_days[s]),
        "day": f"2020-01-{int(sample_days[s]) + 1:02d}",
        "lat": float(data["lat"][lat_ids[s]]),
        "lon": float(data["lon"][lon_ids[s]]),
    } for s in range(3)]

    signature = inspect.signature(train_model)
    epochs = signature.parameters["epochs"].default
    batch_size = signature.parameters["batch_size"].default
    learning_rate = signature.parameters["learning_rate"].default

    metrics_json = {
        "model_name": "OceanEmbed CNN encoder-decoder (PyTorch)",
        "architecture": "convolutional encoder-decoder with a 16-dim learned embedding (autoencoder)",
        "n_parameters": int(n_parameters),
        "train_time_seconds": round(float(train_time), 3),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "embedding_dim": 16,
        "depth_levels_m": [float(d) for d in DEPTHS],
        "validation_design": (f"chronological: first {n_train} days train, "
                              f"last {n_days - n_train} days validation"),
        "input_data": "ocean_data.npz (synthetic)",
        "metrics": [{"depth_m": float(depth), "rmse_degC": float(rmse),
                     "r2": float(r2), "bias_degC": float(bias)}
                    for depth, rmse, r2, bias in rows],
        "sample_points": sample_points,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open("deep_learning_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics_json, fh, indent=2)

    np.savez_compressed(
        "deep_learning_samples.npz",
        depth_m=DEPTHS,
        sample_day=sample_days.astype(np.int64),
        sample_lat=data["lat"][lat_ids].astype(np.float32),
        sample_lon=data["lon"][lon_ids].astype(np.float32),
        pred_dl=pred_profiles.astype(np.float32),
        actual=actual_profiles.astype(np.float32),
    )

    print(f"\nParameters      : {n_parameters:,}")
    print(f"Training time   : {train_time:.1f} s")
    print("saved: deep_learning_metrics.json, deep_learning_samples.npz")


if __name__ == "__main__":
    main()
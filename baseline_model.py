"""baseline_model.py  (OceanEmbed - baseline)
=====================================================================
Trains one scikit-learn Random Forest per depth level (0, 50, 100, 500,
1000 m) that predicts subsurface temperature from the seven standardized
surface variables (SST, SSH, SSS, u/v current, u/v wind) at every grid
point and day.

* Part 1 deliverable: the primary baseline predicts 100 m temperature and
  its metrics are reported explicitly at the top of baseline_results.txt.
* Dashboard deliverable: per-depth metrics for all five levels so the RF
  baseline can be compared side by side against the deep-learning model.

Input : ocean_data_preprocessed.npz  (from preprocess.py, standardized)
Output: baseline_results.txt         (metrics per depth, incl. 100 m)
        baseline_predictions.npz     (per-depth metrics + predicted/actual
                                      temperature profiles at the 3 standard
                                      sample validation points also used by
                                      visualize.py)

Validation: the chronologically last 20% of days are held out — the same
split used by the OceanEmbed CNN — so both models are compared on an
identical, realistic temporal test.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

SURFACE_KEYS = ("sst", "ssh", "sss", "u_current", "v_current",
                "u_wind", "v_wind")
PREPROCESSED = "ocean_data_preprocessed.npz"
DEPTHS_M = np.array([0.0, 50.0, 100.0, 500.0, 1000.0])
PRIMARY_DEPTH_M = 100.0
SEED = 42
HOLD_OUT_FRACTION = 0.20
N_ESTIMATORS = 250


def scaler_for(var_names, scaled_mean, scaled_std, target_key):
    """Return (mean, std) in physical units for a target_tensor variable."""
    index = list(var_names).index(target_key)
    return float(scaled_mean[index]), float(scaled_std[index])


def sample_points(n_lat, n_lon, n_days):
    """3 standard sample validation points (visualize.py convention).

    Points sit at lat/lon index [2, mid, size-3] and at the first/middle/
    last day of the chronological validation window used by the DL model.
    """
    lat_ids = np.array([2, n_lat // 2, n_lat - 3])
    lon_ids = np.array([2, n_lon // 2, n_lon - 3])
    n_train = int(n_days * 0.8)
    valid_days = np.arange(n_train, n_days)
    day_pos = np.linspace(0, len(valid_days) - 1, 3).astype(int)
    return lat_ids, lon_ids, valid_days[day_pos]
def main():
    with np.load(PREPROCESSED) as z:
        surface = np.stack([z[k] for k in SURFACE_KEYS], axis=-1)
        feature_matrix = surface.reshape(-1, len(SURFACE_KEYS)).astype(np.float64)
        target_std = z["subsurface_temp"]            # (day, depth, lat, lon)
        var_names = list(z["var_names"])
        scaled_mean_arr = z["scaled_mean"]
        scaled_std_arr = z["scaled_std"]
        n_days, n_lat, n_lon = (target_std.shape[0], z["lat"].size, z["lon"].size)
        lat_arr, lon_arr = z["lat"], z["lon"]

    # --- chronological last-20% hold-out (matches the CNN validation split) ---
    n_train = int(n_days * (1.0 - HOLD_OUT_FRACTION))
    n_grid = n_lat * n_lon
    train_idx = np.arange(n_train * n_grid)
    val_idx = np.arange(n_train * n_grid, n_days * n_grid)

    # --- train one Random Forest per depth, score on the hold-out ----------
    metrics = []     # (depth_m, rmse, r2, bias, train_rmse)
    rf_models = {}
    for d_index, depth_m in enumerate(DEPTHS_M):
        target_key = f"temp_d{int(depth_m)}"
        mean_t, std_t = scaler_for(var_names, scaled_mean_arr, scaled_std_arr,
                                   target_key)
        target_z = target_std[:, d_index].reshape(-1).astype(np.float64)
        target_phys = target_z * std_t + mean_t

        rf = RandomForestRegressor(n_estimators=N_ESTIMATORS,
                                   min_samples_leaf=2,
                                   random_state=SEED + d_index, n_jobs=-1)
        rf.fit(feature_matrix[train_idx], target_z[train_idx])
        rf_models[d_index] = rf

        pred_val = rf.predict(feature_matrix[val_idx]) * std_t + mean_t
        obs_val = target_phys[val_idx]
        rmse = float(np.sqrt(mean_squared_error(obs_val, pred_val)))
        r2 = float(r2_score(obs_val, pred_val))
        bias = float(np.mean(pred_val - obs_val))
        train_rmse = float(np.sqrt(mean_squared_error(
            target_phys[train_idx],
            rf.predict(feature_matrix[train_idx]) * std_t + mean_t)))
        metrics.append((float(depth_m), rmse, r2, bias, train_rmse))

    # --- temperature profiles at the 3 standard validation sample points ---
    lat_ids, lon_ids, sample_days = sample_points(n_lat, n_lon, n_days)
    grid_cells = n_lat * n_lon
    pred_profiles = np.empty((3, len(DEPTHS_M)))
    actual_profiles = np.empty((3, len(DEPTHS_M)))
    for s, (day, lat_id, lon_id) in enumerate(zip(sample_days, lat_ids, lon_ids)):
        row = int(day * grid_cells + lat_id * n_lon + lon_id)
        for d_index, depth_m in enumerate(DEPTHS_M):
            target_key = f"temp_d{int(depth_m)}"
            mean_t, std_t = scaler_for(var_names, scaled_mean_arr,
                                       scaled_std_arr, target_key)
            pred_profiles[s, d_index] = rf_models[d_index].predict(
                feature_matrix[row:row + 1])[0] * std_t + mean_t
            actual_profiles[s, d_index] = (target_std[sample_days[s],
                                                      d_index,
                                                      lat_id,
                                                      lon_id] * std_t + mean_t)

    primary_model = rf_models[int(np.argmin(np.abs(DEPTHS_M - PRIMARY_DEPTH_M)))]
    feature_importance = dict(zip(SURFACE_KEYS, primary_model.feature_importances_))
# ---------------------------------------------------------------- report
    primary = next(m for m in metrics if m[0] == PRIMARY_DEPTH_M)
    lines = [
        "OceanEmbed baseline model - results",
        "=" * 53,
        f"generated UTC        : {datetime.now(timezone.utc).isoformat()}",
        f"model                : RandomForestRegressor (scikit-learn {sklearn.__version__})",
        f"predicting           : subsurface temperature, deg C "
        f"(primary target: {PRIMARY_DEPTH_M:.0f} m)",
        f"features             : 7 surface vars ({' '.join(SURFACE_KEYS)})",
        f"input data           : {PREPROCESSED} (standardized)",
        f"total samples        : {feature_matrix.shape[0]:,} "
        f"({n_days} days x {n_lat * n_lon} grid points)",
        f"training samples     : {train_idx.size:,}  ({1 - HOLD_OUT_FRACTION:.0%})",
        f"hold-out (chronological) : {val_idx.size:,}  "
        f"({HOLD_OUT_FRACTION:.0%}, last {n_days - n_train} days)",
        "",
        f"Primary baseline target ({PRIMARY_DEPTH_M:.0f} m) held-out metrics:",
        "-" * 53,
        f"  RMSE  = {primary[1]:.4f} deg C",
        f"  R2    = {primary[2]:.4f}",
        f"  BIAS  = {primary[3]:+.4f} deg C",
        "-" * 53,
        f"Reference (in-sample train) RMSE @ {PRIMARY_DEPTH_M:.0f} m = "
        f"{primary[4]:.4f} deg C",
        "",
        "Per-depth held-out (chronological last-20% days) metrics",
        "depth_m,rmse_degC,r2,bias_degC,train_rmse_degC",
    ]
    for depth_m, rmse, r2, bias, train_rmse in metrics:
        lines.append(f"{depth_m:.0f},{rmse:.4f},{r2:.4f},{bias:+.4f},{train_rmse:.4f}")
    lines += [
        "",
        "Sample validation points (visualize.py convention)",
        "point,day_index,lat,lon",
    ]
    for s, (day, lat_id, lon_id) in enumerate(zip(sample_days, lat_ids, lon_ids)):
        lines.append(f"P{s + 1},{int(day)},{lat_arr[lat_id]:.2f},{lon_arr[lon_id]:.2f}")
    lines += [
        "",
        "Feature importances (primary 100 m model): " + ", ".join(
            f"{name}={value:.3f}" for name, value in
            sorted(feature_importance.items(), key=lambda kv: kv[1], reverse=True)),
        "",
        "NOTE: Data are SYNTHETIC. Metrics gauge internal consistency,",
        "not real-world skill.",
    ]

    report = "\n".join(lines)
    print(report)
    with open("baseline_results.txt", "w", encoding="utf-8") as fh:
        fh.write(report)

    np.savez_compressed(
        "baseline_predictions.npz",
        depth_m=DEPTHS_M,
        rmse=np.array([m[1] for m in metrics]),
        r2=np.array([m[2] for m in metrics]),
        bias=np.array([m[3] for m in metrics]),
        train_rmse=np.array([m[4] for m in metrics]),
        sample_day=sample_days.astype(np.int64),
        sample_lat=lat_arr[lat_ids].astype(np.float32),
        sample_lon=lon_arr[lon_ids].astype(np.float32),
        pred_rf=pred_profiles.astype(np.float32),
        actual=actual_profiles.astype(np.float32),
        n_train=np.array([train_idx.size], dtype=np.int64),
        n_val=np.array([val_idx.size], dtype=np.int64),
    )
    print("\nsaved: baseline_results.txt, baseline_predictions.npz")


if __name__ == "__main__":
    main()
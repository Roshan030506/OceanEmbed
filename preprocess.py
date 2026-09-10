"""preprocess.py  (OceanEmbed - data module)
=====================================================================
Loads ``ocean_data.npz`` and:

  1. finds and fills any missing values - linear interpolation along the
     time axis first, spatial fallback after;
  2. standardizes every variable to zero mean / unit variance
     (subsurface temperature is standardized per depth level);
  3. verifies that the final data has no NaNs.

Writes ``ocean_data_preprocessed.npz``: the same key layout as the shared
contract, with data standardized, plus per-variable scaling statistics
(``scaled_mean``, ``scaled_std``, ``var_names``) so a downstream model can
map standardized values back to physical units.
"""
from __future__ import annotations

import json

import numpy as np

SURFACE_KEYS = ("sst", "ssh", "sss", "u_current", "v_current",
                "u_wind", "v_wind")
DEPTH_NAMES = ("temp_d0", "temp_d50", "temp_d100", "temp_d500", "temp_d1000")
IN_PATH = "ocean_data.npz"
OUT_PATH = "ocean_data_preprocessed.npz"


def fill_along_time(var: np.ndarray) -> np.ndarray:
    """Linear interpolation along the leading (time) axis, cell by cell."""
    nt = var.shape[0]
    t = np.arange(nt)
    for idx in np.ndindex(var.shape[1:]):
        column = var[(slice(None),) + idx]
        good = ~np.isnan(column)
        if good.all() or not good.any():
            continue
        tg = t[good]
        var[(slice(None),) + idx] = np.interp(
            t, tg, column[good], left=column[good][0], right=column[good][-1])
    return var


def fill_spatial_fallback(var: np.ndarray) -> np.ndarray:
    """Anything still missing: time-mean of that cell; global mean as last resort."""
    for idx in np.ndindex(var.shape[1:]):
        column = var[(slice(None),) + idx]
        if np.isnan(column).any():
            var[(slice(None),) + idx] = np.where(
                np.isnan(column), np.nanmean(column), column)
    if np.isnan(var).any():
        var = np.where(np.isnan(var), np.nanmean(var), var)
    return var


def fill_missing(var: np.ndarray) -> np.ndarray:
    """Interpolate missing values; no-op when there are none."""
    var = np.array(var, dtype=np.float64)
    if np.isnan(var).any():
        var = fill_along_time(var)
        var = fill_spatial_fallback(var)
    return var


def standardize(arr: np.ndarray):
    """Zero-mean / unit-variance scaling of a 1-level field."""
    m = float(np.nanmean(arr))
    s = float(np.nanstd(arr)) or 1.0
    return (arr - m) / s, m, s


def main() -> None:
    with np.load(IN_PATH) as archive:
        raw = {key: archive[key] for key in archive.files}
    meta = json.loads(str(raw["meta"]))
    print("=" * 72)
    print("preprocess.py  -  OceanEmbed data preprocessing")
    print("=" * 72)

    total_missing = 0
    normalized = {}
    var_names, scaled_mean, scaled_std = [], [], []

    print(f"{'variable':<14}{'missing':>9}{'mean':>10}{'std':>10}   status")
    for key in SURFACE_KEYS:
        n_missing = int(np.isnan(raw[key]).sum())
        total_missing += n_missing
        cleaned = fill_missing(raw[key])
        norm, m, s = standardize(cleaned)
        normalized[key] = np.asarray(norm, dtype=np.float32)
        var_names.append(key)
        scaled_mean.append(m)
        scaled_std.append(s)
        print(f"{key:<14}{n_missing:9d}{m:10.4f}{s:10.4f}   standardized")

    subsurface_missing = int(np.isnan(raw["subsurface_temp"]).sum())
    total_missing += subsurface_missing
    cleaned_sub = fill_missing(raw["subsurface_temp"])
    sub_norm = np.empty_like(cleaned_sub)
    for d, name in enumerate(DEPTH_NAMES):
        n_missing = int(np.isnan(raw["subsurface_temp"][:, d]).sum())
        layer_norm, m, s = standardize(cleaned_sub[:, d])
        sub_norm[:, d] = layer_norm
        var_names.append(name)
        scaled_mean.append(m)
        scaled_std.append(s)
        print(f"{name:<14}{n_missing:9d}{m:10.4f}{s:10.4f}   standardized")
    normalized["subsurface_temp"] = np.asarray(sub_norm, dtype=np.float32)

    nan_remaining = sum(int(np.isnan(normalized[k]).sum()) for k in normalized)
    print("-" * 72)
    print(f"missing values found in raw file : {total_missing}")
    print(f"missing values after filling      : {nan_remaining}")
    print(f"NaNs remaining                    : {nan_remaining}")
    assert nan_remaining == 0, "final data still contains missing values!"
    print("\u2713 final data has no NaNs")

    out = dict(raw)
    for key, value in normalized.items():
        out[key] = value
    out["scaled_mean"] = np.asarray(scaled_mean, dtype=np.float64)
    out["scaled_std"] = np.asarray(scaled_std, dtype=np.float64)
    out["var_names"] = np.asarray(var_names, dtype="U12")
    out["n_missing_found"] = np.asarray(total_missing, dtype=np.int64)
    out["nan_remaining"] = np.asarray(nan_remaining, dtype=np.int64)

    meta["preprocessed"] = {
        "script": "preprocess.py",
        "normalization": "zero mean / unit variance per variable "
                         "(temperature per depth level)",
        "missing_handling": "linear interpolation along time; spatial fallback",
        "variables_standardized": var_names,
        "n_missing_found": total_missing,
        "nan_remaining": nan_remaining,
    }
    out["meta"] = np.array(json.dumps(meta, indent=2))
    np.savez_compressed(OUT_PATH, **out)
    print(f"saved: {OUT_PATH}  (same contract keys, standardized)")


if __name__ == "__main__":
    main()
"""dashboard.py  (OceanEmbed - results dashboard)
=====================================================================
Single-page Streamlit dashboard for the OceanEmbed prototype.
Reads real output files — no hardcoded numbers.

  * ocean_data.npz ............... synthetic data + metadata
  * baseline_results.txt ......... Random-Forest baseline metrics
  * baseline_predictions.npz ..... RF per-depth metrics + sample profiles
  * deep_learning_metrics.json ... OceanEmbed CNN metrics + architecture
  * deep_learning_samples.npz .... CNN predicted/actual sample profiles

Run:
    streamlit run dashboard.py
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="OceanEmbed", page_icon="🌊", layout="wide")

# Clean, minimal color palette
RF_COLOR = "#2563EB"   # blue — Random Forest
DL_COLOR = "#DC2626"   # red — OceanEmbed CNN
NEUTRAL  = "#64748B"   # slate — secondary

REQUIRED_FILES = (
    "ocean_data.npz",
    "baseline_results.txt",
    "baseline_predictions.npz",
    "deep_learning_metrics.json",
    "deep_learning_samples.npz",
)


# --------------------------------------------------------------------------
# Data loaders
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_ocean_data() -> dict:
    with np.load("ocean_data.npz") as z:
        return {
            "synthetic": bool(z["synthetic"]),
            "meta": json.loads(str(z["meta"])),
            "lat": np.asarray(z["lat"], dtype=float),
            "lon": np.asarray(z["lon"], dtype=float),
            "depth": np.asarray(z["depth"], dtype=float),
            "sst": z["sst"],
            "time_iso": [str(t) for t in z["time_iso"]],
        }


@st.cache_data(show_spinner=False)
def load_baseline_results() -> dict:
    text = open("baseline_results.txt", encoding="utf-8").read()

    def grab(label: str):
        match = re.search(rf"{re.escape(label)}\s*=\s*([+-]?\d+\.\d+)", text)
        return float(match.group(1)) if match else None

    header = "depth_m,rmse_degC,r2,bias_degC,train_rmse_degC"
    start = text.index(header) + len(header)
    rows = [line.split(",") for line in text[start:].splitlines()
            if line.strip() and line[0].isdigit()]
    metrics = pd.DataFrame(
        rows,
        columns=["depth_m", "rmse_degC", "r2", "bias_degC", "train_rmse_degC"]
    ).astype(float)

    sp_header = "point,day_index,lat,lon"
    start = text.index(sp_header) + len(sp_header)
    sp_rows = [line.split(",") for line in text[start:].splitlines()
               if line.strip() and line[0].startswith("P")]
    samples = pd.DataFrame(sp_rows, columns=["point", "day_index", "lat", "lon"])
    samples[["day_index", "lat", "lon"]] = samples[["day_index", "lat", "lon"]].astype(float)

    model_line = next((ln for ln in text.splitlines() if ln.startswith("model")), "")

    return {
        "metrics": metrics,
        "samples": samples,
        "rmse_100m": grab("RMSE"),
        "r2_100m": grab("R2"),
        "bias_100m": grab("BIAS"),
        "model_line": model_line,
    }


@st.cache_data(show_spinner=False)
def load_baseline_predictions() -> dict:
    with np.load("baseline_predictions.npz") as z:
        return {
            "depth_m": np.asarray(z["depth_m"], dtype=float),
            "sample_lat": np.asarray(z["sample_lat"], dtype=float),
            "sample_lon": np.asarray(z["sample_lon"], dtype=float),
            "pred_rf": z["pred_rf"],
            "actual": z["actual"],
        }


@st.cache_data(show_spinner=False)
def load_dl_metrics() -> dict:
    with open("deep_learning_metrics.json", encoding="utf-8") as fh:
        payload = json.load(fh)
    return {
        "metrics": pd.DataFrame(payload["metrics"]),
        "model_name": payload["model_name"],
        "architecture": payload["architecture"],
        "n_parameters": int(payload["n_parameters"]),
        "train_time_seconds": float(payload["train_time_seconds"]),
        "epochs": payload.get("epochs"),
        "batch_size": payload.get("batch_size"),
        "learning_rate": payload.get("learning_rate"),
        "validation_design": payload["validation_design"],
    }


@st.cache_data(show_spinner=False)
def load_dl_samples() -> dict:
    with np.load("deep_learning_samples.npz") as z:
        return {
            "depth_m": np.asarray(z["depth_m"], dtype=float),
            "pred_dl": z["pred_dl"],
            "actual": z["actual"],
        }


# --------------------------------------------------------------------------
# Chart builders
# --------------------------------------------------------------------------
def region_map_figure(data, sample_lat, sample_lon):
    """SST heatmap with validation points marked."""
    sst0 = np.asarray(data["sst"][0])
    lat, lon = data["lat"], data["lon"]
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=lon, y=lat, z=sst0,
        colorscale="RdYlBu_r",
        zmin=float(np.percentile(sst0, 1)),
        zmax=float(np.percentile(sst0, 99)),
        colorbar={"title": "SST (°C)"},
        hovertemplate="lon %{x:.2f}E<br>lat %{y:.2f}N<br>SST %{z:.2f} °C",
        name="SST"))
    fig.add_trace(go.Scatter(
        x=sample_lon, y=sample_lat,
        mode="markers+text",
        text=["P1", "P2", "P3"],
        textposition="top center",
        marker={"size": 12, "symbol": "circle", "color": "#111111",
                "line": {"width": 2, "color": "white"}},
        name="Validation points"))
    fig.update_layout(
        height=400,
        xaxis_title="Longitude (°E)",
        yaxis_title="Latitude (°N)",
        xaxis={"range": [float(lon.min()) - 0.4, float(lon.max()) + 0.4]},
        yaxis={"range": [float(lat.min()) - 0.4, float(lat.max()) + 0.4],
               "scaleanchor": "x"},
        margin={"l": 50, "r": 20, "t": 30, "b": 50},
        font=dict(size=12))
    return fig


def residual_figure(depths, actual, pred_rf, pred_dl):
    """Error (predicted - actual) per depth — shows real differences clearly."""
    rf_error = pred_rf - actual
    dl_error = pred_dl - actual
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rf_error, y=depths, name="Random Forest",
        mode="lines+markers",
        line=dict(color=RF_COLOR, width=2),
        marker=dict(size=8)))
    fig.add_trace(go.Scatter(
        x=dl_error, y=depths, name="OceanEmbed CNN",
        mode="lines+markers",
        line=dict(color=DL_COLOR, width=2),
        marker=dict(size=8)))
    fig.add_vline(x=0, line_dash="solid", line_color="#111111", line_width=1)
    fig.update_yaxes(autorange="reversed", title="Depth (m)")
    fig.update_xaxes(title="Error (predicted − actual, °C)")
    fig.update_layout(
        height=350,
        font=dict(size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin={"l": 60, "r": 20, "t": 40, "b": 50})
    return fig


# --------------------------------------------------------------------------
# Page rendering
# --------------------------------------------------------------------------
def main():
    missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
    if missing:
        st.error("Some result files are missing: " + ", ".join(missing))
        st.markdown("Run the pipeline first (in this folder):\n\n"
                    "```text\n"
                    "python generate_data.py\n"
                    "python preprocess.py\n"
                    "python baseline_model.py\n"
                    "python evaluate.py\n"
                    "streamlit run dashboard.py\n"
                    "```")
        st.stop()

    data = load_ocean_data()
    bl = load_baseline_results()
    bl_pred = load_baseline_predictions()
    dl = load_dl_metrics()
    dl_pred = load_dl_samples()

    # --- synthetic-data disclaimer (kept at top) ---
    st.warning("⚠️ **Synthetic data** — all results shown here are from "
               "synthetically generated ocean fields (Bay of Bengal, January 2020). "
               "Metrics measure internal consistency, not real-world skill.")

    # --- header ---
    st.title("OceanEmbed — model comparison")
    lat_min, lat_max = float(data["lat"].min()), float(data["lat"].max())
    lon_min, lon_max = float(data["lon"].min()), float(data["lon"].max())
    st.caption(f"Synthetic prototype · {data['time_iso'][0]} to {data['time_iso'][-1]} "
               f"· {lat_min:.1f}–{lat_max:.1f}°N, {lon_min:.1f}–{lon_max:.1f}°E "
               f"· {data['lat'].size}×{data['lon'].size} grid")

    # --- region map with validation points ---
    st.subheader("Region map")
    st.plotly_chart(region_map_figure(data, bl_pred["sample_lat"],
                                      bl_pred["sample_lon"]),
                    use_container_width=True)

    # --- model comparison table ---
    st.subheader("Model comparison")
    rf = bl["metrics"].set_index("depth_m")
    cnn = dl["metrics"].set_index("depth_m")

    rows = []
    for depth in rf.index:
        rows.append({
            "Depth (m)": int(depth),
            "RF RMSE": round(float(rf.loc[depth]["rmse_degC"]), 4),
            "CNN RMSE": round(float(cnn.loc[depth]["rmse_degC"]), 4),
            "RF R²": round(float(rf.loc[depth]["r2"]), 4),
            "CNN R²": round(float(cnn.loc[depth]["r2"]), 4),
            "RF Bias": round(float(rf.loc[depth]["bias_degC"]), 4),
            "CNN Bias": round(float(cnn.loc[depth]["bias_degC"]), 4),
        })
    table = pd.DataFrame(rows)

    # Neutral summary line (not a "winner" banner)
    st.markdown("**Comparison at prototype scale — both models validated on the "
                "chronological last 20% of days. See table below for per-depth results.**")
    st.dataframe(table, hide_index=True, use_container_width=True)

    # --- error/residual chart ---
    st.subheader("Prediction error by depth")
    samples = bl["samples"].set_index("point")
    option_map = {}
    for i, (lat, lon) in enumerate(
            zip(bl_pred["sample_lat"], bl_pred["sample_lon"])):
        day_no = int(samples.loc[f"P{i + 1}", "day_index"]) + 1
        option_map[f"P{i + 1}  ({lat:.2f}°N, {lon:.2f}°E) — Jan {day_no}, 2020"] = i
    choice = st.selectbox("Sample validation point", list(option_map.keys()))
    s_idx = option_map[choice]
    fig_residual = residual_figure(
        depths=dl_pred["depth_m"],
        actual=dl_pred["actual"][s_idx],
        pred_rf=bl_pred["pred_rf"][s_idx],
        pred_dl=dl_pred["pred_dl"][s_idx])
    st.plotly_chart(fig_residual, use_container_width=True)
    st.caption("Error = predicted minus actual. Zero line = perfect prediction. "
               "Positive values mean the model is too warm.")

    # --- model info ---
    st.subheader("Model info")
    i1, i2 = st.columns(2)
    with i1:
        st.markdown("**Random Forest baseline (scikit-learn)**")
        baseline_name = (bl["model_line"].split(":", 1)[-1].strip()
                         or "RandomForestRegressor (scikit-learn)")
        st.markdown(
            f"- {baseline_name}\n"
            "- One ensemble per depth level (5 models)\n"
            "- Features: 7 surface variables per grid point/day\n"
            "- Validation: chronological last 20% of days")
    with i2:
        st.markdown("**OceanEmbed CNN (PyTorch)**")
        n_params = dl["n_parameters"]
        st.markdown(
            f"- Architecture: {dl['architecture']}\n"
            f"- Parameters: **{n_params:,}**\n"
            f"- Training: **{dl['train_time_seconds']:.1f} s** "
            f"({dl['epochs']} epochs, batch {dl['batch_size']})\n"
            f"- Validation: {dl['validation_design']}")

    st.markdown("---")
    st.caption("Proof-of-concept: single region, one month, 5 depth levels. "
               "The full OceanEmbed system is not represented here.")


if __name__ == "__main__":
    main()
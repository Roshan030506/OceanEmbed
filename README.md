# 🌊 OceanEmbed

### Satellite Embedding-Based Deep Learning Framework for Subsurface Ocean Temperature Reconstruction

OceanEmbed is an AI-based deep learning framework that reconstructs **subsurface ocean temperature profiles from satellite-observable surface conditions**.

The system combines multi-source satellite/oceanographic observations with deep learning to estimate ocean temperature at different depths, visualize subsurface structure, quantify uncertainty, and support applications such as **marine heatwave detection and ocean monitoring**.

---

## 🚀 Key Idea

Satellites provide frequent observations of the ocean surface, but direct measurements of subsurface temperature are sparse because they depend heavily on instruments such as **ARGO floats**.

OceanEmbed addresses this gap by learning the relationship between:

**Surface Ocean Observations → Learned Ocean Embedding → Subsurface Temperature**

The framework uses surface variables such as:

* 🌡️ Sea Surface Temperature (SST)
* 🌊 Sea Surface Height (SSH/SLA)
* 🧂 Sea Surface Salinity (SSS)
* 🌀 Surface ocean currents
* 💨 Surface wind components

These variables are encoded into a compact representation called an **Ocean Embedding**, which is then used to reconstruct temperature throughout the water column.

---

# 🎯 Objectives

* Reconstruct subsurface ocean temperature using surface observations.
* Generate temperature profiles at multiple depths.
* Learn spatial and temporal ocean patterns using deep learning.
* Compare AI predictions with **ARGO observations**.
* Estimate prediction uncertainty.
* Detect potential marine heatwave conditions below the surface.
* Provide interactive visualization through a Streamlit dashboard.
* Build a scalable framework capable of integrating multiple ocean datasets.

---

# 🧠 System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    OCEAN DATA SOURCES                       │
│                                                             │
│  Satellite Data     Reanalysis       ARGO Observations      │
│  SST / SSH / SSS    Currents / Wind   Temperature Profiles  │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                 DATA HARMONIZATION & QC                     │
│                                                             │
│   Quality Control → Regridding → Missing Data Handling      │
│   Temporal Alignment → Normalization → Feature Generation  │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  SURFACE FEATURE ENCODER                    │
│                                                             │
│       CNN / MAE Encoder / OceanEmbed Encoder               │
│                                                             │
│              Surface Observations                           │
│                       ↓                                     │
│              Compact Ocean Embedding                        │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                 SUBSURFACE RECONSTRUCTION                   │
│                                                             │
│   Depth Transformer │ FNO │ Diffusion │ Physics Attention  │
│                                                             │
│             ↓       ↓       ↓       ↓                       │
│                                                             │
│       Temperature at Multiple Ocean Depths                  │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                PHYSICS & DATA ASSIMILATION                  │
│                                                             │
│   Physical Constraints → ARGO Correction → Uncertainty     │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                        │
│                                                             │
│  3D Ocean Temperature │ Vertical Profiles │ Thermocline     │
│  ARGO Comparison      │ Marine Heatwaves  │ Uncertainty     │
│  Model Performance    │ Ocean Embeddings  │ Ocean Map       │
└─────────────────────────────────────────────────────────────┘
```

---

# ✨ Major Features

## 1. Multi-Source Ocean Data

OceanEmbed is designed to integrate multiple publicly available ocean datasets, including:

* OSTIA
* SMAP / SMOS
* DUACS
* OSCAR
* ASCAT
* GLORYS / Copernicus Marine
* ARGO

The data pipeline provides a common interface for downloading, harmonizing, preprocessing, and validating ocean observations.

---

## 2. Ocean Embedding

The core concept of OceanEmbed is to convert multiple surface observations into a compact learned representation.

```text
SST ──────────┐
SSH ──────────┤
SSS ──────────┤
U Current ────┤
V Current ────┼──► Deep Encoder ──► Ocean Embedding
U Wind ───────┤
V Wind ───────┘
```

The embedding captures important spatial and physical characteristics of the upper ocean and provides a compact representation for downstream reconstruction tasks.

---

## 3. Deep Learning Models

The prototype includes multiple model components for experimentation and comparison:

### OceanEmbed

Primary encoder-decoder architecture for surface-to-subsurface reconstruction.

### Depth Transformer

Uses attention mechanisms to model relationships between surface features and different ocean depths.

### Fourier Neural Operator (FNO)

Designed to learn spatial ocean dynamics efficiently using frequency-domain representations.

### Diffusion Model

Provides a probabilistic approach for generating possible subsurface temperature fields and representing uncertainty.

### MAE Encoder

Masked Autoencoder-based representation learning for extracting robust surface-ocean embeddings.

### Physical Attention

Introduces physically meaningful relationships into the attention mechanism.

### Baseline Models

Traditional machine-learning baselines are included for comparison.

---

# 🌡️ Subsurface Temperature Reconstruction

OceanEmbed predicts temperature at multiple depths.

Example:

```text
Surface
  │
  ├── 0 m
  │
  ├── 50 m
  │
  ├── 100 m
  │
  ├── 200 m
  │
  ├── 500 m
  │
  └── 1000 m
```

The system can generate:

* Horizontal temperature maps
* Vertical temperature profiles
* 3D subsurface temperature structures
* Thermocline information
* Depth-wise prediction errors
* Uncertainty estimates

---

# 🌊 ARGO Validation

ARGO float observations provide independent subsurface temperature measurements.

OceanEmbed compares AI-reconstructed temperature against ARGO observations to evaluate model performance.

```text
OceanEmbed Prediction
          │
          │
          ▼
   Temperature Profile
          │
          ├──────────────┐
          │              │
          ▼              ▼
     ARGO Profile    Prediction Error
```

Evaluation can include:

* RMSE
* MAE
* R²
* Bias
* Correlation
* Depth-wise error
* Prediction uncertainty

---

# 🔥 Marine Heatwave Detection

The framework includes a marine heatwave detection module.

It can analyze reconstructed temperature fields to identify abnormal subsurface warming events.

The system can help visualize:

* Heatwave intensity
* Duration
* Spatial extent
* Vertical penetration
* Temperature anomaly
* Potential subsurface heat accumulation

This provides a pathway toward identifying marine heatwave conditions that may not be visible from surface SST alone.

---

# ⚛️ Physics-Informed Processing

OceanEmbed incorporates physical constraints to improve the physical consistency of reconstructed temperature fields.

Examples include:

* Temperature-depth consistency
* Vertical stratification
* Smoothness constraints
* Physically plausible temperature ranges
* Thermocline behavior
* Spatial consistency

The objective is not only to minimize prediction error but also to produce **physically meaningful ocean states**.

---

# 📊 Interactive Dashboard

OceanEmbed includes a Streamlit-based interactive dashboard.

Launch it using:

```bash
streamlit run dashboard/app.py
```

The dashboard contains modules for:

* 📌 Overview
* 🗺️ India Ocean Map
* 🌡️ Surface Inputs
* 🧠 Ocean Embedding
* 🌊 Subsurface 3D Visualization
* 📈 Vertical Temperature Profile
* 🔬 ARGO Comparison
* 🌡️ Thermocline Analysis
* 🔥 Marine Heatwave Detection
* 🎯 Model Performance
* ⚠️ Prediction Uncertainty

---

# 📁 Project Structure

```text
OceanEmbed/
│
├── dashboard/
│   ├── app.py
│   ├── components/
│   │   ├── data_loader.py
│   │   └── model_inference.py
│   │
│   └── pages/
│       ├── overview.py
│       ├── india_ocean_map.py
│       ├── surface_inputs.py
│       ├── ocean_embedding.py
│       ├── subsurface_3d.py
│       ├── vertical_profile.py
│       ├── argo_comparison.py
│       ├── thermocline.py
│       ├── marine_heatwave.py
│       ├── model_performance.py
│       └── uncertainty.py
│
├── configs/
│   ├── data.yaml
│   ├── experiment.yaml
│   ├── features.yaml
│   ├── marine_heatwave.yaml
│   ├── model.yaml
│   └── training.yaml
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── scripts/
│   ├── create_dataset.py
│   ├── download_data.py
│   ├── evaluate_model.py
│   ├── generate_report.py
│   ├── inspect_data.py
│   ├── preprocess_data.py
│   ├── pretrain_embedding.py
│   ├── realtime_inference.py
│   ├── run_ablations.py
│   ├── run_inference.py
│   └── train_model.py
│
├── src/
│   ├── assimilation/
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   ├── interfaces/
│   ├── models/
│   │   ├── baselines/
│   │   ├── depth_transformer.py
│   │   ├── diffusion.py
│   │   ├── fno.py
│   │   ├── mae_encoder.py
│   │   ├── oceanembed.py
│   │   └── physical_attention.py
│   ├── physics/
│   ├── preprocessing/
│   ├── training/
│   ├── utils/
│   └── visualization/
│
├── tests/
│   ├── test_download.py
│   ├── test_features.py
│   ├── test_inference.py
│   ├── test_model.py
│   └── test_preprocessing.py
│
├── DATASET_CITATIONS.md
├── environment.yml
├── pyproject.toml
├── requirements.txt
├── run_demo.py
└── README.md
```

---

# ⚙️ Installation

## Clone the repository

```bash
git clone https://github.com/Roshan030506/OceanEmbed.git
cd OceanEmbed
```

## Create a virtual environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Configuration

Copy the example environment file:

```powershell
copy .env.example .env
```

Add the required dataset/API credentials to `.env` when needed.

**Never commit `.env` or private API credentials to GitHub.**

---

# 🗃️ Data Pipeline

The general pipeline is:

```text
Download
   ↓
Quality Control
   ↓
Harmonization
   ↓
Regridding
   ↓
Feature Engineering
   ↓
Dataset Creation
   ↓
Model Training
   ↓
Inference
   ↓
Evaluation
   ↓
Dashboard
```

Download available datasets:

```bash
python scripts/download_data.py
```

Preprocess the data:

```bash
python scripts/preprocess_data.py
```

Create the training dataset:

```bash
python scripts/create_dataset.py
```

---

# 🧠 Training

Train the OceanEmbed model:

```bash
python scripts/train_model.py
```

Pretrain the embedding model:

```bash
python scripts/pretrain_embedding.py
```

Run model inference:

```bash
python scripts/run_inference.py
```

---

# 📈 Evaluation

Evaluate model performance:

```bash
python scripts/evaluate_model.py
```

Generate a report:

```bash
python scripts/generate_report.py
```

Run ablation experiments:

```bash
python scripts/run_ablations.py
```

---

# ⚡ Real-Time Inference

The prototype also includes a real-time inference pipeline:

```bash
python scripts/realtime_inference.py
```

The pipeline is designed to support incoming surface-ocean observations and generate updated subsurface predictions.

---

# 🧪 Testing

Run the test suite:

```bash
pytest
```

Tests cover:

* Data downloading
* Feature engineering
* Model inference
* Model architecture
* Preprocessing

---

# 📊 Model Evaluation

OceanEmbed evaluates predictions using standard regression and scientific metrics.

| Metric      | Purpose                               |
| ----------- | ------------------------------------- |
| RMSE        | Measures temperature prediction error |
| MAE         | Measures average absolute error       |
| R²          | Measures explained variance           |
| Bias        | Measures systematic prediction error  |
| Correlation | Measures agreement with observations  |

Evaluation is performed both **spatially and depth-wise**.

---

# 🌍 Intended Applications

OceanEmbed can support research and operational applications such as:

* 🌊 Subsurface ocean monitoring
* 🔥 Marine heatwave detection
* 🌡️ Thermocline monitoring
* 🌀 Ocean circulation analysis
* 🐟 Marine ecosystem studies
* 🌍 Climate and ocean research
* 🚢 Oceanographic decision support
* 📡 Satellite-based ocean intelligence

---

# 🔬 Research Contribution

The OceanEmbed prototype focuses on combining several capabilities into a single framework:

```text
Multi-Source Ocean Data
          +
Deep Representation Learning
          +
Subsurface Reconstruction
          +
Physics Constraints
          +
ARGO Validation
          +
Uncertainty Estimation
          +
Marine Heatwave Detection
          =
Integrated AI Ocean Intelligence Framework
```

The key idea is to transform frequently available **surface observations into a learned representation of the ocean's subsurface state**.

---

# ⚠️ Important Disclaimer

OceanEmbed is a **research and prototype system**.

Predictions should not be interpreted as direct measurements unless independently validated against appropriate observational data.

Performance can vary depending on:

* Region
* Season
* Dataset quality
* Spatial resolution
* Temporal coverage
* Depth
* Availability of observations

Synthetic or demonstration data, when used, must not be interpreted as real ocean observations.

---

# 📚 Dataset References

The project is designed to work with publicly available oceanographic and satellite datasets.

See:

```text
DATASET_CITATIONS.md
```

for dataset-specific references and attribution information.

---

# 📄 License

This project is distributed under the license specified in:

```text
LICENSE
```

---

# 👨‍💻 Project

**OceanEmbed**

Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations.

Built as a research prototype for AI-based ocean intelligence and subsurface ocean monitoring.

"""generate_data.py  (OceanEmbed - data module)
=====================================================================
Creates ``ocean_data.npz``: a SYNTHETIC ocean dataset for a 5 deg x 5 deg
box in the Bay of Bengal (15.0-20.0 N, 85.0-90.0 E) for January 2020
(31 daily snapshots) at 0.25 deg resolution (21 x 21 grid points).

The file follows the OceanEmbed Shared Data Contract used by the other
components (see README.md / model.py).  Required keys and shapes:

    sst, ssh, sss, u_current, v_current, u_wind, v_wind
        float32, shape [day, lat, lon] = (31, 21, 21)
    subsurface_temp
        float32, shape [day, depth, lat, lon] = (31, 5, 21, 21)
        depth order [0, 50, 100, 500, 1000] m
    lat, lon
        1-D float arrays of length 21

Informational extra keys (safe for other agents to ignore):
    depth (5,) float32        level depths [0, 50, 100, 500, 1000] m
    time  (31,) int64         day-of-month - 1 for January 2020
    time_iso (31,) <U10       ISO-8601 date string of each snapshot
    synthetic () bool         True - data are SYNTHETIC
    meta () str               JSON metadata (units, provenance, disclaimer)

All values are SYNTHETIC. They are statistically plausible and respect
basic ocean physics - warmer surface / colder deeper water with stable
stratification, geostrophic surface currents around SSH anomalies, a
fresh-north-east salinity pattern, NE-monsoon January winds and two
embedded mesoscale eddies (one warm-core anticyclone, one cool-core
cyclone) - but they are NOT observations.
=====================================================================
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

# ---------------------------------------------------------------------------
# Grid / configuration
# ---------------------------------------------------------------------------
LON = np.arange(85.0, 90.0 + 1e-9, 0.25)          # 21 pts: 85.00..90.00 deg E
LAT = np.arange(15.0, 20.0 + 1e-9, 0.25)          # 21 pts: 15.00..20.00 deg N
DEPTH = np.array([0.0, 50.0, 100.0, 500.0, 1000.0], dtype=np.float32)
N_TIME = 31                                        # January 2020 has 31 days

NLAT, NLON = len(LAT), len(LON)
LON2D, LAT2D = np.meshgrid(LON, LAT)              # (nlat, nlon)
RNG = np.random.default_rng(20200101)             # reproducible synthetic data

# Eddy centres (slow westward drift approximates Rossby wave propagation)
WARM_LON0, WARM_LAT = 87.45, 17.35                # warm-core anticyclonic eddy
COOL_LON0, COOL_LAT = 88.80, 18.70                # cool-core cyclonic eddy
DRIFT_WARM = 0.045                                # deg/day westward
DRIFT_COOL = -0.020                               # deg/day eastward

# ---------------------------------------------------------------------------
# Small building blocks
# ---------------------------------------------------------------------------

def gaussian2d(lonc: float, latc: float, sigma_deg: float) -> np.ndarray:
    """Isotropic Gaussian bump (peak 1.0) centred at (lonc, latc)."""
    return np.exp(-((LON2D - lonc) ** 2 + (LAT2D - latc) ** 2)
                  / (2.0 * sigma_deg ** 2))


def smooth_noise(sigma_cells: float = 2.0, amp: float = 1.0) -> np.ndarray:
    """Unit-std spatial noise, smoothed over sigma_cells grid cells."""
    noise = RNG.standard_normal((NLAT, NLON))
    noise = gaussian_filter(noise, sigma=sigma_cells, mode="reflect")
    return amp * noise / (noise.std() + 1e-12)


def temporal_wave(amp: float = 1.0, n_waves: int = 3) -> np.ndarray:
    """Unit-std time series: sinusoids + red (AR-1) noise."""
    t = np.arange(N_TIME)
    signal = np.zeros(N_TIME)
    for _ in range(n_waves):
        period = RNG.uniform(4.0, 14.0)
        signal += np.sin(2.0 * np.pi * t / period + RNG.uniform(0.0, 2.0 * np.pi))
    white = RNG.standard_normal(N_TIME)
    red = np.zeros(N_TIME)
    red[0] = white[0]
    for i in range(1, N_TIME):
        red[i] = 0.85 * red[i - 1] + 0.30 * white[i]
    signal = signal / n_waves + red
    return amp * signal / (signal.std() + 1e-12)


def wind_swirl(lonc: float, latc: float, sigma_deg: float,
               strength: float, anticyclonic: bool = True):
    """Tangential (rotational) wind field around an eddy, m/s."""
    dx = LON2D - lonc
    dy = LAT2D - latc
    radius = np.hypot(dx, dy) + 1e-9
    env = np.exp(-(dx ** 2 + dy ** 2) / (2.0 * sigma_deg ** 2))
    # counter-clockwise tangential unit vector = (-dy, dx) / r in (lon, lat)
    u = -dy / radius
    v = dx / radius
    sign = -1.0 if anticyclonic else +1.0          # NH: anti -> clockwise
    return sign * strength * env * u, sign * strength * env * v


# Shared large-scale spatio-temporal anomaly: keeps surface + subsurface
# coherent (e.g. warm SST pattern -> warm patches down through the thermocline)
common = np.empty((N_TIME, NLAT, NLON))
for td in range(N_TIME):
    common[td] = smooth_noise(sigma_cells=2.4, amp=1.0)
common = common * temporal_wave(amp=1.0, n_waves=3)[:, None, None]
common = gaussian_filter1d(common, sigma=1.2, axis=0)     # smooth in time
common = common / (common.std() + 1e-12)                  # unit std

# ---------------------------------------------------------------------------
# Vertical profile / depth weights
# ---------------------------------------------------------------------------
PROFILE = np.array([28.6, 27.0, 24.7, 9.3, 4.9])   # mean temp per level, deg C
LAT_GRAD = np.array([0.24, 0.22, 0.18, 0.05, 0.015])  # deg C cooling per deg N
W_COMMON = np.array([1.00, 0.85, 0.60, 0.20, 0.06])  # shared-anomaly depth weight
W_WARM = np.array([1.00, 1.05, 0.95, 0.25, 0.06])     # warm eddy depth weight
W_COOL = np.array([1.00, 1.00, 0.90, 0.30, 0.06])     # cool eddy depth weight
SSH_DEPTH = np.array([0.0, 2.5, 5.0, -1.2, -0.1])      # deg C per m of eddy SSH
DEPTH_NOISE = np.array([0.30, 0.25, 0.35, 0.20, 0.08])  # per-level noise std

sst = np.empty((N_TIME, NLAT, NLON))
ssh = np.empty_like(sst)
sss = np.empty_like(sst)
u_current = np.empty_like(sst)
v_current = np.empty_like(sst)
u_wind = np.empty_like(sst)
v_wind = np.empty_like(sst)
subsurface_temp = np.empty((N_TIME, DEPTH.size, NLAT, NLON))
wind_syn = temporal_wave(amp=1.6, n_waves=2)          # synoptic wind scale

# ---------------------------------------------------------------------------
# Generate the fields, day by day
# ---------------------------------------------------------------------------
DAY = np.arange(N_TIME)

for td in DAY:
    # --- eddy geometries (slowly drifting centres) -----------------------
    warm_lon = WARM_LON0 - DRIFT_WARM * td
    cool_lon = COOL_LON0 - DRIFT_COOL * td
    # Eddies: slightly irregular (not perfectly smooth Gaussians)
    gg_warm = gaussian2d(warm_lon, WARM_LAT, 0.85) + 0.08 * smooth_noise(0.7, 1.0)
    gg_cool = gaussian2d(cool_lon, COOL_LAT, 0.70) + 0.08 * smooth_noise(0.7, 1.0)

    # --- sea surface temperature (deg C) ---------------------------------
    sst_t = 28.7 - 0.24 * (LAT2D - 15.0) + 0.06 * (LON2D - 85.0) / 5.0
    sst_t = sst_t + 1.30 * gg_warm - 0.85 * gg_cool + 0.55 * common[td]
    sst_t = sst_t + smooth_noise(sigma_cells=1.2, amp=0.12) + 0.05 * RNG.standard_normal((NLAT, NLON))
    sst[td] = sst_t

    # --- sea surface height / SLA (m) ------------------------------------
    ssh_warm = 0.22 * gaussian2d(warm_lon, WARM_LAT, 1.05)       # positive dome
    ssh_cool = -0.15 * gaussian2d(cool_lon, COOL_LAT, 0.75)      # negative trough
    ssh_t = (ssh_warm + ssh_cool
             - 0.015 * (LAT2D - 15.0) / 5.0 + 0.010 * (LON2D - 85.0) / 5.0
             + 0.030 * common[td] + smooth_noise(1.0, 0.004))
    ssh[td] = ssh_t

    # --- sea surface salinity (PSU) --------------------------------------
    sss_t = 33.9 - 0.70 * (LAT2D - 15.0) / 5.0 - 0.15 * (LON2D - 85.0) / 5.0
    sss_t = sss_t + 0.12 * gg_warm - 0.12 * gg_cool
    sss_t = sss_t + 0.08 * common[td] + smooth_noise(1.4, 0.03)
    sss[td] = sss_t

    # --- surface geostrophic currents (m/s) from SSH ---------------------
    f_cor = 2.0 * 7.292115e-5 * np.sin(np.radians(LAT2D))
    dlon_m = np.deg2rad(LON[1] - LON[0]) * 6371000.0 * np.cos(np.radians(LAT2D))
    dlat_m = np.deg2rad(LAT[1] - LAT[0]) * 6371000.0
    dssh_dx = np.gradient(ssh_t, axis=1) / dlon_m
    dssh_dy = np.gradient(ssh_t, axis=0) / dlat_m
    u_geo = -(9.81 / f_cor) * dssh_dy
    v_geo = (9.81 / f_cor) * dssh_dx
    u_current[td] = np.clip(u_geo + smooth_noise(1.5, 0.015) + 0.03 * common[td],
                            -0.9, 0.9)
    v_current[td] = np.clip(v_geo + smooth_noise(1.5, 0.015) - 0.02 * common[td],
                            -0.9, 0.9)

    # --- surface winds (m/s): NE monsoon + synoptic + eddy swirl ---------
    u_wind_t = -3.6 - 0.15 * (LAT2D - 15.0) / 5.0 + 0.10 * (LON2D - 85.0) / 5.0
    v_wind_t = -2.8 + 0.10 * (LAT2D - 15.0) / 5.0 - 0.15 * (LON2D - 85.0) / 5.0
    u_wind_t = u_wind_t + wind_syn[td] * smooth_noise(2.6, 1.0)
    v_wind_t = v_wind_t + wind_syn[td] * smooth_noise(2.6, 1.0)
    u_wind_t = u_wind_t + wind_swirl(warm_lon, WARM_LAT, 1.2, 1.2, True)[0]
    v_wind_t = v_wind_t + wind_swirl(warm_lon, WARM_LAT, 1.2, 1.2, True)[1]
    u_wind_t = u_wind_t + wind_swirl(cool_lon, COOL_LAT, 1.0, 1.0, False)[0]
    v_wind_t = v_wind_t + wind_swirl(cool_lon, COOL_LAT, 1.0, 1.0, False)[1]
    u_wind[td] = u_wind_t
    v_wind[td] = v_wind_t

    # --- subsurface temperature (deg C), one profile per depth -----------
    for d in range(DEPTH.size):
        layer = PROFILE[d] - LAT_GRAD[d] * (LAT2D - 15.0)   # mean profile
        layer = layer + W_COMMON[d] * common[td]            # shared anomaly
        layer = layer + 1.30 * W_WARM[d] * gg_warm          # warm eddy
        layer = layer - 0.85 * W_COOL[d] * gg_cool          # cool eddy
        layer = layer + SSH_DEPTH[d] * (ssh_warm + ssh_cool)  # thermocline heave
        layer = layer + DEPTH_NOISE[d] * smooth_noise(1.5, 1.0)
        subsurface_temp[td, d] = layer

# ---------------------------------------------------------------------------
# Post-processing: temporal smoothing, surface consistency, stratification
# ---------------------------------------------------------------------------
for field in (sst, ssh, sss, u_current, v_current, u_wind, v_wind,
              subsurface_temp):
    field[:] = gaussian_filter1d(field, sigma=0.8, axis=0)

# 0-m level must equal SST (single coherent surface field)
subsurface_temp[:, 0, :, :] = sst

# enforce stable stratification: T(d) >= T(d+1) + 0.02 deg C at every point
for d in range(DEPTH.size - 1):
    subsurface_temp[:, d] = np.maximum(subsurface_temp[:, d],
                                       subsurface_temp[:, d + 1] + 0.02)
subsurface_temp = np.clip(subsurface_temp, -2.0, 34.0)
sst = np.clip(sst, 15.0, 32.0)

# ---------------------------------------------------------------------------
# Metadata + save
# ---------------------------------------------------------------------------
meta = {
    "dataset": "OceanEmbed - synthetic Bay-of-Bengal prototype (data half)",
    "synthetic": True,
    "disclaimer": "SYNTHETIC DATA GENERATED FOR SOFTWARE-PROTOTYPE USE ONLY. "
                  "Not observations; must not be used for real-world decisions.",
    "region": {"box": "15-20 deg N, 85-90 deg E", "dx_deg": 0.25, "dy_deg": 0.25},
    "time": {"period": "January 2020", "n_days": 31, "step": "1 day"},
    "units": {
        "sst": "deg C", "ssh": "m", "sss": "PSU",
        "u_current": "m/s (eastward positive)", "v_current": "m/s (northward positive)",
        "u_wind": "m/s (eastward positive)", "v_wind": "m/s (northward positive)",
        "subsurface_temp": "deg C",
    },
    "subsurface_levels_m": [0.0, 50.0, 100.0, 500.0, 1000.0],
    "features": [
        "poleward-cooling annual mean field",
        "stable stratification (monotonic cooling with depth)",
        "warm-core anticyclonic eddy + cool-core cyclonic eddy (mesoscale)",
        "geostrophic surface currents from SSH gradients",
        "NE-monsoon January surface winds with synoptic variability",
        "river-influenced fresher water in the north-east",
    ],
    "generator": "generate_data.py",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "reproducibility": "numpy.random.default_rng(seed=20200101)",
    "contract_version": "1.0",
}

time_iso = np.array([datetime(2020, 1, 1 + d).strftime("%Y-%m-%d")
                     for d in DAY], dtype="U10")

np.savez_compressed(
    "ocean_data.npz",
    lon=LON.astype(np.float32),
    lat=LAT.astype(np.float32),
    time=DAY.astype(np.int64),
    time_iso=time_iso,
    depth=DEPTH,
    sst=sst.astype(np.float32),
    ssh=ssh.astype(np.float32),
    sss=sss.astype(np.float32),
    u_current=u_current.astype(np.float32),
    v_current=v_current.astype(np.float32),
    u_wind=u_wind.astype(np.float32),
    v_wind=v_wind.astype(np.float32),
    subsurface_temp=subsurface_temp.astype(np.float32),
    synthetic=np.array(True, dtype=np.bool_),
    is_synthetic=np.array(True, dtype=np.bool_),
    meta=np.array(json.dumps(meta, indent=2)),
)

# ---------------------------------------------------------------------------
# Self-check + summary
# ---------------------------------------------------------------------------
print("=" * 72)
print("generate_data.py  ->  ocean_data.npz  (SYNTHETIC)")
print("=" * 72)
for key in ("sst", "ssh", "sss", "u_current", "v_current",
            "u_wind", "v_wind", "subsurface_temp"):
    arr = np.load("ocean_data.npz")[key]
    print(f"  {key:<18s} shape={str(arr.shape):<24s} "
          f"min={arr.min():8.3f}  mean={arr.mean():8.3f}  max={arr.max():8.3f}")
assert not np.isnan(np.load("ocean_data.npz")["subsurface_temp"]).any()

# stratification + eddy checks
with np.load("ocean_data.npz") as z:
    t = z["subsurface_temp"]
    ok = all(np.all(t[:, d] >= t[:, d + 1] + 0.02) for d in range(4))
    sst0 = z["sst"][0]
    lon2, lat2 = np.meshgrid(z["lon"], z["lat"])
    bg = 28.7 - 0.24 * (lat2 - 15.0) + 0.06 * (lon2 - 85.0) / 5.0  # planar mean
    anom = sst0 - bg

    def local_extremum(clon, clat, want_max):
        """Local max/min of the background-removed anomaly near an eddy."""
        within = np.sqrt((lon2 - clon) ** 2 + (lat2 - clat) ** 2) <= 1.5
        masked = np.where(within, anom, -np.inf if want_max else np.inf)
        iy, ix = np.unravel_index(int(np.argmax(masked) if want_max
                                      else np.argmin(masked)), anom.shape)
        return z["lat"][iy], z["lon"][ix], float(anom[iy, ix])

    wy, wx, wv = local_extremum(WARM_LON0, WARM_LAT, True)
    cy, cx, cv = local_extremum(COOL_LON0, COOL_LAT, False)
    print(f"  stable stratification (T above >= T below):        {ok}")
    print(f"  warm-core eddy (nominal 17.35 N, 87.45 E): "
          f"SST anomaly +{wv:.2f} deg C near ({wy:.2f} N, {wx:.2f} E)")
    print(f"  cool-core eddy (nominal 18.70 N, 88.80 E): "
          f"SST anomaly {cv:.2f} deg C near ({cy:.2f} N, {cx:.2f} E)")
    print(f"  no missing values (NaNs):                         "
          f"{not bool(np.isnan(t).any())}")
print("  saved: ocean_data.npz  (contract version 1.0, marked synthetic)")
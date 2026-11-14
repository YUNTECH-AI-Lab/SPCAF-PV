"""Configuration of the SynPV generator.

All keys are upper-case and correspond one-to-one to the parameters described in
the paper.  Values set to ``"auto"`` are resolved from the chosen location and
its NASA POWER weather when the simulation starts; the resolved values are
written to ``config_used.json`` in the output folder, so every run is
reproducible.
"""
from __future__ import annotations

import copy
import json
import os
from typing import Any

DEFAULT_CONFIG: dict[str, Any] = dict(
    # --- Site ---
    LATITUDE=40.7259, LONGITUDE=-73.5143,
    ALTITUDE_M="auto",                 # "auto" -> elevation reported by NASA POWER
    SITE_NAME="auto",                  # "auto" -> "lat, lon"
    TIMEZONE="auto",                   # "auto" -> IANA zone from coordinates

    # --- Period / resolution (timestamps label the START of each interval) ---
    START_DATE="2024-01-01", END_DATE="2024-12-31", FREQ_MIN=5,
    CLOUD_SUBSTEPS=5,                  # sub-samples averaged inside each interval

    # --- Population ---
    N_HOUSES=100, PV_OWNERSHIP=0.65,
    SYSTEM_KW_MEDIAN=7.0, SYSTEM_KW_SIGMA=0.35,   # lognormal DC size
    SYSTEM_KW_RANGE=(3.0, 14.0),
    MIN_PANELS=8, MAX_PANELS=36,

    # --- Fallback module (only if the CEC database cannot be loaded) ---
    MODULE_WATTS=400.0, MODULE_EFF=0.21, GAMMA_PMAX=-0.0032,
    G_STC=1000.0, T_STC=25.0,

    # --- Roof geometry ---
    TILT_RANGE=(15.0, 40.0),
    AZIMUTH_CHOICES=[180.0, 135.0, 225.0, 90.0, 270.0],   # northern-hemisphere values
    AZIMUTH_WEIGHTS=[0.40, 0.20, 0.20, 0.10, 0.10],
    AZIMUTH_JITTER_DEG=8.0,
    EQUATOR_FACING="auto",             # "auto": rotate azimuths/obstacles by 180 deg south of the equator
    PORTRAIT_FRACTION=0.85,

    # --- Optics ---
    ALBEDO=0.20, TRANSPOSITION="perez", IAM_A_R=0.16,
    SKYVIEW_LOSS_RANGE=(0.00, 0.08),

    # --- Cell temperature ---
    THERMAL_MODEL="noct_sam",          # "noct_sam" | "faiman"
    MOUNT_STANDOFF_IN=[1.0, 2.0, 3.0, 4.0],
    MOUNT_STANDOFF_WEIGHTS=[0.10, 0.30, 0.45, 0.15],
    TWO_STORY_FRACTION=0.5,
    MODULE_NOCT_DEFAULT=45.0,
    FAIMAN_U0=25.0, FAIMAN_U1=6.84,

    # --- DC / BOS losses ---
    DC_WIRING_LOSS=0.020, CONNECTION_LOSS=0.005,
    MISMATCH_LOSS=0.020, MISMATCH_LOSS_OPTIMIZER=0.005,
    OPTIMIZER_EFF=0.988, LID_LOSS=0.015,
    NAMEPLATE_TOL_RANGE=(-0.02, 0.01),
    AGE_RANGE_YEARS=(0.0, 10.0), DEGRADATION_RANGE=(0.003, 0.008),
    DC_AC_RATIO_RANGE=(1.05, 1.30),
    DESIGN_T_MIN="auto",               # "auto" -> extreme annual mean minimum (NASA POWER daily, 10 yr)
    DESIGN_T_CELL_HOT=65.0,
    DESIGN_T_CELL_COLD_MPP=0.0,
    VDC_ABS_MAX=600.0,                 # 600 V: NEC 690.7 (US dwellings); 1000 V typical elsewhere
    MPPT_INPUTS_CURRENT_FACTOR=1.25,

    # --- Soiling (Kimber-type) ---
    SOILING_RATE_RANGE=(0.0005, 0.0015),
    SOILING_MAX=0.04, SOILING_CLEAN_MM=5.0,

    # --- Shading (horizon obstacles) ---
    BASE_HORIZON_RANGE=(2.0, 8.0),
    OBSTACLE_HORIZON_RANGE=(8.0, 30.0),
    OBSTACLE_TRANSMITTANCE=(0.0, 0.5),

    # --- Snow (ground pack + NREL/Marion panel model) ---
    SNOW_PARTITION="wetbulb",          # "wetbulb" (Stull 2011) | "drybulb"
    SNOW_ALL_BELOW_C=0.0, RAIN_ALL_ABOVE_C=2.0,
    WETBULB_SNOW_C=0.0, WETBULB_RAIN_C=1.5,
    SNOW_LIQUID_RATIO=10.0,
    DEGREE_DAY_FACTOR=3.0,
    MARION_THRESHOLD_CM_HR=1.0, MARION_M=-80.0, MARION_SLIDE=0.197,

    # --- Cloud field ---
    NEIGHBOURHOOD_KM=3.0,
    CLOUD_N_MODES=64, CLOUD_LAMBDA_KM=(1.0, 20.0),
    CLOUD_LIFETIME_MIN_PER_KM=20.0,
    CLOUD_EDGE_WIDTH=0.15,
    CLOUD_TRANSMITTANCE=0.35,
    CLOUD_THICKNESS_SIGMA=0.25,
    CLOUD_ENHANCEMENT=0.25,
    CLOUD_SPEED_RANGE=(3.0, 30.0),
    CLOUD_PREVAILING_FROM_DEG="auto",  # "auto" -> climatological wind belt of the latitude
    CLOUD_DIR_SIGMA_DEG=55.0,

    # --- Metering realism (applied to the 'measured' output only) ---
    METER_NOISE_REL=0.005, METER_NOISE_ABS_W=5.0,
    GAP_EVENTS_PER_YEAR=12.0, GAP_MEDIAN_HOURS=1.0,
    OUTAGE_EVENTS_PER_YEAR=0.5, OUTAGE_DAYS_RANGE=(1, 10),
    EXPORT_LIMIT_FRACTION_OF_HOUSES=0.05, EXPORT_LIMIT_REL=0.8,
    CLOCK_OFFSET_FRACTION=0.05,

    # --- Run / IO ---
    SEED=42, SAVE_CSV=True, COMPRESS=True,
    EXPORT_GROUND_TRUTH=True,
    MAKE_FIGURES=True,
    NASA_HOUR_CONVENTION="auto",       # "auto" | "start" | "end"
    NASA_CACHE_DIR="nasa_cache",
    ALLOW_SYNTHETIC_WEATHER=False,     # True: fall back to synthetic weather if NASA POWER fails
    OUTPUT_DIR="auto",                 # "auto" -> sim_output_<period>
    FLEET_REFERENCE_YEAR=2024,
    PVWATTS_BENCHMARK=True,
)

# Settings that reproduce the case study of the paper exactly
# (Levittown, NY, 100 houses, 2024).
PAPER_PRESET: dict[str, Any] = dict(
    LATITUDE=40.7259, LONGITUDE=-73.5143, ALTITUDE_M=20.0,
    SITE_NAME="Levittown, NY", TIMEZONE="America/New_York",
    START_DATE="2024-01-01", END_DATE="2024-12-31",
    N_HOUSES=100, DESIGN_T_MIN=-18.0, CLOUD_PREVAILING_FROM_DEG=260.0,
    EQUATOR_FACING=False, SEED=42,
)

_TUPLE_KEYS = {k for k, v in DEFAULT_CONFIG.items() if isinstance(v, tuple)}


def default_config(**overrides: Any) -> dict[str, Any]:
    """Return a copy of the default configuration, updated with ``overrides``."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    update_config(cfg, overrides)
    return cfg


def paper_config(**overrides: Any) -> dict[str, Any]:
    """Configuration of the paper case study (optionally modified)."""
    cfg = default_config(**PAPER_PRESET)
    update_config(cfg, overrides)
    return cfg


def update_config(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    unknown = [k for k in overrides if k not in DEFAULT_CONFIG]
    if unknown:
        raise KeyError(f"unknown configuration key(s): {', '.join(sorted(unknown))}")
    for k, v in overrides.items():
        cfg[k] = tuple(v) if (k in _TUPLE_KEYS and isinstance(v, list)) else v
    return cfg


def load_config(path: str) -> dict[str, Any]:
    """Load a JSON (or YAML, if PyYAML is installed) file of overrides."""
    with open(path, encoding="utf-8") as fh:
        if path.lower().endswith((".yaml", ".yml")):
            import yaml  # optional dependency
            data = yaml.safe_load(fh) or {}
        else:
            data = json.load(fh)
    preset = data.pop("PRESET", None)
    base = paper_config() if preset == "paper" else default_config()
    return update_config(base, data)


def save_config(cfg: dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, default=str)


def prevailing_wind_from(lat: float) -> float:
    """Climatological direction (deg, FROM) of the cloud-steering flow.

    Trade easterlies in the tropics, westerlies at mid-latitudes and polar
    easterlies at high latitudes.  This is a coarse default; override it when
    the local steering-level climatology is known.
    """
    a = abs(lat)
    if a < 25.0:
        return 70.0 if lat >= 0 else 110.0
    if a < 65.0:
        return 260.0 if lat >= 0 else 280.0
    return 80.0 if lat >= 0 else 100.0


def timezone_for(lat: float, lon: float) -> str:
    """IANA time zone of a coordinate (falls back to a fixed UTC offset)."""
    try:
        from timezonefinder import TimezoneFinder
        tz = TimezoneFinder().timezone_at(lat=lat, lng=lon)
        if tz:
            return tz
    except Exception:
        pass
    off = int(round(lon / 15.0))
    return "UTC" if off == 0 else f"Etc/GMT{-off:+d}"   # POSIX sign convention

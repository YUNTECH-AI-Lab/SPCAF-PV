"""Human-readable description of an exported dataset."""
from __future__ import annotations

import datetime as _dt

from . import __version__

FLAG_CODES = {0: "normal", 1: "communication gap (value missing)", 2: "inverter outage (zero output)",
              3: "export curtailment", 4: "clock offset"}


def write_dataset_readme(sim, path: str) -> None:
    cfg, m = sim.cfg, sim.metrics
    lines = [
        f"# SynPV dataset - {cfg['SITE_NAME']}",
        "",
        f"Generated with SynPV {__version__} on {_dt.datetime.now():%Y-%m-%d %H:%M}.",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Location | {cfg['LATITUDE']:.4f}, {cfg['LONGITUDE']:.4f} (elevation {cfg['ALTITUDE_M']:.0f} m, "
        f"time zone {cfg['TIMEZONE']}) |",
        f"| Period | {cfg['START_DATE']} to {cfg['END_DATE']} at {cfg['FREQ_MIN']}-min resolution "
        f"({m['time_steps']:,} steps) |",
        f"| Houses | {m['n_houses']} ({m['n_pv_houses']} with PV, {m['dc_capacity_kw']:.1f} kWdc / "
        f"{m['ac_capacity_kw']:.1f} kWac) |",
        f"| Weather source | {m['weather_source']} (hourly, satellite/reanalysis-derived) |",
        f"| Random seed | {cfg['SEED']} |",
        "",
        "Time stamps are in local time (column `timestamp_interval_start`) and label the START of each",
        "interval; every value is the average over the interval.  Power is in W.  Houses without PV have",
        "zero power.  The complete configuration (including values resolved from `auto`) is in",
        "`config_used.json`; running SynPV with that file reproduces this dataset exactly.",
        "",
        "## Files",
        "",
        "| File | Content |",
        "|---|---|",
        "| `pv_output_5min.csv.gz` | Ideal AC power of every house (W) |",
        "| `pv_output_5min_measured.csv.gz` | Measured AC power with metering artefacts (W, NaN = missing) |",
        "| `meter_flags.csv.gz` | Quality flag of every measured sample (codes below) |",
        "| `weather_data.csv.gz` | Regional weather, clear sky, cloud cover and cloud motion |",
        "| `pv_metadata.csv` | System metadata and annual indicators of every house |",
        "| `per_house_results.csv` | Compact per-house summary |",
        "| `pv_dataset.npz` | All arrays (time x house), including ground-truth irradiance, effective "
        "irradiance, cell temperature, DC power and voltage, cloud mask, soiling, shading and snow |",
        "| `summary.json` | Validation metrics and checks |",
        "| `fig*.png` | Diagnostic figures |",
        "",
        "## Quality flags",
        "",
        "| Code | Meaning |",
        "|---|---|",
    ] + [f"| {k} | {v} |" for k, v in FLAG_CODES.items()] + [
        "",
        "## Loading in Python",
        "",
        "```python",
        "import numpy as np, pandas as pd",
        "p = pd.read_csv('pv_output_5min.csv.gz', index_col=0, parse_dates=True)",
        "d = np.load('pv_dataset.npz')          # e.g. d['cell_temp_c'], d['cloud_mask']",
        "```",
        "",
        "## Citation",
        "",
        "If you use this dataset, please cite the SynPV paper and software (see `CITATION.cff` in the",
        "SynPV repository).",
        "",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

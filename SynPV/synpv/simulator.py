"""SynPV simulation engine.

The generator converts hourly NASA POWER meteorology into 5-min AC power of a
synthetic neighbourhood of residential PV systems:

    1  CEC module / inverter databases
    2  NASA POWER hourly meteorology (download + cache)
    3  time base, solar geometry and clear sky
    4  clear-sky index, interpolation to interval mid-points
    5  heterogeneous PV fleet (CEC hardware, string sizing, orientation ...)
    6  advecting cloud field -> per-house GHI / DNI / DHI
    7  snowpack, soiling and horizon shading
    8  Perez transposition, IAM, effective irradiance
    9  panel snow coverage (Marion et al. 2013)
    10 SAM NOCT temperature, CEC single diode, MPPT window, Sandia inverter
    11 metering realism layer (measured output + quality flags)
    12 validation (PR, specific yield, PVWatts v5 benchmark, variability)
    13 figures and dataset export

Usage::

    from synpv import Simulation, default_config
    cfg = default_config(LATITUDE=48.14, LONGITUDE=11.58, START_DATE="2023-01-01",
                         END_DATE="2023-12-31", N_HOUSES=50)
    result = Simulation(cfg).run()
    print(result.metrics["performance_ratio_mean"], result.output_dir)
"""
from __future__ import annotations

import json
import math
import os
import time as _time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
import pvlib
from scipy.special import expit, ndtri

from . import __version__
from .cec import (DEFAULT_BRAND_W, INVERTER_BRAND_WEIGHTS, load_cec_inverters,
                  load_cec_modules)
from .config import (default_config, prevailing_wind_from, save_config,
                     timezone_for, update_config)
from .weather import (NASA_FIRST_DATE, WeatherError, design_minimum_temperature,
                      load_or_fetch, synthetic_weather)

STEPS = [
    "CEC databases", "NASA POWER weather", "Solar geometry", "Clear-sky index",
    "PV fleet", "Cloud field", "Snow, soiling, shading", "Plane-of-array irradiance",
    "Panel snow cover", "Thermal and electrical model", "Metering layer",
    "Validation", "Figures", "Export",
]


class SimulationCancelled(RuntimeError):
    """Raised when :meth:`Simulation.cancel` was called during a run."""


@dataclass
class SimulationResult:
    output_dir: str
    config: dict
    metrics: dict
    checks: list
    files: list
    fleet: pd.DataFrame = field(repr=False)
    runtime_s: float = 0.0


# ==============================================================================
# helpers
# ==============================================================================
def _iam_beam(aoi, a_r):
    if hasattr(pvlib, "iam") and hasattr(pvlib.iam, "martin_ruiz"):
        return np.asarray(pvlib.iam.martin_ruiz(aoi, a_r=a_r), dtype=float)
    aoi = np.asarray(aoi, dtype=float)
    return np.where(aoi < 90.0,
                    (1 - np.exp(-np.cos(np.radians(aoi)) / a_r)) / (1 - np.exp(-1 / a_r)), 0.0)


def _iam_diffuse(tilt, a_r):
    """Martin-Ruiz diffuse IAMs (sky, ground) for a given tilt."""
    try:
        sky, gnd = pvlib.iam.martin_ruiz_diffuse(tilt, a_r=a_r)
        return float(sky), float(gnd)
    except Exception:
        return 0.95, 0.85


def _nanmean(a):
    a = np.asarray(a, dtype=float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def _pct(a, q):
    a = np.asarray(a, dtype=float)
    return float(np.percentile(a, q)) if a.size else float("nan")


def wet_bulb_stull(t, rh):
    """Stull (2011) wet-bulb temperature [C] from air T [C] and RH [%]."""
    rh = np.clip(rh, 5, 99)
    return (t * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) + np.arctan(t + rh)
            - np.arctan(rh - 1.676331) + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh) - 4.686035)


def inverter_efficiency_fallback(load_ratio):
    r = np.array([0.000, 0.005, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00, 1.40])
    e = np.array([0.000, 0.500, 0.850, 0.900, 0.930, 0.955, 0.968, 0.975, 0.972, 0.965, 0.960])
    return np.interp(load_ratio, r, e)


def estimate_memory_gb(cfg) -> float:
    """Approximate peak memory of a run (GB)."""
    n_t = (pd.Timestamp(cfg["END_DATE"]) - pd.Timestamp(cfg["START_DATE"])).days + 1
    n_t *= 24 * 60 / cfg["FREQ_MIN"]
    return float(n_t * cfg["N_HOUSES"] * 4 * 26 / 1e9)


def _string_layout(n_pan, mod, inv, cfg, solaredge):
    """Return (Ns, Np) satisfying inverter limits, or None."""
    voc, vmp, imp = mod["V_oc_ref"], mod["V_mp_ref"], mod["I_mp_ref"]
    beta_rel = mod["beta_oc"] / voc                       # 1/K (negative)
    voc_cold = voc * (1 + beta_rel * (cfg["DESIGN_T_MIN"] - 25))
    vmp_hot = vmp * (1 + beta_rel * (cfg["DESIGN_T_CELL_HOT"] - 25))
    vmp_cold = vmp * (1 + beta_rel * (cfg["DESIGN_T_CELL_COLD_MPP"] - 25))
    # in the CEC/SAM table 'Vdcmax' is (in practice) the top of the MPPT range;
    # the absolute input limit is the larger of it and the code limit (600 V NEC)
    v_abs = max(inv["Vdcmax"], cfg["VDC_ABS_MAX"])
    for np_ in (1, 2, 3):
        ns = int(round(n_pan / np_))
        if ns < 1:
            continue
        if solaredge:
            if 8 <= ns <= 25:
                return ns, np_
            continue
        if (ns * voc_cold <= v_abs and ns * vmp_hot >= inv["Mppt_low"]
                and ns * vmp_cold <= inv["Mppt_high"] * 1.05
                and np_ * imp <= inv["Idcmax"] * cfg["MPPT_INPUTS_CURRENT_FACTOR"]):
            return ns, np_
    return None


def build_pv_fleet(cfg, rng, cec_mods, cec_invs, az_rot=0.0):
    """Heterogeneous residential fleet.  ``az_rot`` = 180 for equator-facing
    roofs in the southern hemisphere (0 reproduces the northern case exactly)."""
    n = cfg["N_HOUSES"]
    half = cfg["NEIGHBOURHOOD_KM"] / 2
    x_km = rng.uniform(-half, half, n)
    y_km = rng.uniform(-half, half, n)
    lat = cfg["LATITUDE"] + y_km / 111.0
    lon = cfg["LONGITUDE"] + x_km / (111.0 * np.cos(np.radians(cfg["LATITUDE"])))
    has_pv = rng.random(n) < cfg["PV_OWNERSHIP"]
    az_choices = list(cfg["AZIMUTH_CHOICES"]) if not az_rot else \
        [(a + az_rot) % 360 for a in cfg["AZIMUTH_CHOICES"]]

    rows = []
    for i in range(n):
        r = dict(house_id=f"house_{i + 1:03d}", latitude=lat[i], longitude=lon[i],
                 x_km=x_km[i], y_km=y_km[i], has_pv=bool(has_pv[i]))
        if not has_pv[i]:
            rows.append(r)
            continue
        target_kw = float(np.clip(rng.lognormal(np.log(cfg["SYSTEM_KW_MEDIAN"]), cfg["SYSTEM_KW_SIGMA"]),
                                  *cfg["SYSTEM_KW_RANGE"]))
        dc_ac_target = rng.uniform(*cfg["DC_AC_RATIO_RANGE"])
        chosen = None
        for _attempt in range(300):
            if cec_mods is not None:
                mname = cec_mods.columns[rng.integers(cec_mods.shape[1])]
                mod = cec_mods[mname].to_dict()
            else:
                mname = "Generic_400W_Mono"
                mod = dict(STC=cfg["MODULE_WATTS"], A_c=cfg["MODULE_WATTS"] / (cfg["MODULE_EFF"] * 1000),
                           N_s=66, I_sc_ref=12.3, V_oc_ref=41.2, I_mp_ref=11.5, V_mp_ref=34.8,
                           alpha_sc=0.0045, beta_oc=-0.11, T_NOCT=cfg["MODULE_NOCT_DEFAULT"],
                           gamma_r=cfg["GAMMA_PMAX"] * 100)
            n_pan = int(np.clip(round(target_kw * 1000 / mod["STC"]), cfg["MIN_PANELS"], cfg["MAX_PANELS"]))
            if cec_invs is None:
                ns, np_ = n_pan, 1
                chosen = (mname, mod, "Generic_fallback", None, ns, np_)
                break
            pdc = n_pan * mod["STC"]
            paco = cec_invs.loc["Paco"].to_numpy()
            cand = np.where(np.abs(paco - pdc / dc_ac_target) / (pdc / dc_ac_target) < 0.12)[0]
            # weighted random order (Efraimidis-Spirakis): market-share-like mix
            w = np.array([INVERTER_BRAND_WEIGHTS.get(str(cec_invs.columns[c]).split("_")[0].lower(),
                                                     DEFAULT_BRAND_W) for c in cand])
            cand = cand[np.argsort(-np.log(rng.random(len(cand))) / np.maximum(w, 1e-6))]
            for c in cand[:40]:
                iname = cec_invs.columns[c]
                inv = cec_invs[iname].to_dict()
                se = iname.lower().startswith("solaredge")
                lay = _string_layout(n_pan, mod, inv, cfg, se)
                if lay is not None:
                    chosen = (mname, mod, iname, inv, *lay)
                    break
            if chosen:
                break
        if chosen is None:
            raise RuntimeError(f"could not configure house {i} (check DC_AC_RATIO_RANGE / VDC_ABS_MAX)")
        mname, mod, iname, inv, ns, np_ = chosen
        n_pan = ns * np_
        cap_kw = n_pan * mod["STC"] / 1000
        if inv is None:
            paco = cap_kw * 1000 / dc_ac_target
            inv = dict(Paco=paco, Vdco=400.0, Vdcmax=600.0, Mppt_low=100.0, Mppt_high=550.0)
        se = iname.lower().startswith("solaredge")

        # horizon profile (72 bins x 5 deg): base horizon + 0-2 obstacles
        hz_el = np.full(72, rng.uniform(*cfg["BASE_HORIZON_RANGE"]))
        hz_tr = np.zeros(72)
        for _ in range(rng.integers(0, 3)):
            c = rng.uniform(60, 300)
            if az_rot:
                c = (c + az_rot) % 360
            w = rng.uniform(15, 55)
            el = rng.uniform(*cfg["OBSTACLE_HORIZON_RANGE"])
            tr = rng.uniform(*cfg["OBSTACLE_TRANSMITTANCE"])
            bins = np.abs(((np.arange(72) * 5 + 2.5 - c + 180) % 360) - 180) < w
            upd = bins & (el > hz_el)
            hz_el[upd] = el
            hz_tr[upd] = tr

        r.update(
            number_of_panels=n_pan, Ns=ns, Np=np_, capacity_kw=round(cap_kw, 3),
            array_area_m2=round(n_pan * mod["A_c"], 2),
            tilt=round(rng.uniform(*cfg["TILT_RANGE"]), 1),
            azimuth=round((rng.choice(az_choices, p=cfg["AZIMUTH_WEIGHTS"])
                           + rng.normal(0, cfg["AZIMUTH_JITTER_DEG"])) % 360, 1),
            portrait=bool(rng.random() < cfg["PORTRAIT_FRACTION"]),
            module_manufacturer=mname.split("_")[0], module_model=mname,
            module_pmp_w=mod["STC"], module_area_m2=mod["A_c"],
            module_efficiency=round(mod["STC"] / (mod["A_c"] * 1000), 4),
            module_voc_v=mod["V_oc_ref"], module_isc_a=mod["I_sc_ref"],
            module_vmp_v=mod["V_mp_ref"], module_imp_a=mod["I_mp_ref"],
            module_alpha_sc=mod["alpha_sc"], module_beta_oc=mod["beta_oc"],
            module_cells=int(mod["N_s"]),
            inverter_manufacturer=iname.split("_")[0], inverter_model=iname,
            inverter_paco_w=inv["Paco"], inverter_vdco_v=inv["Vdco"],
            inverter_vdcmax_v=inv["Vdcmax"], inverter_mppt_low_v=inv["Mppt_low"],
            inverter_mppt_high_v=inv["Mppt_high"],
            inverter_capacity_kw=round(inv["Paco"] / 1000, 3),
            dc_ac_ratio=round(cap_kw * 1000 / inv["Paco"], 3),
            optimizer_architecture=se,
            age_years=round(rng.uniform(*cfg["AGE_RANGE_YEARS"]), 2),
            degradation_rate=round(rng.uniform(*cfg["DEGRADATION_RANGE"]), 4),
            nameplate_tolerance=round(rng.uniform(*cfg["NAMEPLATE_TOL_RANGE"]), 4),
            skyview_loss=round(rng.uniform(*cfg["SKYVIEW_LOSS_RANGE"]), 4),
            soiling_rate=round(rng.uniform(*cfg["SOILING_RATE_RANGE"]), 5),
            _mod=mod, _inv=inv, _hz_el=hz_el, _hz_tr=hz_tr,
        )
        rows.append(r)
    df = pd.DataFrame(rows)
    for c in ("number_of_panels", "Ns", "Np", "capacity_kw", "tilt", "azimuth", "inverter_capacity_kw",
              "dc_ac_ratio", "module_manufacturer", "inverter_manufacturer", "module_model", "inverter_model"):
        if c not in df.columns:                      # no PV house at all
            df[c] = np.nan
    num_fill = {"number_of_panels": 0, "Ns": 0, "Np": 0, "capacity_kw": 0.0, "tilt": 0.0,
                "azimuth": 180.0, "inverter_capacity_kw": 0.0, "dc_ac_ratio": 0.0}
    df = df.fillna(value=num_fill)
    for c in ("module_manufacturer", "inverter_manufacturer", "module_model", "inverter_model"):
        df[c] = df[c].fillna("-")
    df[["number_of_panels", "Ns", "Np"]] = df[["number_of_panels", "Ns", "Np"]].astype(int)
    return df


# ==============================================================================
# simulation
# ==============================================================================
class Simulation:
    """One SynPV run.  Arrays of every stage remain available as attributes
    (e.g. ``sim.pac_all`` [W], shape time x houses) after :meth:`run`."""

    def __init__(self, config: dict | None = None,
                 log: Callable[[str], None] | None = None,
                 progress: Callable[[int, int, str], None] | None = None):
        self.cfg = default_config() if config is None else update_config(default_config(), config)
        self._log = log or print
        self._progress = progress
        self._cancel = False
        self.files: list[str] = []

    # ------------------------------------------------------------ plumbing
    def log(self, msg: str = "") -> None:
        self._log(msg)

    def cancel(self) -> None:
        self._cancel = True

    def _check_cancel(self):
        if self._cancel:
            raise SimulationCancelled("simulation cancelled by user")

    def _out(self, name):
        return os.path.join(self.OUT_DIR, name)

    def _resolve_static(self):
        """Resolve 'auto' values that depend only on the coordinates."""
        cfg = self.cfg
        lat, lon = float(cfg["LATITUDE"]), float(cfg["LONGITUDE"])
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("LATITUDE must be in [-90, 90] and LONGITUDE in [-180, 180]")
        t0, t1 = pd.Timestamp(cfg["START_DATE"]), pd.Timestamp(cfg["END_DATE"])
        if t1 < t0:
            raise ValueError("END_DATE must not be before START_DATE")
        if t0 < pd.Timestamp(NASA_FIRST_DATE):
            raise ValueError(f"NASA POWER hourly data start on {NASA_FIRST_DATE}")
        if 60 % int(cfg["FREQ_MIN"]):
            raise ValueError("FREQ_MIN must divide 60 (e.g. 1, 5, 10, 15, 30, 60)")
        if cfg["SITE_NAME"] in (None, "", "auto"):
            cfg["SITE_NAME"] = f"{lat:.4f}, {lon:.4f}"
        if cfg["TIMEZONE"] in (None, "", "auto"):
            cfg["TIMEZONE"] = timezone_for(lat, lon)
        if cfg["CLOUD_PREVAILING_FROM_DEG"] in (None, "", "auto"):
            cfg["CLOUD_PREVAILING_FROM_DEG"] = prevailing_wind_from(lat)
        if cfg["EQUATOR_FACING"] in (None, "", "auto"):
            cfg["EQUATOR_FACING"] = bool(lat < 0)
        self.SOUTHERN = lat < 0
        self.AZ_ROT = 180.0 if (cfg["EQUATOR_FACING"] and lat < 0) else 0.0

    # ------------------------------------------------------------ run
    def run(self) -> SimulationResult:
        T0 = _time.time()
        self._resolve_static()
        cfg = self.cfg
        # fleet streams depend on SEED only (same houses in every simulated year);
        # weather-dependent streams depend on (SEED, start date) (new clouds each year)
        SS = np.random.SeedSequence(cfg["SEED"])
        _ss = SS.spawn(7)
        self.RNG_FLEET, self.RNG_MISC = np.random.default_rng(_ss[0]), np.random.default_rng(_ss[6])
        _period_key = int(str(cfg["START_DATE"]).replace("-", ""))
        (self.RNG_CLOUD, self.RNG_SOIL, self.RNG_METER, self.RNG_WX) = [
            np.random.default_rng(s) for s in np.random.SeedSequence([cfg["SEED"], _period_key]).spawn(4)]

        if cfg["OUTPUT_DIR"] in (None, "", "auto"):
            _a, _b = cfg["START_DATE"], cfg["END_DATE"]
            self.OUT_DIR = (f"sim_output_{_a[:4]}" if (_a[:4] == _b[:4] and _a[5:] == "01-01" and _b[5:] == "12-31")
                            else f"sim_output_{_a}_{_b}")
        else:
            self.OUT_DIR = cfg["OUTPUT_DIR"]
        os.makedirs(self.OUT_DIR, exist_ok=True)

        self.log("=" * 78)
        self.log(f"SynPV {__version__} - SYNTHETIC RESIDENTIAL PV FLEET GENERATOR")
        self.log("=" * 78)
        self.log(f"Site      : {cfg['SITE_NAME']}  ({cfg['LATITUDE']}, {cfg['LONGITUDE']})  tz {cfg['TIMEZONE']}")
        self.log(f"Period    : {cfg['START_DATE']} .. {cfg['END_DATE']} @ {cfg['FREQ_MIN']} min "
                 f"(interval averages, {cfg['CLOUD_SUBSTEPS']} sub-samples)")
        self.log(f"Houses    : {cfg['N_HOUSES']}  (PV ownership {cfg['PV_OWNERSHIP']:.0%})")
        self.log(f"pvlib     : {pvlib.__version__}   pandas {pd.__version__}   numpy {np.__version__}")
        self.log(f"Output    : {self.OUT_DIR}")
        self.log()

        steps = [self._step_cec, self._step_fetch_weather, self._step_geometry, self._step_clearsky_index,
                 self._step_fleet, self._step_clouds, self._step_environment, self._step_poa,
                 self._step_panel_snow, self._step_electrical, self._step_metering,
                 self._step_validation, self._step_figures, self._step_export]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*(Mean of empty|invalid value|divide).*")
            for i, step in enumerate(steps):
                self._check_cancel()
                if self._progress:
                    self._progress(i, len(steps), STEPS[i])
                step()
        if self._progress:
            self._progress(len(steps), len(steps), "Done")
        self.runtime_s = _time.time() - T0
        self.metrics["runtime_s"] = round(self.runtime_s, 1)
        self.log(f"Run time           : {self.runtime_s / 60:.1f} min")
        with open(self._out("summary.json"), "w", encoding="utf-8") as fh:
            json.dump({"metrics": self.metrics, "checks": self.checks}, fh, indent=2, default=float)
        return SimulationResult(output_dir=os.path.abspath(self.OUT_DIR), config=dict(self.cfg),
                                metrics=self.metrics, checks=self.checks, files=list(self.files),
                                fleet=self.meta, runtime_s=self.runtime_s)

    # ==========================================================================
    # STEP 1 - CEC MODULE & INVERTER DATABASES
    # ==========================================================================
    def _step_cec(self):
        self.CEC_MODULES = load_cec_modules(self.log)
        self.CEC_INVERTERS = load_cec_inverters(self.log)
        self.log()

    # ==========================================================================
    # STEP 2 - METEOROLOGY (NASA POWER hourly, satellite/reanalysis-derived)
    # ==========================================================================
    def _step_fetch_weather(self):
        cfg = self.cfg
        self.log("[STEP 2] Downloading NASA POWER hourly meteorology ...")
        self.NASA_OK = False
        elev = None
        try:
            wx, elev, _ = load_or_fetch(cfg["LATITUDE"], cfg["LONGITUDE"], cfg["START_DATE"],
                                        cfg["END_DATE"], cfg["NASA_CACHE_DIR"], log=self.log)
            self.NASA_OK = True
            self.log(f"          OK - {len(wx)} hourly records (UTC)")
        except Exception as exc:
            self.log(f"          FAILED ({exc.__class__.__name__}: {exc})")
            if not cfg["ALLOW_SYNTHETIC_WEATHER"]:
                raise WeatherError(f"NASA POWER data could not be obtained: {exc}. Check the internet "
                                   "connection, or set ALLOW_SYNTHETIC_WEATHER=True for an offline "
                                   "test run.") from exc
            self.log("          -> SYNTHETIC climatological weather (results are illustrative only)")
            wx = synthetic_weather(cfg["START_DATE"], cfg["END_DATE"], cfg["LATITUDE"], self.RNG_WX,
                                   lon=cfg["LONGITUDE"])

        # precipitation units (NASA hourly occasionally arrives as mm/day)
        if self.NASA_OK:
            _ann = wx["PRECTOTCORR"].clip(lower=0).sum() / (len(wx) / 8766.0)
            if _ann > 5000:
                wx["PRECTOTCORR"] = wx["PRECTOTCORR"].clip(lower=0) / 24.0
                self.log(f"          precipitation /24 correction: {_ann:,.0f} -> "
                         f"{wx['PRECTOTCORR'].sum() / (len(wx) / 8766.0):,.0f} mm/yr")
        wx["PRECTOTCORR"] = wx["PRECTOTCORR"].clip(lower=0).fillna(0)
        self.wx = wx

        if cfg["ALTITUDE_M"] in (None, "", "auto"):
            cfg["ALTITUDE_M"] = float(elev) if elev is not None else 0.0
            self.log(f"          site elevation        : {cfg['ALTITUDE_M']:.0f} m (NASA POWER)")
        if cfg["DESIGN_T_MIN"] in (None, "", "auto"):
            try:
                if not self.NASA_OK:
                    raise WeatherError("no NASA POWER connection")
                tdes, y0, y1 = design_minimum_temperature(
                    cfg["LATITUDE"], cfg["LONGITUDE"], pd.Timestamp(cfg["END_DATE"]).year,
                    cfg["NASA_CACHE_DIR"], log=self.log)
                cfg["DESIGN_T_MIN"] = float(np.floor(tdes))
                self.log(f"          design minimum temp.  : {cfg['DESIGN_T_MIN']:.0f} C "
                         f"(mean of yearly minima, NASA POWER daily {y0}-{y1})")
            except Exception as exc:
                tmin = float(np.nanmin(wx["T2M"].to_numpy(dtype=float)))
                cfg["DESIGN_T_MIN"] = float(np.floor(min(tmin - 5.0, 0.0)))
                self.log(f"          design minimum temp.  : {cfg['DESIGN_T_MIN']:.0f} C "
                         f"(daily climatology unavailable: {exc}; min(period minimum - 5 C, 0 C))")

    # ==========================================================================
    # STEP 3 - TIME BASE, SOLAR GEOMETRY, CLEAR SKY
    # ==========================================================================
    def _step_geometry(self):
        cfg = self.cfg
        DT_MIN = self.DT_MIN = cfg["FREQ_MIN"]
        self.DT_H = DT_MIN / 60.0
        self.STEPS_PER_HOUR = int(round(60 / DT_MIN))
        times = pd.date_range(f"{cfg['START_DATE']} 00:00", f"{cfg['END_DATE']} 23:59",
                              freq=f"{DT_MIN}min", tz="UTC")           # interval starts
        tmid = times + pd.Timedelta(minutes=DT_MIN / 2)                 # physics evaluated here
        self.times, self.tmid, self.N_T = times, tmid, len(times)

        site = pvlib.location.Location(cfg["LATITUDE"], cfg["LONGITUDE"], tz="UTC",
                                       altitude=cfg["ALTITUDE_M"], name=cfg["SITE_NAME"])
        self.log(f"[STEP 3] {self.N_T:,} steps; solar position (NREL SPA) at interval mid-points ...")
        solpos = site.get_solarposition(tmid)
        zenith = solpos["apparent_zenith"].to_numpy()
        self.zenith = zenith
        self.azimuth_sun = solpos["azimuth"].to_numpy()
        self.elevation = solpos["apparent_elevation"].to_numpy()
        IS_DAY = self.IS_DAY = zenith < 90.0
        self.cosz = np.clip(np.cos(np.radians(zenith)), 0, None)

        cs = site.get_clearsky(tmid, model="ineichen", solar_position=solpos)
        self.ghi_cs = np.where(IS_DAY, cs["ghi"].to_numpy(), 0.0)
        self.dni_cs = np.where(IS_DAY, cs["dni"].to_numpy(), 0.0)
        self.dhi_cs = np.where(IS_DAY, cs["dhi"].to_numpy(), 0.0)
        self.dni_extra = pvlib.irradiance.get_extra_radiation(tmid).to_numpy()
        self.airmass = np.asarray(pvlib.atmosphere.get_relative_airmass(np.where(IS_DAY, zenith, 89.9)))

    # ==========================================================================
    # STEP 4 - CLEAR-SKY INDEX, TIME-STAMP CONVENTION, INTERPOLATION
    # ==========================================================================
    def _step_clearsky_index(self):
        cfg, wx, tmid = self.cfg, self.wx, self.tmid
        self.log("[STEP 4] Clear-sky index and interpolation to interval mid-points ...")
        _cs5 = pd.Series(self.ghi_cs, index=tmid)
        _lab_start = tmid.floor("h")
        CS_HOURLY = {"start": _cs5.groupby(_lab_start).mean(),
                     "end": _cs5.groupby(_lab_start + pd.Timedelta(hours=1)).mean()}

        def _kt_from_allsky(conv):
            cs_h = CS_HOURLY[conv].reindex(wx.index)
            return (wx["ALLSKY_SFC_SW_DWN"] / cs_h.where(cs_h > 30.0)), cs_h

        if self.NASA_OK:
            if cfg["NASA_HOUR_CONVENTION"] == "auto":
                # correct alignment => kt is flat through clear days (no am/pm tilt)
                score = {}
                for conv in ("start", "end"):
                    kt_c, cs_h = _kt_from_allsky(conv)
                    ok = (cs_h > 200)
                    day_med = kt_c[ok].groupby(kt_c[ok].index.date).median()
                    clear_days = set(day_med[day_med > 0.80].index)
                    sel = ok & pd.Series(wx.index.date, index=wx.index).isin(list(clear_days))
                    score[conv] = kt_c[sel].groupby(kt_c[sel].index.date).std().mean()
                if any(np.isnan(v) for v in score.values()):
                    score = {"start": 0.0, "end": 1.0}          # not enough clear days -> default
                HOUR_CONV = min(score, key=score.get)
                self.log(f"          hour-stamp convention detected: '{HOUR_CONV}' "
                         f"(clear-day kt std: start={score['start']:.3f}, end={score['end']:.3f})")
            else:
                HOUR_CONV = cfg["NASA_HOUR_CONVENTION"]
            wx["KT_CS"], _ = _kt_from_allsky(HOUR_CONV)
        else:
            HOUR_CONV = "start"
        self.HOUR_CONV = HOUR_CONV

        _center_shift = pd.Timedelta(minutes=30) if HOUR_CONV == "start" else pd.Timedelta(minutes=-30)
        _label_of_step = _lab_start if HOUR_CONV == "start" else _lab_start + pd.Timedelta(hours=1)
        STEPS_PER_HOUR = self.STEPS_PER_HOUR

        def to_steps_interp(series_hourly):
            """Interval-average hourly series -> values at mid-points (linear
            interpolation between hour centres)."""
            s = series_hourly.copy()
            s.index = s.index + _center_shift
            j = s.reindex(s.index.union(tmid)).interpolate(method="time").reindex(tmid)
            return j.ffill().bfill().to_numpy()

        def to_steps_conserve(series_hourly):
            """Hourly totals -> per-step totals, mass conserving."""
            return (series_hourly.reindex(_label_of_step).fillna(0).to_numpy() / STEPS_PER_HOUR)

        self.temp_air = to_steps_interp(wx["T2M"])
        self.wind = np.clip(to_steps_interp(wx["WS10M"]), 0.1, None)
        self.rh = to_steps_interp(wx["RH2M"])
        self.precip = to_steps_conserve(wx["PRECTOTCORR"])           # mm per step

        kt_h = wx["KT_CS"].copy()
        kt_h[CS_HOURLY[HOUR_CONV].reindex(wx.index).fillna(0).to_numpy() < 30] = np.nan
        kt_h = kt_h.clip(0.03, 1.15)
        kt_h = kt_h.interpolate(method="time", limit=4, limit_direction="both").fillna(0.7)
        kt_reg = np.clip(to_steps_interp(kt_h), 0.03, 1.15)
        kt_reg[~self.IS_DAY] = 0.0
        self.kt_reg = kt_reg
        self.ghi_reg = self.ghi_cs * kt_reg

        N_T, DT_H = self.N_T, self.DT_H
        self.log(f"          annual precipitation  : {self.precip.sum() / (N_T * DT_H / 8766):,.0f} mm/yr")
        self.log(f"          regional GHI          : {self.ghi_reg.sum() * DT_H / 1000:,.0f} kWh/m2 over period")
        self.log(f"          mean daytime kt       : {kt_reg[self.ghi_cs > 50].mean():.3f}")

    # ==========================================================================
    # STEP 5 - PV FLEET
    # ==========================================================================
    def _step_fleet(self):
        cfg = self.cfg
        self.log("[STEP 5] Generating residential PV fleet ...")
        fleet = build_pv_fleet(cfg, self.RNG_FLEET, self.CEC_MODULES, self.CEC_INVERTERS, az_rot=self.AZ_ROT)
        # mounting / thermal attributes (separate stream -> fleet above unchanged)
        _hp = fleet["has_pv"].to_numpy()
        fleet["mount_standoff_in"] = np.where(_hp, self.RNG_MISC.choice(cfg["MOUNT_STANDOFF_IN"], len(fleet),
                                                                        p=cfg["MOUNT_STANDOFF_WEIGHTS"]), np.nan)
        fleet["array_height_stories"] = np.where(_hp, 1 + (self.RNG_MISC.random(len(fleet)) < cfg["TWO_STORY_FRACTION"]), 0)
        fleet["module_noct_c"] = [float(m["T_NOCT"]) if isinstance(m, dict) else np.nan
                                  for m in fleet.get("_mod", pd.Series([None] * len(fleet)))]
        fleet["module_gamma_pmp"] = [float(m["gamma_r"]) / 100 if isinstance(m, dict) else np.nan
                                     for m in fleet.get("_mod", pd.Series([None] * len(fleet)))]
        times = self.times
        _mid = times[len(times) // 2]
        YEAR_OFFSET = (_mid - pd.Timestamp(f"{cfg['FLEET_REFERENCE_YEAR']}-07-01", tz="UTC")).days / 365.25
        fleet["age_at_simulation_years"] = np.where(
            _hp, np.maximum(fleet.get("age_years", pd.Series(np.nan, index=fleet.index)).astype(float) + YEAR_OFFSET, 0.0), np.nan)
        self.fleet = fleet
        self.N_H = cfg["N_HOUSES"]
        self.solar_idx = np.flatnonzero(fleet["has_pv"].to_numpy())
        self.N_SOLAR = len(self.solar_idx)
        if self.N_SOLAR == 0:
            raise ValueError("no house received a PV system - increase N_HOUSES or PV_OWNERSHIP")
        pvf = fleet.loc[fleet.has_pv]
        self.log(f"          houses with PV : {self.N_SOLAR} / {self.N_H}")
        self.log(f"          DC capacity    : {pvf.capacity_kw.min():.1f} .. {pvf.capacity_kw.max():.1f} kW "
                 f"(fleet {pvf.capacity_kw.sum():.1f} kW; module {pvf.module_pmp_w.min():.0f}-"
                 f"{pvf.module_pmp_w.max():.0f} W)")
        self.log(f"          DC/AC ratio    : {pvf.dc_ac_ratio.min():.2f} .. {pvf.dc_ac_ratio.max():.2f}")
        self.log(f"          strings        : Ns {pvf.Ns.min()}-{pvf.Ns.max()}, Np {pvf.Np.min()}-{pvf.Np.max()}")
        self.log("          module mix     : " + ", ".join(f"{m} x{c}" for m, c in pvf.module_manufacturer.value_counts().items()))
        self.log("          inverter mix   : " + ", ".join(f"{m} x{c}" for m, c in pvf.inverter_manufacturer.value_counts().items()))

    # ==========================================================================
    # STEP 6 - ADVECTING CLOUD FIELD -> per-house GHI / DNI / DHI
    # ==========================================================================
    # Cloud-cover fraction f(t) and in-cloud transmittance are set from the
    # hourly NASA clear-sky index; WHERE the clouds are is given by a Gaussian
    # random field built from Fourier modes that (i) advect with a day-varying
    # steering wind (Taylor's frozen-turbulence hypothesis) and (ii) evolve
    # (finite eddy lifetimes).  A house is under cloud where the field exceeds
    # the quantile threshold for f(t), so the expected cover equals f(t).
    def _steering_wind(self, rng):
        cfg, tmid = self.cfg, self.tmid
        days = pd.date_range(cfg["START_DATE"], cfg["END_DATE"], freq="D", tz="UTC")
        ws_d = pd.Series(self.wind, index=tmid).groupby(tmid.floor("D")).mean().reindex(days).ffill().bfill()
        nd = len(days)
        spd = np.clip(np.exp(np.log(2.0 * ws_d.to_numpy() + 4.0) + rng.normal(0, 0.30, nd)),
                      *cfg["CLOUD_SPEED_RANGE"])
        dev = np.zeros(nd)
        for d in range(nd):
            dev[d] = (0.6 * dev[d - 1] if d else 0) + rng.normal(0, cfg["CLOUD_DIR_SIGMA_DEG"] * 0.8)
        toward = np.radians(cfg["CLOUD_PREVAILING_FROM_DEG"] + 180 + dev)     # compass
        u = spd * np.sin(toward) * 0.06       # km/min east
        v = spd * np.cos(toward) * 0.06       # km/min north
        noon = days + pd.Timedelta(hours=12)
        tnum = (noon - noon[0]).total_seconds().to_numpy() / 60
        tm = (tmid - noon[0]).total_seconds().to_numpy() / 60
        return np.interp(tm, tnum, u), np.interp(tm, tnum, v), spd

    @staticmethod
    def _spectral_modes(cfg, rng):
        M = cfg["CLOUD_N_MODES"]
        lam = np.exp(rng.uniform(*np.log(cfg["CLOUD_LAMBDA_KM"]), M))
        ang = rng.uniform(0, 2 * np.pi, M)
        k = 2 * np.pi / lam
        kx, ky = k * np.cos(ang), k * np.sin(ang)
        amp = lam ** 0.5
        amp /= np.sqrt((amp ** 2).sum() / 2)                      # unit variance
        tau = cfg["CLOUD_LIFETIME_MIN_PER_KM"] * lam ** 0.7        # minutes
        omega = rng.choice([-1, 1], M) * 2 * np.pi / tau
        phi = rng.uniform(0, 2 * np.pi, M)
        return kx, ky, amp, omega, phi

    @staticmethod
    def _eval_field(modes, xh, yh, Dx, Dy, tmin):
        kx, ky, amp, omega, phi = modes
        A = np.outer(xh, kx) + np.outer(yh, ky)                   # H x M
        B = np.outer(Dx, kx) + np.outer(Dy, ky) - np.outer(tmin, omega) - phi   # T x M
        return (np.cos(B) @ (amp * np.cos(A)).T + np.sin(B) @ (amp * np.sin(A)).T).astype(np.float32)

    def _step_clouds(self):
        cfg, fleet = self.cfg, self.fleet
        N_T, N_H, DT_MIN = self.N_T, self.N_H, self.DT_MIN
        kt_reg, ghi_cs, dni_cs, IS_DAY = self.kt_reg, self.ghi_cs, self.dni_cs, self.IS_DAY
        self.log("[STEP 6] Advecting cloud field (all houses, sub-sampled intervals) ...")
        u_km, v_km, cloud_speed_daily = self._steering_wind(self.RNG_CLOUD)
        modes_cov = self._spectral_modes(cfg, self.RNG_CLOUD)
        modes_thk = self._spectral_modes(cfg, self.RNG_CLOUD)
        xh = fleet["x_km"].to_numpy()
        yh = fleet["y_km"].to_numpy()
        SUB = cfg["CLOUD_SUBSTEPS"]
        dt_sub = DT_MIN / SUB

        # displacement of the frozen field at every sub-sample
        u_sub = np.repeat(u_km, SUB)
        v_sub = np.repeat(v_km, SUB)
        Dx_all = np.cumsum(u_sub) * dt_sub
        Dy_all = np.cumsum(v_sub) * dt_sub
        t_sub_min = np.arange(N_T * SUB) * dt_sub

        k_clr = np.maximum(1.0, kt_reg)                         # clear part
        k_c_nom = cfg["CLOUD_TRANSMITTANCE"]
        f_cov = np.clip((k_clr - kt_reg) / (k_clr - k_c_nom), 0.0, 1.0)
        k_c = np.where(kt_reg < k_c_nom, kt_reg, k_c_nom)       # overcast thinner/thicker
        f_cl = np.clip(f_cov, 1e-3, 1 - 1e-3)
        thr = ndtri(1 - f_cl)                                   # field quantile threshold

        kt_house = np.zeros((N_T, N_H), dtype=np.float32)
        beam_frac = np.zeros((N_T, N_H), dtype=np.float32)      # fraction of time sun unobscured
        cloud_mask = np.zeros((N_T, N_H), dtype=np.float32)
        w_edge = cfg["CLOUD_EDGE_WIDTH"]
        sig_t = cfg["CLOUD_THICKNESS_SIGMA"]
        CHUNK = 4096
        for s in range(0, N_T, CHUNK):
            self._check_cancel()
            e = min(s + CHUNK, N_T)
            ss, se_ = s * SUB, e * SUB
            z = self._eval_field(modes_cov, xh, yh, Dx_all[ss:se_], Dy_all[ss:se_], t_sub_min[ss:se_])
            z2 = self._eval_field(modes_thk, xh, yh, Dx_all[ss:se_], Dy_all[ss:se_], t_sub_min[ss:se_])
            rep = lambda a: np.repeat(a[s:e], SUB)[:, None]
            m = expit((z - rep(thr)) / w_edge)
            m = np.where(rep(f_cov) <= 1e-3, 0.0, np.where(rep(f_cov) >= 1 - 1e-3, 1.0, m))
            k_in = rep(k_c) * np.exp(sig_t * z2 - sig_t ** 2 / 2)
            # enhancement: diffuse boost from nearby cloud sides while the sun is
            # clear; strongest for broken fields (f ~ 0.5)
            fr = rep(f_cov)
            ce = cfg["CLOUD_ENHANCEMENT"] * 4 * fr * (1 - fr) * (1 - m) * expit(2.0 * z2)
            kt_s = rep(k_clr) * (1 - m) + k_in * m + ce
            sh = (e - s, SUB, N_H)
            kt_house[s:e] = kt_s.reshape(sh).mean(axis=1)
            beam_frac[s:e] = (1 - m).reshape(sh).mean(axis=1)
            cloud_mask[s:e] = m.reshape(sh).mean(axis=1)

        # daily energy closure with the NASA source (keeps sub-daily local structure)
        _day = self._day = self.tmid.floor("D")
        tgt = pd.Series(self.ghi_reg, index=self.tmid).groupby(_day).sum()
        sim = pd.Series((ghi_cs[:, None] * kt_house).mean(axis=1), index=self.tmid).groupby(_day).sum()
        corr_d = (tgt / sim.where(sim > 1)).fillna(1.0).clip(0.85, 1.18)
        corr = corr_d.reindex(_day).to_numpy()[:, None]

        ghi_house = (ghi_cs[:, None] * kt_house * corr).astype(np.float32)
        dni_house = (dni_cs[:, None] * k_clr[:, None] * beam_frac * corr).astype(np.float32)
        dhi_house = np.maximum(ghi_house - dni_house * self.cosz[:, None], 0).astype(np.float32)
        for a in (ghi_house, dni_house, dhi_house, cloud_mask):
            a[~IS_DAY] = 0.0
        del kt_house

        self.u_km, self.v_km, self.cloud_speed_daily = u_km, v_km, cloud_speed_daily
        self.xh, self.yh, self.f_cov = xh, yh, f_cov
        self.ghi_house, self.dni_house, self.dhi_house, self.cloud_mask = ghi_house, dni_house, dhi_house, cloud_mask
        self.d01 = np.hypot(xh[:, None] - xh[None, :], yh[:, None] - yh[None, :])
        self.log(f"          cloud speed (daily)  : {np.median(cloud_speed_daily):.1f} m/s median "
                 f"({cloud_speed_daily.min():.1f}-{cloud_speed_daily.max():.1f}), prevailing from "
                 f"{cfg['CLOUD_PREVAILING_FROM_DEG']:.0f} deg")
        self.log(f"          per-house GHI peak   : {ghi_house.max():.0f} W/m2 (cloud enhancement allowed)")
        self.log(f"          GHI energy closure   : sim/NASA = "
                 f"{ghi_house.mean(axis=1).sum() / self.ghi_reg.sum():.4f}")

    # ==========================================================================
    # STEP 7 - SNOW (ground pack), SOILING, HORIZON SHADING
    # ==========================================================================
    def _step_environment(self):
        cfg, fleet = self.cfg, self.fleet
        N_T, N_H, DT_H, tmid, _day = self.N_T, self.N_H, self.DT_H, self.tmid, self._day
        temp_air, rh, precip = self.temp_air, self.rh, self.precip
        self.log("[STEP 7] Snowpack, soiling and horizon shading ...")
        if cfg["SNOW_PARTITION"] == "wetbulb":
            t_wet = wet_bulb_stull(temp_air, rh)
            snow_frac = np.clip((cfg["WETBULB_RAIN_C"] - t_wet) /
                                (cfg["WETBULB_RAIN_C"] - cfg["WETBULB_SNOW_C"]), 0, 1)
        else:
            snow_frac = np.clip((cfg["RAIN_ALL_ABOVE_C"] - temp_air) /
                                (cfg["RAIN_ALL_ABOVE_C"] - cfg["SNOW_ALL_BELOW_C"]), 0, 1)
        snow_mm = precip * snow_frac                         # liquid-equivalent snowfall per step
        rain_mm = precip - snow_mm
        snowfall_cm = snow_mm * cfg["SNOW_LIQUID_RATIO"] / 10.0

        swe = 0.0
        age = 99.0
        ground_depth_cm = np.zeros(N_T)
        ddf = cfg["DEGREE_DAY_FACTOR"] * DT_H / 24.0
        for t in range(N_T):
            swe += snow_mm[t]
            if snow_mm[t] > 0.05:
                age = 0.0
            else:
                age += DT_H / 24
            melt = ddf * max(temp_air[t], 0.0) + 0.0125 * rain_mm[t] * max(temp_air[t], 0.0)
            swe = max(swe - melt, 0.0)
            rho = 0.10 + 0.20 * (1 - math.exp(-age / 10.0))  # fresh -> settled snow
            ground_depth_cm[t] = swe / rho / 10.0 if swe > 0 else 0.0

        # soiling (daily, Kimber-type)
        daily_rain = pd.Series(rain_mm, index=tmid).groupby(_day).sum()
        daily_snow_depth = pd.Series(ground_depth_cm, index=tmid).groupby(_day).max()
        rate = fleet.get("soiling_rate", pd.Series(0.001, index=fleet.index)).fillna(0.001).to_numpy()
        soil_d = np.zeros((len(daily_rain), N_H))
        acc = np.zeros(N_H)
        for d, (r_mm, sd) in enumerate(zip(daily_rain.to_numpy(), daily_snow_depth.to_numpy())):
            if r_mm >= cfg["SOILING_CLEAN_MM"] or sd > 2.0:
                acc[:] = 0.0
            else:
                acc = np.minimum(acc + rate, cfg["SOILING_MAX"])
            soil_d[d] = acc
        soiling_loss = pd.DataFrame(soil_d, index=daily_rain.index).reindex(_day).to_numpy().astype(np.float32)

        # horizon shading of the BEAM component
        az_bin = (np.floor(self.azimuth_sun / 5.0).astype(int)) % 72
        shade_beam = np.ones((N_T, N_H), dtype=np.float32)
        for h in self.solar_idx:
            hz_el, hz_tr = fleet.at[h, "_hz_el"], fleet.at[h, "_hz_tr"]
            blocked = self.elevation < hz_el[az_bin]
            shade_beam[:, h] = np.where(blocked, hz_tr[az_bin], 1.0)
        self.log(f"          ground snow days (>1 cm): "
                 f"{(daily_snow_depth > 1).sum()}  | max depth {ground_depth_cm.max():.1f} cm")
        self.snowfall_cm, self.ground_depth_cm = snowfall_cm, ground_depth_cm
        self.soiling_loss, self.shade_beam = soiling_loss, shade_beam

    # ==========================================================================
    # STEP 8 - POA, IAM, EFFECTIVE IRRADIANCE (per house)
    # ==========================================================================
    def _step_poa(self):
        cfg, fleet = self.cfg, self.fleet
        N_T, N_H, IS_DAY = self.N_T, self.N_H, self.IS_DAY
        zenith, azimuth_sun = self.zenith, self.azimuth_sun
        self.log("[STEP 8] Perez transposition, IAM, effective irradiance ...")
        poa_all = np.zeros((N_T, N_H), dtype=np.float32)      # incident POA (for PR)
        poa_dir_all = np.zeros((N_T, N_H), dtype=np.float32)  # incident beam POA, unshaded
        eff_all = np.zeros((N_T, N_H), dtype=np.float32)      # effective irradiance to cells
        abs_all = np.zeros((N_T, N_H), dtype=np.float32)      # absorbed (for thermal)
        for h in self.solar_idx:
            self._check_cancel()
            tilt, az = fleet.at[h, "tilt"], fleet.at[h, "azimuth"]
            poa = pvlib.irradiance.get_total_irradiance(
                surface_tilt=tilt, surface_azimuth=az,
                solar_zenith=zenith, solar_azimuth=azimuth_sun,
                dni=self.dni_house[:, h].astype(float), ghi=self.ghi_house[:, h].astype(float),
                dhi=self.dhi_house[:, h].astype(float), dni_extra=self.dni_extra, airmass=self.airmass,
                albedo=cfg["ALBEDO"], model=cfg["TRANSPOSITION"])
            p_dir = np.nan_to_num(np.asarray(poa["poa_direct"], float))
            p_sky = np.nan_to_num(np.asarray(poa["poa_sky_diffuse"], float))
            p_gnd = np.nan_to_num(np.asarray(poa["poa_ground_diffuse"], float))
            p_dir[~IS_DAY] = p_sky[~IS_DAY] = p_gnd[~IS_DAY] = 0.0
            aoi = pvlib.irradiance.aoi(tilt, az, zenith, azimuth_sun)
            iam_b = np.clip(np.nan_to_num(_iam_beam(aoi, cfg["IAM_A_R"])), 0, 1)
            iam_s, iam_g = _iam_diffuse(tilt, cfg["IAM_A_R"])
            svf = 1.0 - fleet.at[h, "skyview_loss"]
            beam = p_dir * self.shade_beam[:, h]
            poa_all[:, h] = p_dir + p_sky + p_gnd
            poa_dir_all[:, h] = p_dir
            abs_all[:, h] = beam + p_sky * svf + p_gnd
            eff_all[:, h] = (beam * iam_b + p_sky * svf * iam_s + p_gnd * iam_g) * (1 - self.soiling_loss[:, h])
        self.poa_all, self.poa_dir_all, self.eff_all, self.abs_all = poa_all, poa_dir_all, eff_all, abs_all

    # ==========================================================================
    # STEP 9 - SNOW COVERAGE ON PANELS (NREL / Marion 2013, vectorised)
    # ==========================================================================
    def _step_panel_snow(self):
        cfg, fleet = self.cfg, self.fleet
        N_T, N_H, DT_H = self.N_T, self.N_H, self.DT_H
        self.log("[STEP 9] Panel snow coverage (Marion et al. 2013 / SAM) ...")
        tilt_v = self.tilt_v = fleet["tilt"].to_numpy(dtype=float)
        new_snow = (self.snowfall_cm / DT_H) >= cfg["MARION_THRESHOLD_CM_HR"]
        slide_base = cfg["MARION_SLIDE"] * np.sin(np.radians(tilt_v)) * DT_H
        snow_cov = np.zeros((N_T, N_H), dtype=np.float32)
        cov = np.zeros(N_H)
        temp_air, ground_depth_cm, poa_all = self.temp_air, self.ground_depth_cm, self.poa_all
        for t in range(N_T):
            if new_snow[t]:
                cov[:] = 1.0
            elif ground_depth_cm[t] < 1.0:
                cov[:] = 0.0
            else:
                can = temp_air[t] > poa_all[t] / cfg["MARION_M"]
                cov = np.maximum(cov - np.where(can, slide_base, 0.0), 0.0)
            snow_cov[t] = cov
        _portrait = fleet["portrait"].astype("boolean").fillna(True).to_numpy(dtype=bool)
        n_str = np.where(_portrait, 1, 3)
        snow_dc_loss = (np.ceil(snow_cov * n_str[None, :]) / n_str[None, :]).astype(np.float32)
        snow_dc_loss[:, ~fleet["has_pv"].to_numpy()] = 0.0
        snow_cov[:, ~fleet["has_pv"].to_numpy()] = 0.0
        self.snow_cov, self.snow_dc_loss = snow_cov, snow_dc_loss

    # ==========================================================================
    # STEP 10 - THERMAL, SINGLE-DIODE DC, MPPT WINDOW, INVERTER
    # ==========================================================================
    def _step_electrical(self):
        cfg, fleet = self.cfg, self.fleet
        N_T, N_H, DT_H, IS_DAY = self.N_T, self.N_H, self.DT_H, self.IS_DAY
        temp_air, wind = self.temp_air, self.wind
        CEC_MODULES, CEC_INVERTERS = self.CEC_MODULES, self.CEC_INVERTERS
        self.log(f"[STEP 10] {'SAM NOCT' if cfg['THERMAL_MODEL'] == 'noct_sam' else 'Faiman'} temperature "
                 "-> CEC single-diode -> MPPT -> Sandia inverter ...")
        tcell_all = np.zeros((N_T, N_H), dtype=np.float32)
        pdc_all = np.zeros((N_T, N_H), dtype=np.float32)
        vdc_all = np.zeros((N_T, N_H), dtype=np.float32)
        pac_all = np.zeros((N_T, N_H), dtype=np.float32)
        mppt_out_frac = np.zeros(N_H)
        snow_lost_wh = np.zeros(N_H)
        dc_pre_snow_wh = np.zeros(N_H)
        step_print = max(1, int(math.ceil(self.N_SOLAR / 5)))
        for k, h in enumerate(self.solar_idx):
            self._check_cancel()
            row = fleet.loc[h]
            mod, inv = row["_mod"], row["_inv"]
            ns, np_ = int(row["Ns"]), int(row["Np"])
            poa_in = self.abs_all[:, h].astype(float) * (1 - self.snow_cov[:, h])
            if cfg["THERMAL_MODEL"] == "noct_sam":
                lit = poa_in > 1.0
                pg = np.where(lit, poa_in, 1.0)
                ge = np.where(lit, np.minimum(self.eff_all[:, h], poa_in), 1.0)
                tc = np.asarray(pvlib.temperature.noct_sam(
                    pg, temp_air, wind, noct=row["module_noct_c"], module_efficiency=row["module_efficiency"],
                    effective_irradiance=ge, array_height=int(row["array_height_stories"]),
                    mount_standoff=float(row["mount_standoff_in"])), float)
                tc = np.where(lit, tc, temp_air)
            else:
                tc = np.asarray(pvlib.temperature.faiman(poa_in, temp_air, wind,
                                                         u0=cfg["FAIMAN_U0"], u1=cfg["FAIMAN_U1"]), float)
            tcell_all[:, h] = tc
            G = self.eff_all[:, h].astype(float)
            on = IS_DAY & (G > 1.0)
            p_mod = np.zeros(N_T)
            v_mod = np.zeros(N_T)
            se = bool(row["optimizer_architecture"])

            if CEC_MODULES is not None:
                IL, I0, Rs, Rsh, nVth = pvlib.pvsystem.calcparams_cec(
                    effective_irradiance=G[on], temp_cell=tc[on], alpha_sc=mod["alpha_sc"],
                    a_ref=mod["a_ref"], I_L_ref=mod["I_L_ref"], I_o_ref=mod["I_o_ref"],
                    R_sh_ref=mod["R_sh_ref"], R_s=mod["R_s"], Adjust=mod["Adjust"])
                sd = pvlib.pvsystem.singlediode(IL, I0, Rs, Rsh, nVth)
                pm = np.nan_to_num(np.asarray(sd["p_mp"], float))
                vm = np.nan_to_num(np.asarray(sd["v_mp"], float))
                if (CEC_INVERTERS is not None) and not se:
                    # string inverter: operating voltage limited to the MPPT window
                    lo, hi = inv["Mppt_low"] / ns, inv["Mppt_high"] / ns
                    out = (vm < lo) | (vm > hi)
                    if out.any():
                        v_op = np.clip(vm[out], lo, hi)
                        i_op = np.asarray(pvlib.pvsystem.i_from_v(
                            v_op, np.broadcast_to(IL, vm.shape)[out], np.broadcast_to(I0, vm.shape)[out],
                            np.broadcast_to(Rs, vm.shape)[out], np.broadcast_to(Rsh, vm.shape)[out],
                            np.broadcast_to(nVth, vm.shape)[out]), float)
                        pm[out] = np.maximum(np.nan_to_num(v_op * i_op), 0.0)
                        vm[out] = v_op
                    mppt_out_frac[h] = (out & (pm > 5)).mean()
                p_mod[on], v_mod[on] = pm, vm
            else:
                gam = cfg["GAMMA_PMAX"]
                p_mod[on] = mod["STC"] * G[on] / 1000 * (1 + gam * (tc[on] - 25))
                v_mod[on] = mod["V_mp_ref"] * (1 + mod["beta_oc"] / mod["V_oc_ref"] * (tc[on] - 25))

            mism = cfg["MISMATCH_LOSS_OPTIMIZER"] if se else cfg["MISMATCH_LOSS"]
            derate = ((1 - cfg["DC_WIRING_LOSS"]) * (1 - cfg["CONNECTION_LOSS"]) * (1 - mism)
                      * (1 - cfg["LID_LOSS"]) * (1 + row["nameplate_tolerance"])
                      * (1 - row["degradation_rate"] * row["age_at_simulation_years"])
                      * (cfg["OPTIMIZER_EFF"] if se else 1.0))
            p_dc_pre = p_mod * ns * np_ * derate
            p_dc_pre[~on] = 0.0
            p_dc = p_dc_pre * (1 - self.snow_dc_loss[:, h])
            snow_lost_wh[h] = (p_dc_pre * self.snow_dc_loss[:, h]).sum() * DT_H
            dc_pre_snow_wh[h] = p_dc_pre.sum() * DT_H
            v_dc = np.full(N_T, inv["Vdco"]) if se else v_mod * ns * (1 - cfg["DC_WIRING_LOSS"] / 2)
            p_dc[~on] = 0.0
            v_dc[p_dc <= 0] = 0.0
            assert np.all(np.isfinite(p_dc)) and p_dc.min() >= 0
            pdc_all[:, h] = p_dc
            vdc_all[:, h] = v_dc

            if CEC_INVERTERS is not None:
                p_ac = np.asarray(pvlib.inverter.sandia(v_dc=np.where(p_dc > 0, v_dc, inv["Vdco"]),
                                                        p_dc=p_dc, inverter=pd.Series(inv)), float)
            else:
                p_ac = p_dc * inverter_efficiency_fallback(p_dc / inv["Paco"])
            p_ac = np.clip(np.nan_to_num(p_ac), 0.0, inv["Paco"])       # night tare (-Pnt) -> 0 export
            p_ac[p_dc <= 0] = 0.0
            pac_all[:, h] = p_ac
            if (k + 1) % step_print == 0 or k == self.N_SOLAR - 1:
                self.log(f"          modelled {k + 1:3d}/{self.N_SOLAR}")

        assert pac_all[~IS_DAY].max() == 0.0, "night-time AC must be zero"
        assert np.all(pac_all.max(axis=0) <= fleet["inverter_paco_w"].fillna(0).to_numpy(dtype=float) + 1e-3)
        self.tcell_all, self.pdc_all, self.vdc_all, self.pac_all = tcell_all, pdc_all, vdc_all, pac_all
        self.mppt_out_frac, self.snow_lost_wh, self.dc_pre_snow_wh = mppt_out_frac, snow_lost_wh, dc_pre_snow_wh

    # ==========================================================================
    # STEP 11 - METERING REALISM LAYER (measured output)
    # ==========================================================================
    def _step_metering(self):
        cfg, fleet = self.cfg, self.fleet
        N_T, N_H, DT_H, STEPS_PER_HOUR = self.N_T, self.N_H, self.DT_H, self.STEPS_PER_HOUR
        solar_idx = self.solar_idx
        self.log("[STEP 11] Metering realism layer ...")
        rng = self.RNG_METER
        years = N_T * DT_H / 8766
        pac_meas = self.pac_all.copy()
        flags = np.zeros((N_T, N_H), dtype=np.int8)      # 0 ok, 1 gap, 2 outage, 3 curtailed, 4 clock-shifted
        export_limit_w = np.full(N_H, np.inf)
        clock_offset_steps = np.zeros(N_H, dtype=int)
        for h in solar_idx:
            paco = fleet.at[h, "inverter_paco_w"]
            for _ in range(rng.poisson(cfg["OUTAGE_EVENTS_PER_YEAR"] * years)):
                s = rng.integers(0, N_T)
                L = rng.integers(*cfg["OUTAGE_DAYS_RANGE"]) * 24 * STEPS_PER_HOUR
                pac_meas[s:s + L, h] = 0.0
                flags[s:s + L, h] = 2
            if rng.random() < cfg["EXPORT_LIMIT_FRACTION_OF_HOUSES"]:
                export_limit_w[h] = cfg["EXPORT_LIMIT_REL"] * paco
                cut = pac_meas[:, h] > export_limit_w[h]
                pac_meas[cut, h] = export_limit_w[h]
                flags[cut & (flags[:, h] == 0), h] = 3
            noise = rng.normal(0, 1, N_T) * (cfg["METER_NOISE_REL"] * pac_meas[:, h] + cfg["METER_NOISE_ABS_W"])
            pac_meas[:, h] = np.where(pac_meas[:, h] > 0, np.clip(pac_meas[:, h] + noise, 0, paco), 0.0)
            pac_meas[:, h] = np.round(pac_meas[:, h])
            if rng.random() < cfg["CLOCK_OFFSET_FRACTION"]:
                off = int(rng.choice([-2, -1, 1, 2]))
                clock_offset_steps[h] = off
                pac_meas[:, h] = np.roll(pac_meas[:, h], off)
                flags[:, h] = np.where(flags[:, h] == 0, 4, flags[:, h])
            for _ in range(rng.poisson(cfg["GAP_EVENTS_PER_YEAR"] * years)):
                s = rng.integers(0, N_T)
                L = max(1, int(rng.lognormal(np.log(cfg["GAP_MEDIAN_HOURS"] * STEPS_PER_HOUR), 1.2)))
                L = min(L, 3 * 24 * STEPS_PER_HOUR)
                pac_meas[s:s + L, h] = np.nan
                flags[s:s + L, h] = 1
        pac_meas[:, ~fleet["has_pv"].to_numpy()] = 0.0
        self.log(f"          missing data : {np.isnan(pac_meas[:, solar_idx]).mean() * 100:.2f} %   "
                 f"outage steps: {(flags[:, solar_idx] == 2).mean() * 100:.2f} %   "
                 f"export-limited houses: {np.isfinite(export_limit_w).sum()}   "
                 f"clock-offset houses: {(clock_offset_steps != 0).sum()}")
        self.pac_meas, self.flags = pac_meas, flags
        self.export_limit_w, self.clock_offset_steps = export_limit_w, clock_offset_steps

    # ==========================================================================
    # STEP 12 - VALIDATION
    # ==========================================================================
    def _step_validation(self):
        cfg, fleet = self.cfg, self.fleet
        N_T, N_H, DT_H, STEPS_PER_HOUR = self.N_T, self.N_H, self.DT_H, self.STEPS_PER_HOUR
        solar_idx, N_SOLAR, IS_DAY = self.solar_idx, self.N_SOLAR, self.IS_DAY
        times, tmid, zenith, azimuth_sun = self.times, self.tmid, self.zenith, self.azimuth_sun
        pac_all, pdc_all, poa_all, poa_dir_all = self.pac_all, self.pdc_all, self.poa_all, self.poa_dir_all
        eff_all, abs_all, ghi_house, ghi_cs = self.eff_all, self.abs_all, self.ghi_house, self.ghi_cs
        u_km, v_km, xh, yh, f_cov = self.u_km, self.v_km, self.xh, self.yh, self.f_cov
        temp_air, wind = self.temp_air, self.wind

        hours = N_T * DT_H
        energy_kwh = pac_all.sum(axis=0) * DT_H / 1000
        cap_kw = fleet["capacity_kw"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            sy = np.where(cap_kw > 0, energy_kwh / cap_kw, np.nan)
            cf = np.where(cap_kw > 0, energy_kwh / (cap_kw * hours), np.nan)
            poa_kwh = poa_all.sum(axis=0) * DT_H / 1000
            pr = np.where(cap_kw > 0, energy_kwh / (poa_kwh * cap_kw), np.nan)
        inv_w = fleet["inverter_paco_w"].fillna(0).to_numpy(dtype=float)
        clip_frac = float((pdc_all[:, solar_idx] > inv_w[solar_idx]).mean())
        clip_loss = float((np.maximum(pdc_all[:, solar_idx] * 0.97 - inv_w[solar_idx], 0).sum())
                          / max(pdc_all[:, solar_idx].sum(), 1e-9))

        t_local = times.tz_convert(cfg["TIMEZONE"])
        fleet_kw = pac_all[:, solar_idx].sum(axis=1) / 1000
        df_fleet = pd.DataFrame({"p_kw": fleet_kw}, index=t_local)
        _mon = df_fleet["p_kw"].groupby([t_local.year, t_local.month])
        monthly_mwh = (_mon.sum() * DT_H / 1000)[_mon.size() > 10 * 24 * STEPS_PER_HOUR]
        monthly_mwh.index = pd.Index([f"{y}-{m:02d}" for y, m in monthly_mwh.index], name="month")

        # loss bookkeeping (energy-weighted, PV houses)
        poa_s = poa_all[:, solar_idx].sum()
        loss_soil = 1 - (eff_all[:, solar_idx].sum() /
                         (eff_all[:, solar_idx] / np.maximum(1 - self.soiling_loss[:, solar_idx], 1e-6)).sum())
        loss_snow = float(self.snow_lost_wh.sum() / max(self.dc_pre_snow_wh.sum(), 1e-9))
        loss_shade = 1 - abs_all[:, solar_idx].sum() / poa_s

        # spatial correlation of 5-min clear-sky-index increments vs distance
        sel = (zenith < 70)
        ksi = ghi_house[sel][:, solar_idx] / np.maximum(ghi_cs[sel][:, None], 1)
        dk = np.diff(ksi, axis=0)
        bins = np.array([0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.5])
        if N_SOLAR >= 2 and dk.shape[0] > 2:
            C = np.corrcoef(dk.T)
            D = self.d01[np.ix_(solar_idx, solar_idx)]
            iu = np.triu_indices(N_SOLAR, 1)
            dist_pairs, corr_pairs = D[iu], C[iu]
        else:
            dist_pairs = corr_pairs = np.array([])
        corr_by_d = [_nanmean(corr_pairs[(dist_pairs >= a) & (dist_pairs < b)]) for a, b in zip(bins[:-1], bins[1:])]
        # ramps
        ramps = np.abs(np.diff(pac_all[:, solar_idx], axis=0)) / inv_w[solar_idx]
        ramp_10 = float((ramps > 0.10).mean() * 100)
        fleet_ramp = np.abs(np.diff(fleet_kw)) / (inv_w[solar_idx].sum() / 1000)

        # advection: transit time across the neighbourhood and lagged correlation
        spd_km_min = np.hypot(u_km, v_km)
        broken = IS_DAY & (zenith < 70) & (f_cov > 0.2) & (f_cov < 0.8)
        transit_min = cfg["NEIGHBOURHOOD_KM"] / np.maximum(spd_km_min[broken], 1e-6)
        _dd = pd.Series(np.arange(N_T)).groupby(tmid.floor("D").to_numpy())
        lag_x = {l: [] for l in range(4)}
        lag_y = {l: [] for l in range(4)}
        _half = cfg["NEIGHBOURHOOD_KM"] / 2
        for _, ix in _dd:
            ix = ix.to_numpy()
            ix = ix[broken[ix]]
            if len(ix) < 24 or np.any(np.diff(ix) != 1):
                continue
            uu, vv = u_km[ix].mean(), v_km[ix].mean()
            nrm = np.hypot(uu, vv)
            proj = (xh[solar_idx] * uu + yh[solar_idx] * vv) / nrm
            up, dn = solar_idx[proj < -0.4 * _half], solar_idx[proj > 0.4 * _half]
            if len(up) < 3 or len(dn) < 3:
                continue
            k_up = np.diff(ghi_house[ix][:, up].mean(axis=1) / np.maximum(ghi_cs[ix], 1))
            k_dn = np.diff(ghi_house[ix][:, dn].mean(axis=1) / np.maximum(ghi_cs[ix], 1))
            for l in range(4):
                lag_x[l].append(k_up[:len(k_up) - l])
                lag_y[l].append(k_dn[l:])
        lag_corr = [float(np.corrcoef(np.concatenate(lag_x[l]), np.concatenate(lag_y[l]))[0, 1])
                    if lag_x[l] else float("nan") for l in range(4)]

        # PVWatts v5 benchmark (hourly, independent thermal & loss assumptions)
        pvw_ratio = np.full(N_H, np.nan)
        if cfg["PVWATTS_BENCHMARK"]:
            self.log("[STEP 12] PVWatts v5 benchmark (hourly, Fuentes roof-mount, default losses) ...")
            _hr = times.floor("h")
            _hidx = pd.DatetimeIndex(np.unique(_hr))
            _H = lambda a: pd.Series(a, index=_hr).groupby(level=0).mean().reindex(_hidx).to_numpy()
            ta_h, ws_h = _H(temp_air), _H(wind)
            pvw_loss = 1 - pvlib.pvsystem.pvwatts_losses(availability=0, age=0) / 100   # 11.4 %
            for h in solar_idx:
                self._check_cancel()
                tilt, az = fleet.at[h, "tilt"], fleet.at[h, "azimuth"]
                aoi = pvlib.irradiance.aoi(tilt, az, zenith, azimuth_sun)
                iam = np.nan_to_num(np.asarray(pvlib.iam.physical(aoi), float))
                tr = (poa_all[:, h] - poa_dir_all[:, h]) + poa_dir_all[:, h] * iam
                poa_h, tr_h = _H(poa_all[:, h]), _H(tr)
                tc_h = pvlib.temperature.fuentes(pd.Series(poa_h, index=_hidx), pd.Series(ta_h, index=_hidx),
                                                 pd.Series(ws_h, index=_hidx), noct_installed=49.0,
                                                 surface_tilt=tilt).to_numpy()
                pdc0 = cap_kw[h] * 1000
                pdc_h = np.asarray(pvlib.pvsystem.pvwatts_dc(tr_h, tc_h, pdc0, -0.0037), float) * pvw_loss
                pac_h = np.asarray(pvlib.inverter.pvwatts(pdc_h, inv_w[h] / 0.96, 0.96), float)
                pac_h = np.clip(np.nan_to_num(pac_h), 0, None)
                pvw_ratio[h] = energy_kwh[h] / (pac_h.sum() / 1000)

        # thermal summary (clear, high-irradiance steps)
        _hot = (poa_all[:, solar_idx] > 800)
        dT_cell = (self.tcell_all[:, solar_idx] - temp_air[:, None])[_hot]

        L = self.log
        L()
        L("=" * 78)
        L("VALIDATION")
        L("=" * 78)
        L(f"Weather source       : {'NASA POWER (satellite/reanalysis-derived)' if self.NASA_OK else 'SYNTHETIC climatological fallback'}")
        L(f"GHI (fleet mean)     : {ghi_house[:, solar_idx].mean(axis=1).sum() * DT_H / 1000:,.0f} kWh/m2 "
          f"vs source {self.ghi_reg.sum() * DT_H / 1000:,.0f}")
        L(f"POA (fleet mean)     : {np.nanmean(poa_kwh[solar_idx]):,.0f} kWh/m2")
        L(f"Fleet AC energy      : {energy_kwh.sum():,.0f} kWh")
        L(f"Specific yield       : {np.nanmean(sy):.0f} kWh/kWdc (range {np.nanmin(sy):.0f}-{np.nanmax(sy):.0f})")
        L(f"Capacity factor (DC) : {np.nanmean(cf) * 100:.1f} %")
        L(f"Performance ratio    : {np.nanmean(pr) * 100:.1f} % (range {np.nanmin(pr) * 100:.1f}-{np.nanmax(pr) * 100:.1f})")
        L(f"Losses (energy-wt.)  : shading {loss_shade * 100:.2f} %, soiling {loss_soil * 100:.2f} %, "
          f"snow {loss_snow * 100:.2f} %, clipping ~{clip_loss * 100:.2f} %")
        L(f"Clipping timesteps   : {clip_frac * 100:.2f} %   | MPPT-window limited: "
          f"{self.mppt_out_frac[solar_idx].mean() * 100:.3f} % of steps")
        L(f"Peak fleet AC        : {fleet_kw.max():.1f} kW of {inv_w[solar_idx].sum() / 1000:.1f} kW AC")
        L(f"5-min ramps >10% AC  : house {ramp_10:.2f} % of steps | fleet max ramp "
          f"{fleet_ramp.max() * 100:.1f} %/5min (smoothing by aggregation)")
        L(f"Cell - air at POA>800: median {_pct(dT_cell, 50):.1f} C, 95th pct {_pct(dT_cell, 95):.1f} C "
          f"({cfg['THERMAL_MODEL']})")
        L(f"Max 5-min GHI (house): {ghi_house.max():.0f} W/m2 = {np.nanmax(ghi_house / np.maximum(ghi_cs, 50)[:, None]):.2f} "
          "x clear-sky (cloud enhancement)")
        if cfg["PVWATTS_BENCHMARK"]:
            L(f"Sim / PVWatts v5     : {np.nanmean(pvw_ratio):.3f} (range {np.nanmin(pvw_ratio):.3f}-"
              f"{np.nanmax(pvw_ratio):.3f}) on identical POA irradiance")
        L(f"Cloud transit across the {cfg['NEIGHBOURHOOD_KM']:.0f} km neighbourhood (broken skies): "
          f"median {_pct(transit_min, 50):.1f} min (5-95 %: {_pct(transit_min, 5):.1f}-"
          f"{_pct(transit_min, 95):.1f})")
        L("Upwind -> downwind corr. of 5-min d(kt) at lag 0/1/2/3 steps: "
          + " / ".join(f"{c:.2f}" for c in lag_corr))
        L("Spatial corr. of 5-min d(kt) by distance [km]:")
        for (a, b), c in zip(zip(bins[:-1], bins[1:]), corr_by_d):
            L(f"      {a:3.1f}-{b:3.1f} km : {c:5.2f}")
        L("\nMonthly fleet energy (MWh):")
        L(monthly_mwh.round(2).to_string() if len(monthly_mwh) else "      (no complete month)")

        checks = []

        def _check(name, value, lo, hi, unit=""):
            ok = bool(lo <= value <= hi)
            L(f"  [{'OK  ' if ok else 'WARN'}] {name:<26} {value:8.2f}{unit}   expected {lo}-{hi}{unit}")
            checks.append(dict(name=name, value=float(value), low=lo, high=hi, unit=unit.strip(), ok=ok))

        L()
        full_year = hours / 24 >= 350
        _check("GHI closure (sim/source)", ghi_house[:, solar_idx].mean(axis=1).sum() / self.ghi_reg.sum(), 0.97, 1.03)
        _check("PR max (must be < 1)", np.nanmax(pr), 0.5, 0.99)
        _check("corr <0.5 km > corr >3 km", corr_by_d[0] - corr_by_d[-1], 0.05, 1.0)
        if full_year:
            _check("specific yield", np.nanmean(sy), 1050, 1450, " kWh/kW")
            _check("capacity factor (DC)", np.nanmean(cf) * 100, 12, 17, " %")
            _check("performance ratio", np.nanmean(pr) * 100, 75, 86, " %")
            _check("snow loss", loss_snow * 100, 0.0, 6.0, " %")
            _check("soiling loss", loss_soil * 100, 0.2, 3.0, " %")
        if cfg["PVWATTS_BENCHMARK"] and full_year:      # PVWatts has no snow loss -> annual only
            _check("sim / PVWatts v5", np.nanmean(pvw_ratio), 0.95, 1.05)
        _check("cell-air dT @POA>800 (med)", _pct(dT_cell, 50), 15, 35, " C")
        L(f"  -> {sum(c['ok'] for c in checks)}/{len(checks)} checks passed")
        if full_year:
            L("  Expected ranges refer to typical mid-latitude residential rooftop PV (northeastern U.S.);")
            L("  other climates may legitimately fall outside them.  Residential PR is typically 0.75-0.85")
            L("  (IEC 61724); the PVWatts v5 benchmark (Dobos 2014) uses the same POA, its own thermal model")
            L("  and default losses, so sim/PVWatts ~ 1 means the loss budget is mainstream.")
        else:
            L("  Annual checks (yield, PR, losses, PVWatts) are skipped for periods shorter than 350 days.")

        self.checks = checks
        self.energy_kwh, self.cap_kw, self.sy, self.cf, self.pr = energy_kwh, cap_kw, sy, cf, pr
        self.poa_kwh, self.inv_w, self.t_local, self.fleet_kw = poa_kwh, inv_w, t_local, fleet_kw
        self.df_fleet, self.monthly_mwh, self.pvw_ratio = df_fleet, monthly_mwh, pvw_ratio
        self.sel, self.bins, self.corr_by_d = sel, bins, corr_by_d
        self.dist_pairs, self.corr_pairs = dist_pairs, corr_pairs
        self.metrics = dict(
            site_name=cfg["SITE_NAME"], latitude=float(cfg["LATITUDE"]), longitude=float(cfg["LONGITUDE"]),
            weather_source="NASA POWER" if self.NASA_OK else "synthetic fallback",
            start_date=cfg["START_DATE"], end_date=cfg["END_DATE"], time_steps=int(N_T),
            n_houses=int(N_H), n_pv_houses=int(N_SOLAR),
            dc_capacity_kw=float(cap_kw.sum()), ac_capacity_kw=float(inv_w[solar_idx].sum() / 1000),
            ghi_source_kwh_m2=float(self.ghi_reg.sum() * DT_H / 1000),
            ghi_fleet_kwh_m2=float(ghi_house[:, solar_idx].mean(axis=1).sum() * DT_H / 1000),
            ghi_closure=float(ghi_house[:, solar_idx].mean(axis=1).sum() / self.ghi_reg.sum()),
            poa_kwh_m2=float(np.nanmean(poa_kwh[solar_idx])),
            energy_kwh=float(energy_kwh.sum()),
            specific_yield_mean=float(np.nanmean(sy)), specific_yield_min=float(np.nanmin(sy)),
            specific_yield_max=float(np.nanmax(sy)),
            capacity_factor_mean=float(np.nanmean(cf)),
            performance_ratio_mean=float(np.nanmean(pr)), performance_ratio_min=float(np.nanmin(pr)),
            performance_ratio_max=float(np.nanmax(pr)),
            loss_shading=float(loss_shade), loss_soiling=float(loss_soil), loss_snow=float(loss_snow),
            loss_clipping=float(clip_loss), clipping_steps=float(clip_frac),
            peak_fleet_kw=float(fleet_kw.max()), ramp_gt10_house_pct=ramp_10,
            fleet_max_ramp_pct=float(fleet_ramp.max() * 100),
            cell_air_dT_median=_pct(dT_cell, 50), cell_air_dT_p95=_pct(dT_cell, 95),
            max_ghi=float(ghi_house.max()),
            pvwatts_ratio_mean=_nanmean(pvw_ratio),
            pvwatts_ratio_min=float(np.nanmin(pvw_ratio)) if np.isfinite(pvw_ratio).any() else float("nan"),
            pvwatts_ratio_max=float(np.nanmax(pvw_ratio)) if np.isfinite(pvw_ratio).any() else float("nan"),
            cloud_speed_median_ms=float(np.median(self.cloud_speed_daily)),
            transit_median_min=_pct(transit_min, 50),
            lag_corr=lag_corr,
            corr_by_distance={f"{a:.1f}-{b:.1f} km": c for (a, b), c in zip(zip(bins[:-1], bins[1:]), corr_by_d)},
            monthly_mwh={k: float(v) for k, v in monthly_mwh.round(3).items()},
            missing_pct=float(np.isnan(self.pac_meas[:, solar_idx]).mean() * 100),
        )

    # ==========================================================================
    # STEP 13 - FIGURES
    # ==========================================================================
    def _step_figures(self):
        if not self.cfg["MAKE_FIGURES"]:
            return
        self.log("\n[STEP 13] Figures ...")
        try:
            from .figures import make_figures
            self.files += make_figures(self)
        except Exception as exc:              # figures are optional; never lose the dataset
            self.log(f"          WARNING: figures skipped ({exc.__class__.__name__}: {exc}). "
                     "The dataset is still exported. Reinstall matplotlib to get the figures: "
                     "python -m pip install --force-reinstall matplotlib")

    # ==========================================================================
    # STEP 14 - TABLES & EXPORT
    # ==========================================================================
    def _step_export(self):
        cfg, fleet = self.cfg, self.fleet
        N_H, IS_DAY, _out = self.N_H, self.IS_DAY, self._out
        energy_kwh, sy, cf, pr = self.energy_kwh, self.sy, self.cf, self.pr
        pac_all, t_local = self.pac_all, self.t_local
        self.log("\n" + "=" * 78)
        self.log(f"PER-HOUSE RESULTS - ALL {N_H} HOUSES")
        self.log("=" * 78)
        results_table = pd.DataFrame({
            "house": fleet["house_id"], "PV": np.where(fleet["has_pv"], "yes", "-"),
            "panels": fleet["number_of_panels"], "Ns": fleet["Ns"], "Np": fleet["Np"],
            "kW_dc": fleet["capacity_kw"].round(2), "kW_ac": fleet["inverter_capacity_kw"].round(2),
            "tilt": fleet["tilt"].round(0).astype(int), "azim": fleet["azimuth"].round(0).astype(int),
            "module": fleet["module_manufacturer"], "inverter": fleet["inverter_manufacturer"],
            "peak_kW": (pac_all.max(axis=0) / 1000).round(2), "annual_kWh": energy_kwh.round(0),
            "kWh_per_kW": np.round(sy, 0), "CF_%": np.round(cf * 100, 1), "PR_%": np.round(pr * 100, 1),
        })
        with pd.option_context("display.max_rows", None, "display.width", 220):
            self.log(results_table.to_string(index=False))
        results_table.to_csv(_out("per_house_results.csv"), index=False)

        self.log("\n[STEP 14] Exporting dataset ...")
        meta = fleet.drop(columns=[c for c in fleet.columns if c.startswith("_")]).copy()
        meta["annual_energy_kwh"] = energy_kwh.round(1)
        meta["specific_yield_kwh_per_kw"] = np.round(sy, 1)
        meta["capacity_factor"] = np.round(cf, 4)
        meta["performance_ratio"] = np.round(pr, 4)
        meta["mean_soiling_loss"] = self.soiling_loss.mean(axis=0).round(4)
        meta["mean_beam_shade_factor_day"] = self.shade_beam[IS_DAY].mean(axis=0).round(4)
        meta["snow_energy_loss_frac"] = np.round(np.where(self.dc_pre_snow_wh > 0, self.snow_lost_wh
                                                          / np.maximum(self.dc_pre_snow_wh, 1), 0), 4)
        meta["export_limit_w"] = np.where(np.isfinite(self.export_limit_w), self.export_limit_w, np.nan)
        meta["clock_offset_steps"] = self.clock_offset_steps
        meta["sim_over_pvwatts"] = np.round(self.pvw_ratio, 4)
        meta.to_csv(_out("pv_metadata.csv"), index=False)
        self.meta = meta
        files = ["per_house_results.csv", "pv_metadata.csv"]

        pv_df = pd.DataFrame(np.round(pac_all, 1), index=t_local, columns=fleet["house_id"])
        pv_df.index.name = "timestamp_interval_start"
        pv_meas_df = pd.DataFrame(self.pac_meas, index=t_local, columns=fleet["house_id"])
        pv_meas_df.index.name = "timestamp_interval_start"
        flags_df = pd.DataFrame(self.flags, index=t_local, columns=fleet["house_id"])
        flags_df.index.name = "timestamp_interval_start"
        weather_df = pd.DataFrame({
            "ghi_clearsky_wm2": np.round(self.ghi_cs, 1), "dni_clearsky_wm2": np.round(self.dni_cs, 1),
            "ghi_regional_wm2": np.round(self.ghi_reg, 1), "clear_sky_index_regional": np.round(self.kt_reg, 4),
            "cloud_cover_fraction": np.round(self.f_cov, 4), "cloud_u_km_min": np.round(self.u_km, 4),
            "cloud_v_km_min": np.round(self.v_km, 4), "temp_air_c": np.round(self.temp_air, 2),
            "wind_speed_ms": np.round(self.wind, 2), "relative_humidity_pct": np.round(self.rh, 1),
            "precipitation_mm": np.round(self.precip, 3), "snowfall_cm": np.round(self.snowfall_cm, 3),
            "ground_snow_depth_cm": np.round(self.ground_depth_cm, 2),
            "solar_zenith_deg": np.round(self.zenith, 2), "solar_azimuth_deg": np.round(self.azimuth_sun, 2),
            "is_daylight": IS_DAY.astype(int)}, index=t_local)
        weather_df.index.name = "timestamp_interval_start"

        if cfg["SAVE_CSV"]:
            comp = "gzip" if cfg["COMPRESS"] else None
            ext = ".csv.gz" if cfg["COMPRESS"] else ".csv"
            pv_df.to_csv(_out("pv_output_5min" + ext), compression=comp)
            pv_meas_df.to_csv(_out("pv_output_5min_measured" + ext), compression=comp)
            flags_df.to_csv(_out("meter_flags" + ext), compression=comp)
            weather_df.to_csv(_out("weather_data" + ext), compression=comp)
            files += [f"pv_output_5min{ext}", f"pv_output_5min_measured{ext}", f"meter_flags{ext}",
                      f"weather_data{ext}"]

        hz_el = np.stack([fleet.at[h, "_hz_el"] if fleet.at[h, "has_pv"] else np.zeros(72) for h in range(N_H)])
        hz_tr = np.stack([fleet.at[h, "_hz_tr"] if fleet.at[h, "has_pv"] else np.ones(72) for h in range(N_H)])
        core = dict(
            timestamps=self.times.asi8, pv_ac_w=pac_all, pv_ac_measured_w=self.pac_meas, meter_flags=self.flags,
            pv_dc_w=self.pdc_all, house_id=fleet["house_id"].to_numpy().astype(str),
            has_pv=fleet["has_pv"].to_numpy(), capacity_kw=self.cap_kw,
            inverter_kw=fleet["inverter_capacity_kw"].to_numpy(dtype=float),
            tilt=self.tilt_v, azimuth=fleet["azimuth"].to_numpy(dtype=float),
            x_km=self.xh, y_km=self.yh, ghi_clearsky_wm2=self.ghi_cs.astype(np.float32),
            temp_air_c=self.temp_air.astype(np.float32), wind_ms=self.wind.astype(np.float32),
            zenith_deg=self.zenith.astype(np.float32), azimuth_deg=self.azimuth_sun.astype(np.float32),
            is_day=IS_DAY)
        if cfg["EXPORT_GROUND_TRUTH"]:
            core.update(ghi_house_wm2=self.ghi_house, dni_house_wm2=self.dni_house, dhi_house_wm2=self.dhi_house,
                        poa_wm2=self.poa_all, poa_direct_unshaded_wm2=self.poa_dir_all,
                        effective_irradiance_wm2=self.eff_all, cell_temp_c=self.tcell_all,
                        dc_voltage_v=self.vdc_all, cloud_mask=self.cloud_mask, soiling_loss=self.soiling_loss,
                        beam_shade_factor=self.shade_beam, snow_coverage=self.snow_cov,
                        snow_dc_loss=self.snow_dc_loss,
                        horizon_elevation_deg=hz_el, horizon_transmittance=hz_tr,
                        cloud_u_km_min=self.u_km.astype(np.float32), cloud_v_km_min=self.v_km.astype(np.float32))
        np.savez_compressed(_out("pv_dataset.npz"), **core)
        files.append("pv_dataset.npz")

        save_config(self.cfg, _out("config_used.json"))
        files.append("config_used.json")
        from .export import write_dataset_readme
        write_dataset_readme(self, _out("README_dataset.md"))
        files += ["README_dataset.md", "summary.json"]
        self.files = files + self.files
        self.log(f"          {self.OUT_DIR}: " + ", ".join(files))

        cap_kw, inv_w, solar_idx = self.cap_kw, self.inv_w, self.solar_idx
        self.log("\n" + "=" * 78)
        self.log("SIMULATION COMPLETE")
        self.log("=" * 78)
        self.log(f"PV houses          : {self.N_SOLAR} / {N_H}")
        self.log(f"Installed capacity : {cap_kw.sum():.1f} kW DC / {inv_w[solar_idx].sum() / 1000:.1f} kW AC")
        self.log(f"Energy             : {energy_kwh.sum():,.0f} kWh ({np.nanmean(sy):.0f} kWh/kW)")
        self.log(f"Mean PR            : {np.nanmean(pr) * 100:.1f} %")
        self.log(f"Outputs            : {self.OUT_DIR}")
        self.log(f"Weather source     : {'NASA POWER (satellite/reanalysis-derived)' if self.NASA_OK else 'SYNTHETIC FALLBACK'}")


def run(config: dict | None = None, **overrides: Any) -> SimulationResult:
    """Convenience wrapper: ``synpv.run(LATITUDE=..., LONGITUDE=...)``."""
    cfg = default_config() if config is None else update_config(default_config(), config)
    update_config(cfg, overrides)
    return Simulation(cfg).run()

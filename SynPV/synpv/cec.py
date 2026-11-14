"""CEC module and inverter databases (NREL SAM tables, via pvlib)."""
from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
import pvlib

SAM_MODULES_URL = "https://raw.githubusercontent.com/NREL/SAM/develop/deploy/libraries/CEC%20Modules.csv"
SAM_INVERTERS_URL = "https://raw.githubusercontent.com/NREL/SAM/develop/deploy/libraries/CEC%20Inverters.csv"

MODULE_BRANDS = ["LONGi", "Jinko", "JA_Solar", "Trina", "Canadian_Solar", "REC",
                 "SunPower", "Maxeon", "Hanwha", "Q_CELLS", "LG_Electronics",
                 "Panasonic", "Hyundai", "Risen", "Aptos", "Silfab", "Mission_Solar"]
INVERTER_BRANDS = ["SolarEdge", "SMA", "Fronius", "Huawei", "ABB", "Sungrow",
                   "Growatt", "KACO", "Delta", "Generac", "Tesla"]
# approximate US residential string-inverter market weights (used for sampling)
INVERTER_BRAND_WEIGHTS = {"solaredge": 0.45, "sma": 0.15, "tesla": 0.10, "fronius": 0.08,
                          "generac": 0.06, "delta": 0.05, "abb": 0.05}
DEFAULT_BRAND_W = 0.02


def _num(x, default=np.nan):
    v = np.array(pd.to_numeric(pd.Series(x), errors="coerce"), dtype=float, copy=True)
    v[np.isnan(v)] = default
    return v


@lru_cache(maxsize=4)
def _retrieve(name_bundled, url):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # SAM tables contain duplicate names
        try:
            db = pvlib.pvsystem.retrieve_sam(path=url)
            src = "NREL/SAM GitHub"
        except Exception:
            db = pvlib.pvsystem.retrieve_sam(name_bundled)
            src = "bundled pvlib"
    return db, src


def _brand_mask(columns, brands):
    names = pd.Index(columns).astype(str).str.lower()
    mask = np.zeros(len(names), dtype=bool)
    for b in brands:
        mask |= names.str.startswith(b.lower())
    return mask


def load_cec_modules(log=print):
    try:
        cec, src = _retrieve("CECMod", SAM_MODULES_URL)
        cec = cec.copy()
        log(f"[CEC MOD] {cec.shape[1]} modules from {src}")
        cec = cec.loc[:, _brand_mask(cec.columns, MODULE_BRANDS)]
        stc = _num(cec.loc["STC"])
        tech = cec.loc["Technology"].astype(str).str.lower().to_numpy()
        bif = _num(cec.loc["Bifacial"], 0) if "Bifacial" in cec.index else np.zeros(len(stc))
        bipv = cec.loc["BIPV"].astype(str).str.upper().to_numpy() if "BIPV" in cec.index \
            else np.array(["N"] * len(stc))
        year = pd.to_datetime(cec.loc["Date"], errors="coerce").dt.year.to_numpy() \
            if "Date" in cec.index else np.full(len(stc), 2020)
        keep = ((stc >= 300) & (stc <= 600) & np.char.startswith(tech.astype(str), "mono")
                & (bif == 0) & (bipv != "Y") & (np.nan_to_num(year, nan=2020) >= 2017))
        cec = cec.loc[:, keep]
        need = ["STC", "A_c", "N_s", "I_sc_ref", "V_oc_ref", "I_mp_ref", "V_mp_ref",
                "alpha_sc", "beta_oc", "a_ref", "I_L_ref", "I_o_ref", "R_s", "R_sh_ref", "Adjust"]
        missing = [p for p in need if p not in cec.index]
        if missing:
            raise ValueError(f"missing params {missing}")
        extra = cec.loc[[r for r in ("T_NOCT", "gamma_r") if r in cec.index]].apply(pd.to_numeric, errors="coerce")
        cec = cec.loc[need].apply(pd.to_numeric, errors="coerce").dropna(axis=1)
        noct = extra.loc["T_NOCT"].reindex(cec.columns) if "T_NOCT" in extra.index else pd.Series(np.nan, cec.columns)
        cec.loc["T_NOCT"] = noct.where((noct > 38) & (noct < 52), 45.0).fillna(45.0)
        gam = extra.loc["gamma_r"].reindex(cec.columns) if "gamma_r" in extra.index else pd.Series(np.nan, cec.columns)
        cec.loc["gamma_r"] = gam.where((gam > -0.6) & (gam < -0.2), -0.37).fillna(-0.37)
        # drop parameter sets that are numerically suspicious
        eff = cec.loc["STC"] / (cec.loc["A_c"] * 1000.0)
        cec = cec.loc[:, (eff > 0.17) & (eff < 0.25) & (cec.loc["R_sh_ref"] > 50)]
        if cec.shape[1] == 0:
            raise ValueError("no modules after filtering")
        log(f"[CEC MOD] {cec.shape[1]} modern residential mono-Si modules kept "
            f"({cec.loc['STC'].min():.0f}-{cec.loc['STC'].max():.0f} W)")
        return cec
    except Exception as exc:
        log(f"[CEC MOD] failed ({exc}) - using generic 400 W fallback")
        return None


def load_cec_inverters(log=print):
    try:
        cec, src = _retrieve("cecinverter", SAM_INVERTERS_URL)
        cec = cec.copy()
        log(f"[CEC INV] {cec.shape[1]} inverters from {src}")
        cec = cec.loc[:, _brand_mask(cec.columns, INVERTER_BRANDS)]
        need = ["Vac", "Paco", "Pdco", "Vdco", "Pso", "C0", "C1", "C2", "C3", "Pnt",
                "Vdcmax", "Idcmax", "Mppt_low", "Mppt_high"]
        cec = cec.loc[need].apply(pd.to_numeric, errors="coerce").dropna(axis=1)
        keep = ((cec.loc["Vac"] == 240) & (cec.loc["Paco"] >= 2500) & (cec.loc["Paco"] <= 12500)
                & (cec.loc["Vdcmax"] >= 300) & (cec.loc["Mppt_high"] > cec.loc["Mppt_low"]))
        cec = cec.loc[:, keep]
        if cec.shape[1] == 0:
            raise ValueError("no inverters after filtering")
        log(f"[CEC INV] {cec.shape[1]} residential 240 V string inverters kept")
        return cec
    except Exception as exc:
        log(f"[CEC INV] failed ({exc}) - using fallback efficiency curve")
        return None

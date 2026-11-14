"""NASA POWER hourly meteorology: download, cache, and offline fallback."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import requests

NASA_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
NASA_DAILY_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_PARAMS = ["ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN", "T2M", "WS10M", "RH2M", "PRECTOTCORR"]
NASA_FIRST_DATE = "2001-01-01"      # first day of the POWER hourly record


class WeatherError(RuntimeError):
    """Raised when NASA POWER data cannot be obtained."""


def _fetch_chunk(lat, lon, start, end, parameters, timeout):
    params = {"parameters": ",".join(parameters), "community": "RE",
              "latitude": lat, "longitude": lon,
              "start": start.replace("-", ""), "end": end.replace("-", ""),
              "format": "JSON", "time-standard": "UTC"}
    r = requests.get(NASA_URL, params=params, timeout=timeout,
                     headers={"User-Agent": "SynPV (synthetic PV fleet generator)"})
    if r.status_code != 200:
        try:
            msg = "; ".join(r.json().get("messages", [])) or r.text[:300]
        except ValueError:
            msg = r.text[:300]
        raise WeatherError(f"NASA POWER returned HTTP {r.status_code}: {msg}")
    js = r.json()
    df = pd.DataFrame(js["properties"]["parameter"])
    df.index = pd.to_datetime(df.index, format="%Y%m%d%H").tz_localize("UTC")
    coords = js.get("geometry", {}).get("coordinates", [None, None, None])
    elev = coords[2] if len(coords) > 2 else None
    return df.replace(-999.0, np.nan).sort_index(), elev


def fetch_nasa_power(lat, lon, start, end, parameters=NASA_PARAMS, timeout=180, log=print):
    """Download hourly NASA POWER data (UTC), one request per calendar year.

    Returns ``(dataframe, elevation_m)``.
    """
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    if t1 < t0:
        raise WeatherError("END_DATE is before START_DATE")
    frames, elev = [], None
    for y in range(t0.year, t1.year + 1):
        a = max(t0, pd.Timestamp(f"{y}-01-01")).strftime("%Y-%m-%d")
        b = min(t1, pd.Timestamp(f"{y}-12-31")).strftime("%Y-%m-%d")
        if t1.year > t0.year:
            log(f"          requesting {a} .. {b}")
        try:
            df, e = _fetch_chunk(lat, lon, a, b, parameters, timeout)
        except requests.RequestException as exc:
            raise WeatherError(f"could not reach NASA POWER ({exc.__class__.__name__}: {exc})") from exc
        frames.append(df)
        elev = e if e is not None else elev
    wx = pd.concat(frames).sort_index()
    return wx[~wx.index.duplicated()], elev


def cache_path(cache_dir, lat, lon, start, end):
    return os.path.join(cache_dir, f"nasa_power_{lat:.4f}_{lon:.4f}_{start}_{end}.csv")


def load_or_fetch(lat, lon, start, end, cache_dir, log=print):
    """Return ``(dataframe, elevation_m, source_text)`` using the on-disk cache."""
    path = cache_path(cache_dir, lat, lon, start, end)
    meta_path = path[:-4] + ".meta.json"
    if os.path.exists(path):
        wx = pd.read_csv(path, index_col=0)
        wx.index = pd.to_datetime(wx.index, utc=True)
        elev = None
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as fh:
                elev = json.load(fh).get("elevation_m")
        log(f"          loaded from cache {path}")
        return wx, elev, "cache"
    wx, elev = fetch_nasa_power(lat, lon, start, end, log=log)
    os.makedirs(cache_dir, exist_ok=True)
    wx.to_csv(path)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump({"elevation_m": elev, "latitude": lat, "longitude": lon,
                   "start": start, "end": end, "source": NASA_URL}, fh, indent=2)
    log(f"          cached to {path}")
    return wx, elev, "download"


def design_minimum_temperature(lat, lon, end_year, cache_dir, years=10, timeout=120, log=print):
    """Extreme annual mean minimum temperature (C), the statistic used for
    string sizing by NEC 690.7 (ASHRAE design dry bulb).

    Computed as the mean of the yearly minima of NASA POWER daily ``T2M_MIN``
    over the ``years`` complete years before ``end_year``.
    """
    y1 = max(1981, int(end_year) - 1)
    y0 = max(1981, y1 - years + 1)
    path = os.path.join(cache_dir, f"nasa_power_daily_tmin_{lat:.4f}_{lon:.4f}_{y0}_{y1}.csv")
    if os.path.exists(path):
        ser = pd.read_csv(path, index_col=0).iloc[:, 0]
        ser.index = pd.to_datetime(ser.index)
    else:
        params = {"parameters": "T2M_MIN", "community": "RE", "latitude": lat, "longitude": lon,
                  "start": f"{y0}0101", "end": f"{y1}1231", "format": "JSON", "time-standard": "UTC"}
        try:
            r = requests.get(NASA_DAILY_URL, params=params, timeout=timeout,
                             headers={"User-Agent": "SynPV (synthetic PV fleet generator)"})
        except requests.RequestException as exc:
            raise WeatherError(f"could not reach NASA POWER ({exc.__class__.__name__})") from exc
        if r.status_code != 200:
            raise WeatherError(f"NASA POWER daily request returned HTTP {r.status_code}")
        d = r.json()["properties"]["parameter"]["T2M_MIN"]
        ser = pd.Series(d, dtype=float).replace(-999.0, np.nan).dropna()
        ser.index = pd.to_datetime(ser.index, format="%Y%m%d")
        os.makedirs(cache_dir, exist_ok=True)
        ser.rename("T2M_MIN").to_csv(path)
    yearly = ser.groupby(ser.index.year).min()
    if len(yearly) == 0:
        raise WeatherError("no daily minimum temperatures returned")
    return float(yearly.mean()), int(yearly.index.min()), int(yearly.index.max())


def synthetic_weather(start, end, lat, rng, lon=-75.0):
    """Climatologically plausible hourly NY-like weather (offline fallback only).

    Daily sky regimes (clear / broken / overcast) follow a seasonal Markov chain;
    KT_CS is the hourly clear-sky index relative to pvlib Ineichen.  The diurnal
    cycle follows the solar time of ``lon`` and the seasons are flipped south of
    the equator.  Intended for offline testing only - the climate is not that of
    the chosen site.
    """
    season_shift = 182.6 if lat < 0 else 0.0
    idx = pd.date_range(f"{start} 00:00", f"{end} 23:00", freq="h", tz="UTC")
    days = pd.date_range(start, end, freq="D", tz="UTC")
    nd = len(days)
    doy = days.dayofyear.to_numpy()
    summer = 0.5 * (1 + np.cos(2 * np.pi * (doy - 196 - season_shift) / 365.25))   # 1 in July (NH)
    p_w, p_s = np.array([0.36, 0.30, 0.34]), np.array([0.46, 0.40, 0.14])
    regime = np.zeros(nd, dtype=int)
    for d in range(nd):
        p = p_w + (p_s - p_w) * summer[d]
        regime[d] = regime[d - 1] if (d > 0 and rng.random() < 0.45) else rng.choice(3, p=p / p.sum())
    kt_day = np.select([regime == 0, regime == 1, regime == 2],
                       [rng.normal(0.97, 0.03, nd), rng.normal(0.66, 0.09, nd),
                        rng.normal(0.27, 0.08, nd)])
    noise_sd = np.array([0.02, 0.13, 0.07])[regime]
    t_syn = rng.normal(0, 3.0, nd)
    for d in range(1, nd):
        t_syn[d] = 0.75 * t_syn[d - 1] + 0.66 * t_syn[d]
    ws_day = np.clip(3.2 + 1.2 * (regime == 2) + rng.gamma(2.0, 0.6, nd) - 1.2, 0.8, 12)

    dmap = pd.Series(np.arange(nd), index=days.date)
    di = dmap.reindex(idx.date).to_numpy()
    hl = (idx.hour.to_numpy() + int(round(lon / 15.0))) % 24   # approx local solar hour
    kt = np.empty(len(idx))
    e = 0.0
    for i in range(len(idx)):
        e = 0.6 * e + rng.normal(0, 1)
        kt[i] = kt_day[di[i]] + noise_sd[di[i]] * e
    kt = np.clip(kt, 0.05, 1.08)

    t_mean = 12.5 - 12.0 * np.cos(2 * np.pi * (idx.dayofyear.to_numpy() - 20 - season_shift) / 365.25)
    amp = 3.5 + 3.0 * (regime[di] == 0)
    temp = t_mean + t_syn[di] + amp * np.sin(2 * np.pi * (hl - 9) / 24) + rng.normal(0, 0.5, len(idx))
    rh = np.clip(70 - 18 * np.sin(2 * np.pi * (hl - 9) / 24) + 15 * (regime[di] == 2)
                 + rng.normal(0, 5, len(idx)), 15, 100)
    wind = np.clip(ws_day[di] * (1 + 0.25 * np.sin(2 * np.pi * (hl - 10) / 24))
                   + rng.normal(0, 0.4, len(idx)), 0.3, None)

    precip = np.zeros(len(idx))
    for d in range(nd):
        prob, mean = [(0.03, 2.0), (0.25, 6.0), (0.85, 12.0)][regime[d]]
        if rng.random() < prob:
            total = rng.exponential(mean)
            n_h = rng.integers(3, 11)
            h0 = rng.integers(0, 24 - n_h + 1)
            sl = np.where(di == d)[0][h0:h0 + n_h]
            w = rng.random(len(sl)) + 0.2
            precip[sl] = total * w / w.sum()
    return pd.DataFrame({"T2M": temp, "WS10M": wind, "RH2M": rh,
                         "PRECTOTCORR": precip, "KT_CS": kt}, index=idx)

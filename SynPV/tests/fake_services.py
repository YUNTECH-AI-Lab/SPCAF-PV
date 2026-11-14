"""Offline stand-ins for NASA POWER and Nominatim, used by the tests.

``install()`` patches ``requests.get`` so that NASA POWER returns hourly data
built from SynPV's synthetic weather generator (in the real API format), and
Nominatim returns a small fixed gazetteer.
"""
import json

import numpy as np
import pandas as pd
import pvlib
import requests

_real_get = requests.get

PLACES = [
    ("Levittown, Nassau County, New York, United States", 40.7259, -73.5143),
    ("Munich, Bavaria, Germany", 48.1374, 11.5755),
    ("Sydney, New South Wales, Australia", -33.8688, 151.2093),
    ("Taipei, Taiwan", 25.0330, 121.5654),
]


class _Resp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status
        self.text = json.dumps(payload)

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def _nasa(params):
    from synpv.weather import synthetic_weather
    lat, lon = float(params["latitude"]), float(params["longitude"])
    start = pd.Timestamp(params["start"]).strftime("%Y-%m-%d")
    end = pd.Timestamp(params["end"]).strftime("%Y-%m-%d")
    rng = np.random.default_rng(abs(hash((round(lat, 3), round(lon, 3), start))) % 2**32)
    wx = synthetic_weather(start, end, lat, rng, lon=lon)
    # hourly clear-sky mean over each hour (interval-start convention, like NASA)
    t5 = pd.date_range(f"{start} 00:00", f"{end} 23:59", freq="5min", tz="UTC") + pd.Timedelta(minutes=2.5)
    loc = pvlib.location.Location(lat, lon, altitude=50)
    cs = loc.get_clearsky(t5)["ghi"].clip(lower=0)
    cs_h = cs.groupby(t5.floor("h")).mean().reindex(wx.index).fillna(0).to_numpy()
    allsky = cs_h * wx["KT_CS"].to_numpy()
    if abs(lat) < 35:                       # warmer sites: no snow in the stand-in data
        wx["T2M"] = wx["T2M"].to_numpy() + 10.0
    par = {"ALLSKY_SFC_SW_DWN": allsky, "CLRSKY_SFC_SW_DWN": cs_h, "T2M": wx["T2M"].to_numpy(),
           "WS10M": wx["WS10M"].to_numpy(), "RH2M": wx["RH2M"].to_numpy(),
           "PRECTOTCORR": wx["PRECTOTCORR"].to_numpy()}
    keys = wx.index.strftime("%Y%m%d%H")
    out = {k: {kk: float(round(v, 3)) for kk, v in zip(keys, arr)} for k, arr in par.items()}
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat, 37.5]},
            "properties": {"parameter": out}, "messages": []}


def _nasa_daily(params):
    lat = float(params["latitude"])
    days = pd.date_range(pd.Timestamp(params["start"]), pd.Timestamp(params["end"]), freq="D")
    doy = days.dayofyear.to_numpy()
    shift = 182.6 if lat < 0 else 0.0
    base = 25.0 - 0.45 * abs(lat)
    tmin = base - 9.0 * np.cos(2 * np.pi * (doy - 20 - shift) / 365.25) - 4.0
    tmin += np.random.default_rng(7).normal(0, 3.0, len(days))
    return {"properties": {"parameter": {"T2M_MIN": {d.strftime("%Y%m%d"): float(round(v, 2))
                                                     for d, v in zip(days, tmin)}}}}


def _fake_get(url, params=None, **kw):
    params = params or {}
    if "power.larc.nasa.gov" in url and "/daily/" in url:
        return _Resp(_nasa_daily(params))
    if "power.larc.nasa.gov" in url:
        return _Resp(_nasa(params))
    if "nominatim" in url and url.endswith("/search"):
        q = params.get("q", "").lower()
        hits = [dict(display_name=n, lat=str(a), lon=str(b)) for n, a, b in PLACES if q.split(",")[0] in n.lower()]
        return _Resp(hits)
    if "nominatim" in url and url.endswith("/reverse"):
        lat, lon = float(params["lat"]), float(params["lon"])
        n, a, b = min(PLACES, key=lambda p: (p[1] - lat) ** 2 + (p[2] - lon) ** 2)
        if (a - lat) ** 2 + (b - lon) ** 2 > 4:
            return _Resp({"display_name": ""})
        parts = n.split(", ")
        return _Resp({"display_name": n, "address": {"city": parts[0], "state": parts[-2] if len(parts) > 2 else None,
                                                      "country_code": "xx"}})
    return _real_get(url, params=params, **kw)


def install():
    requests.get = _fake_get
    import synpv.weather as w
    w.requests.get = _fake_get

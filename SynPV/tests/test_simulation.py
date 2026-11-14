import json
import os

import numpy as np
import pytest

from synpv import Simulation, WeatherError, default_config


def _cfg(tmp_path, **kw):
    base = dict(LATITUDE=48.1374, LONGITUDE=11.5755, START_DATE="2023-06-01", END_DATE="2023-06-10",
                N_HOUSES=16, OUTPUT_DIR=str(tmp_path / "out"), NASA_CACHE_DIR=str(tmp_path / "cache"),
                MAKE_FIGURES=False, PVWATTS_BENCHMARK=True)
    base.update(kw)
    return default_config(**base)


@pytest.fixture(scope="module")
def munich(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("munich")
    sim = Simulation(_cfg(tmp, MAKE_FIGURES=True), log=lambda *_: None)
    return sim, sim.run()


def test_outputs_exist(munich):
    _, res = munich
    for f in ("pv_output_5min.csv.gz", "pv_output_5min_measured.csv.gz", "meter_flags.csv.gz",
              "weather_data.csv.gz", "pv_metadata.csv", "pv_dataset.npz", "config_used.json",
              "summary.json", "README_dataset.md", "fig1_irradiance.png", "fig8_variability.png"):
        assert os.path.exists(os.path.join(res.output_dir, f)), f


def test_physics(munich):
    sim, res = munich
    assert sim.pac_all[~sim.IS_DAY].max() == 0                        # no export at night
    paco = sim.fleet["inverter_paco_w"].fillna(0).to_numpy()
    assert np.all(sim.pac_all.max(axis=0) <= paco + 1e-3)             # clipping respected
    assert 0.6 < res.metrics["performance_ratio_mean"] < 0.95
    assert abs(res.metrics["ghi_closure"] - 1) < 0.03                 # energy conserving
    c = res.metrics["corr_by_distance"]
    assert c["0.0-0.5 km"] > c["2.0-3.0 km"]                          # spatial decorrelation


def test_auto_values_resolved(munich):
    _, res = munich
    cfg = json.load(open(os.path.join(res.output_dir, "config_used.json")))
    assert cfg["TIMEZONE"] == "Europe/Berlin"
    assert isinstance(cfg["ALTITUDE_M"], float) and isinstance(cfg["DESIGN_T_MIN"], float)
    assert cfg["CLOUD_PREVAILING_FROM_DEG"] == 260.0


def test_reproducible(munich, tmp_path):
    sim1, _ = munich
    sim2 = Simulation(_cfg(tmp_path), log=lambda *_: None)
    sim2.run()
    assert np.array_equal(sim1.pac_all, sim2.pac_all)
    assert np.array_equal(sim1.pac_meas, sim2.pac_meas, equal_nan=True)


def test_measured_flags(munich):
    sim, _ = munich
    flags = sim.flags[:, sim.solar_idx]
    assert np.all(np.isnan(sim.pac_meas[:, sim.solar_idx][flags == 1]))   # gaps are missing values


def test_southern_hemisphere(tmp_path):
    sim = Simulation(_cfg(tmp_path, LATITUDE=-33.87, LONGITUDE=151.21, START_DATE="2023-01-01",
                          END_DATE="2023-01-05", PVWATTS_BENCHMARK=False), log=lambda *_: None)
    res = sim.run()
    az = res.fleet.loc[res.fleet.has_pv, "azimuth"].to_numpy()
    assert np.all((az < 120) | (az > 240))                            # roofs face the equator (north)
    assert res.config["TIMEZONE"] == "Australia/Sydney"


def test_nasa_failure_is_reported(tmp_path, monkeypatch):
    import synpv.weather as w

    def boom(*a, **k):
        raise w.requests.ConnectionError("offline")
    monkeypatch.setattr(w.requests, "get", boom)
    with pytest.raises(WeatherError):
        Simulation(_cfg(tmp_path, NASA_CACHE_DIR=str(tmp_path / "empty")), log=lambda *_: None).run()


def test_cache_is_used(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, PVWATTS_BENCHMARK=False, END_DATE="2023-06-03")
    Simulation(cfg, log=lambda *_: None).run()
    import synpv.weather as w
    monkeypatch.setattr(w.requests, "get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no network")))
    cfg2 = _cfg(tmp_path, PVWATTS_BENCHMARK=False, END_DATE="2023-06-03", OUTPUT_DIR=str(tmp_path / "out2"))
    Simulation(cfg2, log=lambda *_: None).run()                       # served from the cache

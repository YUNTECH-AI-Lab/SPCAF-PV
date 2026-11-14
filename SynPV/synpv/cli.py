"""Command-line interface.

    synpv app                              # web app with the interactive globe
    synpv run --lat 48.137 --lon 11.575 --start 2023-01-01 --end 2023-12-31 --houses 50
    synpv run --config examples/paper_levittown_2024.json
    synpv config > my_config.json          # write the full default configuration
"""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .config import DEFAULT_CONFIG, default_config, load_config, paper_config, update_config


def _cmd_run(a):
    if a.config:
        cfg = load_config(a.config)
    elif a.paper:
        cfg = paper_config()
    else:
        cfg = default_config()
    over = {}
    for key, val in (("LATITUDE", a.lat), ("LONGITUDE", a.lon), ("START_DATE", a.start), ("END_DATE", a.end),
                     ("N_HOUSES", a.houses), ("PV_OWNERSHIP", a.pv_share), ("FREQ_MIN", a.freq),
                     ("SEED", a.seed), ("SITE_NAME", a.name), ("OUTPUT_DIR", a.out),
                     ("NASA_CACHE_DIR", a.cache)):
        if val is not None:
            over[key] = val
    if a.no_figures:
        over["MAKE_FIGURES"] = False
    if a.allow_synthetic:
        over["ALLOW_SYNTHETIC_WEATHER"] = True
    for item in a.set or []:
        k, _, v = item.partition("=")
        k = k.strip().upper()
        if k not in DEFAULT_CONFIG:
            sys.exit(f"unknown configuration key: {k}")
        try:
            over[k] = json.loads(v)
        except json.JSONDecodeError:
            over[k] = v
    update_config(cfg, over)
    if not a.config and not a.paper and (a.lat is None or a.lon is None):
        sys.exit("give --lat and --lon, --config FILE, or --paper (run 'synpv run -h' for help)")
    from .simulator import Simulation
    from .weather import WeatherError
    try:
        res = Simulation(cfg).run()
    except (WeatherError, ValueError) as exc:
        sys.exit(f"\nSynPV stopped: {exc}")
    passed = sum(c["ok"] for c in res.checks)
    print(f"\nDone: {res.output_dir}  ({passed}/{len(res.checks)} validation checks passed)")


def _cmd_app(a):
    from .webapp.server import serve
    serve(host=a.host, port=a.port, runs_dir=a.runs_dir, open_browser=not a.no_browser)


def _cmd_config(a):
    cfg = paper_config() if a.paper else default_config()
    print(json.dumps(cfg, indent=2, default=list))


def main(argv=None):
    p = argparse.ArgumentParser(prog="synpv", description="SynPV - synthetic residential PV fleet generator "
                                "driven by NASA POWER weather.")
    p.add_argument("--version", action="version", version=f"SynPV {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="generate a dataset from the command line")
    r.add_argument("--config", help="JSON/YAML file with configuration overrides")
    r.add_argument("--paper", action="store_true", help="reproduce the paper case study (Levittown, NY, 2024)")
    r.add_argument("--lat", type=float, help="latitude in degrees (north positive)")
    r.add_argument("--lon", type=float, help="longitude in degrees (east positive)")
    r.add_argument("--start", help="start date YYYY-MM-DD")
    r.add_argument("--end", help="end date YYYY-MM-DD")
    r.add_argument("--houses", type=int, help="number of houses")
    r.add_argument("--pv-share", type=float, help="fraction of houses with PV (0-1)")
    r.add_argument("--freq", type=int, help="resolution in minutes (divisor of 60)")
    r.add_argument("--seed", type=int, help="random seed")
    r.add_argument("--name", help="site name")
    r.add_argument("--out", help="output folder")
    r.add_argument("--cache", help="NASA POWER cache folder")
    r.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="any configuration key, e.g. --set VDC_ABS_MAX=1000 --set TILT_RANGE=[10,35]")
    r.add_argument("--no-figures", action="store_true", help="skip the diagnostic figures")
    r.add_argument("--allow-synthetic", action="store_true",
                   help="use synthetic weather if NASA POWER cannot be reached (testing only)")
    r.set_defaults(func=_cmd_run)

    w = sub.add_parser("app", help="start the web app with the interactive globe")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=8050)
    w.add_argument("--runs-dir", default="synpv_runs", help="folder for the generated datasets")
    w.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    w.set_defaults(func=_cmd_app)

    c = sub.add_parser("config", help="print the default (or paper) configuration as JSON")
    c.add_argument("--paper", action="store_true")
    c.set_defaults(func=_cmd_config)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()

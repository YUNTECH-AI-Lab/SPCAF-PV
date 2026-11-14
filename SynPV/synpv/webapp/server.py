"""SynPV web application: pick a location on the globe, fetch NASA POWER, generate.

Start with ``synpv app`` (or ``python -m synpv app``) and open the printed URL.
"""
from __future__ import annotations

import datetime as _dt
import io
import math
import os
import re
import threading
import time
import traceback
import uuid
import zipfile

import requests
from flask import Flask, abort, jsonify, render_template, request, send_file, send_from_directory

from .. import __version__
from ..config import DEFAULT_CONFIG, PAPER_PRESET, default_config, update_config
from ..simulator import STEPS, Simulation, SimulationCancelled, estimate_memory_gb

NOMINATIM = "https://nominatim.openstreetmap.org"
UA = {"User-Agent": f"SynPV/{__version__} (research PV dataset generator)", "Accept-Language": "en"}

# keys the browser may set (everything else keeps its default)
EDITABLE = {
    "LATITUDE", "LONGITUDE", "SITE_NAME", "ALTITUDE_M", "TIMEZONE", "START_DATE", "END_DATE", "FREQ_MIN",
    "N_HOUSES", "PV_OWNERSHIP", "NEIGHBOURHOOD_KM", "SEED", "SYSTEM_KW_MEDIAN", "SYSTEM_KW_RANGE",
    "DC_AC_RATIO_RANGE", "TILT_RANGE", "VDC_ABS_MAX", "DESIGN_T_MIN", "CLOUD_PREVAILING_FROM_DEG",
    "EQUATOR_FACING", "ALBEDO", "PVWATTS_BENCHMARK", "EXPORT_GROUND_TRUTH", "SAVE_CSV",
    "OUTAGE_EVENTS_PER_YEAR", "GAP_EVENTS_PER_YEAR", "EXPORT_LIMIT_FRACTION_OF_HOUSES",
    "CLOCK_OFFSET_FRACTION", "METER_NOISE_REL", "ALLOW_SYNTHETIC_WEATHER",
}
MAX_HOUSES = 1000
MAX_MEMORY_GB = float(os.environ.get("SYNPV_MAX_MEMORY_GB", "24"))


def _clean(o):
    """JSON-safe copy (NaN/inf -> None, tuples -> lists, numpy scalars -> python)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        try:
            o = o.item()
        except Exception:
            pass
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


class Job:
    def __init__(self, cfg):
        self.id = uuid.uuid4().hex[:12]
        self.cfg = cfg
        self.status = "queued"          # queued | running | done | error | cancelled
        self.step, self.n_steps, self.label = 0, len(STEPS), "Starting"
        self.log: list[str] = []
        self.result = None
        self.error = None
        self.started = time.time()
        self.finished = None
        self.sim: Simulation | None = None

    def snapshot(self, since=0):
        d = dict(id=self.id, status=self.status, step=self.step, n_steps=self.n_steps, label=self.label,
                 steps=STEPS, log=self.log[since:], log_next=len(self.log), error=self.error,
                 elapsed_s=round((self.finished or time.time()) - self.started, 1),
                 output_dir=self.cfg["OUTPUT_DIR"])
        if self.result is not None:
            r = self.result
            d.update(metrics=r.metrics, checks=r.checks, files=r.files, config=r.config)
        return _clean(d)


class JobManager:
    def __init__(self, runs_dir):
        self.runs_dir = os.path.abspath(runs_dir)
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()

    def busy(self):
        return any(j.status in ("queued", "running") for j in self.jobs.values())

    def start(self, cfg):
        with self.lock:
            if self.busy():
                raise RuntimeError("a simulation is already running - wait for it or cancel it")
            job = Job(cfg)
            slug = re.sub(r"[^A-Za-z0-9]+", "_", str(cfg.get("SITE_NAME") or "site")).strip("_")[:40] or "site"
            cfg["OUTPUT_DIR"] = os.path.join(self.runs_dir, f"{_dt.datetime.now():%Y%m%d_%H%M%S}_{slug}_{job.id[:6]}")
            cfg["NASA_CACHE_DIR"] = os.path.join(self.runs_dir, "nasa_cache")
            self.jobs[job.id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def _run(self, job):
        job.status = "running"

        def log(msg=""):
            job.log.extend(str(msg).split("\n"))

        def progress(i, n, label):
            job.step, job.n_steps, job.label = i, n, label

        try:
            job.sim = Simulation(job.cfg, log=log, progress=progress)
            job.result = job.sim.run()
            job.status = "done"
        except SimulationCancelled:
            job.status = "cancelled"
            log("Simulation cancelled.")
        except Exception as exc:                       # report every failure to the browser
            job.status = "error"
            job.error = f"{exc.__class__.__name__}: {exc}"
            log("ERROR: " + job.error)
            log(traceback.format_exc())
        finally:
            job.finished = time.time()
            job.sim = None                             # free the large arrays; results are on disk


def create_app(runs_dir="synpv_runs"):
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.json.sort_keys = False
    jm = JobManager(runs_dir)
    os.makedirs(jm.runs_dir, exist_ok=True)

    @app.get("/")
    def index():
        return render_template("index.html", version=__version__)

    @app.get("/api/defaults")
    def defaults():
        today = _dt.date.today()
        return jsonify(_clean(dict(config=DEFAULT_CONFIG, paper=PAPER_PRESET, steps=STEPS, version=__version__,
                                   max_houses=MAX_HOUSES, max_memory_gb=MAX_MEMORY_GB,
                                   min_date="2001-01-01",
                                   max_date=(today - _dt.timedelta(days=7)).isoformat(),
                                   runs_dir=jm.runs_dir)))

    @app.get("/api/geocode")
    def geocode():
        q = request.args.get("q", "").strip()
        if not q:
            return jsonify([])
        try:
            r = requests.get(f"{NOMINATIM}/search", params=dict(q=q, format="json", limit=6), headers=UA, timeout=15)
            r.raise_for_status()
            return jsonify([dict(name=x["display_name"], lat=float(x["lat"]), lon=float(x["lon"])) for x in r.json()])
        except Exception as exc:
            return jsonify(dict(error=f"place search unavailable ({exc.__class__.__name__})")), 502

    @app.get("/api/reverse")
    def reverse():
        try:
            lat, lon = float(request.args["lat"]), float(request.args["lon"])
            r = requests.get(f"{NOMINATIM}/reverse", params=dict(lat=lat, lon=lon, format="json", zoom=10),
                             headers=UA, timeout=15)
            r.raise_for_status()
            js = r.json()
            a = js.get("address", {})
            place = a.get("city") or a.get("town") or a.get("village") or a.get("municipality") or a.get("county")
            region = a.get("state") or a.get("region")
            parts = [p for p in (place, region, a.get("country_code", "").upper() or None) if p]
            return jsonify(dict(name=", ".join(parts) if parts else js.get("display_name", "")))
        except Exception as exc:
            return jsonify(dict(name="", error=f"place lookup unavailable ({exc.__class__.__name__})"))

    @app.post("/api/run")
    def run():
        body = request.get_json(force=True, silent=True) or {}
        over = {k: v for k, v in body.items() if k in EDITABLE}
        try:
            cfg = default_config()
            update_config(cfg, over)
            cfg["LATITUDE"], cfg["LONGITUDE"] = float(cfg["LATITUDE"]), float(cfg["LONGITUDE"])
            cfg["N_HOUSES"] = int(cfg["N_HOUSES"])
            cfg["SEED"] = int(cfg["SEED"])
            cfg["FREQ_MIN"] = int(cfg["FREQ_MIN"])
            if not 1 <= cfg["N_HOUSES"] <= MAX_HOUSES:
                raise ValueError(f"number of houses must be between 1 and {MAX_HOUSES}")
            mem = estimate_memory_gb(cfg)
            if mem > MAX_MEMORY_GB:
                raise ValueError(f"this run needs about {mem:.0f} GB of memory (limit {MAX_MEMORY_GB:.0f} GB); "
                                 "reduce the number of houses or the period, or split it into several runs")
            job = jm.start(cfg)
        except (ValueError, KeyError, TypeError) as exc:
            return jsonify(dict(error=str(exc))), 400
        except RuntimeError as exc:
            return jsonify(dict(error=str(exc))), 409
        return jsonify(dict(id=job.id))

    def _job(job_id):
        job = jm.jobs.get(job_id)
        if job is None:
            abort(404)
        return job

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id):
        return jsonify(_job(job_id).snapshot(int(request.args.get("since", 0))))

    @app.post("/api/jobs/<job_id>/cancel")
    def job_cancel(job_id):
        job = _job(job_id)
        if job.sim is not None:
            job.sim.cancel()
        return jsonify(dict(ok=True))

    @app.get("/api/jobs/<job_id>/files/<path:name>")
    def job_file(job_id, name):
        job = _job(job_id)
        return send_from_directory(job.cfg["OUTPUT_DIR"], name, as_attachment=request.args.get("dl") == "1")

    @app.get("/api/jobs/<job_id>/download.zip")
    def job_zip(job_id):
        job = _job(job_id)
        if job.status != "done":
            abort(409)
        folder = job.cfg["OUTPUT_DIR"]
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(os.listdir(folder)):
                zf.write(os.path.join(folder, f), arcname=os.path.join(os.path.basename(folder), f))
        buf.seek(0)
        return send_file(buf, mimetype="application/zip", as_attachment=True,
                         download_name=os.path.basename(folder) + ".zip")

    return app


def serve(host="127.0.0.1", port=8050, runs_dir="synpv_runs", open_browser=True):
    app = create_app(runs_dir)
    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}"
    print(f"SynPV {__version__} web app running at {url}   (Ctrl+C to stop)")
    print(f"Runs are saved in {os.path.abspath(runs_dir)}")
    if open_browser:
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    import logging
    import flask.cli
    flask.cli.show_server_banner = lambda *a, **k: None      # this is a local, single-user app
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    app.run(host=host, port=port, threaded=True, debug=False, use_reloader=False)

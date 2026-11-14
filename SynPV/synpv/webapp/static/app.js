/* SynPV web app: globe location picker, run control, results. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const FIG_CAPTIONS = {
    "fig1_irradiance.png": "Irradiance components of a representative house on a clear, a variable and a winter day",
    "fig2_output_vs_irradiance.png": "AC power versus irradiance, coloured by cell temperature",
    "fig3_module_temperature.png": "Air and cell temperature on a clear day",
    "fig4_multi_house.png": "Houses ordered along the direction of cloud motion",
    "fig5_profiles.png": "Mean daily profile by month and monthly fleet energy",
    "fig6_fleet_stats.png": "System size, specific yield, and yield versus orientation",
    "fig7_all_houses.png": "Whole fleet: daily profiles, cloud passage, clipping, energy",
    "fig8_variability.png": "Spatial decorrelation, cloud mask and ramp distribution",
    "fig9_snow_soiling.png": "Ground snow, panel snow cover and soiling",
  };
  const FILE_INFO = {
    "pv_output_5min.csv.gz": "Ideal AC power of every house (W)",
    "pv_output_5min.csv": "Ideal AC power of every house (W)",
    "pv_output_5min_measured.csv.gz": "Measured AC power with metering artefacts (W)",
    "pv_output_5min_measured.csv": "Measured AC power with metering artefacts (W)",
    "meter_flags.csv.gz": "Quality flag of every measured sample",
    "meter_flags.csv": "Quality flag of every measured sample",
    "weather_data.csv.gz": "Regional weather, clear sky, cloud cover and motion",
    "weather_data.csv": "Regional weather, clear sky, cloud cover and motion",
    "pv_metadata.csv": "System metadata and indicators of every house",
    "per_house_results.csv": "Compact per-house summary",
    "pv_dataset.npz": "All arrays, including ground-truth variables (NumPy)",
    "config_used.json": "Complete configuration; rerun it to reproduce this dataset",
    "README_dataset.md": "Description of the files and quality flags",
    "summary.json": "Validation metrics and checks",
  };

  const CHECK_NAMES = {
    "GHI closure (sim/source)": "Irradiance closure, simulated over NASA POWER",
    "PR max (must be < 1)": "Highest performance ratio of any house",
    "corr <0.5 km > corr >3 km": "Correlation drop from nearby (<0.5 km) to distant (>3 km) houses",
    "specific yield": "Specific yield",
    "capacity factor (DC)": "Capacity factor (DC)",
    "performance ratio": "Mean performance ratio",
    "snow loss": "Snow loss",
    "soiling loss": "Soiling loss",
    "sim / PVWatts v5": "Ratio to the PVWatts v5 benchmark",
    "cell-air dT @POA>800 (med)": "Cell minus air temperature above 800 W/m\u00b2 (median)",
  };

  let defaults = null;
  let site = null;              // {lat, lon}
  let siteNameTouched = false;
  let presetExtra = {};         // values only valid for the loaded preset location
  let presetActive = false;     // paper preset loaded and the site not moved since
  let job = null;               // {id, since}

  // ------------------------------------------------------------------ globe
  const globeEl = $("globe");
  const globe = Globe({ animateIn: !reduceMotion })(globeEl)
    .globeImageUrl("/static/img/earth-blue-marble.jpg")
    .bumpImageUrl("/static/img/earth-topology.png")
    .backgroundColor("rgba(0,0,0,0)")
    .showAtmosphere(true)
    .atmosphereColor("#8fbfff")
    .atmosphereAltitude(0.17)
    .pointsData([])
    .pointLat("lat").pointLng("lon")
    .pointColor(() => "#f0a93b")
    .pointAltitude(0.015)
    .pointRadius(0.32)
    .pointsMerge(false)
    .ringsData([])
    .ringLat("lat").ringLng("lon")
    .ringColor(() => (t) => `rgba(240,169,59,${Math.max(0, 1 - t)})`)
    .ringMaxRadius(5)
    .ringPropagationSpeed(2.2)
    .ringRepeatPeriod(reduceMotion ? 0 : 1100)
    .onGlobeClick(({ lat, lng }) => chooseSite(lat, lng, { fly: false, lookup: true }))
    .onPointClick((p) => chooseSite(p.lat, p.lon, { fly: false, lookup: false }));

  const controls = globe.controls();
  controls.autoRotate = !reduceMotion;
  controls.autoRotateSpeed = 0.35;
  controls.enableDamping = true;
  globe.pointOfView({ lat: 28, lng: -40, altitude: 2.4 });

  function resizeGlobe() {
    globe.width(globeEl.clientWidth).height(globeEl.clientHeight);
  }
  window.addEventListener("resize", resizeGlobe);
  resizeGlobe();

  // ------------------------------------------------------------------ site
  const fmtLat = (v) => `${Math.abs(v).toFixed(4)}° ${v >= 0 ? "N" : "S"}`;
  const fmtLon = (v) => `${Math.abs(v).toFixed(4)}° ${v >= 0 ? "E" : "W"}`;

  function chooseSite(lat, lon, { fly = true, lookup = true, name = null, preset = false } = {}) {
    lat = Math.max(-90, Math.min(90, +lat));
    lon = ((+lon + 540) % 360) - 180;
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;
    const moved = !site || Math.abs(site.lat - lat) > 1e-6 || Math.abs(site.lon - lon) > 1e-6;
    site = { lat, lon };
    if (moved && !preset && presetActive) {
      // the paper's site-specific values do not apply to a new location
      presetActive = false;
      presetExtra = {};
      ["tmin", "cloudFrom", "alt"].forEach((id) => { $(id).value = ""; });
      siteNameTouched = false;
    }
    controls.autoRotate = false;
    $("lat").value = lat.toFixed(4);
    $("lon").value = lon.toFixed(4);
    $("readout").classList.remove("empty");
    $("latText").textContent = fmtLat(lat);
    $("lonText").textContent = fmtLon(lon);
    globe.pointsData([site]).ringsData([site]);
    if (fly) globe.pointOfView({ lat, lng: lon, altitude: 1.5 }, reduceMotion ? 0 : 1400);
    if (name) setPlace(name, true);
    else if (lookup) reverseLookup(lat, lon);
    updateMeta();
    updateEstimate();
  }

  function setPlace(name, fill) {
    $("placeText").textContent = name || "Unnamed location";
    if (fill && (!siteNameTouched || !$("siteName").value)) {
      $("siteName").value = name || "";
      siteNameTouched = false;
    }
  }

  let reverseTimer = null;
  function reverseLookup(lat, lon) {
    $("placeText").textContent = "Looking up the place name";
    clearTimeout(reverseTimer);
    reverseTimer = setTimeout(async () => {
      try {
        const r = await fetch(`/api/reverse?lat=${lat}&lon=${lon}`);
        const js = await r.json();
        if (!(site && site.lat === lat && site.lon === lon)) return;
        if (js.error) { $("placeText").textContent = "Place name unavailable offline"; return; }
        setPlace(js.name || "Open ocean or remote area", true);
      } catch { setPlace("", false); }
    }, 350);
  }

  function updateMeta() {
    if (!site) return;
    const hemi = site.lat < 0 ? "Southern hemisphere: roofs face north" : "Northern hemisphere: roofs face south";
    const a = Math.abs(site.lat);
    const belt = a < 25 ? "trade easterlies" : a < 65 ? "westerlies" : "polar easterlies";
    $("metaText").textContent = `${hemi}. Clouds move with the ${belt}.`;
  }

  ["lat", "lon"].forEach((id) => $(id).addEventListener("change", () => {
    const lat = parseFloat($("lat").value), lon = parseFloat($("lon").value);
    if (Number.isFinite(lat) && Number.isFinite(lon)) chooseSite(lat, lon, { fly: true, lookup: true });
  }));
  $("siteName").addEventListener("input", () => { siteNameTouched = true; });

  // ------------------------------------------------------------------ search
  async function search() {
    const q = $("q").value.trim();
    const hits = $("hits");
    if (!q) { hits.hidden = true; return; }
    hits.hidden = false;
    hits.innerHTML = '<li class="empty">Searching</li>';
    try {
      const r = await fetch(`/api/geocode?q=${encodeURIComponent(q)}`);
      const js = await r.json();
      if (!r.ok) throw new Error(js.error || "search failed");
      if (!js.length) { hits.innerHTML = '<li class="empty">No place found. Try another name, or click the globe.</li>'; return; }
      hits.innerHTML = "";
      js.forEach((h) => {
        const li = document.createElement("li");
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = h.name;
        b.addEventListener("click", () => {
          hits.hidden = true;
          const short = h.name.split(",").slice(0, 2).join(",").trim();
          chooseSite(h.lat, h.lon, { fly: true, lookup: false, name: short });
        });
        li.appendChild(b);
        hits.appendChild(li);
      });
    } catch (e) {
      hits.innerHTML = `<li class="empty">Place search is unavailable (${e.message}). Enter the coordinates or click the globe.</li>`;
    }
  }
  $("searchBtn").addEventListener("click", search);
  $("q").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); search(); } });

  // ------------------------------------------------------------------ form
  const num = (id) => { const v = $(id).value.trim(); return v === "" ? null : Number(v); };

  function fillDefaults(c) {
    $("start").value = c.START_DATE; $("end").value = c.END_DATE;
    $("nHouses").value = c.N_HOUSES; $("pvShare").value = Math.round(c.PV_OWNERSHIP * 100);
    $("freq").value = String(c.FREQ_MIN); $("area").value = c.NEIGHBOURHOOD_KM; $("seed").value = c.SEED;
    $("kwMed").value = c.SYSTEM_KW_MEDIAN; $("vmax").value = String(c.VDC_ABS_MAX);
    $("dcacLo").value = c.DC_AC_RATIO_RANGE[0]; $("dcacHi").value = c.DC_AC_RATIO_RANGE[1];
    $("tiltLo").value = c.TILT_RANGE[0]; $("tiltHi").value = c.TILT_RANGE[1];
    $("albedo").value = c.ALBEDO;
    $("tmin").value = typeof c.DESIGN_T_MIN === "number" ? c.DESIGN_T_MIN : "";
    $("cloudFrom").value = typeof c.CLOUD_PREVAILING_FROM_DEG === "number" ? c.CLOUD_PREVAILING_FROM_DEG : "";
    $("alt").value = typeof c.ALTITUDE_M === "number" ? c.ALTITUDE_M : "";
  }

  function days() {
    const a = new Date($("start").value), b = new Date($("end").value);
    return Math.round((b - a) / 86400000) + 1;
  }

  function updateEstimate() {
    const d = days(), n = num("nHouses") || 0, f = Number($("freq").value);
    if (!(d > 0) || !(n > 0)) { $("estimate").textContent = ""; return; }
    const steps = d * 1440 / f;
    const gb = steps * n * 4 * 26 / 1e9;
    const min = 0.4 + 2.6 * (steps * n) / (105408 * 100);
    const t = min < 1.5 ? "about a minute" : `about ${Math.round(min)} minutes`;
    $("estimate").textContent = `${n} houses, ${d} days at ${f}-min resolution: ${t} and ${gb < 1 ? gb.toFixed(1) : Math.round(gb)} GB of memory, plus the NASA POWER download.`;
  }
  ["start", "end", "nHouses", "freq"].forEach((id) => $(id).addEventListener("input", updateEstimate));

  function collect() {
    const lat = num("lat"), lon = num("lon");
    if (lat === null || lon === null) throw new Error("Choose a site: click the globe, search a place, or type the coordinates.");
    if (!$("start").value || !$("end").value) throw new Error("Set the start and end dates.");
    if (days() < 1) throw new Error("The end date must be on or after the start date.");
    if ($("start").value < defaults.min_date) throw new Error(`NASA POWER hourly data start on ${defaults.min_date}.`);
    if ($("end").value > defaults.max_date) throw new Error(`NASA POWER data are available up to about ${defaults.max_date}.`);
    const lo = num("dcacLo"), hi = num("dcacHi"), tl = num("tiltLo"), th = num("tiltHi");
    if (lo > hi) throw new Error("The minimum DC/AC ratio must not exceed the maximum.");
    if (tl > th) throw new Error("The minimum tilt must not exceed the maximum.");
    const meter = $("meter").checked;
    const c = {
      LATITUDE: lat, LONGITUDE: lon,
      SITE_NAME: $("siteName").value.trim() || "auto",
      START_DATE: $("start").value, END_DATE: $("end").value,
      FREQ_MIN: Number($("freq").value), N_HOUSES: num("nHouses"),
      PV_OWNERSHIP: Math.min(1, Math.max(0.01, (num("pvShare") ?? 65) / 100)),
      NEIGHBOURHOOD_KM: num("area"), SEED: num("seed") ?? 42,
      SYSTEM_KW_MEDIAN: num("kwMed"), VDC_ABS_MAX: Number($("vmax").value),
      DC_AC_RATIO_RANGE: [lo, hi], TILT_RANGE: [tl, th], ALBEDO: num("albedo"),
      DESIGN_T_MIN: num("tmin") ?? "auto",
      CLOUD_PREVAILING_FROM_DEG: num("cloudFrom") ?? "auto",
      ALTITUDE_M: num("alt") ?? "auto",
      PVWATTS_BENCHMARK: $("pvwatts").checked, EXPORT_GROUND_TRUTH: $("gt").checked,
      ALLOW_SYNTHETIC_WEATHER: $("synth").checked,
    };
    if (!meter) Object.assign(c, { OUTAGE_EVENTS_PER_YEAR: 0, GAP_EVENTS_PER_YEAR: 0,
      EXPORT_LIMIT_FRACTION_OF_HOUSES: 0, CLOCK_OFFSET_FRACTION: 0, METER_NOISE_REL: 0 });
    return Object.assign(c, presetExtra);
  }

  function showFormError(msg) {
    const el = $("formError");
    el.textContent = msg || "";
    el.hidden = !msg;
  }

  $("paperBtn").addEventListener("click", () => {
    const p = defaults.paper;
    fillDefaults(Object.assign({}, defaults.config, p));
    $("siteName").value = p.SITE_NAME; siteNameTouched = true;
    chooseSite(p.LATITUDE, p.LONGITUDE, { fly: true, lookup: false, name: p.SITE_NAME, preset: true });
    presetExtra = { TIMEZONE: p.TIMEZONE, EQUATOR_FACING: p.EQUATOR_FACING };
    presetActive = true;
    showFormError("");
  });

  // ------------------------------------------------------------------ run
  $("setup").addEventListener("submit", async (e) => {
    e.preventDefault();
    let cfg;
    try { cfg = collect(); } catch (err) { showFormError(err.message); return; }
    showFormError("");
    $("runBtn").disabled = true;
    try {
      const r = await fetch("/api/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(cfg) });
      const js = await r.json();
      if (!r.ok) throw new Error(js.error || `server error ${r.status}`);
      job = { id: js.id, since: 0, name: cfg.SITE_NAME };
      startProgress(cfg);
      poll();
    } catch (err) {
      showFormError(err.message);
    } finally {
      $("runBtn").disabled = false;
    }
  });

  function startProgress(cfg) {
    $("setup").hidden = true;
    $("progress").hidden = false;
    $("runError").hidden = true;
    $("cancelBtn").hidden = false;
    $("newBtn").hidden = true;
    $("progTitle").textContent = "Generating the dataset";
    const name = cfg.SITE_NAME === "auto" ? `${fmtLat(cfg.LATITUDE)}, ${fmtLon(cfg.LONGITUDE)}` : cfg.SITE_NAME;
    $("progSub").textContent = `${name}, ${cfg.START_DATE} to ${cfg.END_DATE}, ${cfg.N_HOUSES} houses`;
    $("log").textContent = "";
    $("barFill").style.width = "0%";
    const ol = $("steps");
    ol.innerHTML = "";
    defaults.steps.forEach((s) => { const li = document.createElement("li"); li.textContent = s; ol.appendChild(li); });
  }

  async function poll() {
    if (!job) return;
    let js;
    try {
      const r = await fetch(`/api/jobs/${job.id}?since=${job.since}`);
      js = await r.json();
    } catch { setTimeout(poll, 1500); return; }
    job.since = js.log_next;
    if (js.log.length) {
      const pre = $("log");
      const atEnd = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 8;
      pre.textContent += js.log.join("\n") + "\n";
      if (atEnd) pre.scrollTop = pre.scrollHeight;
    }
    const pct = Math.round(100 * js.step / js.n_steps);
    $("barFill").style.width = `${pct}%`;
    $("bar").setAttribute("aria-valuenow", pct);
    [...$("steps").children].forEach((li, i) => {
      li.className = i < js.step ? "done" : (i === js.step && js.status === "running" ? "active" : "");
    });
    if (js.status === "running" || js.status === "queued") { setTimeout(poll, 700); return; }
    $("cancelBtn").hidden = true;
    $("newBtn").hidden = false;
    if (js.status === "done") {
      $("progTitle").textContent = "Dataset ready";
      $("progSub").textContent = `Finished in ${fmtDur(js.elapsed_s)}. Saved in ${js.output_dir}`;
      showResults(js);
    } else if (js.status === "cancelled") {
      $("progTitle").textContent = "Cancelled";
      $("progSub").textContent = "No dataset was written. Change the settings and generate again.";
    } else {
      $("progTitle").textContent = "The run stopped";
      $("runError").textContent = js.error || "Unknown error. See the console output.";
      $("runError").hidden = false;
      $("logBox").open = true;
    }
  }

  $("cancelBtn").addEventListener("click", async () => {
    if (!job) return;
    $("cancelBtn").disabled = true;
    try { await fetch(`/api/jobs/${job.id}/cancel`, { method: "POST" }); } finally { $("cancelBtn").disabled = false; }
  });
  $("newBtn").addEventListener("click", () => {
    $("progress").hidden = true;
    $("setup").hidden = false;
    $("results").hidden = true;
  });
  $("backGlobe").addEventListener("click", () => { $("results").hidden = true; resizeGlobe(); });

  const fmtDur = (s) => (s < 90 ? `${Math.round(s)} s` : `${(s / 60).toFixed(1)} min`);

  // ------------------------------------------------------------------ results
  function fmt(v, d = 0) {
    if (v === null || v === undefined || Number.isNaN(v)) return "n/a";
    return Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  }

  function showResults(js) {
    const m = js.metrics, cfg = js.config;
    const base = `/api/jobs/${js.id}/files/`;
    $("resTitle").textContent = `Dataset ready: ${m.site_name}`;
    const full = (new Date(m.end_date) - new Date(m.start_date)) / 86400000 >= 349;
    $("resSub").textContent =
      `${m.n_pv_houses} of ${m.n_houses} houses with PV (${fmt(m.dc_capacity_kw, 1)} kWdc), ${m.start_date} to ${m.end_date} at ` +
      `${cfg.FREQ_MIN}-min resolution. Weather from ${m.weather_source}; elevation ${fmt(cfg.ALTITUDE_M)} m, time zone ${cfg.TIMEZONE}.`;
    $("zipLink").href = `/api/jobs/${js.id}/download.zip`;

    const kpi = [
      ["Fleet energy", m.energy_kwh >= 1e4 ? fmt(m.energy_kwh / 1000, 1) : fmt(m.energy_kwh), m.energy_kwh >= 1e4 ? "MWh" : "kWh"],
      [full ? "Specific yield" : "Specific yield (period)", fmt(m.specific_yield_mean), "kWh/kW"],
      ["Performance ratio", fmt(m.performance_ratio_mean * 100, 1), "%"],
      ["Ratio to PVWatts v5", fmt(m.pvwatts_ratio_mean, 3), ""],
      ["Irradiance closure", fmt(m.ghi_closure, 3), ""],
      ["Peak fleet power", fmt(m.peak_fleet_kw), `of ${fmt(m.ac_capacity_kw)} kW`],
    ];
    $("kpis").innerHTML = kpi.map(([k, v, u]) => `<div><dt>${k}</dt><dd>${v}${u ? `<small>${u}</small>` : ""}</dd></div>`).join("");

    $("checks").innerHTML = (js.checks || []).map((c) => {
      const val = c.value === null ? "n/a" : `${fmt(c.value, 2)}${c.unit ? " " + c.unit : ""}`;
      return `<li class="${c.ok ? "ok" : "warn"}"><span class="mark" aria-label="${c.ok ? "passed" : "outside the expected range"}">${c.ok ? "✓" : "!"}</span>` +
             `<span>${CHECK_NAMES[c.name] || c.name}</span><span class="val">${val} (expected ${c.low} to ${c.high})</span></li>`;
    }).join("") || "<li>No checks were run.</li>";

    const files = js.files.filter((f) => !f.endsWith(".png"));
    $("files").innerHTML = files.map((f) =>
      `<li><a href="${base}${encodeURIComponent(f)}?dl=1">${f}</a><span>${FILE_INFO[f] || ""}</span></li>`).join("");

    const figs = js.files.filter((f) => f.endsWith(".png"));
    $("figs").innerHTML = "";
    figs.forEach((f) => {
      const fig = document.createElement("figure");
      const b = document.createElement("button");
      b.type = "button";
      const img = new Image();
      img.loading = "lazy";
      img.src = base + f;
      img.alt = FIG_CAPTIONS[f] || f;
      b.appendChild(img);
      b.addEventListener("click", () => {
        $("lbImg").src = img.src; $("lbImg").alt = img.alt; $("lbCap").textContent = img.alt;
        $("lightbox").showModal();
      });
      const cap = document.createElement("figcaption");
      cap.textContent = FIG_CAPTIONS[f] || f;
      fig.append(b, cap);
      $("figs").appendChild(fig);
    });
    $("results").hidden = false;
    $("results").scrollTop = 0;
  }
  $("lightbox").addEventListener("click", (e) => { if (e.target === $("lightbox")) $("lightbox").close(); });

  // ------------------------------------------------------------------ init
  (async () => {
    try {
      defaults = await (await fetch("/api/defaults")).json();
    } catch {
      showFormError("Could not reach the SynPV server. Restart it with: synpv app");
      return;
    }
    fillDefaults(defaults.config);
    $("start").min = $("end").min = defaults.min_date;
    $("start").max = $("end").max = defaults.max_date;
    updateEstimate();
  })();
})();

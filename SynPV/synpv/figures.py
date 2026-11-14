"""Diagnostic figures of a SynPV run.

Figures are drawn with the object-oriented matplotlib API (no pyplot), so they
can be produced safely from background threads such as the web application.
"""
from __future__ import annotations

import os

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

STYLE = {
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 200, "figure.dpi": 110,
}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    return os.path.basename(path)


def _hour_axis(ax):
    ax.set_xlim(0, 24)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xlabel("Local time (h)")


def make_figures(sim) -> list[str]:
    """Write the nine diagnostic figures; returns the file names."""
    with matplotlib.rc_context(STYLE):
        return _make(sim)


def _make(sim):
    cfg, fleet = sim.cfg, sim.fleet
    t_local, solar_idx = sim.t_local, sim.solar_idx
    ghi_cs, ghi_house, dni_house = sim.ghi_cs, sim.ghi_house, sim.dni_house
    poa_all, eff_all, pac_all, tcell_all = sim.poa_all, sim.eff_all, sim.pac_all, sim.tcell_all
    inv_w, cap_kw, fleet_kw = sim.inv_w, sim.cap_kw, sim.fleet_kw
    out = []

    # ---- representative house and days (hemisphere aware) -------------------
    ref = solar_idx[int(np.argmax(cap_kw[solar_idx]))]
    hid = fleet.at[ref, "house_id"]
    dates_local = pd.Series(t_local.date)
    hod = (t_local.hour + t_local.minute / 60).to_numpy()
    day_e = sim.df_fleet["p_kw"].groupby(t_local.date).sum()
    day_var = pd.Series(np.abs(np.diff(fleet_kw, prepend=0)), index=t_local).groupby(t_local.date).sum()
    summer_m, winter_m = ((12, 1, 2), (6, 7)) if sim.SOUTHERN else ((6, 7, 8), (12, 1))
    summer = [d for d in day_e.index if d.month in summer_m] or list(day_e.index)
    winter = [d for d in day_e.index if d.month in winter_m] or list(day_e.index)
    clear_day = day_e.loc[summer].idxmax()
    mixed = day_e.loc[summer]
    mixed = mixed[(mixed > mixed.quantile(0.35)) & (mixed < mixed.quantile(0.85))]
    cloudy_day = (day_var.loc[mixed.index] / mixed).idxmax() if len(mixed) else clear_day
    winter_day = day_e.loc[winter].idxmax()
    M = lambda d: (dates_local == d).to_numpy()
    m_clr, m_cld, m_win = M(clear_day), M(cloudy_day), M(winter_day)

    # ---- Fig 1: irradiance components ------------------------------------------
    fig = Figure(figsize=(12, 3.4))
    ax = fig.subplots(1, 3, sharey=False)
    for a, m, lbl in [(ax[0], m_clr, f"(a) Clear day, {clear_day}"), (ax[1], m_cld, f"(b) Variable day, {cloudy_day}"),
                      (ax[2], m_win, f"(c) Winter day, {winter_day}")]:
        a.plot(hod[m], ghi_cs[m], "k:", lw=1, label="Clear-sky GHI")
        a.plot(hod[m], ghi_house[m, ref], lw=1.1, label="GHI")
        a.plot(hod[m], dni_house[m, ref], lw=0.9, alpha=0.85, label="DNI")
        a.plot(hod[m], poa_all[m, ref], lw=1.1, label="POA")
        a.set_title(lbl, loc="left")
        _hour_axis(a)
    ax[0].set_ylabel("Irradiance (W/m$^2$)")
    ax[0].legend(loc="upper left")
    fig.suptitle(f"Irradiance components of {hid} (tilt {fleet.at[ref, 'tilt']:.0f}$^\\circ$, "
                 f"azimuth {fleet.at[ref, 'azimuth']:.0f}$^\\circ$)", y=1.02, fontsize=10)
    out.append(_save(fig, sim._out("fig1_irradiance.png")))

    # ---- Fig 2: output vs irradiance -------------------------------------------
    fig = Figure(figsize=(10, 3.4))
    ax = fig.subplots(1, 2)
    ax[0].plot(hod[m_cld], eff_all[m_cld, ref], color="tab:orange", lw=1, label="Effective irradiance")
    a2 = ax[0].twinx()
    a2.plot(hod[m_cld], pac_all[m_cld, ref] / 1000, color="tab:blue", lw=1, label="AC power")
    a2.grid(False)
    a2.spines["right"].set_visible(True)
    ax[0].set_ylabel("Effective irradiance (W/m$^2$)")
    a2.set_ylabel("AC power (kW)")
    ax[0].set_title(f"(a) Variable day, {cloudy_day}", loc="left")
    _hour_axis(ax[0])
    sub = poa_all[:, ref] > 0
    sc = ax[1].scatter(poa_all[sub, ref], pac_all[sub, ref] / 1000, s=2, alpha=0.15,
                       c=tcell_all[sub, ref], cmap="coolwarm", rasterized=True)
    ax[1].axhline(inv_w[ref] / 1000, color="red", ls="--", lw=1, label=r"Inverter rating $P_{\mathrm{ac},0}$")
    ax[1].set_xlabel("POA irradiance (W/m$^2$)")
    ax[1].set_ylabel("AC power (kW)")
    ax[1].legend(loc="lower right")
    ax[1].set_title("(b) Transfer function", loc="left")
    cb = fig.colorbar(sc, ax=ax[1], pad=0.01)
    cb.set_label("Cell temperature ($^\\circ$C)")
    cb.solids.set_alpha(1)
    out.append(_save(fig, sim._out("fig2_output_vs_irradiance.png")))

    # ---- Fig 3: temperatures ----------------------------------------------------
    fig = Figure(figsize=(7, 3))
    ax = fig.subplots()
    ax.plot(hod[m_clr], sim.temp_air[m_clr], lw=1.2, label="Air temperature")
    ax.plot(hod[m_clr], tcell_all[m_clr, ref], lw=1.2, label=f"Cell temperature ({cfg['THERMAL_MODEL']})")
    ax.axhline(25, color="grey", ls=":", lw=1, label="STC (25 $^\\circ$C)")
    ax.set_ylabel("Temperature ($^\\circ$C)")
    _hour_axis(ax)
    ax.legend()
    ax.set_title(f"Air and cell temperature of {hid}, {clear_day}", loc="left")
    out.append(_save(fig, sim._out("fig3_module_temperature.png")))

    # ---- Fig 4: houses ordered upwind -> downwind ----------------------------------
    fig = Figure(figsize=(10, 3.4))
    ax = fig.subplots()
    _uu, _vv = sim.u_km[m_cld].mean(), sim.v_km[m_cld].mean()
    _proj = (sim.xh * _uu + sim.yh * _vv) / np.hypot(_uu, _vv)
    far = solar_idx[np.argsort(_proj[solar_idx])]
    pick = np.unique(np.r_[far[:2], far[len(far) // 2: len(far) // 2 + 2], far[-2:]])
    _tr = cfg["NEIGHBOURHOOD_KM"] / np.hypot(_uu, _vv)
    for h in pick:
        ax.plot(hod[m_cld], pac_all[m_cld, h] / inv_w[h], lw=0.9,
                label=f"{fleet.at[h, 'house_id']}, $s$ = {_proj[h]:+.1f} km")
    ax.set_ylabel(r"$P_{\mathrm{ac}}/P_{\mathrm{ac},0}$")
    _hour_axis(ax)
    ax.legend(ncol=3, loc="upper left")
    ax.set_title(f"Houses ordered along the cloud motion ($s$), {cloudy_day}; transit across the "
                 f"neighbourhood ~{_tr:.1f} min", loc="left")
    out.append(_save(fig, sim._out("fig4_multi_house.png")))

    # ---- Fig 5: daily and seasonal profiles ---------------------------------------
    df = sim.df_fleet
    mh = df.groupby([df.index.month, df.index.hour])["p_kw"].mean().unstack(0)
    fig = Figure(figsize=(10, 3.4))
    ax = fig.subplots(1, 2)
    cmap = matplotlib.colormaps["viridis"]
    for mth in mh.columns:
        ax[0].plot(mh.index, mh[mth], color=cmap((mth - 1) / 11), lw=1, label=f"M{mth}")
    _hour_axis(ax[0])
    ax[0].set_ylabel("Mean fleet power (kW)")
    ax[0].legend(fontsize=6, ncol=3)
    ax[0].set_title("(a) Mean daily profile by month", loc="left")
    mm = sim.monthly_mwh
    ax[1].bar(range(len(mm)), mm.to_numpy(), color="tab:orange")
    ax[1].set_xticks(range(len(mm)), mm.index, rotation=45, fontsize=7)
    ax[1].set_ylabel("Fleet energy (MWh)")
    ax[1].set_title("(b) Monthly fleet energy", loc="left")
    out.append(_save(fig, sim._out("fig5_profiles.png")))

    # ---- Fig 6: fleet statistics ------------------------------------------------------
    fig = Figure(figsize=(12, 3.4))
    ax = fig.subplots(1, 3)
    ax[0].hist(cap_kw[solar_idx], bins=15, color="tab:blue", edgecolor="w")
    ax[0].set_xlabel("DC capacity (kW$_{\\mathrm{dc}}$)")
    ax[0].set_ylabel("Number of houses")
    ax[0].set_title("(a) System size", loc="left")
    ax[1].hist(sim.sy[solar_idx], bins=15, color="tab:green", edgecolor="w")
    ax[1].set_xlabel("Specific yield (kWh/kW$_{\\mathrm{dc}}$)")
    ax[1].set_ylabel("Number of houses")
    ax[1].set_title("(b) Specific yield", loc="left")
    sc = ax[2].scatter(fleet["azimuth"].to_numpy()[solar_idx], sim.sy[solar_idx], c=sim.tilt_v[solar_idx],
                       cmap="viridis", s=18, edgecolor="k", linewidth=0.3)
    ax[2].set_xlabel("Azimuth ($^\\circ$)")
    ax[2].set_ylabel("Specific yield (kWh/kW$_{\\mathrm{dc}}$)")
    ax[2].set_title("(c) Yield versus orientation", loc="left")
    cb = fig.colorbar(sc, ax=ax[2], pad=0.01)
    cb.set_label("Tilt ($^\\circ$)")
    out.append(_save(fig, sim._out("fig6_fleet_stats.png")))

    # ---- Fig 7: full fleet --------------------------------------------------------------
    fig = Figure(figsize=(12, 7))
    axes = fig.subplots(2, 2)
    prof = pd.DataFrame(pac_all[:, solar_idx], index=t_local)
    mean_day = prof.groupby([prof.index.hour, prof.index.minute]).mean().to_numpy()
    hod_m = np.arange(mean_day.shape[0]) * sim.DT_MIN / 60
    cmin, cmax = cap_kw[solar_idx].min(), cap_kw[solar_idx].max()
    for j, h in enumerate(solar_idx):
        axes[0, 0].plot(hod_m, mean_day[:, j] / 1000, lw=0.7, alpha=0.6,
                        color=cmap((cap_kw[h] - cmin) / (cmax - cmin + 1e-9)))
    axes[0, 0].set_title(f"(a) Mean daily profile of the {sim.N_SOLAR} PV houses", loc="left")
    axes[0, 0].set_ylabel("AC power (kW)")
    _hour_axis(axes[0, 0])
    order = solar_idx[np.argsort(sim.xh[solar_idx])]
    im = axes[0, 1].imshow((pac_all[m_cld][:, order] / inv_w[order]).T, aspect="auto", cmap="inferno",
                           origin="lower", extent=[hod[m_cld].min(), hod[m_cld].max() + sim.DT_MIN / 60, 0, sim.N_SOLAR])
    axes[0, 1].grid(False)
    fig.colorbar(im, ax=axes[0, 1], pad=0.01).set_label(r"$P_{\mathrm{ac}}/P_{\mathrm{ac},0}$")
    axes[0, 1].set_title(f"(b) All PV houses on {cloudy_day}, sorted west to east", loc="left")
    axes[0, 1].set_ylabel("House (west $\\rightarrow$ east)")
    _hour_axis(axes[0, 1])
    axes[1, 0].scatter(cap_kw[solar_idx], pac_all.max(axis=0)[solar_idx] / 1000, c="tab:blue", s=24,
                       edgecolor="k", linewidth=0.3, label="Peak AC power", zorder=3)
    axes[1, 0].scatter(cap_kw[solar_idx], inv_w[solar_idx] / 1000, marker="_", s=90, color="red",
                       label=r"Inverter rating $P_{\mathrm{ac},0}$")
    axes[1, 0].plot([cmin, cmax], [cmin, cmax], "k--", lw=0.8, label=r"$P_{\mathrm{ac}} = P_{\mathrm{dc}}$")
    axes[1, 0].legend(loc="lower right")
    axes[1, 0].set_xlabel("DC capacity (kW$_{\\mathrm{dc}}$)")
    axes[1, 0].set_ylabel("AC power (kW$_{\\mathrm{ac}}$)")
    axes[1, 0].set_title("(c) Peak AC power versus capacity", loc="left")
    srt = solar_idx[np.argsort(sim.energy_kwh[solar_idx])]
    axes[1, 1].bar(range(len(srt)), sim.energy_kwh[srt], color="tab:orange")
    axes[1, 1].set_xlabel("House (sorted by energy)")
    axes[1, 1].set_ylabel("Energy over the period (kWh)")
    axes[1, 1].set_title("(d) Energy per house", loc="left")
    out.append(_save(fig, sim._out("fig7_all_houses.png")))

    # ---- Fig 8: spatio-temporal variability --------------------------------------------
    fig = Figure(figsize=(13, 3.8))
    ax = fig.subplots(1, 3)
    ax[0].scatter(sim.dist_pairs, sim.corr_pairs, s=2, alpha=0.2, rasterized=True, label="House pairs")
    ax[0].plot((sim.bins[:-1] + sim.bins[1:]) / 2, sim.corr_by_d, "r-o", ms=4, label="Binned mean")
    ax[0].set_xlabel("Separation (km)")
    ax[0].set_ylabel(r"Correlation of 5-min $\Delta k_t$")
    ax[0].legend(loc="upper right")
    ax[0].set_title("(a) Spatial decorrelation", loc="left")
    idx_cld = np.flatnonzero(m_cld)
    tsnap = idx_cld[len(idx_cld) // 2]
    sc = ax[1].scatter(sim.xh, sim.yh, c=sim.cloud_mask[tsnap], cmap="Greys", vmin=0, vmax=1, s=40,
                       edgecolor="k", linewidth=0.5)
    half = cfg["NEIGHBOURHOOD_KM"] / 2
    ax[1].annotate("", xy=(sim.u_km[tsnap] / np.hypot(sim.u_km[tsnap], sim.v_km[tsnap]) * half * 0.9,
                           sim.v_km[tsnap] / np.hypot(sim.u_km[tsnap], sim.v_km[tsnap]) * half * 0.9),
                   xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="tab:blue", lw=1.5))
    ax[1].set_aspect("equal")
    ax[1].set_xlabel("$x$ (km)")
    ax[1].set_ylabel("$y$ (km)")
    ax[1].set_title(f"(b) Cloud mask, {t_local[tsnap]:%Y-%m-%d %H:%M} (arrow: motion)", loc="left")
    fig.colorbar(sc, ax=ax[1], pad=0.01).set_label("Cloud mask")
    sel = sim.sel
    ax[2].hist(np.diff(ghi_house[sel][:, ref] / np.maximum(ghi_cs[sel], 1)), bins=120, log=True, alpha=0.6,
               label="Single house")
    ax[2].hist(np.diff(ghi_house[sel][:, solar_idx].mean(axis=1) / np.maximum(ghi_cs[sel], 1)), bins=120,
               log=True, alpha=0.6, label="Fleet mean")
    ax[2].set_xlabel(r"5-min change in clear-sky index $\Delta k_t$")
    ax[2].set_ylabel("Count")
    ax[2].legend()
    ax[2].set_title("(c) Ramp distribution", loc="left")
    out.append(_save(fig, sim._out("fig8_variability.png")))

    # ---- Fig 9: snow and soiling --------------------------------------------------------
    fig = Figure(figsize=(11, 5))
    ax = fig.subplots(2, 1, sharex=True)
    tl = t_local.tz_localize(None)
    ax[0].plot(tl, sim.ground_depth_cm, color="tab:blue", lw=1, label="Ground snow depth (cm)")
    ax[0].plot(tl, sim.snow_cov[:, ref] * 100, color="k", lw=0.7, label=f"Panel snow coverage of {hid} (%)")
    ax[0].set_ylabel("Snow")
    ax[0].legend(loc="upper right")
    ax[0].set_title("(a) Snow", loc="left")
    ax[1].plot(tl, sim.soiling_loss[:, solar_idx].mean(axis=1) * 100, color="tab:brown", lw=1)
    ax[1].set_ylabel("Mean soiling loss (%)")
    ax[1].set_title("(b) Soiling", loc="left")
    out.append(_save(fig, sim._out("fig9_snow_soiling.png")))
    return out

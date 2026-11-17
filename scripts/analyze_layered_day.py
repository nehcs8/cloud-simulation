"""
Layer statistics of layered_day runs from their (5-min) snapshots: per height band — low (Cu, < 3.3 km), mid (layer A,
3.3–7.5 km), high (cirrus, ≥ 7.5 km) — cloud cover, visible cover (column optical depth τ > 0.1), median τ of the cloudy
columns, patchiness (coefficient of variation of τ over the cloudy columns), depth of levels with cloud fraction > 5 %,
ice-phase fraction, w standard deviation inside the band; plus falling-ice diagnostics (ice mass in the dry slot 6–8.3 km
relative to the high band) and surface precipitation from diag.csv. Optical depth as in the renderer: κ 150 m²/kg liquid,
55 ice (α(T) split, θ₀ from profiles.npz).

Run: ./venv/bin/python3 scripts/analyze_layered_day.py RUN_DIR [RUN_DIR …] [--t0 300 --t1 480] [--series]
Output: one summary line per run for the window [t0, t1] (min); --series prints every 30 min as well.
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualize'))
from render_clouds import _reference, _phases, KAPPA_LIQ, KAPPA_ICE   # noqa: E402

BANDS = {"low": (0.0, 3300.0), "mid": (3300.0, 7500.0), "high": (7500.0, 12500.0)}
Q_CLOUD = 1e-5
W_LEVELS = {"low": 1000.0, "mid": 4500.0, "high": 9300.0}


def snapshot_stats(path, ref, z, dz):
    d = np.load(path)
    ql, qi, _ = _phases(d, ref)
    rho = ref["rho0"]
    w = d["w"]
    out = {"t": float(d["t"]) / 60.0}
    for name, (z0, z1) in BANDS.items():
        k = (z >= z0) & (z < z1)
        q = ql[:, :, k] + qi[:, :, k]
        cloudy = q > Q_CLOUD
        col = cloudy.any(axis=2)
        tau = (KAPPA_LIQ * ql[:, :, k] + KAPPA_ICE * qi[:, :, k]) @ (rho[k] * dz[k])
        cf = cloudy.mean(axis=(0, 1))
        lev = np.nonzero(cf > 0.05)[0]
        mass = (q @ (rho[k] * dz[k])).sum()
        kw = np.argmin(abs(z - W_LEVELS[name]))
        out[name] = dict(
            cover=col.mean(), vis=(tau > 0.1).mean(),
            tau=np.median(tau[col]) if col.any() else 0.0,
            cv=tau[col].std() / tau[col].mean() if col.sum() > 3 and tau[col].mean() > 0 else np.nan,
            depth=dz[k][lev].sum() if len(lev) else 0.0,
            base=z[k][lev[0]] if len(lev) else np.nan, top=z[k][lev[-1]] if len(lev) else np.nan,
            ice=(qi[:, :, k] @ (rho[k] * dz[k])).sum() / mass if mass > 0 else np.nan,
            wstd=w[:, :, kw].std(), path=mass / q[:, :, 0].size * 1e3)            # g/m² domain mean
    k_slot = (z >= 6000.0) & (z < 8300.0)
    k_hi = z >= 7500.0
    ice_slot = (qi[:, :, k_slot] @ (rho[k_slot] * dz[k_slot])).sum()
    ice_hi = (qi[:, :, k_hi] @ (rho[k_hi] * dz[k_hi])).sum()
    out["fall"] = ice_slot / ice_hi if ice_hi > 0 else np.nan
    return out


def diag_precip(run, t0, t1):
    rows = list(csv.DictReader(open(os.path.join(run, "diag.csv"))))
    p = [float(r["precip_mmhr"]) for r in rows if t0 <= float(r["t"]) / 60 <= t1]
    wm = [float(r["w_max"]) for r in rows if t0 <= float(r["t"]) / 60 <= t1]
    return np.mean(p) if p else np.nan, np.max(wm) if wm else np.nan


def label(run):
    b = os.path.basename(run.rstrip("/"))
    return b.split("_ld2")[-1] if "_ld2" in b else b[-24:]


def fmt(s, name):
    b = s[name]
    return (f"{100*b['cover']:4.0f} {100*b['vis']:4.0f} {b['tau']:5.2f} {b['cv']:4.1f} {b['depth']/1e3:4.1f} "
            f"{b['wstd']:4.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--t0", type=float, default=300.0)
    ap.add_argument("--t1", type=float, default=480.0)
    ap.add_argument("--series", action="store_true")
    a = ap.parse_args()
    hdr = "cov  vis   tau   cv  dep  wsd"
    print(f"{'run':5} {'t':>5} | low  {hdr} | mid  {hdr} | high {hdr} | fall  ice_hi  P mm/h  wmax")
    for run in a.runs:
        snaps = sorted(glob.glob(os.path.join(run, "snapshots", "snap_*.npz")))
        d0 = np.load(snaps[0])
        z, dz = d0["z"], np.diff(d0["z_face"])
        eff = {}
        for line in open(os.path.join(run, "effective_config.txt")):
            k, _, v = line.partition(" = ")
            eff[k] = v.strip()
        ref = _reference(run, z, eff)
        stats = [snapshot_stats(p, ref, z, dz) for p in snaps]
        win = [s for s in stats if a.t0 <= s["t"] <= a.t1]
        if a.series:
            for s in stats:
                if s["t"] % 30 == 0:
                    print(f"{label(run):5} {s['t']:5.0f} | {fmt(s, 'low')} | {fmt(s, 'mid')} | {fmt(s, 'high')} | "
                          f"{s['fall']:5.2f}  {s['high']['ice']:5.2f}")
        mean = {"t": np.nan, "fall": np.nanmean([s["fall"] for s in win])}
        for name in BANDS:
            mean[name] = {k: np.nanmean([s[name][k] for s in win]) for k in win[0][name]}
        p, wm = diag_precip(run, a.t0, a.t1)
        first = {n: next((s["t"] for s in stats if s[n]["cover"] > 0.05), np.nan) for n in ("mid", "high")}
        print(f"{label(run):5} {'mean':>5} | {fmt(mean, 'low')} | {fmt(mean, 'mid')} | {fmt(mean, 'high')} | "
              f"{mean['fall']:5.2f}  {mean['high']['ice']:5.2f}  {p:6.3f}  {wm:4.1f}"
              f"   (mid >5% from {first['mid']:.0f} min, high from {first['high']:.0f} min;"
              f" high {mean['high']['base']/1e3:.1f}–{mean['high']['top']/1e3:.1f} km, path {mean['high']['path']:.1f} g/m²)",
              flush=True)


if __name__ == "__main__":
    main()

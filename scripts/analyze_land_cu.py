"""
Summarise a land_cu run against the Task 10 targets.

Usage: python scripts/analyze_land_cu.py runs/<run_dir> [t_start_s] [t_end_s]
       (default window: second hour, 3600–7200 s)

From diag.csv (every OUTPUT_EVERY):
  first cloud time, mean cloud_frac / CB / CT / w_max, surface precip, cloud_persistence.
  Note: diag CB/CT are domain extremes (lowest/highest level with any qc > 0).
From snapshots (qc > 1e-5 per column):
  typical cloud base/top = median over cloudy columns of each column's lowest/highest
  cloudy level, plus 5th/95th percentiles.
"""
import csv
import glob
import os
import sys
import numpy as np

QC_THRESH = 1e-5   # kg/kg, same threshold as the cloud mask in simulation.py

run = sys.argv[1]
t0 = float(sys.argv[2]) if len(sys.argv) > 2 else 3600.0
t1 = float(sys.argv[3]) if len(sys.argv) > 3 else 7200.0

rows = list(csv.DictReader(open(os.path.join(run, "diag.csv"))))
col = lambda k: np.array([float(r[k]) for r in rows])
t = col("t")
win = (t >= t0 - 1e-6) & (t <= t1 + 1e-6)

cf, cb, ct, wmax, prec, pers = (col(k) for k in
                                ("cloud_frac", "CB", "CT", "w_max", "precip_mmhr", "cloud_persistence"))
cloudy = cf > 0

print(f"Run: {run}")
print(f"Diagnostics: {len(t)} rows, t = {t[0]:.0f}–{t[-1]:.0f} s; window {t0:.0f}–{t1:.0f} s")
print()
first = t[np.argmax(cloudy)] if cloudy.any() else None
print(f"First cloud:                {first/60:.1f} min" if first is not None else "First cloud: none")

print(f"\n── Window means (diag.csv) ──────────────────────────────────────  target")
print(f"cloud_frac:                 {100*cf[win].mean():6.1f} %   "
      f"(min {100*cf[win].min():.1f}, max {100*cf[win].max():.1f})   5–20 %")
wc = win & cloudy
if wc.any():
    print(f"cloud base, lowest qc>0:    {cb[wc].mean():6.0f} m   (min {cb[wc].min():.0f})       0.9–1.3 km")
    print(f"cloud top, highest qc>0:    {ct[wc].mean():6.0f} m   (max {ct[wc].max():.0f})       < ~2.5 km")
print(f"w_max:                      {wmax[win].mean():6.2f} m/s (max {wmax[win].max():.2f})      < ~8 m/s")
dt_out = np.diff(t, prepend=t[0])
accum = np.sum(prec[win] * dt_out[win] / 3600.0)
print(f"surface precip:             {prec[win].mean():6.3f} mm/h mean (max {prec[win].max():.3f}), "
      f"{accum:.3f} mm accumulated")
print(f"cloud_persistence (20 min): {pers[win].mean():6.2f}     (max {pers[win].max():.2f})       < 0.4")

# ── Snapshot-based typical base/top ──────────────────────────────────────────
snaps = sorted(glob.glob(os.path.join(run, "snapshots", "snap_*.npz")))
bases, tops, base_t = [], [], []
print(f"\n── Snapshots (column-based, qc > {QC_THRESH:g}) ─────────────────────────")
print(f"{'t (min)':>8} {'cf %':>6} {'base p5/med':>13} {'top med/p95':>13} {'max qc g/kg':>12}")
for f in snaps:
    d = np.load(f)
    ts = float(d["t"])
    if not (t0 - 1e-6 <= ts <= t1 + 1e-6):
        continue
    z = d["z"]
    mask = d["qc"] > QC_THRESH                       # (nx, ny, nz)
    has = mask.any(axis=2)
    if not has.any():
        print(f"{ts/60:8.0f} {0:6.1f}   (no cloud)")
        continue
    k_lo = np.argmax(mask, axis=2)[has]
    k_hi = (mask.shape[2] - 1 - np.argmax(mask[:, :, ::-1], axis=2))[has]
    b, tp = z[k_lo], z[k_hi]
    bases.append(b); tops.append(tp); base_t.append(ts / 3600.0)
    print(f"{ts/60:8.0f} {100*has.mean():6.1f} {np.percentile(b,5):6.0f}/{np.median(b):5.0f} "
          f"{np.median(tp):6.0f}/{np.percentile(tp,95):5.0f} {1e3*d['qc'].max():12.2f}")
if bases:
    b = np.concatenate(bases); tp = np.concatenate(tops)
    print(f"{'window':>8} {'':>6} {np.percentile(b,5):6.0f}/{np.median(b):5.0f} "
          f"{np.median(tp):6.0f}/{np.percentile(tp,95):5.0f}")

# ── Mean-state drift (profiles.npz) ──────────────────────────────────────────
# Linear-fit rates over the window; a steady run keeps these near zero.
prof_file = os.path.join(run, "profiles.npz")
if os.path.exists(prof_file):
    p = np.load(prof_file)
    pt, z = p["times"], p["z"]
    pw = (pt >= t0 - 1e-6) & (pt <= t1 + 1e-6)
    if pw.sum() >= 2:
        h = pt[pw] / 3600.0
        rate = lambda y: np.polyfit(h, y, 1)[0]
        ml = z < 500.0
        th_ml = p["theta"][pw][:, ml].mean(axis=1)
        qv_ml = 1e3 * p["qv"][pw][:, ml].mean(axis=1)
        qv_2k = np.array([np.interp(2000.0, z, q) for q in 1e3 * p["qv"][pw]])
        th_2k = np.array([np.interp(2000.0, z, q) for q in p["theta"][pw]])
        print(f"\n── Mean-state drift (profiles.npz, {pw.sum()} profiles) ──────────────  target")
        print(f"θ mixed layer (<500 m):   {th_ml[0]:7.2f} → {th_ml[-1]:7.2f} K     "
              f"{rate(th_ml):+6.2f} K/h      |·| < 0.2")
        print(f"qv mixed layer (<500 m):  {qv_ml[0]:7.2f} → {qv_ml[-1]:7.2f} g/kg  "
              f"{rate(qv_ml):+6.2f} g/kg/h   |·| < 0.2")
        print(f"qv at 2.0 km:             {qv_2k[0]:7.2f} → {qv_2k[-1]:7.2f} g/kg  "
              f"{rate(qv_2k):+6.2f} g/kg/h   |·| < 0.5")
        print(f"θ at 2.0 km:              {th_2k[0]:7.2f} → {th_2k[-1]:7.2f} K     "
              f"{rate(th_2k):+6.2f} K/h")
        if bases:
            med_b = np.array([np.median(b) for b in bases])
            print(f"median cloud base:        {med_b[0]:7.0f} → {med_b[-1]:7.0f} m     "
                  f"{np.polyfit(base_t, med_b, 1)[0]:+6.0f} m/h      |·| < 30")

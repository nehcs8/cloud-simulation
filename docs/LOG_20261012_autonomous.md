# Autonomous run log — 2026-10-12

Plan: `docs/PLAN_20261012_land_cu.md`. Run unattended by Claude; every decision is logged here.

Standing instructions from Simone:
- Run all simulations back to back; analyse each and adjust the next one without asking.
- Snapshots every 20 s (`SNAPSHOT_EVERY=20.0`) instead of 60 s; one GIF per run, `--fps 9`
  (1080 frames → 2 min for 6 h).
- After the main run, if time and disk allow (≥ 20 GB spare), one evening run:
  heterogeneous surface at 128² (not 160², which needs ~45 GB at 20 s output).
- Pilot snapshots may be deleted once their GIFs are made and checked (keep diag, profiles,
  logs and a few snapshots).
- Stop a run and log the reason if it blows up or free disk drops below 15 GB.

## Timeline

### 10:05 — Sim 1 launched (pilot pair, 64², 100 m, 6 h, 2 threads each)
Machine idle (fresh boot, load 0.02). Forcing exactly as in the plan + `SNAPSHOT_EVERY=20.0`:
`SHF=70.0 RAD_COOLING=-3.5 TAU_NUDGE_T=3600.0 TAU_NUDGE_Q=3600.0 NUDGE_Z_TOP=1800.0 W_SUBS=0.005`
- a: `runs/20261012_1005_land_cu_360min_pilot_a`
- b: `runs/20261012_1005_land_cu_360min_pilot_b` (+ `ENABLE_CLOUD_SHADING=True`)

### 10:56 — Sim 1 finished; analysis (`analyze_land_cu.py <run> 3600 21600`)
Timing (2 threads each, both running at once): a 51.1 min wall, 6284 steps, mean dt 3.44 s,
0.49 s/step, 8.5 wall-min per simulated hour; b 50.3 min, 6132 steps, mean dt 3.52 s.
Snapshots: ~5 MB each at 64², ~5.5 GB per 6-h pilot at 20 s output.

| check (1–6 h)          | target          | pilot a           | pilot b (shading) |
|------------------------|-----------------|-------------------|-------------------|
| first cloud            | < 60 min        | 35 min            | 35 min            |
| cloud cover mean       | 5–20 %          | 11.3 % (7.3–14.6) | 11.4 % (7.4–14.8) |
| θ_ml drift             | \|·\| < 0.25 K/h | +0.24 ✓ (marginal) | +0.21 ✓          |
| qv_ml drift            | (< 0.2)         | −0.13 g/kg/h      | −0.11 g/kg/h      |
| qv(2 km) drift         | < 0.5 g/kg/h    | +0.25 ✓           | +0.21 ✓           |
| median cloud base drift| \|·\| < 40 m/h   | +49 ✗             | +44 ✗             |
| p95 top (window)       | < 2.5 km        | 1.90 km ✓         | 1.82 km ✓         |
| w_max mean (max)       | < 8 m/s         | 6.83 (10.6) ✓     | 6.68 (12.6) ✓     |
| precip accumulated     |                 | 0.053 mm          | 0.035 mm          |
| persistence (20 min)   | < 0.4           | 0.03              | 0.02              |

Hourly median cloud base (a / b): h1 869/866, h2 916/910, h3 963/949, h4 1012/995, h5 1067/1043 m.
The drift is steady (≈45–50 m/h), not decaying. The base still stays inside the Task-10 range
(0.9–1.3 km) for the whole 6 h, but it fails the plan's 40 m/h criterion. In a, the last-hour p95
top climbs to 2.45 km; b stays at ~1.9–2.0 km.

The forcing change worked compared with the 10-11 steady run: θ_ml drift went from 0.48 to
0.21–0.24 K/h, qv(2 km) drift from 0.67 to 0.21–0.25 g/kg/h, cloud-base drift from 93 to 44–49 m/h.

**Decision:** b is at least as good as a on every metric (slightly less drift, steadier tops,
less rain), so shading stays on. Both fail the cloud-base drift check, so per the plan I do one
corrective pilot with one knob changed. Cloud-base drift is roughly half θ_ml warming and half
qv_ml drying (LCL ≈ 125 m per K of dewpoint depression; +0.21 K/h and −0.11 g/kg/h ≈ −0.17 K/h
of dewpoint give ~50 m/h). The table row "still warming → stronger RAD_COOLING" applies. I use
−5.0 K/day instead of the plan's −4.5: at −4.5 the extra cooling is only 0.04 K/h, which would
take the drift to ~39 m/h, right on the limit. At −5.0 it's 0.06 K/h, giving an expected ~35 m/h.

### 11:02 — Pilot c launched (64², 4 threads, alone on the machine)
b forcing with `RAD_COOLING=-5.0`: `runs/20261012_1102_land_cu_360min_pilot_c`. Free disk 77 GB.
Decision rule for c: if the base drift is < 40 m/h and cover is still 5–20 %, the main run uses c's
forcing. If c doesn't improve on b, the main run uses b's forcing anyway (it is inside the Task-10
range for 6 h), rather than spending a third pilot.

### 11:08 — Pilot GIFs done
`cloud_gifs/20261012_land_cu_360min_pilot_a.gif`, `..._pilot_b.gif`: 1081 snapshots, `--fps 9 --smooth 0`,
~3 min render each (`--jobs 2`, nice 19, alongside pilot c), ~230 MB each, 119 s playback. Identical
cloud-free spin-up frames are merged by the GIF writer (1013 stored frames). A test frame of b at
240 min (`outputs/plots/20261012_pilot_b_t240min.png`) shows ~15 well-separated cumulus towers,
base ~0.9 km, tops ~2.1 km, with cloud shadows on the ground. Looks realistic.

### 11:47 — Pilot c finished (4 threads alone: 45.0 min wall, 6233 steps, mean dt 3.47 s, 0.43 s/step, 7.5 wall-min/sim-h)

| check (1–6 h)          | target          | b (−3.5 K/d)      | **c (−5.0 K/d)**  |
|------------------------|-----------------|-------------------|-------------------|
| cloud cover mean       | 5–20 %          | 11.4 %            | 12.3 % (7.9–18.5) ✓ |
| θ_ml drift             | \|·\| < 0.25 K/h | +0.21             | +0.15 ✓           |
| qv_ml drift            |                 | −0.11             | −0.11 g/kg/h      |
| qv(2 km) drift         | < 0.5 g/kg/h    | +0.21             | +0.24 ✓           |
| median cloud base drift| \|·\| < 40 m/h   | +44 ✗             | +36 ✓             |
| p95 top (window)       | < 2.5 km        | 1.82 km           | 1.86 km ✓         |
| w_max mean (max)       | < 8 m/s         | 6.68 (12.6)       | 6.76 (10.9) ✓     |
| precip accumulated     |                 | 0.035 mm          | 0.052 mm          |
| persistence            | < 0.4           | 0.02              | 0.01              |

Hourly median base in c: 863, 898, 928, 970, 1004 m (1–5 h). Base drift is down by 8 m/h, as
predicted (~35 m/h). The remaining drift comes from qv_ml drying (−0.11 g/kg/h, unchanged), which
cooling can't fix; a later knob could be weak nudging of qv in the ML, or a little more LHF.
Extra cooling raised cover slightly (12.3 vs 11.4 %), with no taller tops.

**Decision:** c passes everything, so the main run uses c's forcing. GIF:
`cloud_gifs/20261012_land_cu_360min_pilot_c.gif` (146 s render with `--jobs 4`).

### 11:50 — Sim 2 launched (128×128, 50 m, 6.4 km, 6 h, 4 threads, nothing else running)
`run_land_cu.py 21600 NX=128 NY=128 DX=50.0 DY=50.0 SHF=70.0 RAD_COOLING=-5.0 TAU_NUDGE_T=3600.0
TAU_NUDGE_Q=3600.0 NUDGE_Z_TOP=1800.0 W_SUBS=0.005 SNAPSHOT_EVERY=20.0 ENABLE_CLOUD_SHADING=True TAG=main_50m`
→ `runs/20261012_1150_land_cu_360min_main_50m`. `effective_config.txt` checked: NX/NY 128, DX/DY 50,
forcing as above. Snapshots are 18.7 MB (not the 26 MB in the plan), so ≈ 20 GB for 1081 of them.

Disk cleanup (approved by Simone): pilot a/b/c snapshots reduced to the 7 hourly ones each
(t = 0, 1 h … 6 h); diag, profiles, budgets and logs kept. Free disk 69 → 90 GB.

### 12:21 — Main run, 30 wall-min checkpoint
t = 78 min simulated after 30 wall-min (spin-up at dt 5 s is fast). Convective phase:
0.47 wall-s per simulated s (measured over the last 600 s of snapshots, including 20-s output)
≈ 28 wall-min per simulated hour, dt ≈ 2.9 s. Projected finish ≈ 14:35 (≈ 2.75 h in total),
far below the 6 h abort limit, so the run continues for the full 6 h. Cover at 78 min: 15.4 %,
44 clouds, CB 820 m, CT 1.9 km, w_max 5 m/s. Snapshots shrink with compression (12 MB now,
18.7 MB at the start), so expect less than 20 GB.

### 12:44 — Main run, t = 2 h checkpoint: pass
1–2 h: cover 12–17 % (40–50 clouds), w_max 4.2–8.2 m/s (no blow-up), dt 2.2–3.0 s (stable, not
collapsing), CB 820 m, CT 1.7–2.1 km, precip ≤ 0.01 mm/h. Cover is a bit higher than in pilot c
at the same time (~12–13 %). Continuing.

### 15:47 — Main run killed at t = 5 h 41 min (out of memory, not the model)
At 15:47:57 the kernel OOM killer fired (`journalctl -k`): 8 `node` processes that started ~15:44
held ~12.9 GB of the 15 GB RAM (the simulation used 0.5 GB). The OOM kill took down VS Code; the
simulation was a child of the Claude session's shell inside VS Code's snap scope, so it died with
it. They weren't started by this session. Last data: console/diag t = 20460 s, last snapshot
t = 20502 s (1026 snapshots, 27 GB; late snapshots are ~28 MB, more than the 18.7 MB at the
start). `profiles.npz` is only written at the end, so it was missing.

**Decision: don't rerun the missing 19 min.** The 1–5.67 h window answers every question in the
plan; a rerun would cost ~2.75 h for 5 % more data. Recovery:
- `profiles.npz` rebuilt from the snapshots' `prof_*` arrays every 600 s (35 profiles, the model's
  cadence). Snapshots store θ′, so θ₀ = pilot c `profiles.theta[0]` − pilot c `snap_0000000.prof_theta`
  (same sounding and z grid). Check: pilot c's 1-h profile rebuilt this way matches the real one
  to 3e-5 K. RH and w_var aren't rebuilt (the analysis doesn't use them).
- From now on, long runs go under `systemd-run --user` (their own unit, outside VS Code's cgroup).

### 15:50 — Main run results (`analyze_land_cu.py <run> 3600 20400`), compared with pilot c on the same window

| check (1–5.67 h)       | target          | **main 50 m, 128²** | pilot c 100 m, 64² |
|------------------------|-----------------|-------------------|-------------------|
| first cloud            |                 | 31 min            | 35 min            |
| cloud cover mean       | 5–20 %          | 15.1 % (10.8–18.3) ✓ | 12.3 % (7.9–18.5) |
| θ_ml drift             | \|·\| < 0.25 K/h | +0.15 ✓           | +0.15             |
| qv_ml drift            |                 | −0.10 g/kg/h      | −0.11             |
| qv(2 km) drift         | < 0.5 g/kg/h    | +0.22 ✓           | +0.25             |
| median cloud base drift| \|·\| < 40 m/h   | +37 ✓ (1026 snaps) | +36 (all snaps, 11:47) |
| p95 top (window)       | < 2.5 km        | 1.86 km ✓         | 1.82 km           |
| w_max mean (max)       | < 8 m/s         | 7.21 (10.8) ✓     | 6.67 (10.9)       |
| precip accumulated     |                 | 0.055 mm          | 0.046 mm          |
| persistence            | < 0.4           | 0.00              | 0.01              |
| mean dt (t > 1 h)      |                 | 2.38 s            | 3.50 s            |
| clouds in domain (mean)|                 | 37 (6.4 km)²      | 25 (6.4 km)²      |

Note: the pilot c base drift recomputed today gives 28 m/h because only its 7 hourly snapshots
are left. The 36 m/h from the full set (11:47) is the right comparison.

Hourly in main: cover 14.5, 16.1, 15.3, 15.4, 14.1 %; median base 862, 888, 936, 987, 993 m;
p95 top 1.66, 1.84, 1.87, 1.92, 1.94 km.

**Unknown 4 (50 m vs 100 m):** the mean-state drift is identical, so the forcing controls it, not
resolution. At 50 m, cover is +2.8 points (15.1 vs 12.3 %, about 25 % more), there are ~50 % more
distinct clouds in the same area, and w_max is +0.5 m/s. Cloud base and p95 top don't change
(±40 m). Halving the Smagorinsky Δ lets more, smaller-scale cloud survive; tops are limited by the
inversion, not by the resolution.

**Unknown 3 (cost at 50 m):** dt 2.38 s (0.68 × the 100 m dt, not 0.5 ×, since the vertical CFL at
dz = 40 m limits both). Wall: 0.47 wall-s per simulated s in the convective phase, including 20-s
snapshot output ≈ 28 wall-min per simulated hour; the full 6 h ≈ 2.75 h wall (the plan estimated
3–4 h). Snapshots: 12–28 MB (grow as clouds fill the domain, compression), ≈ 27 GB for 6 h at
20 s output.

GIF: `cloud_gifs/20261012_land_cu_341min_50m.gif` (1026 frames, 0–5 h 41 min, `--fps 9`, 223 MB,
150 s render). The name says 341 min instead of the plan's 360 min, because the run stopped early.
Frame at 4 h: `outputs/plots/20261012_main_50m_t240min.png`. The clouds are visibly more detailed
and irregular than at 100 m, with merging towers. Realistic.

### 15:54 — Evening run launched: heterogeneous surface, 128² / 50 m, 6 h
Disk 64 GB free before the run (needs ~28 GB → ~36 GB spare, rule ≥ 20 GB ✓). Same forcing as
main + `SHF_SIGMA=0.25 SHF_CORR=2000.0`, so the two compare directly. Launched with
`systemd-run --user --unit=landcu-hetero` → `runs/20261012_1554_land_cu_360min_hetero_50m`
(effective_config checked). Expected finish ≈ 18:40.
Note: with a 6.4 km domain and a 2 km correlation length, there are only ~3 warm/cold patches per
side; enough to see whether clouds lock onto warm patches, not a statistical sample.

### 18:56 — Evening run (hetero_50m) finished: 184.9 min wall, 8802 steps, mean dt 2.45 s, 30.8 wall-min/sim-h

| (1–5.67 h)             | main_50m (homog.) | hetero_50m |
|------------------------|-------------------|------------|
| cloud cover mean       | 15.1 %            | 15.4 % (10.0–19.1) |
| θ_ml drift             | +0.15 K/h         | +0.03 K/h  |
| qv_ml drift            | −0.10 g/kg/h      | +0.06 g/kg/h |
| qv(2 km) drift         | +0.22 g/kg/h      | +0.19 g/kg/h |
| median cloud base      | 860 → 1020 m, +37 m/h | 820 → 860 m, +5 m/h |
| p95 top                | 1.86 km           | 1.86 km    |
| w_max mean (max)       | 7.21 (10.8)       | 7.39 (11.8) |
| precip accumulated     | 0.055 mm          | 0.073 mm   |

**This run is confounded by a bug in `forcing.build_flux_map`**, found when checking why the mean
state stopped drifting. The Gaussian-filtered random field is divided by its std, but **its mean
is never subtracted**. With SHF_CORR = 2 km on a 6.4 km domain, the sample mean is far from 0.
Reproduced with the run's seed (RANDOM_SEED 42 + 1):
- SHF map mean **35.1 W/m²** instead of 70 (range 5–71, the 5 W/m² clip is active);
- LHF map mean **333.7 W/m²** instead of 300;
- SHF and LHF come from two independent draws: corr(SHF, LHF) = −0.56 (warm patches are dry).
  `FluxPattern` (the evolving version) does it properly: mean removed, `mode='wrap'`, one shared field.
- `gaussian_filter` uses the default `mode='reflect'`, so the map isn't periodic on the periodic domain.
So the drift result mostly reflects SHF 35 / LHF 334 (Bowen ratio 0.10 instead of 0.23), not the
heterogeneity. Side result worth keeping: with half the SHF and +11 % LHF the mean state is almost
steady (θ_ml +0.03 K/h, base +5 m/h) while cover stays at 15 %. That's a lead for the forcing, not
a clean test. Pilots and main_50m used SHF_SIGMA = 0 and are unaffected.

**Valid part: clouds lock onto the warm patches.** Time-mean column cloud frequency (281 snapshots,
1–5.67 h) vs the SHF map: r = +0.61; cloud frequency 21.0 % over the warmest quarter vs 9.9 % over
the coldest. Control (main_50m on the same map): r = 0.00, 15.2 vs 15.1 %. The std of the
cloud-frequency map is 7.5 % (hetero) vs 2.6 % (homogeneous).

GIF: `cloud_gifs/20261012_land_cu_360min_50m_hetero.gif` (1081 frames, 148 s render). Frame at 4 h:
`outputs/plots/20261012_hetero_50m_t240min.png`: more numerous, smaller, flatter clouds than main.

**Decision: no rerun tonight.** Free disk is 36 GB; another 6-h 50 m run needs ~28 GB, which
would leave ~8 GB, below the 15 GB stop rule. The model code needs a fix first (mean removal is
clear; using one shared SHF/LHF field is a physics choice for Simone). I didn't change
`forcing.py` and didn't delete the hetero snapshots.

### ~20:00 — Simone's answers; fix, cleanup, rerun
Simone: SHF and LHF patterns should be **independent**; snapshots of **all runs** can be thinned to
one every 30 min. Wants to move away from periodic lateral boundaries eventually (later, not tonight).

- **Cleanup:** every `runs/*/snapshots` now keeps the snapshot nearest each 30-min mark plus the
  final one (the final state is useful for restarts). 2946 files / 72.2 GB deleted; free disk 36 → 103 GB.
  Pilots a/b/c were already hourly (thinned earlier today), so they stay hourly. `experiments/`
  (71 GB), legacy `snapshots/` (7.1 GB), `nested_runs/` and `restart_snapshots/` untouched.
- **Fix in `forcing.build_flux_map`:** zero-mean fields, `mode='wrap'` (periodic, matching the domain),
  the LHF pattern projected out of the SHF pattern so they're exactly uncorrelated (two random draws
  with only a few patches per domain are otherwise correlated by chance, −0.56 last time), and the
  means restored after clipping. Check at 128²/50 m: SHF 70.00 ± 17.5 (41–110), LHF 300.00 ± 75,
  corr 0.000, edge jump equal to the interior neighbour difference.
- **SHF_CORR 2000 → 500 m (my choice):** SHF_CORR is the Gaussian σ, so 2 km gives patches ~5 km
  across, which is only 2 warm and 2 cold on 6.4 km. 500 m gives 6 + 6 patches ~1–2 km across,
  comparable to the boundary-layer depth (~1 km), where surface heterogeneity should matter most.

### 20:08 — hetero2_50m launched (systemd unit `landcu-hetero2`)
main_50m setup + `SHF_SIGMA=0.25 SHF_CORR=500.0` → `runs/20261012_2008_land_cu_360min_hetero2_50m`.
Effective config checked. Expected finish ≈ 23:15.

### 20:15–22:00 — Non-periodic lateral boundaries, step 1: fringe zone with recycled inflow
Simone asked to start on non-periodic boundaries. Route chosen: keep the periodic FFT pressure solver,
make x an inflow/outflow direction with a fringe zone (Spalart/Nordström) fed by recycling.
- Code: `lateral_bc.py` (new), config keys `LATERAL_BC` (default "periodic"), `FRINGE_WIDTH` 1600 m,
  `FRINGE_TAU` 20 s, `RECYCLE_X` 4800 m, `HETERO_BUFFER` 1000 m, `HETERO_TAPER` 500 m; hook in
  `rk3_step` before the final BCs; `build_flux_map` confines patches to the test region
  (identical maps in periodic mode); fringe term in the `run_land_cu.py` water budget; run header line.
- Tests: `tests/test_fringe.py` passes. test_land_cu and test_projection pass. test_water_budget shows a
  1.5 % residual, but its saved output from 2026-10-11 12:37 had 1.3 %; the default-case test is stale,
  and its code path is unchanged by this work.
- New analysis: `scripts/analyze_fringe.py`.
- Dynamic test `runs/20261012_2059_land_cu_180min_fringe_test_homog` (128×64, 100 m, 3 h, homogeneous,
  2 threads, 42 min wall): fetch vs test region over 1–3 h: cover 11.1 / 11.1 %, w′²(500 m) 0.668 / 0.669,
  LWP 8.69 / 8.73 g/m², u_ml 5.09 both; no jump at x = 0 vs x = RECYCLE_X. The fringe's first half shows a
  dip (cover 8.8 %, w′² 0.55, from blending two uncorrelated turbulent fields); it's confined to the fringe.
- Bug found and fixed: `apply_fringe` relaxed θ_l/q_t but not the diagnostic θ/qv/qc, so the water
  budget booked the fringe's exchange as a −0.16 % "dynamics" residual. Now all are relaxed together. Check run
  `runs/20261012_*_fringe_budget_check` (90 min): residual 0.00000 %, fringe term +0.02 % in 1.5 h.

### 23:15 — hetero2_50m finished (corrected flux maps, SHF_CORR 500 m): 189.1 min wall, 31.5 wall-min/sim-h

| (1–6 h)                | main_50m (homog.) | hetero2_50m (σ 0.25, 500 m) |
|------------------------|-------------------|------------------------------|
| cloud cover mean       | 15.1 %            | 15.4 % (11.2–18.8)           |
| θ_ml / qv_ml drift     | +0.15 / −0.10     | +0.15 K/h / −0.10 g/kg/h     |
| qv(2 km) drift         | +0.22             | +0.21 g/kg/h                 |
| median base drift      | +37 m/h           | +38 m/h (860 → 1020 m)       |
| p95 top                | 1.86 km           | 1.90 km                      |
| w_max mean (max)       | 7.21 (10.8)       | 7.36 (11.2)                  |
| precip accumulated     | 0.055 mm          | 0.064 mm                     |
| cloud freq vs SHF map  | r = +0.02 (11 snaps), 15.1 / 14.7 % | r = +0.25 (901 snaps), warm 16.5 % / cold 14.3 % |

- With correct mean fluxes the mean state is identical to the homogeneous run. So the near-zero drift of
  the buggy hetero_50m came from its wrong fluxes (SHF 35 / LHF 334), not from heterogeneity. That bug
  result still suggests the drift fix for the steady forcing: lower SHF / higher LHF (Bowen ratio).
- Patches of ~1–2 km at 25 % amplitude organise clouds only weakly (r = +0.25; +15 % cloud frequency
  over the warmest quarter). Versus the buggy run (r = +0.61 with ~5 km patches at ~50 % amplitude):
  at 5 m/s, air crosses a 1–2 km patch in 3–7 min, shorter than a thermal's development time, so the
  signal is smeared. Larger patches (≥ 2–3 × BL depth) or stronger contrast are needed for a clear
  response. The fringe setup (air crosses the test region once) is the clean way to test that.
- GIF `cloud_gifs/20261012_land_cu_360min_50m_hetero2.gif` (144 s render). Snapshots of hetero2 and
  the fringe test run thinned to 30 min (31.5 GB); free disk 103 GB.

### 2026-10-14 10:19 — fringe_hetero_50m launched (systemd unit `landcu-fringe-hetero50`)
Fringe + flux patches at 50 m: 256×128 (12.8 × 6.4 km), main_50m forcing + `LATERAL_BC='fringe'
SHF_SIGMA=0.4 SHF_CORR=1000.0`, 20-s snapshots, 6 h → `runs/20261014_1019_land_cu_360min_fringe_hetero_50m`
(stdout `outputs/logs/fringe_hetero_50m_runner.log`). Effective config checked.
Map check before launch: test region [5800, 11200] m gets SHF 69 ± 28 (27–133 W/m²), no clipping, fetch exactly
70; ~2 warm patches ~2.5 km across (larger corr would still give 2 in a 5.4 km region; σ 0.4 for contrast).
Disk 101 GB free before (needs ~55 GB). Machine idle (load 1). Expected ≈ 5.5 h wall → ~16:00.
Analysis: `scripts/analyze_fringe.py RUN_DIR 3600 21600 --shf-map`.

### 2026-10-14 13:20 — fringe_hetero_50m: the test region is a twin of the fetch
Simone noticed the w field looks artificial. At t = 188 min, w(x, y) at x and at x + 4.8 km (= RECYCLE_X)
correlate at 0.99–1.00 at every height (300 m–2 km); other shifts give |r| < 0.15. Inside the test region it
decays from 1.000 (5.8 km) to 0.89 (11.2 km). Cause: the fringe feeds x = 0 with the state at RECYCLE_X, and the
test region's inflow is the same state, so both regions evolve from identical inflow; chaotic divergence over
the ~16-min crossing is small. This is not a coding bug: the patches act (θ at 20 m, test minus twin, has
r = 0.80 with the SHF map, 0.40 at 100 m, 0.05 at 500 m), but the 2026-10-12 validation (fetch = test) was trivial.
Plan: analyse this run as a paired experiment (test minus twin) and crop the GIF to the region downstream of
RECYCLE_X. Plot: `outputs/plots/20261014_fringe_hetero_w_check.png`.

### 2026-10-14 16:29 — fringe_hetero_50m finished: 373 min wall (62 wall-min/sim-h), 8760 steps, mean dt 2.47 s
Direct statistics (1–6 h; `scripts/analyze_fringe.py --shf-map`): cover fetch 15.5 / test 15.7 / fringe 14.2 %;
cloud frequency vs SHF map in place r = −0.03 (warm quarter 15.4 %, cold 15.7 %), i.e. nothing *above* the patches.
Water budget residual 0.

Paired analysis (new `scripts/analyze_fringe_paired.py`, test region minus its twin 4.8 km upstream; full output
`outputs/logs/20261014_fringe_hetero_50m_paired.txt`, maps `outputs/plots/20261014_fringe_hetero_50m_paired.png`):
the response is there, advected downstream, and develops in steps:

| field (time mean, test − twin) | r with SHF map, no shift | best r | at downstream shift |
|-------------------------------|--------------------------|--------|---------------------|
| θ at 20 m                     | +0.89                    | +0.98  | 0.5 km              |
| w′² at 500 m                  | +0.13                    | +0.91  | 2.5 km              |
| cloud frequency               | +0.04                    | +0.83  | 3.5 km              |
| condensed-water path          | +0.02                    | +0.80  | 3.5 km              |

The best cross-wind shift (0.25 km) matches the mean v/u = 0.40/4.97 drift. At the fixed physical shifts none of
60 unrelated flux maps reaches these r (null −0.05 ± 0.42; the spread is large because there are only ~2 warm
patches). At 5 m/s, 2.5–3.5 km means 8–12 min, about one thermal development time: surface heating, then stronger
thermals, then clouds. Δcover is still rising at the fringe (+0.7 points at 10.8 km), so the response is cut off,
and the effect is local redistribution (paired cover 15.65 vs 15.46 %).
This also explains hetero2's weak r = +0.25: it correlated clouds with the map *in place*.

GIF `cloud_gifs/20261014_land_cu_360min_50m_fringe_hetero_test_region.gif` (x = 4.8–11.2 km, `--zexp 1.0`,
1081 frames, 168 s). Renderer fix: `render_clouds.py` assumed dx = 100 m (snapshots don't store it), so all earlier
50 m GIFs are 2× stretched horizontally; it now reads DX/DY from `effective_config.txt` and has `--xrange`.
Snapshots thinned to 30 min (1068 files, 58.5 GB); 2-D fields kept in `reduced_2d.npz` (0.9 GB). Free disk 99 GB.

### 2026-10-14 17:02 — fringe_stripe_50m launched (systemd unit `landcu-stripe50`)
Question: how does the cloud response develop and decay downstream of a warm surface, and how far does it reach?
The 2-patch map gave only ~2 independent samples and a cut-off response, so this run uses one warm stripe that is
uniform in y (averaged over y and time) and a long homogeneous stretch downstream.
New config: `SHF_STRIPE_AMP/_X0/_WIDTH/_TAPER` (forcing.stripe_profile; test in `tests/test_fringe.py`).
Layout (352×96, 50 m, Lx 17.6 km, Ly 4.8 km): fetch 0–8 km (RECYCLE_X = 8000, so every test x has a twin
upstream), stripe SHF 70 → 105 W/m² (+50 %, LHF unchanged) over 9–11 km (250 m cos² edges), then 5 km of
homogeneous surface to the fringe at 16 km. Same forcing as main_50m; 60-s snapshots (~19 GB expected).
Cost ≈ hetero run (33.8 k vs 32.8 k columns): ~6.4 h wall → ~23:30.
→ `runs/20261014_1702_land_cu_360min_fringe_stripe_50m`, stdout `outputs/logs/fringe_stripe_50m_runner.log`.
Analysis: `scripts/analyze_fringe_paired.py RUN 3600 21600` (Δ along x is the response curve).

### 2026-10-14 23:15 — fringe_stripe_50m finished; GIF
GIF `cloud_gifs/20261014_land_cu_360min_50m_fringe_stripe_test_region.gif` (x = 8–16 km: stripe at 9–11 km +
5 km downstream; `--zexp 1.0 --smooth 0 --fps 6`, 361 frames, 54 s). Frame at 4 h:
`outputs/plots/20261014_fringe_stripe_50m_t240min.png`. Analysis not yet run; snapshots not yet thinned.

# land_cu plan — 2026-10-12

Goal of the day: a 6 h, 50 m land_cu run with realistic cumulus whose mean state does not drift
out of the Task-10 targets. Two simulations: a cheap 64² pilot (run as a pair) that settles the
forcing, then the main 128² / 50 m run.

## Where we are (end of 2026-10-11)

- `runs/20261011_1958_land_cu_180min_steady` (64², SHF 110, nudging τ 1.5 h, W_SUBS 5 mm/s):
  clouds good (cover 12.6 %, base 0.9–1.06 km, p95 top 1.86 km), but the mean state drifts:
  θ_ml +0.48 K/h, qv_ml −0.20 g/kg/h, qv(2 km) +0.67 g/kg/h, cloud base +93 m/h.
  Over 6 h that would take cloud base to ~1.4 km and moisten the inversion further.
- Numba (`kernels.py`) is in and validated: 64² 0.30 s/step, 128² 1.05 s/step (4–5× numpy).
  Measured on a quiet machine without output. A 20-min run with output, sharing the CPU with a
  numpy run, took 0.63 s/step at 64², so real cost is still unknown (see unknown 3).
- Cloud shading (`ENABLE_CLOUD_SHADING`, Beer's law on LWP, sun straight overhead) exists in
  `surface_layer.py` but has never been used in a land_cu run.

## Unknowns to answer

1. Can SHF + radiative cooling + nudging hold θ_ml, qv and cloud base steady for 6 h, and still
   keep cover in the 5–20 % range?
2. What does cloud shading do to cover, cloud size and drift? (It lowers mean flux by roughly
   cover × 0.8, so it has to be tuned together with the forcing, which is why it goes in the pilot.)
3. Real cost per simulated hour at 50 m: dt (CFL is set by w/dz = 40 m, so dt should drop much
   less than 2×), wall time per step with output, snapshot write time (~26 MB each at 128²).
4. Does the 50 m grid change cover, cloud base and w_max compared with 100 m (Smagorinsky Δ halves)?

## Sim 1: pilot pair (64×64, 100 m, 6 h), both at once, 2 threads each

Same forcing for both; b adds shading. Both finish in about 1 h (check `uptime` first).

Forcing change compared with the steady run: lower SHF (θ_ml drift) + stronger cooling below the
inversion + faster nudging starting below the inversion (qv(2 km) drift).
A rough linear estimate puts θ_ml drift near 0.25 K/h. That's an estimate, the run decides.

```bash
F="SHF=70.0 RAD_COOLING=-3.5 TAU_NUDGE_T=3600.0 TAU_NUDGE_Q=3600.0 NUDGE_Z_TOP=1800.0 W_SUBS=0.005"
NUMBA_NUM_THREADS=2 MPLBACKEND=Agg nohup ./venv/bin/python3 run_land_cu.py 21600 $F TAG=pilot_a \
    > outputs/logs/land_cu_pilot_a_runner.log 2>&1 &
NUMBA_NUM_THREADS=2 MPLBACKEND=Agg nohup ./venv/bin/python3 run_land_cu.py 21600 $F ENABLE_CLOUD_SHADING=True TAG=pilot_b \
    > outputs/logs/land_cu_pilot_b_runner.log 2>&1 &
```
Judge with `scripts/analyze_land_cu.py runs/<run> 3600 21600` (drift section at the bottom).

| check (window 1–6 h)        | pass            | if it fails                                            |
|-----------------------------|-----------------|--------------------------------------------------------|
| cloud cover mean            | 5–20 %          | < 5 % or first cloud > 60 min → SHF 85                  |
| θ_ml drift                  | \|·\| < 0.25 K/h  | still warming → RAD_COOLING −4.5; cooling → SHF 85     |
| qv(2 km) drift              | < 0.5 g/kg/h    | TAU_NUDGE_Q 2400, NUDGE_Z_TOP 1700                     |
| median cloud base drift     | \|·\| < 40 m/h    | follows θ_ml / qv_ml, fix those first                  |
| p95 top, w_max              | < 2.5 km, < 8 m/s | tops too high → the FT is still moistening (row above) |

Pick for Sim 2: b if it passes and its clouds look at least as good as a's (render a few frames with
`--single`); otherwise a. If both fail, adjust one knob per the table and rerun one pilot
(~40 min with 4 threads) rather than starting Sim 2 on a drifting setup.
Also record from `runner.log`: wall-min per simulated hour and mean dt after 1 h.

## Sim 2: main run (128×128, 50 m, 6.4 km domain, 6 h), 4 threads, nothing else running

```bash
MPLBACKEND=Agg nohup ./venv/bin/python3 run_land_cu.py 21600 NX=128 NY=128 DX=50.0 DY=50.0 \
    <forcing from Sim 1> TAG=main_50m > outputs/logs/land_cu_main_50m_runner.log 2>&1 &
```
(`run_land_cu.py` sets NX=NY=64 itself, but the KEY=VALUE overrides are applied afterwards, so
NX/NY here take effect. Confirm in `effective_config.txt` before walking away.)

- Expected cost: dt ≈ 2–2.5 s, ~1.1 s/step + output → about 3–4 h wall. Disk ≈ 10 GB of snapshots.
- **Checkpoint at 30 wall-min:** from `console.log` (simulated t vs wall time) project the finish.
  If it projects > 6 h, restart with T_END 14400 (4 h) instead of letting it run into the night.
- **Checkpoint at t = 2 h simulated:** cover 5–20 %, no blow-up (w_max < 15 m/s), dt not collapsing.
- Afterwards: `analyze_land_cu.py` (1–6 h window); compare with the pilot (unknown 4);
  GIF with `--fps 6` → `cloud_gifs/20261012_land_cu_360min_50m.gif`; record results in
  `docs/EXPERIMENT_LOG.md` and the drift and timing numbers in CLAUDE.md.

## If there is time in the evening

Overnight candidates (cost: one more run): 160×160 / 50 m (8 km domain, ~5–6 h wall);
a heterogeneous surface (`SHF_SIGMA` 0.2–0.3, `SHF_CORR` 2000 m) to see cloud organisation over
warm/cold patches; or fuse the leftover numpy in `compute_tendencies` / `project` (~25 % more speed).

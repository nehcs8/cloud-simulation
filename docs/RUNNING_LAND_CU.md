# Running the land_cu simulations (shallow-cumulus field)

How to reproduce the land_cu runs of 2026-10-11 … 10-14. The code state they used is commit
`249a909` ("Add land_cu case, numba kernels, …"). After that commit the runner was
generalised to `run_case.py CASE[:VARIANT] …`. `run_land_cu.py` is kept as a wrapper and is
identical to `run_case.py land_cu …`, so every command below still works unchanged.
For the exact code, `git checkout 249a909`. Each run folder also keeps its `config.py` copy and
`effective_config.txt`, which holds the full set of values after overrides.

## Command form

```bash
MPLBACKEND=Agg ./venv/bin/python3 run_land_cu.py T_END_s [KEY=VALUE ...] [TAG=name] > runs/<run>/runner.log 2>&1
# identical:  run_case.py land_cu T_END_s [KEY=VALUE ...] [TAG=name]
```
- The defaults come from `cases/land_cu.py: configure()`: 64×64 at 100 m, 40 m levels to 3 km
  stretched to 4 km, SHF 150 / LHF 300 W/m² ramped over 45 min, u_g = 5 → 9 m/s sheared, subsidence
  4 mm/s, nudging τ 6 h above 1.9 km, radiative cooling −2 K/day, sponge from 3.2 km, dt ≤ 5 s.
- Overrides are Python literals; quote strings for the shell: `"LATERAL_BC='fringe'"`.
- Output goes to `runs/<YYYYMMDD_HHMM>_land_cu_<min>min[_<TAG>]/`.
- For long runs, use `systemd-run --user --unit=<name> --working-directory="$PWD" /bin/bash -c '…'`
  so the run survives a VS Code crash.

## Runs and their exact overrides

The overrides were recovered from each run's `runner.log` (copies in `outputs/logs/*_runner.log`).
`STEADY` stands for the recommended steady forcing:
`SHF=70.0 RAD_COOLING=-5.0 TAU_NUDGE_T=3600.0 TAU_NUDGE_Q=3600.0 NUDGE_Z_TOP=1800.0 W_SUBS=0.005 ENABLE_CLOUD_SHADING=True`

| Run folder (runs/…) | Command arguments |
|---|---|
| 20261011_1515_land_cu_30min | `1800` |
| 20261011_1527_land_cu_120min | `7200` |
| 20261011_1616_land_cu_240min | `14400` |
| 20261011_1958_land_cu_180min_steady | `10800 SHF=110.0 TAU_NUDGE_T=5400.0 TAU_NUDGE_Q=5400.0 W_SUBS=0.005 TAG=steady` |
| 20261012_1005_…_pilot_a | `21600 SHF=70.0 RAD_COOLING=-3.5 TAU_NUDGE_T=3600.0 TAU_NUDGE_Q=3600.0 NUDGE_Z_TOP=1800.0 W_SUBS=0.005 SNAPSHOT_EVERY=20.0 TAG=pilot_a` |
| 20261012_1005_…_pilot_b | as pilot_a `+ ENABLE_CLOUD_SHADING=True TAG=pilot_b` |
| 20261012_1102_…_pilot_c | `21600 STEADY SNAPSHOT_EVERY=20.0 TAG=pilot_c` |
| 20261012_1150_…_main_50m | `21600 NX=128 NY=128 DX=50.0 DY=50.0 STEADY SNAPSHOT_EVERY=20.0 TAG=main_50m` |
| 20261012_1554_…_hetero_50m | main_50m `+ SHF_SIGMA=0.25 SHF_CORR=2000.0 TAG=hetero_50m` |
| 20261012_2008_…_hetero2_50m | main_50m `+ SHF_SIGMA=0.25 SHF_CORR=500.0 TAG=hetero2_50m` |
| 20261012_2059_…_180min_fringe_test_homog | `10800 NX=128 NY=64 DX=100.0 DY=100.0 STEADY SNAPSHOT_EVERY=60.0 "LATERAL_BC='fringe'" TAG=fringe_test_homog` |
| 20261012_2143_…_90min_fringe_budget_check | `5400 NX=128 NY=64 DX=100.0 DY=100.0 STEADY SNAPSHOT_EVERY=1800.0 "LATERAL_BC='fringe'" TAG=fringe_budget_check` |
| 20261014_1019_…_fringe_hetero_50m | `21600 NX=256 NY=128 DX=50.0 DY=50.0 STEADY SNAPSHOT_EVERY=20.0 "LATERAL_BC='fringe'" SHF_SIGMA=0.4 SHF_CORR=1000.0 TAG=fringe_hetero_50m` |
| 20261014_1702_…_fringe_stripe_50m | `21600 NX=352 NY=96 DX=50.0 DY=50.0 STEADY SNAPSHOT_EVERY=60.0 "LATERAL_BC='fringe'" RECYCLE_X=8000.0 SHF_STRIPE_AMP=35.0 SHF_STRIPE_X0=9000.0 SHF_STRIPE_WIDTH=2000.0 TAG=fringe_stripe_50m` |

(`STEADY` is shorthand only: write the values out in the command.) Pilots a and b were started
together and ran side by side.

Example, main_50m in full:
```bash
MPLBACKEND=Agg ./venv/bin/python3 run_land_cu.py 21600 NX=128 NY=128 DX=50.0 DY=50.0 \
  SHF=70.0 RAD_COOLING=-5.0 TAU_NUDGE_T=3600.0 TAU_NUDGE_Q=3600.0 NUDGE_Z_TOP=1800.0 \
  W_SUBS=0.005 ENABLE_CLOUD_SHADING=True SNAPSHOT_EVERY=20.0 TAG=main_50m
```

## Analysis and rendering

```bash
./venv/bin/python3 scripts/analyze_land_cu.py runs/<run_dir> 3600 7200
./venv/bin/python3 scripts/analyze_fringe.py runs/<run_dir> [t0 t1] [--shf-map]
./venv/bin/python3 scripts/analyze_fringe_paired.py runs/<run_dir> [t0 t1]
VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow ./venv/bin/python3 scripts/visualize/render_clouds.py \
    runs/<run_dir> --out cloud_gifs/YYYYMMDD_<description>.gif --fps 2 --smooth 0 --jobs 4
```
Results and the reasons behind the settings: `docs/EXPERIMENT_LOG.md` (2026-10-12 onwards) and
the "land_cu run" section of `CLAUDE.md`.

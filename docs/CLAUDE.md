# Cloud Simulation — Reference

Cumulus cloud simulations on laptop (12×12×5 km domain) with nested high-resolution runs.

---

## Environment

- Python: `./venv/bin/python3` (local venv)
- Working dir: `/home/simone/Documents/Python/Python i guess/Clouds/cloud-simulation`
- Packages: numpy, scipy, matplotlib, pyvista, imageio, pyyaml, cdsapi

---

## Code Structure

### Core (root)
- `main.py`, `main_nested.py` — entry points
- `config.py`, `config_nested.py` — all parameters
- `simulation.py` — time loop, CFL limiter, snapshots
- `state.py` — fields + reference atmosphere
- `grid.py` — NX×NY×NZ, exponential z-stretch
- `dynamics.py` — WENO-3 advection + Smagorinsky SGS
- `pressure.py` — FFT projection (2D FFT x-y, tridiagonal z)
- `microphysics.py` — Kessler warm-rain
- `forcing.py`, `forcing_nested.py` — SHF/LHF, subsidence, nudging, sponge
- `boundary_conditions.py` — relaxation BCs for nested
- `terrain.py` — terrain handling

### Scripts
- `scripts/find_cloud.py` — detect clouds in snapshots
- `scripts/setup_nested.py` — create nested ICs from parent
- `scripts/track_cloud.py` — track cloud lifetime
- `scripts/run_experiment.py` — automated experiments
- `scripts/run_manager.py` — YAML metadata system (2026-09-22)
- `scripts/visualize/` — visualization tools

### Organization
- `experiments/parent/` — parent simulation runs
- `nested_runs/` — nested simulation runs (YAML metadata)
- `restart_snapshots/` — protected restart files (immutable)
- `snapshots/` — ephemeral outputs (cleared each run)
- `outputs/logs/`, `outputs/plots/` — archived outputs
- `docs/` — documentation

---

## Physics Configuration

### Parent Simulation (100m resolution)
```python
Grid: 120×120×80, dx=100m, domain 12×12×5km, dz=27m→120m
Time: dt=2.0s, T_END=7200s (2 hr), output every 60s
Runtime: ~30-40 min wall-clock

# Moisture budget (tuned for 10-20% cloud cover)
QV_SURF = 0.016       # kg/kg
QV_SCALE = 4000.0     # m (was 2500 → dried BL too fast)
TAU_NUDGE_Q = 3600.0  # s (was 1800 → removed moisture too aggressively)

# Forcing
SHF = 200.0, LHF = 100.0  # W/m²
W_SUBS = 0.005            # m/s subsidence
TAU_NUDGE_T = 3600.0      # s theta nudging

# SGS
Smagorinsky-Lilly: CS=0.18, PR_T=0.33
```

### Nested Simulation (25m resolution, working config 2026-09-22)
```python
Grid: 100×100×60, dx=25m, domain 2.5×2.5×3km
Time: dt=0.5s, T_END=900s (15 min), output every 30s
Runtime: ~25 min wall-clock (0.65× realtime)

# Initial conditions: interpolated from parent snapshot
# No surface fluxes (cloud already exists)
# Relaxation BC: 8-cell zone, tau_min=5.0s → nudge toward parent at edges
```

**Failed 10m resolution attempt:** 250×250×60 grid → 0.04× realtime (25× slower). Bottleneck: WENO-3 advection + Smagorinsky (many gradients) scales as O(n²) horizontally. 25m grid (6.25× fewer cells) gives 16× speedup.

---

## Run Management (2026-09-22)

### Metadata System
All runs tracked with YAML metadata:
```bash
# Nested run structure
nested_runs/YYYY-MM-DD_HHMM_nest_cloudN_dxXXm/
├── metadata.yaml          # parent lineage, grid, config, status
├── config_nested.yaml     # archived config
├── nested_ic.npz          # initial conditions
├── diag.csv               # diagnostics
├── sim.log                # output log
└── snapshots/             # snap_*.npz

# Archive current run
./venv/bin/python3 scripts/archive_current_nested.py
```

Metadata includes: run ID, parent source (experiment/snapshot/cloud), grid config, physics params, completion status.

---

## Common Workflows

### Run parent simulation
```bash
MPLBACKEND=Agg nohup ./venv/bin/python3 main.py > sim.log 2>&1 &
tail -f diag.csv | column -t -s,
```

### Find cloud for nested run
```bash
# Search snapshot
./venv/bin/python3 scripts/find_cloud.py \
  --snapshot experiments/.../snap_0001200.npz \
  --plot

# Track cloud lifetime
./venv/bin/python3 scripts/track_cloud.py
```

### Setup and run nested simulation
```bash
# 1. Create nested IC
./venv/bin/python3 scripts/setup_nested.py \
  --parent-snapshot experiments/.../snap_0001200.npz \
  --cloud-center 44,56,44 \
  --nest-size 2.5 \
  --dx-fine 25.0 \
  --output nested_ic.npz

# 2. Update config_nested.py: verify NX, NY, DX, DT, T_END

# 3. Run
MPLBACKEND=Agg nohup ./venv/bin/python3 main_nested.py > nested_sim.log 2>&1 &

# 4. Archive when complete
./venv/bin/python3 scripts/archive_current_nested.py
```

### Visualize
```bash
./venv/bin/python3 scripts/visualize/visualize3d.py  # GIF animation
./venv/bin/python3 scripts/visualize/visualize3d.py --fps 10 --out test.mp4
```

---

## Diagnostics

Key `diag.csv` columns:
- `qv_bl` — BL moisture (g/kg), should stay ~10-11 g/kg
- `cloud_frac` — cloud cover, target 0.10-0.20 for parent
- `qc_max` — max cloud water (g/kg), healthy clouds 0.5-5.0
- `w_max` — max vertical velocity (m/s), expect 5-15
- `n_clouds` — cloud count
- `CB`, `CT` — cloud base/top (m)

---

## Critical Issues & Solutions

### Moisture budget imbalance (resolved 2026-09-21)
**Problem:** QV_SCALE=2500m → BL mean qv 25% lower than surface → dried over time → LCL too high.
**Solution:** QV_SCALE=4000m, TAU_NUDGE_Q=3600s → stable ~10 g/kg BL moisture → 10-20% cloud cover.

### Restart file corruption (resolved 2026-09-21)
**Problem:** `snapshots/` shared across experiments → overwrite snap_0001200.npz → contaminated IC.
**Solution:** `restart_snapshots/` directory (immutable), copy to `snapshots/` before each run.

### Nested performance bottleneck (resolved 2026-09-22)
**Problem:** 10m resolution (250×250 grid) → 25× slower than realtime → 6 hours for 15 min.
**Solution:** Use 25m resolution (100×100 grid) → 0.65× realtime → 25 min wall-clock. Still 4× finer than parent.
**Bottleneck:** WENO-3 advection + Smagorinsky (many gradient computations), scales O(n²) horizontally.

---

## Working with Claude Code

**DO NOT create throwaway scripts** — directory already cluttered.

For one-off tests:
```bash
# ✅ GOOD: inline code
./venv/bin/python3 << 'EOF'
import numpy as np
# test code here
EOF

# ❌ BAD: test_xyz.py, debug_abc.py, temp_*.py
```

Use existing tools: `find_cloud.py`, `track_cloud.py`, `visualize3d.py`.

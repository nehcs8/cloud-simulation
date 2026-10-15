"""land_cu run on a 64x64 grid with a process-by-process water budget and dt statistics.

Usage: python run_land_cu.py [T_END_seconds] [KEY=VALUE ...] [TAG=name]   (default 7200)

KEY=VALUE overrides a config value after land_cu.configure() (value parsed as a
Python literal, e.g. SHF=110 TAU_NUDGE_Q=5400). TAG=name is appended to the run
folder name: runs/<ts>_land_cu_<min>min_<name>/.

Total water W = Σ ρ₀ (qv + qc + qr) dV [kg]. Each physics call in rk3_step is
wrapped to record its ΔW; the residual (ΔW minus all physics terms) is what
the dynamics (advection + projection) and clipping did. Written every
OUTPUT_EVERY to <run dir>/water_budget.csv.
"""
import ast
import csv
import sys
import time
import numpy as np
import config as cfg
from cases import land_cu

cfg.CASE = 'land_cu'
land_cu.configure()         # case settings; overrides below take precedence
cfg.NX = 64
cfg.NY = 64
args = sys.argv[1:]
cfg.T_END = float(args.pop(0)) if args and '=' not in args[0] else 7200.0
cfg.CASE_NAME = f'land_cu_{int(cfg.T_END / 60)}min'
cfg.OUTPUT_EVERY = 60.0

for arg in args:
    key, val = arg.split('=', 1)
    if key == 'TAG':
        cfg.CASE_NAME += f'_{val}'
        continue
    if not hasattr(cfg, key):
        sys.exit(f'Unknown config key: {key}')
    setattr(cfg, key, ast.literal_eval(val))
    print(f'override: {key} = {getattr(cfg, key)!r}')

import timestepping
import simulation
from simulation import Simulation

PROCESSES = ['apply_surface_layer', 'apply_microphysics', 'apply_cold_pool',
             'apply_subsidence', 'apply_profile_nudging', 'apply_radiative_cooling',
             'apply_sponge', 'apply_fringe']
budget = {p: 0.0 for p in PROCESSES}


def total_water(s, g):
    dV = g.dx * g.dy * g.dz_face
    return float(((s.qv + s.qc + s.qr).sum(axis=(0, 1)) * s.rho0 * dV).sum())


def _wrap(name):
    fn = getattr(timestepping, name)
    def wrapped(s, g, *args, **kw):
        w0 = total_water(s, g)
        out = fn(s, g, *args, **kw)
        budget[name] += total_water(s, g) - w0
        return out
    setattr(timestepping, name, wrapped)


for p in PROCESSES:
    _wrap(p)


class BudgetSimulation(Simulation):
    def __init__(self):
        super().__init__()
        self.W0 = total_water(self.state, self.grid)
        self._bf = open(f"{simulation.OUTPUT_DIR}/water_budget.csv", 'w', newline='')
        self._bw = csv.writer(self._bf)
        self._bw.writerow(['t', 'W_kg', 'dW_pct'] + [f'{p}_pct' for p in PROCESSES]
                          + ['residual_dynamics_pct'])
        self._next_budget = 0.0
        self.dts = []
        self.wall0 = time.time()

    def _advance(self, dt):
        super()._advance(dt)
        self.dts.append(dt)
        if self.t + dt >= self._next_budget - 1e-6:
            W = total_water(self.state, self.grid)
            pct = lambda x: 100.0 * x / self.W0
            dW = W - self.W0
            resid = dW - sum(budget.values())
            self._bw.writerow([f"{self.t + dt:.1f}", f"{W:.8e}", f"{pct(dW):+.5f}"]
                              + [f"{pct(budget[p]):+.5f}" for p in PROCESSES]
                              + [f"{pct(resid):+.5f}"])
            self._bf.flush()
            self._next_budget += cfg.OUTPUT_EVERY


print('=' * 70)
print(f'land_cu: {cfg.NX}x{cfg.NY} grid, {cfg.T_END/60:.0f} min simulated, water budget tracking')
print('=' * 70)

sim = BudgetSimulation()
try:
    sim.run()
finally:
    sim._bf.close()

dts = np.array(sim.dts)
wall = time.time() - sim.wall0
print(f"\nSteps: {len(dts)}  mean dt = {dts.mean():.2f} s  (min {dts.min():.2f}, max {dts.max():.2f}; "
      f"output-aligned steps included)")
print(f"Wall time: {wall/60:.1f} min  ({wall/len(dts):.2f} s/step, "
      f"{wall/(sim.t/3600)/60:.1f} wall-min per simulated hour)")
print('\n' + '=' * 70)
print('land_cu RUN COMPLETED')
print('=' * 70)

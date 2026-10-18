"""Run a case with a process-by-process water budget and dt statistics.

Usage: python run_case.py CASE[:VARIANT] [T_END_seconds] [KEY=VALUE ...] [TAG=name]   (default 7200)

CASE is a module in cases/ (land_cu, single_cu). VARIANT picks a preset from the case's
VARIANTS dict (e.g. single_cu:bubble), applied after case.configure(). KEY=VALUE overrides
a config value after that (value parsed as a Python literal, e.g. SHF=110 TAU_NUDGE_Q=5400).
The run folder is runs/<ts>_<case>[_<variant>]_<min>min[_<tag>]/.

Total water W = Σ ρ₀ (qv + qc + qr) dV [kg]. Each physics call in rk3_step is
wrapped to record its ΔW; the residual (ΔW minus all physics terms) is what
the dynamics (advection + projection) and clipping did; water crossing open lateral
boundaries (open_bc.py) is its own column, boundary_outflow_pct. Written every
OUTPUT_EVERY to <run dir>/water_budget.csv.
"""
import ast
import csv
import sys
import time
import numpy as np
import config as cfg
import cases

args = sys.argv[1:]
if not args or args[0].split(':')[0] not in cases.CASES:
    sys.exit(f'Usage: run_case.py CASE[:VARIANT] [T_END] [KEY=VALUE ...]   CASE in {cases.CASES}')
case_name, _, variant = args.pop(0).partition(':')
case = cases.get(case_name)
cfg.CASE = case_name
case.configure()            # case settings; variant and overrides below take precedence
if variant:
    presets = getattr(case, 'VARIANTS', {})
    if variant not in presets:
        sys.exit(f'Unknown variant {variant!r} for {case_name}: {list(presets)}')
    for key, val in presets[variant].items():
        setattr(cfg, key, val)
    print(f'variant: {variant} {presets[variant]}')
cfg.T_END = float(args.pop(0)) if args and '=' not in args[0] else 7200.0
cfg.CASE_NAME = '_'.join(filter(None, [case_name, variant, f'{int(cfg.T_END / 60)}min']))
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
             'apply_sponge', 'apply_fringe', 'apply_open_relaxation']
budget = {p: 0.0 for p in PROCESSES}


def total_water(s, g):
    dV = g.dx * g.dy * g.dz_face
    if g.terrain is not None:                       # fluid cells only (solid cells are reset)
        q = np.where(g.terrain.mask, 0.0, s.qv + s.qc + s.qr)
        return float((q.sum(axis=(0, 1)) * s.rho0 * dV).sum())
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
                          + ['boundary_outflow_pct', 'residual_dynamics_pct'])
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
            out = -self.state.boundary_water_out           # water gained through open boundaries
            resid = dW - sum(budget.values()) - out
            self._bw.writerow([f"{self.t + dt:.1f}", f"{W:.8e}", f"{pct(dW):+.5f}"]
                              + [f"{pct(budget[p]):+.5f}" for p in PROCESSES]
                              + [f"{pct(out):+.5f}", f"{pct(resid):+.5f}"])
            self._bf.flush()
            self._next_budget += cfg.OUTPUT_EVERY


print('=' * 70)
print(f'{cfg.CASE_NAME}: {cfg.NX}x{cfg.NY} grid, {cfg.T_END/60:.0f} min simulated, water budget tracking')
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
print(f'{cfg.CASE_NAME} RUN COMPLETED')
print('=' * 70)

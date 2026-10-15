"""Smoke test runner: 64x64x80 grid, 10 min simulation."""
import config as cfg

# Override config for smoke test
cfg.NX = 64
cfg.NY = 64
cfg.NZ = 80
cfg.T_END = 600.0  # 10 minutes
cfg.OUTPUT_EVERY = 60.0
cfg.CASE_NAME = 'smoke_test'

# Import and run
from simulation import Simulation

print('='*70)
print('SMOKE TEST: 64x64x80 grid, 10 min simulated time')
print('='*70)

sim = Simulation()
sim.run()

print('\n' + '='*70)
print('SMOKE TEST COMPLETED SUCCESSFULLY')
print('='*70)

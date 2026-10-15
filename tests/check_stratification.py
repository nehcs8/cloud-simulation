"""
Check base-state stratification and horizontal-mean theta at t=0.
"""
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config as cfg
from grid import Grid
from state import State

# Use same config as bubble test
cfg.NX = 32
cfg.NY = 32
cfg.NZ = 80
cfg.DX = 100.0
cfg.DY = 100.0
cfg.Z_TOP = 5000.0

grid = Grid(terrain=None)
state = State(grid)

# Initialize with dry warm bubble
x_c = cfg.NX * cfg.DX / 2.0
y_c = cfg.NY * cfg.DY / 2.0
z_c = 500.0
R = 500.0

x = np.arange(cfg.NX) * cfg.DX + cfg.DX / 2.0
y = np.arange(cfg.NY) * cfg.DY + cfg.DY / 2.0
z = grid.z

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
r = np.sqrt((X - x_c)**2 + (Y - y_c)**2 + (Z - z_c)**2)

theta_pert = np.where(r < R, 1.0 * np.cos(0.5 * np.pi * r / R)**2, 0.0)
state.theta[:] = theta_pert

print("="*80)
print("BASE-STATE STRATIFICATION CHECK")
print("="*80)
print()

# Compute dtheta0/dz
dtheta0_dz = np.gradient(state.theta0, z)

print("theta0(z) profile (every 10 levels):")
print(f"{'z (m)':>8} {'theta0 (K)':>12} {'dtheta0/dz (K/m)':>18} {'dtheta0/dz (K/km)':>18}")
print("-"*80)
for i in range(0, len(z), 10):
    print(f"{z[i]:8.1f} {state.theta0[i]:12.6f} {dtheta0_dz[i]:18.9f} {dtheta0_dz[i]*1000:18.6f}")

print()
print("Key levels:")
print(f"  z=0 m (surface):     theta0 = {state.theta0[0]:.6f} K, dtheta0/dz = {dtheta0_dz[0]*1000:.6f} K/km")
idx_500 = np.argmin(np.abs(z - 500))
print(f"  z=500 m (bubble):    theta0 = {state.theta0[idx_500]:.6f} K, dtheta0/dz = {dtheta0_dz[idx_500]*1000:.6f} K/km")
idx_1000 = np.argmin(np.abs(z - 1000))
print(f"  z=1000 m:            theta0 = {state.theta0[idx_1000]:.6f} K, dtheta0/dz = {dtheta0_dz[idx_1000]*1000:.6f} K/km")
idx_1500 = np.argmin(np.abs(z - 1500))
print(f"  z=1500 m (BL top):   theta0 = {state.theta0[idx_1500]:.6f} K, dtheta0/dz = {dtheta0_dz[idx_1500]*1000:.6f} K/km")
idx_2000 = np.argmin(np.abs(z - 2000))
print(f"  z=2000 m (FT):       theta0 = {state.theta0[idx_2000]:.6f} K, dtheta0/dz = {dtheta0_dz[idx_2000]*1000:.6f} K/km")

print()
print("Horizontal-mean theta (absolute, not perturbation) at t=0:")
theta_abs = state.theta + state.theta0[np.newaxis, np.newaxis, :]
theta_mean = theta_abs.mean(axis=(0, 1))  # (nz,) horizontal mean at each level

print(f"{'z (m)':>8} {'<theta> (K)':>14} {'theta0 (K)':>12} {'<theta> - theta0 (K)':>20}")
print("-"*80)
for i in range(0, len(z), 10):
    print(f"{z[i]:8.1f} {theta_mean[i]:14.6f} {state.theta0[i]:12.6f} {theta_mean[i] - state.theta0[i]:20.6f}")

print()
print("="*80)
print("STRATIFICATION DIAGNOSIS")
print("="*80)

# Expected stratification
print("Expected from config:")
print(f"  GAMMA_BL = {cfg.GAMMA_BL:.3f} K/km (well-mixed BL)")
print(f"  Z_BL = {cfg.Z_BL:.1f} m")
if hasattr(cfg, 'Z_INVERSION'):
    print(f"  Z_INVERSION = {cfg.Z_INVERSION:.1f} m")
    print(f"  INVERSION_STRENGTH = {cfg.INVERSION_STRENGTH:.1f} K")
print(f"  GAMMA_FT = {cfg.GAMMA_FT:.3f} K/km (free troposphere)")

print()
# Check if stratification is strong enough
if dtheta0_dz[idx_2000] * 1000 > 5.0:
    print(f"✓ FT stratification = {dtheta0_dz[idx_2000]*1000:.2f} K/km is strong enough to cap bubble")
else:
    print(f"✗ FT stratification = {dtheta0_dz[idx_2000]*1000:.2f} K/km is TOO WEAK")
    print("  Bubble will rise through entire domain (neutral atmosphere)")

print("="*80)

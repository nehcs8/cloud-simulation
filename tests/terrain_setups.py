"""
Shared idealised set-ups for the terrain validation tests (tests/test_terrain_rest.py,
tests/test_mountain_waves.py): a dry atmosphere with constant buoyancy frequency N from the
ground up, no surface fluxes, no surface layer, no initial noise, run through run_case.py.
"""

import os
import sys
import glob
import runpy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

N_BV = 0.01                      # 1/s
THETA_S = 303.0                  # K (single_cu mixed-layer θ, kept from the case)


def dtheta_dz(n=N_BV, theta=THETA_S, g=9.81):
    """Constant dθ/dz giving buoyancy frequency n near the ground."""
    return theta * n**2 / g


def run(t_end: float, overrides: list[str], tag: str):
    """Run single_cu:hotspot (trigger switched off) with the dry constant-N sounding; return the run dir."""
    import cases.single_cu as sc
    import timestepping
    sc.THETA_NOISE = 0.0
    sc.QV_NOISE = 0.0
    sc.Z_TROP = 1.0e5                            # no tropopause inside the domain
    timestepping.apply_surface_layer = lambda *a, **k: None   # free-slip ground, no fluxes
    base = ['SHF=0.0', 'LHF=0.0', 'SHF_SPOT_AMP=0.0', 'SHF_SPOT_LHF_AMP=0.0', 'BUBBLE_DTHETA=0.0',
            'SCU_THETA_ML=303.0', 'SCU_Z_ML=30.0', 'SCU_RH_ML_TOP=0.05',
            f'SCU_LAPSE=((100000.0, {dtheta_dz():.6e}),)', 'SCU_RH=((30.0, 0.05), (100000.0, 0.05))']
    sys.argv = ['run_case.py', 'single_cu:hotspot', str(t_end)] + base + overrides + [f'TAG={tag}']
    cwd = os.getcwd()
    runpy.run_path(os.path.join(cwd, 'run_case.py'), run_name='__main__')
    return sorted(glob.glob(os.path.join(cwd, 'runs', f'*_{tag}')))[-1]

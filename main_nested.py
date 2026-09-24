#!/usr/bin/env python3
"""
Nested cloud simulation - main entry point.

Runs high-resolution simulation of a single cloud from parent domain.
Uses relaxation BC instead of periodic, and parent profiles for forcing.
"""

import sys
import time
import numpy as np
from pathlib import Path

# CRITICAL: Replace config with config_nested BEFORE importing other modules
# This ensures Grid, State, etc. use nested parameters
import config_nested
sys.modules['config'] = config_nested
import config as cfg  # Now points to config_nested

# Standard simulation components (reuse from parent)
from grid import Grid
from state import State
from dynamics import compute_tendencies
from pressure import project
from microphysics import apply_microphysics

# Nested-specific modules
from boundary_conditions import create_boundary_handler
import forcing_nested as forcing

# Simulation infrastructure
from simulation import Simulation
from scripts.run_manager import RunManager, config_to_dict

print("=" * 70)
print("NESTED CLOUD SIMULATION")
print("=" * 70)
print(f"Resolution: {cfg.DX}m × {cfg.DY}m")
print(f"Domain: {cfg.NX*cfg.DX/1000:.1f} × {cfg.NY*cfg.DY/1000:.1f} × {cfg.Z_TOP/1000:.1f} km")
print(f"Duration: {cfg.T_END}s ({cfg.T_END/60:.1f} min)")
print("=" * 70)

# Extract cloud info from config for metadata
def _extract_cloud_info():
    """Extract cloud information from config."""
    # Load parent snapshot to get cloud properties
    parent_snap = np.load(cfg.PARENT_SNAPSHOT)
    i, j, k = cfg.CLOUD_CENTER
    qc_max = (parent_snap['qc'][i, j, k] * 1000)  # g/kg
    t_parent = float(parent_snap['t'])

    # Try to infer cloud ID from archived runs or default
    cloud_id = getattr(cfg, 'CLOUD_ID', 'unknown')

    return {
        'cloud_id': cloud_id,
        'center': cfg.CLOUD_CENTER,
        'time_parent': t_parent,
        'qc_max': qc_max,
    }


class NestedSimulation(Simulation):
    """
    Nested simulation with modified BC and forcing.

    Inherits most functionality from parent Simulation class,
    overrides boundary conditions and forcing functions.
    """

    def __init__(self, run_dir: Path):
        """Initialize nested simulation.

        Parameters
        ----------
        run_dir : Path
            Directory for this simulation run (created by RunManager)
        """
        self.run_dir = run_dir
        self.snapshot_dir = run_dir / "snapshots"
        self.diag_path = run_dir / "diag.csv"

        # Call parent init (creates grid, state, viz, etc.)
        # But use config_nested instead of config
        self.grid = Grid(terrain=None)  # flat domain
        self.state = State(self.grid, restart_file=cfg.RESTART_FROM)

        # No surface fluxes in nested (cloud already exists)
        self.shf_map = np.zeros((self.grid.nx, self.grid.ny))
        self.lhf_map = np.zeros((self.grid.nx, self.grid.ny))

        self.t = 0.0
        self.step = 0

        # No live visualization (too slow for fine grid)
        self.viz = None
        self.enable_viz = False  # Flag for run() method

        # Setup CSV output in run directory
        import csv
        self.snapshot_dir.mkdir(exist_ok=True)
        self._csv = open(self.diag_path, "w", newline="")
        from simulation import _DIAG_FIELDS
        self._writer = csv.DictWriter(self._csv, fieldnames=_DIAG_FIELDS)
        self._writer.writeheader()
        self._csv.flush()

        # Setup boundary conditions
        self.bc_handler = create_boundary_handler(cfg)

        # Load parent data for BC and forcing
        self._load_parent_data()

        print(f"\n{self.grid}")
        print(f"  Restart: {cfg.RESTART_FROM}")
        print(f"  dt={cfg.DT}s, t_end={cfg.T_END}s")

    def run(self) -> None:
        """Run without visualization (override parent method)."""
        next_output = 0.0
        last_dt = cfg.DT

        try:
            while self.t < cfg.T_END:
                last_dt = self._cfl_dt()
                dt = last_dt
                if self.t + dt > next_output:
                    dt = max(next_output - self.t, 1e-6)

                self._advance(dt)
                self.t += dt
                self.step += 1

                if self.t >= next_output - 1e-6:
                    d = self._diagnostics(last_dt)
                    self._report(d)
                    self._writer.writerow({k: f"{v:.6g}" for k, v in d.items()})
                    self._csv.flush()
                    # Skip visualization
                    self._save_snapshot(d)
                    next_output += cfg.OUTPUT_EVERY
        finally:
            self._csv.close()

    def _load_parent_data(self):
        """Load parent fields for BC and forcing."""
        print("\nLoading parent data for BC and forcing...")

        # Parent fields are in the nested_ic file
        data = np.load(cfg.RESTART_FROM)

        # For BC: need parent fields at same resolution as nest
        # (already interpolated during setup)
        self.parent_fields = {
            'u': data['u'],
            'v': data['v'],
            'w': data['w'],
            'theta': data['theta'],
            'qv': data['qv'],
            'qc': data['qc'],
            'qr': data['qr'],
        }

        # For forcing: parent mean profiles
        self.parent_profiles = {
            'theta0': data['prof_theta'],
            'qv0': data['prof_qv'],
            'theta_mean': data['theta_mean'],
            'qv_mean': data['qv_mean'],
        }

        print(f"  ✓ Loaded parent data")

    def _advance(self, dt: float) -> None:
        """Single timestep with nested BC and forcing."""
        s, g = self.state, self.grid

        # 1. Tendencies (same as parent)
        tend = compute_tendencies(s, g)

        # 2. Forward-Euler
        s.u += dt * tend["u"]
        s.v += dt * tend["v"]
        s.w += dt * tend["w"]
        s.theta += dt * tend["theta"]
        s.qv += dt * tend["qv"]
        s.qc += dt * tend["qc"]
        s.qr += dt * tend["qr"]

        s.qc = np.maximum(s.qc, 0.0)
        s.qr = np.maximum(s.qr, 0.0)

        # 3. NO surface fluxes (cloud already exists)

        # 4. Microphysics (same)
        apply_microphysics(s, g, dt)

        # 5. Cold pool (same)
        forcing.apply_cold_pool_nested(s, g, dt)

        # 6. Pressure projection (same)
        project(s.u, s.v, s.w, g, dt)

        # 7. NESTED FORCING (use parent profiles)
        forcing.apply_subsidence_nested(s, self.parent_profiles, g, dt)
        forcing.apply_profile_nudging_nested(s, self.parent_profiles, g, dt)

        # 8. Sponge (same)
        forcing.apply_sponge_nested(s, g, dt)

        # 9. RELAXATION BC (key difference from parent)
        if self.bc_handler is not None:
            self.bc_handler.apply(s, self.parent_fields, dt)

        # 10. Standard BCs
        s.u[:, :, 0] = 0.0
        s.v[:, :, 0] = 0.0
        s.qv = np.maximum(s.qv, 0.0)
        s.qc = np.maximum(s.qc, 0.0)
        s.qr = np.maximum(s.qr, 0.0)

    def _save_snapshot(self, d: dict) -> None:
        """Save snapshot to run directory (override parent method)."""
        s, g = self.state, self.grid
        path = self.snapshot_dir / f"snap_{self.t:07.0f}.npz"
        np.savez_compressed(
            path,
            t=np.array(self.t),
            qc=s.qc.astype(np.float32),
            qr=s.qr.astype(np.float32),
            qv=s.qv.astype(np.float32),
            w=(0.5 * (s.w[:, :, :-1] + s.w[:, :, 1:])).astype(np.float32),
            theta=s.theta.astype(np.float32),
            z_face=g.z_face.astype(np.float32),
            z=g.z.astype(np.float32),
            prof_qc=s.qc.mean(axis=(0, 1)).astype(np.float32),
            prof_qr=s.qr.mean(axis=(0, 1)).astype(np.float32),
            prof_qv=s.qv.mean(axis=(0, 1)).astype(np.float32),
            prof_theta=s.theta.mean(axis=(0, 1)).astype(np.float32),
            prof_w=(0.5 * (s.w[:, :, :-1] + s.w[:, :, 1:])).mean(axis=(0, 1)).astype(np.float32),
        )


def main():
    """Run nested simulation with automatic metadata and organization."""
    print("\nSetting up run directory and metadata...")

    # Create run manager
    manager = RunManager()

    # Extract cloud info
    cloud_info = _extract_cloud_info()

    # Create run directory with metadata
    run_dir = manager.create_nested_run(
        parent_snapshot=cfg.PARENT_SNAPSHOT,
        cloud_info=cloud_info,
        config_dict=config_to_dict(cfg),
        resolution_m=cfg.DX,
        duration_s=cfg.T_END,
    )

    print(f"\nRun directory: {run_dir.name}")
    print("\nInitializing nested simulation...")
    start_time = time.time()

    sim = NestedSimulation(run_dir)

    init_time = time.time() - start_time
    print(f"Initialization complete ({init_time:.1f}s)")

    print(f"\nStarting time integration...")
    print(f"Expected timesteps: ~{int(cfg.T_END / cfg.DT)}")

    run_start = time.time()
    status = "complete"
    notes = ""

    try:
        sim.run()
        run_time = time.time() - run_start
        total_time = time.time() - start_time

        # Compile statistics for notes
        import csv
        with open(sim.diag_path, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                last_line = lines[-1].strip().split(',')
                header = lines[0].strip().split(',')
                data = dict(zip(header, last_line))

                notes = f"""Simulation completed successfully.
Wall-clock time: {total_time:.1f}s ({total_time/60:.1f} min)
Speed: {cfg.T_END/total_time:.2f}× realtime

Final statistics (t={cfg.T_END}s):
  - Max vertical velocity: {float(data.get('w_max', 0)):.2f} m/s
  - Max cloud water: {float(data.get('qc_max', 0)):.2f} g/kg
  - Cloud cover: {float(data.get('cloud_frac', 0))*100:.1f}%
  - Total timesteps: {int(float(data.get('step', 0)))}
  - Average dt: {cfg.T_END/sim.step:.3f}s
  - Snapshots saved: {len(list(sim.snapshot_dir.glob('*.npz')))}"""

        print(f"\n{'='*70}")
        print(f"SIMULATION COMPLETE")
        print(f"{'='*70}")
        print(f"Wall-clock time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"Simulation time: {cfg.T_END:.0f}s ({cfg.T_END/60:.1f} min)")
        print(f"Speed: {cfg.T_END/total_time:.2f}× realtime")
        print(f"Timesteps: {sim.step}")
        print(f"Average dt: {cfg.T_END/sim.step:.3f}s")

    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
        status = "cancelled"
        notes = f"Interrupted at t={sim.t:.0f}s (step {sim.step})"
    except Exception as e:
        print(f"\n\nSimulation FAILED: {e}")
        status = "failed"
        notes = f"Failed at t={sim.t:.0f}s: {str(e)}"
        import traceback
        traceback.print_exc()

    # Finalize metadata
    manager.finalize_run(run_dir, status=status, notes=notes)

    print(f"\n{'='*70}")
    print(f"Run directory: {run_dir}")
    print(f"Metadata: {run_dir / 'metadata.yaml'}")
    print(f"Diagnostics: {run_dir / 'diag.csv'}")
    print(f"Snapshots: {run_dir / 'snapshots'}/ ({len(list(sim.snapshot_dir.glob('*.npz')))} files)")
    print(f"{'='*70}")

    return 0 if status == "complete" else 1


if __name__ == '__main__':
    sys.exit(main())

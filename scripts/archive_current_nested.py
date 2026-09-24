#!/usr/bin/env python3
"""
Archive the currently completed nested simulation using the run management system.
"""

import sys
import shutil
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_manager import RunManager
import config_nested as cfg

def main():
    """Archive current nested simulation."""

    # Initialize run manager
    manager = RunManager()

    # Cloud information (from our cloud selection)
    cloud_info = {
        "cloud_id": 16,
        "center": (44, 56, 44),
        "time_parent": 1200.0,  # seconds
        "qc_max": 3.95,  # g/kg
    }

    # Parent snapshot source
    parent_snapshot = "experiments/phase2_nudging/exp13_tau3600/snapshots/snap_0001200.npz"

    # Create config dict from config_nested
    config_dict = {
        k: getattr(cfg, k)
        for k in dir(cfg)
        if not k.startswith("_") and k.isupper()
    }

    # Create nested run directory
    run_dir = manager.create_nested_run(
        parent_snapshot=parent_snapshot,
        cloud_info=cloud_info,
        config_dict=config_dict,
        resolution_m=cfg.DX,
        duration_s=cfg.T_END,
    )

    print(f"\nRun directory created: {run_dir}")

    # Move outputs to run directory
    base_dir = Path.cwd()

    # Move snapshots
    snapshots_src = base_dir / "snapshots"
    if snapshots_src.exists():
        print("\nMoving snapshots...")
        for snap in snapshots_src.glob("snap_*.npz"):
            shutil.move(str(snap), str(run_dir / "snapshots" / snap.name))

    # Move diagnostics
    diag_src = base_dir / "diag.csv"
    if diag_src.exists():
        print("Moving diag.csv...")
        shutil.move(str(diag_src), str(run_dir / "diag.csv"))

    # Move log
    log_src = base_dir / "outputs" / "logs" / "nested_sim.log"
    if log_src.exists() and log_src.stat().st_size > 0:
        print("Moving nested_sim.log...")
        shutil.move(str(log_src), str(run_dir / "sim.log"))

    # Copy nested IC
    ic_src = base_dir / "nested_ic.npz"
    if ic_src.exists():
        print("Copying nested_ic.npz...")
        shutil.copy(str(ic_src), str(run_dir / "nested_ic.npz"))

    # Finalize run with statistics
    print("\nFinalizing run metadata...")

    # Read final diagnostics for summary
    diag_path = run_dir / "diag.csv"
    if diag_path.exists():
        # Read last line of CSV (no pandas needed)
        with open(diag_path, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                last_line = lines[-1].strip().split(',')
                header = lines[0].strip().split(',')
                data = dict(zip(header, last_line))

                notes = f"""Simulation completed successfully.
Final statistics (t={cfg.T_END}s):
  - Max vertical velocity: {float(data.get('w_max', 0)):.2f} m/s
  - Max cloud water: {float(data.get('qc_max', 0)):.2f} g/kg
  - Cloud cover: {float(data.get('cloud_frac', 0))*100:.1f}%
  - Total timesteps: {int(float(data.get('step', 0)))}
  - Snapshots saved: {len(list((run_dir / 'snapshots').glob('*.npz')))}
"""
            else:
                notes = "Diagnostics file empty."
    else:
        notes = "Outputs archived successfully."

    manager.finalize_run(run_dir, status="complete", notes=notes)

    print(f"\n{'='*60}")
    print("NESTED RUN ARCHIVED")
    print(f"{'='*60}")
    print(f"Location: {run_dir}")
    print(f"Run ID: {run_dir.name}")
    print(f"\nMetadata saved to: {run_dir / 'metadata.yaml'}")
    print(f"Config archived to: {run_dir / 'config_nested.yaml'}")

    # Display metadata
    import yaml
    with open(run_dir / "metadata.yaml") as f:
        metadata = yaml.safe_load(f)

    print(f"\n{'='*60}")
    print("RUN METADATA")
    print(f"{'='*60}")
    print(yaml.dump(metadata, default_flow_style=False, sort_keys=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())

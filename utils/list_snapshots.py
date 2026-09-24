#!/usr/bin/env python3
"""
List available snapshots with useful metadata for choosing restart points.
"""

import os
import numpy as np
from pathlib import Path

SNAPSHOT_DIR = "snapshots"

def list_snapshots():
    """List all snapshots with key metrics."""
    snap_dir = Path(SNAPSHOT_DIR)

    if not snap_dir.exists():
        print(f"No snapshot directory found: {SNAPSHOT_DIR}")
        return

    snapshots = sorted(snap_dir.glob("snap_*.npz"))

    if not snapshots:
        print(f"No snapshots found in {SNAPSHOT_DIR}")
        return

    print(f"Available snapshots in {SNAPSHOT_DIR}:")
    print(f"{'File':<25} {'Time':>8} {'qc_max':>10} {'qr_max':>10} {'w_max':>8} {'Cloud%':>8}")
    print("-" * 85)

    for snap_file in snapshots:
        try:
            data = np.load(snap_file)
            t = float(data['t'])
            qc = data['qc']
            qr = data['qr']
            w = data['w']

            qc_max = qc.max() * 1e3  # g/kg
            qr_max = qr.max() * 1e3  # g/kg
            w_max = w.max()  # m/s

            # Cloud fraction (column-max qc > threshold)
            cloud_frac = (qc.max(axis=2) > 1e-5).mean() * 100  # %

            print(f"{snap_file.name:<25} {t:7.0f}s {qc_max:8.2f} g/kg {qr_max:8.2f} g/kg "
                  f"{w_max:7.2f} m/s {cloud_frac:7.1f}%")
        except Exception as e:
            print(f"{snap_file.name:<25} Error reading: {e}")

    print()
    print("To restart from a snapshot, set in config.py:")
    print('  RESTART_FROM = "snapshots/snap_XXXXXXX.npz"')


if __name__ == "__main__":
    list_snapshots()

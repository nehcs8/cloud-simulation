#!/usr/bin/env python3
"""Monitor progress of nested simulation."""

import sys
import time
import os

# Check if simulation is running
os.system("ps aux | grep '[m]ain_nested.py' > /dev/null")
if os.system("ps aux | grep '[m]ain_nested.py' > /dev/null") != 0:
    print("❌ Nested simulation is NOT running")
    sys.exit(1)

print("✓ Nested simulation is running\n")

# Read last line of diag.csv
try:
    with open("diag.csv", "r") as f:
        lines = f.readlines()
        if len(lines) > 1:
            header = lines[0].strip().split(',')
            last_line = lines[-1].strip().split(',')

            # Parse key fields
            data = dict(zip(header, last_line))
            t = float(data['t'])
            qc_max = float(data['qc_max']) * 1000  # g/kg
            w_max = float(data['w_max'])
            cloud_frac = float(data['cloud_frac']) * 100  # percent
            n_clouds = int(data['n_clouds'])

            # Calculate progress
            t_end = 7200.0  # from config_nested
            progress_pct = 100 * t / t_end

            # Estimate time remaining (assuming 1.1× realtime)
            elapsed_simtime = t
            remaining_simtime = t_end - t
            remaining_walltime = remaining_simtime / 1.1  # seconds

            print(f"Progress: {progress_pct:.1f}% ({t:.0f}s / {t_end:.0f}s)")
            print(f"ETA: {remaining_walltime/60:.0f} minutes remaining")
            print(f"\nCloud state:")
            print(f"  qc_max:      {qc_max:.2f} g/kg")
            print(f"  w_max:       {w_max:.1f} m/s")
            print(f"  cloud_frac:  {cloud_frac:.1f}%")
            print(f"  n_clouds:    {n_clouds}")

            # Count snapshots
            import glob
            snapshots = glob.glob("snapshots/snap_*.npz")
            print(f"\nSnapshots created: {len(snapshots)}")

        else:
            print("Simulation just started, no data yet")

except FileNotFoundError:
    print("diag.csv not found - simulation may not have started yet")
except Exception as e:
    print(f"Error reading diagnostics: {e}")

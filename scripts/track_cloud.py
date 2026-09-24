#!/usr/bin/env python3
"""Track a cloud through time in parent snapshots."""

import numpy as np
from pathlib import Path
from scipy.ndimage import label

# Cloud location at t=1200s
target_i, target_j, target_k = 44, 56, 44

# Find all snapshots
snapshot_dir = Path("experiments/phase2_nudging/exp13_tau3600/snapshots")
snapshots = sorted(snapshot_dir.glob("snap_*.npz"))

print(f"Tracking cloud at parent grid location ({target_i}, {target_j}, {target_k})")
print(f"Found {len(snapshots)} snapshots from t=0 to t={len(snapshots)*60}s")
print()
print(f"{'Time (s)':<10} {'Time (min)':<12} {'qc @ center':<15} {'Max qc nearby':<15} {'Cloud cells':<12} {'Status':<20}")
print("-" * 95)

cloud_active = []

for snap_path in snapshots:
    data = np.load(snap_path)
    t = float(data['t'])
    qc = data['qc'] * 1000  # g/kg

    # Check qc at the target location
    qc_at_center = qc[target_i, target_j, target_k]

    # Check nearby region (±5 cells in x,y, ±3 in z)
    i_min, i_max = max(0, target_i-5), min(qc.shape[0], target_i+6)
    j_min, j_max = max(0, target_j-5), min(qc.shape[1], target_j+6)
    k_min, k_max = max(0, target_k-3), min(qc.shape[2], target_k+4)

    region = qc[i_min:i_max, j_min:j_max, k_min:k_max]
    qc_max_nearby = region.max()

    # Count cloud cells in region (qc > 0.5 g/kg)
    cloud_mask = region > 0.5
    cloud_cells = cloud_mask.sum()

    # Determine status
    if qc_at_center > 1.0:
        status = "ACTIVE (center)"
        is_active = True
    elif qc_max_nearby > 1.0:
        status = "ACTIVE (nearby)"
        is_active = True
    elif qc_max_nearby > 0.1:
        status = "Weak/forming"
        is_active = False
    else:
        status = "No cloud"
        is_active = False

    cloud_active.append((t, is_active, qc_max_nearby))

    print(f"{t:<10.0f} {t/60:<12.1f} {qc_at_center:<15.3f} {qc_max_nearby:<15.3f} {cloud_cells:<12d} {status:<20}")

# Analyze lifetime
print("\n" + "=" * 95)
print("CLOUD LIFETIME ANALYSIS:")
print("=" * 95)

# Find when cloud becomes active (qc > 1.0 g/kg)
active_times = [t for t, active, _ in cloud_active if active]

if active_times:
    t_birth = active_times[0]
    t_death = active_times[-1]
    lifetime = t_death - t_birth

    print(f"\nCloud appears (qc > 1.0 g/kg):    t = {t_birth:.0f}s  ({t_birth/60:.1f} min)")
    print(f"Cloud dissipates (qc < 1.0 g/kg): t = {t_death:.0f}s  ({t_death/60:.1f} min)")
    print(f"Cloud lifetime:                   {lifetime:.0f}s ({lifetime/60:.1f} min)")

    # Find peak
    peak_t, _, peak_qc = max(cloud_active, key=lambda x: x[2])
    print(f"\nPeak intensity: {peak_qc:.2f} g/kg at t = {peak_t:.0f}s ({peak_t/60:.1f} min)")

    # Check if we're capturing it at t=1200
    print(f"\nSelected snapshot at t=1200s ({1200/60:.1f} min):")
    if t_birth <= 1200 <= t_death:
        age_at_snapshot = 1200 - t_birth
        remaining = t_death - 1200
        print(f"  Cloud age: {age_at_snapshot:.0f}s ({age_at_snapshot/60:.1f} min)")
        print(f"  Remaining lifetime: {remaining:.0f}s ({remaining/60:.1f} min)")

        if 1200 < peak_t:
            print(f"  Status: Still growing (peak at t={peak_t:.0f}s)")
        elif 1200 > peak_t:
            print(f"  Status: Decaying (peaked at t={peak_t:.0f}s)")
        else:
            print(f"  Status: At peak intensity")
    else:
        print(f"  WARNING: Cloud not active at t=1200s!")
else:
    print("\nNo active cloud found in this region!")

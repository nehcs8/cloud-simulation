#!/usr/bin/env python3
"""
Plot horizontal slices at multiple altitudes for a selected cumulus cloud.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

# Load snapshot
snapshot_path = "experiments/phase2_nudging/exp13_tau3600/snapshots/snap_0001200.npz"
data = np.load(snapshot_path)

qc = data['qc'] * 1000  # g/kg
z = data['z']
t = float(data['t'])

# Find Cloud 16 (center at i=44, j=56, k=44)
cloud_mask = qc > 1.0
labeled, num_clouds = label(cloud_mask)

# Get cloud 16 properties
# Find the cloud at position (44, 56, 44)
target_i, target_j, target_k = 44, 56, 44
cloud_id = labeled[target_i, target_j, target_k]

if cloud_id == 0:
    print("Warning: No cloud found at target position, using strongest cloud")
    # Find strongest cloud instead
    cloud_id = None
    max_qc = 0
    for cid in range(1, num_clouds + 1):
        cloud_mask_i = labeled == cid
        qc_max = qc[cloud_mask_i].max()
        if qc_max > max_qc:
            max_qc = qc_max
            cloud_id = cid

cloud_cells = labeled == cloud_id
indices = np.where(cloud_cells)

# Get cloud extent
k_min, k_max = indices[2].min(), indices[2].max()
k_range = k_max - k_min

# Select 4 altitude levels: lowest, two intermediate, highest
k_levels = [
    k_min,
    k_min + k_range // 3,
    k_min + 2 * k_range // 3,
    k_max
]

# Create 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, k in enumerate(k_levels):
    ax = axes[idx]

    # Horizontal slice at this altitude
    qc_slice = qc[:, :, k]

    # Plot
    im = ax.contourf(
        qc_slice,
        levels=np.linspace(0, 6, 25),
        cmap='Blues',
        extend='max'
    )

    # Contour lines
    ax.contour(
        qc_slice,
        levels=[0.5, 1.0, 2.0, 4.0],
        colors='black',
        linewidths=0.8,
        alpha=0.6
    )

    # Mark cloud cells at this level
    cloud_slice = cloud_cells[:, :, k]
    if cloud_slice.any():
        ax.contour(
            cloud_slice,
            levels=[0.5],
            colors='red',
            linewidths=2,
            linestyles='--',
            alpha=0.8
        )

    # Labels
    altitude_label = ['Lowest', '1/3 height', '2/3 height', 'Highest'][idx]
    ax.set_title(f'{altitude_label} level: z = {z[k]:.0f} m (k={k})', fontsize=12, fontweight='bold')
    ax.set_xlabel('x (grid index)')
    ax.set_ylabel('y (grid index)')
    ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_aspect('equal')

    # Add colorbar to each subplot
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('qc (g/kg)', fontsize=10)

    # Add stats text
    qc_at_level = qc_slice[cloud_slice] if cloud_slice.any() else []
    if len(qc_at_level) > 0:
        stats_text = f'Cloud cells: {cloud_slice.sum()}\nMax qc: {qc_at_level.max():.2f} g/kg\nMean qc: {qc_at_level.mean():.2f} g/kg'
    else:
        stats_text = 'No cloud at this level'

    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Overall title
fig.suptitle(f'Cumulus Cloud Horizontal Slices\n'
             f'Cloud ID {cloud_id} at t={t/60:.1f} min\n'
             f'Cloud base: {z[k_min]:.0f}m, Cloud top: {z[k_max]:.0f}m',
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('cumulus_horizontal_slices.png', dpi=200, bbox_inches='tight')
print(f"\n✓ Plot saved to cumulus_horizontal_slices.png")
print(f"\nCloud properties:")
print(f"  Cloud ID: {cloud_id}")
print(f"  Altitude range: {z[k_min]:.0f} - {z[k_max]:.0f} m")
print(f"  Vertical extent: {z[k_max] - z[k_min]:.0f} m")
print(f"  Total volume: {cloud_cells.sum()} cells")
print(f"  Max qc: {qc[cloud_cells].max():.2f} g/kg")
print(f"  Mean qc: {qc[cloud_cells].mean():.2f} g/kg")

plt.show()

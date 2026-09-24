#!/usr/bin/env python3
"""
Find suitable clouds in parent simulation snapshots for nested runs.

Usage:
    python find_cloud.py --snapshot experiments/phase1_qvsurf/exp08_qv0160/snapshots/snap_0001200.npz
"""

import argparse
import numpy as np
from pathlib import Path
from scipy.ndimage import label, center_of_mass
import matplotlib.pyplot as plt

def find_clouds(snapshot_path, min_qc=1.0, min_volume=10):
    """
    Detect cloud objects in a snapshot.

    Parameters:
    -----------
    snapshot_path : Path
        Parent snapshot file
    min_qc : float
        Minimum qc threshold (g/kg) to define cloud
    min_volume : int
        Minimum number of cells to count as cloud

    Returns:
    --------
    clouds : list of dict
        Each dict contains: center (i,j,k), size, qc_max, bounds
    """
    data = np.load(snapshot_path)
    qc = data['qc'] * 1000  # Convert to g/kg
    t = float(data['t'])

    # 3D cloud detection
    cloud_mask = qc > min_qc
    labeled, num_clouds = label(cloud_mask)

    print(f"\nSnapshot: {snapshot_path.name}")
    print(f"Time: t = {t:.0f}s ({t/60:.1f} min)")
    print(f"Found {num_clouds} cloud objects with qc > {min_qc} g/kg")
    print()

    clouds = []
    for cloud_id in range(1, num_clouds + 1):
        cloud_cells = labeled == cloud_id
        volume = cloud_cells.sum()

        if volume < min_volume:
            continue

        # Cloud properties
        qc_cloud = qc[cloud_cells]
        qc_max = qc_cloud.max()
        qc_mean = qc_cloud.mean()

        # Center of mass
        center = center_of_mass(cloud_cells)
        i_c, j_c, k_c = [int(round(c)) for c in center]

        # Bounding box
        indices = np.where(cloud_cells)
        i_min, i_max = indices[0].min(), indices[0].max()
        j_min, j_max = indices[1].min(), indices[1].max()
        k_min, k_max = indices[2].min(), indices[2].max()

        # Horizontal extent (in meters, assuming dx=100m)
        h_extent = max(i_max - i_min, j_max - j_min) * 100
        v_extent = (data['z'][k_max] - data['z'][k_min])

        clouds.append({
            'id': cloud_id,
            'center': (i_c, j_c, k_c),
            'center_z_m': data['z'][k_c],
            'size': volume,
            'qc_max': qc_max,
            'qc_mean': qc_mean,
            'bounds': ((i_min, i_max), (j_min, j_max), (k_min, k_max)),
            'h_extent_m': h_extent,
            'v_extent_m': v_extent,
        })

    # Sort by qc_max (strongest clouds first)
    clouds.sort(key=lambda c: c['qc_max'], reverse=True)

    return clouds, data


def report_clouds(clouds, top_n=10):
    """Print summary of detected clouds."""
    print(f"{'ID':<4} {'Center (i,j,k)':<18} {'Z(m)':<6} {'Vol':<7} {'qc_max':<8} {'qc_mean':<8} {'H×V extent (m)':<20}")
    print("-" * 95)

    for cloud in clouds[:top_n]:
        i, j, k = cloud['center']
        print(
            f"{cloud['id']:<4} "
            f"({i:3d},{j:3d},{k:2d})"
            f"      {cloud['center_z_m']:6.0f}  "
            f"{cloud['size']:6d}  "
            f"{cloud['qc_max']:6.2f}  "
            f"{cloud['qc_mean']:6.2f}  "
            f"{cloud['h_extent_m']:6.0f} × {cloud['v_extent_m']:6.0f}"
        )


def recommend_cloud(clouds, target_size_km=2.0, parent_nx=120):
    """
    Recommend a cloud for nesting based on:
    - Moderate size (fits in nested domain)
    - Strong enough (qc_max > 2 g/kg)
    - Not too close to domain edges
    """
    target_cells = int(target_size_km * 1000 / 100)  # parent cells needed
    margin = target_cells // 2 + 5  # cells from edge

    print(f"\nRecommendations for {target_size_km} km nested domain:")
    print(f"  Required margin from edges: {margin} cells")
    print()

    for cloud in clouds:
        i, j, k = cloud['center']

        # Check if cloud center has enough margin
        if i < margin or i > parent_nx - margin:
            reason = f"Too close to x-boundary (i={i})"
        elif j < margin or j > parent_nx - margin:
            reason = f"Too close to y-boundary (j={j})"
        elif cloud['qc_max'] < 2.0:
            reason = f"Too weak (qc_max={cloud['qc_max']:.1f})"
        elif cloud['h_extent_m'] > target_size_km * 1000 * 0.8:
            reason = f"Too large ({cloud['h_extent_m']:.0f}m > 80% of domain)"
        else:
            reason = "✓ GOOD CANDIDATE"

        if "GOOD" in reason or cloud == clouds[0]:  # Always show best cloud
            print(
                f"Cloud {cloud['id']:2d} @ ({i:3d},{j:3d},{k:2d}): "
                f"qc_max={cloud['qc_max']:5.2f} g/kg, "
                f"size={cloud['h_extent_m']:.0f}×{cloud['v_extent_m']:.0f}m - "
                f"{reason}"
            )


def visualize_slice(data, clouds, selected_cloud_id=None):
    """Quick x-z slice visualization."""
    qc = data['qc'] * 1000
    z = data['z']
    t = float(data['t'])

    # Middle y-slice
    j_mid = qc.shape[1] // 2
    qc_slice = qc[:, j_mid, :]

    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot qc
    im = ax.contourf(
        np.arange(qc.shape[0]),
        z,
        qc_slice.T,
        levels=np.linspace(0, 5, 21),
        cmap='Blues',
        extend='max'
    )
    ax.contour(
        np.arange(qc.shape[0]),
        z,
        qc_slice.T,
        levels=[0.5, 1.0, 2.0],
        colors='black',
        linewidths=0.5,
        alpha=0.5
    )

    # Mark cloud centers
    for cloud in clouds[:5]:  # Top 5 clouds
        i, j, k = cloud['center']
        marker = 'r*' if cloud['id'] == selected_cloud_id else 'ro'
        size = 200 if cloud['id'] == selected_cloud_id else 80
        ax.plot(i, z[k], marker, markersize=size/20,
                label=f"Cloud {cloud['id']}" if cloud['id'] <= 3 else None)

    plt.colorbar(im, ax=ax, label='qc (g/kg)')
    ax.set_xlabel('x (grid index)')
    ax.set_ylabel('Height (m)')
    ax.set_title(f'Cloud water at t={t/60:.1f} min (y-slice at j={j_mid})')
    ax.grid(alpha=0.3)
    if clouds:
        ax.legend(loc='upper right')

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Find clouds for nested simulation')
    parser.add_argument('--snapshot', type=str, required=True,
                       help='Path to parent snapshot file')
    parser.add_argument('--min-qc', type=float, default=1.0,
                       help='Minimum qc threshold (g/kg)')
    parser.add_argument('--nest-size', type=float, default=2.0,
                       help='Target nested domain size (km)')
    parser.add_argument('--plot', action='store_true',
                       help='Show visualization')
    parser.add_argument('--recommend', type=int, default=None,
                       help='Cloud ID to recommend (highlights in plot)')

    args = parser.parse_args()

    snapshot_path = Path(args.snapshot)
    if not snapshot_path.exists():
        print(f"Error: {snapshot_path} not found")
        return 1

    # Find clouds
    clouds, data = find_clouds(snapshot_path, min_qc=args.min_qc)

    if not clouds:
        print("No clouds found!")
        return 1

    # Report
    report_clouds(clouds)
    recommend_cloud(clouds, target_size_km=args.nest_size)

    # Visualize
    if args.plot:
        fig = visualize_slice(data, clouds, selected_cloud_id=args.recommend)
        plt.savefig('cloud_detection.png', dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to cloud_detection.png")
        plt.show()

    # Print setup command for recommended cloud
    if clouds:
        best = clouds[0] if args.recommend is None else next(
            (c for c in clouds if c['id'] == args.recommend), clouds[0]
        )
        i, j, k = best['center']
        print(f"\nTo setup nested simulation with Cloud {best['id']}:")
        print(f"  python setup_nested.py \\")
        print(f"    --parent-snapshot {args.snapshot} \\")
        print(f"    --cloud-center {i},{j},{k} \\")
        print(f"    --nest-size {args.nest_size} \\")
        print(f"    --dx-fine 25.0")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

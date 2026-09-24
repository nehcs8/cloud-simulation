#!/usr/bin/env python3
"""
Setup nested simulation from parent snapshot.

Extracts subregion from parent domain and interpolates to fine grid.

Usage:
    python setup_nested.py \\
        --parent-snapshot experiments/phase1_qvsurf/exp08_qv0160/snapshots/snap_0001200.npz \\
        --cloud-center 45,76,44 \\
        --nest-size 2.0 \\
        --dx-fine 25.0 \\
        --output nested_ic.npz
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import config_nested as cfg


def extract_parent_region(snapshot_path, cloud_center, nest_size_km, dx_parent=100.0):
    """
    Extract subregion around cloud from parent snapshot.

    Parameters:
    -----------
    snapshot_path : Path
        Parent simulation snapshot
    cloud_center : tuple (i, j, k)
        Cloud center in parent grid indices
    nest_size_km : float
        Nested domain size (km)
    dx_parent : float
        Parent resolution (m), default 100

    Returns:
    --------
    parent_data : dict
        Extracted fields on parent grid
    """
    print(f"Loading parent snapshot: {snapshot_path}")
    data = np.load(snapshot_path)

    i_c, j_c, k_c = cloud_center
    n_cells = int(nest_size_km * 1000 / dx_parent)  # cells in parent

    # Extract region centered on cloud
    i_start = max(0, i_c - n_cells // 2)
    i_end = min(data['qc'].shape[0], i_c + n_cells // 2)
    j_start = max(0, j_c - n_cells // 2)
    j_end = min(data['qc'].shape[1], j_c + n_cells // 2)

    print(f"  Cloud center: ({i_c}, {j_c}, {k_c})")
    print(f"  Extracting parent region: i={i_start}:{i_end}, j={j_start}:{j_end}")
    print(f"  Parent cells: {i_end-i_start} × {j_end-j_start} × {data['qc'].shape[2]}")

    # Note: w in snapshot is already cell-centered (converted in simulation.py)
    # But we need face-staggered w for the nested grid
    # Reconstruct face-staggered from cell-centered approximation
    w_cell = data['w'][i_start:i_end, j_start:j_end, :]
    # Simple approach: linearly interpolate to faces, add boundaries
    nz_cell = w_cell.shape[2]
    w_face = np.zeros((w_cell.shape[0], w_cell.shape[1], nz_cell + 1))

    # Interior faces: average of adjacent cells
    for k in range(1, nz_cell):
        w_face[:, :, k] = 0.5 * (w_cell[:, :, k-1] + w_cell[:, :, k])

    # Boundaries (rigid lid)
    w_face[:, :, 0] = 0.0
    w_face[:, :, -1] = 0.0

    # Check if u,v are in snapshot (new format) or need to be initialized to zero (old format)
    shape_3d = (i_end - i_start, j_end - j_start, data['qc'].shape[2])

    if 'u' in data and 'v' in data:
        # New format: u,v are saved in snapshot
        u_region = data['u'][i_start:i_end, j_start:j_end, :]
        v_region = data['v'][i_start:i_end, j_start:j_end, :]
        print(f"  ✓ Loaded u,v from parent snapshot")
        print(f"    u range: {u_region.min():.2f} - {u_region.max():.2f} m/s")
        print(f"    v range: {v_region.min():.2f} - {v_region.max():.2f} m/s")
    else:
        # Old format: u,v not saved, initialize to zero
        print(f"  ⚠ Warning: u,v not in parent snapshot, initializing to zero")
        print(f"    This will cause artificial rotation in clouds!")
        print(f"    Run parent simulation with updated simulation.py to save u,v")
        u_region = np.zeros(shape_3d, dtype=np.float32)
        v_region = np.zeros(shape_3d, dtype=np.float32)

    parent_region = {
        'u': u_region,
        'v': v_region,
        'w': w_face,  # face-staggered, reconstructed from saved data
        'theta': data['theta'][i_start:i_end, j_start:j_end, :],
        'qv': data['qv'][i_start:i_end, j_start:j_end, :],
        'qc': data['qc'][i_start:i_end, j_start:j_end, :],
        'qr': data['qr'][i_start:i_end, j_start:j_end, :],
        'z': data['z'],
        'z_face': data['z_face'],
        't': float(data['t']),
        # Mean profiles for forcing
        'prof_theta': data['prof_theta'],
        'prof_qv': data['prof_qv'],
        # Domain statistics
        'parent_shape': shape_3d,
        'parent_offset': (i_start, j_start, 0),
    }

    print(f"  Extracted t={parent_region['t']:.0f}s")
    print(f"  qc range: {parent_region['qc'].min()*1000:.3f} - {parent_region['qc'].max()*1000:.3f} g/kg")

    return parent_region


def interpolate_to_nest_grid(parent_data, nx_nest, ny_nest, nz_nest,
                             dx_fine, dy_fine, z_top_nest):
    """
    Interpolate parent fields to fine nested grid.

    Parameters:
    -----------
    parent_data : dict
        Extracted parent region
    nx_nest, ny_nest, nz_nest : int
        Nested grid dimensions
    dx_fine, dy_fine : float
        Nested resolution (m)
    z_top_nest : float
        Nested domain top (m)

    Returns:
    --------
    nest_data : dict
        Fields interpolated to nested grid
    """
    print(f"\nInterpolating to nested grid:")
    print(f"  Target: {nx_nest} × {ny_nest} × {nz_nest}")
    print(f"  Resolution: dx={dx_fine}m, dy={dy_fine}m")

    # Parent grid coordinates
    nx_p, ny_p, nz_p = parent_data['parent_shape']
    x_p = np.arange(nx_p) * 100.0  # parent dx = 100m
    y_p = np.arange(ny_p) * 100.0
    z_p = parent_data['z']

    # Nested grid coordinates
    x_n = np.arange(nx_nest) * dx_fine
    y_n = np.arange(ny_nest) * dy_fine

    # Vertical: truncate parent z-levels to nest domain top
    z_n_mask = parent_data['z'] <= z_top_nest
    z_n = parent_data['z'][z_n_mask]
    nz_actual = len(z_n)

    if nz_actual < nz_nest:
        print(f"  Warning: Only {nz_actual} z-levels available below {z_top_nest}m")
        print(f"           Adjusting nz_nest: {nz_nest} → {nz_actual}")
        nz_nest = nz_actual

    print(f"  Vertical levels: {nz_nest} (z_top = {z_n[-1]:.1f}m)")

    # Interpolate each field
    nest_data = {}

    for field_name in ['u', 'v', 'theta', 'qv', 'qc', 'qr']:
        print(f"  Interpolating {field_name}...", end='', flush=True)

        # Extract parent field (truncate vertical)
        field_parent = parent_data[field_name][:, :, :nz_nest]

        # Create interpolator (linear in 3D)
        interp = RegularGridInterpolator(
            (x_p, y_p, z_n),
            field_parent,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )

        # Create nested mesh
        xn_mesh, yn_mesh, zn_mesh = np.meshgrid(x_n, y_n, z_n, indexing='ij')
        points = np.stack([xn_mesh.ravel(), yn_mesh.ravel(), zn_mesh.ravel()], axis=1)

        # Interpolate
        nest_data[field_name] = interp(points).reshape(nx_nest, ny_nest, nz_nest)
        print(f" done")

    # For w (face-staggered): need nz+1 levels
    print(f"  Interpolating w (face-staggered)...", end='', flush=True)
    z_face_p = parent_data['z_face']
    z_face_n_mask = z_face_p <= z_top_nest
    z_face_n = z_face_p[z_face_n_mask]

    # Ensure we have nz_nest+1 face levels
    if len(z_face_n) < nz_nest + 1:
        # Extend by one level if needed
        dz_top = z_face_n[-1] - z_face_n[-2]
        z_face_n = np.append(z_face_n, z_face_n[-1] + dz_top)

    w_parent = parent_data['w'][:, :, :len(z_face_n)]

    interp_w = RegularGridInterpolator(
        (x_p, y_p, z_face_n),
        w_parent,
        method='linear',
        bounds_error=False,
        fill_value=0.0
    )

    xn_mesh, yn_mesh, zfn_mesh = np.meshgrid(x_n, y_n, z_face_n, indexing='ij')
    points_w = np.stack([xn_mesh.ravel(), yn_mesh.ravel(), zfn_mesh.ravel()], axis=1)
    nest_data['w'] = interp_w(points_w).reshape(nx_nest, ny_nest, len(z_face_n))

    # Enforce rigid lid
    nest_data['w'][:, :, 0] = 0.0
    nest_data['w'][:, :, -1] = 0.0
    print(f" done")

    # Copy metadata
    nest_data['z'] = z_n
    nest_data['z_face'] = z_face_n
    nest_data['t'] = parent_data['t']
    nest_data['prof_theta'] = parent_data['prof_theta'][:nz_nest]
    nest_data['prof_qv'] = parent_data['prof_qv'][:nz_nest]

    # Compute domain means for forcing
    nest_data['theta_mean'] = np.mean(nest_data['theta'], axis=(0, 1))
    nest_data['qv_mean'] = np.mean(nest_data['qv'], axis=(0, 1))

    print(f"\nNested grid statistics:")
    print(f"  qc range: {nest_data['qc'].min()*1000:.3f} - {nest_data['qc'].max()*1000:.3f} g/kg")
    print(f"  w range: {nest_data['w'].min():.2f} - {nest_data['w'].max():.2f} m/s")
    print(f"  theta mean: {nest_data['theta_mean'].mean():.3f} K")

    return nest_data, nz_nest


def save_nested_ic(nest_data, output_path):
    """Save nested initial conditions to file."""
    print(f"\nSaving nested IC to: {output_path}")

    np.savez_compressed(
        output_path,
        # State fields
        u=nest_data['u'].astype(np.float32),
        v=nest_data['v'].astype(np.float32),
        w=nest_data['w'].astype(np.float32),
        theta=nest_data['theta'].astype(np.float32),
        qv=nest_data['qv'].astype(np.float32),
        qc=nest_data['qc'].astype(np.float32),
        qr=nest_data['qr'].astype(np.float32),
        # Grid
        z=nest_data['z'].astype(np.float32),
        z_face=nest_data['z_face'].astype(np.float32),
        # Time
        t=np.array(nest_data['t']),
        # Reference profiles (for forcing)
        prof_theta=nest_data['prof_theta'].astype(np.float32),
        prof_qv=nest_data['prof_qv'].astype(np.float32),
        # Domain means (for nested nudging)
        theta_mean=nest_data['theta_mean'].astype(np.float32),
        qv_mean=nest_data['qv_mean'].astype(np.float32),
    )

    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description='Setup nested cloud simulation')
    parser.add_argument('--parent-snapshot', type=str, required=True,
                       help='Path to parent simulation snapshot')
    parser.add_argument('--cloud-center', type=str, required=True,
                       help='Cloud center as i,j,k (e.g., 45,76,44)')
    parser.add_argument('--nest-size', type=float, default=2.0,
                       help='Nested domain size (km)')
    parser.add_argument('--dx-fine', type=float, default=25.0,
                       help='Nested resolution (m)')
    parser.add_argument('--output', type=str, default='nested_ic.npz',
                       help='Output file')

    args = parser.parse_args()

    # Parse cloud center
    cloud_center = tuple(map(int, args.cloud_center.split(',')))
    if len(cloud_center) != 3:
        print("Error: cloud-center must be i,j,k")
        return 1

    # Check parent snapshot exists
    parent_path = Path(args.parent_snapshot)
    if not parent_path.exists():
        print(f"Error: {parent_path} not found")
        return 1

    # Extract parent region
    parent_data = extract_parent_region(
        parent_path,
        cloud_center,
        args.nest_size
    )

    # Interpolate to nested grid
    nest_data, nz_actual = interpolate_to_nest_grid(
        parent_data,
        cfg.NX, cfg.NY, cfg.NZ,
        args.dx_fine, args.dx_fine,
        cfg.Z_TOP
    )

    # Save
    output_path = Path(args.output)
    save_nested_ic(nest_data, output_path)

    print(f"\n✓ Setup complete!")
    print(f"\nNext steps:")
    print(f"  1. Update config_nested.py:")
    print(f"       RESTART_FROM = '{output_path}'")
    if nz_actual < cfg.NZ:
        print(f"       NZ = {nz_actual}  # (adjusted)")
    print(f"  2. Run nested simulation:")
    print(f"       python main_nested.py")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

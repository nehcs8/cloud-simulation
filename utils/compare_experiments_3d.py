#!/usr/bin/env python3
"""
Generate side-by-side 3D volume rendering comparison from multiple experiments.

Usage:
    python compare_experiments_3d.py \
        --experiments exp06_qv0140 exp07_qv0150 exp08_qv0160 \
        --phase phase1_qvsurf \
        --output comparison_3d.gif
"""

import argparse
import numpy as np
import pyvista as pv
from pathlib import Path
import imageio
from grid import Grid as SimGrid
import config as cfg

Z_EXP = 2.0    # vertical exaggeration
QC_MAX = 1.5   # g/kg — top of colour scale


def load_snapshots(exp_dir: Path):
    """Load all snapshots from an experiment directory."""
    snap_dir = exp_dir / 'snapshots'
    if not snap_dir.exists():
        return []
    snapshots = sorted(snap_dir.glob('snap_*.npz'))
    return snapshots


def make_grid():
    """Create PyVista rectilinear grid."""
    g = SimGrid()
    xf = np.arange(cfg.NX + 1, dtype=np.float64) * cfg.DX
    yf = np.arange(cfg.NY + 1, dtype=np.float64) * cfg.DY
    zf = g.z_face * Z_EXP
    return pv.RectilinearGrid(xf, yf, zf)


def render_comparison_frame(snapshot_paths, titles, vmax=1.5):
    """Render a single frame with multiple experiments in grid layout."""
    n_exp = len(snapshot_paths)

    # Determine optimal layout based on number of experiments
    if n_exp <= 4:
        # 2×2 grid for 4 or fewer experiments
        nrows, ncols = 2, 2
        single_width = 550   # Wider to accommodate diagonal view
        single_height = 500
    else:
        # 2×3 grid for 5 experiments (3 top, 2 bottom)
        nrows, ncols = 2, 3
        single_width = 500
        single_height = 480

    total_width = single_width * ncols
    total_height = single_height * nrows

    # Create plotter with grid
    pl = pv.Plotter(off_screen=True, window_size=(total_width, total_height), shape=(nrows, ncols))

    g = SimGrid()
    xm = cfg.NX * cfg.DX
    ym = cfg.NY * cfg.DY
    zm = g.z_face[-1] * Z_EXP

    vol_kwargs = dict(
        clim=[0.0, vmax],
        cmap="Blues",
        opacity=[0, 0, 0.4, 0.75, 0.92],
        show_scalar_bar=False,  # Too cluttered with multiple views
    )

    # Layout positions based on number of experiments
    if n_exp <= 4:
        # 2×2 grid
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    else:
        # 2×3 grid (3 top, 2 bottom)
        positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]

    t = None
    for i, (snap_path, title) in enumerate(zip(snapshot_paths, titles)):
        if i >= len(positions):
            break

        row, col = positions[i]
        pl.subplot(row, col)

        data = np.load(snap_path)
        t = float(data['t'])
        qc = data['qc'] * 1000  # Convert to g/kg

        # Create grid for this subplot
        grid = make_grid()
        grid.cell_data["qc"] = qc.astype(np.float32).ravel(order="F")

        # Add ground plane
        pl.add_mesh(
            pv.Plane(center=(xm/2, ym/2, 0), direction=(0,0,1), i_size=xm, j_size=ym),
            color="#7B9E6B", opacity=0.85,
        )

        # Add domain wireframe
        pl.add_mesh(
            pv.Box(bounds=(0, xm, 0, ym, 0, zm)),
            style="wireframe", color="gray", opacity=0.2, line_width=1,
        )

        # Add volume
        pl.add_volume(grid, scalars="qc", **vol_kwargs)

        # Add prominent title for this subplot
        pl.add_text(title, position="upper_edge", font_size=16, color="black", font="times")

        # Set camera position
        pl.camera_position = [
            (xm * 1.5, ym * 1.5, zm * 1.2),
            (xm * 0.5, ym * 0.5, g.z_face[25] * Z_EXP),
            (0, 0, 1),
        ]

    # Add global time stamp to appropriate cell based on layout
    if t is not None:
        if n_exp <= 4:
            # For 2×2, add to bottom-right cell
            pl.subplot(1, 1)
            pl.add_text(f't = {t/60:.1f} min', position='lower_right', font_size=24, color='black', font="times")
        else:
            # For 2×3, add to empty bottom-right cell
            pl.subplot(1, 2)
            pl.add_text(f't = {t/60:.1f} min', position='upper_left', font_size=28, color='black', font="times")

    # Render and capture
    pl.render()
    img = pl.screenshot(return_img=True)
    pl.close()

    return img


def main():
    parser = argparse.ArgumentParser(description='Compare multiple experiments in 3D')
    parser.add_argument('--experiments', nargs='+', required=True,
                       help='Experiment IDs to compare')
    parser.add_argument('--phase', required=True,
                       help='Phase directory (e.g., phase1_qvsurf)')
    parser.add_argument('--output', default='comparison_3d.gif',
                       help='Output GIF filename')
    parser.add_argument('--fps', type=float, default=8.0,
                       help='Frames per second')
    parser.add_argument('--vmax', type=float, default=1.5,
                       help='Max value for colorbar (g/kg)')

    args = parser.parse_args()

    # Load experiment directories
    base_dir = Path('experiments') / args.phase
    exp_dirs = [base_dir / exp_id for exp_id in args.experiments]

    # Check all exist
    for exp_dir in exp_dirs:
        if not exp_dir.exists():
            print(f"Error: {exp_dir} does not exist")
            return 1

    # Load snapshots from each experiment
    print(f"Loading snapshots from {len(exp_dirs)} experiments...")
    all_snapshots = [load_snapshots(exp_dir) for exp_dir in exp_dirs]

    # Check all have same number of snapshots
    n_frames = min(len(snaps) for snaps in all_snapshots)
    if n_frames == 0:
        print("Error: No snapshots found in experiments")
        return 1

    print(f"Found {n_frames} frames per experiment")

    # Generate titles with parameter values
    titles = []
    for exp_id in args.experiments:
        # Extract parameter from exp_id (e.g., exp01_qv0110 -> QV=0.011)
        if 'qv' in exp_id:
            param_val = exp_id.split('_qv')[-1]
            param_val = f"0.{param_val}"
            titles.append(f"QV_SURF={param_val}")
        else:
            titles.append(exp_id)

    # Filter to first 30 minutes only
    max_time = 1800.0  # 30 minutes in seconds
    filtered_snapshots = []
    for snaps in all_snapshots:
        filtered = []
        for snap in snaps:
            t = float(np.load(snap)['t'])
            if t <= max_time:
                filtered.append(snap)
        filtered_snapshots.append(filtered)

    n_frames = min(len(snaps) for snaps in filtered_snapshots)
    print(f"Filtered to first 30 minutes: {n_frames} frames")

    # Render frames
    print(f"Rendering 3D comparison frames...")
    images = []
    for frame_idx in range(n_frames):
        snapshot_paths = [snaps[frame_idx] for snaps in filtered_snapshots]
        img = render_comparison_frame(
            snapshot_paths, titles,
            vmax=args.vmax
        )
        images.append(img)
        if (frame_idx + 1) % 5 == 0:
            print(f"  Rendered {frame_idx + 1}/{n_frames} frames", end='\r')

    print(f"\n  Rendered {n_frames} frames")

    # Save GIF
    print(f"Saving to {args.output}...")
    imageio.mimsave(args.output, images, fps=args.fps, loop=0)

    print(f"✓ Saved: {args.output}")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

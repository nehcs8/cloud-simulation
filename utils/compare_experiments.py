#!/usr/bin/env python3
"""
Generate side-by-side comparison GIF from multiple experiments.

Usage:
    python compare_experiments.py \
        --experiments exp01_qv0110 exp02_qv0120 exp03_qv0125 \
        --phase phase1_qvsurf \
        --output comparison_qvsurf.gif
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import imageio
from grid import Grid


def load_snapshots(exp_dir: Path):
    """Load all snapshots from an experiment directory."""
    snap_dir = exp_dir / 'snapshots'
    if not snap_dir.exists():
        return []

    snapshots = sorted(snap_dir.glob('snap_*.npz'))
    return snapshots


def render_comparison_frame(snapshot_paths, grid, titles, cmap='Blues', vmin=0, vmax=0.5):
    """Render a single frame with multiple experiments side-by-side."""
    n_exp = len(snapshot_paths)
    fig, axes = plt.subplots(1, n_exp, figsize=(5*n_exp, 4))

    if n_exp == 1:
        axes = [axes]

    xkm = grid.x / 1e3
    zkm = grid.z / 1e3

    t = None
    for i, (snap_path, title) in enumerate(zip(snapshot_paths, titles)):
        data = np.load(snap_path)
        t = float(data['t'])
        qc = data['qc'] * 1000  # Convert to g/kg

        # x-z cross-section at y = NY/2
        jmid = grid.ny // 2
        qc_xz = qc[:, jmid, :].T

        ax = axes[i]
        im = ax.imshow(
            qc_xz,
            origin='lower',
            aspect='auto',
            extent=[xkm[0], xkm[-1], zkm[0], zkm[-1]],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )
        ax.set_xlabel('x [km]')
        ax.set_ylabel('z [km]')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='qc [g/kg]', fraction=0.046)

    if t is not None:
        fig.suptitle(f't = {t/60:.1f} min', fontsize=14, y=0.98)

    plt.tight_layout()

    # Convert to image array
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    image = image[:, :, :3]  # Drop alpha channel
    plt.close(fig)

    return image


def main():
    parser = argparse.ArgumentParser(description='Compare multiple experiments')
    parser.add_argument('--experiments', nargs='+', required=True,
                       help='Experiment IDs to compare')
    parser.add_argument('--phase', required=True,
                       help='Phase directory (e.g., phase1_qvsurf)')
    parser.add_argument('--output', default='comparison.gif',
                       help='Output GIF filename')
    parser.add_argument('--fps', type=float, default=8.0,
                       help='Frames per second')
    parser.add_argument('--vmax', type=float, default=0.5,
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

    # Create grid
    grid = Grid()

    # Generate titles with parameter values
    titles = []
    for exp_id in args.experiments:
        # Extract parameter from exp_id (e.g., exp01_qv0110 -> QV=0.011)
        if 'qv' in exp_id:
            param_val = exp_id.split('_qv')[-1]
            param_val = f"0.{param_val}"
            titles.append(f"QV_SURF={param_val}")
        elif 'tau' in exp_id:
            param_val = exp_id.split('_tau')[-1]
            titles.append(f"TAU_Q={param_val}s")
        elif 'lhf' in exp_id:
            param_val = exp_id.split('_lhf')[-1]
            titles.append(f"LHF={param_val}W/m²")
        elif 'scale' in exp_id:
            param_val = exp_id.split('_scale')[-1]
            titles.append(f"QV_SCALE={param_val}m")
        elif 'wsubs' in exp_id:
            param_val = exp_id.split('_wsubs')[-1]
            param_val = f"0.{param_val.zfill(3)}"
            titles.append(f"W_SUBS={param_val}m/s")
        else:
            titles.append(exp_id)

    # Render frames
    print(f"Rendering comparison frames...")
    images = []
    for frame_idx in range(n_frames):
        snapshot_paths = [snaps[frame_idx] for snaps in all_snapshots]
        img = render_comparison_frame(
            snapshot_paths, grid, titles,
            vmin=0, vmax=args.vmax
        )
        images.append(img)
        if (frame_idx + 1) % 10 == 0:
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

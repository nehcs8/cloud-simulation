"""
Side-by-side comparison: Parent cloud (left) vs Nested cloud evolution (right).

Shows the parent snapshot at t=1200s (dx=100m) next to the nested cloud
evolution at 4× finer resolution (dx=25m).

Usage:
    python3 visualize_comparison.py                    # → comparison.gif
    python3 visualize_comparison.py --fps 4            # slower playback
    python3 visualize_comparison.py --out compare.mp4  # MP4 format
"""

import sys
import os
import glob
import argparse
from pathlib import Path
import numpy as np
import pyvista as pv
import imageio

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load both configurations
import config as cfg_parent
import config_nested as cfg_nested

from grid import Grid


def _make_grid_parent(parent_data):
    """Create grid for parent simulation from actual data dimensions."""
    nx, ny, nz = parent_data['qc'].shape

    # Get actual z coordinates from parent data if available, else create uniform
    if 'z' in parent_data:
        z_centers = parent_data['z']
        # Create face coordinates (nz+1 points)
        z_face = np.zeros(nz + 1)
        z_face[0] = 0.0
        z_face[1:-1] = 0.5 * (z_centers[:-1] + z_centers[1:])
        z_face[-1] = 2 * z_centers[-1] - z_face[-2]
    else:
        # Fallback to uniform grid
        z_face = np.linspace(0, 5000, nz + 1)

    xf = np.arange(nx + 1) * 100.0  # dx = 100m for parent
    yf = np.arange(ny + 1) * 100.0
    zf = z_face * 2.0  # Z_EXP
    return pv.RectilinearGrid(xf, yf, zf)


def _make_grid_nested(nested_data):
    """Create grid for nested simulation from actual data dimensions."""
    nx, ny, nz = nested_data['qc'].shape

    # Get z coordinates from nested data if available
    if 'z' in nested_data:
        z_centers = nested_data['z']
        z_face = np.zeros(nz + 1)
        z_face[0] = 0.0
        z_face[1:-1] = 0.5 * (z_centers[:-1] + z_centers[1:])
        z_face[-1] = 2 * z_centers[-1] - z_face[-2]
    else:
        # Use nested config values
        z_face = np.linspace(0, cfg_nested.Z_TOP, nz + 1)

    xf = np.arange(nx + 1) * cfg_nested.DX
    yf = np.arange(ny + 1) * cfg_nested.DY
    zf = z_face * 1.5  # Z_EXP for nested
    return pv.RectilinearGrid(xf, yf, zf)


def _make_plotter_sidebyside(parent_data):
    """
    Create a plotter with two viewports side-by-side.
    Returns plotter and two dicts of volume kwargs for parent/nested.
    """
    pl = pv.Plotter(off_screen=True, window_size=(2560, 720), shape=(1, 2))
    pl.set_background("white")

    # ── Left viewport: Parent cloud (static) ──────────────────────────────────
    pl.subplot(0, 0)

    # Parent domain dimensions from actual data
    nx_p, ny_p, nz_p = parent_data['qc'].shape
    xm_p = nx_p * 100.0  # dx = 100m
    ym_p = ny_p * 100.0

    # Get z_face from parent data
    if 'z' in parent_data:
        z_centers = parent_data['z']
        z_face = np.zeros(nz_p + 1)
        z_face[0] = 0.0
        z_face[1:-1] = 0.5 * (z_centers[:-1] + z_centers[1:])
        z_face[-1] = 2 * z_centers[-1] - z_face[-2]
    else:
        z_face = np.linspace(0, 5000, nz_p + 1)

    zm_p = z_face[-1] * 2.0

    pl.add_mesh(
        pv.Plane(center=(xm_p/2, ym_p/2, 0), direction=(0,0,1), i_size=xm_p, j_size=ym_p),
        color="#7B9E6B", opacity=0.85,
    )
    pl.add_mesh(
        pv.Box(bounds=(0, xm_p, 0, ym_p, 0, zm_p)),
        style="wireframe", color="white", opacity=0.12, line_width=1,
    )

    # Camera for parent (wider domain)
    pl.camera_position = [
        (xm_p * 1.5,  ym_p * 1.5,  zm_p * 1.2),
        (xm_p * 0.5,  ym_p * 0.5,  z_face[25] * 2.0 if nz_p > 25 else zm_p * 0.5),
        (0, 0, 1),
    ]

    vol_kwargs_parent = dict(
        clim=[0.0, 2.0],  # g/kg
        cmap="Blues",
        opacity=[0, 0, 0.4, 0.75, 0.92],
        scalar_bar_args=dict(
            title="qc [g/kg]", title_font_size=11, label_font_size=9,
            position_x=0.42, position_y=0.05, height=0.3, width=0.03,
        ),
    )

    # ── Right viewport: Nested cloud (animated) ───────────────────────────────
    pl.subplot(0, 1)

    # Nested domain - will be set up dynamically with first frame
    # Use config values for initial setup
    xm_n = cfg_nested.NX * cfg_nested.DX
    ym_n = cfg_nested.NY * cfg_nested.DY
    zm_n = cfg_nested.Z_TOP * 1.5

    pl.add_mesh(
        pv.Plane(center=(xm_n/2, ym_n/2, 0), direction=(0,0,1), i_size=xm_n, j_size=ym_n),
        color="#7B9E6B", opacity=0.85,
    )
    pl.add_mesh(
        pv.Box(bounds=(0, xm_n, 0, ym_n, 0, zm_n)),
        style="wireframe", color="white", opacity=0.12, line_width=1,
    )

    # Camera for nested (smaller domain)
    pl.camera_position = [
        (xm_n * 1.8,  ym_n * 1.8,  zm_n * 0.9),
        (xm_n * 0.5,  ym_n * 0.5,  zm_n * 0.4),
        (0, 0, 1),
    ]

    vol_kwargs_nested = dict(
        clim=[0.0, 3.0],  # g/kg (nested has stronger clouds)
        cmap="Blues",
        opacity=[0, 0, 0.35, 0.70, 0.90],
        scalar_bar_args=dict(
            title="qc [g/kg]", title_font_size=11, label_font_size=9,
            position_x=0.92, position_y=0.05, height=0.3, width=0.03,
        ),
    )

    return pl, vol_kwargs_parent, vol_kwargs_nested


def _render_comparison_frame(pl, vol_kw_parent, vol_kw_nested,
                              grid_parent, grid_nested,
                              qc_parent_gkg, qc_nested_gkg, t_nested):
    """
    Render one frame with parent (left, static) and nested (right, time-varying).
    """
    # Left panel: Parent cloud (static)
    pl.subplot(0, 0)
    grid_parent.cell_data["qc"] = qc_parent_gkg
    vol_parent = pl.add_volume(grid_parent, scalars="qc", **vol_kw_parent)
    # Use named text actor for stable replacement
    pl.add_text("Parent (dx=100m, t=1200s)", position="upper_edge",
                font_size=11, color="black", name="title_parent")

    # Right panel: Nested cloud (animated)
    pl.subplot(0, 1)
    grid_nested.cell_data["qc"] = qc_nested_gkg
    vol_nested = pl.add_volume(grid_nested, scalars="qc", **vol_kw_nested)
    # Remove old title if exists, then add new one
    if pl.renderer.GetActors2D().GetNumberOfItems() > 0:
        pl.remove_actor("title_nested", reset_camera=False)
    pl.add_text(f"Nested (dx=25m, t={t_nested:.0f}s)", position="upper_edge",
                font_size=11, color="black", name="title_nested")

    # Render and screenshot
    pl.render()
    img = pl.screenshot(return_img=True)

    # Clean up volumes for next frame (but keep titles)
    pl.subplot(0, 0)
    pl.remove_actor(vol_parent, reset_camera=False)
    pl.remove_actor("title_parent", reset_camera=False)
    pl.subplot(0, 1)
    pl.remove_actor(vol_nested, reset_camera=False)

    return img


def render_comparison(out=None, fps=4.0):
    """
    Render side-by-side comparison GIF.

    Parent cloud (left): static at t=1200s
    Nested cloud (right): animated from t=0 to t=300s
    """
    # Load parent snapshot
    parent_path = cfg_nested.PARENT_SNAPSHOT
    if not os.path.exists(parent_path):
        print(f"Error: Parent snapshot not found: {parent_path}")
        sys.exit(1)

    print(f"Loading parent snapshot: {parent_path}")
    parent_data = np.load(parent_path)
    qc_parent = (parent_data['qc'] * 1e3).astype(np.float32).ravel(order='F')

    # Find nested snapshots
    nested_paths = sorted(glob.glob("snapshots/snap_*.npz"))
    nested_paths_filtered = []
    for p in nested_paths:
        d = np.load(p)
        t = float(d['t'])
        if t <= cfg_nested.T_END + 1e-3:
            nested_paths_filtered.append(p)

    if not nested_paths_filtered:
        print("Error: No nested snapshots found in snapshots/")
        sys.exit(1)

    print(f"Found {len(nested_paths_filtered)} nested snapshots (t=0 to t={cfg_nested.T_END}s)")

    # Load first nested snapshot to get dimensions
    first_nested = np.load(nested_paths_filtered[0])

    # Output path
    if out is None:
        os.makedirs("cloud_gifs", exist_ok=True)
        out = "cloud_gifs/comparison.gif"

    # Setup grids and plotter
    grid_parent = _make_grid_parent(parent_data)
    grid_nested = _make_grid_nested(first_nested)
    pl, vol_kw_p, vol_kw_n = _make_plotter_sidebyside(parent_data)

    print(f"Rendering {len(nested_paths_filtered)} frames → {out} (fps={fps})")
    print(f"  Parent: {parent_data['qc'].shape} cells at dx=100m")
    print(f"  Nested: {first_nested['qc'].shape} cells at dx=25m")

    imgs = []
    for i, nested_path in enumerate(nested_paths_filtered):
        nested_data = np.load(nested_path)
        t_nested = float(nested_data['t'])
        qc_nested = (nested_data['qc'] * 1e3).astype(np.float32).ravel(order='F')

        img = _render_comparison_frame(
            pl, vol_kw_p, vol_kw_n,
            grid_parent, grid_nested,
            qc_parent, qc_nested, t_nested
        )
        imgs.append(img)
        print(f"  frame {i+1:2d}/{len(nested_paths_filtered)}  nested t={t_nested:.0f}s",
              end="\r", flush=True)

    pl.close()
    print()

    # Save
    if out.endswith('.mp4'):
        imageio.mimsave(out, imgs, fps=fps, codec='libx264', quality=8)
    else:
        imageio.mimsave(out, imgs, fps=fps, loop=0)

    print(f"Saved: {out} ({len(imgs)} frames)")
    print(f"\nComparison shows:")
    print(f"  Left:  Parent cloud at t=1200s (12×12 km, dx=100m)")
    print(f"  Right: Nested cloud evolution t=0-300s (2×2 km, dx=25m)")


def main():
    p = argparse.ArgumentParser(description="Side-by-side parent vs nested visualization")
    p.add_argument("--out", default=None, help="Output file (.gif or .mp4)")
    p.add_argument("--fps", type=float, default=4.0, help="Frames per second")
    args = p.parse_args()

    render_comparison(out=args.out, fps=args.fps)


if __name__ == "__main__":
    main()

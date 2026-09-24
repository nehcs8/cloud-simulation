"""
Off-screen 3D cloud renderer for NESTED simulation.

Renders high-resolution nested cloud evolution (dx=25m) to GIF or MP4.
Uses config_nested instead of config for proper grid dimensions.

Usage:
    python3 visualize3d_nested.py                    # → nested_clouds.gif
    python3 visualize3d_nested.py --out nested.mp4   # MP4 format
    python3 visualize3d_nested.py --fps 10           # faster playback
"""

import sys
import os
import re
import glob
import argparse
from pathlib import Path
import numpy as np
import pyvista as pv
import imageio

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# CRITICAL: Use nested config (same trick as main_nested.py)
import config_nested
sys.modules['config'] = config_nested
import config as cfg

from grid import Grid as SimGrid

SNAPSHOT_DIR = "snapshots"
GIF_DIR      = "cloud_gifs"


def _next_output_path(ext: str, prefix: str = "nested_clouds") -> str:
    """Return cloud_gifs/nested_clouds_N.<ext> where N is one above the current max."""
    os.makedirs(GIF_DIR, exist_ok=True)
    existing = glob.glob(os.path.join(GIF_DIR, f"{prefix}_*.{ext}"))
    nums = [int(m.group(1)) for f in existing
            if (m := re.search(rf"{prefix}_(\d+)\.", os.path.basename(f)))]
    n = max(nums) + 1 if nums else 1
    return os.path.join(GIF_DIR, f"{prefix}_{n}.{ext}")


Z_EXP   = 1.5    # vertical exaggeration (nested domain is taller relative to width)
QC_MAX  = 3.0    # g/kg — top of colour scale (nested has stronger clouds)


# ── Opacity transfer function ──────────────────────────────────────────────────

def _opacity(n: int = 256) -> np.ndarray:
    """
    Starts near-transparent above a small threshold, ramps up with sqrt.
    Every cloudy cell is visible; denser core is darker/more opaque.
    """
    t  = np.linspace(0.0, 1.0, n)
    t0 = 0.02 / QC_MAX          # 0.02 g/kg threshold normalised
    ramp = np.where(t > t0, (t - t0) / (1.0 - t0), 0.0)
    op   = np.sqrt(np.maximum(ramp, 0.0)) * 0.88
    return op.astype(np.float32)


# ── Grid (built once) ──────────────────────────────────────────────────────────

def _make_grid() -> pv.RectilinearGrid:
    g  = SimGrid()
    xf = np.arange(cfg.NX + 1, dtype=np.float64) * cfg.DX
    yf = np.arange(cfg.NY + 1, dtype=np.float64) * cfg.DY
    zf = g.z_face * Z_EXP
    return pv.RectilinearGrid(xf, yf, zf)


# ── Plotter setup (fixed camera, built once) ───────────────────────────────────

def _make_plotter() -> tuple[pv.Plotter, dict]:
    """
    Build the plotter with all static elements (ground, box, axes, camera).
    Returns the plotter and a dict of params needed to re-add the volume each frame.
    """
    g  = SimGrid()
    xm = cfg.NX * cfg.DX
    ym = cfg.NY * cfg.DY
    zm = g.z_face[-1] * Z_EXP

    pl = pv.Plotter(off_screen=True, window_size=(1280, 720))
    pl.set_background("white")

    # Ground
    pl.add_mesh(
        pv.Plane(center=(xm/2, ym/2, 0), direction=(0,0,1), i_size=xm, j_size=ym),
        color="#7B9E6B", opacity=0.85, name="ground",
    )
    # Domain wireframe
    pl.add_mesh(
        pv.Box(bounds=(0, xm, 0, ym, 0, zm)),
        style="wireframe", color="white", opacity=0.12, line_width=1, name="box",
    )
    pl.add_axes(line_width=3)

    # Fixed camera — outside the +x+y corner, elevated, looking at cloud-level centre
    # Adjusted for smaller nested domain (2×2×3 km)
    pl.camera_position = [
        (xm * 1.8,  ym * 1.8,  zm * 0.9),   # farther out for good view
        (xm * 0.5,  ym * 0.5,  g.z_face[30] * Z_EXP),  # look at mid-cloud level
        (0, 0, 1),
    ]

    vol_kwargs = dict(
        clim    = [0.0, QC_MAX],
        cmap    = "Blues",
        opacity = [0, 0, 0.35, 0.70, 0.90],   # transparent→opaque with scalar value
        scalar_bar_args = dict(
            title="qc [g/kg]", title_font_size=13, label_font_size=11,
            position_x=0.87, position_y=0.05, height=0.35, width=0.04,
        ),
    )
    return pl, vol_kwargs


# ── Render one frame: set data → add volume → screenshot → remove volume ───────

def _render_frame(pl: pv.Plotter, vol_kwargs: dict, grid: pv.RectilinearGrid,
                  qc_gkg: np.ndarray, t: float) -> np.ndarray:
    """
    Data must be in the grid BEFORE add_volume — VTK doesn't pick up in-place updates.
    So we: set data → add_volume → render → screenshot → remove volume actor.
    """
    grid.cell_data["qc"] = qc_gkg
    vol_actor = pl.add_volume(grid, scalars="qc", name="cloud_vol", **vol_kwargs)

    # Title with nested domain info
    pl.add_text(f"Nested Cloud (dx=25m)  —  t = {t:.0f}s",
                position="upper_edge", font_size=12, color="black", name="title")

    pl.render()
    img = pl.screenshot(return_img=True)
    pl.remove_actor(vol_actor)
    return img


# ── Single frame ───────────────────────────────────────────────────────────────

def render_single(path: str, out: str = "nested_cloud_frame.png") -> None:
    d       = np.load(path)
    t       = float(d["t"])
    grid    = _make_grid()
    pl, kw  = _make_plotter()
    img     = _render_frame(pl, kw, grid,
                            (d["qc"] * 1e3).astype(np.float32).ravel(order="F"), t)
    pl.close()
    imageio.imwrite(out, img)
    print(f"Saved: {out}")


# ── Full animation ─────────────────────────────────────────────────────────────

def render_all(out: str | None = None, fps: float = 8.0) -> None:
    # Only render nested snapshots (t <= 300s)
    all_paths = sorted(glob.glob(f"{SNAPSHOT_DIR}/snap_*.npz"))
    if not all_paths:
        print(f"No snapshots in '{SNAPSHOT_DIR}/'. Run main_nested.py first.")
        sys.exit(1)

    # Filter to nested simulation duration (t <= 300s)
    paths = []
    for p in all_paths:
        try:
            d = np.load(p)
            t = float(d["t"])
            # Only include snapshots within nested simulation time
            if t <= cfg.T_END + 1e-3:
                paths.append(p)
        except Exception as e:
            print(f"Warning: Could not load {p}: {e}")
            continue

    if not paths:
        print(f"No snapshots found for t <= {cfg.T_END}s. Check SNAPSHOT_DIR.")
        sys.exit(1)

    ext = "mp4" if (out and out.endswith(".mp4")) else "gif"
    if out is None:
        out = _next_output_path(ext)
    else:
        # If user specifies path, ensure directory exists
        out_dir = os.path.dirname(out)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

    print(f"Rendering {len(paths)} frames → {out}  (fps={fps})")
    print(f"Nested domain: {cfg.NX}×{cfg.NY}×{cfg.NZ}, dx={cfg.DX}m")

    grid   = _make_grid()
    pl, kw = _make_plotter()

    imgs = []
    for i, path in enumerate(paths):
        d      = np.load(path)
        t      = float(d["t"])
        qc_gkg = (d["qc"] * 1e3).astype(np.float32).ravel(order="F")
        imgs.append(_render_frame(pl, kw, grid, qc_gkg, t))
        print(f"  frame {i+1:3d}/{len(paths)}  t={t:5.0f}s", end="\r", flush=True)

    pl.close()
    print()

    if out.endswith(".mp4"):
        imageio.mimsave(out, imgs, fps=fps, codec="libx264", quality=8)
    else:
        imageio.mimsave(out, imgs, fps=fps, loop=0)

    print(f"Saved: {out}  ({len(imgs)} frames)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Render 3D animation of nested cloud simulation")
    p.add_argument("--out",    default=None, help="Output file (.gif or .mp4); default auto-numbers")
    p.add_argument("--fps",    type=float, default=8.0, help="Frames per second")
    p.add_argument("--single", metavar="SNAP.npz", help="Render one frame to PNG")
    args = p.parse_args()

    if args.single:
        render_single(args.single)
    else:
        render_all(out=args.out, fps=args.fps)


if __name__ == "__main__":
    main()

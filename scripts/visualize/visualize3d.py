"""
Off-screen 3D cloud renderer — volume rendering of qc.

Renders each snapshot to a frame and saves a GIF or MP4.
No interactive window, fixed camera, lightweight.

Usage:
    python3 visualize3d.py                       # → clouds.gif
    python3 visualize3d.py --out clouds.mp4      # → MP4 (needs ffmpeg)
    python3 visualize3d.py --fps 6               # playback speed
    python3 visualize3d.py --single snap_0003600.npz   # one PNG
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

import config as cfg
from grid import Grid as SimGrid

SNAPSHOT_DIR = "snapshots"
GIF_DIR      = "cloud_gifs"


def _next_output_path(ext: str) -> str:
    """Return cloud_gifs/clouds_N.<ext> where N is one above the current max."""
    os.makedirs(GIF_DIR, exist_ok=True)
    existing = glob.glob(os.path.join(GIF_DIR, f"clouds_*.{ext}"))
    nums = [int(m.group(1)) for f in existing
            if (m := re.search(r"clouds_(\d+)\.", os.path.basename(f)))]
    n = max(nums) + 1 if nums else 1
    return os.path.join(GIF_DIR, f"clouds_{n}.{ext}")


Z_EXP   = 2.0    # vertical exaggeration so clouds don't look pancake-flat
QC_MAX  = 1.5    # g/kg — top of colour scale
QR_MAX  = 2.0    # kg/m² — rain water path for ground overlay


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

def _make_grid(nx=None, ny=None, dx=None, dy=None, z_face=None) -> pv.RectilinearGrid:
    """Make grid from snapshot dimensions or config defaults."""
    if nx is None:
        g = SimGrid()
        nx, ny = cfg.NX, cfg.NY
        dx, dy = cfg.DX, cfg.DY
        z_face = g.z_face
    xf = np.arange(nx + 1, dtype=np.float64) * dx
    yf = np.arange(ny + 1, dtype=np.float64) * dy
    zf = z_face * Z_EXP
    return pv.RectilinearGrid(xf, yf, zf)


# ── Plotter setup (fixed camera, built once) ───────────────────────────────────

def _make_plotter(nx=None, ny=None, dx=None, dy=None, z_face=None) -> tuple[pv.Plotter, dict]:
    """
    Build the plotter with all static elements (ground, box, axes, camera).
    Returns the plotter and a dict of params needed to re-add the volume each frame.
    """
    if nx is None:
        g = SimGrid()
        nx, ny = cfg.NX, cfg.NY
        dx, dy = cfg.DX, cfg.DY
        z_face = g.z_face

    xm = nx * dx
    ym = ny * dy
    zm = z_face[-1] * Z_EXP

    pl = pv.Plotter(off_screen=True, window_size=(1280, 720))
    pl.set_background("white")

    # Ground mesh (will be updated per frame with rain colors)
    # Create a structured grid for the ground to show spatial rain patterns
    x_ground = np.linspace(0, xm, nx)
    y_ground = np.linspace(0, ym, ny)
    xx, yy = np.meshgrid(x_ground, y_ground, indexing='ij')
    zz = np.zeros_like(xx)
    ground_mesh = pv.StructuredGrid(xx, yy, zz)

    # Domain wireframe
    pl.add_mesh(
        pv.Box(bounds=(0, xm, 0, ym, 0, zm)),
        style="wireframe", color="white", opacity=0.12, line_width=1, name="box",
    )
    pl.add_axes(line_width=3)

    # Fixed camera — outside the +x+y corner, elevated, looking at cloud-level centre
    cam_z_idx = min(25, len(z_face) - 1)
    pl.camera_position = [
        (xm * 1.5,  ym * 1.5,  zm * 1.2),
        (xm * 0.5,  ym * 0.5,  z_face[cam_z_idx] * Z_EXP),
        (0, 0, 1),
    ]

    vol_kwargs = dict(
        clim    = [0.0, QC_MAX],
        cmap    = "Blues",
        opacity = [0, 0, 0.4, 0.75, 0.92],   # transparent→opaque with scalar value
        scalar_bar_args = dict(
            title="qc [g/kg]", title_font_size=13, label_font_size=11,
            position_x=0.87, position_y=0.05, height=0.35, width=0.04,
        ),
    )
    return pl, vol_kwargs, ground_mesh


# ── Render one frame: set data → add volume → screenshot → remove volume ───────

def _render_frame(pl: pv.Plotter, vol_kwargs: dict, grid: pv.RectilinearGrid,
                  ground_mesh: pv.StructuredGrid, qc_gkg: np.ndarray,
                  qr_kgkg: np.ndarray, dz_face: np.ndarray, t: float, nx: int, ny: int, nz: int) -> np.ndarray:
    """
    Data must be in the grid BEFORE add_volume — VTK doesn't pick up in-place updates.
    So we: set data → add_volume → render → screenshot → remove volume actor.

    Also computes rain water path (RWP) and colors the ground accordingly.
    """
    grid.cell_data["qc"] = qc_gkg
    vol_actor = pl.add_volume(grid, scalars="qc", name="cloud_vol", **vol_kwargs)

    # Calculate rain water path (RWP) = column integral of qr [kg/m²]
    # qr_kgkg has shape (nx, ny, nz) in Fortran order, need to reshape
    qr_3d = qr_kgkg.reshape((nx, ny, nz), order='F')
    rwp = np.sum(qr_3d * dz_face[np.newaxis, np.newaxis, :], axis=2)  # (nx, ny) in kg/m²

    # Color ground based on rain: green (no rain) → blue (heavy rain)
    # Flatten for pyvista (structured grid point data)
    rwp_flat = rwp.ravel(order='F')
    ground_mesh.point_data["rwp"] = rwp_flat

    # Add ground with rain coloring
    ground_actor = pl.add_mesh(
        ground_mesh,
        scalars="rwp",
        clim=[0.0, QR_MAX],
        cmap="Greens_r",  # Green (dry) → white (wet)
        opacity=0.85,
        name="ground",
    )

    pl.add_text(f"Cloud water qc  —  t = {t/60:.0f} min",
                position="upper_edge", font_size=12, color="black", name="title")
    pl.render()
    img = pl.screenshot(return_img=True)
    pl.remove_actor(vol_actor)
    pl.remove_actor(ground_actor)
    return img


# ── Single frame ───────────────────────────────────────────────────────────────

def render_single(path: str, out: str = "cloud_frame.png") -> None:
    d = np.load(path)
    t = float(d["t"])

    # Extract dimensions from snapshot shape
    qc_shape = d["qc"].shape
    nx, ny, nz = qc_shape

    # Extract grid parameters from snapshot
    dx = d.get("dx", cfg.DX)
    dy = d.get("dy", cfg.DY)
    z_face = d["z_face"] if "z_face" in d else None

    # If z_face not in snapshot, fall back to SimGrid (but this should always exist)
    if z_face is None:
        print("Warning: z_face not in snapshot, using SimGrid")
        sim_grid = SimGrid()
        z_face = sim_grid.z_face
        dz_face = sim_grid.dz_face
    else:
        # Compute dz_face from z_face
        dz_face = np.diff(z_face)

    grid = _make_grid(nx=nx, ny=ny, dx=dx, dy=dy, z_face=z_face)
    pl, kw, gmesh = _make_plotter(nx=nx, ny=ny, dx=dx, dy=dy, z_face=z_face)

    img = _render_frame(
        pl, kw, grid, gmesh,
        (d["qc"] * 1e3).astype(np.float32).ravel(order="F"),
        d["qr"].astype(np.float32).ravel(order="F"),
        dz_face,
        t,
        nx, ny, nz
    )
    pl.close()
    imageio.imwrite(out, img)
    print(f"Saved: {out}")


# ── Full animation ─────────────────────────────────────────────────────────────

def render_all(out: str | None = None, fps: float = 8.0, snapshot_dir: str | None = None) -> None:
    snap_dir = snapshot_dir or SNAPSHOT_DIR
    all_paths = sorted(glob.glob(f"{snap_dir}/snap_*.npz"))
    if not all_paths:
        print(f"No snapshots in '{snap_dir}/'. Run main.py first.")
        sys.exit(1)

    # filter to simulation duration
    paths = [p for p in all_paths if float(np.load(p)["t"]) <= cfg.T_END + 1e-3]
    if not paths:
        paths = all_paths  # fallback: render everything

    ext = "mp4" if (out and out.endswith(".mp4")) else "gif"
    if out is None:
        out = _next_output_path(ext)
    else:
        os.makedirs(GIF_DIR, exist_ok=True)

    # Load first snapshot to extract grid dimensions
    first_data = np.load(paths[0])

    # Extract dimensions from snapshot shape
    nx, ny, nz = first_data["qc"].shape

    # Extract grid parameters from snapshot
    dx = first_data.get("dx", cfg.DX)
    dy = first_data.get("dy", cfg.DY)
    z_face = first_data["z_face"] if "z_face" in first_data else None

    # If z_face not in snapshot, fall back to SimGrid (but this should always exist)
    if z_face is None:
        print("Warning: z_face not in snapshot, using SimGrid")
        sim_grid = SimGrid()
        z_face = sim_grid.z_face
        dz_face = sim_grid.dz_face
    else:
        # Compute dz_face from z_face
        dz_face = np.diff(z_face)

    print(f"Rendering {len(paths)} frames → {out}  (fps={fps})")
    print(f"Grid: {nx}×{ny}×{nz}, dx={dx}m, dy={dy}m")

    grid = _make_grid(nx=nx, ny=ny, dx=dx, dy=dy, z_face=z_face)
    pl, kw, gmesh = _make_plotter(nx=nx, ny=ny, dx=dx, dy=dy, z_face=z_face)

    imgs = []
    for i, path in enumerate(paths):
        d      = np.load(path)
        t      = float(d["t"])
        qc_gkg = (d["qc"] * 1e3).astype(np.float32).ravel(order="F")
        qr_kgkg = d["qr"].astype(np.float32).ravel(order="F")
        imgs.append(_render_frame(pl, kw, grid, gmesh, qc_gkg, qr_kgkg, dz_face, t, nx, ny, nz))
        print(f"  frame {i+1:3d}/{len(paths)}  t={t/60:5.1f} min", end="\r", flush=True)

    pl.close()
    print()

    if out.endswith(".mp4"):
        imageio.mimsave(out, imgs, fps=fps, codec="libx264", quality=8)
    else:
        imageio.mimsave(out, imgs, fps=fps, loop=0)

    print(f"Saved: {out}  ({len(imgs)} frames)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out",    default=None, help="Output file (.gif or .mp4); default auto-numbers in cloud_gifs/")
    p.add_argument("--fps",    type=float, default=8.0)
    p.add_argument("--single", metavar="SNAP.npz", help="Render one frame to PNG")
    p.add_argument("--snapshot-dir", default=None, help="Custom snapshot directory to render")
    args = p.parse_args()

    if args.single:
        render_single(args.single)
    else:
        render_all(out=args.out, fps=args.fps, snapshot_dir=args.snapshot_dir)


if __name__ == "__main__":
    main()

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
import numpy as np
import pyvista as pv
import imageio
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
    pl.camera_position = [
        (xm * 1.5,  ym * 1.5,  zm * 1.2),
        (xm * 0.5,  ym * 0.5,  g.z_face[25] * Z_EXP),
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
    pl.add_text(f"Cloud water qc  —  t = {t/60:.0f} min",
                position="upper_edge", font_size=12, color="black", name="title")
    pl.render()
    img = pl.screenshot(return_img=True)
    pl.remove_actor(vol_actor)
    return img


# ── Single frame ───────────────────────────────────────────────────────────────

def render_single(path: str, out: str = "cloud_frame.png") -> None:
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
    all_paths = sorted(glob.glob(f"{SNAPSHOT_DIR}/snap_*.npz"))
    if not all_paths:
        print(f"No snapshots in '{SNAPSHOT_DIR}/'. Run main.py first.")
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

    print(f"Rendering {len(paths)} frames → {out}  (fps={fps})")
    grid   = _make_grid()
    pl, kw = _make_plotter()

    imgs = []
    for i, path in enumerate(paths):
        d      = np.load(path)
        t      = float(d["t"])
        qc_gkg = (d["qc"] * 1e3).astype(np.float32).ravel(order="F")
        imgs.append(_render_frame(pl, kw, grid, qc_gkg, t))
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
    args = p.parse_args()

    if args.single:
        render_single(args.single)
    else:
        render_all(out=args.out, fps=args.fps)


if __name__ == "__main__":
    main()

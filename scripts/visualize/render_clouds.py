"""
Fast, lit 3-D cloud renderer for simulation snapshots → GIF / MP4.

Clouds are drawn as a smoothed iso-surface of qc (white, sun-lit) instead of a
volume ray-cast, which is ~20–50× cheaper on a software (Mesa/llvmpipe)
renderer and still fast on a GPU. The ground shows the clouds' shadows cast
along the sun direction (computed in numpy from the liquid-water path, using
the periodic domain) and a blue-grey tint where rain reaches the surface.

Usage:
    python scripts/visualize/render_clouds.py RUN_DIR --out cloud_gifs/YYYYMMDD_desc.gif
    python scripts/visualize/render_clouds.py RUN_DIR --single 7200 --out frame.png
Options:
    --fps 4            playback speed
    --zexp 1.5         vertical exaggeration
    --mode volume      white volume rendering instead of iso-surfaces (slower, softer)
    --jobs N           render frames in N parallel processes (default: 1)
    --smooth 0.8       gaussian sigma (grid cells) applied to qc before contouring;
                       0 = raw field (faceted, true to the grid), default 0.8
    --xrange X0 X1     show only x in [X0, X1] km (e.g. the test region of a fringe run)
Grid spacing is read from the run's effective_config.txt (snapshots don't store it).

Headless / no GPU: prefix with VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow.
"""

import argparse
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyvista as pv
import imageio
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap

QC_ISO      = 0.01     # g/kg, cloud surface (same as the 1e-5 kg/kg cloud mask in simulation.py)
QC_SMOOTH   = 0.8      # default gaussian sigma (cells) applied to qc before contouring
EXTINCTION  = 150.0    # m²/kg, shadow optical depth = EXTINCTION · LWP
SUN_ELEV    = 45.0     # deg above horizon
SUN_AZIM    = 225.0    # deg, direction the sun shines FROM (225 = from the south-west)
RHO_AIR     = 1.1      # kg/m³, for the liquid-water path (only affects shadow darkness)
WINDOW      = (1280, 720)

GROUND_RGB  = np.array([0.42, 0.55, 0.30])   # sunlit fields
SHADOW_RGB  = np.array([0.20, 0.28, 0.17])   # fully shaded fields
RAIN_RGB    = np.array([0.35, 0.42, 0.55])   # wet ground tint
# Cloud colour by height above cloud base: grey base → white tops (smooth gradient)
CLOUD_CMAP  = LinearSegmentedColormap.from_list("cloud", ["#8e98a3", "#d9dee4", "#ffffff"])
CLOUD_GRAD  = 600.0    # m (unexaggerated) over which the base grey fades to white


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _cloud_levels(paths) -> tuple[int, int]:
    """Vertical index range containing cloud in any snapshot (plus a margin)."""
    k_lo, k_hi = None, None
    for p in paths:
        qc = np.load(p)["qc"]
        any_k = np.nonzero((qc > QC_ISO * 1e-3 * 0.5).any(axis=(0, 1)))[0]
        if len(any_k):
            k_lo = any_k[0] if k_lo is None else min(k_lo, any_k[0])
            k_hi = any_k[-1] if k_hi is None else max(k_hi, any_k[-1])
    if k_lo is None:
        return 0, 1
    return max(k_lo - 2, 0), k_hi + 3


def _ground_colors(qc, qr, z_face, dx, dy) -> np.ndarray:
    """(nx, ny, 3) ground RGB: sun-projected cloud shadows + rain tint (periodic domain)."""
    nx, ny, nz = qc.shape
    dz = np.diff(z_face)
    zc = 0.5 * (z_face[:-1] + z_face[1:])
    # Horizontal shift of each level's shadow on the ground, away from the sun
    run = zc / np.tan(np.radians(SUN_ELEV))
    az = np.radians(SUN_AZIM)
    sx = np.rint(-run * np.sin(az) / dx).astype(int)   # sun from SW → shadows toward NE
    sy = np.rint(-run * np.cos(az) / dy).astype(int)
    tau = np.zeros((nx, ny))
    for k in np.nonzero(qc.max(axis=(0, 1)) > 0)[0]:
        lwp_k = RHO_AIR * qc[:, :, k] * dz[k]
        tau += np.roll(lwp_k, (sx[k], sy[k]), axis=(0, 1))
    tau = gaussian_filter(tau, sigma=0.7, mode="wrap")          # soften 100 m pixels
    shade = np.exp(-EXTINCTION * tau / np.sin(np.radians(SUN_ELEV)))[..., None]
    rgb = SHADOW_RGB + (GROUND_RGB - SHADOW_RGB) * shade
    rain = np.clip(RHO_AIR * (qr[:, :, 0] * 1e3) / 0.05, 0.0, 1.0)[..., None]   # surface qr
    return rgb * (1 - 0.7 * rain) + RAIN_RGB * 0.7 * rain


def _caption(d, z) -> str:
    qc = d["qc"]
    col = qc.max(axis=2) > QC_ISO * 1e-3
    txt = f"t = {float(d['t'])/60:.0f} min    cloud cover {100*col.mean():.0f}%"
    cloudy_k = np.nonzero((qc > QC_ISO * 1e-3).any(axis=(0, 1)))[0]
    if len(cloudy_k):
        txt += f"    base {z[cloudy_k[0]]:.0f} m    top {z[cloudy_k[-1]]:.0f} m"
    return txt


# ── Frame rendering ───────────────────────────────────────────────────────────

def render_frame(path: str, k_range: tuple[int, int], zexp: float, mode: str,
                 smooth: float = QC_SMOOTH, dx: float = 100.0, dy: float = 100.0,
                 xrange: tuple[float, float] | None = None) -> np.ndarray:
    d = np.load(path)
    qc = d["qc"].astype(np.float32) * 1e3                 # g/kg
    qr = d["qr"].astype(np.float32)
    z_face = d["z_face"]
    z = d["z"]
    rgb = _ground_colors(qc * 1e-3, qr, z_face, dx, dy)   # shadows on the full (periodic) domain
    periodic_x = xrange is None
    if xrange is not None:
        i0, i1 = int(round(xrange[0] * 1e3 / dx)), int(round(xrange[1] * 1e3 / dx))
        qc, rgb = qc[i0:i1], rgb[i0:i1]
    nx, ny, _ = qc.shape
    Lx, Ly = nx * dx, ny * dy

    pl = pv.Plotter(off_screen=True, window_size=WINDOW, lighting="none")
    pl.set_background("#9cc4e8", top="#3d7cc9")          # sky gradient

    # Sun + soft fill light
    az, el = np.radians(SUN_AZIM), np.radians(SUN_ELEV)
    sun_dir = np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])
    centre = np.array([Lx / 2, Ly / 2, 0.0])
    pl.add_light(pv.Light(position=tuple(centre + 1e5 * sun_dir), focal_point=tuple(centre),
                          color="#fff6e8", intensity=1.0, light_type="scene light"))
    pl.add_light(pv.Light(position=tuple(centre + 1e5 * np.array([-sun_dir[0], -sun_dir[1], 0.6])),
                          focal_point=tuple(centre), color="#cfe0ff", intensity=0.35,
                          light_type="scene light"))

    # Ground with shadows and rain
    gx, gy = np.meshgrid(np.arange(nx + 1) * dx, np.arange(ny + 1) * dy, indexing="ij")
    ground = pv.StructuredGrid(gx, gy, np.zeros_like(gx))
    ground.cell_data["rgb"] = (rgb.reshape(-1, 3, order="F") * 255).astype(np.uint8)
    pl.add_mesh(ground, scalars="rgb", rgb=True, ambient=0.45, diffuse=0.6, specular=0.0)

    # Clouds: cropped to the cloud layer, smoothed (periodic in x, y)
    k0, k1 = k_range
    k1 = min(k1, qc.shape[2])
    sub = qc[:, :, k0:k1]
    if smooth > 0:
        sub = gaussian_filter(sub, sigma=smooth,
                              mode=("wrap" if periodic_x else "nearest", "wrap", "nearest"))
    # Zero-pad the lateral edges so clouds cut by the periodic boundary are closed surfaces
    sub = np.pad(sub, ((1, 1), (1, 1), (0, 0)))
    zz = z[k0:k1] * zexp
    xs = np.concatenate([[0.0], (np.arange(nx) + 0.5) * dx, [Lx]])
    ys = np.concatenate([[0.0], (np.arange(ny) + 0.5) * dy, [Ly]])
    grid = pv.RectilinearGrid(xs, ys, zz)
    grid.point_data["qc"] = sub.ravel(order="F")

    if mode == "volume":
        opac = np.clip(np.linspace(-0.05, 1.0, 64), 0, None) ** 0.6 * 0.9
        pl.add_volume(grid, scalars="qc",
                      cmap="Greys", clim=[0.0, 1.5], opacity=opac, shade=True,
                      show_scalar_bar=False, ambient=0.5, diffuse=0.7)
    elif sub.max() > QC_ISO:
        surf = grid.contour([QC_ISO], scalars="qc").compute_normals(auto_orient_normals=True)
        # Brighter at the top (sunlit), slightly grey at the base: colour by height
        cloudy = np.nonzero((qc > QC_ISO).any(axis=(0, 1)))[0]
        z_base = z[cloudy[0]] * zexp if len(cloudy) else zz[0]
        surf.point_data["h"] = surf.points[:, 2]
        pl.add_mesh(surf, scalars="h", cmap=CLOUD_CMAP,
                    clim=[z_base, z_base + CLOUD_GRAD * zexp], show_scalar_bar=False,
                    smooth_shading=True,
                    ambient=0.35, diffuse=0.75, specular=0.15, specular_power=12)

    pl.add_text(_caption(d, z), position="upper_left", font_size=11, color="white", shadow=True)

    # Camera: oblique view from the south-west, looking at the cloud layer
    focal = (Lx / 2, Ly / 2, 0.35 * zz[-1])
    dist = 1.25 * max(Lx, Ly)
    pl.camera.focal_point = focal
    pl.camera.position = (Lx / 2 - 0.75 * dist, Ly / 2 - 0.95 * dist, 0.55 * dist)
    pl.camera.up = (0, 0, 1)
    pl.camera.view_angle = 32

    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def _render_one(args):
    return render_frame(*args)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--out", required=True, help=".gif, .mp4, or .png with --single")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--zexp", type=float, default=1.5)
    ap.add_argument("--mode", choices=["iso", "volume"], default="iso")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--smooth", type=float, default=QC_SMOOTH,
                    help="gaussian sigma in grid cells applied to qc (0 = no smoothing)")
    ap.add_argument("--xrange", type=float, nargs=2, metavar=("X0_KM", "X1_KM"),
                    help="show only this x range (km)")
    ap.add_argument("--single", type=float, metavar="T_SECONDS", help="render only the snapshot at this time")
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.run_dir, "snapshots", "snap_*.npz")))
    if not paths:
        raise SystemExit(f"No snapshots in {a.run_dir}/snapshots")
    k_range = _cloud_levels(paths)
    dx = dy = 100.0
    eff = os.path.join(a.run_dir, "effective_config.txt")
    if os.path.exists(eff):
        for line in open(eff):
            if line.startswith("DX = "):
                dx = float(line.split("=")[1])
            elif line.startswith("DY = "):
                dy = float(line.split("=")[1])
    else:
        print("No effective_config.txt: assuming dx = dy = 100 m")
    xr = tuple(a.xrange) if a.xrange else None

    if a.single is not None:
        times = np.array([float(np.load(p)["t"]) for p in paths])
        p = paths[int(np.argmin(np.abs(times - a.single)))]
        t0 = time.time()
        imageio.imwrite(a.out, render_frame(p, k_range, a.zexp, a.mode, a.smooth, dx, dy, xr))
        print(f"Saved {a.out} ({time.time() - t0:.1f} s)")
        return

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    print(f"Rendering {len(paths)} frames ({a.mode}, smooth={a.smooth}, "
          f"levels {k_range[0]}–{k_range[1]}) → {a.out}", flush=True)
    t0 = time.time()
    jobs = [(p, k_range, a.zexp, a.mode, a.smooth, dx, dy, xr) for p in paths]
    if a.jobs > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            imgs = list(ex.map(_render_one, jobs))
    else:
        imgs = []
        for i, j in enumerate(jobs):
            imgs.append(_render_one(j))
            print(f"  frame {i+1}/{len(jobs)}", flush=True)
    if a.out.endswith(".mp4"):
        imageio.mimsave(a.out, imgs, fps=a.fps, codec="libx264", quality=8)
    else:
        imageio.mimsave(a.out, imgs, fps=a.fps, loop=0)
    print(f"Saved {a.out}: {len(imgs)} frames in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()

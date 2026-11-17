"""
Fast, lit 3-D cloud renderer for simulation snapshots → MP4 (or GIF / PNG).

Liquid cloud is drawn as an opaque iso-surface of the liquid part of qc (white, sun-lit, grey at the
base of each layer), which is ~20–50× cheaper than a volume ray-cast on a software (Mesa/llvmpipe)
renderer. Ice (cloud ice + graupel/snow) is a white volume whose opacity follows its extinction:
1 − exp(−β Δs), β = ρ₀ (κ_ice q_i + κ_gr q_g), so thin cirrus is a see-through veil and a dense anvil
is opaque. Liquid/ice split as in the model: liquid fraction α(T) linear between T_ICE_00 and T_ICE_0,
T = (θ₀ + θ′) π₀ with θ₀ from the run's profiles.npz (without it, or without ICE_SCHEME, everything is
liquid). The ground shows shadows from the optical depth along the sun direction and a blue-grey
tint where rain reaches the surface; a height ruler (km) stands at the back-left corner.

Usage:
    python scripts/visualize/render_clouds.py RUN_DIR --out cloud_gifs/YYYYMMDD_desc.mp4
    python scripts/visualize/render_clouds.py RUN_DIR --single 7200 --out frame.png
Options:
    --fps 4            playback speed
    --zexp 1.5         vertical exaggeration
    --mode volume      white volume rendering instead of iso-surfaces (slower, softer)
    --jobs N           render frames in N parallel processes (default: 1)
    --smooth 0.8       gaussian sigma (grid cells) applied to qc before contouring;
                       0 = raw field (faceted, true to the grid), default 0.8
    --xrange X0 X1     show only x in [X0, X1] km (e.g. the test region of a fringe run)
    --trange T0 T1     only snapshots with T0 <= t <= T1 (s)
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
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction

QC_ISO      = 0.01     # g/kg, cloud surface (same as the 1e-5 kg/kg cloud mask in simulation.py)
QC_SMOOTH   = 0.8      # default gaussian sigma (cells) applied to qc before contouring
KAPPA_LIQ   = 150.0    # m²/kg, mass extinction of droplets (r_eff 10 µm: 3 / (2 ρ_w r))
KAPPA_ICE   = 55.0     # m²/kg, cloud ice (r_eff 30 µm: 3 / (2 ρ_ice r))
KAPPA_GR    = 8.0      # m²/kg, graupel/snow (mm-sized, ρ 400 kg/m³)
ICE_UNIT    = 50.0     # m, path length on which the ice volume's opacity table is defined
ICE_BETA_MAX = 5.0 / ICE_UNIT   # 1/m, extinction above which an ICE_UNIT path is opaque (0.993)
SUN_ELEV    = 45.0     # deg above horizon
SUN_AZIM    = 225.0    # deg, direction the sun shines FROM (225 = from the south-west)
WINDOW      = (1280, 720)
G, CP, RD, P00 = 9.81, 1004.0, 287.0, 1.0e5

GROUND_RGB  = np.array([0.42, 0.55, 0.30])   # sunlit fields
SHADOW_RGB  = np.array([0.20, 0.28, 0.17])   # fully shaded fields
RAIN_RGB    = np.array([0.35, 0.42, 0.55])   # wet ground tint
# Cloud colour by height above cloud base: grey base → white tops (smooth gradient)
CLOUD_CMAP  = LinearSegmentedColormap.from_list("cloud", ["#8e98a3", "#d9dee4", "#ffffff"])
CLOUD_GRAD  = 600.0    # m (unexaggerated) over which the base grey fades to white
ICE_CMAP    = LinearSegmentedColormap.from_list("ice", ["#ffffff", "#eef1f5"])   # ice volume, dense parts faintly grey


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _cloud_levels(paths) -> tuple[int, int]:
    """Vertical index range containing cloud in any snapshot (plus a margin)."""
    k_lo, k_hi = None, None
    for p in paths:
        d = np.load(p)
        q = d["qc"] + d["qg"] if "qg" in d else d["qc"]
        any_k = np.nonzero((q > QC_ISO * 1e-3 * 0.5).any(axis=(0, 1)))[0]
        if len(any_k):
            k_lo = any_k[0] if k_lo is None else min(k_lo, any_k[0])
            k_hi = any_k[-1] if k_hi is None else max(k_hi, any_k[-1])
    if k_lo is None:
        return 0, 1
    return max(k_lo - 2, 0), k_hi + 3


def _shift(a: np.ndarray, sx: int, sy: int, open_x: bool, open_y: bool) -> np.ndarray:
    """Shift a 2-D field by (sx, sy) cells: wrap-around in periodic directions, zero fill in open ones."""
    out = np.roll(a, (sx, sy), axis=(0, 1))
    if open_x and sx:
        if sx > 0:
            out[:sx] = 0.0
        else:
            out[sx:] = 0.0
    if open_y and sy:
        if sy > 0:
            out[:, :sy] = 0.0
        else:
            out[:, sy:] = 0.0
    return out


def _reference(run_dir: str, z: np.ndarray, eff: dict) -> dict:
    """Reference θ₀, π₀, ρ₀ of the run and the ice switch. θ₀ = first profile of profiles.npz minus the
    first snapshot's mean θ′ (snapshots store θ′). Without profiles.npz: exponential ρ₀, all cloud liquid."""
    ice = eff.get("ICE_SCHEME", "None") not in ("None", "'none'", "'warm'")
    prof = os.path.join(run_dir, "profiles.npz")
    snap0 = sorted(glob.glob(os.path.join(run_dir, "snapshots", "snap_*.npz")))[0]
    if not os.path.exists(prof):
        if ice:
            print("No profiles.npz (run not finished?): ice drawn as liquid")
        return dict(ice=False, rho0=1.2 * np.exp(-z / 8500.0))
    theta0 = np.load(prof)["theta"][0] - np.load(snap0)["prof_theta"]
    pi0 = np.ones_like(z)
    for k in range(1, len(z)):
        pi0[k] = pi0[k - 1] - G * (z[k] - z[k - 1]) / (CP * 0.5 * (theta0[k] + theta0[k - 1]))
    return dict(ice=ice, theta0=theta0, pi0=pi0, rho0=P00 * pi0 ** (CP / RD) / (RD * theta0 * pi0),
                t_ice_0=float(eff.get("T_ICE_0", 273.15)), t_ice_00=float(eff.get("T_ICE_00", 253.15)))


def _phases(d, ref: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(q_liq, q_ice, q_g) [kg/kg]: cloud condensate split by the model's α(T), plus graupel/snow."""
    qc = d["qc"].astype(np.float32)
    qg = d["qg"].astype(np.float32) if "qg" in d else np.zeros_like(qc)
    if not ref["ice"]:
        return qc, np.zeros_like(qc), qg
    T = (ref["theta0"] + d["theta"]) * ref["pi0"]
    alpha = np.clip((T - ref["t_ice_00"]) / (ref["t_ice_0"] - ref["t_ice_00"]), 0.0, 1.0).astype(np.float32)
    return alpha * qc, (1.0 - alpha) * qc, qg


def _ground_colors(beta, qr, z_face, dx, dy, open_x=False, open_y=False) -> np.ndarray:
    """(nx, ny, 3) ground RGB: sun-projected cloud shadows from the extinction β [1/m] + rain tint.
    Shadows wrap around only in periodic directions (open: they leave the domain)."""
    nx, ny, nz = beta.shape
    dz = np.diff(z_face)
    zc = 0.5 * (z_face[:-1] + z_face[1:])
    # Horizontal shift of each level's shadow on the ground, away from the sun
    run = zc / np.tan(np.radians(SUN_ELEV))
    az = np.radians(SUN_AZIM)
    sx = np.rint(-run * np.sin(az) / dx).astype(int)   # sun from SW → shadows toward NE
    sy = np.rint(-run * np.cos(az) / dy).astype(int)
    tau = np.zeros((nx, ny))
    for k in np.nonzero(beta.max(axis=(0, 1)) > 0)[0]:
        tau += _shift(beta[:, :, k] * dz[k], sx[k], sy[k], open_x, open_y)
    tau = gaussian_filter(tau, sigma=0.7, mode=("nearest" if open_x else "wrap",
                                                "nearest" if open_y else "wrap"))   # soften pixels
    shade = np.exp(-tau / np.sin(np.radians(SUN_ELEV)))[..., None]
    rgb = SHADOW_RGB + (GROUND_RGB - SHADOW_RGB) * shade
    rain = np.clip(1.1 * (qr[:, :, 0] * 1e3) / 0.05, 0.0, 1.0)[..., None]   # surface qr
    return rgb * (1 - 0.7 * rain) + RAIN_RGB * 0.7 * rain


def _layers(q, max_gap: int = 2) -> list[tuple[int, int, float]]:
    """Cloud layers as (k_base, k_top, cover): runs of cloudy levels (any column above QC_ISO) split where
    more than max_gap levels are clear; cover = fraction of columns with cloud anywhere in the layer."""
    cloudy = q > QC_ISO * 1e-3
    ks = np.nonzero(cloudy.any(axis=(0, 1)))[0]
    if not len(ks):
        return []
    runs = np.split(ks, np.nonzero(np.diff(ks) > max_gap + 1)[0] + 1)
    return [(r[0], r[-1], cloudy[:, :, r[0]:r[-1] + 1].any(axis=2).mean()) for r in runs]


def _caption(t, q_liq, q_ice, z, dz, rho0) -> str:
    """Time, total cover, then per layer (highest first): base–top, phase, cover, median optical depth of its
    cloudy columns, and visible cover (columns with τ > 0.1)."""
    qc = q_liq + q_ice
    col = qc.max(axis=2) > QC_ISO * 1e-3
    txt = f"t = {t/60:.0f} min    cloud cover {100*col.mean():.0f}%"
    for k0, k1, cov in _layers(qc)[::-1]:
        s = slice(k0, k1 + 1)
        w = rho0[s] * dz[s]
        tau = (KAPPA_LIQ * q_liq[:, :, s] + KAPPA_ICE * q_ice[:, :, s]) @ w
        lay = (qc[:, :, s] > QC_ISO * 1e-3).any(axis=2)
        f_ice = (q_ice[:, :, s] @ w).sum() / max((qc[:, :, s] @ w).sum(), 1e-30)
        phase = "ice" if f_ice > 0.9 else "liquid" if f_ice < 0.1 else "mixed"
        txt += (f"\n  {z[k0]/1e3:4.1f}–{z[k1]/1e3:4.1f} km  {phase:6s} {100*cov:3.0f}%"
                f"   tau {np.median(tau[lay]):4.1f}   visible (tau > 0.1) {100*(tau > 0.1).mean():3.0f}%")
    return txt


def _height_axis(pl, Lx: float, Ly: float, top: float, zexp: float) -> None:
    """Height ruler at the left corner (0, Ly) in true km, with faint km lines on the two back walls
    (y = Ly and x = Lx) so layer heights can be read against the clouds."""
    step = 500.0 if top <= 3000.0 else 1000.0 if top <= 12000.0 else 2000.0
    ticks = np.arange(0.0, top + 1.0, step)
    zt = np.append(ticks, top) * zexp if ticks[-1] < top - 0.3 * step else ticks * zexp
    pl.add_mesh(pv.Line((0.0, Ly, 0.0), (0.0, Ly, zt[-1])), color="white", line_width=3, lighting=False)
    for h in ticks[1:] * zexp:
        wall = pv.MultipleLines([(0.0, Ly, h), (Lx, Ly, h), (Lx, 0.0, h)])
        pl.add_mesh(wall, color="white", opacity=0.35, line_width=1, lighting=False)
        pl.add_mesh(pv.Line((-0.02 * Lx, Ly, h), (0.0, Ly, h)), color="white", line_width=3, lighting=False)
    fmt = (lambda h: f"{h/1e3:.1f} km") if step < 1000.0 else (lambda h: f"{h/1e3:.0f} km")
    pl.add_point_labels(np.array([(-0.03 * Lx, Ly, h * zexp) for h in ticks[1:]]), [fmt(h) for h in ticks[1:]],
                        font_size=14, text_color="white", shape=None, show_points=False,
                        always_visible=True, justification_horizontal="right")


# ── Frame rendering ───────────────────────────────────────────────────────────

def render_frame(path: str, k_range: tuple[int, int], zexp: float, mode: str,
                 smooth: float = QC_SMOOTH, dx: float = 100.0, dy: float = 100.0,
                 xrange: tuple[float, float] | None = None,
                 open_xy: tuple[bool, bool] = (False, False),
                 terrain: np.ndarray | None = None, ref: dict | None = None) -> np.ndarray:
    d = np.load(path)
    z_face = d["z_face"]
    z = d["z"]
    dz = np.diff(z_face)
    if ref is None:
        ref = dict(ice=False, rho0=1.2 * np.exp(-z / 8500.0))
    q_liq, q_ice, q_g = _phases(d, ref)
    rho0 = ref["rho0"].astype(np.float32)
    beta_ice = rho0 * (KAPPA_ICE * q_ice + KAPPA_GR * q_g)                 # 1/m
    beta = beta_ice + rho0 * KAPPA_LIQ * q_liq
    qr = d["qr"].astype(np.float32)
    open_x, open_y = open_xy
    rgb = _ground_colors(beta, qr, z_face, dx, dy, open_x, open_y)   # shadows on the full domain
    caption = _caption(float(d["t"]), q_liq, q_ice, z, dz, rho0)
    qc = q_liq * 1e3                                                        # g/kg, opaque surface
    periodic_x = xrange is None and not open_x
    periodic_y = not open_y
    if xrange is not None:
        i0, i1 = int(round(xrange[0] * 1e3 / dx)), int(round(xrange[1] * 1e3 / dx))
        qc, rgb, beta_ice = qc[i0:i1], rgb[i0:i1], beta_ice[i0:i1]
        if terrain is not None:
            terrain = terrain[i0:i1 + 1]
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
    gz = np.zeros_like(gx) if terrain is None else terrain * zexp
    ground = pv.StructuredGrid(gx, gy, gz)
    ground.cell_data["rgb"] = (rgb.reshape(-1, 3, order="F") * 255).astype(np.uint8)
    pl.add_mesh(ground, scalars="rgb", rgb=True, ambient=0.45, diffuse=0.6, specular=0.0)
    if terrain is not None and terrain.max() > 0:            # height contours every 100 m
        ground.point_data["h"] = terrain.ravel(order="F")
        levels = np.arange(100.0, terrain.max(), 100.0)
        if len(levels):
            lines = ground.contour(isosurfaces=levels, scalars="h")
            lines.points[:, 2] += 5.0 * zexp                    # lift off the surface (no z-fighting)
            pl.add_mesh(lines, color="#3b3220", line_width=2, lighting=False)

    # Clouds: cropped to the cloud layer, smoothed (wrap-around only in periodic directions)
    k0, k1 = k_range
    k1 = min(k1, qc.shape[2])
    smooth_mode = ("wrap" if periodic_x else "nearest", "wrap" if periodic_y else "nearest", "nearest")
    sub = qc[:, :, k0:k1]
    if smooth > 0:
        sub = gaussian_filter(sub, sigma=smooth, mode=smooth_mode)
    # Zero-pad the lateral edges so clouds cut by the domain boundary are closed surfaces
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
        # Brighter at the top (sunlit), grey at the base: colour by height above the base of the point's layer
        bases = np.array([z[kb] for kb, _, _ in _layers(qc * 1e-3)])
        pz = surf.points[:, 2] / zexp
        below = bases[None, :] <= pz[:, None] + 0.5 * dz[k0]
        base_of = np.where(below.any(axis=1), np.where(below, bases[None, :], -np.inf).max(axis=1), bases[0])
        surf.point_data["h"] = pz - base_of
        pl.add_mesh(surf, scalars="h", cmap=CLOUD_CMAP,
                    clim=[0.0, CLOUD_GRAD], show_scalar_bar=False,
                    smooth_shading=True,
                    ambient=0.35, diffuse=0.75, specular=0.15, specular_power=12)

    # Ice and graupel/snow: white volume, opacity 1 − exp(−β·Δs) per ICE_UNIT of path
    bsub = beta_ice[:, :, k0:k1]
    if mode != "volume" and bsub.max() > 1e-6:
        if smooth > 0:
            bsub = gaussian_filter(bsub, sigma=smooth, mode=smooth_mode)
        vol = pv.RectilinearGrid(xs, ys, zz)
        vol.point_data["beta"] = np.pad(bsub, ((1, 1), (1, 1), (0, 0))).ravel(order="F")
        actor = pl.add_volume(vol, scalars="beta", clim=[0.0, ICE_BETA_MAX], opacity="linear",
                              opacity_unit_distance=ICE_UNIT, cmap=ICE_CMAP, shade=False,
                              show_scalar_bar=False)
        # Exact float opacity table (pyvista's arrays are 8-bit, which loses τ ≈ 0.03 cirrus steps)
        otf = vtkPiecewiseFunction()
        for b in np.linspace(0.0, ICE_BETA_MAX, 128) ** 2 / ICE_BETA_MAX:      # dense near 0
            otf.AddPoint(b, 1.0 - np.exp(-b * ICE_UNIT))
        actor.GetProperty().SetScalarOpacity(otf)
        actor.GetProperty().SetScalarOpacityUnitDistance(ICE_UNIT)

    pl.add_text(caption, position="upper_left", font_size=11, color="white", shadow=True)
    _height_axis(pl, Lx, Ly, zz[-1] / zexp, zexp)

    # Camera: oblique view from the south-west, pulled back until the ground and the highest cloud level
    # of the whole run (zz[-1], the same for every frame) fit in the frame
    top = zz[-1]
    focal = np.array([Lx / 2, Ly / 2, 0.4 * top])
    direction = np.array([-0.75, -0.95, 0.55])
    dist = _fit_distance(focal, direction, (Lx, Ly, top), view_angle=32.0,
                         aspect=WINDOW[0] / WINDOW[1])
    pl.camera.focal_point = tuple(focal)
    pl.camera.position = tuple(focal + direction / np.linalg.norm(direction) * dist)
    pl.camera.up = (0, 0, 1)
    pl.camera.view_angle = 32

    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def _terrain_nodes(eff: str, nx: int, ny: int, dx: float, dy: float) -> np.ndarray | None:
    """Ground height (nx+1, ny+1) at the cell corners from the run's TERRAIN_* settings
    (analytic shapes centred in the domain, as terrain.py builds them); None for flat or unknown."""
    if not os.path.exists(eff):
        return None
    cfg = {}
    for line in open(eff):
        key, _, val = line.partition(" = ")
        if key.startswith("TERRAIN_"):
            cfg[key] = val.strip().strip("'\"")
    kind = cfg.get("TERRAIN_TYPE", "flat")
    if kind == "flat":
        return None
    h0, s = float(cfg.get("TERRAIN_H_MAX", 0)), float(cfg.get("TERRAIN_SIGMA", 1))
    X, Y = np.meshgrid(np.arange(nx + 1) * dx - nx * dx / 2, np.arange(ny + 1) * dy - ny * dy / 2,
                       indexing="ij")
    if kind == "gaussian_hill":
        return h0 * np.exp(-(X**2 + Y**2) / (2 * s**2))
    if kind == "agnesi_ridge":
        return h0 * s**2 / (X**2 + s**2)
    print(f"TERRAIN_TYPE {kind!r}: not drawn (flat ground)")
    return None


def _fit_distance(focal, direction, box, view_angle: float, aspect: float, margin: float = 1.06) -> float:
    """Smallest camera distance along `direction` from `focal` at which all corners of the box
    [0, Lx] × [0, Ly] × [0, top] lie inside the view (vertical angle view_angle, width/height aspect)."""
    Lx, Ly, top = box
    d = np.asarray(direction, float) / np.linalg.norm(direction)       # focal → camera
    fwd = -d
    right = np.cross(fwd, (0.0, 0.0, 1.0)); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    tan_v = np.tan(np.radians(view_angle / 2)) / margin
    tan_h = tan_v * aspect
    corners = np.array([(x, y, z) for x in (0, Lx) for y in (0, Ly) for z in (0, top)]) - focal
    dist = 0.0
    for c in corners:                       # each corner: depth along fwd grows with dist
        a, h, v = c @ fwd, abs(c @ right), abs(c @ up)
        dist = max(dist, h / tan_h - a, v / tan_v - a)
    return dist


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
    ap.add_argument("--trange", type=float, nargs=2, metavar=("T0_S", "T1_S"),
                    help="only snapshots with T0 <= t <= T1 (s)")
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.run_dir, "snapshots", "snap_*.npz")))
    if not paths:
        raise SystemExit(f"No snapshots in {a.run_dir}/snapshots")
    if a.trange:
        t_of = lambda p: float(os.path.basename(p)[5:-4])          # snap_<t>.npz
        paths = [p for p in paths if a.trange[0] <= t_of(p) <= a.trange[1]]
    k_range = _cloud_levels(paths)
    dx = dy = 100.0
    open_xy = [False, False]                       # lateral BCs of the run (periodic unless "open")
    eff = os.path.join(a.run_dir, "effective_config.txt")
    if os.path.exists(eff):
        for line in open(eff):
            if line.startswith("BC_X = "):
                open_xy[0] = "open" in line
            elif line.startswith("BC_Y = "):
                open_xy[1] = "open" in line
            elif line.startswith("DX = "):
                dx = float(line.split("=")[1])
            elif line.startswith("DY = "):
                dy = float(line.split("=")[1])
    else:
        print("No effective_config.txt: assuming dx = dy = 100 m")
    xr = tuple(a.xrange) if a.xrange else None
    terrain = _terrain_nodes(eff, len(np.load(paths[0])["qc"]), np.load(paths[0])["qc"].shape[1], dx, dy)
    eff_vals = {}
    if os.path.exists(eff):
        for line in open(eff):
            key, _, val = line.partition(" = ")
            eff_vals[key] = val.strip()
    ref = _reference(a.run_dir, np.load(paths[0])["z"], eff_vals)

    if a.single is not None:
        times = np.array([float(np.load(p)["t"]) for p in paths])
        p = paths[int(np.argmin(np.abs(times - a.single)))]
        t0 = time.time()
        imageio.imwrite(a.out, render_frame(p, k_range, a.zexp, a.mode, a.smooth, dx, dy, xr,
                                              tuple(open_xy), terrain, ref))
        print(f"Saved {a.out} ({time.time() - t0:.1f} s)")
        return

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if any(open_xy):
        print(f"Open lateral boundaries (x {open_xy[0]}, y {open_xy[1]}): no wrap-around of shadows/smoothing")
    print(f"Rendering {len(paths)} frames ({a.mode}, smooth={a.smooth}, "
          f"levels {k_range[0]}–{k_range[1]}) → {a.out}", flush=True)
    t0 = time.time()
    jobs = [(p, k_range, a.zexp, a.mode, a.smooth, dx, dy, xr, tuple(open_xy), terrain, ref) for p in paths]
    if a.jobs > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            imgs = list(ex.map(_render_one, jobs))
    else:
        imgs = []
        for i, j in enumerate(jobs):
            imgs.append(_render_one(j))
            print(f"  frame {i+1}/{len(jobs)}", flush=True)
    if a.out.endswith(".mp4"):
        imageio.mimsave(a.out, imgs, fps=a.fps, codec="libx264", quality=8, macro_block_size=1,
                        pixelformat="yuv420p")
    else:
        imageio.mimsave(a.out, imgs, fps=a.fps, loop=0)
    print(f"Saved {a.out}: {len(imgs)} frames in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()

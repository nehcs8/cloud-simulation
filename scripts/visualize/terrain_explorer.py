"""Explore the Copernicus GLO-30 DEM of the Alps (data/dem/copernicus_glo30/).

Usage (from the repo root):
    ./venv/bin/python3 scripts/visualize/terrain_explorer.py --list
    ./venv/bin/python3 scripts/visualize/terrain_explorer.py matterhorn
    ./venv/bin/python3 scripts/visualize/terrain_explorer.py 46.55,8.00 --size 15
    ./venv/bin/python3 scripts/visualize/terrain_explorer.py chamonix --compare 100   # + model view at 100 m
    ./venv/bin/python3 scripts/visualize/terrain_explorer.py --overview               # 2-D map of all tiles
    ... --out outputs/plots/terrain_matterhorn.png                                   # off-screen PNG

Interactive window (pyvista): left drag rotates, right drag / wheel zooms, shift+drag pans,
'r' resets the camera, 'q' quits.

The crop is put on a local metric grid (east/north in m from the centre, equirectangular,
fine for crops up to ~100 km). --compare DX adds a second panel with the terrain block-averaged
to DX and drawn as flat steps -- roughly what a model grid of that spacing sees.
Note: GLO-30 is a surface model (forest canopy and buildings included, ~20-30 m on wooded slopes).
"""
import argparse
import glob
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEM_DIR = os.path.join(ROOT, 'data', 'dem', 'copernicus_glo30')
R_EARTH = 6371000.0

PLACES = {  # name: (lat, lon)
    'matterhorn': (45.976, 7.658), 'monte_rosa': (45.937, 7.867), 'mont_blanc': (45.833, 6.865),
    'chamonix': (45.924, 6.870), 'eiger': (46.577, 8.005), 'jungfrau': (46.537, 7.963),
    'zermatt': (46.020, 7.749), 'aletsch': (46.450, 8.050), 'st_moritz': (46.498, 9.838),
    'bernina': (46.382, 9.908), 'gotthard': (46.559, 8.561), 'grossglockner': (47.074, 12.694),
    'zugspitze': (47.421, 10.985), 'innsbruck': (47.269, 11.404), 'ortler': (46.509, 10.545),
    'dolomites': (46.500, 11.850), 'marmolada': (46.434, 11.851), 'tre_cime': (46.619, 12.302),
    'cortina': (46.540, 12.136), 'bolzano': (46.498, 11.354), 'lake_garda': (45.650, 10.700),
    'lake_como': (46.000, 9.250), 'aosta': (45.737, 7.320), 'gran_paradiso': (45.517, 7.267),
    'monviso': (44.667, 7.090), 'ecrins': (44.922, 6.360), 'grenoble': (45.188, 5.724),
    'triglav': (46.378, 13.837), 'dachstein': (47.475, 13.606), 'lugano': (46.004, 8.951),
}


def tile_name(lat, lon):
    ns, ew = ('N' if lat >= 0 else 'S'), ('E' if lon >= 0 else 'W')
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM.tif"


def load_crop(lat0, lon0, size_km, res_m):
    """Height field (ny, nx) on a local metric grid centred on (lat0, lon0); x east, y north."""
    import rasterio
    from rasterio.merge import merge
    from scipy.interpolate import RegularGridInterpolator

    half = 0.5 * size_km * 1000.0
    dlat = math.degrees(half / R_EARTH)
    dlon = math.degrees(half / (R_EARTH * math.cos(math.radians(lat0))))
    w, e, s, n = lon0 - dlon, lon0 + dlon, lat0 - dlat, lat0 + dlat
    paths = [os.path.join(DEM_DIR, tile_name(la, lo))
             for la in range(math.floor(s), math.floor(n) + 1)
             for lo in range(math.floor(w), math.floor(e) + 1)]
    have = [p for p in paths if os.path.exists(p)]
    if not have:
        sys.exit(f"No tiles for this area in {DEM_DIR} (sea, or outside the downloaded box).")
    if len(have) < len(paths):
        print(f"  {len(paths) - len(have)} tile(s) missing (sea / not downloaded) -> filled with 0 m")
    pad = 2.0 / 3600.0  # a couple of source pixels around the box for the interpolation
    srcs = [rasterio.open(p) for p in have]
    try:
        arr, tr = merge(srcs, bounds=(w - pad, s - pad, e + pad, n + pad), nodata=0.0)
    finally:
        for f in srcs:
            f.close()
    z = arr[0].astype(np.float64)
    z[~np.isfinite(z)] = 0.0
    rows, cols = z.shape
    lons = tr.c + tr.a * (np.arange(cols) + 0.5)
    lats = tr.f + tr.e * (np.arange(rows) + 0.5)          # descending
    lats, z = lats[::-1], z[::-1]                          # ascending for the interpolator
    interp = RegularGridInterpolator((lats, lons), z, bounds_error=False, fill_value=None)

    nx = int(round(size_km * 1000.0 / res_m))
    xc = (np.arange(nx) + 0.5) * res_m - half              # cell centres, m
    X, Y = np.meshgrid(xc, xc)
    lat_q = lat0 + np.degrees(Y / R_EARTH)
    lon_q = lon0 + np.degrees(X / (R_EARTH * math.cos(math.radians(lat0))))
    return xc, interp(np.stack([lat_q, lon_q], axis=-1))


def block_average(xc, z, dx_src, dx_model):
    k = max(1, int(round(dx_model / dx_src)))
    ny, nx = (z.shape[0] // k) * k, (z.shape[1] // k) * k
    zb = z[:ny, :nx].reshape(ny // k, k, nx // k, k).mean(axis=(1, 3))
    return xc[:nx].reshape(-1, k).mean(axis=1), zb, k * dx_src


def surface_mesh(xc, z, zexp):
    import pyvista as pv
    X, Y = np.meshgrid(xc, xc[:z.shape[0]])
    m = pv.StructuredGrid(X, Y, z * zexp)
    m['elevation [m]'] = z.ravel(order='F')
    return m


def step_mesh(xc, z, dx, zexp):
    """Flat-topped columns: each cell a horizontal quad at its mean height, plus vertical walls."""
    import pyvista as pv
    ny, nx = z.shape
    xe = np.concatenate([xc - dx / 2, [xc[-1] + dx / 2]])
    ye = xe[:ny + 1]
    pts, faces, elev = [], [], []

    def quad(p4, h):
        i = len(pts)
        pts.extend(p4)
        faces.extend([4, i, i + 1, i + 2, i + 3])
        elev.append(h)

    zz = z * zexp
    for j in range(ny):
        for i in range(nx):
            h = zz[j, i]
            quad([(xe[i], ye[j], h), (xe[i + 1], ye[j], h), (xe[i + 1], ye[j + 1], h),
                  (xe[i], ye[j + 1], h)], z[j, i])
            if i + 1 < nx:   # wall to the east neighbour
                h2 = zz[j, i + 1]
                quad([(xe[i + 1], ye[j], h), (xe[i + 1], ye[j + 1], h), (xe[i + 1], ye[j + 1], h2),
                      (xe[i + 1], ye[j], h2)], max(z[j, i], z[j, i + 1]))
            if j + 1 < ny:   # wall to the north neighbour
                h2 = zz[j + 1, i]
                quad([(xe[i], ye[j + 1], h), (xe[i + 1], ye[j + 1], h), (xe[i + 1], ye[j + 1], h2),
                      (xe[i], ye[j + 1], h2)], max(z[j, i], z[j + 1, i]))
    m = pv.PolyData(np.array(pts, dtype=float), np.array(faces))
    m.cell_data['elevation [m]'] = np.array(elev)
    return m


def show(meshes_titles, clim, zexp, out, title):
    import pyvista as pv
    n = len(meshes_titles)
    pl = pv.Plotter(shape=(1, n), off_screen=bool(out), window_size=(900 * n, 800))
    for k, (mesh, sub) in enumerate(meshes_titles):
        pl.subplot(0, k)
        pl.add_mesh(mesh, scalars='elevation [m]', cmap='gist_earth', clim=clim,
                    smooth_shading=isinstance(mesh, pv.StructuredGrid), specular=0.15,
                    scalar_bar_args=dict(title='elevation [m]', fmt='%.0f', position_x=0.25,
                                         position_y=0.03, width=0.5, height=0.06))
        pl.add_text(sub, font_size=10)
        pl.show_grid(xtitle='east [m]', ytitle='north [m]', ztitle=f'z × {zexp:g} [m]', font_size=8)
        pl.view_isometric()   # from the south-east, looking down ~35°
        pl.camera.zoom(1.2)
    if n > 1:
        pl.link_views()
    if out:
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        pl.screenshot(out)
        print(f"  wrote {out}")
    else:
        pl.show(title=title)


def overview(out):
    import matplotlib
    if out:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    import rasterio
    from rasterio.enums import Resampling
    paths = sorted(glob.glob(os.path.join(DEM_DIR, '*.tif')))
    if not paths:
        sys.exit(f"No tiles in {DEM_DIR}")
    dec = 60  # 3600 -> 60 px per degree (~1.5 km)
    W, S, E, N = 5, 43, 17, 48
    mosaic = np.zeros(((N - S) * dec, (E - W) * dec))   # missing tiles (open sea) stay 0
    for p in paths:
        with rasterio.open(p) as f:
            z = f.read(1, out_shape=(dec, dec), resampling=Resampling.average)
            b = f.bounds
        r0, c0 = int(round((N - b.top) * dec)), int(round((b.left - W) * dec))
        if 0 <= r0 < mosaic.shape[0] and 0 <= c0 < mosaic.shape[1]:
            mosaic[r0:r0 + dec, c0:c0 + dec] = np.where(np.isfinite(z), z, 0.0)
    fig, ax = plt.subplots(figsize=(15, 7.5))
    lat_c = 46.0
    ls = LightSource(azdeg=315, altdeg=40)   # shade once, so no seams between tiles
    rgb = ls.shade(mosaic, cmap=plt.cm.gist_earth, vert_exag=1.0, blend_mode='soft', vmin=0, vmax=4500,
                   dx=1852.0 / dec * 60 * math.cos(math.radians(lat_c)), dy=1852.0 / dec * 60)
    ax.imshow(rgb, extent=(W, E, S, N), origin='upper')
    for name, (la, lo) in PLACES.items():
        ax.plot(lo, la, 'r.', ms=4)
        ax.text(lo + 0.03, la + 0.03, name, fontsize=6, color='k')
    ax.set_xlim(5, 17); ax.set_ylim(43, 48)
    ax.set_aspect(1 / math.cos(math.radians(lat_c)))
    ax.set_xlabel('lon [°E]'); ax.set_ylabel('lat [°N]')
    ax.set_title(f'Copernicus GLO-30, {len(paths)} tiles (shown at ~1.5 km)')
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
        print(f"  wrote {out}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('where', nargs='?', help="place name (see --list) or 'lat,lon'")
    ap.add_argument('--size', type=float, default=12.0, help='crop side [km] (default 12)')
    ap.add_argument('--res', type=float, default=30.0, help='resampling of the crop [m] (default 30 = native)')
    ap.add_argument('--zexp', type=float, default=1.0, help='vertical exaggeration (default 1)')
    ap.add_argument('--compare', type=float, metavar='DX', help='add a panel block-averaged to DX metres')
    ap.add_argument('--out', help='write a PNG instead of opening a window')
    ap.add_argument('--list', action='store_true', help='list built-in place names')
    ap.add_argument('--overview', action='store_true', help='2-D shaded map of all tiles')
    a = ap.parse_args()

    if a.list:
        for k, (la, lo) in sorted(PLACES.items()):
            print(f"  {k:15s} {la:7.3f} N {lo:7.3f} E")
        return
    if a.overview:
        overview(a.out)
        return
    if not a.where:
        ap.error("give a place name or 'lat,lon' (or --list / --overview)")
    if a.where.lower() in PLACES:
        lat0, lon0 = PLACES[a.where.lower()]
    else:
        try:
            lat0, lon0 = (float(v) for v in a.where.split(','))
        except ValueError:
            ap.error(f"unknown place '{a.where}' (see --list) and not 'lat,lon'")

    if a.compare:   # make the model spacing a whole number of crop cells (100 m -> 25 m crop)
        a.res = a.compare / math.ceil(a.compare / a.res - 1e-9)
    print(f"  centre {lat0:.4f} N {lon0:.4f} E, {a.size:g} km, {a.res:g} m")
    xc, z = load_crop(lat0, lon0, a.size, a.res)
    gy, gx = np.gradient(z, a.res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    print(f"  elevation {z.min():.0f}-{z.max():.0f} m, relief {z.max() - z.min():.0f} m; "
          f"slope median {np.median(slope):.0f}°, 95th pct {np.percentile(slope, 95):.0f}°, max {slope.max():.0f}°")
    clim = (float(z.min()), float(z.max()))
    panels = [(surface_mesh(xc, z, a.zexp), f'GLO-30 at {a.res:g} m')]
    if a.compare:
        xb, zb, dxb = block_average(xc, z, a.res, a.compare)
        if zb.size > 250_000:
            sys.exit("--compare grid too large for the step view; use a larger DX or smaller --size")
        steps = np.abs(np.diff(zb, axis=1)).max(), np.abs(np.diff(zb, axis=0)).max()
        print(f"  {dxb:g} m view: {zb.shape[1]}×{zb.shape[0]} cells, largest step between neighbours "
              f"{max(steps):.0f} m, range {zb.min():.0f}-{zb.max():.0f} m")
        panels.append((step_mesh(xb, zb, dxb, a.zexp), f'block mean at {dxb:g} m (model view)'))
    show(panels, clim, a.zexp, a.out, f'{a.where} ({lat0:.3f}, {lon0:.3f})')


if __name__ == '__main__':
    main()

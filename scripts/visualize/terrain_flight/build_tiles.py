"""Build the level-of-detail tile pyramid for the terrain flight viewer.

    ./venv/bin/python3 scripts/visualize/terrain_flight/build_tiles.py [--jobs 8]

Reads the Copernicus GLO-30 tiles in data/dem/copernicus_glo30/ (5-17 E, 43-48 N) and writes
data/dem/glo30_lod/<L>/<row>/<col>.png plus meta.json and overview.png.

Sample (r, c) of level L sits at lon = W + ((c + 0.5) * 2^L - 0.5) / PPD, lat = N - (same with r);
level L has a sample spacing of 2^L arcsec (L0 = native 1" ~ 31 m N-S / 21 m E-W at 46 N).
A chunk is CHUNK = 225 cells (226 x 226 samples, the last row/column shared with the neighbour),
so an L0 chunk is 1/16 degree and an L4 chunk exactly 1 degree. Heights are stored as
Terrarium-encoded RGB (h = R*256 + G + B/256 - 32768 m), lossless to 1/256 m.
"""
import argparse
import glob
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
SRC = os.path.join(ROOT, 'data', 'dem', 'copernicus_glo30')
OUT = os.path.join(ROOT, 'data', 'dem', 'glo30_lod')
W, S, E, N = 5, 43, 17, 48          # degrees
PPD = 3600                          # samples per degree at L0
CHUNK = 225
NLEV = 7                            # L0 .. L6 (L6 chunk = 4 degrees)


def load_mosaic():
    import rasterio
    z = np.zeros(((N - S) * PPD, (E - W) * PPD), dtype=np.float32)   # missing tile = sea = 0
    # GLO-30 is point-registered: row 0 / col 0 of tile N<lat> E<lon> sit exactly on (lat+1, lon), so
    # mosaic sample (R, C) is at lat = N - R/PPD, lon = W + C/PPD. Place tiles by the name, not the bounds.
    for p in sorted(glob.glob(os.path.join(SRC, '*.tif'))):
        parts = os.path.basename(p).split('_')       # Copernicus_DSM_COG_10_N45_00_E007_00_DEM.tif
        lat, lon = int(parts[4][1:]), int(parts[6][1:])
        with rasterio.open(p) as f:
            a = f.read(1)
        if a.shape != (PPD, PPD):
            raise SystemExit(f"unexpected tile shape {a.shape} in {p}")
        r0, c0 = (N - 1 - lat) * PPD, (lon - W) * PPD
        z[r0:r0 + PPD, c0:c0 + PPD] = np.where(np.isfinite(a), a, 0.0)
    return z


def encode(h):
    v = np.clip(h + 32768.0, 0, 65535.99)
    r = np.floor(v / 256.0)
    g = np.floor(v - r * 256.0)
    b = np.floor((v - r * 256.0 - g) * 256.0)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


_Z = None


def _init(path, shape):
    global _Z
    _Z = np.load(path, mmap_mode='r') if shape is None else None


def write_row(args):
    from PIL import Image
    lev, row, ncol = args
    z = _Z
    nr, nc = z.shape
    os.makedirs(os.path.join(OUT, str(lev), str(row)), exist_ok=True)
    zmin, zmax = [], []
    for col in range(ncol):
        r0, c0 = row * CHUNK, col * CHUNK
        rr = np.minimum(np.arange(r0, r0 + CHUNK + 1), nr - 1)   # clamp at the outer edge
        cc = np.minimum(np.arange(c0, c0 + CHUNK + 1), nc - 1)
        h = np.asarray(z[np.ix_(rr, cc)], dtype=np.float32)
        Image.fromarray(encode(h), 'RGB').save(os.path.join(OUT, str(lev), str(row), f'{col}.png'),
                                              compress_level=6)
        zmin.append(float(h.min())); zmax.append(float(h.max()))
    return lev, row, zmin, zmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=8)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    print("reading tiles ...", flush=True)
    z = load_mosaic()
    meta = dict(west=W, south=S, east=E, north=N, ppd=PPD, chunk=CHUNK, levels=[],
                lat0=0.5 * (S + N), lon0=0.5 * (W + E), encoding='terrarium')
    tmp = os.path.join(OUT, '_level.npy')
    for lev in range(NLEV):
        if lev > 0:   # 2x2 mean (pad odd sizes by edge replication)
            if z.shape[0] % 2: z = np.vstack([z, z[-1:]])
            if z.shape[1] % 2: z = np.hstack([z, z[:, -1:]])
            z = 0.25 * (z[0::2, 0::2] + z[1::2, 0::2] + z[0::2, 1::2] + z[1::2, 1::2])
        np.save(tmp, z)
        nrow, ncol = math.ceil(z.shape[0] / CHUNK), math.ceil(z.shape[1] / CHUNK)
        print(f"L{lev}: {z.shape[1]}x{z.shape[0]} samples, {ncol}x{nrow} chunks", flush=True)
        zmin = [[0.0] * ncol for _ in range(nrow)]
        zmax = [[0.0] * ncol for _ in range(nrow)]
        with ProcessPoolExecutor(a.jobs, initializer=_init, initargs=(tmp, None)) as ex:
            for lv, row, mn, mx in ex.map(write_row, [(lev, r, ncol) for r in range(nrow)]):
                zmin[row], zmax[row] = [round(v, 1) for v in mn], [round(v, 1) for v in mx]
        meta['levels'].append(dict(level=lev, rows=nrow, cols=ncol, samples=[z.shape[1], z.shape[0]],
                                   zmin=zmin, zmax=zmax))
        if lev == 4:   # minimap: ~1.9 km per pixel, 360 x 150 ... upsampled by the browser
            ov = z
    os.remove(tmp)

    from PIL import Image
    from matplotlib import cm
    from matplotlib.colors import LightSource
    ls = LightSource(azdeg=315, altdeg=40)
    lat_c = meta['lat0']
    sp = 16 * 1852.0 / 60.0   # L4 spacing in m (N-S)
    rgb = ls.shade(ov, cmap=cm.gist_earth, vert_exag=1.0, blend_mode='soft', vmin=0, vmax=4500,
                   dx=sp * math.cos(math.radians(lat_c)), dy=sp)
    Image.fromarray((rgb[..., :3] * 255).astype(np.uint8)).save(os.path.join(OUT, 'overview.png'))
    with open(os.path.join(OUT, 'meta.json'), 'w') as f:
        json.dump(meta, f)
    print(f"done -> {OUT}")


if __name__ == '__main__':
    main()

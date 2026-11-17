"""Yearly direct-sun hours and sunless days for the whole Alpine DEM (5-17 E, 43-48 N).

    ./venv/bin/python3 scripts/sun_alps.py run      [--stride 3] [--year 2026]   # resumable, writes strips
    ./venv/bin/python3 scripts/sun_alps.py status   [--stride 3] [--year 2026]   # how far along, time left
    ./venv/bin/python3 scripts/sun_alps.py assemble [--stride 3] [--year 2026]   # GeoTIFF + overview PNG
    ./venv/bin/python3 scripts/sun_alps.py tiles    [--stride 3] [--year 2026]   # overlay for the flight viewer

`run` does assemble + tiles at the end. Long runs: start it outside VS Code's cgroup, e.g.
    systemd-run --user --unit=sun-alps --working-directory="$PWD" \
        ./venv/bin/python3 scripts/sun_alps.py run --stride 3
Progress: data/sun/alps_<res>m_<year>/progress.log (one line per strip with % done and ETA), or `status`.

Same method as `terrain_sun.py --map` (same numba kernel): horizon per point in 1-deg azimuth bins up to
--radius km (default 60), sun every 5 min, sun table per strip at the strip's latitude. Points are every
--stride-th DEM sample (stride 3 = 3" ~ 90 m N-S / 64 m E-W), each value is the point at the cell centre
(not a cell average). Sea (DEM <= 0.5 m) is skipped (-1). Points within --radius of the domain edge see a
truncated horizon (outside the data = no terrain).
Output dir data/sun/alps_<res>m_<year>/: strips/strip_NNN.npz, sun_alps_<res>m_<year>.tif (int16 bands:
hours/year, sunless days, ground m; nodata -1), viewer/ (overlay tiles), and outputs/plots/sunmap_alps_*.png.
"""
import argparse
import datetime as dt
import glob
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'visualize', 'terrain_flight'))
import terrain_sun as ts            # noqa: E402
import build_tiles as bt            # noqa: E402

W, S, E, N, PPD = bt.W, bt.S, bt.E, bt.N, bt.PPD
SEA = 0.5
NLEV = 7


def out_dir(a):
    res = round(a.stride * 30)
    return os.path.join(ROOT, 'data', 'sun', f'alps_{res}m_{a.year}')


def grid(stride):
    off = stride // 2
    rows = np.arange(off, (N - S) * PPD, stride)
    cols = np.arange(off, (E - W) * PPD, stride)
    return off, rows, cols


def fmt_dur(s):
    s = int(round(s))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def build_pyramid(z):
    """All levels (1", 2", ... 64") in one flat float32 array, as the kernel wants it."""
    shapes = [z.shape]
    for _ in range(NLEV - 1):
        r, c = shapes[-1]
        shapes.append(((r + 1) // 2, (c + 1) // 2))
    sizes = [r * c for r, c in shapes]
    flat = np.empty(sum(sizes), np.float32)
    off = np.cumsum([0] + sizes[:-1]).astype(np.int64)
    flat[:sizes[0]] = z.ravel()
    lv = z
    for L in range(1, NLEV):
        if lv.shape[0] % 2: lv = np.vstack([lv, lv[-1:]])
        if lv.shape[1] % 2: lv = np.hstack([lv, lv[:, -1:]])
        lv = 0.25 * (lv[0::2, 0::2] + lv[1::2, 0::2] + lv[0::2, 1::2] + lv[1::2, 1::2])
        flat[off[L]:off[L] + sizes[L]] = lv.ravel()
    nr = np.array([s[0] for s in shapes], np.int64)
    nc = np.array([s[1] for s in shapes], np.int64)
    return flat, off, nr, nc


def write_progress(od, info, line):
    with open(os.path.join(od, 'progress.log'), 'a') as f:
        f.write(line + '\n')
    tmp = os.path.join(od, 'progress.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(info, f, indent=1)
    os.replace(tmp, os.path.join(od, 'progress.json'))
    print(line, flush=True)


def cmd_run(a):
    od = out_dir(a)
    os.makedirs(os.path.join(od, 'strips'), exist_ok=True)
    off, rows, cols = grid(a.stride)
    nstrip = math.ceil(rows.size / a.strip_rows)
    print(f"output {od}\ngrid {cols.size} x {rows.size} points (stride {a.stride}), {nstrip} strips of "
          f"{a.strip_rows} rows, horizon radius {a.radius:g} km, year {a.year}", flush=True)
    t_load = time.time()
    print("loading DEM mosaic ...", flush=True)
    z = bt.load_mosaic()
    land = z[np.ix_(rows, cols)] > SEA
    land_per_strip = [int(land[k * a.strip_rows:(k + 1) * a.strip_rows].sum()) for k in range(nstrip)]
    land_total = sum(land_per_strip)
    del land
    # early-stop bound per output row: highest terrain within the radius in latitude (all longitudes)
    from scipy.ndimage import maximum_filter1d
    rowmax = z.max(axis=1)
    win = 2 * int(math.ceil(a.radius * 1000.0 / ts.DEG_M * PPD)) + 1
    zmax_rows = maximum_filter1d(rowmax, size=win, mode='nearest')[rows].astype(np.float64)
    flat, loff, nr, nc = build_pyramid(z)
    ground_all = z[np.ix_(rows, cols)].astype(np.float32)
    del z
    print(f"  ready in {time.time() - t_load:.0f} s; {land_total / 1e6:.1f} M land points "
          f"({100 * land_total / (rows.size * cols.size):.0f} % of the grid)", flush=True)

    kern = ts._map_kernel()
    m_row = ts.DEG_M / PPD
    curv = (1 - ts.K_REFR) / (2 * ts.R_EARTH)
    todo = [k for k in range(nstrip) if not os.path.exists(os.path.join(od, 'strips', f'strip_{k:03d}.npz'))]
    done_land = sum(land_per_strip[k] for k in range(nstrip) if k not in todo)
    t_session = time.time()
    recent = []          # (land points, seconds) of the last strips: ETA from the current rate, not the
                         # session mean (a busy machine for a while would skew it for the rest of the run)
    info = dict(stride=a.stride, year=a.year, radius_km=a.radius, nstrip=nstrip, land_total=land_total,
                started=dt.datetime.now().isoformat(timespec='seconds'), pid=os.getpid())
    write_progress(od, dict(info, done_land=done_land, strips_done=nstrip - len(todo), state='running'),
                   f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] start: {nstrip - len(todo)}/{nstrip} strips already done "
                   f"({100 * done_land / land_total:.1f} % of land points)")
    for k in todo:
        r = rows[k * a.strip_rows:(k + 1) * a.strip_rows]
        lat_c = N - r.mean() / PPD
        st = ts.sun_table(lat_c, 0.5 * (W + E), a.year)
        coslat = np.cos(np.radians(N - r / PPD))
        shape = (r.size, cols.size)
        out_n = np.zeros(shape, np.int64); out_s = np.zeros(shape, np.int64); out_h = np.zeros(shape)
        t0 = time.time()
        kern(flat, loff, nr, nc, r.astype(np.int64), cols.astype(np.int64), coslat, st['az_c'], m_row, curv,
             a.obs_height, zmax_rows[k * a.strip_rows:(k + 1) * a.strip_rows], SEA, a.radius * 1000.0,
             st['el_sorted'], st['bin_off'], st['day_off'], st['s_bin_d'], st['s_el_d'], out_n, out_s, out_h)
        secs = time.time() - t0
        hours = np.where(out_n < 0, -1, np.round(out_n * st['step_min'] / 60.0)).astype(np.int16)
        tmp = os.path.join(od, 'strips', f'strip_{k:03d}.tmp.npz')
        np.savez_compressed(tmp, hours=hours, sunless=out_s.astype(np.int16), seconds=secs,
                            ground=ground_all[k * a.strip_rows:(k + 1) * a.strip_rows].round().astype(np.int16),
                            astro_h=st['astro_h'], lat_c=lat_c)
        os.replace(tmp, os.path.join(od, 'strips', f'strip_{k:03d}.npz'))
        done_land += land_per_strip[k]
        el = time.time() - t_session
        recent = (recent + [(land_per_strip[k], secs)])[-8:]
        rate = sum(n for n, _ in recent) / max(sum(t for _, t in recent), 1e-9)   # land points/s, last 8 strips
        eta = (land_total - done_land) / rate if rate > 0 else float('nan')
        fin = dt.datetime.now() + dt.timedelta(seconds=eta) if np.isfinite(eta) else None
        strips_done = nstrip - len(todo) + todo.index(k) + 1
        left = f"left ~{fmt_dur(eta)} (finish ~{fin:%a %H:%M})" if fin else "left ?"
        line = (f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] strip {k + 1:3d}/{nstrip} ({lat_c:.2f} N, "
                f"{land_per_strip[k] / 1e3:.0f} k land pts) in {fmt_dur(secs)} | done {100 * done_land / land_total:5.1f} % "
                f"of land points, {strips_done}/{nstrip} strips | elapsed {fmt_dur(el)} | {left}")
        write_progress(od, dict(info, done_land=done_land, strips_done=strips_done, state='running',
                                rate_pts_s=rate, eta_s=eta, finish=fin.isoformat(timespec='seconds') if fin else None,
                                updated=dt.datetime.now().isoformat(timespec='seconds')), line)
    write_progress(od, dict(info, done_land=land_total, strips_done=nstrip, state='computed',
                            updated=dt.datetime.now().isoformat(timespec='seconds')),
                   f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] all strips done; assembling ...")
    del flat
    cmd_assemble(a)
    cmd_tiles(a)
    write_progress(od, dict(info, done_land=land_total, strips_done=nstrip, state='finished',
                            updated=dt.datetime.now().isoformat(timespec='seconds')),
                   f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] finished")


def cmd_status(a):
    od = out_dir(a)
    pj = os.path.join(od, 'progress.json')
    if not os.path.exists(pj):
        print(f"no run in {od}"); return
    p = json.load(open(pj))
    n_files = len(glob.glob(os.path.join(od, 'strips', 'strip_[0-9][0-9][0-9].npz')))
    alive = False
    try:
        os.kill(p['pid'], 0); alive = True
    except (OSError, KeyError):
        pass
    print(f"{od}\nstate: {p['state']}{'' if alive or p['state'] == 'finished' else ' (process not running -> rerun `run` to resume)'}")
    print(f"strips: {n_files}/{p['nstrip']}, land points: {100 * p['done_land'] / p['land_total']:.1f} % "
          f"of {p['land_total'] / 1e6:.1f} M")
    if p.get('eta_s') and p['state'] == 'running':
        upd = dt.datetime.fromisoformat(p['updated'])
        since = (dt.datetime.now() - upd).total_seconds()
        left = max(0.0, p['eta_s'] - since)
        fin = dt.datetime.now() + dt.timedelta(seconds=left)
        print(f"rate {p['rate_pts_s']:.0f} land points/s (last 8 strips), time left ~{fmt_dur(left)} "
              f"(finish ~{fin:%a %H:%M}), last strip finished {fmt_dur(since)} ago")
    print("\nlast log lines:")
    with open(os.path.join(od, 'progress.log')) as f:
        print(''.join(f.readlines()[-3:]), end='')


def load_strips(a):
    od = out_dir(a)
    off, rows, cols = grid(a.stride)
    nstrip = math.ceil(rows.size / a.strip_rows)
    hours = np.full((rows.size, cols.size), -1, np.int16)
    sunless = np.full_like(hours, -1); ground = np.zeros_like(hours)
    missing = 0
    for k in range(nstrip):
        p = os.path.join(od, 'strips', f'strip_{k:03d}.npz')
        if not os.path.exists(p):
            missing += 1; continue
        d = np.load(p)
        sl = slice(k * a.strip_rows, (k + 1) * a.strip_rows)
        hours[sl], sunless[sl], ground[sl] = d['hours'], d['sunless'], d['ground']
    return hours, sunless, ground, off, missing, nstrip


def cmd_assemble(a):
    import rasterio
    from rasterio.transform import from_origin
    od = out_dir(a)
    hours, sunless, ground, off, missing, nstrip = load_strips(a)
    if missing:
        print(f"WARNING: {missing}/{nstrip} strips missing (left as nodata)")
    res = round(a.stride * 30)
    step = a.stride / PPD
    tr = from_origin(W + (off - a.stride / 2) / PPD, N - (off - a.stride / 2) / PPD, step, step)
    tif = os.path.join(od, f'sun_alps_{res}m_{a.year}.tif')
    with rasterio.open(tif, 'w', driver='GTiff', width=hours.shape[1], height=hours.shape[0], count=3,
                       dtype='int16', crs='EPSG:4326', transform=tr, nodata=-1, compress='deflate',
                       predictor=2, tiled=True, blockxsize=512, blockysize=512) as f:
        for i, (arr, desc) in enumerate([(hours, 'direct sun hours per year'),
                                         (sunless, 'days without direct sun'), (ground, 'ground elevation m')], 1):
            f.write(arr, i); f.set_band_description(i, desc)
        f.update_tags(year=a.year, stride_arcsec=a.stride, radius_km=a.radius, source='Copernicus GLO-30',
                      method='scripts/sun_alps.py')
    print(f"wrote {os.path.relpath(tif, ROOT)} ({os.path.getsize(tif) / 1e6:.0f} MB)")
    overview_png(a, hours, sunless, ground)


def overview_png(a, hours, sunless, ground):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    k = max(1, round(8 * 3 / a.stride))      # ~720 m per pixel
    def down(x):
        r, c = (x.shape[0] // k) * k, (x.shape[1] // k) * k
        return x[:r, :c].reshape(r // k, k, c // k, k)
    g = down(ground.astype(np.float32)).mean(axis=(1, 3))
    hh = down(hours.astype(np.float32)); ss = down(sunless.astype(np.float32))
    valid = hh >= 0
    with np.errstate(invalid='ignore'):
        h = np.where(valid.sum(axis=(1, 3)) > 0, np.where(valid, hh, 0).sum(axis=(1, 3)) / valid.sum(axis=(1, 3)), np.nan)
        s = np.where(valid.sum(axis=(1, 3)) > 0, np.where(valid, ss, 0).sum(axis=(1, 3)) / valid.sum(axis=(1, 3)), np.nan)
    lat_c = 0.5 * (S + N)
    px = k * a.stride / PPD * ts.DEG_M
    shade = LightSource(azdeg=315, altdeg=45).hillshade(g, dx=px * math.cos(math.radians(lat_c)), dy=px)
    ext = (W, E, S, N)
    fig, axs = plt.subplots(2, 1, figsize=(16, 14.5))
    for ax, data, cmap, vmax, label in [(axs[0], h, 'magma', 4500, 'direct sun per year [h]'),
                                        (axs[1], s, 'Blues', 150, 'days without direct sun (cell mean)')]:
        ax.imshow(shade, extent=ext, cmap='gray', vmin=0, vmax=1)
        im = ax.imshow(np.ma.masked_invalid(data), extent=ext, cmap=cmap, vmin=0, vmax=vmax, alpha=0.8)
        ax.set_aspect(1 / math.cos(math.radians(lat_c)))
        ax.set_xlabel('lon [°E]'); ax.set_ylabel('lat [°N]')
        fig.colorbar(im, ax=ax, shrink=0.75, label=label)
    axs[0].set_title(f'Possible direct sun per year {a.year} (terrain shading, Copernicus GLO-30, '
                     f'{round(a.stride * 30)} m points, shown as ~{round(px / 10) * 10:.0f} m means)')
    axs[1].set_title('Days without any direct sun')
    fig.tight_layout()
    out = os.path.join(ROOT, 'outputs', 'plots', f'sunmap_alps_{round(a.stride * 30)}m_{a.year}.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {os.path.relpath(out, ROOT)}")


def cmd_tiles(a):
    """Overlay for the flight viewer: per 1-degree tile an RGB PNG (R = hours/4500*254, G = sunless days
    (capped 254), B = 255 land / 0 sea), plus a coarse whole-domain overview.png in the same encoding."""
    from PIL import Image
    od = out_dir(a)
    hours, sunless, ground, off, missing, nstrip = load_strips(a)
    vd = os.path.join(od, 'viewer')
    os.makedirs(vd, exist_ok=True)
    land = hours >= 0
    def enc(h, s, l):
        r = np.where(l, np.clip(np.round(h / 4500.0 * 254), 0, 254), 0)
        g = np.where(l, np.clip(s, 0, 254), 0)
        return np.stack([r, g, np.where(l, 255, 0)], axis=-1).astype(np.uint8)
    per = PPD // a.stride                         # raster cells per degree
    for r in range(N - S):
        for c in range(E - W):
            sl = (slice(r * per, (r + 1) * per), slice(c * per, (c + 1) * per))
            Image.fromarray(enc(hours[sl], sunless[sl], land[sl])).save(os.path.join(vd, f'{r}_{c}.png'), compress_level=6)
    k = 10
    def down(x):
        rr, cc = (x.shape[0] // k) * k, (x.shape[1] // k) * k
        return x[:rr, :cc].reshape(rr // k, k, cc // k, k)
    lw = down(land.astype(np.float32))
    cnt = lw.sum(axis=(1, 3))
    hm = (down(np.where(land, hours, 0).astype(np.float32))).sum(axis=(1, 3)) / np.maximum(cnt, 1)
    sm = (down(np.where(land, sunless, 0).astype(np.float32))).sum(axis=(1, 3)) / np.maximum(cnt, 1)
    ov = enc(hm, sm, cnt > 0)
    ov[..., 2] = np.round(255 * cnt / (k * k)).astype(np.uint8)
    Image.fromarray(ov).save(os.path.join(vd, 'overview.png'))
    meta = dict(stride=a.stride, offset=int(off), cells_per_degree=per, rows=N - S, cols=E - W,
                overview_factor=k, hours_scale=4500.0 / 254, year=a.year, res_m=round(a.stride * 30),
                west=W, north=N, ppd=PPD)
    with open(os.path.join(vd, 'meta.json'), 'w') as f:
        json.dump(meta, f)
    print(f"wrote viewer overlay to {os.path.relpath(vd, ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['run', 'status', 'assemble', 'tiles'])
    ap.add_argument('--stride', type=int, default=3, help='every n-th DEM sample (3 = ~90 m, 1 = native 30 m)')
    ap.add_argument('--year', type=int, default=dt.date.today().year)
    ap.add_argument('--radius', type=float, default=60.0, help='horizon search radius [km]')
    ap.add_argument('--obs-height', type=float, default=2.0)
    ap.add_argument('--strip-rows', type=int, default=0, help='output rows per strip (default: 1/96 of the domain)')
    a = ap.parse_args()
    if a.strip_rows <= 0:
        a.strip_rows = math.ceil(((N - S) * PPD // a.stride) / 96)
    {'run': cmd_run, 'status': cmd_status, 'assemble': cmd_assemble, 'tiles': cmd_tiles}[a.cmd](a)


if __name__ == '__main__':
    main()

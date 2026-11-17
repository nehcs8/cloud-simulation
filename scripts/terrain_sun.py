"""Direct-sun / terrain-shadow times for a location in the Alps, over a whole year.

    ./venv/bin/python3 scripts/terrain_sun.py viganella
    ./venv/bin/python3 scripts/terrain_sun.py 46.052,8.193 --name viganella --year 2027
    ./venv/bin/python3 scripts/terrain_sun.py zermatt --obs-height 2 --radius 100
    ./venv/bin/python3 scripts/terrain_sun.py viganella --map [--size 12] [--stride 1]   # heatmap, ~20 s
    ./venv/bin/python3 scripts/terrain_sun.py viganella --map --size 40 --from-tif data/sun/alps_90m_2026/sun_alps_90m_2026.tif

Map mode: the same horizon test for every DEM sample (~30 m) of a --size km square (horizon radius 60 km,
1-deg azimuth bins, rays stop once no peak in the area could raise the horizon further; distant terrain
from 2x2-averaged coarser levels), sun every 5 min. -> outputs/plots/sunmap_<name>_<size>km_<year>.png with
yearly direct-sun hours and days without direct sun, plus an .npz with the arrays.

Method
1. Horizon profile: from the point, rays every 0.5 deg of azimuth out to --radius km (default 100) through
   the Copernicus GLO-30 DEM (data/dem/copernicus_glo30/): native 1" within 6 km, 4" (~120 m) beyond.
   Horizon angle = max over the ray of atan((h - h0 - (1-k) d^2 / 2R) / d), with Earth curvature and
   terrestrial refraction k = 0.13. h0 = DEM + --obs-height.
2. Sun position every minute of the year (NOAA solar-calculator formulas, ~0.01 deg), apparent elevation
   with Bennett's refraction.
3. Sunlit when apparent sun elevation > horizon angle at the sun's azimuth (the slope the point sits on
   is part of its own horizon, so self-shading is included).

Outputs: outputs/plots/sun_<name>_<year>.png (sun-path diagram, daily sun hours, first/last sun) and
outputs/logs/sun_<name>_<year>.txt (monthly table, days without direct sun).

Caveats: GLO-30 is a surface model (trees and buildings included): a point inside a forest sees the
canopy as horizon -- pick a clearing / open ground, or raise --obs-height. Clouds are ignored: this is
astronomical sun minus terrain, not measured sunshine.
"""
import argparse
import datetime as dt
import glob
import math
import os
import sys
from zoneinfo import ZoneInfo

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'visualize'))
from terrain_explorer import PLACES as _EXPLORER_PLACES, DEM_DIR, tile_name   # noqa: E402

R_EARTH = 6371000.0
DEG_M = R_EARTH * math.pi / 180.0   # m per degree of latitude
K_REFR = 0.13
TZ = ZoneInfo('Europe/Zurich')   # CET/CEST, same rules across the Alpine countries
PLACES = dict(_EXPLORER_PLACES)
PLACES.update({
    'viganella': (46.0525, 8.1934),     # village without winter sun (mirror since 2006)
    'rattenberg': (47.4395, 11.8945),   # Tyrol, shaded by the Stadtberg in winter (street level, ~520 m)
    'milan': (45.4642, 9.1900),
})


# ---------------------------------------------------------------------------------------------- DEM
def load_dem(lat0, lon0, half_m, arcsec):
    """DEM block around the point at `arcsec` resolution -> (z[lat desc, lon asc], lat_top, lon_left, step_deg)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.merge import merge
    dlat = math.degrees(half_m / R_EARTH)
    dlon = math.degrees(half_m / (R_EARTH * math.cos(math.radians(lat0))))
    w, e, s, n = lon0 - dlon, lon0 + dlon, lat0 - dlat, lat0 + dlat
    paths = [os.path.join(DEM_DIR, tile_name(la, lo))
             for la in range(math.floor(s), math.floor(n) + 1)
             for lo in range(math.floor(w), math.floor(e) + 1)]
    have = [p for p in paths if os.path.exists(p)]
    step = arcsec / 3600.0
    srcs = [rasterio.open(p) for p in have]
    try:
        arr, tr = merge(srcs, bounds=(w, s, e, n), res=(step, step), nodata=np.nan,
                        resampling=Resampling.average if arcsec > 1 else Resampling.nearest)
    finally:
        for f in srcs:
            f.close()
    z = arr[0].astype(np.float64)
    # pixel centres: lat = top - (i + 0.5) step, lon = left + (j + 0.5) step
    return z, tr.f - 0.5 * step, tr.c + 0.5 * step, step, len(have) < len(paths)


def sample(dem, lat, lon):
    from scipy.ndimage import map_coordinates
    z, lat_top, lon_left, step, _ = dem
    i = (lat_top - lat) / step
    j = (lon - lon_left) / step
    return map_coordinates(z, [i.ravel(), j.ravel()], order=1, mode='constant', cval=np.nan).reshape(lat.shape)


def horizon_profile(lat0, lon0, radius_km, obs_height, daz=0.5):
    near_r, far_r = 6000.0, radius_km * 1000.0
    near = load_dem(lat0, lon0, near_r + 300, 1)
    far = load_dem(lat0, lon0, far_r + 2000, 4)
    h_ground = float(sample(near, np.array([lat0]), np.array([lon0]))[0])
    if not np.isfinite(h_ground):
        sys.exit("No DEM data at this point (outside data/dem/copernicus_glo30/).")
    h0 = h_ground + obs_height

    d = [30.0]
    while d[-1] < far_r:   # 15 m steps near the point, then growing by 0.4 % per step
        d.append(d[-1] + max(15.0, 0.004 * d[-1]))
    d = np.array(d)
    az = np.arange(0.0, 360.0, daz)
    A, D = np.meshgrid(np.radians(az), d, indexing='ij')            # (naz, nd)
    lat = lat0 + np.degrees(D * np.cos(A) / R_EARTH)
    lon = lon0 + np.degrees(D * np.sin(A) / (R_EARTH * np.cos(np.radians(lat))))
    h = np.where(D <= near_r, sample(near, lat, lon), np.nan)
    farm = D > near_r
    h[farm] = sample(far, lat[farm], lon[farm])
    missing = np.isnan(h)
    ang = np.degrees(np.arctan2(np.where(missing, -1e9, h) - h0 - (1 - K_REFR) * D ** 2 / (2 * R_EARTH), D))
    k = ang.argmax(axis=1)
    hor = ang[np.arange(len(az)), k]
    dist = d[k]
    sun_az = (az > 35) & (az < 325)        # azimuths the sun can reach at these latitudes
    no_data_frac = missing[sun_az].any(axis=1).mean()
    return dict(az=az, angle=hor, dist=dist, h_ground=h_ground, h0=h0, no_data_frac=no_data_frac,
                edge_tiles_missing=near[4] or far[4])


# ---------------------------------------------------------------------------------------------- sun
def sun_position(t_utc, lat, lon):
    """NOAA solar calculator. t_utc: datetime64 array. Returns (azimuth deg from N clockwise, apparent elevation deg)."""
    rad, deg = np.radians, np.degrees
    sec = t_utc.astype('datetime64[s]').astype(np.int64)
    jd = sec / 86400.0 + 2440587.5
    jc = (jd - 2451545.0) / 36525.0
    L = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360
    M = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    ecc = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    C = (np.sin(rad(M)) * (1.914602 - jc * (0.004817 + 0.000014 * jc)) + np.sin(rad(2 * M)) * (0.019993 - 0.000101 * jc)
         + np.sin(rad(3 * M)) * 0.000289)
    app_long = L + C - 0.00569 - 0.00478 * np.sin(rad(125.04 - 1934.136 * jc))
    obliq = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
    obliq_c = obliq + 0.00256 * np.cos(rad(125.04 - 1934.136 * jc))
    decl = np.arcsin(np.sin(rad(obliq_c)) * np.sin(rad(app_long)))
    y = np.tan(rad(obliq_c / 2)) ** 2
    eqt = 4 * deg(y * np.sin(2 * rad(L)) - 2 * ecc * np.sin(rad(M)) + 4 * ecc * y * np.sin(rad(M)) * np.cos(2 * rad(L))
                  - 0.5 * y * y * np.sin(4 * rad(L)) - 1.25 * ecc * ecc * np.sin(2 * rad(M)))
    minutes = (sec % 86400) / 60.0
    tst = (minutes + eqt + 4 * lon) % 1440
    ha = rad(tst / 4 - 180)
    phi = rad(lat)
    cosz = np.clip(np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha), -1, 1)
    zen = np.arccos(cosz)
    az = deg(np.arctan2(np.sin(ha), np.cos(ha) * np.sin(phi) - np.tan(decl) * np.cos(phi))) + 180.0
    el = 90 - deg(zen)
    refr = np.where(el > -1, 1.02 / np.tan(rad(el + 10.3 / (el + 5.11))) / 60.0, 0.0)   # Saemundsson, deg
    return az % 360, el + refr


# ---------------------------------------------------------------------------------------------- year
def year_analysis(hp, lat, lon, year, step_min=1):
    t0 = np.datetime64(f'{year}-01-01T00:00') - np.timedelta64(2, 'h')   # local midnight CET-ish margin
    n_days = (np.datetime64(f'{year + 1}-01-01') - np.datetime64(f'{year}-01-01')).astype(int)
    t = t0 + np.arange(0, (n_days + 1) * 1440, step_min).astype('timedelta64[m]')
    az, el = sun_position(t, lat, lon)
    hz = np.interp(az, hp['az'], hp['angle'], period=360)
    up = el > 0.0          # above the astronomical (flat, sea-level) horizon
    lit = el > hz          # above the terrain
    # local time: CET (+60 min), CEST (+120 min) from 01:00 UTC on the last Sunday of March to that of October
    def last_sunday(month):
        d = dt.date(year, month, 31)
        return np.datetime64(f'{d - dt.timedelta(days=(d.weekday() + 1) % 7)}T01:00')
    off = np.where((t >= last_sunday(3)) & (t < last_sunday(10)), 120, 60)
    local_min = (t - np.datetime64(f'{year}-01-01T00:00')).astype('timedelta64[m]').astype(np.int64) + off
    lday = local_min // 1440
    lmin = local_min % 1440
    days = []
    for d in range(n_days):
        m = lday == d
        tl, tu, mm = lit[m], up[m], lmin[m]
        first = mm[tl][0] if tl.any() else None
        last = mm[tl][-1] + step_min if tl.any() else None
        rise = mm[tu][0] if tu.any() else None
        sset = mm[tu][-1] + step_min if tu.any() else None
        # number of separate sunlit intervals (sun hiding behind a peak and coming back)
        nint = int(np.count_nonzero(np.diff(tl.astype(int)) == 1) + (1 if tl.size and tl[0] else 0))
        days.append(dict(date=dt.date(year, 1, 1) + dt.timedelta(days=d), sun_h=tl.sum() * step_min / 60,
                         astro_h=tu.sum() * step_min / 60, first=first, last=last, rise=rise, set=sset,
                         intervals=nint))
    return days


def fmt_min(m):
    return '  –  ' if m is None else f'{int(m) // 60:02d}:{int(m) % 60:02d}'


def no_sun_periods(days):
    per, cur = [], None
    for d in days:
        if d['sun_h'] == 0:
            cur = [d['date'], d['date']] if cur is None else [cur[0], d['date']]
        elif cur is not None:
            per.append(cur); cur = None
    if cur is not None:
        per.append(cur)
    # a period spanning New Year is split in two; join first and last
    if len(per) > 1 and per[0][0] == days[0]['date'] and per[-1][1] == days[-1]['date']:
        per = [[per[-1][0], per[0][1]]] + per[1:-1]
    return per


def report(name, lat, lon, year, hp, days, out_txt):
    L = []
    L.append(f"Direct sun at {name} ({lat:.4f} N, {lon:.4f} E), {year}, times {TZ.key} (CET/CEST)")
    L.append(f"ground {hp['h_ground']:.0f} m (GLO-30 surface model), observer {hp['h0']:.0f} m; "
             f"horizon out to the DEM edge, curvature + refraction k={K_REFR}")
    if hp['no_data_frac'] > 0:
        L.append(f"WARNING: {100 * hp['no_data_frac']:.0f} % of the horizon rays towards the sun's azimuths leave the DEM "
                 f"(those parts of the horizon are only as high as the terrain inside it)")
    i_max = int(np.argmax(hp['angle']))
    L.append(f"highest horizon {hp['angle'][i_max]:.1f}° at azimuth {hp['az'][i_max]:.0f}° "
             f"({hp['dist'][i_max] / 1000:.1f} km away); due south (180°) {np.interp(180, hp['az'], hp['angle']):.1f}°")
    L.append("")
    L.append("date        sunrise  first sun  last sun  sunset   sun h   day h  lost  pieces")
    for d in days:
        if d['date'].day in (1, 21) or d['date'] == days[-1]['date']:
            L.append(f"{d['date']}  {fmt_min(d['rise'])}    {fmt_min(d['first'])}     {fmt_min(d['last'])}    "
                     f"{fmt_min(d['set'])}   {d['sun_h']:5.2f}  {d['astro_h']:6.2f}  {100 * (1 - d['sun_h'] / max(d['astro_h'], 1e-9)):3.0f} %"
                     f"  {d['intervals']}")
    L.append("")
    tot, ast = sum(d['sun_h'] for d in days), sum(d['astro_h'] for d in days)
    L.append(f"year: {tot:.0f} h of possible direct sun out of {ast:.0f} h above a flat horizon "
             f"({100 * tot / ast:.0f} %)")
    nz = [d for d in days if d['sun_h'] == 0]
    L.append(f"days without any direct sun: {len(nz)}")
    for a, b in no_sun_periods(days):
        L.append(f"   {a:%d %b} – {b:%d %b}  ({(b - a).days + 1 if b >= a else (b - a).days + 1 + len(days)} days)")
    mins = sorted((d for d in days if d['sun_h'] > 0), key=lambda d: d['sun_h'])[:1]
    if mins:
        L.append(f"shortest sunny day: {mins[0]['date']} with {60 * mins[0]['sun_h']:.0f} min")
    txt = '\n'.join(L)
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, 'w') as f:
        f.write(txt + '\n')
    return txt


def plot(name, lat, lon, year, hp, days, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[0, :])
    # sun-path diagram
    az = hp['az']
    ax.fill_between(np.append(az, 360), -5, np.append(hp['angle'], hp['angle'][0]), color='#8c7b6b', alpha=0.85,
                    lw=0, label='terrain horizon', zorder=3)
    cmap = plt.cm.coolwarm
    for mo in range(1, 13):
        d0 = np.datetime64(f'{year}-{mo:02d}-21T00:00')
        t = d0 - np.timedelta64(2, 'h') + np.arange(0, 28 * 60, 2).astype('timedelta64[m]')
        sa, se = sun_position(t, lat, lon)
        keep = se > -2
        col = cmap(1 - abs(mo - 6.5) / 5.5)
        lw = 2.4 if mo in (6, 12) else 1.0
        sa_k, se_k = sa[keep], se[keep]
        ax.plot(sa_k, se_k, '-', color=col, lw=lw, zorder=2,
                label=dt.date(year, mo, 21).strftime('%d %b') if mo in (3, 6, 9, 12) else None)
        if mo in (6, 12, 3):
            for hh in range(4, 22):
                local = dt.datetime(year, mo, 21, hh, tzinfo=TZ)
                tu = np.array([np.datetime64(local.astimezone(dt.timezone.utc).replace(tzinfo=None), 'm')])
                ha_, he_ = sun_position(tu, lat, lon)
                if he_[0] > -1:
                    ax.plot(ha_, he_, 'o', color=col, ms=3.5, zorder=4)
                    ax.annotate(f'{hh}h', (ha_[0], he_[0]), textcoords='offset points', xytext=(0, 5), fontsize=6,
                                ha='center', color=col, zorder=5)
    ax.set_xlim(30, 330); ax.set_ylim(-2, 72)
    ax.set_xticks(range(30, 331, 30))
    ax.set_xticklabels(['30', '60', 'E 90', '120', '150', 'S 180', '210', '240', 'W 270', '300', '330'])
    ax.set_xlabel('azimuth [°]'); ax.set_ylabel('elevation [°]')
    ax.set_title(f'{name}: sun paths on the 21st of each month (local-time markers on 21 Mar / Jun / Dec) and the terrain horizon')
    ax.grid(alpha=0.3); ax.legend(loc='upper right', fontsize=8)

    dates = [np.datetime64(d['date']) for d in days]
    ax2 = fig.add_subplot(gs[1, 0])
    ah = np.array([d['astro_h'] for d in days]); sh = np.array([d['sun_h'] for d in days])
    ax2.fill_between(dates, sh, ah, color='#8c7b6b', alpha=0.5, label='lost to terrain')
    ax2.plot(dates, ah, color='0.4', lw=1, label='sun above flat horizon')
    ax2.plot(dates, sh, color='#e8a317', lw=1.8, label='direct sun at the point')
    ax2.set_ylabel('hours per day'); ax2.set_ylim(0, 16.5); ax2.grid(alpha=0.3); ax2.legend(fontsize=8, loc='upper left')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b')); ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.set_title('daily direct-sun hours')

    ax3 = fig.add_subplot(gs[1, 1])
    def series(k):
        return np.array([np.nan if d[k] is None else d[k] / 60 for d in days])
    ax3.fill_between(dates, series('first'), series('last'), color='#ffd36b', alpha=0.6, label='sun on the point')
    ax3.plot(dates, series('rise'), color='0.4', lw=1, ls='--', label='sunrise / sunset (flat)')
    ax3.plot(dates, series('set'), color='0.4', lw=1, ls='--')
    ax3.plot(dates, series('first'), color='#c77d00', lw=1.2, label='first / last direct sun')
    ax3.plot(dates, series('last'), color='#c77d00', lw=1.2)
    ax3.set_ylim(4, 22); ax3.set_yticks(range(4, 23, 2)); ax3.set_ylabel(f'local time ({TZ.key})')
    ax3.grid(alpha=0.3); ax3.legend(fontsize=8, loc='upper left')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b')); ax3.xaxis.set_major_locator(mdates.MonthLocator())
    ax3.set_title('first and last direct sun (gaps between them not shown)')
    tot, ast = sh.sum(), ah.sum()
    fig.suptitle(f'{name} ({lat:.4f} N, {lon:.4f} E, ground {hp["h_ground"]:.0f} m) — {year}: '
                 f'{tot:.0f} of {ast:.0f} h possible direct sun ({100 * tot / ast:.0f} %), '
                 f'{int((sh == 0).sum())} days without sun', fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- map mode
def _map_kernel():
    import numba
    from numba import prange

    @numba.njit(parallel=True, cache=True, fastmath=True)
    def kern(flat, off, nr, nc, i0s, j0s, coslat, az_c, m_row, curv, obs, zmax_row, skip_below, radius,
             el_sorted, bin_off, day_off, s_bin, s_el, out_n, out_sunless, out_h):
        # zmax_row[ii]: highest terrain within the horizon radius of row ii (early-stop bound);
        # points with ground <= skip_below (sea) get -1 in out_n / out_sunless
        nb = az_c.size
        ca = np.cos(np.radians(az_c))
        sa = np.sin(np.radians(az_c))
        nlev = off.size
        nc0 = nc[0]
        for ii in prange(i0s.size):
            i0 = i0s[ii]
            zmax = zmax_row[ii]
            hz = np.empty(nb)
            for jj in range(j0s.size):
                j0 = j0s[jj]
                hg = flat[i0 * nc0 + j0]
                h0 = hg + obs
                out_h[ii, jj] = hg
                if hg <= skip_below:
                    out_n[ii, jj] = -1
                    out_sunless[ii, jj] = -1
                    continue
                for b in range(nb):
                    di = -ca[b] / m_row                    # L0 rows per metre (rows run south)
                    dj = sa[b] / (m_row * coslat[ii])      # L0 cols per metre
                    best = -1e9
                    d = 20.0
                    while d < radius:
                        step = 15.0 if d < 1500.0 else 0.012 * d
                        L = 0
                        while L + 1 < nlev and 30.0 * 2.0 ** (L + 1) <= step:
                            L += 1
                        f = 2.0 ** L
                        xi = (i0 + di * d - 0.5 * (f - 1.0)) / f
                        xj = (j0 + dj * d - 0.5 * (f - 1.0)) / f
                        ia = int(np.floor(xi)); ja = int(np.floor(xj))
                        ncl = nc[L]
                        if ia < 0 or ja < 0 or ia + 1 >= nr[L] or ja + 1 >= ncl:
                            break
                        fi = xi - ia; fj = xj - ja
                        p = off[L] + ia * ncl + ja
                        h = ((flat[p] * (1 - fj) + flat[p + 1] * fj) * (1 - fi)
                             + (flat[p + ncl] * (1 - fj) + flat[p + ncl + 1] * fj) * fi)
                        drop = curv * d * d
                        t = (h - h0 - drop) / d
                        if t > best:
                            best = t
                        if (zmax - h0 - drop) / d <= best:   # nothing further out can be higher
                            break
                        d += step
                    hz[b] = np.degrees(np.arctan(best))
                # sun samples (5 min each) above the horizon
                n = 0
                for b in range(nb):
                    lo = bin_off[b]; hi = bin_off[b + 1]
                    if hi > lo:
                        n += hi - lo - np.searchsorted(el_sorted[lo:hi], hz[b], side='right')
                out_n[ii, jj] = n
                # days without any direct sun (each day's samples sorted by elevation, highest first)
                cnt = 0
                for dd in range(day_off.size - 1):
                    lit = False
                    for k in range(day_off[dd], day_off[dd + 1]):
                        if s_el[k] > hz[s_bin[k]]:
                            lit = True
                            break
                    if not lit:
                        cnt += 1
                out_sunless[ii, jj] = cnt
    return kern


def sun_table(lat, lon, year, step_min=5):
    """A year of sun positions (every step_min) arranged for the map kernel: binned by 1-deg azimuth with
    sorted elevations, and per local day (CET) sorted highest first."""
    t = np.datetime64(f'{year}-01-01T00:00') - np.timedelta64(1, 'h') \
        + np.arange(0, 366 * 1440, step_min).astype('timedelta64[m]')
    t = t[t < np.datetime64(f'{year + 1}-01-01T00:00') - np.timedelta64(1, 'h')]
    az, el = sun_position(t, lat, lon)
    astro_h = (el > 0).sum() * step_min / 60.0
    keep = el > -3.0
    t, az, el = t[keep], az[keep], el[keep]
    az_min = math.floor(az.min()); az_max = math.ceil(az.max())
    az_c = np.arange(az_min, az_max) + 0.5
    s_bin = np.clip((az - az_min).astype(np.int64), 0, az_c.size - 1)
    order = np.lexsort((el, s_bin))
    el_sorted = el[order]
    bin_off = np.searchsorted(s_bin[order], np.arange(az_c.size + 1)).astype(np.int64)
    lday = ((t + np.timedelta64(60, 'm')) - np.datetime64(f'{year}-01-01T00:00')).astype('timedelta64[D]').astype(np.int64)
    order = np.lexsort((-el, lday))          # per day, highest sun first
    day_off = np.searchsorted(lday[order], np.arange(lday.max() + 2)).astype(np.int64)
    return dict(astro_h=astro_h, az_c=az_c, el_sorted=el_sorted, bin_off=bin_off, day_off=day_off,
                s_bin_d=s_bin[order], s_el_d=el[order], step_min=step_min)


def sun_map(lat0, lon0, size_km, radius_km, year, obs_height, stride, step_min=5):
    import time
    half = 0.5 * size_km * 1000.0
    print(f"loading DEM ({size_km:g} km map + {radius_km:g} km horizon radius) ...", flush=True)
    z, lat_top, lon_left, step, _ = load_dem(lat0, lon0, half + radius_km * 1000.0 + 3000.0, 1)
    nodata = ~np.isfinite(z)
    z = np.where(nodata, -1000.0, z).astype(np.float64)
    levels = []
    lv = z
    for _ in range(7):   # 1", 2", ... 64"
        levels.append(np.ascontiguousarray(lv))
        if lv.shape[0] % 2: lv = np.vstack([lv, lv[-1:]])
        if lv.shape[1] % 2: lv = np.hstack([lv, lv[:, -1:]])
        lv = 0.25 * (lv[0::2, 0::2] + lv[1::2, 0::2] + lv[0::2, 1::2] + lv[1::2, 1::2])
    lats = lat_top - np.arange(z.shape[0]) * step
    lons = lon_left + np.arange(z.shape[1]) * step
    dlat = math.degrees(half / R_EARTH)
    dlon = math.degrees(half / (R_EARTH * math.cos(math.radians(lat0))))
    i0s = np.nonzero((lats <= lat0 + dlat) & (lats >= lat0 - dlat))[0][::stride]
    j0s = np.nonzero((lons >= lon0 - dlon) & (lons <= lon0 + dlon))[0][::stride]
    if nodata[np.ix_(i0s, j0s)].any():
        print("  WARNING: part of the map has no DEM data (outside the downloaded tiles)")

    st = sun_table(lat0, lon0, year, step_min)
    astro_h, az_c = st['astro_h'], st['az_c']
    az_min, az_max = int(az_c[0] - 0.5), int(az_c[-1] + 0.5)

    m_row = step * DEG_M
    coslat = np.cos(np.radians(lats[i0s]))
    out_n = np.zeros((i0s.size, j0s.size), np.int64)
    out_sunless = np.zeros((i0s.size, j0s.size), np.int64)
    out_h = np.zeros((i0s.size, j0s.size))
    kern = _map_kernel()
    print(f"computing {i0s.size} x {j0s.size} = {i0s.size * j0s.size} points x {az_c.size} azimuths "
          f"({az_min}-{az_max}°) ...", flush=True)
    t0 = time.time()
    flat = np.concatenate([l.ravel() for l in levels])
    off = np.cumsum([0] + [l.size for l in levels[:-1]]).astype(np.int64)
    nr = np.array([l.shape[0] for l in levels], np.int64)
    nc = np.array([l.shape[1] for l in levels], np.int64)
    kern(flat, off, nr, nc, i0s.astype(np.int64), j0s.astype(np.int64), coslat, az_c, m_row,
         (1 - K_REFR) / (2 * R_EARTH), obs_height, np.full(i0s.size, float(z.max())), -1e30, radius_km * 1000.0,
         st['el_sorted'], st['bin_off'], st['day_off'], st['s_bin_d'], st['s_el_d'],
         out_n, out_sunless, out_h)
    print(f"  done in {time.time() - t0:.0f} s", flush=True)
    return dict(lat=lats[i0s], lon=lons[j0s], ground=out_h, hours=out_n * step_min / 60.0,
                sunless=out_sunless, astro_h=astro_h, lat0=lat0, lon0=lon0, year=year)


def map_from_tif(tif, lat0, lon0, size_km):
    """Crop a sun_alps.py GeoTIFF to a size_km square around (lat0, lon0) -> same dict as sun_map()."""
    import rasterio
    from rasterio.windows import from_bounds
    half = 0.5 * size_km * 1000.0
    dlat = math.degrees(half / R_EARTH)
    dlon = math.degrees(half / (R_EARTH * math.cos(math.radians(lat0))))
    with rasterio.open(tif) as f:
        win = from_bounds(lon0 - dlon, lat0 - dlat, lon0 + dlon, lat0 + dlat, f.transform).round_offsets().round_lengths()
        hours, sunless, ground = (f.read(i, window=win).astype(np.float64) for i in (1, 2, 3))
        tr = f.window_transform(win)
        year = int(f.tags().get('year', dt.date.today().year))
    lons = tr.c + tr.a * (np.arange(hours.shape[1]) + 0.5)
    lats = tr.f + tr.e * (np.arange(hours.shape[0]) + 0.5)
    sea = hours < 0
    hours[sea] = np.nan; sunless[sea] = np.nan
    return dict(lat=lats, lon=lons, ground=ground, hours=hours, sunless=sunless,
                astro_h=sun_table(lat0, lon0, year)['astro_h'], lat0=lat0, lon0=lon0, year=year)


def plot_map(name, m, out_png, mark=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    kx = DEG_M * math.cos(math.radians(m['lat0']))
    x = (m['lon'] - m['lon0']) * kx / 1000.0
    y = (m['lat'] - m['lat0']) * DEG_M / 1000.0
    ext = (x[0], x[-1], y[-1], y[0])
    g = m['ground']
    dx = (x[1] - x[0]) * 1000.0; dy = (y[0] - y[1]) * 1000.0
    shade = LightSource(azdeg=315, altdeg=45).hillshade(g, dx=dx, dy=dy)
    fig, axs = plt.subplots(1, 2, figsize=(17, 8.2))
    panels = [(m['hours'], 'magma', 0, m['astro_h'], 'direct sun per year [h]',
               f"hours of possible direct sun in {m['year']} (flat horizon: {m['astro_h']:.0f} h)"),
              (m['sunless'], 'Blues', 0, max(30, int(np.nanpercentile(m['sunless'], 99.5))), 'days',
               'days without any direct sun')]
    for ax, (data, cmap, vmin, vmax, cl, title) in zip(axs, panels):
        ax.imshow(shade, extent=ext, cmap='gray', vmin=0, vmax=1)
        im = ax.imshow(data, extent=ext, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.78)
        ax.contour(x, y, g, levels=np.arange(0, 5000, 500), colors='k', linewidths=0.35, alpha=0.5)
        if mark is not None:
            mx = (mark[1] - m['lon0']) * kx / 1000.0; my = (mark[0] - m['lat0']) * DEG_M / 1000.0
            ax.plot(mx, my, 'o', mfc='none', mec='cyan', ms=9, mew=1.5)
        ax.set_xlabel('east [km]'); ax.set_ylabel('north [km]'); ax.set_title(title)
        ax.set_aspect('equal')
        fig.colorbar(im, ax=ax, shrink=0.8, label=cl)
    fig.suptitle(f"{name} ({m['lat0']:.4f} N, {m['lon0']:.4f} E) — terrain shading from GLO-30, "
                 f"contours every 500 m", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('where', help="place name (see terrain_explorer.py --list, plus viganella, rattenberg, milan) or 'lat,lon'")
    ap.add_argument('--name', help='label for titles and file names')
    ap.add_argument('--year', type=int, default=dt.date.today().year)
    ap.add_argument('--radius', type=float, default=100.0, help='horizon search radius [km] (default 100)')
    ap.add_argument('--obs-height', type=float, default=2.0, help='eye height above the DEM surface [m] (default 2)')
    ap.add_argument('--map', action='store_true', help='heatmap of yearly sun hours around the point instead')
    ap.add_argument('--size', type=float, default=12.0, help='map side [km] (default 12)')
    ap.add_argument('--stride', type=int, default=1, help='map: use every n-th DEM sample (1 = ~30 m)')
    ap.add_argument('--from-tif', metavar='TIF', help='map: crop a precomputed sun_alps.py GeoTIFF instead of computing')
    a = ap.parse_args()
    if a.where.lower() in PLACES:
        lat, lon = PLACES[a.where.lower()]
        name = a.name or a.where.lower()
    else:
        try:
            lat, lon = (float(v) for v in a.where.split(','))
        except ValueError:
            ap.error(f"unknown place '{a.where}' and not 'lat,lon'")
        name = a.name or f'{lat:.4f}N_{lon:.4f}E'
    slug = name.replace(' ', '_').lower()
    if a.map:
        if a.from_tif:
            m = map_from_tif(a.from_tif, lat, lon, a.size)
            out_png = os.path.join(ROOT, 'outputs', 'plots',
                                   f'sunmap_{slug}_{a.size:g}km_{m["year"]}_from_{os.path.basename(a.from_tif)[:-4]}.png')
        else:
            radius = min(a.radius, 60.0)
            m = sun_map(lat, lon, a.size, radius, a.year, a.obs_height, a.stride)
            out_png = os.path.join(ROOT, 'outputs', 'plots', f'sunmap_{slug}_{a.size:g}km_{a.year}.png')
            np.savez_compressed(out_png[:-4] + '.npz', **m)
        plot_map(name, m, out_png, mark=(lat, lon))
        h = m['hours']
        print(f"sun hours per year: min {np.nanmin(h):.0f}, median {np.nanmedian(h):.0f}, max {np.nanmax(h):.0f} "
              f"(flat horizon {m['astro_h']:.0f}); points with sunless days: "
              f"{100 * np.nanmean(m['sunless'] > 0):.0f} %, max {np.nanmax(m['sunless']):.0f} days")
        print(f"wrote {os.path.relpath(out_png, ROOT)}" + ("" if a.from_tif else " (+ .npz with the arrays)"))
        return
    hp = horizon_profile(lat, lon, a.radius, a.obs_height)
    days = year_analysis(hp, lat, lon, a.year)
    out_txt = os.path.join(ROOT, 'outputs', 'logs', f'sun_{slug}_{a.year}.txt')
    out_png = os.path.join(ROOT, 'outputs', 'plots', f'sun_{slug}_{a.year}.png')
    print(report(name, lat, lon, a.year, hp, days, out_txt))
    plot(name, lat, lon, a.year, hp, days, out_png)
    print(f"\nwrote {os.path.relpath(out_txt, ROOT)} and {os.path.relpath(out_png, ROOT)}")


if __name__ == '__main__':
    main()

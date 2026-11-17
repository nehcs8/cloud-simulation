"""
Layered day ("layered_day"): boundary-layer cumulus under a mid-level altocumulus layer and upper
cirrus layer, in veering shear — the idealised step before a real multi-layer WegenerNet day.

Sounding (z = height above surface), tabulated in cfg.LD_THETA / cfg.LD_RH and interpolated linearly:
  0–900 m        mixed layer: θ 299 K, q_v constant with RH 0.80 at 900 m (LCL ≈ 1.1 km)
  900–3800 m     4.2 → 3.5 K/km, RH 0.80 → 0.60 (Cu layer, conditionally unstable)
  3800–4700 m    moist layer A (altocumulus): RH 0.92, 1.8 K/km (unstable once saturated), T ≈ −2…−7 °C
  4700–4900 m    stable cap, +1.5 K
  4900–8300 m    dry slot, RH 0.40, 4 K/km
  8500–10000 m   moist layer C (cirrus): RH 0.92, T ≈ −33…−45 °C
  11000 m        tropopause, 15 K/km above (RH 0.20)
RH is relative to the model's own saturation: over the liquid/ice mix of ICE_SCHEME ("graupel" here), so
layer C is 92 % over ice. Wind: u_g(z) = 2 → 20 m/s at 10 km, V_GEO 3 m/s (SW at the surface, W aloft).

Without large-scale ascent or radiation (LS_FORCING None, the "baseline" variant) the moist layers are
only lifted by gravity waves and convection; with LS_FORCING (forcing.apply_large_scale_forcing, the
default "trough" variant) a trough's ascent saturates them over the afternoon.

Grid: 128² at 150 m (19.2 km), 100 m levels to 11 km, 12 stretched levels to 14 km, sponge from 12 km.
Open lateral boundaries with a 64² precursor (it gets the same large-scale forcing, so the layers enter
from the edges too). Uniform SHF 150 / LHF 250 W/m², ramped over 30 min.
"""

import numpy as np
import config as cfg
from thermodynamics import saturation_mixing_ratio, qsat_mixed, _mixed
from cases.land_cu import exner_profile

Z_TROP   = 11000.0     # m
GAMMA_ST = 15.0e-3     # K/m above the tropopause
RH_ST    = 0.20

THETA = ((0.0, 299.0), (900.0, 299.0), (1500.0, 301.5), (3800.0, 309.5), (4700.0, 311.1),
         (4900.0, 312.6), (8500.0, 327.0), (10000.0, 332.25), (Z_TROP, 336.25))
RH = ((900.0, 0.80), (1500.0, 0.70), (3600.0, 0.60), (3800.0, 0.92), (4700.0, 0.92), (4900.0, 0.40),
      (8300.0, 0.40), (8500.0, 0.92), (10000.0, 0.92), (Z_TROP, 0.30))
Z_ML = 900.0

# "turb" variants (2026-11-04): cirrus layer C thinner and drier (8.8–9.6 km, RH 0.88) so it saturates in patches
RH_TURB = ((900.0, 0.80), (1500.0, 0.70), (3600.0, 0.60), (3800.0, 0.92), (4700.0, 0.92), (4900.0, 0.40),
           (8600.0, 0.40), (8800.0, 0.88), (9600.0, 0.88), (9800.0, 0.40), (Z_TROP, 0.30))
RH_MOIST = 0.85        # LD_RH above this (and above 2 km) marks a moist layer for LD_RH_PATCH / LD_THETA_NOISE_UP

Z_NOISE     = 200.0
THETA_NOISE = 0.1
QV_NOISE    = 1.0e-4


def wind_profile(z: np.ndarray) -> np.ndarray:
    """u_g(z) = 2 + 18·min(z/10 km, 1) m/s, plus an optional jet LD_JET_DU·cos²(π(z − LD_JET_Z)/(2 LD_JET_W))
    within ±LD_JET_W of LD_JET_Z (shear for Kelvin–Helmholtz billows at the edges of a moist layer)."""
    u = 2.0 + 18.0 * np.minimum(z / 10000.0, 1.0)
    du, zj, wj = getattr(cfg, 'LD_JET_DU', 0.0), getattr(cfg, 'LD_JET_Z', 5100.0), getattr(cfg, 'LD_JET_W', 500.0)
    if du:
        x = (z - zj) / wj
        u = u + du * np.where(np.abs(x) < 1.0, np.cos(0.5 * np.pi * x) ** 2, 0.0)
    return u


def ls_forcing() -> dict:
    """
    Large-scale forcing of an approaching trough (LS_FORCING = "case", see forcing.apply_large_scale_forcing):
    ascent w_ls(z) = W · sin(π (z − 1 km)/(11 km − 1 km)) above 1 km, peaking at 6 km, zero below 1 km and
    above 11 km; W ramps 0 → LD_W_MAX (4 cm/s) between 1 h and 4 h and stays. No horizontal advection.
    Over the ramp (≈ 1.5 h at full W) the layers rise ~250 m: ~1–2 K adiabatic cooling, enough to
    saturate RH 0.92; later the ascent keeps them saturated.
    """
    z = np.arange(0.0, 14001.0, 250.0)
    t = np.array([0.0, 3600.0, 14400.0, 86400.0])
    amp = cfg.LD_W_MAX * np.array([0.0, 0.0, 1.0, 1.0])
    shape = np.where((z > 1000.0) & (z < Z_TROP), np.sin(np.pi * (z - 1000.0) / (Z_TROP - 1000.0)), 0.0)
    w = amp[:, None] * shape[None, :]
    zeros = np.zeros_like(w)
    return dict(t=t, z=z, w=w, thl_adv=zeros, qt_adv=zeros)


_TURB = dict(LS_FORCING="case", LD_RH=RH_TURB, LD_W_MAX=0.03, SHF=200.0, LHF=320.0,
             LD_RH_PATCH=0.04, LD_PATCH_SCALE=2000.0, LD_THETA_NOISE_UP=0.1)
VARIANTS = {
    "baseline": dict(LS_FORCING=None),
    "trough":   dict(LS_FORCING="case"),       # ls_forcing() below
    # patchy, more turbulent layers (docs/PLAN_layered_day2.md); radiation / ice fall are switched per run
    "turb":     dict(_TURB),
    "turb_jet": dict(_TURB, LD_JET_DU=8.0, LD_JET_Z=5100.0, LD_JET_W=500.0),
}


def configure() -> None:
    """Apply layered_day settings to config (default variant: trough). Applied once."""
    if getattr(cfg, '_LAYERED_DAY_CONFIGURED', False):
        return
    cfg._LAYERED_DAY_CONFIGURED = True

    cfg.NX = 128
    cfg.NY = 128
    cfg.DX = 150.0
    cfg.DY = 150.0

    cfg.SHF = 150.0
    cfg.LHF = 250.0
    cfg.FLUX_RAMP_TIME = 1800.0
    cfg.SHF_SIGMA = 0.0
    cfg.ENABLE_FLUX_PATTERN = False
    cfg.ENABLE_CLOUD_SHADING = False
    cfg.SHF_SPOT_AMP = 0.0
    cfg.SHF_SPOT_LHF_AMP = 0.0
    cfg.BUBBLE_DTHETA = 0.0

    cfg.LD_THETA = THETA
    cfg.LD_RH = RH
    cfg.LD_Z_ML = Z_ML

    cfg.U_GEO_PROFILE = wind_profile
    cfg.U_GEO = 2.0
    cfg.V_GEO = 3.0

    cfg.Z_BL = Z_ML
    cfg.W_SUBS = 0.0
    cfg.RAD_COOLING = 0.0
    cfg.NUDGE_Z_TOP = 1.0e9
    cfg.TAU_NUDGE_T = 1.0e12
    cfg.TAU_NUDGE_Q = 1.0e12
    cfg.LS_FORCING = "case"
    cfg.LD_W_MAX = 0.04
    cfg.LD_RH_PATCH = 0.0          # amplitude (RH units, 1 σ) of smooth humidity patches in the moist layers
    cfg.LD_PATCH_SCALE = 2000.0    # m, size of those patches (~2.5 σ of the Gaussian filter)
    cfg.LD_THETA_NOISE_UP = 0.0    # K, white θ noise in the moist layers
    cfg.LD_JET_DU = 0.0

    cfg.ICE_SCHEME = "graupel"
    cfg.RAIN_MOMENTS = 2

    cfg.BC_X = "open"
    cfg.BC_Y = "open"
    cfg.OPEN_INFLOW_REF = "precursor"
    cfg.PRECURSOR_NX = 64
    cfg.PRECURSOR_NY = 64

    cfg.GRID_TYPE = "uniform_stretched"
    cfg.DZ_UNIFORM = 100.0
    cfg.Z_UNIFORM_TOP = Z_TROP
    cfg.Z_TOP = 14000.0
    cfg.N_STRETCH = 12
    cfg.Z_SPONGE = 12000.0

    cfg.CFL_SUM = True
    cfg.CFL_MAX = 0.8
    cfg.DT = 3.0


def theta_profile(z: np.ndarray) -> np.ndarray:
    zs, th = np.array(cfg.LD_THETA, dtype=float).T
    return np.interp(z, zs, th) + GAMMA_ST * np.maximum(z - zs[-1], 0.0)


def qsat_profile(theta0: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Model q_sat [kg/kg] on the reference state (over the liquid/ice mix when ICE_SCHEME is on)."""
    pi0 = exner_profile(theta0, z)
    T = theta0 * pi0
    if _mixed():
        return qsat_mixed(T, cfg.P00 * pi0 ** (cfg.CP / cfg.RD))[0]
    return saturation_mixing_ratio(T, pi0)


def reference_profiles(z: np.ndarray):
    """(θ₀, q_v0) on cell centres z: constant q_v below LD_Z_ML (RH of the first LD_RH point at LD_Z_ML)."""
    theta0 = theta_profile(z)
    qsat = qsat_profile(theta0, z)
    zs, rhs = np.array(cfg.LD_RH, dtype=float).T
    rh = np.where(z <= zs[-1], np.interp(z, zs, rhs), RH_ST)

    z_ml = cfg.LD_Z_ML
    z_fine = np.arange(z[0], z_ml + 2.0, 1.0)
    qv_ml = rhs[0] * np.interp(z_ml, z_fine, qsat_profile(theta_profile(z_fine), z_fine))
    return theta0, np.where(z <= z_ml, qv_ml, rh * qsat)


def init_perturbations(s, g) -> None:
    """Background q_v plus white-noise θ′, q_v′ in the lowest 200 m. With LD_RH_PATCH > 0: smooth RH patches
    (σ LD_RH_PATCH, size LD_PATCH_SCALE, RH capped at 0.98) in the moist layers; LD_THETA_NOISE_UP: white θ noise
    there. The first random draws are unchanged, so trough/baseline runs start bitwise as before."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    shape = (g.nx, g.ny, g.nz)
    low = g.z_agl() < Z_NOISE
    s.qv[:] = s.qv0[np.newaxis, np.newaxis, :]
    s.theta += THETA_NOISE * rng.uniform(-1.0, 1.0, shape) * low
    s.qv    += QV_NOISE    * rng.uniform(-1.0, 1.0, shape) * low
    if cfg.LD_RH_PATCH <= 0.0 and cfg.LD_THETA_NOISE_UP <= 0.0:
        return
    zs, rhs = np.array(cfg.LD_RH, dtype=float).T
    moist = (np.interp(g.z, zs, rhs) > RH_MOIST) & (g.z > 2000.0)
    qsat = qsat_profile(s.theta0, g.z)
    if cfg.LD_RH_PATCH > 0.0:
        sig = cfg.LD_PATCH_SCALE / 2.5 / cfg.DX
        f = np.empty((g.nx, g.ny, 2))                    # independent patches below / above 7 km
        for i in range(2):
            f[:, :, i] = gaussian_filter(rng.standard_normal((g.nx, g.ny)), sigma=sig, mode="wrap")
            f[:, :, i] = (f[:, :, i] - f[:, :, i].mean()) / f[:, :, i].std()   # zero mean: moisture only redistributed
        f = np.where(g.z < 7000.0, f[:, :, :1], f[:, :, 1:])
        rh = np.minimum(s.qv0 / qsat + cfg.LD_RH_PATCH * f, 0.98)
        s.qv[:] = np.where(moist, rh * qsat, s.qv)
    s.theta += cfg.LD_THETA_NOISE_UP * rng.uniform(-1.0, 1.0, shape) * moist


def print_rh_profile(s, g) -> None:
    """Initial θ₀, T₀, q_v0, RH (over the model saturation) and u_g every ~500 m."""
    pi0 = s._pi0()[0, 0]
    qsat = qsat_profile(s.theta0, g.z)
    print("=" * 66)
    print("layered_day initial sounding")
    print("=" * 66)
    print(f"{'z (m)':>8} {'θ0 (K)':>9} {'T0 (°C)':>8} {'qv0 (g/kg)':>11} {'RH':>6} {'u_g':>6} {'v_g':>6}")
    last = -1e9
    for k in range(g.nz):
        if g.z[k] - last < 480.0:
            continue
        last = g.z[k]
        print(f"{g.z[k]:8.0f} {s.theta0[k]:9.2f} {s.theta0[k] * pi0[k] - 273.15:8.1f} "
              f"{s.qv0[k]*1e3:11.3f} {s.qv0[k] / qsat[k]:6.2f} {s.u_g[k]:6.1f} {cfg.V_GEO:6.1f}")
    print(f"Max qc anywhere: {s.qc.max():.1e} kg/kg")
    print("=" * 66)

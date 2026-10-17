"""
Single cumulus congestus ("single_cu"): one deep warm cloud (tops ~4–7 km) from a
localised trigger, instead of a field of shallow cumulus.

Sounding (z = height above surface), no capping inversion:
  0–600 m       mixed layer:  θ = 300 K, qv constant with RH = 0.75 at 600 m (LCL ≈ 1.2 km)
  600–9500 m    troposphere:  dθ/dz from cfg.SCU_LAPSE: 3.5 K/km to 1200 m (keeps the background
                BL under the LCL), 2.0 K/km to 3000 m (LFC close to the LCL), 4.5 K/km to 9500 m;
                RH linear 0.75 → 0.50 at 5000 m, then 0.50. With cfg.SCU_RH_CL set: RH = SCU_RH_CL from 600 m to
                cfg.SCU_Z_RH_CL, then linear to 0.50 at 5000 m (moister cloud layer)
  > 9500 m      stratosphere: dθ/dz = 15 K/km, RH = 0.20
Parcel (tests/test_single_cu.py): LCL 1175 m, LFC 1575 m, EL 9.3 km, CAPE ≈ 1220 J/kg;
freezing level ≈ 3.55 km. The 2026-10-15 runs used uniform 3.5 K/km (SCU_LAPSE =
((9500.0, 3.5e-3),): LFC 1.9 km), on which the hot spot never got past ~2.7 km. Warm rain only (Kessler): no ice, no anvil.

Triggers (variants, see VARIANTS and run_case.py single_cu:<variant>):
  hotspot  SHF + 300 and LHF + 300 W/m² in a disc 3 km across at the domain centre
           (forcing.spot_profile; the dry +300 SHF spot stayed shallow in 64² tests)
  bubble   +2 K, moist (RH 0.85) cos² bubble centred at 500 m, 3 km across, 1 km deep, t = 0
Both have the same weak background fluxes (SHF 50, LHF 100 W/m²), which alone keep the
boundary layer below the LCL, so only the trigger makes a cloud.

No large-scale forcing (subsidence, nudging, radiative cooling all off), light uniform
wind 1.5 m/s, 128×128 at 100 m, 50 m levels to 7 km stretched to 12 km, sponge from 9.5 km.

Initial perturbations: white noise θ' ∈ [-0.1, 0.1] K, qv' ∈ [-0.1, 0.1] g/kg below
200 m (breaks the symmetry), plus the bubble when BUBBLE_DTHETA > 0.
"""

import numpy as np
import config as cfg
from thermodynamics import saturation_mixing_ratio
from cases.land_cu import exner_profile

THETA_ML  = 300.0      # K, mixed-layer θ
Z_ML      = 600.0      # m, mixed-layer top
GAMMA_TR  = 3.5e-3     # K/m, tropospheric dθ/dz
Z_TROP    = 9500.0     # m, tropopause
GAMMA_ST  = 15.0e-3    # K/m, stratospheric dθ/dz

RH_ML_TOP = 0.75       # RH at the mixed-layer top
RH_MID    = 0.50       # RH from Z_RH_MID up to the tropopause
Z_RH_MID  = 5000.0     # m
RH_ST     = 0.20       # RH above the tropopause

Z_NOISE     = 200.0    # m, perturbations only below this height
THETA_NOISE = 0.1      # K
QV_NOISE    = 1.0e-4   # kg/kg (0.1 g/kg)

def _par(name: str, default):
    """Sounding parameter cfg.SCU_<name> (set by configure(), overridable), else the module default."""
    return getattr(cfg, f"SCU_{name}", default)


def shear_profile(z: np.ndarray) -> np.ndarray:
    """u_g(z) = 4·min(z/5000, 1) m/s: calm at the surface (low-level air stays over the trigger),
    weak shear separates updraft and rain shaft."""
    return 4.0 * np.minimum(z / 5000.0, 1.0)


# Tropical-congestus sounding (2026-10-17): θ 303 K mixed layer to 900 m, qv 15.2 g/kg,
# LCL 1225 m, LFC 1475 m, CAPE ≈ 1790 J/kg; RH 0.85 to 3 km, 0.70 at 5 km; +1.5 K stable
# layer at 5.3–5.8 km near 0 °C … −10 °C as a soft cap for the tops.
CONGESTUS = dict(
    SCU_THETA_ML=303.0, SCU_Z_ML=900.0, SCU_RH_ML_TOP=0.85, Z_BL=900.0,
    SCU_LAPSE=((1300.0, 3.5e-3), (3000.0, 2.0e-3), (5300.0, 4.5e-3), (5800.0, 7.5e-3),
               (Z_TROP, 4.5e-3)),
    SCU_RH=((900.0, 0.85), (3000.0, 0.85), (5000.0, 0.70), (Z_TROP, 0.50)),
    U_GEO_PROFILE=shear_profile,
    NX=160, NY=160,
    SHF_SPOT_AMP=300.0, SHF_SPOT_LHF_AMP=300.0, SHF_SPOT_DIAM=5000.0,
    BUBBLE_DTHETA=1.5, BUBBLE_RH=0.85,
)

# Variants: config values applied after configure() by run_case.py (case:variant)
VARIANTS = {
    "hotspot": dict(SHF_SPOT_AMP=300.0, SHF_SPOT_LHF_AMP=300.0, BUBBLE_DTHETA=0.0),
    "bubble":  dict(SHF_SPOT_AMP=0.0, SHF_SPOT_LHF_AMP=0.0, BUBBLE_DTHETA=2.0, BUBBLE_RH=0.85),
    "congestus": CONGESTUS,      # tropical sounding + 5 km moist hot spot + bubble + shear, 160²
}


def configure() -> None:
    """
    Apply single_cu settings to config (default trigger: hot spot).

    Called by Simulation when CASE == "single_cu", before the grid is built.
    Applied once: a run script may call configure() itself and then override values.
    """
    if getattr(cfg, '_SINGLE_CU_CONFIGURED', False):
        return
    cfg._SINGLE_CU_CONFIGURED = True

    # Domain: 12.8 × 12.8 km at 100 m
    cfg.NX = 128
    cfg.NY = 128
    cfg.DX = 100.0
    cfg.DY = 100.0

    # Weak background surface fluxes, ramped over 30 min, no shading
    cfg.SHF = 50.0
    cfg.LHF = 100.0
    cfg.FLUX_RAMP_TIME = 1800.0
    cfg.SHF_SIGMA = 0.0
    cfg.ENABLE_FLUX_PATTERN = False
    cfg.ENABLE_CLOUD_SHADING = False

    # Trigger (hot spot by default; see VARIANTS)
    for key, val in VARIANTS["hotspot"].items():
        setattr(cfg, key, val)

    # Stratification above Z_ML: ((z_top, dθ/dz [K/m]), ...), last layer ends at Z_TROP.
    # Stable below cloud base (keeps the background BL under the LCL), weakly stable
    # cloud layer (LFC just above the LCL), more stable above 3 km (CAPE, tops).
    # ((Z_TROP, 3.5e-3),) = the sounding of the 2026-10-15 runs.
    cfg.SCU_LAPSE = ((1200.0, 3.5e-3), (3000.0, 2.0e-3), (Z_TROP, 4.5e-3))

    # Mixed layer and humidity above it. SCU_RH: ((z, RH), ...) points above SCU_Z_ML,
    # linearly interpolated (constant beyond the last point, RH_ST above the tropopause);
    # None = the SCU_RH_CL / linear 0.75 → 0.50 profile below.
    cfg.SCU_THETA_ML = THETA_ML
    cfg.SCU_Z_ML = Z_ML
    cfg.SCU_RH_ML_TOP = RH_ML_TOP
    cfg.SCU_RH = None

    # Moister cloud layer (None = linear RH 0.75 → 0.50 from Z_ML to Z_RH_MID)
    cfg.SCU_RH_CL = None
    cfg.SCU_Z_RH_CL = 3000.0

    # Light uniform wind
    cfg.U_GEO_PROFILE = None
    cfg.U_GEO = 1.5
    cfg.V_GEO = 0.0

    # No large-scale forcing
    cfg.Z_BL = Z_ML                      # mixed-layer top (BL diagnostics)
    cfg.W_SUBS = 0.0
    cfg.RAD_COOLING = 0.0
    cfg.NUDGE_Z_TOP = 1.0e9              # nudging mask empty
    cfg.TAU_NUDGE_T = 1.0e12
    cfg.TAU_NUDGE_Q = 1.0e12

    # Vertical grid: 50 m to 7000 m (140 levels), 20 stretched levels to 12 km
    cfg.GRID_TYPE = "uniform_stretched"
    cfg.DZ_UNIFORM = 50.0
    cfg.Z_UNIFORM_TOP = 7000.0
    cfg.Z_TOP = 12000.0
    cfg.N_STRETCH = 20
    cfg.Z_SPONGE = 9500.0

    # Adaptive dt on the summed 3-D Courant number, capped at 3 s (updrafts 10–20 m/s)
    cfg.CFL_SUM = True
    cfg.CFL_MAX = 0.8
    cfg.DT = 3.0


def theta_profile(z: np.ndarray) -> np.ndarray:
    """
    Piecewise-linear θ₀(z) [K]: THETA_ML up to Z_ML, then the (z_top, dθ/dz) layers of
    cfg.SCU_LAPSE (last one ends at Z_TROP), then GAMMA_ST. Without SCU_LAPSE (configure()
    not called): GAMMA_TR from Z_ML to Z_TROP.
    """
    layers = getattr(cfg, 'SCU_LAPSE', None) or ((Z_TROP, GAMMA_TR),)
    theta = np.full_like(z, _par("THETA_ML", THETA_ML), dtype=float)
    z_bot = _par("Z_ML", Z_ML)
    for z_top, gamma in list(layers[:-1]) + [(Z_TROP, layers[-1][1])]:
        theta += gamma * np.clip(z - z_bot, 0.0, z_top - z_bot)
        z_bot = z_top
    return theta + GAMMA_ST * np.maximum(z - Z_TROP, 0.0)


def rh_profile(z: np.ndarray) -> np.ndarray:
    """Target RH above the mixed layer (the mixed layer has constant qv instead)."""
    points = _par("RH", None)
    if points is not None:
        zs, rhs = np.array(points, dtype=float).T
        return np.where(z <= Z_TROP, np.interp(z, zs, rhs), RH_ST)
    rh_cl = getattr(cfg, 'SCU_RH_CL', None)
    if rh_cl is None:
        rh_lo, z_lo = RH_ML_TOP, Z_ML
    else:
        rh_lo, z_lo = rh_cl, cfg.SCU_Z_RH_CL
    frac = np.clip((z - z_lo) / (Z_RH_MID - z_lo), 0.0, 1.0)
    return np.where(z <= Z_TROP, rh_lo + (RH_MID - rh_lo) * frac, RH_ST)


def qsat_profile(theta0: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Model q_sat [kg/kg] on the reference state."""
    pi0 = exner_profile(theta0, z)
    return saturation_mixing_ratio(theta0 * pi0, pi0)


def reference_profiles(z: np.ndarray):
    """
    Return (θ₀, qv₀) on cell centres z. q_sat at Z_ML (between grid levels in general)
    is interpolated from a fine column with the same surface reference (z[0]).
    """
    theta0 = theta_profile(z)
    qsat = qsat_profile(theta0, z)

    z_ml = _par("Z_ML", Z_ML)
    z_fine = np.arange(z[0], z_ml + 2.0, 1.0)
    qv_ml = _par("RH_ML_TOP", RH_ML_TOP) * np.interp(z_ml, z_fine,
                                                     qsat_profile(theta_profile(z_fine), z_fine))

    qv0 = np.where(z <= z_ml, qv_ml, rh_profile(z) * qsat)
    return theta0, qv0


def add_warm_bubble(s, g) -> None:
    """θ′ += BUBBLE_DTHETA·w, qv → qv + w·max(BUBBLE_RH·q_sat(T) − qv, 0), w = cos² bubble."""
    lx, ly = g.nx * g.dx, g.ny * g.dy
    x0 = 0.5 * lx if cfg.BUBBLE_X0 is None else cfg.BUBBLE_X0
    y0 = 0.5 * ly if cfg.BUBBLE_Y0 is None else cfg.BUBBLE_Y0
    ddx = ((g.x - x0 + 0.5 * lx) % lx - 0.5 * lx)[:, None, None]
    ddy = ((g.y - y0 + 0.5 * ly) % ly - 0.5 * ly)[None, :, None]
    ddz = (g.z - cfg.BUBBLE_Z0)[None, None, :]
    r = np.sqrt((ddx**2 + ddy**2) / cfg.BUBBLE_RX**2 + ddz**2 / cfg.BUBBLE_RZ**2)
    w = np.where(r < 1.0, np.cos(0.5 * np.pi * np.minimum(r, 1.0)) ** 2, 0.0)

    s.theta += cfg.BUBBLE_DTHETA * w
    if cfg.BUBBLE_RH > 0.0:
        pi0 = s._pi0()
        qsat = saturation_mixing_ratio((s.theta + s.theta0[None, None, :]) * pi0, pi0)
        s.qv += w * np.maximum(cfg.BUBBLE_RH * qsat - s.qv, 0.0)


def init_perturbations(s, g) -> None:
    """Background qv, white-noise θ', qv' in the lowest 200 m, optional warm bubble."""
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    shape = (g.nx, g.ny, g.nz)
    low = (g.z < Z_NOISE)[np.newaxis, np.newaxis, :]

    s.qv[:] = s.qv0[np.newaxis, np.newaxis, :]
    s.theta += THETA_NOISE * rng.uniform(-1.0, 1.0, shape) * low
    s.qv    += QV_NOISE    * rng.uniform(-1.0, 1.0, shape) * low
    if cfg.BUBBLE_DTHETA != 0.0:
        add_warm_bubble(s, g)


def print_rh_profile(s, g) -> None:
    """Print the initial θ₀, qv₀ and RH profile (horizontal mean and max) and qc max."""
    pi0 = s._pi0()
    T = (s.theta + s.theta0[np.newaxis, np.newaxis, :]) * pi0
    rh = s.qv / saturation_mixing_ratio(T, pi0)
    rh_mean = rh.mean(axis=(0, 1))
    rh_max = rh.max(axis=(0, 1))
    qc_max = s.qc.max(axis=(0, 1))

    print("=" * 74)
    print("single_cu initial sounding")
    print("=" * 74)
    print(f"{'z (m)':>8} {'θ0 (K)':>9} {'T0 (°C)':>8} {'qv0 (g/kg)':>11} {'RH mean':>9} "
          f"{'RH max':>8} {'qc max':>8}")
    for k in range(g.nz):
        if g.z[k] > 1500.0 and k % 10:
            continue   # thin out above the cloud base
        print(f"{g.z[k]:8.0f} {s.theta0[k]:9.2f} {s.theta0[k] * pi0[0, 0, k] - 273.15:8.1f} "
              f"{s.qv0[k]*1e3:11.3f} {rh_mean[k]:9.3f} {rh_max[k]:8.3f} {qc_max[k]:8.1e}")
    print(f"Max RH anywhere: {rh.max():.3f}   Max qc anywhere: {s.qc.max():.1e} kg/kg")
    print("=" * 74)

"""
Single cumulus congestus ("single_cu"): one deep warm cloud (tops ~4–7 km) from a
localised trigger, instead of a field of shallow cumulus.

Sounding (z = height above surface), no capping inversion:
  0–600 m       mixed layer:  θ = 300 K, qv constant with RH = 0.75 at 600 m (LCL ≈ 1.2 km)
  600–9500 m    troposphere:  dθ/dz = 3.5 K/km (T lapse ≈ 6 K/km), RH linear 0.75 → 0.50
                at 5000 m, then 0.50
  > 9500 m      stratosphere: dθ/dz = 15 K/km, RH = 0.20
Parcel CAPE of the mixed-layer air ≈ 1200 J/kg (tests/test_single_cu.py prints it);
freezing level ≈ 3.8 km. Warm rain only (Kessler): no ice, no anvil.

Triggers (variants, see VARIANTS and run_case.py single_cu:<variant>):
  hotspot  SHF + 300 W/m² in a disc 3 km across at the domain centre (forcing.spot_profile)
  bubble   +2 K, moist (RH 0.95) cos² bubble centred at 500 m, 3 km across, 1 km deep, t = 0
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

# Trigger variants: config values applied after configure() by run_case.py (case:variant)
VARIANTS = {
    "hotspot": dict(SHF_SPOT_AMP=300.0, BUBBLE_DTHETA=0.0),
    "bubble":  dict(SHF_SPOT_AMP=0.0, BUBBLE_DTHETA=2.0, BUBBLE_RH=0.95),
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
    """Piecewise-linear θ₀(z) [K]."""
    theta_trop = THETA_ML + GAMMA_TR * (Z_TROP - Z_ML)
    return np.select(
        [z <= Z_ML, z <= Z_TROP],
        [np.full_like(z, THETA_ML), THETA_ML + GAMMA_TR * (z - Z_ML)],
        theta_trop + GAMMA_ST * (z - Z_TROP),
    )


def rh_profile(z: np.ndarray) -> np.ndarray:
    """Target RH above the mixed layer (the mixed layer has constant qv instead)."""
    frac = np.clip((z - Z_ML) / (Z_RH_MID - Z_ML), 0.0, 1.0)
    return np.where(z <= Z_TROP, RH_ML_TOP + (RH_MID - RH_ML_TOP) * frac, RH_ST)


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

    z_fine = np.arange(z[0], Z_ML + 2.0, 1.0)
    qv_ml = RH_ML_TOP * np.interp(Z_ML, z_fine, qsat_profile(theta_profile(z_fine), z_fine))

    qv0 = np.where(z <= Z_ML, qv_ml, rh_profile(z) * qsat)
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

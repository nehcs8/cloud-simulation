#!/usr/bin/env python3
"""
Diagnose why clouds aren't forming by computing LCL and checking thermodynamic profiles.
"""

import numpy as np
import config as cfg

# Physical constants
G = cfg.G
CP = cfg.CP
RD = cfg.RD
RV = cfg.RV
LV = cfg.LV
P00 = cfg.P00
EPS = cfg.EPS

def compute_lcl(T_sfc, p_sfc, qv_sfc):
    """
    Compute Lifting Condensation Level using Bolton (1980) formula.

    Parameters:
    -----------
    T_sfc : float
        Surface temperature [K]
    p_sfc : float
        Surface pressure [Pa]
    qv_sfc : float
        Surface water vapor mixing ratio [kg/kg]

    Returns:
    --------
    z_lcl : float
        LCL height [m]
    T_lcl : float
        LCL temperature [K]
    p_lcl : float
        LCL pressure [Pa]
    """
    # Vapor pressure
    e = p_sfc * qv_sfc / (EPS + qv_sfc)  # Pa

    # Dewpoint (Bolton 1980)
    e_mb = e / 100.0  # hPa
    Td = 243.5 / ((17.67 / np.log(e_mb / 6.112)) - 1) + 273.15  # K

    # LCL temperature (Bolton 1980)
    T_lcl = 1.0 / (1.0/(Td - 56.0) + np.log(T_sfc/Td)/800.0) + 56.0  # K

    # LCL pressure (adiabatic ascent)
    p_lcl = p_sfc * (T_lcl / T_sfc) ** (CP / RD)  # Pa

    # LCL height (hydrostatic with mean temperature)
    T_mean = 0.5 * (T_sfc + T_lcl)
    z_lcl = (RD * T_mean / G) * np.log(p_sfc / p_lcl)  # m

    return z_lcl, T_lcl, p_lcl, Td


def diagnose_atmosphere():
    """Diagnose current atmospheric state and cloud formation potential."""

    print("=" * 80)
    print("CLOUD FORMATION DIAGNOSTICS")
    print("=" * 80)

    # Surface conditions from config
    theta_sfc = cfg.THETA_S
    qv_sfc = cfg.QV_SURF

    # Exner function at surface (pi0 = 1 at surface)
    pi0_sfc = 1.0
    T_sfc = theta_sfc * pi0_sfc  # K
    p_sfc = P00  # Pa (assuming surface is at reference pressure)

    print(f"\n1. SURFACE CONDITIONS:")
    print(f"   θ_sfc     = {theta_sfc:.2f} K")
    print(f"   T_sfc     = {T_sfc:.2f} K  ({T_sfc - 273.15:.2f} °C)")
    print(f"   p_sfc     = {p_sfc/100:.1f} hPa")
    print(f"   qv_sfc    = {qv_sfc*1000:.3f} g/kg")
    print(f"   RH_sfc    = {compute_rh(T_sfc, qv_sfc)*100:.1f}%")

    # Compute LCL
    z_lcl, T_lcl, p_lcl, Td_sfc = compute_lcl(T_sfc, p_sfc, qv_sfc)

    print(f"\n2. LIFTING CONDENSATION LEVEL (LCL):")
    print(f"   Td_sfc    = {Td_sfc:.2f} K  ({Td_sfc - 273.15:.2f} °C)")
    print(f"   T-Td      = {T_sfc - Td_sfc:.2f} K  (dewpoint depression)")
    print(f"   z_LCL     = {z_lcl:.0f} m")
    print(f"   T_LCL     = {T_lcl:.2f} K  ({T_lcl - 273.15:.2f} °C)")
    print(f"   p_LCL     = {p_lcl/100:.1f} hPa")

    # Check if LCL is reachable
    z_bl = cfg.Z_BL
    print(f"\n3. BOUNDARY LAYER:")
    print(f"   z_BL      = {z_bl:.0f} m")
    print(f"   LCL/z_BL  = {z_lcl/z_bl:.2f}")

    if z_lcl > z_bl * 1.5:
        print(f"   ⚠️  WARNING: LCL is {z_lcl/z_bl:.1f}x higher than BL top!")
        print(f"       Thermals unlikely to reach saturation.")
    elif z_lcl > z_bl:
        print(f"   ⚠️  CAUTION: LCL is above BL top (needs strong thermals)")
    else:
        print(f"   ✓ LCL is within BL (clouds should form)")

    # Compute parcel buoyancy at LCL
    # Profile at LCL height
    if z_lcl <= z_bl:
        theta_env = theta_sfc + cfg.GAMMA_BL * 1e-3 * z_lcl
    else:
        theta_env = theta_sfc + cfg.GAMMA_BL * 1e-3 * z_bl + cfg.GAMMA_FT * 1e-3 * (z_lcl - z_bl)

    qv_env = qv_sfc * np.exp(-z_lcl / cfg.QV_SCALE)

    # Parcel properties at LCL (conserved theta, qv until saturation)
    theta_parcel = theta_sfc
    qv_parcel = qv_sfc

    # Buoyancy (simplified, ignoring water loading)
    B_lcl = G * ((theta_parcel - theta_env) / theta_env + 0.61 * (qv_parcel - qv_env))

    print(f"\n4. BUOYANCY AT LCL:")
    print(f"   θ_env     = {theta_env:.2f} K")
    print(f"   θ_parcel  = {theta_parcel:.2f} K")
    print(f"   Δθ        = {theta_parcel - theta_env:+.2f} K")
    print(f"   qv_env    = {qv_env*1000:.3f} g/kg")
    print(f"   B_LCL     = {B_lcl:+.4f} m/s²")

    if B_lcl < 0:
        print(f"   ⚠️  NEGATIVE BUOYANCY: Parcel is denser than environment!")
        print(f"       Even if heated at surface, parcel won't reach LCL.")

    # Estimate required surface heating
    shf = cfg.SHF
    lhf = cfg.LHF

    print(f"\n5. SURFACE FORCING:")
    print(f"   SHF       = {shf:.1f} W/m²")
    print(f"   LHF       = {lhf:.1f} W/m²")

    # Estimate max surface theta perturbation from SHF
    # Assume mixed layer of depth h, heat flux for time dt
    h_mix = 300  # m, typical mixed layer depth
    dt_heat = 3600  # s, 1 hour of heating
    rho0 = P00 / (RD * T_sfc)  # kg/m³

    dtheta_max = (shf * dt_heat) / (rho0 * CP * h_mix)

    print(f"   Max Δθ after 1hr = {dtheta_max:.2f} K  (in {h_mix}m mixed layer)")

    # Check if this is enough to overcome stability
    theta_lcl_env = theta_env
    theta_needed = theta_lcl_env + 0.5  # need slight positive buoyancy
    theta_deficit = theta_needed - theta_sfc

    print(f"\n6. THERMODYNAMIC BARRIER:")
    print(f"   θ needed to reach LCL with buoyancy = {theta_needed:.2f} K")
    print(f"   θ deficit = {theta_deficit:.2f} K")

    if dtheta_max < theta_deficit:
        print(f"   ⚠️  INSUFFICIENT HEATING: Max Δθ={dtheta_max:.2f}K < deficit={theta_deficit:.2f}K")
        print(f"       Surface flux too weak to overcome stability.")

    # Recommendations
    print(f"\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)

    if z_lcl > z_bl * 1.2:
        print("\n1. INCREASE MOISTURE (lower LCL):")
        # Target LCL at ~1000-1200m (within BL)
        qv_target = find_qv_for_target_lcl(T_sfc, p_sfc, 1200.0)
        print(f"   Set QV_SURF = {qv_target:.4f} kg/kg  (currently {qv_sfc:.4f})")
        print(f"   This would place LCL at ~1200m")

    if B_lcl < 0 or dtheta_max < theta_deficit:
        print("\n2. INCREASE SURFACE HEATING:")
        print(f"   Option A: Increase SHF to {shf * 1.5:.0f} W/m²  (+50%)")
        print(f"   Option B: Increase LHF to {lhf * 3:.0f} W/m²  (adds moisture)")

    if cfg.GAMMA_BL > 1.5:
        print("\n3. REDUCE BL STABILITY:")
        print(f"   Set GAMMA_BL = 0.5-1.0 K/km  (currently {cfg.GAMMA_BL} K/km)")
        print(f"   This creates a less stable BL, easier for thermals to rise")

    print("\n" + "=" * 80)


def compute_rh(T, qv):
    """Compute relative humidity from T and qv."""
    # Saturation vapor pressure (Bolton 1980)
    es = 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))  # Pa
    # Saturation mixing ratio
    qvs = EPS * es / (P00 - es)
    return qv / qvs


def find_qv_for_target_lcl(T_sfc, p_sfc, z_target):
    """Find qv_sfc that gives target LCL height."""
    # Binary search
    qv_min, qv_max = 0.001, 0.020
    for _ in range(20):
        qv_mid = 0.5 * (qv_min + qv_max)
        z_lcl, _, _, _ = compute_lcl(T_sfc, p_sfc, qv_mid)
        if z_lcl > z_target:
            qv_min = qv_mid
        else:
            qv_max = qv_mid
    return qv_mid


if __name__ == "__main__":
    diagnose_atmosphere()

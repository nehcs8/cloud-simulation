"""
Test the "single_cu" initial state (cases/single_cu.py) and print the parcel profile.

Success criteria:
  - No grid point is saturated at t=0 (hot spot and bubble variants): qc = 0 everywhere
  - Mixed-layer parcel CAPE (pseudo-adiabatic, warm rain only) in 500–1500 J/kg
  - LCL above the initial mixed layer (the background stays clear until it deepens)
  - Hot spot: SHF/LHF excess = SHF_SPOT_AMP/SHF_SPOT_LHF_AMP at the centre, 0 outside SHF_SPOT_DIAM/2
  - Bubble: θ′ max ≈ BUBBLE_DTHETA, RH max < 1

Parcel: θ_l and q_t of the mixed-layer air conserved while lifting, saturation adjusted
with the model's q_sat; condensate removed (pseudo) or kept (reversible, loading).
Buoyancy as in State.buoyancy: g (θ′/θ₀ + 0.61 qv′ − qc).

Usage: python tests/test_single_cu.py
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config as cfg
cfg.CASE = "single_cu"
from cases import single_cu
single_cu.configure()
cfg.NX = cfg.NY = 64          # smaller column set for the test; profiles do not depend on it

from grid import Grid
from state import State
from forcing import build_flux_map
from thermodynamics import saturation_mixing_ratio


def adjust(thl, qt, pi):
    """(θ, qc) from θ_l, q_t at Exner pi by bisection on qc."""
    if qt <= saturation_mixing_ratio(thl * pi, pi):
        return thl, 0.0
    lo, hi = 0.0, qt
    for _ in range(60):
        qc = 0.5 * (lo + hi)
        th = thl + cfg.LV / (cfg.CP * pi) * qc
        lo, hi = (qc, hi) if qt - qc > saturation_mixing_ratio(th * pi, pi) else (lo, qc)
    qc = 0.5 * (lo + hi)
    return thl + cfg.LV / (cfg.CP * pi) * qc, qc


def parcel_profile(z, theta0, qv0, pi0, loading):
    """Buoyancy (nz,) and qc (nz,) of a parcel lifted from the lowest level."""
    thl, qt = theta0[0], qv0[0]
    B, QC = np.zeros_like(z), np.zeros_like(z)
    for k in range(len(z)):
        th, qc = adjust(thl, qt, pi0[k])
        qv = qt - qc
        B[k] = cfg.G * ((th - theta0[k]) / theta0[k] + 0.61 * (qv - qv0[k]) - (qc if loading else 0.0))
        QC[k] = qc
        if not loading:                 # pseudo-adiabatic: rain out the condensate
            qt, thl = qv, th
    return B, QC


def parcel_stats(g, s, loading):
    pi0 = s._pi0()[0, 0]
    B, qc = parcel_profile(g.z, s.theta0, s.qv0, pi0, loading)
    k_lcl = int(np.argmax(qc > 0))
    pos = np.where((B > 0) & (np.arange(g.nz) >= k_lcl))[0]
    cape = float(np.sum(np.maximum(B, 0)[k_lcl:] * g.dz_face[k_lcl:]))
    k_lfc = pos[0] if len(pos) else k_lcl
    cin = float(np.sum(np.minimum(B, 0)[:k_lfc] * g.dz_face[:k_lfc]))
    lfc = g.z[pos[0]] if len(pos) else np.nan
    el = g.z[pos[-1]] if len(pos) else np.nan
    return dict(LCL=g.z[k_lcl], LFC=lfc, EL=el, CAPE=cape, CIN=cin, B=B, qc=qc)


def test_sounding_and_cape():
    g = Grid(terrain=None)
    s = State(g)
    print(g)

    n_cloudy = int((s.qc > 0.0).sum())
    print(f"Cloudy grid points at t=0: {n_cloudy}  (max qc = {s.qc.max():.2e})")
    assert n_cloudy == 0, f"{n_cloudy} saturated grid points at t=0"

    pi0 = s._pi0()[0, 0]
    T0 = s.theta0 * pi0
    print(f"Freezing level: {np.interp(0.0, -(T0 - 273.15), g.z):.0f} m   "
          f"T at tropopause ({single_cu.Z_TROP:.0f} m): "
          f"{np.interp(single_cu.Z_TROP, g.z, T0) - 273.15:.1f} °C")

    for loading in (False, True):
        p = parcel_stats(g, s, loading)
        print(f"Parcel ({'reversible' if loading else 'pseudo-adiabatic'}): LCL {p['LCL']:.0f} m  "
              f"LFC {p['LFC']:.0f} m  EL {p['EL']:.0f} m  CAPE {p['CAPE']:.0f} J/kg  CIN {p['CIN']:.0f} J/kg")
        if not loading:
            pseudo = p

    print(f"{'z (m)':>7} {'B (m/s²)':>9} {'qc (g/kg)':>10}")
    for k in range(0, g.nz, 10):
        print(f"{g.z[k]:7.0f} {pseudo['B'][k]:9.4f} {pseudo['qc'][k]*1e3:10.2f}")

    assert 500.0 <= pseudo['CAPE'] <= 1500.0, f"CAPE {pseudo['CAPE']:.0f} J/kg outside 500–1500"
    assert pseudo['LCL'] > single_cu.Z_ML + 300.0, "LCL too close to the mixed-layer top"


def test_hotspot_map():
    for key, val in single_cu.VARIANTS["hotspot"].items():
        setattr(cfg, key, val)
    g = Grid(terrain=None)
    shf, lhf = build_flux_map(g)
    excess = shf - cfg.SHF
    lx = g.nx * g.dx
    r = np.hypot(g.x[:, None] - 0.5 * lx, g.y[None, :] - 0.5 * lx)
    area = (excess / cfg.SHF_SPOT_AMP).sum() * g.dx * g.dy
    print(f"Hot spot: max excess {excess.max():.1f} W/m², outside-disc max "
          f"{np.abs(excess[r > 0.5 * cfg.SHF_SPOT_DIAM]).max():.1f}, equivalent diameter "
          f"{2 * np.sqrt(area / np.pi):.0f} m, LHF excess max {(lhf - cfg.LHF).max():.1f} W/m²")
    assert abs(excess.max() - cfg.SHF_SPOT_AMP) < 1.0
    assert np.abs(excess[r > 0.5 * cfg.SHF_SPOT_DIAM]).max() == 0.0
    assert abs((lhf - cfg.LHF).max() - cfg.SHF_SPOT_LHF_AMP) < 1.0
    assert np.allclose(lhf - cfg.LHF, cfg.SHF_SPOT_LHF_AMP / cfg.SHF_SPOT_AMP * excess)


def test_bubble():
    for key, val in single_cu.VARIANTS["bubble"].items():
        setattr(cfg, key, val)
    g = Grid(terrain=None)
    s = State(g)
    pi0 = s._pi0()
    rh = s.qv / saturation_mixing_ratio((s.theta + s.theta0[None, None, :]) * pi0, pi0)
    i, j, k = np.unravel_index(np.argmax(s.theta), s.theta.shape)
    print(f"Bubble: θ′ max {s.theta.max():.2f} K at z = {g.z[k]:.0f} m, RH max {rh.max():.3f}, "
          f"qv excess max {(s.qv - s.qv0[None, None, :]).max()*1e3:.2f} g/kg, qc max {s.qc.max():.1e}")
    assert abs(s.theta.max() - cfg.BUBBLE_DTHETA) < 0.2
    assert rh.max() < 1.0 and s.qc.max() == 0.0
    for key in ("SHF_SPOT_AMP", "SHF_SPOT_LHF_AMP", "BUBBLE_DTHETA", "BUBBLE_RH"):
        setattr(cfg, key, 0.0)


def test_congestus_variant():
    """Tropical-congestus variant: unsaturated start, CAPE 400–1500 J/kg, EL 5.5–8 km."""
    saved = {k: getattr(cfg, k) for k in single_cu.CONGESTUS}
    for key, val in single_cu.CONGESTUS.items():
        setattr(cfg, key, val)
    cfg.NX = cfg.NY = 64
    g = Grid(terrain=None)
    s = State(g)
    p = parcel_stats(g, s, loading=False)
    print(f"congestus: qv_ml {s.qv0[0]*1e3:.1f} g/kg  LCL {p['LCL']:.0f} m  LFC {p['LFC']:.0f} m  "
          f"EL {p['EL']:.0f} m  CAPE {p['CAPE']:.0f} J/kg  CIN {p['CIN']:.0f} J/kg  "
          f"u(0, 5 km) = {s.u_g[0]:.2f}, {np.interp(5000, g.z, s.u_g):.2f} m/s  qc max {s.qc.max():.1e}")
    assert s.qc.max() == 0.0
    assert 400.0 <= p['CAPE'] <= 1500.0
    assert 5500.0 <= p['EL'] <= 8000.0, "parcel EL outside congestus range"
    for key, val in saved.items():
        setattr(cfg, key, val)
    cfg.NX = cfg.NY = 64


if __name__ == "__main__":
    test_sounding_and_cape()
    test_hotspot_map()
    test_bubble()
    test_congestus_variant()
    print("All single_cu tests passed.")

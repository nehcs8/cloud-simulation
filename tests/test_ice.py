"""
Mixed-phase cloud condensate (ICE_SCHEME = "mixed", step 1 of the ice plan).

Checks:
  1. Above T_ICE_0 the mixed adjustment equals the warm one (same q_c, θ).
  2. numba kernels = numpy reference (diagnosis, θ_l, warm_rain melt term).
  3. Round trip (θ_l, q_t) → (θ, q_v, q_c) → θ_l is exact; saturated cells sit at q_s over the mix.
  4. Energy: converting cloud ice to (liquid) rain cools by L_f (1 − α) Δq / (c_p π) (+ L_v for rain evaporation).
  5. Reversible parcel ascent with and without ice: extra warming above the freezing level.

Run: ./venv/bin/python3 tests/test_ice.py
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import config as cfg
import thermodynamics as th
import microphysics as mp


def column(nz=100, ztop=10000.0):
    """Styria-like column: θ0 302 K mixed layer to 1.2 km, 4.5 K/km above; hydrostatic π0."""
    z = (np.arange(nz) + 0.5) * ztop / nz
    theta0 = 302.0 + 4.5e-3 * np.maximum(z - 1200.0, 0.0)
    pi0 = np.ones(nz)
    pi0[0] = 1.0 - cfg.G * z[0] / (cfg.CP * theta0[0])
    for k in range(1, nz):
        pi0[k] = pi0[k - 1] - cfg.G * (z[k] - z[k - 1]) / (cfg.CP * 0.5 * (theta0[k] + theta0[k - 1]))
    return z, theta0, pi0


def random_state(nx=12, ny=10, seed=1):
    z, theta0, pi0 = column()
    rng = np.random.default_rng(seed)
    nz = z.size
    p = cfg.P00 * pi0 ** (cfg.CP / cfg.RD)
    T0 = theta0 * pi0
    qsw = cfg.EPS * 611.2 * np.exp(17.67 * (T0 - 273.15) / (T0 - 273.15 + 243.5)) / (p - 611.2)
    # θ_l′ ±3 K, q_t from 0.6 to 1.4 × water saturation of the reference (clear and cloudy cells)
    theta_l = rng.uniform(-3, 3, (nx, ny, nz))
    qt = qsw[None, None, :] * rng.uniform(0.6, 1.4, (nx, ny, nz))
    return z, theta0, pi0, theta_l, qt


def diag(theta_l, qt, theta0, pi0, numba):
    cfg.USE_NUMBA = numba
    return th.convert_from_theta_l_qt(theta_l, qt, np.zeros_like(qt), theta0, pi0.reshape(1, 1, -1))


def main():
    ok = True
    z, theta0, pi0, theta_l, qt = random_state()
    pi3 = pi0.reshape(1, 1, -1)
    T_l = (theta_l + theta0[None, None, :]) * pi3

    # 1. warm limit
    cfg.ICE_SCHEME = "none"
    tw, qvw, qcw, _ = diag(theta_l, qt, theta0, pi0, numba=False)
    cfg.ICE_SCHEME = "mixed"
    tm, qvm, qcm, _ = diag(theta_l, qt, theta0, pi0, numba=False)
    T_m = (tm + theta0[None, None, :]) * pi3
    warm = T_m > cfg.T_ICE_0 + 0.5
    e1 = np.abs(qcm - qcw)[warm].max()
    e1t = np.abs(tm - tw)[warm].max()
    print(f"1. warm cells ({warm.sum()}): max |Δq_c| {e1:.2e} kg/kg, max |Δθ| {e1t:.2e} K")
    ok &= e1 < 1e-8 and e1t < 1e-4

    # 2. numba = numpy
    tn, qvn, qcn, _ = diag(theta_l, qt, theta0, pi0, numba=True)
    e2 = max(np.abs(tn - tm).max(), np.abs(qcn - qcm).max() * 1e3)
    cfg.USE_NUMBA = True
    thl_n = th.theta_l_from_state(tm, qcm, pi3, theta0)
    cfg.USE_NUMBA = False
    thl_p = th.theta_l_from_state(tm, qcm, pi3, theta0)
    e2b = np.abs(thl_n - thl_p).max()
    print(f"2. numba vs numpy: diagnosis max |Δθ|, |Δq_c|·1e3 {e2:.2e}; θ_l {e2b:.2e}")
    ok &= e2 < 1e-8 and e2b < 1e-12

    # 3. round trip and saturation
    e3 = np.abs(thl_p - theta_l).max()
    p = cfg.P00 * pi3 ** (cfg.CP / cfg.RD) * np.ones_like(qt)
    qs, _ = th.qsat_mixed(T_m, p)
    cloudy = qcm > 0
    e3b = np.abs(qvm - qs)[cloudy].max()
    clear_sub = np.all(qvm[~cloudy] <= qs[~cloudy] * (1 + 1e-12))
    a, _ = th.liquid_fraction(T_m)
    mixed = cloudy & (a > 0) & (a < 1)
    icy = cloudy & (a == 0)
    print(f"3. round trip max |Δθ_l| {e3:.2e} K; cloudy cells {cloudy.sum()} (mixed {mixed.sum()}, ice {icy.sum()}): "
          f"max |q_v − q_s| {e3b:.2e}; clear cells sub-saturated: {clear_sub}")
    ok &= e3 < 1e-8 and e3b < 1e-10 and clear_sub and mixed.sum() > 0 and icy.sum() > 0

    # 4. melt term of warm_rain (numpy and numba): only the conversion, no rain evaporation
    class S:
        pass
    for numba in (False, True):
        cfg.USE_NUMBA = numba
        s = S()
        s.theta, s.qv, s.qc = tm.copy(), qvm.copy(), qcm.copy()
        s.qr = np.where(cloudy, 1e-3, 0.0)
        s.theta0, s.rho0 = theta0, cfg.P00 * pi0 ** (cfg.CP / cfg.RD - 1) / (cfg.RD * theta0)
        s._pi0 = lambda: pi3
        before = s.theta.copy(); qc0 = s.qc.copy(); qr0 = s.qr.copy()
        mp.warm_rain(s, None, 5.0)
        conv = qc0 - s.qc
        evap = qr0 + conv - s.qr       # rain also evaporates in icy cloud (q_v < q_sw there)
        expect = -((cfg.LF / cfg.CP) * (1 - a) * conv + (cfg.LV / cfg.CP) * evap) / pi3
        e4 = np.abs((s.theta - before) - expect)[cloudy].max()
        print(f"4. melt term ({'numba' if numba else 'numpy'}): converted {conv.max()*1e3:.3f} g/kg max, "
              f"max |Δθ − expected| {e4:.2e} K (melt + rain evaporation)")
        ok &= e4 < 1e-10 and conv.max() > 0

    # 5. reversible parcel from the LCL region: θ_l, q_t of a saturated parcel at 1.7 km
    k0 = np.searchsorted(z, 1700.0)
    T_base = theta0[k0] * pi0[k0]
    p_base = cfg.P00 * pi0[k0] ** (cfg.CP / cfg.RD)
    q_parcel = cfg.EPS * 611.2 * np.exp(17.67 * (T_base - 273.15) / (T_base - 273.15 + 243.5)) / (p_base - 611.2)
    print(f"5. reversible ascent from {z[k0]:.0f} m (q_t {q_parcel*1e3:.1f} g/kg), parcel minus warm parcel:")
    res = {}
    for scheme in ("none", "mixed"):
        cfg.ICE_SCHEME = scheme
        cfg.USE_NUMBA = False
        # θ_l of the parcel is conserved: start just saturated at the base (q_c = 0 → θ_l′ = θ′ = 0)
        thl = np.zeros((1, 1, z.size)); q = np.full((1, 1, z.size), q_parcel)
        # θ_l′ relative to θ0(z): the parcel keeps θ_l = θ0[k0]
        thl[0, 0, :] = theta0[k0] - theta0
        t, qv, qc, _ = th.convert_from_theta_l_qt(thl, q, np.zeros_like(q), theta0, pi3)
        res[scheme] = (t[0, 0], qc[0, 0])
    for zz in (3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0):
        k = np.searchsorted(z, zz)
        T = (res["mixed"][0][k] + theta0[k]) * pi0[k]
        dT = (res["mixed"][0][k] - res["none"][0][k]) * pi0[k]
        print(f"   z {z[k]:5.0f} m  T {T - 273.15:6.1f} °C  ΔT {dT:+5.2f} K  "
              f"q_c {res['none'][1][k]*1e3:5.2f} → {res['mixed'][1][k]*1e3:5.2f} g/kg")
    k6 = np.searchsorted(z, 6000.0)
    ok &= (res["mixed"][0][k6] - res["none"][0][k6]) > 0.5
    k3 = np.searchsorted(z, 3000.0)
    ok &= abs(res["mixed"][0][k3] - res["none"][0][k3]) < 1e-6

    cfg.ICE_SCHEME = "none"; cfg.USE_NUMBA = True
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

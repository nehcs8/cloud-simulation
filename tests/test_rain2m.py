"""
Two-moment rain (RAIN_MOMENTS = 2): checks of microphysics.rain_2m, _sediment_2m and the n_r terms of ice_graupel.

  1. numba = numpy (rain_2m warm and with graupel; sedimentation; ice_graupel with n_r).
  2. Total water conserved per cell; Δθ = −[L_v·evap + L_f (1 − α)·conv] / (c_p π); mean drop mass within bounds.
  3. Self-collection/breakup alone (saturated, no cloud): n_r falls for small drops, stays at D_eq, rises for big drops.
  4. Evaporation (sub-saturated, no cloud) of the same q_r: small drops evaporate much faster than big ones; the
     one-moment KW78 rate for comparison.
  5. Sedimentation: column mass loss = surface flux × dt; V_q > V_n (size sorting).

Run: ./venv/bin/python3 tests/test_rain2m.py
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import config as cfg
import thermodynamics as th
import microphysics as mp
from test_ice import random_state


class _S:
    pass


def state(seed=3, scheme="graupel"):
    cfg.ICE_SCHEME = scheme; cfg.RAIN_MOMENTS = 2; cfg.USE_NUMBA = False
    z, theta0, pi0, theta_l, qt = random_state(seed=seed)
    pi3 = pi0.reshape(1, 1, -1)
    rng = np.random.default_rng(seed)
    s = _S()
    s.theta, s.qv, s.qc, _ = th.convert_from_theta_l_qt(theta_l, qt, np.zeros_like(qt), theta0, pi3)
    s.qv = s.qv * rng.uniform(0.8, 1.0, qt.shape)
    s.qr = rng.uniform(0, 3e-3, qt.shape) * (rng.random(qt.shape) < 0.7)
    x = 10 ** rng.uniform(np.log10(cfg.RAIN_XMIN), np.log10(cfg.RAIN_XMAX), qt.shape)
    s.nr = np.where(s.qr > 0, s.qr / x, 0.0)
    s.qg = rng.uniform(0, 2e-3, qt.shape) * (rng.random(qt.shape) < 0.5)
    s.theta0 = theta0
    s.rho0 = cfg.P00 * pi0 ** (cfg.CP / cfg.RD - 1) / (cfg.RD * theta0)
    s._pi0 = lambda: pi3
    return s, z, pi3


def column_cell(dm, sat, qr=1e-3, T=285.0, p=85000.0):
    """1×1×1 state with rain of mean-mass diameter dm [m] and no cloud; sat = q_v / q_sw."""
    cfg.ICE_SCHEME = "none"; cfg.RAIN_MOMENTS = 2
    pi = (p / cfg.P00) ** (cfg.RD / cfg.CP)
    s = _S()
    s.theta0 = np.array([T / pi]); s.theta = np.zeros((1, 1, 1))
    s.rho0 = np.array([p / (cfg.RD * T)])
    s._pi0 = lambda: np.full((1, 1, 1), pi)
    qs = float(mp._q_sat(np.array(T), np.array(pi)))
    s.qv = np.full((1, 1, 1), sat * qs); s.qc = np.zeros((1, 1, 1))
    s.qr = np.full((1, 1, 1), qr)
    s.nr = s.qr / (np.pi / 6 * 1000.0 * dm**3)
    s.qg = np.zeros((1, 1, 1))
    return s


def main():
    ok = True
    lv_cp, lf_cp = cfg.LV / cfg.CP, cfg.LF / cfg.CP

    # 1–2. rain_2m numba vs numpy, conservation, energy, bounds
    for scheme in ("none", "graupel"):
        out = {}
        for numba in (False, True):
            s, z, pi3 = state(scheme=scheme)
            cfg.USE_NUMBA = numba
            T0 = (s.theta + s.theta0) * pi3
            a = th.liquid_fraction(T0)[0] if scheme != "none" else 1.0
            W0, th0, qc0, qv0 = s.qv + s.qc + s.qr, s.theta.copy(), s.qc.copy(), s.qv.copy()
            mp.rain_2m(s, None, 5.0)
            out[numba] = s
            dW = np.abs(s.qv + s.qc + s.qr - W0).max()
            conv, evap = qc0 - s.qc, s.qv - qv0
            lf = lf_cp if scheme != "none" else 0.0
            dE = np.abs(s.theta - th0 + (lv_cp * evap + lf * (1 - a) * conv) / pi3).max()
            r = s.qr > 1e-15
            x = s.qr[r] / s.nr[r]
            inb = (x.min() >= cfg.RAIN_XMIN * (1 - 1e-12)) and (x.max() <= cfg.RAIN_XMAX * (1 + 1e-12))
            print(f"1/2. rain_2m ICE {scheme:7s} {'numba' if numba else 'numpy'}: |Δ water| {dW:.1e}, |Δθ − expected| "
                  f"{dE:.1e} K, converted ≤ {conv.max()*1e3:.3f}, evaporated ≤ {evap.max()*1e3:.3f} g/kg, x in bounds {inb}")
            ok &= dW < 1e-15 and dE < 1e-10 and inb and conv.max() > 0 and evap.max() > 0
        d = max(np.abs(getattr(out[True], k) - getattr(out[False], k)).max() / (np.abs(getattr(out[False], k)).max() + 1e-300)
                for k in ('theta', 'qv', 'qc', 'qr', 'nr'))
        print(f"     numba vs numpy max rel diff {d:.1e}")
        ok &= d < 1e-12

    # ice_graupel with n_r: numba vs numpy, numbers stay ≥ 0
    out = {}
    for numba in (False, True):
        s, z, pi3 = state()
        cfg.USE_NUMBA = numba
        mp.ice_graupel(s, None, 5.0)
        out[numba] = s
    d = np.abs(out[True].nr - out[False].nr).max() / out[False].nr.max()
    print(f"     ice_graupel n_r numba vs numpy max rel diff {d:.1e}, min n_r {out[True].nr.min():.1e}")
    ok &= d < 1e-12 and out[True].nr.min() >= 0

    # 3. self-collection / breakup alone
    cfg.USE_NUMBA = False
    res = []
    for dm in (0.3e-3, 0.8e-3, cfg.RAIN_DEQ, 1.5e-3):
        s = column_cell(dm, sat=1.0)
        n0 = s.nr.copy()
        mp.rain_2m(s, None, 10.0)
        res.append(float(s.nr[0, 0, 0] / n0[0, 0, 0] - 1.0))
    print("3. Δn_r/n_r in 10 s (q_r 1 g/kg, saturated): " + ", ".join(
        f"D {d*1e3:.2f} mm {r*100:+.2f} %" for d, r in zip((0.3e-3, 0.8e-3, cfg.RAIN_DEQ, 1.5e-3), res)))
    ok &= res[1] < 0 and abs(res[2]) < 1e-12 and res[3] > 0

    # 4. evaporation vs drop size (and the one-moment rate)
    rates = []
    for dm in (0.2e-3, 0.5e-3, 1.0e-3, 1.5e-3):
        s = column_cell(dm, sat=0.7, qr=2e-4)
        q0 = float(s.qr[0, 0, 0])
        mp.rain_2m(s, None, 1.0)
        rates.append((q0 - float(s.qr[0, 0, 0])) / q0 * 3600.0)
    cfg.RAIN_MOMENTS = 1; cfg.RAIN_EVAP = "kw78"
    s = column_cell(1e-3, sat=0.7, qr=2e-4); q0 = float(s.qr[0, 0, 0])
    mp.warm_rain(s, None, 1.0)
    one = (q0 - float(s.qr[0, 0, 0])) / q0 * 3600.0
    cfg.RAIN_MOMENTS = 2
    print("4. evaporation of q_r 0.2 g/kg at RH 0.7, 285 K [fraction per hour]: " + ", ".join(
        f"D {d*1e3:.1f} mm {r:.1f}" for d, r in zip((0.2e-3, 0.5e-3, 1.0e-3, 1.5e-3), rates))
        + f"; one-moment KW78 {one:.1f} (its MP mean-mass D "
        f"{(6 * 2e-4 * float(s.rho0[0]) / (np.pi * 1000.0 * 8e6 / 3.0)) ** 0.25 * 1e3:.2f} mm)")
    ok &= rates[0] > rates[1] > rates[2] > rates[3] > 0

    # 5. sedimentation
    res = {}
    for numba in (False, True):
        cfg.USE_NUMBA = numba
        s, z, pi3 = state()
        g = _S(); g.terrain = None; g.nz = z.size; g.dz_face = np.full(z.size, z[1] - z[0])
        col0 = (s.qr * s.rho0 * g.dz_face).sum(axis=2)
        sfc = mp._sediment_2m(s, g, 10.0)
        col1 = (s.qr * s.rho0 * g.dz_face).sum(axis=2)
        err = np.abs(col0 - col1 - sfc / 3600.0 * 10.0).max()
        res[numba] = (s.qr.copy(), s.nr.copy())
        print(f"5. sedimentation ({'numba' if numba else 'numpy'}): column loss − flux·dt max {err:.1e} kg/m²")
        ok &= err < 1e-12
    d = max(np.abs(res[True][i] - res[False][i]).max() / res[False][i].max() for i in (0, 1))
    lam = (np.pi * 1000.0 / 5.24e-7) ** (1 / 3)
    f = lam / (lam + 600.0)
    print(f"   numba vs numpy max rel diff {d:.1e}; 1 mm mean-mass drops at the surface: V_q {9.65 - 10.3 * f**4:.2f}, "
          f"V_n {9.65 - 10.3 * f:.2f} m/s")
    ok &= d < 1e-12

    cfg.ICE_SCHEME = "none"; cfg.RAIN_MOMENTS = 1; cfg.USE_NUMBA = True
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

"""Debug the theta_l/qt conversion issue."""
import numpy as np
import sys
import config as cfg
from thermodynamics import *

# Single test case with known values
theta = 1.0  # K perturbation
qv = 0.010   # 10 g/kg
qc = 0.001   # 1 g/kg
qr = 0.0     # no rain

# Reference profiles
theta0 = 300.0  # K
pi0 = 1.0       # dimensionless

print("=" * 70)
print("DEBUGGING theta_l/qt CONVERSION")
print("=" * 70)
print()
print(f"Input state:")
print(f"  theta (perturbation) = {theta} K")
print(f"  qv    = {qv*1e3:.3f} g/kg")
print(f"  qc    = {qc*1e3:.3f} g/kg")
print(f"  theta0 (background)  = {theta0} K")
print(f"  pi0   = {pi0}")
print()

# Forward conversion
L_cp = cfg.LV / cfg.CP
print(f"L_cp = {L_cp:.2f} K/(kg/kg)")
print()

theta_l_calc = theta - L_cp * qc / pi0
qt_calc = qv + qc + qr

print(f"Forward conversion (theta, qv, qc) -> (theta_l, qt):")
print(f"  theta_l = theta - (Lv/cp)*qc/pi0")
print(f"          = {theta} - {L_cp:.2f}*{qc}/1.0")
print(f"          = {theta} - {L_cp * qc:.4f}")
print(f"          = {theta_l_calc:.4f} K")
print(f"  qt      = qv + qc + qr")
print(f"          = {qv} + {qc} + {qr}")
print(f"          = {qt_calc:.6f} kg/kg")
print()

# Backward conversion - the issue is here
print(f"Backward conversion (theta_l, qt) -> (theta, qv, qc):")
print()
print(f"Step 1: Initial guess qc = 0")
qc_guess = 0.0
print(f"  qc_guess = {qc_guess}")
print()

print(f"Step 2: Compute temperature")
print(f"  theta_abs = theta_l + theta0")
print(f"            = {theta_l_calc} + {theta0}")
theta_abs_initial = theta_l_calc + theta0
print(f"            = {theta_abs_initial} K")
T_initial = theta_abs_initial * pi0
print(f"  T = theta_abs * pi0 = {T_initial} K")
print()

# Saturation mixing ratio
T_C = T_initial - 273.15
e_s = 611.2 * np.exp(17.67 * T_C / (T_C + 243.5))
p = cfg.P00 * pi0 ** (cfg.CP / cfg.RD)
qs_initial = cfg.EPS * e_s / np.maximum(p - e_s, 1.0)
print(f"  qs(T) = {qs_initial*1e3:.3f} g/kg")
print()

print(f"Step 3: Check supersaturation")
print(f"  qt = {qt_calc*1e3:.3f} g/kg")
print(f"  qs = {qs_initial*1e3:.3f} g/kg")
print(f"  qt > qs? {qt_calc > qs_initial}")
print()

if qt_calc > qs_initial:
    print(f"Step 4: Newton iteration to find qc")
    print(f"  Need to find qc such that: qt - qc = qs(T(qc))")
    print(f"  where T depends on qc through: theta = theta_l + (Lv/cp)*qc/pi0")
    print()
    
    # Manual iteration
    for i in range(5):
        # Temperature with current qc_guess
        theta_abs_iter = theta_l_calc + L_cp * qc_guess / pi0 + theta0
        T_iter = theta_abs_iter * pi0
        
        # Saturation at this temperature
        T_C_iter = T_iter - 273.15
        e_s_iter = 611.2 * np.exp(17.67 * T_C_iter / (T_C_iter + 243.5))
        qs_iter = cfg.EPS * e_s_iter / np.maximum(p - e_s_iter, 1.0)
        
        # Residual
        residual = qt_calc - qc_guess - qs_iter
        
        print(f"  Iteration {i}:")
        print(f"    qc_guess = {qc_guess*1e3:.5f} g/kg")
        print(f"    T(qc)    = {T_iter:.3f} K")
        print(f"    qs(T)    = {qs_iter*1e3:.5f} g/kg")
        print(f"    residual = {residual*1e3:.5f} g/kg")
        
        if abs(residual) < 1e-8:
            print(f"    Converged!")
            break
        
        # Newton step
        dqsdT = qs_iter * cfg.LV / (cfg.RV * T_iter**2)
        dTdqc = L_cp * pi0
        dqsdqc = dqsdT * dTdqc
        denominator = 1.0 + dqsdqc
        delta_qc = residual / denominator
        
        print(f"    delta_qc = {delta_qc*1e3:.5f} g/kg")
        
        qc_guess = qc_guess + delta_qc
        qc_guess = np.maximum(qc_guess, 0.0)
        qc_guess = np.minimum(qc_guess, qt_calc)
        print()
    
    qc_recovered = qc_guess
    qv_recovered = qt_calc - qc_recovered
    
    print(f"Step 5: Recover theta from theta_l and qc")
    print(f"  theta = theta_l + (Lv/cp)*qc/pi0")
    print(f"        = {theta_l_calc} + {L_cp}*{qc_recovered}/1.0")
    theta_recovered = theta_l_calc + L_cp * qc_recovered / pi0
    print(f"        = {theta_recovered:.6f} K")
    print()
    
    print(f"Final recovered state:")
    print(f"  theta = {theta_recovered:.6f} K (original: {theta})")
    print(f"  qv    = {qv_recovered*1e3:.5f} g/kg (original: {qv*1e3})")
    print(f"  qc    = {qc_recovered*1e3:.5f} g/kg (original: {qc*1e3})")
    print()
    
    print(f"Errors:")
    print(f"  |theta_rec - theta| = {abs(theta_recovered - theta):.6e} K")
    print(f"  |qv_rec - qv|       = {abs(qv_recovered - qv):.6e} kg/kg")
    print(f"  |qc_rec - qc|       = {abs(qc_recovered - qc):.6e} kg/kg")


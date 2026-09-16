"""
Main time loop.

Steps per iteration:
  1. Compute dynamics tendencies (advection + SGS diffusion)
  2. Forward-Euler update
  3. Apply surface fluxes
  4. Apply microphysics (saturation adjustment, warm rain, sedimentation)
  5. Pressure projection (enforce ∇·u = 0)
  6. Apply sponge layer
  7. Enforce boundary conditions
  8. Advance time; output if due
"""

import csv
import os
import numpy as np
from scipy.ndimage import label
import config as cfg
from grid import Grid
from state import State
from dynamics import compute_tendencies
from pressure import project
from microphysics import apply_microphysics
from forcing import build_flux_map, apply_surface_fluxes, apply_subsidence, apply_profile_nudging, apply_cold_pool, apply_sponge
from visualize import Visualizer

SNAPSHOT_DIR = "snapshots"
DIAG_CSV     = "diag.csv"

_DIAG_FIELDS = [
    "t", "step", "dt",
    "w_max", "w_99",                   # updraft stats
    "qc_max", "qc_mean",               # cloud water
    "qr_max", "precip_mmhr",           # rain / surface precip
    "cloud_frac", "n_clouds",          # cloud cover and object count
    "CB", "CT",                        # cloud base / top
    "qv_bl", "qv_sfc",                 # BL and surface moisture
    "theta_bl",                        # BL mean buoyancy perturbation
    "tke",                             # domain-mean TKE
]


class Simulation:
    def __init__(self) -> None:
        self.grid  = Grid()
        self.state = State(self.grid)
        self.shf_map, self.lhf_map = build_flux_map(self.grid)
        self.t     = 0.0
        self.step  = 0
        self.viz   = Visualizer(self.grid)
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        self._csv  = open(DIAG_CSV, "w", newline="")
        self._writer = csv.DictWriter(self._csv, fieldnames=_DIAG_FIELDS)
        self._writer.writeheader()
        self._csv.flush()
        print(self.grid)
        print(f"  dt={cfg.DT}s, t_end={cfg.T_END}s, output every {cfg.OUTPUT_EVERY}s")

    # ── CFL-limited time step ─────────────────────────────────────────────────
    def _cfl_dt(self) -> float:
        s, g = self.state, self.grid
        u_max = max(
            np.abs(s.u).max(),
            np.abs(s.v).max(),
            (np.abs(s.w).max() * g.dx / g.dz_face.min()),   # scaled to dx
        ) + 1e-6
        dt_cfl = 0.8 * g.dx / u_max
        return min(cfg.DT, dt_cfl)

    # ── Single timestep ───────────────────────────────────────────────────────
    def _advance(self, dt: float) -> None:
        s, g = self.state, self.grid

        # 1. Tendencies
        tend = compute_tendencies(s, g)

        # 2. Forward-Euler update
        s.u     += dt * tend["u"]
        s.v     += dt * tend["v"]
        s.w     += dt * tend["w"]
        s.theta += dt * tend["theta"]
        s.qv    += dt * tend["qv"]
        s.qc    += dt * tend["qc"]
        s.qr    += dt * tend["qr"]

        # 2b. Clip hydrometeors — WENO-3 can undershoot near sharp gradients
        s.qc = np.maximum(s.qc, 0.0)
        s.qr = np.maximum(s.qr, 0.0)

        # 3. Surface fluxes
        apply_surface_fluxes(s, g, dt, self.shf_map, self.lhf_map)

        # 4. Microphysics
        apply_microphysics(s, g, dt)

        # 4b. Cold pool: evaporate sub-cloud rain, cool surface layer
        apply_cold_pool(s, g, dt)

        # 5. Pressure projection
        project(s.u, s.v, s.w, g, dt)

        # 6. Large-scale subsidence + profile nudging
        apply_subsidence(s, g, dt)
        apply_profile_nudging(s, g, dt)

        # 7. Sponge
        apply_sponge(s, g, dt)

        # 8. Boundary conditions (w already enforced in project)
        s.u[:, :, 0]  = 0.0   # no-slip bottom for u (optional: free-slip = s.u[:,:,1])
        s.v[:, :, 0]  = 0.0
        s.qv = np.maximum(s.qv, 0.0)
        s.qc = np.maximum(s.qc, 0.0)
        s.qr = np.maximum(s.qr, 0.0)

    # ── Run ───────────────────────────────────────────────────────────────────
    def run(self) -> None:
        next_output = 0.0
        last_dt     = cfg.DT

        try:
            while self.t < cfg.T_END:
                last_dt = self._cfl_dt()
                dt = last_dt
                # Align to output times
                if self.t + dt > next_output:
                    dt = max(next_output - self.t, 1e-6)

                self._advance(dt)
                self.t    += dt
                self.step += 1

                if self.t >= next_output - 1e-6:
                    d = self._diagnostics(last_dt)
                    self._report(d)
                    self._writer.writerow({k: f"{v:.6g}" for k, v in d.items()})
                    self._csv.flush()
                    self.viz.update(self.state, self.t)
                    self._save_snapshot(d)
                    next_output += cfg.OUTPUT_EVERY
        finally:
            self._csv.close()

    # ── Diagnostics ───────────────────────────────────────────────────────────
    def _diagnostics(self, dt: float) -> dict:
        s, g = self.state, self.grid

        # Cell-centred w
        w_cc = 0.5 * (s.w[:, :, :-1] + s.w[:, :, 1:])

        # Updraft stats
        w_abs  = np.abs(w_cc)
        w_max  = w_abs.max()
        w_99   = float(np.percentile(w_abs, 99))

        # Cloud water
        qc_max  = s.qc.max() * 1e3          # g/kg
        qc_mean = s.qc.mean() * 1e3

        # Rain
        qr_max = s.qr.max() * 1e3
        # Surface precip: sedimentation flux at k=0  (V_t * rho0 * qr → kg/m²/s → mm/hr)
        _A_VT, _B_VT = 36.34, 0.1364
        rho0_sfc = s.rho0[0]
        vt_sfc   = _A_VT * (rho0_sfc * np.maximum(s.qr[:, :, 0], 0.0)) ** _B_VT
        precip   = float((vt_sfc * s.qr[:, :, 0] * rho0_sfc).mean()) * 3600.0  # mm/hr

        # Cloud cover (column-based) and cloud object count
        cloud_col  = s.qc.max(axis=2) > 1e-5  # (nx, ny) bool
        cloud_frac = float(cloud_col.mean())
        _, n_clouds = label(cloud_col)

        # Cloud base / top (lowest/highest level with any qc)
        has_cloud = cloud_col.any()
        if has_cloud:
            any_z = s.qc.any(axis=(0, 1))        # (nz,) bool
            cb    = float(g.z[np.argmax(any_z)])
            ct    = float(g.z[len(g.z) - 1 - np.argmax(any_z[::-1])])
        else:
            cb = ct = 0.0

        # Moisture & buoyancy in the BL
        bl   = g.z < cfg.Z_BL
        qv_bl    = float(s.qv[:, :, bl].mean())
        qv_sfc   = float(s.qv[:, :, 0].mean())
        theta_bl = float(s.theta[:, :, bl].mean())

        # Domain-mean TKE
        tke = float(0.5 * (s.u**2 + s.v**2 + w_cc**2).mean())

        return dict(
            t=self.t, step=self.step, dt=dt,
            w_max=w_max, w_99=w_99,
            qc_max=qc_max, qc_mean=qc_mean,
            qr_max=qr_max, precip_mmhr=precip,
            cloud_frac=cloud_frac, n_clouds=n_clouds,
            CB=cb, CT=ct,
            qv_bl=qv_bl, qv_sfc=qv_sfc,
            theta_bl=theta_bl, tke=tke,
        )

    def _report(self, d: dict) -> None:
        print(
            f"t={d['t']:6.0f}s  step={d['step']:5d}"
            f"  w_max={d['w_max']:5.2f}m/s  w99={d['w_99']:4.2f}m/s"
            f"  qc_max={d['qc_max']:.2f}g/kg"
            f"  cov={d['cloud_frac']*100:4.1f}%  N={d['n_clouds']:3d}"
            f"  CB={d['CB']:.0f}m  CT={d['CT']:.0f}m"
            f"  prec={d['precip_mmhr']:.2f}mm/h"
            f"  qv_bl={d['qv_bl']*1e3:.2f}g/kg"
        )

    def _save_snapshot(self, d: dict) -> None:
        s, g = self.state, self.grid
        path = os.path.join(SNAPSHOT_DIR, f"snap_{self.t:07.0f}.npz")
        np.savez_compressed(
            path,
            t       = np.array(self.t),
            qc      = s.qc.astype(np.float32),
            qr      = s.qr.astype(np.float32),
            qv      = s.qv.astype(np.float32),
            w       = (0.5 * (s.w[:,:,:-1] + s.w[:,:,1:])).astype(np.float32),
            theta   = s.theta.astype(np.float32),
            z_face  = g.z_face.astype(np.float32),
            z       = g.z.astype(np.float32),
            # Mean vertical profiles (cheap summary of domain state)
            prof_qc    = s.qc.mean(axis=(0, 1)).astype(np.float32),
            prof_qr    = s.qr.mean(axis=(0, 1)).astype(np.float32),
            prof_qv    = s.qv.mean(axis=(0, 1)).astype(np.float32),
            prof_theta = s.theta.mean(axis=(0, 1)).astype(np.float32),
            prof_w     = (0.5 * (s.w[:,:,:-1] + s.w[:,:,1:])).mean(axis=(0, 1)).astype(np.float32),
        )


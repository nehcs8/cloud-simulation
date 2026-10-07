"""
Main time loop.

Steps per iteration (RK3 time integration):
  1. RK3 Stage 1: Compute tendencies, update, apply physics
  2. RK3 Stage 2: Compute tendencies, update, apply physics
  3. RK3 Stage 3: Compute tendencies, update, apply physics
  4. Advance time; output if due

Physics applied at each RK3 stage:
  - Surface fluxes
  - Microphysics (saturation adjustment, warm rain, sedimentation)
  - Pressure projection (enforce ∇·u = 0)
  - Large-scale forcing (subsidence, nudging)
  - Sponge layer
  - Boundary conditions

Time integration: Wicker-Skamarock RK3 (2002) - 3rd order, stable to CFL~1
"""

import csv
import os
import numpy as np
from scipy.ndimage import label
import config as cfg
from grid import Grid
from state import State
from terrain import Terrain
from timestepping import rk3_step
from forcing import build_flux_map
from scripts.visualize.visualize import Visualizer

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
    "u_mean", "v_mean",                # domain-mean winds
    "vort_max", "vort_cloud_mean",     # vorticity (max and in-cloud mean)
    "centroid_x", "centroid_y",        # cloud mass centroid (km)
    "centroid_speed",                  # speed of centroid movement (m/s)
]


class Simulation:
    def __init__(self) -> None:
        # ── Terrain setup ─────────────────────────────────────────────────────
        terrain = self._create_terrain()

        # ── Grid and state ────────────────────────────────────────────────────
        self.grid  = Grid(terrain=terrain)
        self.state = State(self.grid, restart_file=cfg.RESTART_FROM)
        self.shf_map, self.lhf_map = build_flux_map(self.grid)
        self.t     = 0.0
        self.step  = 0
        self.viz   = Visualizer(self.grid)

        # Track previous centroid for speed calculation
        self.prev_centroid = None
        self.prev_t = 0.0
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        self._csv  = open(DIAG_CSV, "w", newline="")
        self._writer = csv.DictWriter(self._csv, fieldnames=_DIAG_FIELDS)
        self._writer.writeheader()
        self._csv.flush()
        print(self.grid)
        if terrain is not None:
            print(f"  {terrain}")
        if cfg.RESTART_FROM is not None:
            print(f"  Restart: enabled from {cfg.RESTART_FROM}")
        print(f"  dt={cfg.DT}s, t_end={cfg.T_END}s, output every {cfg.OUTPUT_EVERY}s")

    def _create_terrain(self):
        """Create terrain based on config.TERRAIN_TYPE."""
        if cfg.TERRAIN_TYPE == "flat":
            return None

        # Create temporary grid for terrain generation
        temp_grid = Grid(terrain=None)

        if cfg.TERRAIN_TYPE == "gaussian_hill":
            terrain = Terrain.from_gaussian_hill(
                temp_grid,
                h_max=cfg.TERRAIN_H_MAX,
                sigma=cfg.TERRAIN_SIGMA
            )
            print(f"Created Gaussian hill: h_max={cfg.TERRAIN_H_MAX}m, sigma={cfg.TERRAIN_SIGMA}m")

        elif cfg.TERRAIN_TYPE == "sine_ridge":
            terrain = Terrain.from_sine_ridge(
                temp_grid,
                h_max=cfg.TERRAIN_H_MAX,
                axis='x'
            )
            print(f"Created sine ridge: h_max={cfg.TERRAIN_H_MAX}m along x-axis")

        elif cfg.TERRAIN_TYPE == "random":
            terrain = Terrain.from_random_field(
                temp_grid,
                h_mean=cfg.TERRAIN_H_MAX / 2,
                h_std=cfg.TERRAIN_H_MAX / 4,
                correlation_length=cfg.TERRAIN_SIGMA
            )
            print(f"Created random terrain: h_mean={cfg.TERRAIN_H_MAX/2}m, correlation={cfg.TERRAIN_SIGMA}m")

        elif cfg.TERRAIN_TYPE == "dem":
            if cfg.TERRAIN_FILE is None:
                raise ValueError("TERRAIN_TYPE='dem' but TERRAIN_FILE is not set")
            terrain = Terrain.from_dem_file(temp_grid, cfg.TERRAIN_FILE)
            print(f"Loaded DEM from {cfg.TERRAIN_FILE}")

        else:
            raise ValueError(
                f"Unknown TERRAIN_TYPE='{cfg.TERRAIN_TYPE}'. "
                f"Valid options: 'flat', 'gaussian_hill', 'sine_ridge', 'random', 'dem'"
            )

        return terrain

    # ── CFL-limited time step ─────────────────────────────────────────────────
    def _cfl_dt(self) -> float:
        """
        Compute CFL and diffusion-limited timestep.

        With RK3, we can use CFL ~ 1.0 (vs 0.5 for forward Euler).
        See LES_REBUILD_PLAN.md line 244: "Update CFL condition (can use larger dt with RK3)"

        Also checks diffusive stability: K·Δt/Δz² < 0.5 for explicit diffusion.
        """
        s, g = self.state, self.grid

        # Advective CFL limit
        u_max = max(
            np.abs(s.u).max(),
            np.abs(s.v).max(),
            (np.abs(s.w).max() * g.dx / g.dz_face.min()),   # scaled to dx
        ) + 1e-6
        # RK3 allows CFL ~ 1.0 (more stable than forward Euler's 0.5)
        dt_cfl = 1.0 * g.dx / u_max

        # Diffusive stability limit (explicit diffusion: K·dt/dx² < 0.5)
        # Estimate max K from Smagorinsky: K_M ~ (C_s Δ)² |S|
        # Conservative estimate: use domain max of resolved strain rate
        from dynamics import _smag_K
        K_M, K_H = _smag_K(s, g)
        K_max = max(K_M.max(), K_H.max()) + 1e-10
        dx_min = min(g.dx, g.dy, g.dz_face.min())
        dt_diff = 0.25 * dx_min**2 / K_max  # Factor 0.25 for safety (3D diffusion)

        return min(cfg.DT, dt_cfl, dt_diff)

    # ── Single timestep ───────────────────────────────────────────────────────
    def _advance(self, dt: float) -> None:
        """
        Advance state by one timestep using RK3 time integration.

        All physics (microphysics, forcing, projection, BCs) are applied
        within the RK3 stages in timestepping.rk3_step().
        """
        rk3_step(self.state, self.grid, dt, self.shf_map, self.lhf_map)

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

        # Domain-mean TKE (use cell-centered velocities)
        u_cc = s.u_at_cell()
        v_cc = s.v_at_cell()
        tke = float(0.5 * (u_cc**2 + v_cc**2 + w_cc**2).mean())

        # Domain-mean winds (use cell-centered velocities)
        u_mean = float(u_cc.mean())
        v_mean = float(v_cc.mean())

        # Vertical vorticity: ζ = ∂v/∂x - ∂u/∂y (at cell centers)
        dvdx = np.gradient(v_cc, g.dx, axis=0)
        dudy = np.gradient(u_cc, g.dy, axis=1)
        vort = dvdx - dudy
        vort_max = float(np.abs(vort).max())

        # Mean vorticity in cloudy regions (qc > 0.1 g/kg)
        cloud_mask_3d = s.qc > 1e-4
        if cloud_mask_3d.any():
            vort_cloud_mean = float(np.abs(vort[cloud_mask_3d]).mean())
        else:
            vort_cloud_mean = 0.0

        # Cloud centroid (mass-weighted center of qc)
        if cloud_mask_3d.any():
            x_grid, y_grid = np.meshgrid(np.arange(g.nx) * g.dx, np.arange(g.ny) * g.dy, indexing='ij')
            x_grid_3d = np.broadcast_to(x_grid[:,:,np.newaxis], s.qc.shape)
            y_grid_3d = np.broadcast_to(y_grid[:,:,np.newaxis], s.qc.shape)

            total_qc = s.qc.sum()
            centroid_x = float((s.qc * x_grid_3d).sum() / total_qc / 1000.0)  # km
            centroid_y = float((s.qc * y_grid_3d).sum() / total_qc / 1000.0)  # km

            # Calculate centroid speed (m/s)
            if self.prev_centroid is not None:
                dx_cent = (centroid_x - self.prev_centroid[0]) * 1000.0  # m
                dy_cent = (centroid_y - self.prev_centroid[1]) * 1000.0  # m
                dt_cent = self.t - self.prev_t
                centroid_speed = float(np.sqrt(dx_cent**2 + dy_cent**2) / dt_cent) if dt_cent > 0 else 0.0
            else:
                centroid_speed = 0.0

            self.prev_centroid = (centroid_x, centroid_y)
            self.prev_t = self.t
        else:
            centroid_x = centroid_y = centroid_speed = 0.0

        return dict(
            t=self.t, step=self.step, dt=dt,
            w_max=w_max, w_99=w_99,
            qc_max=qc_max, qc_mean=qc_mean,
            qr_max=qr_max, precip_mmhr=precip,
            cloud_frac=cloud_frac, n_clouds=n_clouds,
            CB=cb, CT=ct,
            qv_bl=qv_bl, qv_sfc=qv_sfc,
            theta_bl=theta_bl, tke=tke,
            u_mean=u_mean, v_mean=v_mean,
            vort_max=vort_max, vort_cloud_mean=vort_cloud_mean,
            centroid_x=centroid_x, centroid_y=centroid_y,
            centroid_speed=centroid_speed,
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

        # Compute 2D vorticity field at mid-cloud height (z ~ 1500m) for visualization
        # Use cell-centered velocities for vorticity
        u_cc = s.u_at_cell()
        v_cc = s.v_at_cell()
        z_mid_idx = np.argmin(np.abs(g.z - 1500.0))
        dvdx = np.gradient(v_cc, g.dx, axis=0)
        dudy = np.gradient(u_cc, g.dy, axis=1)
        vort = dvdx - dudy
        vort_2d_midcloud = vort[:, :, z_mid_idx].astype(np.float32)

        np.savez_compressed(
            path,
            t       = np.array(self.t),
            u       = u_cc.astype(np.float32),   # Save cell-centered for compatibility
            v       = v_cc.astype(np.float32),
            qc      = s.qc.astype(np.float32),
            qr      = s.qr.astype(np.float32),
            qv      = s.qv.astype(np.float32),
            w       = (0.5 * (s.w[:,:,:-1] + s.w[:,:,1:])).astype(np.float32),
            theta   = s.theta.astype(np.float32),
            z_face  = g.z_face.astype(np.float32),
            z       = g.z.astype(np.float32),
            # 2D vorticity field at mid-cloud level for rotation analysis
            vort_2d = vort_2d_midcloud,
            vort_z_level = np.array(g.z[z_mid_idx]),
            # Mean vertical profiles (cheap summary of domain state)
            prof_u     = u_cc.mean(axis=(0, 1)).astype(np.float32),
            prof_v     = v_cc.mean(axis=(0, 1)).astype(np.float32),
            prof_qc    = s.qc.mean(axis=(0, 1)).astype(np.float32),
            prof_qr    = s.qr.mean(axis=(0, 1)).astype(np.float32),
            prof_qv    = s.qv.mean(axis=(0, 1)).astype(np.float32),
            prof_theta = s.theta.mean(axis=(0, 1)).astype(np.float32),
            prof_w     = (0.5 * (s.w[:,:,:-1] + s.w[:,:,1:])).mean(axis=(0, 1)).astype(np.float32),
        )


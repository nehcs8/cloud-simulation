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
import sys
import shutil
import numpy as np
from scipy.ndimage import label
from datetime import datetime
import config as cfg
from grid import Grid
from state import State
from terrain import Terrain
from timestepping import rk3_step
from forcing import build_flux_map, FluxPattern
import cases
import open_bc
from scripts.visualize.visualize import Visualizer

# Create timestamped output directory with case name
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
OUTPUT_DIR = f"runs/{RUN_TIMESTAMP}_{cfg.CASE_NAME}"
SNAPSHOT_DIR = f"{OUTPUT_DIR}/snapshots"
DIAG_CSV     = f"{OUTPUT_DIR}/diag.csv"
LOG_FILE     = f"{OUTPUT_DIR}/console.log"
CONFIG_COPY  = f"{OUTPUT_DIR}/config.py"
PROFILES_NPZ = f"{OUTPUT_DIR}/profiles.npz"

_DIAG_FIELDS = [
    "t", "step", "dt",
    "w_max", "w_99",                   # updraft stats
    "qc_max", "qc_mean",               # cloud water
    "qr_max", "precip_mmhr",           # rain / surface precip
    "cloud_frac", "n_clouds",          # cloud cover and object count
    "cloud_persistence",               # correlation with cloud mask 20 min ago
    "CB", "CT",                        # cloud base / top
    "LCL", "BL_height",                # lifting condensation level and BL top
    "qv_bl", "qv_sfc",                 # BL and surface moisture
    "theta_bl",                        # BL mean buoyancy perturbation
    "tke",                             # domain-mean TKE (from horizontal-mean deviations)
    "u_mean", "v_mean",                # domain-mean winds
    "vort_max", "vort_cloud_mean",     # vorticity (max and in-cloud mean)
    "centroid_x", "centroid_y",        # cloud mass centroid (km)
    "centroid_speed",                  # speed of centroid movement (m/s)
    "qg_max",                          # graupel/snow (ICE_SCHEME "graupel"; 0 otherwise), g/kg
    "Dr_mm",                           # q_r-weighted mean-mass drop diameter where q_r > 0.01 g/kg (RAIN_MOMENTS 2), mm
]


class Simulation:
    def __init__(self) -> None:
        # ── Case-specific configuration (must precede grid construction) ──────
        case = cases.get(cfg.CASE)
        if case is not None:
            case.configure()
        if cfg.LATERAL_BC == "fringe" and open_bc.open_x():
            raise ValueError('LATERAL_BC = "fringe" needs BC_X = "periodic" (the fringe recycles a periodic x)')

        # ── Terrain setup ─────────────────────────────────────────────────────
        terrain = self._create_terrain()

        # ── Grid and state ────────────────────────────────────────────────────
        self.grid  = Grid(terrain=terrain)
        self.state = State(self.grid, restart_file=cfg.RESTART_FROM)
        self.shf_map, self.lhf_map = build_flux_map(self.grid)
        self.flux_pattern = FluxPattern(self.grid) if cfg.ENABLE_FLUX_PATTERN else None
        self.t     = 0.0
        self.step  = 0
        self.viz   = Visualizer(self.grid)

        # Track previous centroid for speed calculation
        self.prev_centroid = None
        self.prev_t = 0.0

        # Cloud persistence tracking (rolling buffer for 20 min correlation)
        self.cloud_mask_history = []  # List of (time, mask) tuples
        self.cloud_persistence_window = 1200.0  # 20 minutes in seconds

        # Profile storage (saved every PROFILE_EVERY)
        self.micro_times, self.micro_rows = [], []
        self.profile_times = []
        self.profile_data = {
            'theta': [],
            'qv': [],
            'qc': [],
            'RH': [],
            'u': [],
            'v': [],
            'w_var': [],  # w'^2
        }

        # Create timestamped output directories (never overwrite)
        os.makedirs(OUTPUT_DIR, exist_ok=False)  # Fail if directory exists
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)

        # Copy config.py to run directory for reproducibility
        import config
        config_source = config.__file__
        shutil.copy2(config_source, CONFIG_COPY)
        # Effective values after case/run-script overrides (config.py copy misses these)
        with open(os.path.join(OUTPUT_DIR, "effective_config.txt"), "w") as f:
            for name in sorted(n for n in dir(cfg) if n.isupper()):
                f.write(f"{name} = {getattr(cfg, name)!r}\n")

        # Redirect stdout to both console and log file
        self._setup_logging()

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
        print(f"  Snapshot interval: {cfg.SNAPSHOT_EVERY}s")
        if cfg.LATERAL_BC == "fringe":
            print(f"  Lateral BC x: fringe {cfg.FRINGE_WIDTH:.0f} m (tau {cfg.FRINGE_TAU:.0f} s), "
                  f"recycling at x = {cfg.RECYCLE_X:.0f} m, flux patterns in "
                  f"[{cfg.RECYCLE_X + cfg.HETERO_BUFFER:.0f}, "
                  f"{self.grid.nx * self.grid.dx - cfg.FRINGE_WIDTH:.0f}] m")
        if open_bc.any_open():
            print(f"  Lateral BCs: x {cfg.BC_X}, y {cfg.BC_Y} (c* = {cfg.OPEN_C_STAR:.0f} m/s, inflow "
                  f"relaxation {cfg.OPEN_RELAX_WIDTH} cells / {cfg.OPEN_RELAX_TAU:.0f} s)")
        print(f"  NaN guard: {'enabled' if cfg.ENABLE_NAN_GUARD else 'disabled'}")
        print(f"  Output directory: {OUTPUT_DIR}")

    def _setup_logging(self):
        """Redirect stdout to both console and log file."""
        class TeeLogger:
            def __init__(self, *files):
                self.files = files
            def write(self, data):
                for f in self.files:
                    f.write(data)
                    f.flush()
            def flush(self):
                for f in self.files:
                    f.flush()

        self._log_file = open(LOG_FILE, "w")
        self._original_stdout = sys.stdout
        sys.stdout = TeeLogger(sys.stdout, self._log_file)

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

        elif cfg.TERRAIN_TYPE == "agnesi_ridge":
            terrain = Terrain.from_agnesi_ridge(temp_grid, h_max=cfg.TERRAIN_H_MAX,
                                                half_width=cfg.TERRAIN_SIGMA)
            print(f"Created Witch-of-Agnesi ridge: h_max={cfg.TERRAIN_H_MAX}m, a={cfg.TERRAIN_SIGMA}m")

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
        if cfg.CFL_SUM:
            # Pointwise 3-D Courant rate |u|/dx + |v|/dy + |w|/dz at cell centres
            rate = (np.abs(s.u_at_cell()) / g.dx + np.abs(s.v_at_cell()) / g.dy
                    + np.abs(s.w_at_cell()) / g.dz_face[np.newaxis, np.newaxis, :])
            dt_cfl = cfg.CFL_MAX / (rate.max() + 1e-12)
        else:
            u_max = max(
                np.abs(s.u).max(),
                np.abs(s.v).max(),
                (np.abs(s.w).max() * g.dx / g.dz_face.min()),   # scaled to dx
            ) + 1e-6
            # RK3 allows CFL ~ 1.0 (more stable than forward Euler's 0.5)
            dt_cfl = cfg.CFL_MAX * g.dx / u_max

        # Diffusive stability limit (explicit diffusion: K·dt/dx² < 0.5)
        # Estimate max K from Smagorinsky: K_M ~ (C_s Δ)² |S|
        # Conservative estimate: use domain max of resolved strain rate
        from dynamics import _smag_K
        K_M, K_H = _smag_K(s, g)
        K_max = max(K_M.max(), K_H.max()) + 1e-10
        dx_min = min(g.dx, g.dy, g.dz_face.min())
        dt_diff = 0.25 * dx_min**2 / K_max  # Factor 0.25 for safety (3D diffusion)

        # Open lateral boundaries: radiation term (u_n ± c*) must satisfy the same Courant limit
        if open_bc.any_open():
            u_b = max([np.abs(s.u[0]).max(), np.abs(s.u[-1]).max()] if open_bc.open_x() else [0.0])
            v_b = max([np.abs(s.v[:, 0]).max(), np.abs(s.v[:, -1]).max()] if open_bc.open_y() else [0.0])
            dt_cfl = min(dt_cfl, cfg.CFL_MAX * min(g.dx, g.dy) / (cfg.OPEN_C_STAR + max(u_b, v_b)))

        return min(cfg.DT, dt_cfl, dt_diff)

    # ── Single timestep ───────────────────────────────────────────────────────
    def _advance(self, dt: float) -> None:
        """
        Advance state by one timestep using RK3 time integration.

        All physics (microphysics, forcing, projection, BCs) are applied
        within the RK3 stages in timestepping.rk3_step().
        """
        if self.flux_pattern is not None:
            self.flux_pattern.update(self.t)
            self.shf_map, self.lhf_map = self.flux_pattern.maps()
        rk3_step(self.state, self.grid, dt, self.shf_map, self.lhf_map, t=self.t)

    def _check_for_nans(self) -> bool:
        """
        Check if any prognostic field contains NaN or Inf.

        Returns True if simulation is healthy, False if NaN/Inf detected.
        """
        if not cfg.ENABLE_NAN_GUARD:
            return True

        s = self.state
        fields_to_check = [
            ('u', s.u),
            ('v', s.v),
            ('w', s.w),
            ('theta_l', s.theta_l),
            ('qt', s.qt),
            ('qr', s.qr),
            ('qg', s.qg),
            ('nr', s.nr),
        ]

        for name, field in fields_to_check:
            if not np.isfinite(field).all():
                # Find location of max |w| for diagnostic
                w_cc = 0.5 * (s.w[:, :, :-1] + s.w[:, :, 1:])
                w_abs = np.abs(w_cc)
                max_idx = np.unravel_index(np.argmax(w_abs), w_abs.shape)
                i, j, k = max_idx
                z_max = self.grid.z[k]

                print(f"\n{'='*70}")
                print(f"ERROR: NaN/Inf detected in field '{name}' at step {self.step}, t={self.t:.1f}s")
                print(f"Location of max |w|: i={i}, j={j}, k={k} (z={z_max:.1f}m)")
                print(f"Max |w| = {w_abs[max_idx]:.3f} m/s")
                print(f"{'='*70}\n")

                # Save crash state
                self._save_crash_state()
                return False

        return True

    def _save_crash_state(self):
        """Save full state to crash_state.npz for post-mortem debugging."""
        crash_path = os.path.join(OUTPUT_DIR, "crash_state.npz")
        s, g = self.state, self.grid

        print(f"Saving crash state to: {crash_path}")

        np.savez_compressed(
            crash_path,
            # Time info
            t=np.array(self.t),
            step=np.array(self.step),
            # Prognostic fields (conserved)
            u=s.u,
            v=s.v,
            w=s.w,
            theta_l=s.theta_l,
            qt=s.qt,
            qr=s.qr,
            qg=s.qg,
            nr=s.nr,
            # Diagnostic fields
            theta=s.theta,
            qv=s.qv,
            qc=s.qc,
            # Reference profiles
            theta0=s.theta0,
            qv0=s.qv0,
            rho0=s.rho0,
            # Grid
            z=g.z,
            z_face=g.z_face,
            dx=np.array(g.dx),
            dy=np.array(g.dy),
        )
        print(f"Crash state saved. Simulation halted.")

    # ── Run ───────────────────────────────────────────────────────────────────
    def run(self) -> None:
        next_output = 0.0
        next_snapshot = 0.0
        next_profile = 0.0
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

                # NaN/Inf guard: check after each timestep
                if not self._check_for_nans():
                    print("Simulation halted due to NaN/Inf detection.")
                    break

                # Regular diagnostic output
                if self.t >= next_output - 1e-6:
                    d = self._diagnostics(last_dt)
                    self._report(d)
                    self._writer.writerow({k: f"{v:.6g}" for k, v in d.items()})
                    self._csv.flush()
                    self.viz.update(self.state, self.t)
                    next_output += cfg.OUTPUT_EVERY

                # Snapshot output (separate interval from diagnostics)
                if self.t >= next_snapshot - 1e-6:
                    # Compute diagnostics if not already done in this timestep
                    if abs(self.t - (next_output - cfg.OUTPUT_EVERY)) > 1e-3:
                        d = self._diagnostics(last_dt)
                    # else: diagnostics were just computed above, reuse them
                    self._save_snapshot(d)
                    next_snapshot += cfg.SNAPSHOT_EVERY

                # Profile output
                if self.t >= next_profile - 1e-6:
                    self._save_profile()
                    next_profile += cfg.PROFILE_EVERY
        finally:
            # Write profiles to file
            self._write_profiles()
            self._csv.close()
            if hasattr(self, '_log_file'):
                self._log_file.close()
            # Restore original stdout
            if hasattr(self, '_original_stdout'):
                sys.stdout = self._original_stdout

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
        # Surface precip: use stored flux from sedimentation (already in mm/hr)
        precip = float(s.surface_precip_flux.mean())

        # Cloud cover (column-based) and cloud object count
        cloud_col  = s.qc.max(axis=2) > 1e-5  # (nx, ny) bool
        cloud_frac = float(cloud_col.mean())
        _, n_clouds = label(cloud_col)

        # Cloud persistence: correlation with cloud mask 20 min ago
        # Update history buffer
        self.cloud_mask_history.append((self.t, cloud_col.copy()))
        # Remove old entries (older than 20 min)
        self.cloud_mask_history = [(t, mask) for t, mask in self.cloud_mask_history
                                   if self.t - t <= self.cloud_persistence_window]
        # Find mask closest to 20 min ago
        target_t = self.t - self.cloud_persistence_window
        if len(self.cloud_mask_history) > 1:
            # Find entry closest to target time
            times = np.array([t for t, _ in self.cloud_mask_history])
            idx = np.argmin(np.abs(times - target_t))
            old_mask = self.cloud_mask_history[idx][1]
            # Compute correlation (Pearson correlation of flattened masks)
            m1 = cloud_col.flatten().astype(float)
            m2 = old_mask.flatten().astype(float)
            if m1.std() > 0 and m2.std() > 0:
                cloud_persistence = float(np.corrcoef(m1, m2)[0, 1])
            else:
                cloud_persistence = 0.0
        else:
            cloud_persistence = 0.0  # Not enough history yet

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

        # Lifting Condensation Level (LCL) from lowest 200m mean
        lowest_200m = g.z < 200.0
        theta_200m = (s.theta[:, :, lowest_200m] + s.theta0[np.newaxis, np.newaxis, lowest_200m]).mean()
        qv_200m = s.qv[:, :, lowest_200m].mean()
        # LCL from dewpoint depression: z_LCL ≈ z_parcel + 125 m/K · (T − T_d).
        # T_d inverts the same Magnus/Buck e_s(T) used by the model's q_sat.
        pi0_200m = s._pi0()[0, 0, lowest_200m].mean()
        T_200m = theta_200m * pi0_200m
        p_200m = cfg.P00 * pi0_200m ** (cfg.CP / cfg.RD)
        e_200m = max(qv_200m * p_200m / (cfg.EPS + qv_200m), 1e-3)   # Pa
        ln_e = np.log(e_200m / 611.2)
        Td_200m = 273.15 + 243.5 * ln_e / (17.67 - ln_e)
        lcl = float(g.z[lowest_200m].mean() + 125.0 * max(T_200m - Td_200m, 0.0))

        # Boundary layer height: level of maximum dθ/dz (inversion strength)
        theta_full = s.theta + s.theta0[np.newaxis, np.newaxis, :]  # (nx, ny, nz)
        theta_prof = theta_full.mean(axis=(0, 1))  # (nz,)
        # Compute vertical gradient dθ/dz at cell centers
        dtheta_dz = np.gradient(theta_prof, g.z)
        # Find level of maximum gradient (inversion)
        k_bl = np.argmax(dtheta_dz)
        bl_height = float(g.z[k_bl])

        # TKE from deviations from horizontal mean at each level
        u_cc = s.u_at_cell()
        v_cc = s.v_at_cell()
        # Horizontal mean at each level
        u_mean_z = u_cc.mean(axis=(0, 1))  # (nz,)
        v_mean_z = v_cc.mean(axis=(0, 1))
        w_mean_z = w_cc.mean(axis=(0, 1))
        # Deviations (broadcast back to 3D)
        u_prime = u_cc - u_mean_z[np.newaxis, np.newaxis, :]
        v_prime = v_cc - v_mean_z[np.newaxis, np.newaxis, :]
        w_prime = w_cc - w_mean_z[np.newaxis, np.newaxis, :]
        # TKE = 0.5 * <u'^2 + v'^2 + w'^2>
        tke = float(0.5 * (u_prime**2 + v_prime**2 + w_prime**2).mean())

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
            cloud_persistence=cloud_persistence,
            CB=cb, CT=ct,
            LCL=lcl, BL_height=bl_height,
            qv_bl=qv_bl, qv_sfc=qv_sfc,
            theta_bl=theta_bl, tke=tke,
            u_mean=u_mean, v_mean=v_mean,
            vort_max=vort_max, vort_cloud_mean=vort_cloud_mean,
            centroid_x=centroid_x, centroid_y=centroid_y,
            centroid_speed=centroid_speed,
            qg_max=s.qg.max() * 1e3,
            Dr_mm=self._drop_diameter_mm(),
        )

    def _drop_diameter_mm(self) -> float:
        s = self.state
        m = (s.qr > 1e-5) & (s.nr > 0)
        if not m.any():
            return 0.0
        d = (6.0 * s.qr[m] / (np.pi * 1000.0 * s.nr[m])) ** (1.0 / 3.0)
        return float((d * s.qr[m]).sum() / s.qr[m].sum() * 1e3)

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
            qg      = s.qg.astype(np.float32),
            nr      = s.nr.astype(np.float32),
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
            prof_qg    = s.qg.mean(axis=(0, 1)).astype(np.float32),
            prof_qv    = s.qv.mean(axis=(0, 1)).astype(np.float32),
            prof_theta = s.theta.mean(axis=(0, 1)).astype(np.float32),
            prof_w     = (0.5 * (s.w[:,:,:-1] + s.w[:,:,1:])).mean(axis=(0, 1)).astype(np.float32),
        )

    def _save_profile(self) -> None:
        """Save mean vertical profiles for this timestep."""
        s, g = self.state, self.grid

        # Horizontal mean profiles
        u_cc = s.u_at_cell()
        v_cc = s.v_at_cell()
        w_cc = 0.5 * (s.w[:, :, :-1] + s.w[:, :, 1:])

        theta_full = s.theta + s.theta0[np.newaxis, np.newaxis, :]
        theta_prof = theta_full.mean(axis=(0, 1))
        qv_prof = s.qv.mean(axis=(0, 1))
        qc_prof = s.qc.mean(axis=(0, 1))

        # Relative humidity
        from microphysics import _q_sat
        pi0_prof = s._pi0()[0, 0, :]  # (nz,)
        T_prof = theta_prof * pi0_prof
        qs_prof = _q_sat(T_prof, pi0_prof)
        RH_prof = qv_prof / np.maximum(qs_prof, 1e-10)

        # Velocity profiles
        u_prof = u_cc.mean(axis=(0, 1))
        v_prof = v_cc.mean(axis=(0, 1))

        # w variance: w'^2 where w' = w - <w>_horizontal
        w_mean_z = w_cc.mean(axis=(0, 1))
        w_prime = w_cc - w_mean_z[np.newaxis, np.newaxis, :]
        w_var_prof = (w_prime**2).mean(axis=(0, 1))

        # Store in buffer
        self.profile_times.append(self.t)
        self.profile_data['theta'].append(theta_prof.astype(np.float32))
        self.profile_data['qv'].append(qv_prof.astype(np.float32))
        self.profile_data['qc'].append(qc_prof.astype(np.float32))
        self.profile_data['RH'].append(RH_prof.astype(np.float32))
        self.profile_data['u'].append(u_prof.astype(np.float32))
        self.profile_data['v'].append(v_prof.astype(np.float32))
        self.profile_data['w_var'].append(w_var_prof.astype(np.float32))
        self._write_micro_budget()

    def _write_micro_budget(self) -> None:
        """Cumulative microphysics process budget [kg per level] at each profile time → micro_budget.npz
        (rewritten every time, so it survives a crash). Processes: microphysics.PROCESSES."""
        import microphysics
        self.micro_times.append(self.t)
        self.micro_rows.append(microphysics.budget_kg(self.state, self.grid))
        np.savez_compressed(f"{OUTPUT_DIR}/micro_budget.npz", times=np.array(self.micro_times),
                            z=self.grid.z, processes=np.array(microphysics.PROCESSES),
                            cumulative_kg=np.array(self.micro_rows))       # (n_times, nz, n_proc)

    def _write_profiles(self) -> None:
        """Write collected profiles to profiles.npz."""
        if len(self.profile_times) == 0:
            return  # No profiles to write

        # Convert lists to arrays
        times = np.array(self.profile_times)
        profiles = {
            'times': times,
            'z': self.grid.z,
            'theta': np.array(self.profile_data['theta']),  # (n_times, nz)
            'qv': np.array(self.profile_data['qv']),
            'qc': np.array(self.profile_data['qc']),
            'RH': np.array(self.profile_data['RH']),
            'u': np.array(self.profile_data['u']),
            'v': np.array(self.profile_data['v']),
            'w_var': np.array(self.profile_data['w_var']),
        }

        np.savez_compressed(PROFILES_NPZ, **profiles)
        print(f"Saved {len(times)} profiles to {PROFILES_NPZ}")


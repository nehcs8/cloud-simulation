"""Model state: fields, reference profiles, and initial conditions."""

import numpy as np
from scipy.ndimage import gaussian_filter
import config as cfg
from grid import Grid
import cases


class State:
    """
    Prognostic fields (all shape nx × ny × nz unless noted):
        u, v   – horizontal wind  [m/s]
        w      – vertical wind    [m/s]  shape nx × ny × (nz+1), face-staggered in z

        theta_l – liquid water potential temperature θ_l  [K]  (Phase 1.4)
        qt      – total water mixing ratio qt [kg/kg]           (Phase 1.4)
        qr      – rain-water mixing ratio [kg/kg]

        theta  – potential temperature perturbation θ'  [K]     (diagnostic)
        qv     – water-vapour mixing ratio  [kg/kg]              (diagnostic)
        qc     – cloud-water mixing ratio   [kg/kg]              (diagnostic)

    Reference profiles (1-D, length nz):
        theta0 – background θ₀(z)           [K]
        qv0    – background qᵥ₀(z)          [kg/kg]
        rho0   – background density ρ₀(z)   [kg/m³]  (Boussinesq reference)
        pi0    – Exner function π₀(z)       [-]
    """

    def __init__(self, grid: Grid, restart_file: str = None) -> None:
        self.grid = grid
        g = grid

        shape   = (g.nx, g.ny, g.nz)
        shape_u = (g.nx + 1, g.ny, g.nz)  # face-staggered in x (Arakawa C-grid)
        shape_v = (g.nx, g.ny + 1, g.nz)  # face-staggered in y
        shape_w = (g.nx, g.ny, g.nz + 1)  # face-staggered in z

        # ── Prognostic fields ─────────────────────────────────────────────────
        # Arakawa C-grid staggering:
        # u: on x-faces (i+1/2, j, k)
        # v: on y-faces (i, j+1/2, k)
        # w: on z-faces (i, j, k+1/2)
        # scalars: at cell centers (i, j, k)
        self.u     = np.zeros(shape_u)
        self.v     = np.zeros(shape_v)
        self.w     = np.zeros(shape_w)

        # Phase 1.4: Moist conserved variables (prognostic)
        self.theta_l = np.zeros(shape)    # Liquid water potential temperature
        self.qt      = np.zeros(shape)    # Total water mixing ratio
        self.qr      = np.zeros(shape)    # Rain water

        # Diagnostic fields (derived from theta_l, qt)
        self.theta = np.zeros(shape)      # θ′  (perturbation) - diagnostic
        self.qv    = np.zeros(shape)      # Water vapor - diagnostic
        self.qc    = np.zeros(shape)      # Cloud water - diagnostic

        # Surface fluxes (diagnostics updated by microphysics)
        self.surface_precip_flux = np.zeros((g.nx, g.ny))  # mm/hr, updated by sedimentation

        # ── Reference profiles ────────────────────────────────────────────────
        self.theta0, self.qv0, self.rho0 = self._build_reference(g.z)

        # Geostrophic wind u_g(z) (nz,): profile if configured, else uniform U_GEO
        if cfg.U_GEO_PROFILE is not None:
            self.u_g = np.asarray(cfg.U_GEO_PROFILE(g.z), dtype=float)
        else:
            self.u_g = np.full(g.nz, cfg.U_GEO)

        # ── Apply initial conditions ──────────────────────────────────────────
        if restart_file is not None:
            self._load_from_snapshot(restart_file, g)
            # Snapshot already has diagnostic variables, just update conserved ones
            self._update_conserved_variables()
        elif cases.get(cfg.CASE) is not None:
            case = cases.get(cfg.CASE)
            case.init_perturbations(self, g)
            self.u[:] = self.u_g[np.newaxis, np.newaxis, :]
            self.v[:] = cfg.V_GEO
            self._update_conserved_variables()
            self._update_diagnostic_variables()
            case.print_rh_profile(self, g)
        else:
            self._init_random_bl(g)
            # Initialize background wind (uniform in space)
            self.u[:] = self.u_g[np.newaxis, np.newaxis, :]
            self.v[:] = cfg.V_GEO
            # Convert to θ_l/qt framework (Phase 1.4)
            # After initialization, convert (θ, qv, qc, qr) → (θ_l, qt, qr)
            self._update_conserved_variables()
            # Then update diagnostics
            self._update_diagnostic_variables()

    # ── Reference atmosphere ──────────────────────────────────────────────────
    def _build_reference(self, z: np.ndarray):
        """
        θ₀(z) with well-mixed BL and capping inversion:
        - Well-mixed BL (constant θ) below Z_BL
        - Smooth inversion transition around Z_INVERSION
        - FT lapse rate above inversion

        qv₀: well-mixed in BL, then exponential decay above.
        ρ₀: from hydrostatic + ideal gas (approximately constant for Boussinesq).
        """
        case = cases.get(cfg.CASE)
        if case is not None:
            theta0, qv0 = case.reference_profiles(z)
            pi0 = case.exner_profile(theta0, z)
            rho0 = cfg.P00 * pi0 ** (cfg.CP / cfg.RD) / (cfg.RD * theta0 * pi0)
            return theta0, qv0, rho0

        z_bl = cfg.Z_BL
        gamma_bl = cfg.GAMMA_BL * 1e-3   # K/m (should be 0.0 for well-mixed)
        gamma_ft = cfg.GAMMA_FT * 1e-3   # K/m

        # Base profile without inversion
        # Well-mixed BL: constant theta up to Z_BL
        theta0_base = np.where(
            z <= z_bl,
            cfg.THETA_S + gamma_bl * z,  # If gamma_bl=0, this is constant
            cfg.THETA_S + gamma_bl * z_bl + gamma_ft * (z - z_bl),
        )

        # Add smooth inversion jump using tanh
        # This creates a ~3K jump over ~200m centered at Z_INVERSION
        if hasattr(cfg, 'Z_INVERSION') and hasattr(cfg, 'INVERSION_STRENGTH'):
            z_inv = cfg.Z_INVERSION
            delta_theta = cfg.INVERSION_STRENGTH
            width = cfg.INVERSION_WIDTH
            # tanh goes from -1 to +1, so 0.5*(1+tanh) goes from 0 to 1
            inversion_profile = 0.5 * delta_theta * (1.0 + np.tanh((z - z_inv) / width))
            theta0 = theta0_base + inversion_profile
        else:
            theta0 = theta0_base

        # Well-mixed BL moisture: constant below Z_BL, then exponential decay
        # This creates ~14.5 g/kg in BL, then drops to ~40-50% RH above
        qv0 = np.where(
            z <= z_bl,
            cfg.QV_SURF,  # Constant in well-mixed BL
            cfg.QV_SURF * np.exp(-(z - z_bl) / cfg.QV_SCALE)  # Decay above BL
        )

        # Exner function π₀ = (p₀/p₀₀)^(Rd/cp) via hydrostatic balance
        # dπ/dz = -g / (cp * θ₀)
        pi0 = np.ones(len(z))
        for k in range(1, len(z)):
            dz = z[k] - z[k - 1]
            theta_mid = 0.5 * (theta0[k] + theta0[k - 1])
            pi0[k] = pi0[k - 1] - cfg.G * dz / (cfg.CP * theta_mid)

        rho0 = cfg.P00 * pi0 ** (cfg.CP / cfg.RD) / (cfg.RD * theta0 * pi0)

        return theta0, qv0, rho0

    # ── Initial conditions ────────────────────────────────────────────────────
    def _init_random_bl(self, g: Grid) -> None:
        """
        Seed the boundary layer with spatially correlated random perturbations.
        This produces many small thermals rather than one big bubble, giving
        realistic scattered cumulus instead of a domain-filling slab.
        """
        rng = np.random.default_rng(cfg.RANDOM_SEED)
        sigma = cfg.NOISE_CORR / g.dx   # correlation length in grid cells

        # background water vapour
        self.qv[:] = self.qv0[np.newaxis, np.newaxis, :]

        # 2-D correlated random fields (one for θ, one for qv)
        def _corr_noise():
            raw = rng.standard_normal((g.nx, g.ny))
            fld = gaussian_filter(raw, sigma=sigma)
            return fld / fld.std()   # unit std, zero mean

        theta_noise_2d = _corr_noise()   # (nx, ny)
        qv_noise_2d    = _corr_noise()

        # Vertical envelope: full amplitude at surface, tapers to 0 at BL top
        z    = g.z                        # (nz,)
        taper = np.maximum(0.0, 1.0 - z / cfg.Z_BL) ** 2   # (nz,)

        self.theta += cfg.THETA_NOISE * theta_noise_2d[:, :, np.newaxis] * taper
        self.qv    += cfg.QV_NOISE    * qv_noise_2d  [:, :, np.newaxis] * taper

        # Zero out terrain cells (if terrain is present)
        if g.terrain is not None:
            self.theta[g.terrain.mask] = 0.0
            self.qv[g.terrain.mask] = 0.0

    # ── Load from snapshot ────────────────────────────────────────────────────
    def _load_from_snapshot(self, snapshot_path: str, g: Grid) -> None:
        """
        Load state from a saved snapshot file.

        Snapshots contain: qc, qr, qv, w, theta (and their mean profiles).
        We need to also initialize u, v (set to zero for simplicity, or could
        compute from geostrophic balance if needed).
        """
        import os
        if not os.path.exists(snapshot_path):
            raise FileNotFoundError(f"Snapshot file not found: {snapshot_path}")

        data = np.load(snapshot_path)
        print(f"Loading initial conditions from: {snapshot_path}")
        print(f"  Snapshot time: t = {data['t']:.1f}s")

        # Check grid compatibility
        if data['qc'].shape != (g.nx, g.ny, g.nz):
            raise ValueError(
                f"Snapshot grid size {data['qc'].shape} does not match "
                f"current grid ({g.nx}, {g.ny}, {g.nz})"
            )

        # Load fields
        self.qc = data['qc'].astype(float)
        self.qr = data['qr'].astype(float)
        self.qv = data['qv'].astype(float)
        self.theta = data['theta'].astype(float)

        # w handling depends on snapshot format
        w_data = data['w']
        if w_data.shape[2] == g.nz:  # cell-centered (old format)
            # Reconstruct face values
            w_cc = w_data
            self.w[:, :, 1:-1] = 0.5 * (w_cc[:, :, :-1] + w_cc[:, :, 1:])
            self.w[:, :, 0] = 0.0
            self.w[:, :, -1] = 0.0
        elif w_data.shape[2] == g.nz + 1:  # face-staggered (nested IC format)
            # Direct copy
            self.w[:] = w_data
            w_cc = 0.5 * (w_data[:, :, :-1] + w_data[:, :, 1:])  # for reporting
        else:
            raise ValueError(f"w shape {w_data.shape} incompatible with grid nz={g.nz}")

        # u, v: load if available, otherwise initialize to zero
        # NOTE: Snapshots save cell-centered u,v but State needs face-staggered
        if 'u' in data:
            u_data = data['u'].astype(float)
            if u_data.shape[0] == g.nx:  # cell-centered (nx, ny, nz)
                # Convert to face-staggered (nx+1, ny, nz) via simple replication
                # Interior faces: average of adjacent cells
                self.u[1:-1, :, :] = 0.5 * (u_data[:-1, :, :] + u_data[1:, :, :])
                # Periodic BCs: wrap around
                self.u[0, :, :] = 0.5 * (u_data[-1, :, :] + u_data[0, :, :])
                self.u[-1, :, :] = self.u[0, :, :]
            else:  # already face-staggered
                self.u[:] = u_data
        else:
            self.u[:] = self.u_g[np.newaxis, np.newaxis, :]

        if 'v' in data:
            v_data = data['v'].astype(float)
            if v_data.shape[1] == g.ny:  # cell-centered (nx, ny, nz)
                # Convert to face-staggered (nx, ny+1, nz)
                self.v[:, 1:-1, :] = 0.5 * (v_data[:, :-1, :] + v_data[:, 1:, :])
                # Periodic BCs
                self.v[:, 0, :] = 0.5 * (v_data[:, -1, :] + v_data[:, 0, :])
                self.v[:, -1, :] = self.v[:, 0, :]
            else:  # already face-staggered
                self.v[:] = v_data
        else:
            self.v[:] = cfg.V_GEO

        # Apply terrain mask if terrain is present
        if g.terrain is not None:
            mask = g.terrain.mask
            self.u[mask] = 0.0
            self.v[mask] = 0.0
            self.theta[mask] = 0.0
            qv0_full = np.broadcast_to(self.qv0, (g.nx, g.ny, g.nz))
            self.qv[mask] = qv0_full[mask]
            self.qc[mask] = 0.0
            self.qr[mask] = 0.0

        print(f"  Loaded: qc_max={self.qc.max()*1e3:.2f}g/kg, "
              f"qr_max={self.qr.max()*1e3:.2f}g/kg, "
              f"w_max={w_cc.max():.2f}m/s")

    # ── C-grid interpolation utilities ────────────────────────────────────────
    def u_at_cell(self) -> np.ndarray:
        """
        Interpolate u from x-faces to cell centers.
        u is on x-faces (nx+1, ny, nz), return (nx, ny, nz).
        """
        return 0.5 * (self.u[:-1, :, :] + self.u[1:, :, :])

    def v_at_cell(self) -> np.ndarray:
        """
        Interpolate v from y-faces to cell centers.
        v is on y-faces (nx, ny+1, nz), return (nx, ny, nz).
        """
        return 0.5 * (self.v[:, :-1, :] + self.v[:, 1:, :])

    def w_at_cell(self) -> np.ndarray:
        """
        Interpolate w from z-faces to cell centers.
        w is on z-faces (nx, ny, nz+1), return (nx, ny, nz).
        """
        return 0.5 * (self.w[:, :, :-1] + self.w[:, :, 1:])

    def u_at_v_faces(self) -> np.ndarray:
        """
        Interpolate u to v-face locations (nx, ny+1, nz).
        Average in both x and y directions, with periodic BCs.
        """
        # First average u in x: (nx+1, ny, nz) -> (nx, ny, nz)
        u_at_cell = 0.5 * (self.u[:-1, :, :] + self.u[1:, :, :])
        # Then average in y with periodic wrapping: (nx, ny, nz) -> (nx, ny+1, nz)
        u_at_v = np.zeros((self.grid.nx, self.grid.ny + 1, self.grid.nz))
        u_at_v[:, 1:-1, :] = 0.5 * (u_at_cell[:, :-1, :] + u_at_cell[:, 1:, :])
        u_at_v[:, 0, :] = 0.5 * (u_at_cell[:, -1, :] + u_at_cell[:, 0, :])  # periodic
        u_at_v[:, -1, :] = u_at_v[:, 0, :]  # periodic
        return u_at_v

    def v_at_u_faces(self) -> np.ndarray:
        """
        Interpolate v to u-face locations (nx+1, ny, nz).
        Average in both y and x directions, with periodic BCs.
        """
        # First average v in y: (nx, ny+1, nz) -> (nx, ny, nz)
        v_at_cell = 0.5 * (self.v[:, :-1, :] + self.v[:, 1:, :])
        # Then average in x with periodic wrapping: (nx, ny, nz) -> (nx+1, ny, nz)
        v_at_u = np.zeros((self.grid.nx + 1, self.grid.ny, self.grid.nz))
        v_at_u[1:-1, :, :] = 0.5 * (v_at_cell[:-1, :, :] + v_at_cell[1:, :, :])
        v_at_u[0, :, :] = 0.5 * (v_at_cell[-1, :, :] + v_at_cell[0, :, :])  # periodic
        v_at_u[-1, :, :] = v_at_u[0, :, :]  # periodic
        return v_at_u

    # ── Derived quantities ────────────────────────────────────────────────────
    def buoyancy(self) -> np.ndarray:
        """
        Boussinesq buoyancy: B = g (θ′/θ₀ + 0.61 qv − qc − qr)
        Shape: (nx, ny, nz)
        """
        theta0 = self.theta0[np.newaxis, np.newaxis, :]
        qv0 = self.qv0[np.newaxis, np.newaxis, :]
        return cfg.G * (
            self.theta / theta0
            + 0.61 * (self.qv - qv0)   # perturbation from background, not total qv
            - self.qc
            - self.qr
        )

    def T(self) -> np.ndarray:
        """Approximate absolute temperature from θ and reference Exner function."""
        # Build pi0 at cell centres (was computed in _build_reference but not stored)
        g = self.grid
        _, _, _ = self._build_reference(g.z)  # cheap; not ideal – cache below
        pi0 = self._pi0()
        return (self.theta + self.theta0[np.newaxis, np.newaxis, :]) * pi0

    def _pi0(self) -> np.ndarray:
        """Exner function π₀(z), broadcast to (1,1,nz)."""
        z = self.grid.z
        pi0 = np.ones(len(z))
        theta0 = self.theta0
        for k in range(1, len(z)):
            dz = z[k] - z[k - 1]
            theta_mid = 0.5 * (theta0[k] + theta0[k - 1])
            pi0[k] = pi0[k - 1] - cfg.G * dz / (cfg.CP * theta_mid)
        return pi0[np.newaxis, np.newaxis, :]

    # ── Phase 1.4: θ_l/qt conversion methods ──────────────────────────────────
    def _update_conserved_variables(self) -> None:
        """
        Update θ_l and qt from current θ, qv, qc, qr.

        Called after physics that modify θ, qv, qc (e.g., microphysics).
        """
        from thermodynamics import convert_to_theta_l_qt
        pi0 = self._pi0()
        self.theta_l, self.qt, self.qr = convert_to_theta_l_qt(
            self.theta, self.qv, self.qc, self.qr, pi0
        )

    def _update_diagnostic_variables(self) -> None:
        """
        Update diagnostic θ, qv, qc from conserved θ_l, qt, qr.

        Called after advection/timestepping of θ_l, qt.
        """
        from thermodynamics import convert_from_theta_l_qt
        pi0 = self._pi0()
        self.theta, self.qv, self.qc, self.qr = convert_from_theta_l_qt(
            self.theta_l, self.qt, self.qr, self.theta0, pi0
        )

"""Model state: fields, reference profiles, and initial conditions."""

import numpy as np
from scipy.ndimage import gaussian_filter
import config as cfg
from grid import Grid


class State:
    """
    Prognostic fields (all shape nx × ny × nz unless noted):
        u, v   – horizontal wind  [m/s]
        w      – vertical wind    [m/s]  shape nx × ny × (nz+1), face-staggered in z
        theta  – potential temperature perturbation θ'  [K]
        qv     – water-vapour mixing ratio  [kg/kg]
        qc     – cloud-water mixing ratio   [kg/kg]
        qr     – rain-water mixing ratio    [kg/kg]

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
        shape_w = (g.nx, g.ny, g.nz + 1)

        # ── Prognostic fields ─────────────────────────────────────────────────
        self.u     = np.zeros(shape)
        self.v     = np.zeros(shape)
        self.w     = np.zeros(shape_w)    # face-staggered in z
        self.theta = np.zeros(shape)      # θ′  (perturbation)
        self.qv    = np.zeros(shape)
        self.qc    = np.zeros(shape)
        self.qr    = np.zeros(shape)

        # ── Reference profiles ────────────────────────────────────────────────
        self.theta0, self.qv0, self.rho0 = self._build_reference(g.z)

        # ── Apply initial conditions ──────────────────────────────────────────
        if restart_file is not None:
            self._load_from_snapshot(restart_file, g)
        else:
            self._init_random_bl(g)
            # Initialize background wind (uniform in space)
            self.u[:] = cfg.U_GEO
            self.v[:] = cfg.V_GEO

    # ── Reference atmosphere ──────────────────────────────────────────────────
    def _build_reference(self, z: np.ndarray):
        """
        Piecewise-linear θ₀(z): BL lapse rate below z_BL, FT lapse rate above.
        qv₀: exponential decay from surface.
        ρ₀: from hydrostatic + ideal gas (approximately constant for Boussinesq).
        """
        z_bl = cfg.Z_BL
        gamma_bl = cfg.GAMMA_BL * 1e-3   # K/m
        gamma_ft = cfg.GAMMA_FT * 1e-3   # K/m

        theta0 = np.where(
            z <= z_bl,
            cfg.THETA_S + gamma_bl * z,
            cfg.THETA_S + gamma_bl * z_bl + gamma_ft * (z - z_bl),
        )

        qv0 = cfg.QV_SURF * np.exp(-z / cfg.QV_SCALE)

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
        if 'u' in data:
            self.u[:] = data['u'].astype(float)
        else:
            self.u[:] = 0.0

        if 'v' in data:
            self.v[:] = data['v'].astype(float)
        else:
            self.v[:] = 0.0

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

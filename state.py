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

    def __init__(self, grid: Grid) -> None:
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
        self._init_random_bl(g)

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

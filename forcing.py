"""
Surface fluxes and sponge-layer damping.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import config as cfg
from grid import Grid
from state import State
from lateral_bc import hetero_window


def build_flux_map(g: Grid) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate fixed 2-D SHF and LHF maps (W/m²) with spatial correlation.
    Hot patches drive individual thermals; cool patches suppress convection.
    Called once at simulation start and reused every timestep.

    SHF and LHF patterns are independent: each is zero-mean, unit-std and periodic
    (mode='wrap'), and the LHF pattern is made exactly uncorrelated with the SHF one
    (with few patches per domain, two random draws are otherwise strongly correlated
    by chance). The domain means stay exactly cfg.SHF and cfg.LHF.

    With LATERAL_BC = "fringe" the patterns are confined to the test region
    (lateral_bc.hetero_window); the fetch and fringe stay homogeneous, and the std is
    SHF_SIGMA inside the full-strength part of the window.
    """
    rng   = np.random.default_rng(cfg.RANDOM_SEED + 1)
    sigma = cfg.SHF_CORR / g.dx     # correlation length in grid cells
    win   = hetero_window(g)[:, None]
    full  = np.broadcast_to(win >= 1.0 - 1e-12, (g.nx, g.ny))

    def _field():
        fld = gaussian_filter(rng.standard_normal((g.nx, g.ny)), sigma=sigma, mode='wrap')
        return win * (fld - (win * fld).sum() / (win.sum() * g.ny))   # zero domain mean

    f_shf = _field()
    f_shf /= f_shf[full].std()
    f_lhf = _field()
    f_lhf -= (f_lhf * f_shf).sum() / (f_shf * f_shf).sum() * f_shf   # uncorrelate from SHF
    f_lhf /= f_lhf[full].std()

    shf_map = cfg.SHF * (1.0 + cfg.SHF_SIGMA * f_shf)
    lhf_map = cfg.LHF * (1.0 + cfg.SHF_SIGMA * f_lhf)

    # Clip to physically reasonable range, then restore the domain means
    shf_map = np.maximum(shf_map, 5.0)
    lhf_map = np.maximum(lhf_map, 2.0)
    shf_map *= cfg.SHF / shf_map.mean()
    lhf_map *= cfg.LHF / lhf_map.mean()

    if cfg.SHF_STRIPE_AMP != 0.0:
        shf_map = shf_map + cfg.SHF_STRIPE_AMP * stripe_profile(g)[:, None]
    if cfg.SHF_SPOT_AMP != 0.0 or cfg.SHF_SPOT_LHF_AMP != 0.0:
        spot = spot_profile(g)
        shf_map = shf_map + cfg.SHF_SPOT_AMP * spot
        lhf_map = lhf_map + cfg.SHF_SPOT_LHF_AMP * spot

    return shf_map, lhf_map


def stripe_profile(g: Grid) -> np.ndarray:
    """(nx,) weight in [0, 1] of the warm stripe (cos² edges; see SHF_STRIPE_* in config)."""
    x0, w = cfg.SHF_STRIPE_X0, cfg.SHF_STRIPE_WIDTH
    tp = max(min(cfg.SHF_STRIPE_TAPER, 0.5 * w), 1e-9)
    ramp = lambda d: np.sin(0.5 * np.pi * np.clip(d / tp, 0.0, 1.0)) ** 2
    return ramp(g.x - x0) * ramp(x0 + w - g.x)


def spot_profile(g: Grid) -> np.ndarray:
    """(nx, ny) weight in [0, 1] of the hot spot (disc with cos² edge or Gaussian; periodic distance; see SHF_SPOT_*)."""
    lx, ly = g.nx * g.dx, g.ny * g.dy
    x0 = 0.5 * lx if cfg.SHF_SPOT_X0 is None else cfg.SHF_SPOT_X0
    y0 = 0.5 * ly if cfg.SHF_SPOT_Y0 is None else cfg.SHF_SPOT_Y0
    ddx = (g.x - x0 + 0.5 * lx) % lx - 0.5 * lx
    ddy = (g.y - y0 + 0.5 * ly) % ly - 0.5 * ly
    r = np.hypot(ddx[:, None], ddy[None, :])
    rad = 0.5 * cfg.SHF_SPOT_DIAM
    if getattr(cfg, 'SHF_SPOT_SHAPE', 'disc') == 'gaussian':
        sigma = cfg.SHF_SPOT_DIAM / (2.0 * np.sqrt(2.0))       # 2πσ² = π (D/2)²
        return np.exp(-0.5 * (r / sigma) ** 2)
    tp = max(min(cfg.SHF_SPOT_TAPER, rad), 1e-9)
    return np.sin(0.5 * np.pi * np.clip((rad - r) / tp, 0.0, 1.0)) ** 2


class FluxPattern:
    """
    Evolving surface-flux pattern (AR(1) red noise in time, Gaussian-correlated in space).

    Every FLUX_PATTERN_DT seconds:  pattern ← a·pattern + sqrt(1−a²)·new_field,
    a = exp(−FLUX_PATTERN_DT / FLUX_PATTERN_TAU). new_field has zero mean and unit std,
    so the pattern keeps unit std. One pattern modulates both SHF and LHF
    (warm patches are also moist), preserving the domain means.
    """

    def __init__(self, g: Grid):
        self.rng = np.random.default_rng(cfg.RANDOM_SEED + 2)
        self.sigma = cfg.SHF_CORR / g.dx
        self.shape = (g.nx, g.ny)
        self.a = np.exp(-cfg.FLUX_PATTERN_DT / cfg.FLUX_PATTERN_TAU)
        self.pattern = self._new_field()
        self.t_last = 0.0

    def _new_field(self) -> np.ndarray:
        fld = gaussian_filter(self.rng.standard_normal(self.shape), sigma=self.sigma, mode='wrap')
        fld -= fld.mean()
        return fld / fld.std()

    def update(self, t: float) -> None:
        """Advance the pattern by however many FLUX_PATTERN_DT intervals have passed."""
        while t - self.t_last >= cfg.FLUX_PATTERN_DT - 1e-9:
            self.pattern = self.a * self.pattern + np.sqrt(1.0 - self.a**2) * self._new_field()
            self.t_last += cfg.FLUX_PATTERN_DT

    def maps(self) -> tuple[np.ndarray, np.ndarray]:
        factor = 1.0 + cfg.FLUX_PATTERN_AMP * self.pattern
        return cfg.SHF * factor, cfg.LHF * factor


def cloud_transmittance(s: State, g: Grid) -> np.ndarray:
    """
    Surface shortwave transmittance under clouds, (nx, ny), in [MIN_TRANSMITTANCE, 1].
    Beer's law on liquid water path: exp(−CLOUD_EXTINCTION · LWP).
    """
    LWP = np.sum(s.rho0[np.newaxis, np.newaxis, :] * s.qc
                 * g.dz_face[np.newaxis, np.newaxis, :], axis=2)   # kg/m²
    return np.maximum(np.exp(-cfg.CLOUD_EXTINCTION * LWP), cfg.MIN_TRANSMITTANCE)


def apply_surface_fluxes(s: State, g: Grid, dt: float,
                         shf_map: np.ndarray, lhf_map: np.ndarray) -> None:
    """
    Apply spatially heterogeneous SHF and LHF into the surface cell.

    For flat domain: surface is at k=0.
    For terrain: surface follows terrain elevation (terrain-following).

    shf_map, lhf_map: (nx, ny) arrays in W/m².

    Optional cloud-radiative feedback:
    - Compute column-integrated cloud water (liquid water path, LWP)
    - Calculate optical depth: τ = extinction × LWP
    - Reduce surface fluxes by transmittance = exp(-τ)
    - Creates self-limiting clouds and oscillating thermals
    """
    # Cloud-radiative feedback (optional)
    if cfg.ENABLE_CLOUD_SHADING:
        transmittance = cloud_transmittance(s, g)
        shf_map = shf_map * transmittance
        lhf_map = lhf_map * transmittance

    if g.terrain is None:
        # ── Flat domain (original code) ──────────────────────────────────────
        rho0_bot = s.rho0[0]
        dz0      = g.dz_face[0]
        pi0_bot  = s._pi0()[0, 0, 0]

        dtheta = shf_map * dt / (rho0_bot * cfg.CP * dz0 * pi0_bot)   # (nx,ny)
        dqv    = lhf_map * dt / (rho0_bot * cfg.LV * dz0)

        s.theta[:, :, 0] += dtheta
        s.qv   [:, :, 0] += dqv

    else:
        # ── Terrain-following surface ────────────────────────────────────────
        pi0 = s._pi0()[0, 0, :]  # (nz,) - Exner function at each level

        # Apply fluxes at terrain surface level for each column
        for i in range(g.nx):
            for j in range(g.ny):
                k = g.terrain.k_sfc[i, j]  # surface level index
                dz = g.dz_face[k]
                rho = s.rho0[k]

                dtheta = shf_map[i, j] * dt / (rho * cfg.CP * dz * pi0[k])
                dqv    = lhf_map[i, j] * dt / (rho * cfg.LV * dz)

                s.theta[i, j, k] += dtheta
                s.qv[i, j, k]    += dqv


def subsidence_profile(z: np.ndarray) -> np.ndarray:
    """
    Large-scale vertical velocity w_ls(z) ≤ 0 [m/s]: linear from 0 at the surface to
    −W_SUBS at SUBS_Z_MAX (default Z_BL); above that constant, or (if SUBS_Z_ZERO
    is set) linear back to 0 at SUBS_Z_ZERO and zero above.
    """
    z_max = cfg.SUBS_Z_MAX if cfg.SUBS_Z_MAX is not None else cfg.Z_BL
    w_ls = -cfg.W_SUBS * np.minimum(z / z_max, 1.0)
    if cfg.SUBS_Z_ZERO is not None:
        taper = np.clip((cfg.SUBS_Z_ZERO - z) / (cfg.SUBS_Z_ZERO - z_max), 0.0, 1.0)
        w_ls = np.where(z > z_max, -cfg.W_SUBS * taper, w_ls)
    return w_ls


def _upwind_gradient_from_above(phi: np.ndarray, g: Grid) -> np.ndarray:
    """dφ/dz using the level above (upwind for w_ls < 0); zero at the top level."""
    grad = np.zeros_like(phi)
    grad[:-1] = (phi[1:] - phi[:-1]) / g.dz_cen
    return grad


def apply_subsidence(s: State, g: Grid, dt: float) -> None:
    """
    Large-scale subsidence: ∂φ/∂t = −w_ls(z) ∂φ̄/∂z, applied uniformly in x, y.
    Adds drying and adiabatic warming, preventing the whole domain
    from becoming saturated over time (standard in fair-weather cumulus LES).

    SUBS_GRADIENT = "reference": φ̄ = reference profiles θ0, qv0 (centred gradient)
    SUBS_GRADIENT = "mean":      φ̄ = current horizontal-mean θ (θ0 + ⟨θ′⟩), qv, qc
                                 (upwind gradient), so the forcing follows the
                                 evolving boundary layer and inversion
    """
    z    = g.z                                                     # (nz,)
    w_ls = subsidence_profile(z)                                   # (nz,) negative (downward)
    w3   = w_ls[np.newaxis, np.newaxis, :]

    if cfg.SUBS_GRADIENT == "mean":
        theta_bar = s.theta0 + s.theta.mean(axis=(0, 1))
        s.theta -= dt * w3 * _upwind_gradient_from_above(theta_bar, g)
        s.qv    -= dt * w3 * _upwind_gradient_from_above(s.qv.mean(axis=(0, 1)), g)
        s.qc    -= dt * w3 * _upwind_gradient_from_above(s.qc.mean(axis=(0, 1)), g)
        return

    dtheta0_dz = np.gradient(s.theta0, z)                         # (nz,)
    dqv0_dz    = np.gradient(s.qv0,    z)                         # (nz,)

    # subsidence advects the reference gradient: ∂φ′/∂t -= w_ls * dφ₀/dz
    s.theta -= dt * w3 * dtheta0_dz[np.newaxis, np.newaxis, :]
    s.qv    -= dt * w3 * dqv0_dz[np.newaxis, np.newaxis, :]


def apply_radiative_cooling(s: State, g: Grid, dt: float) -> None:
    """
    Prescribed clear-sky radiative tendency on θ: RAD_COOLING [K/day] below RAD_Z_TOP,
    tapering linearly to 0 at RAD_Z_ZERO. No-op when RAD_COOLING = 0.
    """
    if cfg.RAD_COOLING == 0.0:
        return
    taper = np.clip((cfg.RAD_Z_ZERO - g.z) / (cfg.RAD_Z_ZERO - cfg.RAD_Z_TOP), 0.0, 1.0)
    s.theta += dt * (cfg.RAD_COOLING / 86400.0) * taper[np.newaxis, np.newaxis, :]


def apply_profile_nudging(s: State, g: Grid, dt: float) -> None:
    """
    Remove domain-mean drift of θ′ and qv toward their reference profiles.
    Represents the large-scale environment preventing unlimited thermodynamic
    drift in a periodic domain.  Only the MEAN is nudged — local variability
    driven by the heterogeneous surface is fully preserved.

    UPDATED: Only nudge ABOVE cfg.NUDGE_Z_TOP to avoid damping BL dynamics.
    """
    # domain-mean profiles at each height
    theta_mean = np.nanmean(s.theta, axis=(0, 1))        # (nz,)
    qv_mean    = np.nanmean(s.qv,   axis=(0, 1))        # (nz,)

    # Create vertical mask: only nudge above NUDGE_Z_TOP (disable in BL)
    if hasattr(cfg, 'NUDGE_Z_TOP'):
        z = g.z  # (nz,)
        nudge_mask = (z > cfg.NUDGE_Z_TOP).astype(float)  # (nz,)
    else:
        nudge_mask = 1.0  # Nudge everywhere if NUDGE_Z_TOP not set

    # relax mean θ′ toward 0, mean qv toward qv0 (only where nudge_mask > 0)
    s.theta -= (dt / cfg.TAU_NUDGE_T) * theta_mean[np.newaxis, np.newaxis, :] * nudge_mask[np.newaxis, np.newaxis, :]
    s.qv    -= (dt / cfg.TAU_NUDGE_Q) * (qv_mean - s.qv0)[np.newaxis, np.newaxis, :] * nudge_mask[np.newaxis, np.newaxis, :]


def apply_cold_pool(s: State, g: Grid, dt: float) -> None:
    """
    Cold pool parameterisation.

    Rain that falls into the lowest CP_DEPTH metres is in the dry sub-cloud
    layer and evaporates rapidly.  We evaporate it instantaneously here,
    concentrating the latent cooling near the surface.  The resulting patch
    of cold, dense air spreads as a density current (handled by the existing
    pressure solver + dynamics), lifts air at its edges, and triggers new
    convection — the primary mechanism maintaining scattered cumulus.
    """
    CP_DEPTH = 300.0  # m — depth of the cold pool layer
    Lcp      = cfg.LV / cfg.CP
    pi0      = s._pi0()                        # (1,1,nz)

    cp_mask = g.z < CP_DEPTH                   # (nz,) bool — ~6 lowest levels
    if not cp_mask.any():
        return

    qr_cp = s.qr[:, :, cp_mask]               # (nx,ny,n_cp)
    evap  = np.maximum(qr_cp, 0.0)            # evaporate all rain in CP layer

    s.qr   [:, :, cp_mask] -= evap
    s.qv   [:, :, cp_mask] += evap
    s.theta[:, :, cp_mask] -= Lcp * evap / pi0[0, 0, cp_mask]

    s.qv = np.maximum(s.qv, 0.0)
    s.qr = np.maximum(s.qr, 0.0)


def apply_sponge(s: State, g: Grid, dt: float) -> None:
    """
    Rayleigh damping toward REFERENCE PROFILES above z_sponge.

    CRITICAL FIX: Relax toward (u_g, v_g, qv0), not toward zero!
    The old code multiplied fields by exp(-α*dt), which drives them to zero.
    This removed q_t from the sponge layer and caused spurious wind drift.

    Correct relaxation: φ(t+dt) = φ + (φ_ref - φ) * (1 - exp(-α*dt))

    Damping coefficient α(z) = sin²(π/2 * (z - z_sp)/(z_top - z_sp)) / τ
    """
    z     = g.z                     # (nz,)
    z_sp  = cfg.Z_SPONGE
    z_top = g.z_face[-1]

    alpha = np.where(
        z > z_sp,
        np.sin(0.5 * np.pi * (z - z_sp) / (z_top - z_sp)) ** 2 / cfg.TAU_SPONGE,
        0.0,
    )                                 # (nz,)

    a3 = alpha[np.newaxis, np.newaxis, :]   # broadcast to (1,1,nz)

    # Relaxation factor: 1 - exp(-α*dt)
    relax = 1.0 - np.exp(-a3 * dt)

    # Relax toward reference profiles
    s.u     += (s.u_g[np.newaxis, np.newaxis, :] - s.u) * relax   # → u_g(z)
    s.v     += (cfg.V_GEO - s.v) * relax         # → v_g
    s.theta += (0.0 - s.theta) * relax           # → θ′ = 0
    s.qv    += (s.qv0 - s.qv) * relax            # → qv0 (reference moisture profile)
    s.qc    += (0.0 - s.qc) * relax              # → 0 (no clouds in sponge)
    s.qr    += (0.0 - s.qr) * relax              # → 0 (no rain in sponge)
    s.qg    += (0.0 - s.qg) * relax              # → 0

    # w on faces: interpolate alpha to faces
    alpha_face = np.zeros(g.nz + 1)
    alpha_face[1:-1] = 0.5 * (alpha[:-1] + alpha[1:])
    alpha_face[0]    = alpha[0]
    alpha_face[-1]   = alpha[-1]

    relax_w = 1.0 - np.exp(-alpha_face[np.newaxis, np.newaxis, :] * dt)
    s.w += (0.0 - s.w) * relax_w  # → 0

    # Keep boundary w = 0
    s.w[:, :, 0]  = 0.0
    s.w[:, :, -1] = 0.0

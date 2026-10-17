"""
Spectral pressure projection (anelastic incompressibility enforcement).

Anelastic constraint: ∇·(ρ₀u) = 0

Solves: (1/ρ₀)·∇·(ρ₀ ∇φ) = (1/ρ₀)·∇·(ρ₀u*) / dt
then corrects:  u ← u* − dt ∂φ/∂x
                v ← v* − dt ∂φ/∂y
                w ← w* − dt ∂φ/∂z

Lateral BCs per direction (cfg.BC_X, cfg.BC_Y):
  "periodic": real FFT, modified wavenumbers of the periodic 2nd-order Laplacian.
  "open":     the normal velocity on the boundary faces is given and kept (∂φ/∂n = 0);
              orthonormal DCT-II, eigenvalues 2(cos(πk/n) − 1)/Δ². Before the solve the net
              mass flux through the open boundaries is set to zero by a uniform correction of
              the boundary normal velocity (solvability with a rigid lid).
Neumann (∂φ/∂z = 0) at top and bottom. w is face-staggered in z.
"""

import os
import numpy as np
from scipy.fft import rfft2, irfft2, rfft, irfft, dct, idct, dctn, idctn
import config as cfg
from grid import Grid

try:
    import kernels
except ImportError:          # numba not installed → numpy Thomas solver
    kernels = None

_FFT_WORKERS = os.cpu_count() or 1


def project(u: np.ndarray, v: np.ndarray, w: np.ndarray,
            g: Grid, dt: float, rho0: np.ndarray = None):
    """
    In-place velocity projection for anelastic constraint: ∇·(ρ₀u) = 0.

    rho0: (nz,) reference density at cell centres. Pass State.rho0 so the
          constraint matches the ρ₀ used by advection/diffusion in dynamics.py.
          If None, falls back to a constant-θ (THETA_S) hydrostatic profile
          (only for standalone tests without a State).

    Returns the pressure φ (nx,ny,nz).
    """
    nx, ny, nz = g.nx, g.ny, g.nz
    dx, dy = g.dx, g.dy
    dz_f = g.dz_face    # (nz,)   cell heights
    dz_c = g.dz_cen     # (nz-1,) inter-centre spacings

    if rho0 is not None:
        rho0_c = np.asarray(rho0)
    else:
        rho0_c = _constant_theta_rho0(g.z)

    # Interpolate to w-faces
    rho0_w = np.zeros(nz + 1)
    rho0_w[0] = rho0_c[0]  # Extrapolate to bottom
    rho0_w[-1] = rho0_c[-1]  # Extrapolate to top
    rho0_w[1:-1] = 0.5 * (rho0_c[:-1] + rho0_c[1:])  # Average to faces

    open_x = getattr(cfg, 'BC_X', 'periodic') == 'open'
    open_y = getattr(cfg, 'BC_Y', 'periodic') == 'open'
    if open_x or open_y:
        return _project_open(u, v, w, g, dt, rho0_c, rho0_w, open_x, open_y)

    # ── Enforce periodic boundary conditions on face-staggered velocities ────────
    # For C-grid with periodic BCs, the "extra" face is the same as the first face
    # u[nx] = u[0] (same x-face due to wraparound)
    # v[:, ny] = v[:, 0] (same y-face due to wraparound)
    u[-1, :, :] = u[0, :, :]
    v[:, -1, :] = v[:, 0, :]

    # ── Divergence of ρ₀u*  (anelastic constraint) ────────────────────────────
    # Anelastic: ∇·(ρ₀u) = 0
    # RHS = (1/ρ₀) * ∇·(ρ₀u*) / dt
    #
    # Horizontal: ρ₀ constant in x,y → ∂(ρ₀u)/∂x = ρ₀ ∂u/∂x
    # Vertical:   ρ₀ = ρ₀(z)        → ∂(ρ₀w)/∂z needs ρ₀ at w-faces

    # Horizontal divergence (ρ₀ doesn't vary in x,y)
    div_u = (u[1:, :, :] - u[:-1, :, :]) / dx         # (nx, ny, nz)
    div_v = (v[:, 1:, :] - v[:, :-1, :]) / dy         # (nx, ny, nz)

    # Vertical divergence: ∂(ρ₀w)/∂z
    # w is on z-faces (nx, ny, nz+1), rho0_w is on z-faces (nz+1)
    # flux at face k: rho0_w[k] * w[:,:,k]
    # divergence in cell k: (flux[k+1] - flux[k]) / dz_f[k]
    rho0w_flux = w * rho0_w[np.newaxis, np.newaxis, :]  # (nx, ny, nz+1)
    div_rho0w = (rho0w_flux[:, :, 1:] - rho0w_flux[:, :, :-1]) / dz_f[np.newaxis, np.newaxis, :]  # (nx, ny, nz)

    # RHS = (1/rho0_c) * ∇·(ρ₀u) / dt
    #     = [ρ₀ (∂u/∂x + ∂v/∂y) + ∂(ρ₀w)/∂z] / (ρ₀ dt)
    # Expand rho0_c to 3D
    rho0_c_3d = rho0_c[np.newaxis, np.newaxis, :]  # (1, 1, nz)
    rhs = (rho0_c_3d * (div_u + div_v) + div_rho0w) / (rho0_c_3d * dt)  # (nx, ny, nz)

    # ── 2-D FFT in x-y ────────────────────────────────────────────────────────
    rhs_hat = rfft2(rhs, axes=(0, 1), workers=_FFT_WORKERS)                     # (nx, ny//2+1, nz)

    # Modified wavenumbers for 2nd-order centred FD on periodic grid
    kx_idx = np.fft.fftfreq(nx, d=1.0 / nx).astype(int)
    ky_idx = np.arange(ny // 2 + 1)

    lam_x = (2.0 * (np.cos(2.0 * np.pi * kx_idx / nx) - 1.0) / dx**2)[:, np.newaxis]  # (nx,1)
    lam_y = (2.0 * (np.cos(2.0 * np.pi * ky_idx / ny) - 1.0) / dy**2)[np.newaxis, :]  # (1,ny//2+1)
    lam_xy = (lam_x + lam_y)[:, :, np.newaxis]            # (nx, ny//2+1, 1)
    # The (0,0) mode is singular (λ = 0 with Neumann top/bottom) and is discarded below;
    # a dummy λ keeps its Thomas sweep from dividing by an exact zero pivot.
    lam_xy[0, 0, 0] = -1.0 / dx**2

    # ── Tridiagonal solve in z for each (kx,ky) mode ──────────────────────────
    # Anelastic operator: (1/ρ₀) ∂/∂z (1/ρ₀ ∂φ/∂z) + (1/ρ₀)(∂²φ/∂x² + ∂²φ/∂y²) = RHS
    #
    # Simplifies to: ∂/∂z (1/ρ₀ ∂φ/∂z) = ρ₀ * RHS - (∂²φ/∂x² + ∂²φ/∂y²)
    #
    # Vertical operator with ρ₀ weighting at w-faces:
    # Interior k (1..nz-2):
    #   a_k φ̂_{k-1} + b_k φ̂_k + c_k φ̂_{k+1} = R̂_k
    #   a_k = rho0_w[k]   / (dz_c[k-1] * dz_f[k] * rho0_c[k])
    #   c_k = rho0_w[k+1] / (dz_c[k]   * dz_f[k] * rho0_c[k])
    #   b_k = -(a_k + c_k) + λ_xy
    # Boundary k=0:    a_0 = 0, c_0 = rho0_w[1]/(dz_c[0]*dz_f[0]*rho0_c[0])
    # Boundary k=nz-1: c_{nz-1} = 0, a_{nz-1} = rho0_w[nz-1]/(dz_c[nz-2]*dz_f[nz-1]*rho0_c[nz-1])

    a = np.zeros(nz)
    b = np.zeros(nz)
    c = np.zeros(nz)

    # Lower coefficients (a: sub-diagonal, from k-1)
    # a[k] couples k-1 to k, uses rho0_w at interface (face k)
    a[1:] = rho0_w[1:-1] / (dz_c * dz_f[1:] * rho0_c[1:])  # nz-1 values

    # Upper coefficients (c: super-diagonal, to k+1)
    # c[k] couples k to k+1, uses rho0_w at interface (face k+1)
    c[:-1] = rho0_w[1:-1] / (dz_c * dz_f[:-1] * rho0_c[:-1])  # nz-1 values

    # Diagonal (without λ_xy)
    b = -(a + c)

    # For k=0: Neumann → φ_ghost = φ_0 → no change (a[0]=0 already)
    # For k=nz-1: Neumann → φ_ghost = φ_{nz-1} → c[nz-1]=0 already

    # Reshape for broadcasting: a,b,c → (1,1,nz)
    a3 = a[np.newaxis, np.newaxis, :]
    b3 = b[np.newaxis, np.newaxis, :]
    c3 = c[np.newaxis, np.newaxis, :]

    if kernels is not None and getattr(cfg, 'USE_NUMBA', True):
        phi_hat = kernels.solve_tridiag_modes(a, b, c, np.ascontiguousarray(lam_xy[:, :, 0]),
                                              np.ascontiguousarray(rhs_hat))
    else:
        phi_hat = _solve_tridiagonal(
            (a3 + 0.0j) * np.ones_like(rhs_hat),
            (b3 + lam_xy) * np.ones_like(rhs_hat),
            (c3 + 0.0j) * np.ones_like(rhs_hat),
            rhs_hat,
            nz,
        )

    # Fix the (0,0) mode: set to 0 (pressure reference; only gradients matter)
    phi_hat[0, 0, :] = 0.0

    # ── Back to physical space ─────────────────────────────────────────────────
    phi = irfft2(phi_hat, s=(nx, ny), axes=(0, 1), workers=_FFT_WORKERS)        # (nx,ny,nz)

    # ── Correct velocities (C-grid gradients matching divergence) ─────────────
    # u is on x-faces: gradient between adjacent cells
    # phi[i] is at cell center i, u[i] is at face i-1/2 (between cells i-1 and i)
    # With periodic BCs, cells wrap: use np.roll or explicit indexing

    # Gradient at x-face i (between cells i-1 and i): (phi[i] - phi[i-1]) / dx
    # u has shape (nx+1, ny, nz); phi has shape (nx, ny, nz)
    # Interior faces i=1..nx-1: use phi[i] - phi[i-1]
    # Boundary faces: i=0 (left) and i=nx (right) use periodic wrapping

    dphi_dx = np.zeros_like(u)  # (nx+1, ny, nz)
    dphi_dx[1:-1, :, :] = (phi[1:, :, :] - phi[:-1, :, :]) / dx
    # Periodic BCs for x:
    dphi_dx[0, :, :] = (phi[0, :, :] - phi[-1, :, :]) / dx   # left face wraps to right
    dphi_dx[-1, :, :] = dphi_dx[0, :, :]                      # right face = left face (periodic)
    u -= dt * dphi_dx

    # v is on y-faces: gradient between adjacent cells
    dphi_dy = np.zeros_like(v)  # (nx, ny+1, nz)
    dphi_dy[:, 1:-1, :] = (phi[:, 1:, :] - phi[:, :-1, :]) / dy
    # Periodic BCs for y:
    dphi_dy[:, 0, :] = (phi[:, 0, :] - phi[:, -1, :]) / dy
    dphi_dy[:, -1, :] = dphi_dy[:, 0, :]
    v -= dt * dphi_dy

    # w is on z-faces: gradient between adjacent cells (already correct stencil)
    dphi_dz = np.zeros_like(w)  # (nx, ny, nz+1)
    dphi_dz[:, :, 1:-1] = (phi[:, :, 1:] - phi[:, :, :-1]) / dz_c[np.newaxis, np.newaxis, :]
    w -= dt * dphi_dz
    # Enforce w=0 at top/bottom boundaries (rigid lid)
    w[:, :, 0]  = 0.0
    w[:, :, -1] = 0.0

    # Enforce zero domain-mean ρ₀w at every interior face.
    # Required by anelastic continuity + periodic BCs + rigid lid:
    # ∂<ρ₀w>/∂z = 0 and <ρ₀w>(z=0) = 0  →  <ρ₀w>(z) = 0 everywhere.
    # So <w>(z) = 0 / ρ₀(z) = 0 (since ρ₀ > 0 everywhere).
    # The (0,0) Fourier mode is singular, so we correct the mean explicitly.
    w[:, :, 1:-1] -= w[:, :, 1:-1].mean(axis=(0, 1), keepdims=True)

    # ── Re-enforce periodic BCs after gradient correction ────────────────────
    # This ensures the redundant faces stay equal (important for consistency)
    u[-1, :, :] = u[0, :, :]
    v[:, -1, :] = v[:, 0, :]

    return phi


def _vertical_coefficients(g: Grid, rho0_c: np.ndarray, rho0_w: np.ndarray):
    """Sub-, main- and super-diagonal (nz,) of the ρ₀-weighted vertical operator (Neumann top/bottom)."""
    nz, dz_f, dz_c = g.nz, g.dz_face, g.dz_cen
    a = np.zeros(nz)
    c = np.zeros(nz)
    a[1:] = rho0_w[1:-1] / (dz_c * dz_f[1:] * rho0_c[1:])
    c[:-1] = rho0_w[1:-1] / (dz_c * dz_f[:-1] * rho0_c[:-1])
    return a, -(a + c), c


def balance_boundary_flux(u: np.ndarray, v: np.ndarray, g: Grid, rho0_c: np.ndarray,
                          open_x: bool, open_y: bool) -> float:
    """
    Make the net mass flux through the open lateral boundaries zero (needed with a rigid
    lid: ∫∇·(ρ₀u) dV = boundary flux). Subtracts one uniform outward velocity from every
    open boundary face; returns that correction [m/s].
    """
    dz = g.dz_face
    flux, area = 0.0, 0.0                      # outward ρ₀·u_n·dA summed, Σ ρ₀ dA
    w_x = rho0_c * dz * g.dy                   # (nz,) ρ₀ dA of one x-face column
    w_y = rho0_c * dz * g.dx
    if open_x:
        flux += float(((u[-1] - u[0]) * w_x).sum())
        area += 2.0 * g.ny * float(w_x.sum())
    if open_y:
        flux += float(((v[:, -1] - v[:, 0]) * w_y).sum())
        area += 2.0 * g.nx * float(w_y.sum())
    corr = flux / area
    if open_x:
        u[0] += corr                           # outward normal is −x at the west face
        u[-1] -= corr
    if open_y:
        v[:, 0] += corr
        v[:, -1] -= corr
    return corr


def _forward(f: np.ndarray, open_x: bool, open_y: bool) -> np.ndarray:
    """Horizontal transform: DCT-II on open axes, real FFT on the periodic one."""
    if open_x and open_y:
        return dctn(f, type=2, axes=(0, 1), norm='ortho', workers=_FFT_WORKERS)
    if open_x:
        return rfft(dct(f, type=2, axis=0, norm='ortho', workers=_FFT_WORKERS),
                    axis=1, workers=_FFT_WORKERS)
    return rfft(dct(f, type=2, axis=1, norm='ortho', workers=_FFT_WORKERS),
                axis=0, workers=_FFT_WORKERS)


def _inverse(fh: np.ndarray, nx: int, ny: int, open_x: bool, open_y: bool) -> np.ndarray:
    if open_x and open_y:
        return idctn(fh, type=2, axes=(0, 1), norm='ortho', workers=_FFT_WORKERS)
    if open_x:
        return idct(irfft(fh, n=ny, axis=1, workers=_FFT_WORKERS),
                    type=2, axis=0, norm='ortho', workers=_FFT_WORKERS)
    return idct(irfft(fh, n=nx, axis=0, workers=_FFT_WORKERS),
                type=2, axis=1, norm='ortho', workers=_FFT_WORKERS)


def _eigenvalues(n: int, d: float, open_axis: bool, half: bool) -> np.ndarray:
    """1-D eigenvalues of the 2nd-order Laplacian: Neumann (DCT-II) or periodic (FFT / rFFT)."""
    if open_axis:
        k = np.arange(n)
        return 2.0 * (np.cos(np.pi * k / n) - 1.0) / d**2
    k = np.arange(n // 2 + 1) if half else np.fft.fftfreq(n, d=1.0 / n)
    return 2.0 * (np.cos(2.0 * np.pi * k / n) - 1.0) / d**2


def _project_open(u, v, w, g: Grid, dt: float, rho0_c, rho0_w, open_x: bool, open_y: bool):
    """project() with at least one open lateral direction (see module docstring)."""
    nx, ny, nz = g.nx, g.ny, g.nz
    dx, dy, dz_f, dz_c = g.dx, g.dy, g.dz_face, g.dz_cen

    if not open_x:
        u[-1, :, :] = u[0, :, :]
    if not open_y:
        v[:, -1, :] = v[:, 0, :]
    balance_boundary_flux(u, v, g, rho0_c, open_x, open_y)

    # RHS = ∇·(ρ₀u*) / (ρ₀ dt), as in the periodic path
    div_h = (u[1:] - u[:-1]) / dx + (v[:, 1:] - v[:, :-1]) / dy
    rflux = w * rho0_w[np.newaxis, np.newaxis, :]
    div_rho0w = (rflux[:, :, 1:] - rflux[:, :, :-1]) / dz_f[np.newaxis, np.newaxis, :]
    r3 = rho0_c[np.newaxis, np.newaxis, :]
    rhs = (r3 * div_h + div_rho0w) / (r3 * dt)

    rhs_hat = _forward(rhs, open_x, open_y)
    lam = (_eigenvalues(nx, dx, open_x, half=(open_y and not open_x))[:, None]
           + _eigenvalues(ny, dy, open_y, half=(open_x and not open_y))[None, :])
    lam[0, 0] = -1.0 / dx**2          # singular (0,0) mode: dummy here, solved separately below

    a, b, c = _vertical_coefficients(g, rho0_c, rho0_w)
    if kernels is not None and getattr(cfg, 'USE_NUMBA', True):
        phi_hat = kernels.solve_tridiag_modes(a, b, c, np.ascontiguousarray(lam),
                                              np.ascontiguousarray(rhs_hat))
    else:
        ones = np.ones(rhs_hat.shape)
        phi_hat = _solve_tridiagonal(a * ones, (b + lam[:, :, None]) * ones, c * ones,
                                     rhs_hat, nz)

    # (0,0) mode: singular (constant φ). The boundary fluxes are balanced, so the system is
    # consistent; pin φ̂ at the lowest level and solve the remaining rows exactly.
    a0, b0, c0 = a.copy(), b.copy(), c.copy()
    b0[0], c0[0] = 1.0, 0.0
    d0 = rhs_hat[0, 0, :].copy()
    d0[0] = 0.0
    phi_hat[0, 0, :] = _solve_tridiagonal(a0, b0, c0, d0, nz)

    phi = _inverse(phi_hat, nx, ny, open_x, open_y)

    # Velocity correction; on open boundary faces ∂φ/∂n = 0, so u_n stays as given
    dpx = np.zeros_like(u)
    dpx[1:-1] = (phi[1:] - phi[:-1]) / dx
    if not open_x:
        dpx[0] = (phi[0] - phi[-1]) / dx
        dpx[-1] = dpx[0]
    u -= dt * dpx
    dpy = np.zeros_like(v)
    dpy[:, 1:-1] = (phi[:, 1:] - phi[:, :-1]) / dy
    if not open_y:
        dpy[:, 0] = (phi[:, 0] - phi[:, -1]) / dy
        dpy[:, -1] = dpy[:, 0]
    v -= dt * dpy
    dpz = np.zeros_like(w)
    dpz[:, :, 1:-1] = (phi[:, :, 1:] - phi[:, :, :-1]) / dz_c[np.newaxis, np.newaxis, :]
    w -= dt * dpz
    w[:, :, 0] = 0.0
    w[:, :, -1] = 0.0
    if not open_x:
        u[-1] = u[0]
    if not open_y:
        v[:, -1] = v[:, 0]
    return phi


def _constant_theta_rho0(z: np.ndarray) -> np.ndarray:
    """Hydrostatic ρ₀(z) for constant θ = THETA_S (fallback when no State is available)."""
    import config as cfg
    pi0 = np.ones(len(z))
    for k in range(1, len(z)):
        pi0[k] = pi0[k - 1] - cfg.G * (z[k] - z[k - 1]) / (cfg.CP * cfg.THETA_S)
    return cfg.P00 * pi0 ** (cfg.CP / cfg.RD) / (cfg.RD * cfg.THETA_S * pi0)


# ── Thomas algorithm for tridiagonal systems ──────────────────────────────────

def _solve_tridiagonal(
    a: np.ndarray,   # (nx, ny//2+1, nz)  sub-diagonal (a[...,0] unused)
    b: np.ndarray,   # (nx, ny//2+1, nz)  diagonal
    c: np.ndarray,   # (nx, ny//2+1, nz)  super-diagonal (c[...,-1] unused)
    d: np.ndarray,   # (nx, ny//2+1, nz)  RHS
    nz: int,
) -> np.ndarray:
    """Vectorised Thomas algorithm over the first two axes."""
    b = b.copy()
    d = d.copy()
    x = np.zeros_like(d)

    # Forward sweep
    for k in range(1, nz):
        m = a[..., k] / b[..., k - 1]
        b[..., k] -= m * c[..., k - 1]
        d[..., k] -= m * d[..., k - 1]

    # Back substitution
    x[..., -1] = d[..., -1] / b[..., -1]
    for k in range(nz - 2, -1, -1):
        x[..., k] = (d[..., k] - c[..., k] * x[..., k + 1]) / b[..., k]

    return x

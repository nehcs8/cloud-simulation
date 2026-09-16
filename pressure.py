"""
FFT-based pressure projection (incompressibility enforcement).

Solves: ∇²φ = (∇·u*) / dt
then corrects:  u ← u* − dt ∂φ/∂x
                v ← v* − dt ∂φ/∂y
                w ← w* − dt ∂φ/∂z

Periodic BCs in x and y; Neumann (∂φ/∂z = 0) at top and bottom.
w is face-staggered in z.
"""

import numpy as np
from numpy.fft import rfft2, irfft2
from grid import Grid


def project(u: np.ndarray, v: np.ndarray, w: np.ndarray,
            g: Grid, dt: float):
    """
    In-place velocity projection.  Returns the pressure φ (nx,ny,nz).
    """
    nx, ny, nz = g.nx, g.ny, g.nz
    dx, dy = g.dx, g.dy
    dz_f = g.dz_face    # (nz,)   cell heights
    dz_c = g.dz_cen     # (nz-1,) inter-centre spacings

    # ── Divergence of u*  (uses staggered w) ──────────────────────────────────
    # Centred differences in x,y (periodic)
    div_u = (np.roll(u, -1, 0) - np.roll(u, 1, 0)) / (2.0 * dx)
    div_v = (np.roll(v, -1, 1) - np.roll(v, 1, 1)) / (2.0 * dy)
    # Finite difference of staggered w across each cell
    div_w = (w[:, :, 1:] - w[:, :, :-1]) / dz_f[np.newaxis, np.newaxis, :]

    rhs = (div_u + div_v + div_w) / dt    # (nx,ny,nz)

    # ── 2-D FFT in x-y ────────────────────────────────────────────────────────
    rhs_hat = rfft2(rhs, axes=(0, 1))                     # (nx, ny//2+1, nz)

    # Modified wavenumbers for 2nd-order centred FD on periodic grid
    kx_idx = np.fft.fftfreq(nx, d=1.0 / nx).astype(int)
    ky_idx = np.arange(ny // 2 + 1)

    lam_x = (2.0 * (np.cos(2.0 * np.pi * kx_idx / nx) - 1.0) / dx**2)[:, np.newaxis]  # (nx,1)
    lam_y = (2.0 * (np.cos(2.0 * np.pi * ky_idx / ny) - 1.0) / dy**2)[np.newaxis, :]  # (1,ny//2+1)
    lam_xy = (lam_x + lam_y)[:, :, np.newaxis]            # (nx, ny//2+1, 1)

    # ── Tridiagonal solve in z for each (kx,ky) mode ──────────────────────────
    # Equation: L_z φ̂ + λ_xy φ̂ = R̂
    # L_z is the non-uniform 2nd-order z-Laplacian with Neumann BCs.
    #
    # Interior k (1..nz-2):
    #   a_k φ̂_{k-1} + b_k φ̂_k + c_k φ̂_{k+1} = R̂_k
    #   a_k = 1 / (dz_c[k-1] * dz_f[k])
    #   c_k = 1 / (dz_c[k]   * dz_f[k])
    #   b_k = -(a_k + c_k) + λ_xy
    # Boundary k=0:    no lower neighbour → a_0 = 0, c_0 = 1/(dz_c[0]*dz_f[0])
    # Boundary k=nz-1: no upper neighbour → c_{nz-1} = 0

    a = np.zeros(nz)
    b = np.zeros(nz)
    c = np.zeros(nz)

    # Lower coefficients (a: sub-diagonal, from k-1)
    a[1:] = 1.0 / (dz_c * dz_f[1:])        # nz-1 values
    # Upper coefficients (c: super-diagonal, to k+1)
    c[:-1] = 1.0 / (dz_c * dz_f[:-1])      # nz-1 values
    # Diagonal (without λ_xy)
    b = -(a + c)

    # For k=0: Neumann → φ_ghost = φ_0 → no change (a[0]=0 already)
    # For k=nz-1: Neumann → φ_ghost = φ_{nz-1} → c[nz-1]=0 already

    # Reshape for broadcasting: a,b,c → (1,1,nz)
    a3 = a[np.newaxis, np.newaxis, :]
    b3 = b[np.newaxis, np.newaxis, :]
    c3 = c[np.newaxis, np.newaxis, :]

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
    phi = irfft2(phi_hat, s=(nx, ny), axes=(0, 1))        # (nx,ny,nz)

    # ── Correct velocities ────────────────────────────────────────────────────
    u -= dt * (np.roll(phi, -1, 0) - np.roll(phi, 1, 0)) / (2.0 * dx)
    v -= dt * (np.roll(phi, -1, 1) - np.roll(phi, 1, 1)) / (2.0 * dy)

    # w correction on interior faces: dφ/dz between cell centres k and k+1
    dphi_dz = np.zeros((g.nx, g.ny, g.nz + 1))
    dphi_dz[:, :, 1:-1] = (phi[:, :, 1:] - phi[:, :, :-1]) / dz_c[np.newaxis, np.newaxis, :]
    w -= dt * dphi_dz
    # Enforce w=0 at boundaries
    w[:, :, 0]  = 0.0
    w[:, :, -1] = 0.0

    # Enforce zero domain-mean w at every interior face.
    # Required by Boussinesq continuity + periodic BCs + rigid lid:
    # ∂<w>/∂z = 0 and <w>(z=0) = 0  →  <w>(z) = 0 everywhere.
    # The (0,0) Fourier mode of the Poisson equation is singular (Neumann
    # Laplacian), so the standard solver cannot correct the mean; we do it
    # explicitly here instead.
    w[:, :, 1:-1] -= w[:, :, 1:-1].mean(axis=(0, 1), keepdims=True)

    return phi


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

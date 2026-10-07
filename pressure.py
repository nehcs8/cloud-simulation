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

    # ── Enforce periodic boundary conditions on face-staggered velocities ────────
    # For C-grid with periodic BCs, the "extra" face is the same as the first face
    # u[nx] = u[0] (same x-face due to wraparound)
    # v[:, ny] = v[:, 0] (same y-face due to wraparound)
    u[-1, :, :] = u[0, :, :]
    v[:, -1, :] = v[:, 0, :]

    # ── Divergence of u*  (C-grid: all velocities face-staggered) ─────────────
    # u is on x-faces (nx+1, ny, nz): simple difference gives cell-centered divergence
    # v is on y-faces (nx, ny+1, nz): simple difference gives cell-centered divergence
    # w is on z-faces (nx, ny, nz+1): simple difference gives cell-centered divergence
    #
    # This matches the Poisson operator eigenvalues exactly! (see plan line 36-38)

    # du/dx: difference across each cell
    # u[i+1, j, k] is at face i+1/2; u[i, j, k] is at face i-1/2
    # cell i has left face i-1/2 and right face i+1/2
    div_u = (u[1:, :, :] - u[:-1, :, :]) / dx         # (nx, ny, nz)

    # dv/dy: difference across each cell
    div_v = (v[:, 1:, :] - v[:, :-1, :]) / dy         # (nx, ny, nz)

    # dw/dz: difference across each cell (already correct)
    div_w = (w[:, :, 1:] - w[:, :, :-1]) / dz_f[np.newaxis, np.newaxis, :]  # (nx, ny, nz)

    rhs = (div_u + div_v + div_w) / dt                # (nx, ny, nz)

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

    # Enforce zero domain-mean w at every interior face.
    # Required by Boussinesq continuity + periodic BCs + rigid lid:
    # ∂<w>/∂z = 0 and <w>(z=0) = 0  →  <w>(z) = 0 everywhere.
    # The (0,0) Fourier mode of the Poisson equation is singular (Neumann
    # Laplacian), so the standard solver cannot correct the mean; we do it
    # explicitly here instead.
    w[:, :, 1:-1] -= w[:, :, 1:-1].mean(axis=(0, 1), keepdims=True)

    # ── Re-enforce periodic BCs after gradient correction ────────────────────
    # This ensures the redundant faces stay equal (important for consistency)
    u[-1, :, :] = u[0, :, :]
    v[:, -1, :] = v[:, 0, :]

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

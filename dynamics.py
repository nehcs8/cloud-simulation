"""
Advection and SGS diffusion tendencies (C-GRID VERSION).

C-grid staggering:
  - u: x-faces (nx+1, ny, nz)
  - v: y-faces (nx, ny+1, nz)
  - w: z-faces (nx, ny, nz+1)
  - scalars (theta, qv, qc, qr): cell centers (nx, ny, nz)

Advection (Phase 1.3 - consistent high-order):
  - Scalars: WENO-3 in all directions (3rd-order, less diffusive)
  - Momentum: upwind-1 for stability (to be upgraded later)

Diffusion: Smagorinsky-Lilly SGS with spatially varying K_M and K_H.
Lateral boundaries: horizontal neighbours come from halo-padded copies (halo.py), periodic
or open per direction (cfg.BC_X / BC_Y).
Vertical: zero-gradient (Neumann) BCs; w=0 on top/bottom faces (rigid lid).
"""

import numpy as np
import config as cfg
from grid import Grid
from state import State
import halo
import open_bc
from halo import H

try:
    import kernels
except ImportError:          # numba not installed → numpy path only
    kernels = None


def _numba() -> bool:
    """Use the kernels.py path (checked per call, so tests can toggle cfg.USE_NUMBA)."""
    return kernels is not None and getattr(cfg, 'USE_NUMBA', True)

# WENO smoothness-indicator floor
# NOTE: This will be scaled by local field magnitude to handle different variable scales
# (e.g., θ ~300 K vs q_t ~1e-8 kg/kg)
_EPS_REL = 1e-6  # Relative epsilon (fraction of local field magnitude)


# ══ WENO-3 advection (for cell-centered scalars) ═════════════════════════════

def _weno3_faces(m1, c, p1, p2, vel, eps):
    """WENO-3 flux vel·φ at the face between cells c and p1 (arrays of equal shape)."""
    up = np.maximum(vel, 0.0)
    un = np.minimum(vel, 0.0)

    # left-biased reconstruction (upwind for vel > 0)
    q0L = -0.5*m1 + 1.5*c;   q1L = 0.5*c + 0.5*p1
    b0L = (c - m1)**2;        b1L = (p1 - c)**2
    # Standard WENO-3 weights: 1/3 upwind, 2/3 centered
    a0L = (1/3) / (eps + b0L)**2;  a1L = (2/3) / (eps + b1L)**2
    w0L = a0L / (a0L + a1L)
    fL  = up * (w0L*q0L + (1 - w0L)*q1L)

    # right-biased reconstruction (upwind for vel < 0)
    q0R = 1.5*p1 - 0.5*p2;   q1R = 0.5*c + 0.5*p1
    b0R = (p2 - p1)**2;       b1R = (p1 - c)**2
    a0R = (1/3) / (eps + b0R)**2;  a1R = (2/3) / (eps + b1R)**2
    w0R = a0R / (a0R + a1R)
    fR  = un * (w0R*q0R + (1 - w0R)*q1R)
    return fL + fR


def _weno_eps(c):
    """Smoothness floor scaled by the local field magnitude (θ ~300 K vs q_t ~1e-8)."""
    return _EPS_REL * np.maximum(np.abs(c), 1e-12)**2


def _weno3_x(P: np.ndarray, u: np.ndarray, dx: float) -> np.ndarray:
    """
    Flux-form WENO-3 d(u·φ)/dx for a cell-centred scalar.
    P: φ halo-padded (halo.pad_c, H = 2); u: (nx+1, ny, nz) all x-faces.
    Fluxes on faces 0…nx; face m lies between cells m−1 and m.
    """
    nx, ny = u.shape[0] - 1, u.shape[1]
    Y = slice(H, H + ny)
    c = P[H-1:H+nx, Y]
    F = _weno3_faces(P[H-2:H+nx-1, Y], c, P[H:H+nx+1, Y], P[H+1:H+nx+2, Y], u, _weno_eps(c))
    return (F[1:] - F[:-1]) / dx


def _weno3_y(P: np.ndarray, v: np.ndarray, dy: float) -> np.ndarray:
    """Flux-form WENO-3 d(v·φ)/dy; P halo-padded, v: (nx, ny+1, nz) all y-faces."""
    nx, ny = v.shape[0], v.shape[1] - 1
    X = slice(H, H + nx)
    c = P[X, H-1:H+ny]
    F = _weno3_faces(P[X, H-2:H+ny-1], c, P[X, H:H+ny+1], P[X, H+1:H+ny+2], v, _weno_eps(c))
    return (F[:, 1:] - F[:, :-1]) / dy


def _weno3_z(phi: np.ndarray, w_face: np.ndarray,
             dz_face: np.ndarray, rho0_c: np.ndarray = None, rho0_w: np.ndarray = None) -> np.ndarray:
    """
    Anelastic flux-form WENO-3: -(1/ρ₀) ∂(ρ₀·w·phi)/∂z for a cell-centred scalar.

    w_face: (nx,ny,nz+1), face-staggered; rigid-lid enforces w=0 at k=0 and k=nz.
    Neumann BCs for phi (zero-gradient ghost cells at top/bottom).
    rho0_c: (nz,) reference density at cell centers (optional, defaults to 1.0)
    rho0_w: (nz+1,) reference density at w-faces (optional, defaults to 1.0)

    Returns: -(1/ρ₀) * ∂(ρ₀*w*φ)/∂z
    """
    if rho0_c is None:
        rho0_c = np.ones(phi.shape[2])
    if rho0_w is None:
        rho0_w = np.ones(w_face.shape[2])
    nz = phi.shape[2]
    dz = dz_face[np.newaxis, np.newaxis, :]    # (1,1,nz)

    # two Neumann ghost cells each side
    p = np.concatenate([phi[:,:,:1], phi[:,:,:1], phi,
                        phi[:,:,-1:], phi[:,:,-1:]], axis=2)   # (nx,ny,nz+4)

    # stencil for face k+1/2  (k = 0 … nz-1)
    pm1 = p[:,:, 1:nz+1]    # cell k-1
    pk  = p[:,:, 2:nz+2]    # cell k
    pp1 = p[:,:, 3:nz+3]    # cell k+1
    pp2 = p[:,:, 4:nz+4]    # cell k+2

    wf = w_face[:,:, 1:]    # w at face k+1/2, shape (nx,ny,nz); w_face[...,nz]=0 → wf[...,-1]=0
    wp = np.maximum(wf, 0.0)
    wn = np.minimum(wf, 0.0)

    # Anelastic: weight by ρ₀ at w-faces
    # rho0_w has shape (nz+1), we need faces 1:nz (same as wf)
    rho0_wf = rho0_w[np.newaxis, np.newaxis, 1:]  # (1,1,nz)

    # Adaptive epsilon scaled by local field magnitude
    phi_scale = np.maximum(np.abs(phi), 1e-12)
    eps = _EPS_REL * phi_scale**2

    # left-biased (w > 0)
    q0L = -0.5*pm1 + 1.5*pk;  q1L = 0.5*pk + 0.5*pp1
    b0L = (pk - pm1)**2;        b1L = (pp1 - pk)**2
    # Standard WENO-3 weights: 1/3 upwind, 2/3 centered
    a0L = (1/3) / (eps + b0L)**2;  a1L = (2/3) / (eps + b1L)**2
    w0L = a0L / (a0L + a1L)
    fL  = rho0_wf * wp * (w0L*q0L + (1 - w0L)*q1L)  # Anelastic: ρ₀*w*φ

    # right-biased (w < 0)
    q0R = 1.5*pp1 - 0.5*pp2;  q1R = 0.5*pk + 0.5*pp1
    b0R = (pp2 - pp1)**2;       b1R = (pp1 - pk)**2
    # Standard WENO-3 weights: 1/3 upwind, 2/3 centered
    a0R = (1/3) / (eps + b0R)**2;  a1R = (2/3) / (eps + b1R)**2
    w0R = a0R / (a0R + a1R)
    fR  = rho0_wf * wn * (w0R*q0R + (1 - w0R)*q1R)  # Anelastic: ρ₀*w*φ

    # flux at face k+1/2: ρ₀*w*φ
    # top face: wf[...,-1]=0 → F[...,-1]=0 automatically
    F     = fL + fR
    # bottom face rigid lid → flux = 0
    F_km1 = np.concatenate([np.zeros_like(F[:,:,:1]), F[:,:,:-1]], axis=2)

    # Anelastic form: -(1/ρ₀) * ∂(ρ₀*w*φ)/∂z
    rho0_c_3d = rho0_c[np.newaxis, np.newaxis, :]  # (1,1,nz)
    return (F - F_km1) / (rho0_c_3d * dz)


# ══ Momentum advection (C-grid face-staggered) ═══════════════════════════════

def advect_momentum(UP, VP, WP, g, rho0_c, rho0_w):
    """
    Anelastic flux-form C-grid momentum advection (energy-conserving
    for anelastic flow with div(rho0*u) = 0).

    Tendencies: -(1/rho0) * div(rho0 * u * phi)
    - For u, v: rho0 at cell centers for vertical flux
    - For w: rho0 at w-faces for vertical flux

    Inputs are halo-padded (H = 2): UP = halo.pad_u(u), VP = halo.pad_v(v), WP = halo.pad_c(w).
    Returns tendencies (tu, tv, tw) with shapes:
      - tu: (nx+1, ny, nz) on all x-faces (incl. both boundary faces)
      - tv: (nx, ny+1, nz) on all y-faces
      - tw: (nx, ny, nz+1) for w on z-faces

    This scheme conserves kinetic energy in the inviscid limit, preventing
    the energy blow-up that occurs with advective-form schemes.
    """
    dx, dy, dzf, dzc = g.dx, g.dy, g.dz_face, g.dz_cen
    nx, ny = g.nx, g.ny
    Xc, Yc = slice(H, H + nx), slice(H, H + ny)            # cells 0…n−1
    Xf, Yf = slice(H, H + nx + 1), slice(H, H + ny + 1)    # faces 0…n (or cells 0…n)
    Xm, Ym = slice(H - 1, H + nx), slice(H - 1, H + ny)    # one to the left of Xf / Yf
    Xp, Yp = slice(H + 1, H + nx + 2), slice(H + 1, H + ny + 2)

    # Expand rho0 profiles for broadcasting
    rho0_c_3d = rho0_c[np.newaxis, np.newaxis, :]  # (1, 1, nz)
    rho0_w_3d = rho0_w[np.newaxis, np.newaxis, :]  # (1, 1, nz+1)
    dzf_3d = dzf[np.newaxis, np.newaxis, :]
    rho0_wf = rho0_w_3d[..., 1:-1]                 # (1, 1, nz-1) at interior w-faces

    # u-momentum on x-faces 0…nx: -(1/rho0) * div(rho0 * u * u)
    U = UP[Xf, Yc]
    ucp = 0.5*(U + UP[Xp, Yc]); ucm = 0.5*(UP[Xm, Yc] + U)
    adv_u = (ucp*ucp - ucm*ucm)/dx
    Yc_m, Yc_p = slice(H - 1, H + ny - 1), slice(H + 1, H + ny + 1)
    Fyx_c = 0.5*(VP[Xm, Yc] + VP[Xf, Yc]) * 0.5*(UP[Xf, Yc_m] + U)
    Fyx_p = 0.5*(VP[Xm, Yc_p] + VP[Xf, Yc_p]) * 0.5*(U + UP[Xf, Yc_p])
    adv_u += (Fyx_p - Fyx_c)/dy
    # Vertical flux: -(1/rho0_c) * d(rho0_w * w * u)/dz, w averaged to the u location
    uz = 0.5*(U[..., :-1] + U[..., 1:])
    wf_for_u = 0.5*(WP[Xm, Yc] + WP[Xf, Yc])
    Fzw_u = np.zeros_like(wf_for_u)
    Fzw_u[..., 1:-1] = rho0_wf * wf_for_u[..., 1:-1] * uz
    adv_u += (Fzw_u[..., 1:] - Fzw_u[..., :-1]) / (rho0_c_3d * dzf_3d)

    # v-momentum on y-faces 0…ny
    V = VP[Xc, Yf]
    vcp = 0.5*(V + VP[Xc, Yp]); vcm = 0.5*(VP[Xc, Ym] + V)
    adv_v = (vcp*vcp - vcm*vcm)/dy
    Xc_m, Xc_p = slice(H - 1, H + nx - 1), slice(H + 1, H + nx + 1)
    Fxy_c = 0.5*(UP[Xc, Ym] + UP[Xc, Yf]) * 0.5*(VP[Xc_m, Yf] + V)
    Fxy_p = 0.5*(UP[Xc_p, Ym] + UP[Xc_p, Yf]) * 0.5*(V + VP[Xc_p, Yf])
    adv_v += (Fxy_p - Fxy_c)/dx
    vz = 0.5*(V[..., :-1] + V[..., 1:])
    wf_for_v = 0.5*(WP[Xc, Ym] + WP[Xc, Yf])
    Fzw_v = np.zeros_like(wf_for_v)
    Fzw_v[..., 1:-1] = rho0_wf * wf_for_v[..., 1:-1] * vz
    adv_v += (Fzw_v[..., 1:] - Fzw_v[..., :-1]) / (rho0_c_3d * dzf_3d)

    # w-momentum: -(1/rho0_w) * div(rho0 * u * w) at interior w-faces of each cell
    a = dzf[:-1]/(dzf[:-1]+dzf[1:]); b = dzf[1:]/(dzf[:-1]+dzf[1:])
    Uf = UP[Xf, Yc]; Vf = VP[Xc, Yf]
    Uw = a*Uf[..., :-1] + b*Uf[..., 1:]; Vw = a*Vf[..., :-1] + b*Vf[..., 1:]
    Fxw = Uw*0.5*(WP[Xm, Yc, 1:-1] + WP[Xf, Yc, 1:-1])          # x-faces 0…nx
    Fyw = Vw*0.5*(WP[Xc, Ym, 1:-1] + WP[Xc, Yf, 1:-1])          # y-faces 0…ny
    W = WP[Xc, Yc]
    wc = 0.5*(W[..., :-1] + W[..., 1:])            # w at cell centres
    Fzz = rho0_c_3d * wc * wc
    dzc_3d = dzc[np.newaxis, np.newaxis, :]
    adv_w = np.zeros_like(W)
    adv_w[..., 1:-1] = ((Fxw[1:]-Fxw[:-1])/dx + (Fyw[:, 1:]-Fyw[:, :-1])/dy
                        + (Fzz[..., 1:]-Fzz[..., :-1]) / (rho0_wf * dzc_3d))

    return -adv_u, -adv_v, -adv_w


# ══ Smagorinsky-Lilly SGS (C-grid) ═══════════════════════════════════════════

def _smag_K(s: State, g: Grid):
    """
    Spatially varying eddy diffusivities via Smagorinsky-Lilly on C-grid:
        K_M = (Cs · Δ)²  |S|
        K_H = K_M / Pr_t
    Returns K_M, K_H both shape (nx, ny, nz) at cell centers.
    Horizontal neighbours from halo-padded cell-centre velocities.
    """
    dx, dy = g.dx, g.dy
    nx, ny = g.nx, g.ny
    dz3 = g.dz_face[np.newaxis, np.newaxis, :]     # (1,1,nz)

    # Interpolate face velocities to cell centers, then pad
    UC = halo.pad_c(s.u_at_cell())
    VC = halo.pad_c(s.v_at_cell())
    WC = halo.pad_c(s.w_at_cell())

    # Terrain: no K in solid cells, and one-sided du/dz, dv/dz at each column's first fluid cell
    # (a centred difference would see the solid cell's u = 0 as no-slip shear on a free-slip ground)
    kb = np.zeros((nx, ny), dtype=np.int64) if g.terrain is None else g.terrain.k_sfc.astype(np.int64)

    if _numba():
        K_M = kernels.smag_KM(UC, VC, WC, s.w, g.z, dx, dy, g.dz_face, cfg.CS, kb)
        return K_M, K_M / cfg.PR_T

    X, Y = slice(H, H + nx), slice(H, H + ny)
    Xp, Xm = slice(H + 1, H + nx + 1), slice(H - 1, H + nx - 1)
    Yp, Ym = slice(H + 1, H + ny + 1), slice(H - 1, H + ny - 1)
    u_cen, v_cen = UC[X, Y], VC[X, Y]

    # Horizontal gradients (centred)
    dudx = (UC[Xp, Y] - UC[Xm, Y]) / (2*dx)
    dvdy = (VC[X, Yp] - VC[X, Ym]) / (2*dy)
    dudy = (UC[X, Yp] - UC[X, Ym]) / (2*dy)
    dvdx = (VC[Xp, Y] - VC[Xm, Y]) / (2*dx)

    # dw/dz from staggered faces (exact on this grid)
    dwdz = (s.w[:,:,1:] - s.w[:,:,:-1]) / dz3

    # Horizontal derivatives of w
    dwdx  = (WC[Xp, Y] - WC[Xm, Y]) / (2*dx)
    dwdy  = (WC[X, Yp] - WC[X, Ym]) / (2*dy)

    # du/dz, dv/dz — centred differences, handles non-uniform spacing
    dudz = np.gradient(u_cen, g.z, axis=2)
    dvdz = np.gradient(v_cen, g.z, axis=2)
    if g.terrain is not None:
        I, J = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
        top = kb < g.nz - 1
        I, J, k0 = I[top], J[top], kb[top]
        h1 = g.z[k0 + 1] - g.z[k0]
        dudz[I, J, k0] = (u_cen[I, J, k0 + 1] - u_cen[I, J, k0]) / h1
        dvdz[I, J, k0] = (v_cen[I, J, k0 + 1] - v_cen[I, J, k0]) / h1

    # strain-rate magnitude squared: 2 S_ij S_ij
    S2 = (2*(dudx**2 + dvdy**2 + dwdz**2)
          + (dudy + dvdx)**2
          + (dudz + dwdx)**2
          + (dvdz + dwdy)**2)

    Delta = (dx * dy * g.dz_face[np.newaxis, np.newaxis, :])**(1/3)
    K_M   = (cfg.CS * Delta)**2 * np.sqrt(np.maximum(S2, 0.0))
    if g.terrain is not None:
        K_M[np.arange(g.nz)[np.newaxis, np.newaxis, :] < kb[:, :, np.newaxis]] = 0.0
    K_H   = K_M / cfg.PR_T
    return K_M, K_H


# ══ Variable-K diffusion (for cell-centered fields) ══════════════════════════

def _kdiff_x(P, KP, dx, mask=None):
    """∂/∂x(K ∂φ/∂x) for a cell-centred field; φ and K halo-padded, fluxes on faces 0…nx.
    mask (nx+1, ny, nz): optional face mask (terrain; 0 = closed face)."""
    nx, ny = P.shape[0] - 2*H, P.shape[1] - 2*H
    Y = slice(H, H + ny)
    Kf   = 0.5*(KP[H-1:H+nx, Y] + KP[H:H+nx+1, Y])
    flux = Kf * (P[H:H+nx+1, Y] - P[H-1:H+nx, Y]) / dx
    if mask is not None:
        flux = mask * flux
    return (flux[1:] - flux[:-1]) / dx

def _kdiff_y(P, KP, dy, mask=None):
    """∂/∂y(K ∂φ/∂y); φ and K halo-padded, fluxes on faces 0…ny; optional face mask (nx, ny+1, nz)."""
    nx, ny = P.shape[0] - 2*H, P.shape[1] - 2*H
    X = slice(H, H + nx)
    Kf   = 0.5*(KP[X, H-1:H+ny] + KP[X, H:H+ny+1])
    flux = Kf * (P[X, H:H+ny+1] - P[X, H-1:H+ny]) / dy
    if mask is not None:
        flux = mask * flux
    return (flux[:, 1:] - flux[:, :-1]) / dy

def _kdiff_z(phi, K, dz_face, dz_cen, rho0_c=None, rho0_w=None, mask=None):
    """
    Anelastic flux-form variable-K diffusion in z: (1/rho0) d/dz(rho0 * K * dphi/dz)
    Neumann BCs (zero flux at boundaries).

    rho0_c: (nz,) reference density at cell centers (optional, defaults to 1.0)
    rho0_w: (nz+1,) reference density at w-faces (optional, defaults to 1.0)
    """
    if rho0_c is None:
        rho0_c = np.ones(phi.shape[2])
    if rho0_w is None:
        rho0_w = np.ones(dz_face.shape[0] + 1)

    Kf       = 0.5*(K[:,:,:-1] + K[:,:,1:])        # (nx,ny,nz-1) at interior faces
    dz_c     = dz_cen[np.newaxis, np.newaxis, :]

    # Anelastic flux at interior w-faces: rho0_w * K * dphi/dz
    # rho0 at interior w-faces (1:nz)
    rho0_wf = rho0_w[np.newaxis, np.newaxis, 1:-1]  # (1, 1, nz-1)
    flux_int = rho0_wf * Kf * (phi[:,:,1:] - phi[:,:,:-1]) / dz_c
    if mask is not None:                       # terrain face mask (nx, ny, nz+1), interior faces
        flux_int = mask[:, :, 1:-1] * flux_int

    zeros    = np.zeros_like(phi[:,:,:1])
    flux     = np.concatenate([zeros, flux_int, zeros], axis=2)   # (nx,ny,nz+1)

    # Anelastic tendency: (1/rho0_c) * d(flux)/dz
    rho0_c_3d = rho0_c[np.newaxis, np.newaxis, :]
    dz_f = dz_face[np.newaxis, np.newaxis, :]
    return (flux[:,:,1:] - flux[:,:,:-1]) / (rho0_c_3d * dz_f)


# ══ Terrain helpers ══════════════════════════════════════════════════════════

def _ground_fill(phi: np.ndarray, ter) -> np.ndarray:
    """Copy of phi with solid cells set to their column's ground-level (first fluid) value."""
    sfc = np.take_along_axis(phi, ter.k_sfc[:, :, np.newaxis], axis=2)
    return np.where(ter.mask, sfc, phi)


def _terrain_masks(g: Grid, rho0_c, rho0_w, open_x: bool, open_y: bool) -> dict:
    """Face masks for diffusion with terrain, cached: mx, my = open fractions of the side faces
    (pressure._terrain_setup apertures; 0/1 for the staircase), mz = open horizontal faces, wmx, wmy
    for the horizontal diffusion of w on interior levels, hc = cell fluid fractions, ax/ay apertures."""
    import pressure
    cache = g.terrain.__dict__.setdefault('_stencil_masks', {})   # on the object (ids get reused)
    key = (open_x, open_y)
    if key not in cache:
        t = pressure._terrain_setup(g, rho0_c, rho0_w, open_x, open_y)
        wz = t['wz'][:, :, 1:-1]                       # fluid interior w faces (nx, ny, nz-1)
        nx, ny = g.nx, g.ny
        wmx = np.zeros((nx + 1, ny, g.nz - 1))
        wmx[1:-1] = wz[:-1] & wz[1:]
        wmx[0] = wz[0] if open_x else (wz[-1] & wz[0])
        wmx[-1] = wz[-1] if open_x else wmx[0]
        wmy = np.zeros((nx, ny + 1, g.nz - 1))
        wmy[:, 1:-1] = wz[:, :-1] & wz[:, 1:]
        wmy[:, 0] = wz[:, 0] if open_y else (wz[:, -1] & wz[:, 0])
        wmy[:, -1] = wz[:, -1] if open_y else wmy[:, 0]
        # diffusion: face fluxes scaled by the open fraction (partial cells; 0/1 for the staircase)
        cache[key] = dict(mx=t['ax'], my=t['ay'], mz=t['mz'], wmx=wmx, wmy=wmy, hc=t['hc'],
                              ax=t['ax'], ay=t['ay'])
    return cache[key]


# ══ Full tendency computation (C-GRID) ═══════════════════════════════════════

def compute_tendencies(s: State, g: Grid) -> dict:
    """
    Compute tendencies for all prognostic variables on C-grid.

    Phase 1.4: Transports conserved variables θ_l, qt (not θ, qv, qc separately).

    Returns dict with keys: u, v, w, theta_l, qt, qr
    Shapes:
      - u: (nx+1, ny, nz)
      - v: (nx, ny+1, nz)
      - w: (nx, ny, nz+1)
      - scalars: (nx, ny, nz)
    """
    dx, dy = g.dx, g.dy
    dz_f   = g.dz_face    # (nz,)
    dz_c   = g.dz_cen     # (nz-1,)

    open_x, open_y = halo.open_x(), halo.open_y()
    X, Y = slice(H, H + g.nx), slice(H, H + g.ny)          # interior of a padded array

    # ── Periodic directions: the last face is the first face ─────────────────
    # u is on x-faces (nx+1): u[0] and u[nx] are the SAME face (periodic wrap)
    # v is on y-faces (ny+1): v[0] and v[ny] are the SAME face (periodic wrap)
    if not open_x:
        s.u[-1, :, :] = s.u[0, :, :]
    if not open_y:
        s.v[:, -1, :] = s.v[:, 0, :]

    # ── Halo-padded copies (horizontal neighbours for all stencils) ──────────
    ter = g.terrain
    UP, VP, WP = halo.pad_u(s.u), halo.pad_v(s.v), halo.pad_c(s.w)
    if ter is None:
        P_thl, P_qt, P_qr = halo.pad_c(s.theta_l), halo.pad_c(s.qt), halo.pad_c(s.qr)
    else:
        # terrain: solid cells take their column's ground-level fluid value, so stencils next to
        # the ground see a zero-gradient continuation (advective fluxes through solid faces are
        # already zero because the projection zeroes those face velocities)
        P_thl, P_qt, P_qr = (halo.pad_c(_ground_fill(f, ter)) for f in (s.theta_l, s.qt, s.qr))
    if open_x or open_y:
        # inflow faces see the environment (sounding) in the ghost cells; outflow keeps zero gradient
        masks, ref = open_bc.inflow_masks(s), open_bc.references(s)
        for P, key in ((P_thl, 'theta_l'), (P_qt, 'qt'), (P_qr, 'qr')):
            open_bc.inflow_ghosts(P, ref[key], masks)

    # ── Scalar advection ──────────────────────────────────────────────────────
    # CONSISTENT WENO-3 for all scalars (Phase 1.3)
    # Phase 1.4: Transport conserved variables θ_l, qt (not θ, qv, qc)
    #
    # CRITICAL FIX: Use true divergence-free face velocities!
    # Passing cell-centered velocities and re-averaging breaks discrete conservation.
    # This was causing spurious sources/sinks of ~0.23/s in uniform-scalar test.
    # Fluxes use the C-grid face velocities on all faces 0…n (incl. both boundary faces).

    # Cell-centered w only for background gradient transport
    w_cen = s.w_at_cell()  # (nx, ny, nz)

    # Get reference density profiles for anelastic advection
    rho0_c = s.rho0  # (nz,) at cell centers
    # Interpolate to w-faces
    rho0_w = np.zeros(g.nz + 1)
    rho0_w[0] = rho0_c[0]  # Extrapolate to bottom
    rho0_w[-1] = rho0_c[-1]  # Extrapolate to top
    rho0_w[1:-1] = 0.5 * (rho0_c[:-1] + rho0_c[1:])  # Average to faces

    tm = None if ter is None else _terrain_masks(g, rho0_c, rho0_w, open_x, open_y)
    # transport velocities: with terrain the flux through a side face is (open fraction) × u
    u_tr, v_tr = (s.u, s.v) if tm is None else (tm['ax'] * s.u, tm['ay'] * s.v)

    def adv_weno3(P):
        """WENO-3 in all directions using true divergence-free face velocities.
        Anelastic: vertical advection uses ρ₀-weighted fluxes. P: halo-padded scalar."""
        if _numba():
            return kernels.weno3_div(P, u_tr, v_tr, s.w, dx, dy, dz_f, rho0_c, rho0_w)
        return (_weno3_x(P, u_tr, dx)
                + _weno3_y(P, v_tr, dy)
                + _weno3_z(P[X, Y], s.w, dz_f, rho0_c, rho0_w))

    # Transport conserved variables (Phase 1.4)
    tend_theta_l = -adv_weno3(P_thl)
    tend_qt      = -adv_weno3(P_qt)
    tend_qr      = -adv_weno3(P_qr)

    # ── Momentum advection ────────────────────────────────────────────────────
    # Anelastic flux-form energy-conserving scheme (prevents KE blow-up)
    if _numba():
        tend_u, tend_v, tend_w = kernels.momentum_adv(UP, VP, WP, dx, dy, dz_f, dz_c,
                                                      rho0_c, rho0_w)
    else:
        tend_u, tend_v, tend_w = advect_momentum(UP, VP, WP, g, rho0_c, rho0_w)

    # ── Smagorinsky SGS diffusion ─────────────────────────────────────────────
    K_M, K_H = _smag_K(s, g)
    P_KH = halo.pad_c(K_H)

    # Scalar diffusion (cell-centered fields)
    # Phase 1.4: Diffuse conserved variables θ_l, qt
    # Anelastic: vertical diffusion uses (1/rho0) d/dz(rho0 * K * dphi/dz)
    def diff_s(P):
        if tm is not None:                       # terrain: no diffusive flux through solid faces
            if _numba():
                return kernels.kdiff_xyz_masked(P, P_KH, tm['mx'], tm['my'], tm['mz'], dx, dy,
                                                dz_f, dz_c, rho0_c, rho0_w)
            return (_kdiff_x(P, P_KH, dx, tm['mx']) + _kdiff_y(P, P_KH, dy, tm['my'])
                    + _kdiff_z(P[X, Y], K_H, dz_f, dz_c, rho0_c, rho0_w, tm['mz']))
        if _numba():
            return kernels.kdiff_xyz(P, P_KH, dx, dy, dz_f, dz_c, rho0_c, rho0_w)
        return (_kdiff_x(P, P_KH, dx)
                + _kdiff_y(P, P_KH, dy)
                + _kdiff_z(P[X, Y], K_H, dz_f, dz_c, rho0_c, rho0_w))

    tend_theta_l += diff_s(P_thl)
    tend_qt      += diff_s(P_qt)
    tend_qr      += diff_s(P_qr)
    if tm is not None:
        # finite volume: face fluxes per unit cell volume → per unit fluid volume (partial cells)
        hc = tm['hc']
        tend_theta_l, tend_qt, tend_qr = (np.where(hc > 0.0, t_ / np.where(hc > 0.0, hc, 1.0), 0.0)
                                          for t_ in (tend_theta_l, tend_qt, tend_qr))

    # Momentum diffusion - apply to cell-centered, then interpolate back
    # (Simplified - proper C-grid diffusion would be on faces)
    # Anelastic: vertical diffusion uses rho0 weighting
    u_diff = diff_s(halo.pad_c(s.u_at_cell()))
    v_diff = diff_s(halo.pad_c(s.v_at_cell()))

    # Distribute back to faces (simplified). Boundary faces: periodic face 0 (= face n)
    # takes the cell next to it on one side as before; open faces take their adjacent cell.
    tend_u_diff = np.zeros_like(s.u)
    tend_u_diff[1:-1, :, :] = 0.5 * (u_diff[:-1, :, :] + u_diff[1:, :, :])
    tend_u_diff[0, :, :] = u_diff[0, :, :]
    tend_u_diff[-1, :, :] = u_diff[-1, :, :] if open_x else tend_u_diff[0, :, :]
    tend_u += tend_u_diff

    tend_v_diff = np.zeros_like(s.v)
    tend_v_diff[:, 1:-1, :] = 0.5 * (v_diff[:, :-1, :] + v_diff[:, 1:, :])
    tend_v_diff[:, 0, :] = v_diff[:, 0, :]
    tend_v_diff[:, -1, :] = v_diff[:, -1, :] if open_y else tend_v_diff[:, 0, :]
    tend_v += tend_v_diff

    # w diffusion (interior faces only)
    w_int = s.w[:,:,1:-1]
    K_M_wface = 0.5*(K_M[:,:,:-1] + K_M[:,:,1:])
    dz_mid = (dz_f[:-1] * dz_f[1:])[np.newaxis, np.newaxis, :]
    P_wi, P_Kw = halo.pad_c(w_int), halo.pad_c(K_M_wface)
    if tm is not None:
        if _numba():
            dw_h = kernels.kdiff_xy_masked(P_wi, P_Kw, tm['wmx'], tm['wmy'], dx, dy)
        else:
            dw_h = _kdiff_x(P_wi, P_Kw, dx, tm['wmx']) + _kdiff_y(P_wi, P_Kw, dy, tm['wmy'])
    elif _numba():
        dw_h = kernels.kdiff_xy(P_wi, P_Kw, dx, dy)
    else:
        dw_h = _kdiff_x(P_wi, P_Kw, dx) + _kdiff_y(P_wi, P_Kw, dy)
    dw_diff = (dw_h
               + K_M_wface * (s.w[:,:,2:] - 2.0*w_int + s.w[:,:,:-2]) / dz_mid)
    tend_w[:,:,1:-1] += dw_diff

    if open_x or open_y:
        # boundary-face normal velocity: radiation condition instead of the interior tendency
        open_bc.radiation_tendency(tend_u, tend_v, s, g)
        # outward total-water flux through the open boundaries [kg/s] (for the water budget);
        # q_t = q_v + q_c + q_r already includes rain
        s.boundary_water_rate = open_bc.boundary_outflow(P_qt, P_KH, s, g)

    # ── Coriolis force (f-plane) ──────────────────────────────────────────────
    # CRITICAL FIX: Include geostrophic wind terms!
    # Correct form: du/dt = +f*(v - v_g),  dv/dt = -f*(u - u_g(z))
    # Without this, wind rotates ~47° over 6h in BOMEX instead of staying steady
    if cfg.ENABLE_CORIOLIS:
        # v at u-faces: (nx+1, ny, nz)
        v_at_u = s.v_at_u_faces()
        tend_u += cfg.CORIOLIS_F * (v_at_u - cfg.V_GEO)

        # u at v-faces: (nx, ny+1, nz)
        u_at_v = s.u_at_v_faces()
        tend_v -= cfg.CORIOLIS_F * (u_at_v - s.u_g[np.newaxis, np.newaxis, :])

    # ── Buoyancy on w-faces ───────────────────────────────────────────────────
    B      = s.buoyancy()  # (nx, ny, nz) at cell centers
    B_face = np.zeros((g.nx, g.ny, g.nz + 1))
    B_face[:,:,1:-1] = 0.5*(B[:,:,:-1] + B[:,:,1:])
    tend_w += B_face

    # ── Background theta gradient carried by w ────────────────────────────────
    # Phase 1.4: This affects θ, which relates to θ_l through latent heating
    # For now, we still apply it to ensure energy consistency
    # (Could be refined: gradient of θ_l_0 instead of θ_0)
    dtheta0_dz = np.gradient(s.theta0, g.z)
    tend_theta_l -= w_cen * dtheta0_dz[np.newaxis, np.newaxis, :]

    return dict(u=tend_u, v=tend_v, w=tend_w,
                theta_l=tend_theta_l, qt=tend_qt, qr=tend_qr)

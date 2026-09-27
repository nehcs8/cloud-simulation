"""
Advection and SGS diffusion tendencies.

Advection: 3rd-order WENO for cell-centred scalars and momentum (u, v);
           1st-order upwind kept for face-staggered w (self-advection in z).
Diffusion:  Smagorinsky-Lilly SGS with spatially varying K_M and K_H.
Periodicity: x and y wrap with np.roll.
Vertical: zero-gradient (Neumann) BCs; w=0 on top/bottom faces (rigid lid).
"""

import numpy as np
import config as cfg
from grid import Grid
from state import State

_EPS = 1e-6    # WENO smoothness-indicator floor


# ══ WENO-3 advection ══════════════════════════════════════════════════════════

def _weno3_x(phi: np.ndarray, u: np.ndarray, dx: float) -> np.ndarray:
    """Flux-form WENO-3 d(u·phi)/dx, periodic in x."""
    m1 = np.roll(phi,  1, 0)
    p1 = np.roll(phi, -1, 0)
    p2 = np.roll(phi, -2, 0)
    uf = 0.5 * (u + np.roll(u, -1, 0))        # u at face i+1/2
    up = np.maximum(uf, 0.0)
    un = np.minimum(uf, 0.0)

    # left-biased reconstruction at i+1/2 (upwind for u > 0)
    q0L = -0.5*m1 + 1.5*phi;   q1L = 0.5*phi + 0.5*p1
    b0L = (phi - m1)**2;         b1L = (p1 - phi)**2
    a0L = (2/3) / (_EPS + b0L)**2;  a1L = (1/3) / (_EPS + b1L)**2
    w0L = a0L / (a0L + a1L)
    fL  = up * (w0L*q0L + (1 - w0L)*q1L)

    # right-biased reconstruction at i+1/2 (upwind for u < 0)
    q0R = 1.5*p1 - 0.5*p2;     q1R = 0.5*phi + 0.5*p1
    b0R = (p2 - p1)**2;          b1R = (p1 - phi)**2
    a0R = (2/3) / (_EPS + b0R)**2;  a1R = (1/3) / (_EPS + b1R)**2
    w0R = a0R / (a0R + a1R)
    fR  = un * (w0R*q0R + (1 - w0R)*q1R)

    F = fL + fR                                # flux at face i+1/2
    return (F - np.roll(F, 1, 0)) / dx        # divergence


def _weno3_y(phi: np.ndarray, v: np.ndarray, dy: float) -> np.ndarray:
    """Flux-form WENO-3 d(v·phi)/dy, periodic in y."""
    m1 = np.roll(phi,  1, 1)
    p1 = np.roll(phi, -1, 1)
    p2 = np.roll(phi, -2, 1)
    vf = 0.5 * (v + np.roll(v, -1, 1))
    vp = np.maximum(vf, 0.0)
    vn = np.minimum(vf, 0.0)

    q0L = -0.5*m1 + 1.5*phi;   q1L = 0.5*phi + 0.5*p1
    b0L = (phi - m1)**2;         b1L = (p1 - phi)**2
    a0L = (2/3) / (_EPS + b0L)**2;  a1L = (1/3) / (_EPS + b1L)**2
    w0L = a0L / (a0L + a1L)
    fL  = vp * (w0L*q0L + (1 - w0L)*q1L)

    q0R = 1.5*p1 - 0.5*p2;     q1R = 0.5*phi + 0.5*p1
    b0R = (p2 - p1)**2;          b1R = (p1 - phi)**2
    a0R = (2/3) / (_EPS + b0R)**2;  a1R = (1/3) / (_EPS + b1R)**2
    w0R = a0R / (a0R + a1R)
    fR  = vn * (w0R*q0R + (1 - w0R)*q1R)

    F = fL + fR
    return (F - np.roll(F, 1, 1)) / dy


def _weno3_z(phi: np.ndarray, w_face: np.ndarray,
             dz_face: np.ndarray) -> np.ndarray:
    """
    Flux-form WENO-3 d(w·phi)/dz for a cell-centred scalar.
    w_face: (nx,ny,nz+1), face-staggered; rigid-lid enforces w=0 at k=0 and k=nz.
    Neumann BCs for phi (zero-gradient ghost cells at top/bottom).
    """
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

    # left-biased (w > 0)
    q0L = -0.5*pm1 + 1.5*pk;  q1L = 0.5*pk + 0.5*pp1
    b0L = (pk - pm1)**2;        b1L = (pp1 - pk)**2
    a0L = (2/3) / (_EPS + b0L)**2;  a1L = (1/3) / (_EPS + b1L)**2
    w0L = a0L / (a0L + a1L)
    fL  = wp * (w0L*q0L + (1 - w0L)*q1L)

    # right-biased (w < 0)
    q0R = 1.5*pp1 - 0.5*pp2;  q1R = 0.5*pk + 0.5*pp1
    b0R = (pp2 - pp1)**2;       b1R = (pp1 - pk)**2
    a0R = (2/3) / (_EPS + b0R)**2;  a1R = (1/3) / (_EPS + b1R)**2
    w0R = a0R / (a0R + a1R)
    fR  = wn * (w0R*q0R + (1 - w0R)*q1R)

    # flux at face k+1/2; top face: wf[...,-1]=0 → F[...,-1]=0 automatically
    F     = fL + fR
    # bottom face rigid lid → flux = 0
    F_km1 = np.concatenate([np.zeros_like(F[:,:,:1]), F[:,:,:-1]], axis=2)
    return (F - F_km1) / dz


# ══ Upwind helpers ════════════════════════════════════════════════════════════

def _upwind_z_scalar(phi: np.ndarray, w_cen: np.ndarray,
                     dz_face: np.ndarray) -> np.ndarray:
    """1st-order upwind d(w·phi)/dz for a cell-centred scalar. Neumann BCs."""
    dz = dz_face[np.newaxis, np.newaxis, :]
    wp = np.maximum(w_cen, 0.0)
    wn = np.minimum(w_cen, 0.0)
    phi_dn = np.concatenate([phi[:,:,:1],  phi[:,:,:-1]], axis=2)   # Neumann bottom
    phi_up = np.concatenate([phi[:,:,1:],  phi[:,:,-1:]], axis=2)   # Neumann top
    return (wp*(phi - phi_dn) + wn*(phi_up - phi)) / dz


def _upwind_x(phi, u, dx):
    up = np.maximum(u, 0.0);  un = np.minimum(u, 0.0)
    return (up*(phi - np.roll(phi, 1, 0)) + un*(np.roll(phi, -1, 0) - phi)) / dx

def _upwind_y(phi, v, dy):
    vp = np.maximum(v, 0.0);  vn = np.minimum(v, 0.0)
    return (vp*(phi - np.roll(phi, 1, 1)) + vn*(np.roll(phi, -1, 1) - phi)) / dy

def _upwind_z_w(w: np.ndarray, dz_face: np.ndarray) -> np.ndarray:
    dw    = np.zeros_like(w)
    w_int = w[:,:,1:-1];  w_up = w[:,:,2:];  w_dn = w[:,:,:-2]
    dz    = (dz_face[:-1] + dz_face[1:])[np.newaxis, np.newaxis, :]
    wp    = np.maximum(w_int, 0.0);  wn = np.minimum(w_int, 0.0)
    dw[:,:,1:-1] = (wp*(w_int - w_dn) + wn*(w_up - w_int)) / dz
    return dw


# ══ Smagorinsky-Lilly SGS ═════════════════════════════════════════════════════

def _smag_K(s: State, g: Grid):
    """
    Spatially varying eddy diffusivities via Smagorinsky-Lilly:
        K_M = (Cs · Δ)²  |S|
        K_H = K_M / Pr_t
    Returns K_M, K_H both shape (nx, ny, nz).
    """
    dx, dy = g.dx, g.dy
    dz3 = g.dz_face[np.newaxis, np.newaxis, :]     # (1,1,nz)

    # horizontal gradients (centred, periodic in x and y)
    dudx = (np.roll(s.u, -1, 0) - np.roll(s.u,  1, 0)) / (2*dx)
    dvdy = (np.roll(s.v, -1, 1) - np.roll(s.v,  1, 1)) / (2*dy)
    dudy = (np.roll(s.u, -1, 1) - np.roll(s.u,  1, 1)) / (2*dy)
    dvdx = (np.roll(s.v, -1, 0) - np.roll(s.v,  1, 0)) / (2*dx)

    # dw/dz from staggered faces (exact on this grid)
    dwdz = (s.w[:,:,1:] - s.w[:,:,:-1]) / dz3

    # w at cell centres for horizontal shear terms
    w_cen = 0.5 * (s.w[:,:,:-1] + s.w[:,:,1:])
    dwdx  = (np.roll(w_cen, -1, 0) - np.roll(w_cen,  1, 0)) / (2*dx)
    dwdy  = (np.roll(w_cen, -1, 1) - np.roll(w_cen,  1, 1)) / (2*dy)

    # du/dz, dv/dz — centred differences, handles non-uniform spacing
    dudz = np.gradient(s.u, g.z, axis=2)
    dvdz = np.gradient(s.v, g.z, axis=2)

    # strain-rate magnitude squared: 2 S_ij S_ij
    S2 = (2*(dudx**2 + dvdy**2 + dwdz**2)
          + (dudy + dvdx)**2
          + (dudz + dwdx)**2
          + (dvdz + dwdy)**2)

    Delta = (dx * dy * g.dz_face[np.newaxis, np.newaxis, :])**(1/3)
    K_M   = (cfg.CS * Delta)**2 * np.sqrt(np.maximum(S2, 0.0))
    K_H   = K_M / cfg.PR_T
    return K_M, K_H


# ══ Variable-K diffusion ══════════════════════════════════════════════════════

def _kdiff_x(phi, K, dx):
    Kf   = 0.5*(K + np.roll(K, -1, 0))
    flux = Kf * (np.roll(phi, -1, 0) - phi) / dx
    return (flux - np.roll(flux, 1, 0)) / dx

def _kdiff_y(phi, K, dy):
    Kf   = 0.5*(K + np.roll(K, -1, 1))
    flux = Kf * (np.roll(phi, -1, 1) - phi) / dy
    return (flux - np.roll(flux, 1, 1)) / dy

def _kdiff_z(phi, K, dz_face, dz_cen):
    """Flux-form variable-K diffusion in z; Neumann BCs (zero flux at boundaries)."""
    Kf       = 0.5*(K[:,:,:-1] + K[:,:,1:])        # (nx,ny,nz-1) at interior faces
    dz_c     = dz_cen[np.newaxis, np.newaxis, :]
    flux_int = Kf * (phi[:,:,1:] - phi[:,:,:-1]) / dz_c
    zeros    = np.zeros_like(phi[:,:,:1])
    flux     = np.concatenate([zeros, flux_int, zeros], axis=2)   # (nx,ny,nz+1)
    return (flux[:,:,1:] - flux[:,:,:-1]) / dz_face[np.newaxis, np.newaxis, :]


# ══ Full tendency computation ══════════════════════════════════════════════════

def compute_tendencies(s: State, g: Grid) -> dict:
    dx, dy = g.dx, g.dy
    dz_f   = g.dz_face    # (nz,)
    dz_c   = g.dz_cen     # (nz-1,)

    # w at cell centres for z-advection
    w_cen = 0.5 * (s.w[:,:,:-1] + s.w[:,:,1:])

    # ── Advection ─────────────────────────────────────────────────────────────
    # theta, qv, u, v: upwind everywhere.
    #   Upwind's implicit diffusion (~50 m²/s) stabilises the dynamically
    #   active fields and prevents explosive convective growth.
    # qc, qr: WENO-3 in x,y + upwind in z.
    #   WENO-3 sharpens horizontal cloud edges and gives the cauliflower
    #   structure; these fields are bounded (clipping + microphysics) so
    #   WENO-3 undershoots are safe. Upwind in z retains vertical stability.
    def adv_upwind(phi):
        return (_upwind_x(phi, s.u, dx)
                + _upwind_y(phi, s.v, dy)
                + _upwind_z_scalar(phi, w_cen, dz_f))

    def adv_weno_h(phi):
        return (_weno3_x(phi, s.u, dx)
                + _weno3_y(phi, s.v, dy)
                + _upwind_z_scalar(phi, w_cen, dz_f))

    tend_theta = -adv_upwind(s.theta)
    tend_qv    = -adv_upwind(s.qv)
    tend_qc    = -adv_weno_h(s.qc)
    tend_qr    = -adv_weno_h(s.qr)

    # ── Momentum advection ─────────────────────────────────────────────────────
    # Use WENO-3 if enabled (reduces numerical diffusion), otherwise upwind
    if cfg.ENABLE_WENO_MOMENTUM:
        tend_u = -adv_weno_h(s.u)
        tend_v = -adv_weno_h(s.v)
    else:
        tend_u = -adv_upwind(s.u)
        tend_v = -adv_upwind(s.v)

    # ── Upwind advection for w (face-staggered) ───────────────────────────────
    u_at_w = np.zeros((g.nx, g.ny, g.nz + 1))
    v_at_w = np.zeros((g.nx, g.ny, g.nz + 1))
    u_at_w[:,:,1:-1] = 0.5*(s.u[:,:,:-1] + s.u[:,:,1:])
    v_at_w[:,:,1:-1] = 0.5*(s.v[:,:,:-1] + s.v[:,:,1:])
    tend_w = (-_upwind_x(s.w, u_at_w, dx)
              - _upwind_y(s.w, v_at_w, dy)
              - _upwind_z_w(s.w, dz_f))
    tend_w[:,:, 0] = 0.0
    tend_w[:,:,-1] = 0.0

    # ── Smagorinsky SGS diffusion ─────────────────────────────────────────────
    K_M, K_H = _smag_K(s, g)

    def diff_s(phi):
        return (_kdiff_x(phi, K_H, dx)
                + _kdiff_y(phi, K_H, dy)
                + _kdiff_z(phi, K_H, dz_f, dz_c))

    def diff_m(phi):
        return (_kdiff_x(phi, K_M, dx)
                + _kdiff_y(phi, K_M, dy)
                + _kdiff_z(phi, K_M, dz_f, dz_c))

    tend_u     += diff_m(s.u)
    tend_v     += diff_m(s.v)
    tend_theta += diff_s(s.theta)
    tend_qv    += diff_s(s.qv)
    tend_qc    += diff_s(s.qc)
    tend_qr    += diff_s(s.qr)

    # w interior diffusion with variable K_M at face positions
    w_int     = s.w[:,:,1:-1]
    K_M_wface = 0.5*(K_M[:,:,:-1] + K_M[:,:,1:])    # (nx,ny,nz-1)
    dz_mid    = (dz_f[:-1] * dz_f[1:])[np.newaxis, np.newaxis, :]
    dw_diff   = (_kdiff_x(w_int, K_M_wface, dx)
                 + _kdiff_y(w_int, K_M_wface, dy)
                 + K_M_wface * (s.w[:,:,2:] - 2.0*w_int + s.w[:,:,:-2]) / dz_mid)
    tend_w[:,:,1:-1] += dw_diff

    # ── Coriolis force (f-plane approximation) ────────────────────────────────
    # Coriolis deflects horizontal winds: du/dt += f*v, dv/dt -= f*u
    # This breaks axisymmetric flow patterns and prevents artificial rotation
    if cfg.ENABLE_CORIOLIS:
        tend_u += cfg.CORIOLIS_F * s.v
        tend_v -= cfg.CORIOLIS_F * s.u

    # ── Buoyancy on w-faces ───────────────────────────────────────────────────
    B      = s.buoyancy()
    B_face = np.zeros((g.nx, g.ny, g.nz + 1))
    B_face[:,:,1:-1] = 0.5*(B[:,:,:-1] + B[:,:,1:])
    tend_w += B_face

    # ── Background theta gradient carried by w ────────────────────────────────
    dtheta0_dz = np.gradient(s.theta0, g.z)
    tend_theta -= w_cen * dtheta0_dz[np.newaxis, np.newaxis, :]

    return dict(u=tend_u, v=tend_v, w=tend_w,
                theta=tend_theta, qv=tend_qv, qc=tend_qc, qr=tend_qr)

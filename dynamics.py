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
Periodicity: x and y wrap.
Vertical: zero-gradient (Neumann) BCs; w=0 on top/bottom faces (rigid lid).
"""

import numpy as np
import config as cfg
from grid import Grid
from state import State

# WENO smoothness-indicator floor
# NOTE: This will be scaled by local field magnitude to handle different variable scales
# (e.g., θ ~300 K vs q_t ~1e-8 kg/kg)
_EPS_REL = 1e-6  # Relative epsilon (fraction of local field magnitude)


# ══ WENO-3 advection (for cell-centered scalars) ═════════════════════════════

def _weno3_x(phi: np.ndarray, u_face: np.ndarray, dx: float) -> np.ndarray:
    """
    Flux-form WENO-3 d(u·phi)/dx for cell-centered scalar, periodic in x.
    phi: (nx, ny, nz) cell-centered
    u_face: (nx, ny, nz) velocity at x-faces between cells i and i+1

    CRITICAL: Uses true divergence-free face velocities, not cell-centered interpolation.
    """
    m1 = np.roll(phi,  1, 0)
    p1 = np.roll(phi, -1, 0)
    p2 = np.roll(phi, -2, 0)
    uf = u_face  # Already at faces - no averaging needed!
    up = np.maximum(uf, 0.0)
    un = np.minimum(uf, 0.0)

    # Adaptive epsilon scaled by local field magnitude
    # This ensures consistent behavior for both θ (~300) and q_t (~1e-8)
    phi_scale = np.maximum(np.abs(phi), 1e-12)  # Avoid division by zero
    eps = _EPS_REL * phi_scale**2

    # left-biased reconstruction at i+1/2 (upwind for u > 0)
    q0L = -0.5*m1 + 1.5*phi;   q1L = 0.5*phi + 0.5*p1
    b0L = (phi - m1)**2;         b1L = (p1 - phi)**2
    # Standard WENO-3 weights: 1/3 upwind, 2/3 centered
    a0L = (1/3) / (eps + b0L)**2;  a1L = (2/3) / (eps + b1L)**2
    w0L = a0L / (a0L + a1L)
    fL  = up * (w0L*q0L + (1 - w0L)*q1L)

    # right-biased reconstruction at i+1/2 (upwind for u < 0)
    q0R = 1.5*p1 - 0.5*p2;     q1R = 0.5*phi + 0.5*p1
    b0R = (p2 - p1)**2;          b1R = (p1 - phi)**2
    # Standard WENO-3 weights: 1/3 upwind, 2/3 centered
    a0R = (1/3) / (eps + b0R)**2;  a1R = (2/3) / (eps + b1R)**2
    w0R = a0R / (a0R + a1R)
    fR  = un * (w0R*q0R + (1 - w0R)*q1R)

    F = fL + fR                                # flux at face i+1/2
    return (F - np.roll(F, 1, 0)) / dx        # divergence


def _weno3_y(phi: np.ndarray, v_face: np.ndarray, dy: float) -> np.ndarray:
    """
    Flux-form WENO-3 d(v·phi)/dy for cell-centered scalar, periodic in y.
    phi: (nx, ny, nz) cell-centered
    v_face: (nx, ny, nz) velocity at y-faces between cells j and j+1

    CRITICAL: Uses true divergence-free face velocities, not cell-centered interpolation.
    """
    m1 = np.roll(phi,  1, 1)
    p1 = np.roll(phi, -1, 1)
    p2 = np.roll(phi, -2, 1)
    vf = v_face  # Already at faces - no averaging needed!
    vp = np.maximum(vf, 0.0)
    vn = np.minimum(vf, 0.0)

    # Adaptive epsilon scaled by local field magnitude
    phi_scale = np.maximum(np.abs(phi), 1e-12)
    eps = _EPS_REL * phi_scale**2

    q0L = -0.5*m1 + 1.5*phi;   q1L = 0.5*phi + 0.5*p1
    b0L = (phi - m1)**2;         b1L = (p1 - phi)**2
    # Standard WENO-3 weights: 1/3 upwind, 2/3 centered
    a0L = (1/3) / (eps + b0L)**2;  a1L = (2/3) / (eps + b1L)**2
    w0L = a0L / (a0L + a1L)
    fL  = vp * (w0L*q0L + (1 - w0L)*q1L)

    q0R = 1.5*p1 - 0.5*p2;     q1R = 0.5*phi + 0.5*p1
    b0R = (p2 - p1)**2;          b1R = (p1 - phi)**2
    # Standard WENO-3 weights: 1/3 upwind, 2/3 centered
    a0R = (1/3) / (eps + b0R)**2;  a1R = (2/3) / (eps + b1R)**2
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

    # Adaptive epsilon scaled by local field magnitude
    phi_scale = np.maximum(np.abs(phi), 1e-12)
    eps = _EPS_REL * phi_scale**2

    # left-biased (w > 0)
    q0L = -0.5*pm1 + 1.5*pk;  q1L = 0.5*pk + 0.5*pp1
    b0L = (pk - pm1)**2;        b1L = (pp1 - pk)**2
    # Standard WENO-3 weights: 1/3 upwind, 2/3 centered
    a0L = (1/3) / (eps + b0L)**2;  a1L = (2/3) / (eps + b1L)**2
    w0L = a0L / (a0L + a1L)
    fL  = wp * (w0L*q0L + (1 - w0L)*q1L)

    # right-biased (w < 0)
    q0R = 1.5*pp1 - 0.5*pp2;  q1R = 0.5*pk + 0.5*pp1
    b0R = (pp2 - pp1)**2;       b1R = (pp1 - pk)**2
    # Standard WENO-3 weights: 1/3 upwind, 2/3 centered
    a0R = (1/3) / (eps + b0R)**2;  a1R = (2/3) / (eps + b1R)**2
    w0R = a0R / (a0R + a1R)
    fR  = wn * (w0R*q0R + (1 - w0R)*q1R)

    # flux at face k+1/2; top face: wf[...,-1]=0 → F[...,-1]=0 automatically
    F     = fL + fR
    # bottom face rigid lid → flux = 0
    F_km1 = np.concatenate([np.zeros_like(F[:,:,:1]), F[:,:,:-1]], axis=2)
    return (F - F_km1) / dz


# ══ Upwind helpers (for cell-centered scalars) ═══════════════════════════════

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
    """1st-order upwind for cell-centered scalar with cell-centered u."""
    up = np.maximum(u, 0.0);  un = np.minimum(u, 0.0)
    return (up*(phi - np.roll(phi, 1, 0)) + un*(np.roll(phi, -1, 0) - phi)) / dx

def _upwind_y(phi, v, dy):
    """1st-order upwind for cell-centered scalar with cell-centered v."""
    vp = np.maximum(v, 0.0);  vn = np.minimum(v, 0.0)
    return (vp*(phi - np.roll(phi, 1, 1)) + vn*(np.roll(phi, -1, 1) - phi)) / dy


# ══ Momentum advection (C-grid face-staggered) ═══════════════════════════════

def advect_momentum(u, v, w, g):
    """
    2nd-order flux-form C-grid momentum advection (energy-conserving
    for discretely divergence-free flow).

    Returns tendencies (tu, tv, tw) with shapes:
      - tu: (nx+1, ny, nz) for u on x-faces
      - tv: (nx, ny+1, nz) for v on y-faces
      - tw: (nx, ny, nz+1) for w on z-faces

    This scheme conserves kinetic energy in the inviscid limit, preventing
    the energy blow-up that occurs with advective-form schemes.
    """
    dx, dy, dzf, dzc = g.dx, g.dy, g.dz_face, g.dz_cen
    U, V, W = u[:-1], v[:, :-1], w                      # unique periodic faces
    xm = lambda a: np.roll(a, 1, 0); xp = lambda a: np.roll(a, -1, 0)
    ym = lambda a: np.roll(a, 1, 1); yp = lambda a: np.roll(a, -1, 1)
    ddz = lambda F: (F[..., 1:] - F[..., :-1]) / dzf

    # u-momentum
    uc = 0.5*(U + xp(U)); Fxx = uc*uc
    adv_u = (Fxx - xm(Fxx))/dx
    Fyx = 0.5*(xm(V)+V) * 0.5*(ym(U)+U)
    adv_u += (yp(Fyx) - Fyx)/dy
    uz = np.zeros_like(W); uz[..., 1:-1] = 0.5*(U[..., :-1] + U[..., 1:])
    adv_u += ddz(0.5*(xm(W)+W) * uz)

    # v-momentum
    vc = 0.5*(V + yp(V)); Fyy = vc*vc
    adv_v = (Fyy - ym(Fyy))/dy
    Fxy = 0.5*(ym(U)+U) * 0.5*(xm(V)+V)
    adv_v += (xp(Fxy) - Fxy)/dx
    vz = np.zeros_like(W); vz[..., 1:-1] = 0.5*(V[..., :-1] + V[..., 1:])
    adv_v += ddz(0.5*(ym(W)+W) * vz)

    # w-momentum (interior faces)
    a = dzf[:-1]/(dzf[:-1]+dzf[1:]); b = dzf[1:]/(dzf[:-1]+dzf[1:])
    Uw = a*U[..., :-1] + b*U[..., 1:]; Vw = a*V[..., :-1] + b*V[..., 1:]
    Wi = W[..., 1:-1]
    Fxw = Uw*0.5*(xm(Wi)+Wi); Fyw = Vw*0.5*(ym(Wi)+Wi)
    wc = 0.5*(W[..., :-1] + W[..., 1:]); Fzz = wc*wc
    adv_w = np.zeros_like(W)
    adv_w[..., 1:-1] = ((xp(Fxw)-Fxw)/dx + (yp(Fyw)-Fyw)/dy
                        + (Fzz[..., 1:]-Fzz[..., :-1])/dzc)

    tu = np.concatenate([-adv_u, -adv_u[:1]], 0)        # restore duplicate face
    tv = np.concatenate([-adv_v, -adv_v[:, :1]], 1)
    return tu, tv, -adv_w


# ══ Smagorinsky-Lilly SGS (C-grid) ═══════════════════════════════════════════

def _smag_K(s: State, g: Grid):
    """
    Spatially varying eddy diffusivities via Smagorinsky-Lilly on C-grid:
        K_M = (Cs · Δ)²  |S|
        K_H = K_M / Pr_t
    Returns K_M, K_H both shape (nx, ny, nz) at cell centers.
    """
    dx, dy = g.dx, g.dy
    dz3 = g.dz_face[np.newaxis, np.newaxis, :]     # (1,1,nz)

    # Interpolate face velocities to cell centers
    u_cen = s.u_at_cell()  # (nx, ny, nz)
    v_cen = s.v_at_cell()  # (nx, ny, nz)
    w_cen = s.w_at_cell()  # (nx, ny, nz)

    # Horizontal gradients (centred, periodic in x and y)
    dudx = (np.roll(u_cen, -1, 0) - np.roll(u_cen,  1, 0)) / (2*dx)
    dvdy = (np.roll(v_cen, -1, 1) - np.roll(v_cen,  1, 1)) / (2*dy)
    dudy = (np.roll(u_cen, -1, 1) - np.roll(u_cen,  1, 1)) / (2*dy)
    dvdx = (np.roll(v_cen, -1, 0) - np.roll(v_cen,  1, 0)) / (2*dx)

    # dw/dz from staggered faces (exact on this grid)
    # This is actually exact: dwdz at cell center k is (w[k+1] - w[k]) / dz[k]
    dwdz = (s.w[:,:,1:] - s.w[:,:,:-1]) / dz3

    # Horizontal derivatives of w
    dwdx  = (np.roll(w_cen, -1, 0) - np.roll(w_cen,  1, 0)) / (2*dx)
    dwdy  = (np.roll(w_cen, -1, 1) - np.roll(w_cen,  1, 1)) / (2*dy)

    # du/dz, dv/dz — centred differences, handles non-uniform spacing
    dudz = np.gradient(u_cen, g.z, axis=2)
    dvdz = np.gradient(v_cen, g.z, axis=2)

    # strain-rate magnitude squared: 2 S_ij S_ij
    S2 = (2*(dudx**2 + dvdy**2 + dwdz**2)
          + (dudy + dvdx)**2
          + (dudz + dwdx)**2
          + (dvdz + dwdy)**2)

    Delta = (dx * dy * g.dz_face[np.newaxis, np.newaxis, :])**(1/3)
    K_M   = (cfg.CS * Delta)**2 * np.sqrt(np.maximum(S2, 0.0))
    K_H   = K_M / cfg.PR_T
    return K_M, K_H


# ══ Variable-K diffusion (for cell-centered fields) ══════════════════════════

def _kdiff_x(phi, K, dx):
    """Diffusion for cell-centered scalar with cell-centered K."""
    Kf   = 0.5*(K + np.roll(K, -1, 0))
    flux = Kf * (np.roll(phi, -1, 0) - phi) / dx
    return (flux - np.roll(flux, 1, 0)) / dx

def _kdiff_y(phi, K, dy):
    """Diffusion for cell-centered scalar with cell-centered K."""
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

    # ── Enforce periodic BCs for face-staggered velocities ───────────────────
    # CRITICAL for eliminating seam artifacts at periodic boundaries!
    # u is on x-faces (nx+1): u[0] and u[nx] are the SAME face (periodic wrap)
    # v is on y-faces (ny+1): v[0] and v[ny] are the SAME face (periodic wrap)
    s.u[-1, :, :] = s.u[0, :, :]
    s.v[:, -1, :] = s.v[:, 0, :]

    # ── Scalar advection ──────────────────────────────────────────────────────
    # CONSISTENT WENO-3 for all scalars (Phase 1.3)
    # Phase 1.4: Transport conserved variables θ_l, qt (not θ, qv, qc)
    #
    # CRITICAL FIX: Use true divergence-free face velocities!
    # Passing cell-centered velocities and re-averaging breaks discrete conservation.
    # This was causing spurious sources/sinks of ~0.23/s in uniform-scalar test.
    #
    # For WENO flux computation: u_face[i] should be velocity at RIGHT face of cell i
    # Right face of cell i is at x-face i+1, so u_face[i] = u[i+1]
    # With periodic BCs enforced (u[nx] = u[0]), using u[1:] gives correct periodicity
    u_face = s.u[1:]      # (nx, ny, nz) - x-face velocities (u[1] through u[nx])
    v_face = s.v[:,1:]    # (nx, ny, nz) - y-face velocities (v[1] through v[ny])

    # Cell-centered w only for background gradient transport
    w_cen = s.w_at_cell()  # (nx, ny, nz)

    def adv_weno3(phi):
        """WENO-3 in all directions using true divergence-free face velocities."""
        return (_weno3_x(phi, u_face, dx)
                + _weno3_y(phi, v_face, dy)
                + _weno3_z(phi, s.w, dz_f))

    # Transport conserved variables (Phase 1.4)
    tend_theta_l = -adv_weno3(s.theta_l)
    tend_qt      = -adv_weno3(s.qt)
    tend_qr      = -adv_weno3(s.qr)

    # ── Momentum advection ────────────────────────────────────────────────────
    # Flux-form energy-conserving scheme (prevents KE blow-up)
    tend_u, tend_v, tend_w = advect_momentum(s.u, s.v, s.w, g)

    # ── Smagorinsky SGS diffusion ─────────────────────────────────────────────
    K_M, K_H = _smag_K(s, g)

    # Scalar diffusion (cell-centered fields)
    # Phase 1.4: Diffuse conserved variables θ_l, qt
    def diff_s(phi):
        return (_kdiff_x(phi, K_H, dx)
                + _kdiff_y(phi, K_H, dy)
                + _kdiff_z(phi, K_H, dz_f, dz_c))

    tend_theta_l += diff_s(s.theta_l)
    tend_qt      += diff_s(s.qt)
    tend_qr      += diff_s(s.qr)

    # Momentum diffusion - apply to cell-centered, then interpolate back
    # (Simplified - proper C-grid diffusion would be on faces)
    u_cen = s.u_at_cell()  # (nx, ny, nz)
    v_cen = s.v_at_cell()  # (nx, ny, nz)
    u_diff = diff_s(u_cen)
    v_diff = diff_s(v_cen)

    # Distribute back to faces (simplified)
    # For u on x-faces (nx+1, ny, nz), interpolate from cell centers
    tend_u_diff = np.zeros_like(s.u)
    tend_u_diff[1:-1, :, :] = 0.5 * (u_diff[:-1, :, :] + u_diff[1:, :, :])
    tend_u_diff[0, :, :] = u_diff[0, :, :]   # boundary (periodic)
    tend_u_diff[-1, :, :] = tend_u_diff[0, :, :]
    tend_u += tend_u_diff

    tend_v_diff = np.zeros_like(s.v)
    tend_v_diff[:, 1:-1, :] = 0.5 * (v_diff[:, :-1, :] + v_diff[:, 1:, :])
    tend_v_diff[:, 0, :] = v_diff[:, 0, :]
    tend_v_diff[:, -1, :] = tend_v_diff[:, 0, :]
    tend_v += tend_v_diff

    # w diffusion (interior faces only)
    w_int = s.w[:,:,1:-1]
    K_M_wface = 0.5*(K_M[:,:,:-1] + K_M[:,:,1:])
    dz_mid = (dz_f[:-1] * dz_f[1:])[np.newaxis, np.newaxis, :]
    dw_diff = (_kdiff_x(w_int, K_M_wface, dx)
               + _kdiff_y(w_int, K_M_wface, dy)
               + K_M_wface * (s.w[:,:,2:] - 2.0*w_int + s.w[:,:,:-2]) / dz_mid)
    tend_w[:,:,1:-1] += dw_diff

    # ── Coriolis force (f-plane) ──────────────────────────────────────────────
    # CRITICAL FIX: Include geostrophic wind terms!
    # Correct form: du/dt = +f*(v - v_g),  dv/dt = -f*(u - u_g)
    # Without this, wind rotates ~47° over 6h in BOMEX instead of staying steady
    if cfg.ENABLE_CORIOLIS:
        # v at u-faces: (nx+1, ny, nz)
        v_at_u = s.v_at_u_faces()
        tend_u += cfg.CORIOLIS_F * (v_at_u - cfg.V_GEO)

        # u at v-faces: (nx, ny+1, nz)
        u_at_v = s.u_at_v_faces()
        tend_v -= cfg.CORIOLIS_F * (u_at_v - cfg.U_GEO)

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

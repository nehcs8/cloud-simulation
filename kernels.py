"""
Numba kernels for the expensive parts of dynamics.compute_tendencies.

Each kernel reproduces one numpy routine in dynamics.py loop by loop (same
stencils, same boundary handling) and runs in parallel over x with prange.
The numpy versions stay in dynamics.py as the reference; tests/test_kernels.py
checks that both agree to round-off.

  weno3_div        ↔ _weno3_x + _weno3_y + _weno3_z
  kdiff_xy         ↔ _kdiff_x + _kdiff_y
  kdiff_xyz        ↔ _kdiff_x + _kdiff_y + _kdiff_z
  smag_KM          ↔ K_M of _smag_K
  momentum_adv     ↔ advect_momentum (returns -adv, i.e. tendencies)
  gather_xy        ↔ halo padding (halo.py)

Horizontal neighbours come from halo-padded copies (halo.py, H = 2 ghost cells), so the
same kernels serve periodic and open lateral boundaries.

Threads: one per physical core by default (hyperthreads slow these kernels
down); override with the NUMBA_NUM_THREADS environment variable.
"""

import os
import numpy as np
import numba
from numba import njit, prange

if 'NUMBA_NUM_THREADS' not in os.environ:
    numba.set_num_threads(max(1, min(numba.config.NUMBA_NUM_THREADS, (os.cpu_count() or 2) // 2)))

_EPS_REL = 1e-6   # must match dynamics._EPS_REL


# ══ Halo padding ═════════════════════════════════════════════════════════════

@njit(parallel=True, cache=True)
def gather_xy(a, mx, my):
    """out[I, J, :] = a[mx[I], my[J], :] — halo padding with index maps (see halo.py)."""
    nxp = mx.shape[0]; nyp = my.shape[0]; nz = a.shape[2]
    out = np.empty((nxp, nyp, nz))
    for I in prange(nxp):
        si = mx[I]
        for J in range(nyp):
            sj = my[J]
            for k in range(nz):
                out[I, J, k] = a[si, sj, k]
    return out


# ══ WENO-3 scalar advection ══════════════════════════════════════════════════

@njit(inline='always')
def _weno3_face(m1, c, p1, p2, vel, eps):
    """Upwind-weighted WENO-3 flux vel·φ at the face between cells c and p1."""
    q0L = -0.5*m1 + 1.5*c;   q1L = 0.5*c + 0.5*p1
    b0L = (c - m1)**2;        b1L = (p1 - c)**2
    a0L = (1/3) / (eps + b0L)**2;  a1L = (2/3) / (eps + b1L)**2
    w0L = a0L / (a0L + a1L)
    fL = max(vel, 0.0) * (w0L*q0L + (1 - w0L)*q1L)

    q0R = 1.5*p1 - 0.5*p2;   q1R = 0.5*c + 0.5*p1
    b0R = (p2 - p1)**2;       b1R = (p1 - c)**2
    a0R = (1/3) / (eps + b0R)**2;  a1R = (2/3) / (eps + b1R)**2
    w0R = a0R / (a0R + a1R)
    fR = min(vel, 0.0) * (w0R*q0R + (1 - w0R)*q1R)
    return fL + fR


@njit(inline='always')
def _eps(phi):
    s = max(abs(phi), 1e-12)
    return _EPS_REL * s * s


@njit(parallel=True, cache=True)
def weno3_div(P, u, v, w, dx, dy, dz_face, rho0_c, rho0_w):
    """
    ∂(uφ)/∂x + ∂(vφ)/∂y + (1/ρ₀) ∂(ρ₀wφ)/∂z, WENO-3 flux form.

    P: φ padded with H = 2 ghost cells (halo.pad_c), shape (nx+4, ny+4, nz);
    u (nx+1,ny,nz), v (nx,ny+1,nz), w (nx,ny,nz+1) C-grid velocities (all faces,
    including the two boundary faces). Neumann ghost cells and zero flux through
    the bottom face in z.
    """
    nx = P.shape[0] - 4; ny = P.shape[1] - 4; nz = P.shape[2]
    out = np.empty((nx, ny, nz))
    for i in prange(nx):
        I = i + 2
        for j in range(ny):
            J = j + 2
            F_below = 0.0                       # flux through bottom face (rigid lid)
            for k in range(nz):
                c = P[I, J, k]
                # x: faces i+1/2 and i-1/2
                Fx = _weno3_face(P[I - 1, J, k], c, P[I + 1, J, k], P[I + 2, J, k],
                                 u[i + 1, j, k], _eps(c))
                cm = P[I - 1, J, k]
                Fxm = _weno3_face(P[I - 2, J, k], cm, c, P[I + 1, J, k],
                                  u[i, j, k], _eps(cm))
                # y
                Fy = _weno3_face(P[I, J - 1, k], c, P[I, J + 1, k], P[I, J + 2, k],
                                 v[i, j + 1, k], _eps(c))
                cm = P[I, J - 1, k]
                Fym = _weno3_face(P[I, J - 2, k], cm, c, P[I, J + 1, k],
                                  v[i, j, k], _eps(cm))
                # z: face k+1/2 with Neumann ghosts
                km1 = max(k - 1, 0); kp1 = min(k + 1, nz - 1); kp2 = min(k + 2, nz - 1)
                Fz = _weno3_face(P[I, J, km1], c, P[I, J, kp1], P[I, J, kp2],
                                 rho0_w[k + 1] * w[i, j, k + 1], _eps(c))
                out[i, j, k] = ((Fx - Fxm) / dx + (Fy - Fym) / dy
                                + (Fz - F_below) / (rho0_c[k] * dz_face[k]))
                F_below = Fz
    return out


# ══ Variable-K diffusion ═════════════════════════════════════════════════════

@njit(parallel=True, cache=True)
def kdiff_xy(P, KP, dx, dy):
    """∂/∂x(K ∂φ/∂x) + ∂/∂y(K ∂φ/∂y), φ and K at cell centres, both halo-padded (H = 2)."""
    nx = P.shape[0] - 4; ny = P.shape[1] - 4; nz = P.shape[2]
    out = np.empty((nx, ny, nz))
    for i in prange(nx):
        I = i + 2
        for j in range(ny):
            J = j + 2
            for k in range(nz):
                c = P[I, J, k]; Kc = KP[I, J, k]
                fxp = 0.5*(Kc + KP[I + 1, J, k]) * (P[I + 1, J, k] - c) / dx
                fxm = 0.5*(KP[I - 1, J, k] + Kc) * (c - P[I - 1, J, k]) / dx
                fyp = 0.5*(Kc + KP[I, J + 1, k]) * (P[I, J + 1, k] - c) / dy
                fym = 0.5*(KP[I, J - 1, k] + Kc) * (c - P[I, J - 1, k]) / dy
                out[i, j, k] = (fxp - fxm) / dx + (fyp - fym) / dy
    return out


@njit(parallel=True, cache=True)
def kdiff_xyz(P, KP, dx, dy, dz_face, dz_cen, rho0_c, rho0_w):
    """kdiff_xy + (1/ρ₀) ∂/∂z(ρ₀ K ∂φ/∂z), zero flux at bottom and top (padded inputs)."""
    nx = P.shape[0] - 4; ny = P.shape[1] - 4; nz = P.shape[2]
    out = kdiff_xy(P, KP, dx, dy)
    for i in prange(nx):
        I = i + 2
        for j in range(ny):
            J = j + 2
            f_below = 0.0
            for k in range(nz):
                if k < nz - 1:
                    f_above = (rho0_w[k + 1] * (0.5*(KP[I, J, k] + KP[I, J, k + 1]))
                               * (P[I, J, k + 1] - P[I, J, k]) / dz_cen[k])
                else:
                    f_above = 0.0
                out[i, j, k] += (f_above - f_below) / (rho0_c[k] * dz_face[k])
                f_below = f_above
    return out


@njit(parallel=True, cache=True)
def kdiff_xy_masked(P, KP, mx, my, dx, dy):
    """kdiff_xy with face masks (terrain): mx (nx+1,ny,nz), my (nx,ny+1,nz), 0 = closed face."""
    nx = P.shape[0] - 4; ny = P.shape[1] - 4; nz = P.shape[2]
    out = np.empty((nx, ny, nz))
    for i in prange(nx):
        I = i + 2
        for j in range(ny):
            J = j + 2
            for k in range(nz):
                c = P[I, J, k]; Kc = KP[I, J, k]
                fxp = mx[i + 1, j, k] * 0.5*(Kc + KP[I + 1, J, k]) * (P[I + 1, J, k] - c) / dx
                fxm = mx[i, j, k] * 0.5*(KP[I - 1, J, k] + Kc) * (c - P[I - 1, J, k]) / dx
                fyp = my[i, j + 1, k] * 0.5*(Kc + KP[I, J + 1, k]) * (P[I, J + 1, k] - c) / dy
                fym = my[i, j, k] * 0.5*(KP[I, J - 1, k] + Kc) * (c - P[I, J - 1, k]) / dy
                out[i, j, k] = (fxp - fxm) / dx + (fyp - fym) / dy
    return out


@njit(parallel=True, cache=True)
def kdiff_xyz_masked(P, KP, mx, my, mz, dx, dy, dz_face, dz_cen, rho0_c, rho0_w):
    """kdiff_xyz with face masks (terrain); mz (nx,ny,nz+1)."""
    nx = P.shape[0] - 4; ny = P.shape[1] - 4; nz = P.shape[2]
    out = kdiff_xy_masked(P, KP, mx, my, dx, dy)
    for i in prange(nx):
        I = i + 2
        for j in range(ny):
            J = j + 2
            f_below = 0.0
            for k in range(nz):
                if k < nz - 1:
                    f_above = (mz[i, j, k + 1] * rho0_w[k + 1] * (0.5*(KP[I, J, k] + KP[I, J, k + 1]))
                               * (P[I, J, k + 1] - P[I, J, k]) / dz_cen[k])
                else:
                    f_above = 0.0
                out[i, j, k] += (f_above - f_below) / (rho0_c[k] * dz_face[k])
                f_below = f_above
    return out


# ══ Smagorinsky K_M ══════════════════════════════════════════════════════════

@njit(parallel=True, cache=True)
def smag_KM(UC, VC, WC, w, z, dx, dy, dz_face, cs):
    """
    K_M = (Cs·Δ)² |S| at cell centres; same stencils as dynamics._smag_K
    (centred horizontal differences, np.gradient-style du/dz, dv/dz).
    UC, VC, WC: cell-centred u, v, w, halo-padded (H = 2); w: unpadded z-faces.
    """
    nx = UC.shape[0] - 4; ny = UC.shape[1] - 4; nz = UC.shape[2]
    KM = np.empty((nx, ny, nz))
    for i in prange(nx):
        I = i + 2
        for j in range(ny):
            J = j + 2
            for k in range(nz):
                dudx = (UC[I + 1, J, k] - UC[I - 1, J, k]) / (2*dx)
                dvdy = (VC[I, J + 1, k] - VC[I, J - 1, k]) / (2*dy)
                dudy = (UC[I, J + 1, k] - UC[I, J - 1, k]) / (2*dy)
                dvdx = (VC[I + 1, J, k] - VC[I - 1, J, k]) / (2*dx)
                dwdz = (w[i, j, k + 1] - w[i, j, k]) / dz_face[k]
                dwdx = (WC[I + 1, J, k] - WC[I - 1, J, k]) / (2*dx)
                dwdy = (WC[I, J + 1, k] - WC[I, J - 1, k]) / (2*dy)
                # np.gradient(·, z): one-sided at the ends, 2nd-order non-uniform inside
                if k == 0:
                    h = z[1] - z[0]
                    dudz = (UC[I, J, 1] - UC[I, J, 0]) / h
                    dvdz = (VC[I, J, 1] - VC[I, J, 0]) / h
                elif k == nz - 1:
                    h = z[nz - 1] - z[nz - 2]
                    dudz = (UC[I, J, k] - UC[I, J, k - 1]) / h
                    dvdz = (VC[I, J, k] - VC[I, J, k - 1]) / h
                else:
                    h1 = z[k] - z[k - 1]; h2 = z[k + 1] - z[k]
                    a = -h2 / (h1 * (h1 + h2))
                    b = (h2 - h1) / (h1 * h2)
                    c = h1 / (h2 * (h1 + h2))
                    dudz = a*UC[I, J, k - 1] + b*UC[I, J, k] + c*UC[I, J, k + 1]
                    dvdz = a*VC[I, J, k - 1] + b*VC[I, J, k] + c*VC[I, J, k + 1]
                S2 = (2*(dudx**2 + dvdy**2 + dwdz**2)
                      + (dudy + dvdx)**2
                      + (dudz + dwdx)**2
                      + (dvdz + dwdy)**2)
                Delta = (dx * dy * dz_face[k])**(1/3)
                KM[i, j, k] = (cs * Delta)**2 * np.sqrt(max(S2, 0.0))
    return KM


# ══ Momentum advection ═══════════════════════════════════════════════════════

@njit(parallel=True, cache=True)
def momentum_adv(UP, VP, WP, dx, dy, dz_face, dz_cen, rho0_c, rho0_w):
    """
    Tendencies (tu, tv, tw) = −advection, same scheme as dynamics.advect_momentum.
    Halo-padded inputs (H = 2): UP = halo.pad_u(u) (nx+1+4, ny+4, nz),
    VP = halo.pad_v(v) (nx+4, ny+1+4, nz), WP = halo.pad_c(w) (nx+4, ny+4, nz+1).
    tu is computed on all x-faces 0…nx (incl. both boundary faces), tv on all y-faces.
    """
    nx = WP.shape[0] - 4; ny = WP.shape[1] - 4; nz = WP.shape[2] - 1
    tu = np.empty((nx + 1, ny, nz)); tv = np.empty((nx, ny + 1, nz)); tw = np.zeros((nx, ny, nz + 1))

    # ── u at x-face i (i = 0…nx); in UP face i is index i+2, in VP/WP cell i is index i+2 ──
    for i in prange(nx + 1):
        I = i + 2
        for j in range(ny):
            J = j + 2
            Fzu_below = 0.0
            for k in range(nz):
                ucp = 0.5*(UP[I, J, k] + UP[I + 1, J, k]);  ucm = 0.5*(UP[I - 1, J, k] + UP[I, J, k])
                adv = (ucp*ucp - ucm*ucm) / dx
                Fyx_p = 0.5*(VP[I - 1, J + 1, k] + VP[I, J + 1, k]) * 0.5*(UP[I, J, k] + UP[I, J + 1, k])
                Fyx_c = 0.5*(VP[I - 1, J, k] + VP[I, J, k]) * 0.5*(UP[I, J - 1, k] + UP[I, J, k])
                adv += (Fyx_p - Fyx_c) / dy
                if k < nz - 1:
                    Fzu = (rho0_w[k + 1] * (0.5*(WP[I - 1, J, k + 1] + WP[I, J, k + 1]))
                           * (0.5*(UP[I, J, k] + UP[I, J, k + 1])))
                else:
                    Fzu = 0.0
                adv += (Fzu - Fzu_below) / (rho0_c[k] * dz_face[k])
                Fzu_below = Fzu
                tu[i, j, k] = -adv

    # ── v at y-face j (j = 0…ny) ──
    for i in prange(nx):
        I = i + 2
        for j in range(ny + 1):
            J = j + 2
            Fzv_below = 0.0
            for k in range(nz):
                vcp = 0.5*(VP[I, J, k] + VP[I, J + 1, k]);  vcm = 0.5*(VP[I, J - 1, k] + VP[I, J, k])
                adv = (vcp*vcp - vcm*vcm) / dy
                Fxy_p = 0.5*(UP[I + 1, J - 1, k] + UP[I + 1, J, k]) * 0.5*(VP[I, J, k] + VP[I + 1, J, k])
                Fxy_c = 0.5*(UP[I, J - 1, k] + UP[I, J, k]) * 0.5*(VP[I - 1, J, k] + VP[I, J, k])
                adv += (Fxy_p - Fxy_c) / dx
                if k < nz - 1:
                    Fzv = (rho0_w[k + 1] * (0.5*(WP[I, J - 1, k + 1] + WP[I, J, k + 1]))
                           * (0.5*(VP[I, J, k] + VP[I, J, k + 1])))
                else:
                    Fzv = 0.0
                adv += (Fzv - Fzv_below) / (rho0_c[k] * dz_face[k])
                Fzv_below = Fzv
                tv[i, j, k] = -adv

    # ── w at interior z-faces k = 1..nz-1 of cell (i, j) ──
    for i in prange(nx):
        I = i + 2
        for j in range(ny):
            J = j + 2
            for k in range(1, nz):
                m = k - 1
                a = dz_face[m] / (dz_face[m] + dz_face[m + 1])
                b = dz_face[m + 1] / (dz_face[m] + dz_face[m + 1])
                Uw_c = a*UP[I, J, m] + b*UP[I, J, m + 1]
                Uw_p = a*UP[I + 1, J, m] + b*UP[I + 1, J, m + 1]
                Vw_c = a*VP[I, J, m] + b*VP[I, J, m + 1]
                Vw_p = a*VP[I, J + 1, m] + b*VP[I, J + 1, m + 1]
                Fxw_c = Uw_c * 0.5*(WP[I - 1, J, k] + WP[I, J, k])
                Fxw_p = Uw_p * 0.5*(WP[I, J, k] + WP[I + 1, J, k])
                Fyw_c = Vw_c * 0.5*(WP[I, J - 1, k] + WP[I, J, k])
                Fyw_p = Vw_p * 0.5*(WP[I, J, k] + WP[I, J + 1, k])
                wc_lo = 0.5*(WP[I, J, k - 1] + WP[I, J, k])
                wc_hi = 0.5*(WP[I, J, k] + WP[I, J, k + 1])
                Fzz_lo = rho0_c[m] * wc_lo * wc_lo
                Fzz_hi = rho0_c[m + 1] * wc_hi * wc_hi
                tw[i, j, k] = -((Fxw_p - Fxw_c) / dx + (Fyw_p - Fyw_c) / dy
                                + (Fzz_hi - Fzz_lo) / (rho0_w[k] * dz_cen[m]))
    return tu, tv, tw


# ══ Pressure with terrain: masked anelastic Poisson operator ══════════════════

@njit(parallel=True, cache=True)
def poisson_masked(phi, mx, my, mz, Tx, Ty, Tz, fluid):
    """
    (L φ)_c = Σ_faces m_f T_f (φ_nb − φ_c) on fluid cells, 0 on solid cells.
    mx (nx+1,ny,nz), my (nx,ny+1,nz), mz (nx,ny,nz+1): 1 where the face connects two fluid
    cells (0 on solid faces, open boundary faces, top and bottom); x/y neighbours wrap
    (periodic faces 0 and n carry the same mask). Tx, Ty (nz,), Tz (nz+1,) conductances.
    """
    nx, ny, nz = phi.shape
    out = np.zeros_like(phi)
    for i in prange(nx):
        im = (i - 1) % nx; ip = (i + 1) % nx
        for j in range(ny):
            jm = (j - 1) % ny; jp = (j + 1) % ny
            for k in range(nz):
                if not fluid[i, j, k]:
                    continue
                c = phi[i, j, k]
                acc = (mx[i + 1, j, k] * Tx[k] * (phi[ip, j, k] - c)
                       + mx[i, j, k] * Tx[k] * (phi[im, j, k] - c)
                       + my[i, j + 1, k] * Ty[k] * (phi[i, jp, k] - c)
                       + my[i, j, k] * Ty[k] * (phi[i, jm, k] - c))
                if k < nz - 1:
                    acc += mz[i, j, k + 1] * Tz[k + 1] * (phi[i, j, k + 1] - c)
                if k > 0:
                    acc += mz[i, j, k] * Tz[k] * (phi[i, j, k - 1] - c)
                out[i, j, k] = acc
    return out


# ══ Pressure: tridiagonal solve per horizontal Fourier mode ══════════════════

@njit(parallel=True, cache=True)
def solve_tridiag_modes(a, b, c, lam_xy, d):
    """
    Thomas algorithm for every (kx, ky) mode: a φ_{k-1} + (b + λ_xy) φ_k + c φ_{k+1} = d.
    a, b, c (nz,) real; lam_xy (nx, nyh) real; d (nx, nyh, nz) complex.
    Same sweep as pressure._solve_tridiagonal.
    """
    nx, nyh, nz = d.shape
    x = np.empty_like(d)
    for i in prange(nx):
        bb = np.empty(nz)
        dd = np.empty(nz, dtype=d.dtype)
        for j in range(nyh):
            lam = lam_xy[i, j]
            bb[0] = b[0] + lam
            dd[0] = d[i, j, 0]
            for k in range(1, nz):
                m = a[k] / bb[k - 1]
                bb[k] = (b[k] + lam) - m * c[k - 1]
                dd[k] = d[i, j, k] - m * dd[k - 1]
            x[i, j, nz - 1] = dd[nz - 1] / bb[nz - 1]
            for k in range(nz - 2, -1, -1):
                x[i, j, k] = (dd[k] - c[k] * x[i, j, k + 1]) / bb[k]
    return x


# ══ Saturation adjustment (θ_l, qt → qv, qc) ═════════════════════════════════

@njit(parallel=True, cache=True)
def diagnose_cloud_water(theta_l, qt, theta0, pi0, p, L_cp, lv, rv, eps_rd, max_iterations):
    """
    Per-cell version of thermodynamics.diagnose_cloud_water (Buck q_sat, Newton
    iterations on qc until |qt − qc − q_s| ≤ 1e-8). pi0, p, theta0 are (nz,)
    profiles; p = P00·π₀^(cp/Rd). Returns (qv, qc, any_supersaturated).
    """
    nx, ny, nz = theta_l.shape
    qv = np.empty_like(qt)
    qc = np.zeros_like(qt)
    flags = np.zeros(nx, dtype=np.bool_)
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                q = qt[i, j, k]
                T = (theta_l[i, j, k] + theta0[k]) * pi0[k]
                T_C = T - 273.15
                e_s = 611.2 * np.exp(17.67 * T_C / (T_C + 243.5))
                qs = eps_rd * e_s / max(p[k] - e_s, 1.0)
                c = 0.0
                if q > qs:
                    flags[i] = True
                    for it in range(max_iterations):
                        T = (theta_l[i, j, k] + L_cp * c / pi0[k] + theta0[k]) * pi0[k]
                        T_C = T - 273.15
                        e_s = 611.2 * np.exp(17.67 * T_C / (T_C + 243.5))
                        qs = eps_rd * e_s / max(p[k] - e_s, 1.0)
                        residual = q - c - qs
                        if abs(residual) <= 1e-8:
                            break
                        dqsdT = qs * lv / (rv * T**2)
                        c = c + residual / (1.0 + dqsdT * L_cp)
                        c = min(max(c, 0.0), q)
                qc[i, j, k] = c
                qv[i, j, k] = max(q - c, 0.0)
    return qv, qc, flags.any()


# ══ Kessler microphysics (microphysics.py) ═══════════════════════════════════

@njit(inline='always')
def _qsat_buck(T, p_k, eps_rd):
    T_C = T - 273.15
    e_s = 611.2 * np.exp(17.67 * T_C / (T_C + 243.5))
    return eps_rd * e_s / max(p_k - e_s, 1.0)


@njit(parallel=True, cache=True)
def saturation_adjustment(theta, qv, qc, theta0, pi0, p, L_cp, lv, rv, eps_rd):
    """In-place one-step condensation/evaporation, as microphysics.saturation_adjustment."""
    nx, ny, nz = theta.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                T = (theta[i, j, k] + theta0[k]) * pi0[k]
                q_s = _qsat_buck(T, p[k], eps_rd)
                excess = qv[i, j, k] - q_s
                if excess > 0.0:
                    delta = excess / (1.0 + L_cp * (q_s * lv / (rv * T**2)))
                    qc[i, j, k] += delta
                    qv[i, j, k] -= delta
                    theta[i, j, k] += L_cp * delta / pi0[k]
                elif excess < 0.0 and qc[i, j, k] > 0.0:
                    delta = min(-excess / (1.0 + L_cp * (q_s * lv / (rv * T**2))), qc[i, j, k])
                    qc[i, j, k] -= delta
                    qv[i, j, k] += delta
                    theta[i, j, k] -= L_cp * delta / pi0[k]


@njit(parallel=True, cache=True)
def warm_rain(theta, qv, qc, qr, theta0, pi0, p, eps_rd, dt,
              auto_rate, auto_thresh, k2, k_evp, L_cp):
    """In-place autoconversion, accretion and rain evaporation, as microphysics.warm_rain."""
    nx, ny, nz = theta.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                T = (theta[i, j, k] + theta0[k]) * pi0[k]
                q_s = _qsat_buck(T, p[k], eps_rd)
                c = qc[i, j, k]; r = qr[i, j, k]
                auto = min(max(0.0, auto_rate * (c - auto_thresh)) * dt, c)
                accr = min(k2 * c * r ** 0.875 * dt, c)
                c -= auto + accr
                r += auto + accr
                deficit = max(0.0, q_s - qv[i, j, k])
                evap = min(k_evp * deficit * r ** 0.525 * dt, r)
                qc[i, j, k] = c
                qr[i, j, k] = r - evap
                qv[i, j, k] += evap
                theta[i, j, k] -= L_cp * evap / pi0[k]


@njit(parallel=True, cache=True)
def sedimentation(qr, rho0, dz_face, dt, a_vt, b_vt):
    """
    Implicit top-down rain sedimentation, as microphysics.sedimentation.
    Updates qr in place (after clipping at 0); returns the surface flux F_out[:, :, 0].
    """
    nx, ny, nz = qr.shape
    sfc = np.empty((nx, ny))
    for i in prange(nx):
        for j in range(ny):
            F_in = 0.0
            for k in range(nz - 1, -1, -1):
                q = max(qr[i, j, k], 0.0)
                V_t = a_vt * ((rho0[k] * q / 1000.0) ** b_vt) * np.sqrt(rho0[0] / rho0[k])
                q = (q + (dt / dz_face[k]) * (F_in / rho0[k])) / (1.0 + V_t * dt / dz_face[k])
                q = max(q, 0.0)
                qr[i, j, k] = q
                F_in = V_t * rho0[k] * q
            sfc[i, j] = F_in
    return sfc


@njit(parallel=True, cache=True)
def sedimentation_kb(qr, rho0, dz_face, dt, a_vt, b_vt, kb, dzb):
    """sedimentation with terrain: each column's sweep ends at its ground level kb[i, j]
    (first fluid cell, fluid thickness dzb[i, j] — less than Δz for partial cells);
    returns the flux leaving that cell (surface precipitation)."""
    nx, ny, nz = qr.shape
    sfc = np.empty((nx, ny))
    for i in prange(nx):
        for j in range(ny):
            F_in = 0.0
            for k in range(nz - 1, kb[i, j] - 1, -1):
                q = max(qr[i, j, k], 0.0)
                V_t = a_vt * ((rho0[k] * q / 1000.0) ** b_vt) * np.sqrt(rho0[0] / rho0[k])
                dzk = dzb[i, j] if k == kb[i, j] else dz_face[k]
                q = (q + (dt / dzk) * (F_in / rho0[k])) / (1.0 + V_t * dt / dzk)
                q = max(q, 0.0)
                qr[i, j, k] = q
                F_in = V_t * rho0[k] * q
            sfc[i, j] = F_in
    return sfc

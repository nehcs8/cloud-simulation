"""
Open lateral boundary conditions (stage 3 of docs/PLAN_open_boundaries.md).

Used for every direction with cfg.BC_X / BC_Y = "open". Reference ("environment") state:
θ_l′ = 0, q_t = qv0(z), q_r = 0, u = u_g(z), v = V_GEO, w = 0 — the case sounding and wind.
With OPEN_INFLOW_REF = "mean" (update_reference, once per step) θ_l′, q_t (and θ′, q_v for the relaxation)
follow the time-filtered mean of the clear columns below OPEN_REF_Z1, tapered to the sounding at OPEN_REF_Z2:
the inflow boundary layer is heated and moistened like the domain's. OPEN_CPM_AMP > 0 adds the cell
perturbation method (apply_cell_perturbations) so the inflow is turbulent too.

1. Inflow / outflow per boundary face (ny × nz for x-boundaries, nx × nz for y-boundaries),
   from the sign of the normal velocity on that face.
2. Scalar ghost cells (inflow_ghosts): inflow faces see the reference profile in the halo,
   outflow faces keep the zero-gradient ghosts of halo.py (anomalies leave freely).
3. Normal velocity on the boundary faces (radiation): Klemp & Wilhelmson (1978)
       ∂u_n/∂t = −(u_n ± c*) ∂u_n/∂n     (one-sided from the interior; c* = OPEN_C_STAR,
                                          outgoing gravity waves) − (u_n − u_ref)/OPEN_TAU_NORMAL
   replacing the momentum tendency on those faces; the projection then only balances the net
   boundary mass flux (pressure.balance_boundary_flux).
4. Relaxation zone (apply_relaxation, physics step): the OPEN_RELAX_WIDTH cells next to an inflow
   face relax θ, q_v, q_c, q_r, u, v, w toward the reference with weight ((W − d)/W)² / OPEN_RELAX_TAU
   (d = 0 at the boundary cell); outflow faces are left alone.
5. Water budget: outward flux of total water (ρ₀ q_t — q_t includes q_r; advection + diffusion) through the
   open boundary faces, set by dynamics.compute_tendencies each RK stage; rk3_step accumulates the
   last stage's rate × dt in s.boundary_water_out [kg] (Wicker–Skamarock: s^{n+1} = s^n + dt·RHS(stage 3)).
"""

import numpy as np
import config as cfg
from halo import H, open_x, open_y


def any_open() -> bool:
    return open_x() or open_y()


def references(s) -> dict:
    """Reference (nz,) profiles of the prognostic fields (θ, q_v, q_c also for the relaxation zone)."""
    nz = s.theta_l.shape[2]
    ref = dict(theta_l=np.zeros(nz), qt=np.asarray(s.qv0, dtype=float), qr=np.zeros(nz), qg=np.zeros(nz), nr=np.zeros(nz),
               u=np.asarray(s.u_g, dtype=float), v=np.full(nz, cfg.V_GEO),
               theta=np.zeros(nz), qv=np.asarray(s.qv0, dtype=float))
    if getattr(s, 'open_ref', None) is not None:
        ref.update(s.open_ref)
    return ref


def clear_columns(s, g) -> np.ndarray:
    """(nx, ny) mask of columns without hydrometeors and outside cold pools."""
    q = s.qc + s.qr + s.qg
    clear = q.max(axis=2) < cfg.OPEN_REF_QMIN
    k0 = 0 if g.terrain is None else g.terrain.k_sfc
    th_sfc = s.theta[:, :, 0] if g.terrain is None else np.take_along_axis(s.theta, k0[..., None], 2)[..., 0]
    clear &= th_sfc >= np.median(th_sfc) - cfg.OPEN_REF_COLD
    return clear


def update_reference(s, g, dt: float) -> None:
    """OPEN_INFLOW_REF = "mean": update s.open_ref (θ′, θ_l′, q_v, q_t) toward the clear-column mean."""
    if getattr(cfg, 'OPEN_INFLOW_REF', 'sounding') != 'mean' or not any_open():
        return
    qv0 = np.asarray(s.qv0, dtype=float)
    if getattr(s, 'open_ref', None) is None:
        nz = qv0.size
        s.open_ref = dict(theta=np.zeros(nz), theta_l=np.zeros(nz), qv=qv0.copy(), qt=qv0.copy())
    m = clear_columns(s, g)
    if m.sum() < 0.05 * m.size:          # (almost) overcast: keep the previous reference
        return
    wz = np.clip((cfg.OPEN_REF_Z2 - g.z) / (cfg.OPEN_REF_Z2 - cfg.OPEN_REF_Z1), 0.0, 1.0)
    fluid = None if g.terrain is None else ~g.terrain.mask
    a = 1.0 - np.exp(-dt / cfg.OPEN_REF_TAU)
    for key, f, base in (('theta', s.theta, 0.0), ('theta_l', s.theta_l, 0.0), ('qv', s.qv, qv0), ('qt', s.qt, qv0)):
        if fluid is None:
            mean = f[m].mean(axis=0)
        else:                            # levels below the ground in some columns: mean over fluid cells
            wgt = fluid & m[..., None]
            n = wgt.sum(axis=(0, 1))
            mean = np.where(n > 0, (f * wgt).sum(axis=(0, 1)) / np.maximum(n, 1), base)
        target = base + wz * (mean - base)
        s.open_ref[key] += a * (target - s.open_ref[key])


def inflow_masks(s) -> dict:
    """Boolean inflow masks on the four boundary face sets (only for open directions)."""
    m = {}
    if open_x():
        m['w'] = s.u[0] > 0.0            # (ny, nz): west face, flow into the domain
        m['e'] = s.u[-1] < 0.0
    if open_y():
        m['s'] = s.v[:, 0] > 0.0         # (nx, nz)
        m['n'] = s.v[:, -1] < 0.0
    return m


def inflow_ghosts(P: np.ndarray, ref: np.ndarray, masks: dict) -> np.ndarray:
    """Set the ghost cells of a halo-padded cell-centred scalar to ref(z) on inflow faces."""
    nx, ny = P.shape[0] - 2*H, P.shape[1] - 2*H
    Y, X = slice(H, H + ny), slice(H, H + nx)
    if 'w' in masks:
        for g in range(H):
            P[g, Y] = np.where(masks['w'], ref, P[g, Y])
            P[H + nx + g, Y] = np.where(masks['e'], ref, P[H + nx + g, Y])
    if 's' in masks:
        for g in range(H):
            P[X, g] = np.where(masks['s'], ref, P[X, g])
            P[X, H + ny + g] = np.where(masks['n'], ref, P[X, H + ny + g])
    return P


def radiation_tendency(tend_u: np.ndarray, tend_v: np.ndarray, s, g) -> None:
    """Replace the tendencies of the boundary-face normal velocities (in place)."""
    c = cfg.OPEN_C_STAR
    tau = cfg.OPEN_TAU_NORMAL
    if open_x():
        u = s.u
        ref = s.u_g[np.newaxis, :]
        relax_w = (ref - u[0]) / tau if tau > 0 else 0.0
        relax_e = (ref - u[-1]) / tau if tau > 0 else 0.0
        cw = np.minimum(u[0] - c, 0.0)                 # outgoing (−x) phase speed at the west face
        ce = np.maximum(u[-1] + c, 0.0)                # outgoing (+x) at the east face
        tend_u[0] = -cw * (u[1] - u[0]) / g.dx + relax_w
        tend_u[-1] = -ce * (u[-1] - u[-2]) / g.dx + relax_e
    if open_y():
        v = s.v
        ref = cfg.V_GEO
        relax_s = (ref - v[:, 0]) / tau if tau > 0 else 0.0
        relax_n = (ref - v[:, -1]) / tau if tau > 0 else 0.0
        cs = np.minimum(v[:, 0] - c, 0.0)
        cn = np.maximum(v[:, -1] + c, 0.0)
        tend_v[:, 0] = -cs * (v[:, 1] - v[:, 0]) / g.dy + relax_s
        tend_v[:, -1] = -cn * (v[:, -1] - v[:, -2]) / g.dy + relax_n


def _weno_face(P, sl_m2, sl_m1, sl_0, sl_p1, vel):
    """WENO-3 flux vel·φ at a set of faces from four padded slices (cells m−2 … m+1)."""
    from dynamics import _weno3_faces, _weno_eps
    c = P[sl_m1]
    return _weno3_faces(P[sl_m2], c, P[sl_0], P[sl_p1], vel, _weno_eps(c))


def boundary_outflow(P: np.ndarray, KP: np.ndarray, s, g) -> float:
    """
    Outward flux Σ ρ₀ (u_n φ − K ∂φ/∂n) dA [φ-units · kg/s] through all open boundary faces,
    for a halo-padded scalar P with the same face values as dynamics (WENO-3 + K-diffusion).
    """
    nx, ny = g.nx, g.ny
    total = 0.0
    if open_x():
        dA = (g.dy * g.dz_face * s.rho0)[np.newaxis, :]          # (1, nz)
        Y = slice(H, H + ny)
        for m, sign in ((0, -1.0), (nx, 1.0)):                    # face m: between cells m−1, m
            I = H + m
            F = _weno_face(P, (I - 2, Y), (I - 1, Y), (I, Y), (I + 1, Y), s.u[m])
            Kf = 0.5 * (KP[I - 1, Y] + KP[I, Y])
            D = -Kf * (P[I, Y] - P[I - 1, Y]) / g.dx              # diffusive flux in +x
            total += sign * float(((F + D) * dA).sum())
    if open_y():
        dA = (g.dx * g.dz_face * s.rho0)[np.newaxis, :]
        X = slice(H, H + nx)
        for m, sign in ((0, -1.0), (ny, 1.0)):
            J = H + m
            F = _weno_face(P, (X, J - 2), (X, J - 1), (X, J), (X, J + 1), s.v[:, m])
            Kf = 0.5 * (KP[X, J - 1] + KP[X, J])
            D = -Kf * (P[X, J] - P[X, J - 1]) / g.dy
            total += sign * float(((F + D) * dA).sum())
    return total


def apply_relaxation(s, g, dt: float) -> None:
    """Davies-type relaxation toward the reference in the inflow cells next to open boundaries."""
    update_reference(s, g, dt)
    W = int(cfg.OPEN_RELAX_WIDTH)
    if W <= 0 or not any_open():
        apply_cell_perturbations(s, g, dt)
        return
    masks = inflow_masks(s)
    a = 1.0 - np.exp(-dt / cfg.OPEN_RELAX_TAU)
    wts = a * ((W - np.arange(W)) / W) ** 2                     # d = 0 … W−1
    ref = references(s)
    th_ref, qv_ref = ref['theta'], ref['qv']

    def relax_cells(sel, m, wt):
        """Relax cell-centred fields at index tuple sel where inflow mask m (…, nz)."""
        r = wt * m
        s.theta[sel] += (th_ref - s.theta[sel]) * r
        s.qv[sel] += (qv_ref - s.qv[sel]) * r
        s.qc[sel] += (0.0 - s.qc[sel]) * r
        s.qr[sel] += (0.0 - s.qr[sel]) * r
        s.qg[sel] += (0.0 - s.qg[sel]) * r
        s.nr[sel] += (0.0 - s.nr[sel]) * r
        if getattr(cfg, 'OPEN_RELAX_W', True):                    # False: inflow thermals keep their w
            w_c = s.w[sel]                                       # w levels 0…nz-1 of these cells
            r_w = np.concatenate([r, r[..., -1:]], axis=-1)      # (…, nz+1)
            s.w[sel] = w_c + (0.0 - w_c) * r_w

    nx, ny = g.nx, g.ny
    if open_x():
        for d in range(min(W, nx // 2)):
            for i, key in ((d, 'w'), (nx - 1 - d, 'e')):
                m = masks[key].astype(float)                     # (ny, nz)
                relax_cells((i,), m, wts[d])
                s.v[i] += (ref['v'] - s.v[i]) * wts[d] * np.concatenate([m, m[-1:]], axis=0)
            # interior u faces next to the boundary (the boundary faces follow the radiation BC)
            for f, key in ((d + 1, 'w'), (nx - 1 - d, 'e')):
                m = masks[key].astype(float)
                s.u[f] += (ref['u'] - s.u[f]) * wts[d] * m
    if open_y():
        for d in range(min(W, ny // 2)):
            for j, key in ((d, 's'), (ny - 1 - d, 'n')):
                m = masks[key].astype(float)                     # (nx, nz)
                relax_cells((slice(None), j), m, wts[d])
                s.u[:, j] += (ref['u'] - s.u[:, j]) * wts[d] * np.concatenate([m, m[-1:]], axis=0)
            for f, key in ((d + 1, 's'), (ny - 1 - d, 'n')):
                m = masks[key].astype(float)
                s.v[:, f] += (ref['v'] - s.v[:, f]) * wts[d] * m
    s.w[:, :, 0] = 0.0
    s.w[:, :, -1] = 0.0
    apply_cell_perturbations(s, g, dt)


def apply_cell_perturbations(s, g, dt: float) -> None:
    """Cell perturbation method: random θ offsets in blocks next to inflow faces, every OPEN_CPM_PERIOD."""
    amp = getattr(cfg, 'OPEN_CPM_AMP', 0.0)
    if amp <= 0.0 or not any_open():
        return
    c, nb = int(cfg.OPEN_CPM_CELL), int(cfg.OPEN_CPM_BANDS)
    kz = np.nonzero(g.z < cfg.OPEN_CPM_ZTOP)[0]
    if getattr(s, 'cpm_rng', None) is None:
        s.cpm_rng = np.random.default_rng(cfg.OPEN_CPM_SEED)
        s.cpm_clock = np.inf                                       # perturb at the first call
    period = cfg.OPEN_CPM_PERIOD
    if period <= 0.0:
        ref = references(s)
        speed = np.hypot(ref['u'][kz], ref['v'][kz]).mean()
        period = c * min(g.dx, g.dy) / max(speed, 1.0)
    s.cpm_clock += dt
    if s.cpm_clock < period:
        return
    s.cpm_clock = 0.0
    masks = inflow_masks(s)
    nx, ny = g.nx, g.ny

    def blocks(n_along, n_across):
        """Random field (n_along, n_across, len(kz)) constant over c × c blocks."""
        r = s.cpm_rng.uniform(-amp, amp, (-(-n_along // c), -(-n_across // c), kz.size))
        return np.repeat(np.repeat(r, c, 0), c, 1)[:n_along, :n_across]

    d = min(nb * c, nx // 2 if open_x() else nb * c, ny // 2 if open_y() else nb * c)
    if open_x():
        for key, sl in (('w', slice(0, d)), ('e', slice(nx - d, nx))):
            m = masks[key][:, kz].astype(float)                    # (ny, nkz): inflow at that face
            p = blocks(d, ny) if key == 'w' else blocks(d, ny)[::-1]
            s.theta[sl, :, kz[0]:kz[-1] + 1] += p * m[None]
    if open_y():
        for key, sl in (('s', slice(0, d)), ('n', slice(ny - d, ny))):
            m = masks[key][:, kz].astype(float)                    # (nx, nkz)
            p = blocks(d, nx).transpose(1, 0, 2)
            if key == 'n':
                p = p[:, ::-1]
            s.theta[:, sl, kz[0]:kz[-1] + 1] += p * m[:, None]

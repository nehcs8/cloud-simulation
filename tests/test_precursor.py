"""
Test the concurrent precursor's inflow on every open face (precursor.py, OPEN_INFLOW_REF = "precursor").

Main 24 × 20 open/open, precursor 16 × 12, random precursor fields, a main flow that enters through
west and south and leaves through east and north, plus a few reversed points on each face.

  1. tiling: Precursor.at returns the precursor cell (i mod 16, j mod 12) for any main index, incl.
     negative ghost indices and face index n (= face 0)
  2. ghosts: inflow points of all four faces get the precursor cells under the ghosts, outflow points keep
     the zero-gradient ghosts
  3. boundary-face normal velocity: radiation tendency everywhere, plus at inflow points relaxation of the
     deviation from the face mean toward the precursor's deviation (PRECURSOR_TAU_U); the face mean is not forced
  4. relaxation zone: inflow cells next to the west and south faces move toward the precursor cells under
     them (all fields incl. w), the outflow (east, north) zones are left alone
  5. periodic main directions must be multiples of the precursor size

Usage: python tests/test_precursor.py
"""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config as cfg
cfg.NX, cfg.NY, cfg.DX, cfg.DY = 24, 20, 100.0, 100.0
cfg.GRID_TYPE = "uniform_stretched"
cfg.DZ_UNIFORM, cfg.Z_UNIFORM_TOP, cfg.Z_TOP, cfg.N_STRETCH = 50.0, 1000.0, 2000.0, 6
cfg.BC_X = cfg.BC_Y = "open"
cfg.U_GEO, cfg.V_GEO = 3.0, 2.0
cfg.OPEN_INFLOW_REF = "precursor"
cfg.PRECURSOR_NX, cfg.PRECURSOR_NY = 16, 12

from grid import Grid
from state import State
from precursor import Precursor
import halo
import open_bc

H = halo.H
g = Grid(terrain=None)
s = State(g)
pre = Precursor(g)
s.precursor = pre
rng = np.random.default_rng(1)
ps = pre.s
for k in ('theta_l', 'qt', 'theta', 'qv', 'qc', 'qr', 'w'):
    setattr(ps, k, getattr(ps, k) + rng.normal(0.0, 1.0, getattr(ps, k).shape))
ps.u = 3.0 + rng.normal(0.0, 0.5, ps.u.shape)
ps.v = 2.0 + rng.normal(0.0, 0.5, ps.v.shape)
ps.u[-1], ps.v[:, -1] = ps.u[0], ps.v[:, 0]            # periodic: face n is face 0

s.u[:] = 3.0
s.v[:] = 2.0
s.u[0, :3] = s.u[-1, :3] = -1.0                        # reversed points: west outflow, east inflow
s.v[:3, 0] = s.v[:3, -1] = -1.0
nx, ny = g.nx, g.ny
ok = True


def check(name, cond):
    global ok
    print(f"{name}: {'ok' if cond else 'FAILED'}")
    ok &= bool(cond)


# 1. tiling
check("tiling (incl. ghosts, face n = face 0)",
      np.array_equal(pre.at('theta_l', [-1, 0, 16, 23], [-2, 13]),
                     ps.theta_l[np.ix_([15, 0, 0, 7], [10, 1])])
      and np.array_equal(pre.at('u', [24], [0])[0, 0], ps.u[8, 0]))

# 2. ghosts
masks = open_bc.inflow_masks(s)
P = halo.pad_c(s.theta_l)
edge = P.copy()                                         # zero-gradient ghosts before inflow_ghosts
open_bc.inflow_ghosts(P, np.zeros(g.nz), masks, pre.ghost_planes('theta_l', H, nx, ny, masks))
G = np.arange(H)
west = pre.at('theta_l', G - H, np.arange(ny))
north = pre.at('theta_l', np.arange(nx), ny + G)
mw = masks['w'][None]
mn = masks['n'][:, None]
check("west ghosts: precursor at inflow, edge at outflow",
      np.array_equal(P[:H, H:H + ny], np.where(mw, west, edge[:H, H:H + ny])) and masks['w'].sum() < masks['w'].size)
check("north ghosts: precursor at inflow points only",
      np.array_equal(P[H:H + nx, H + ny:], np.where(mn, north, edge[H:H + nx, H + ny:])) and masks['n'].any())

# 3. normal velocity on the boundary faces
tu, tv = np.zeros_like(s.u), np.zeros_like(s.v)
open_bc.radiation_tendency(tu, tv, s, g)
cfg_pre = cfg.OPEN_INFLOW_REF
s.precursor = None
ru, rv = np.zeros_like(s.u), np.zeros_like(s.v)
open_bc.radiation_tendency(ru, rv, s, g)
s.precursor = pre
J, I = np.arange(ny), np.arange(nx)
pert = lambda p, f: ((p - p.mean(0)) - (f - f.mean(0))) / cfg.PRECURSOR_TAU_U
exp_w = ru[0] + (s.u[0] > 0) * pert(pre.at('u', [0], J)[0], s.u[0])
exp_e = ru[-1] + (s.u[-1] < 0) * pert(pre.at('u', [nx], J)[0], s.u[-1])
exp_n = rv[:, -1] + (s.v[:, -1] < 0) * pert(pre.at('v', I, [ny])[:, 0], s.v[:, -1])
check("u_n: radiation + precursor turbulence at inflow points, radiation only at outflow points (west, east, north)",
      np.allclose(tu[0], exp_w) and np.allclose(tu[-1], exp_e) and np.allclose(tv[:, -1], exp_n)
      and not np.allclose(tu[-1], ru[-1]))

# 4. relaxation zone
before = {k: getattr(s, k).copy() for k in ('theta', 'w', 'u')}
open_bc.apply_relaxation(s, g, 30.0)
tw = pre.at('theta', [0], J)[0]
ts = pre.at('theta', I, [0])[:, 0]
dist = lambda f, t: np.abs(f - t)
m_w = masks['w']
m_s = masks['s'][3:]                                    # skip the corner columns the west zone also relaxes
check("west zone moves toward the precursor (θ, w)",
      (dist(s.theta[0], tw)[m_w] < dist(before['theta'][0], tw)[m_w]).all()
      and (dist(s.w[0, :, 1:-1], pre.at('w', [0], J)[0][:, 1:-1])[m_w[:, 1:]]
           < dist(before['w'][0, :, 1:-1], pre.at('w', [0], J)[0][:, 1:-1])[m_w[:, 1:]]).all())
check("south zone moves toward the precursor (θ)",
      (dist(s.theta[4:, 0], ts[4:])[m_s[1:]] < dist(before['theta'][4:, 0], ts[4:])[m_s[1:]]).all())
check("east / north outflow zones untouched (θ)",
      np.array_equal(s.theta[-1, 4:-4][~masks['e'][4:-4]], before['theta'][-1, 4:-4][~masks['e'][4:-4]])
      and np.array_equal(s.theta[8:-4, -1][~masks['n'][8:-4]], before['theta'][8:-4, -1][~masks['n'][8:-4]]))

# 5. periodic main direction not a multiple of the precursor
cfg.BC_Y = "periodic"
try:
    Precursor(g)
    check("periodic NY = 20 vs precursor 12 refused", False)
except ValueError:
    check("periodic NY = 20 vs precursor 12 refused", True)
cfg.BC_Y = "open"

print("All precursor tests passed." if ok else "PRECURSOR TESTS FAILED")
sys.exit(0 if ok else 1)

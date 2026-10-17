"""
Halo (ghost-cell) padding for the horizontal stencils (stage 2 of docs/PLAN_open_boundaries.md).

Stencils in dynamics.py / kernels.py read horizontal neighbours from a padded copy of each
field instead of wrapping indices. The padding is a gather with one index map per direction:

  BC "periodic": ghost cells wrap around (identical values to the old np.roll / % n stencils).
                 For face-staggered arrays (u in x, v in y) the unique faces are 0 … n−1
                 (face n is face 0).
  BC "open":     ghost cells repeat the nearest boundary cell / boundary face (zero gradient).
                 Placeholder until stage 3 fills inflow / outflow ghost values.

Padded shapes (H = 2 ghost cells per side):
  pad_c(phi): cell-centred (nx, ny, …)     → (nx + 2H, ny + 2H, …)
  pad_u(u):   x-faces      (nx + 1, ny, nz) → (nx + 1 + 2H, ny + 2H, nz)
  pad_v(v):   y-faces      (nx, ny + 1, nz) → (nx + 2H, ny + 1 + 2H, nz)
Interior point (i, j) sits at (i + H, j + H).
"""

import numpy as np
import config as cfg

try:
    import kernels
except ImportError:
    kernels = None

H = 2


def open_x() -> bool:
    return getattr(cfg, 'BC_X', 'periodic') == 'open'


def open_y() -> bool:
    return getattr(cfg, 'BC_Y', 'periodic') == 'open'


_maps = {}


def index_map(n: int, face: bool, is_open: bool) -> np.ndarray:
    """Source index for each padded position along one axis (length n + 2H, or n + 1 + 2H for faces)."""
    key = (n, face, is_open)
    if key not in _maps:
        m = n + 1 if face else n
        idx = np.arange(-H, m + H)
        if is_open:
            idx = np.clip(idx, 0, m - 1)
        else:
            idx = idx % n
        _maps[key] = idx.astype(np.int64)
    return _maps[key]


def _gather(a: np.ndarray, mx: np.ndarray, my: np.ndarray) -> np.ndarray:
    if kernels is not None and getattr(cfg, 'USE_NUMBA', True):
        return kernels.gather_xy(np.ascontiguousarray(a), mx, my)
    return a[mx][:, my]


def pad_c(phi: np.ndarray) -> np.ndarray:
    """Cell-centred field (nx, ny, …) with H ghost cells on each horizontal side."""
    nx, ny = phi.shape[0], phi.shape[1]
    return _gather(phi, index_map(nx, False, open_x()), index_map(ny, False, open_y()))


def pad_u(u: np.ndarray) -> np.ndarray:
    """x-face field (nx+1, ny, nz): faces padded in x, cells in y."""
    nx, ny = u.shape[0] - 1, u.shape[1]
    return _gather(u, index_map(nx, True, open_x()), index_map(ny, False, open_y()))


def pad_v(v: np.ndarray) -> np.ndarray:
    """y-face field (nx, ny+1, nz): cells in x, faces padded in y."""
    nx, ny = v.shape[0], v.shape[1] - 1
    return _gather(v, index_map(nx, False, open_x()), index_map(ny, True, open_y()))

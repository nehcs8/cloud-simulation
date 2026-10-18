"""Grid geometry: uniform horizontal, exponentially stretched vertical."""

import numpy as np
from typing import Optional
import config as cfg


class Grid:
    """Holds all geometric arrays needed by the solver."""

    def __init__(self, terrain: Optional['Terrain'] = None) -> None:
        # ── Horizontal ────────────────────────────────────────────────────────
        self.nx = cfg.NX
        self.ny = cfg.NY
        self.dx = cfg.DX
        self.dy = cfg.DY

        self.x = (np.arange(cfg.NX) + 0.5) * cfg.DX   # cell centres
        self.y = (np.arange(cfg.NY) + 0.5) * cfg.DY

        # ── Vertical grid ─────────────────────────────────────────────────────
        # z_face: face positions (nz+1 values, including z=0 and z=Z_TOP)
        if cfg.GRID_TYPE == "uniform_stretched":
            self.z_face = _uniform_stretched_faces(
                cfg.DZ_UNIFORM, cfg.Z_UNIFORM_TOP, cfg.Z_TOP, cfg.N_STRETCH)
            cfg.NZ = len(self.z_face) - 1                 # NZ is derived for this grid
        else:
            # exponential stretching
            eta = np.linspace(0.0, 1.0, cfg.NZ + 1)
            beta = 1.5                                     # stretching factor
            self.z_face = cfg.Z_TOP * (np.exp(beta * eta) - 1.0) / (np.exp(beta) - 1.0)
        self.nz = len(self.z_face) - 1

        # cell-centre positions
        self.z = 0.5 * (self.z_face[:-1] + self.z_face[1:])

        # cell height (nz)
        self.dz_face = np.diff(self.z_face)               # Δz of each cell

        # distance between adjacent cell centres (nz-1, for internal faces)
        self.dz_cen = np.diff(self.z)                     # used in pressure solve

        # ── 3-D versions for broadcasting ─────────────────────────────────────
        # shape (1, 1, nz)
        self.Z3   = self.z[np.newaxis, np.newaxis, :]
        self.DZ3  = self.dz_face[np.newaxis, np.newaxis, :]

        # ── Terrain (optional) ────────────────────────────────────────────────
        self.terrain = terrain  # None for flat domain, or Terrain instance

    def z_agl(self) -> np.ndarray:
        """Height of the cell centres above the ground: (1, 1, nz) without terrain,
        (nx, ny, nz) with terrain (ground = bottom face of each column's first fluid cell)."""
        if self.terrain is None:
            return self.Z3
        ground = self.z_face[self.terrain.k_sfc]
        return self.z[np.newaxis, np.newaxis, :] - ground[:, :, np.newaxis]

    # ── Convenience ───────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        return (
            f"Grid({self.nx}×{self.ny}×{self.nz}, "
            f"dx={self.dx}m, dz={self.dz_face[0]:.1f}–{self.dz_face[-1]:.1f}m, "
            f"z_top={self.z_face[-1]:.0f}m)"
        )


def _uniform_stretched_faces(dz0: float, z_uniform: float, z_top: float,
                             n_stretch: int) -> np.ndarray:
    """
    Faces with uniform spacing dz0 up to z_uniform, then n_stretch levels with
    geometric growth dz0·r, dz0·r², … chosen (bisection on r) to end exactly at z_top.
    """
    n_uniform = int(round(z_uniform / dz0))
    if abs(n_uniform * dz0 - z_uniform) > 1e-6:
        raise ValueError(f"Z_UNIFORM_TOP={z_uniform} is not a multiple of DZ_UNIFORM={dz0}")
    depth = z_top - z_uniform
    if depth <= 0.0:
        return np.arange(n_uniform + 1) * dz0
    if depth <= n_stretch * dz0:
        raise ValueError("Stretched layer too shallow for N_STRETCH levels of at least DZ_UNIFORM")

    total = lambda r: dz0 * np.sum(r ** np.arange(1, n_stretch + 1))
    lo, hi = 1.0, 2.0
    while total(hi) < depth:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if total(mid) < depth else (lo, mid)
    r = 0.5 * (lo + hi)

    dz = np.concatenate([np.full(n_uniform, dz0), dz0 * r ** np.arange(1, n_stretch + 1)])
    z_face = np.concatenate([[0.0], np.cumsum(dz)])
    z_face[-1] = z_top                                     # remove round-off
    return z_face


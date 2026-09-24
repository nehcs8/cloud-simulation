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
        self.nz = cfg.NZ
        self.dx = cfg.DX
        self.dy = cfg.DY

        self.x = (np.arange(cfg.NX) + 0.5) * cfg.DX   # cell centres
        self.y = (np.arange(cfg.NY) + 0.5) * cfg.DY

        # ── Vertical (exponential stretching) ─────────────────────────────────
        # z_face: face positions (nz+1 values, including z=0 and z=Z_TOP)
        eta = np.linspace(0.0, 1.0, cfg.NZ + 1)
        beta = 1.5                                         # stretching factor
        self.z_face = cfg.Z_TOP * (np.exp(beta * eta) - 1.0) / (np.exp(beta) - 1.0)

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

    # ── Convenience ───────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        return (
            f"Grid({self.nx}×{self.ny}×{self.nz}, "
            f"dx={self.dx}m, dz={self.dz_face[0]:.1f}–{self.dz_face[-1]:.1f}m, "
            f"z_top={self.z_face[-1]:.0f}m)"
        )

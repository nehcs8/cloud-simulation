"""
Relaxation boundary conditions for nested simulations.

Nudges the outer rim of the nested domain toward parent solution to
prevent reflections and allow disturbances to exit cleanly.
"""

import numpy as np


class RelaxationBC:
    """
    Manages relaxation boundary conditions for nested domain.

    The outer n_relax cells are nudged toward parent fields with a
    spatially-varying timescale that decreases toward the boundary.
    """

    def __init__(self, nx, ny, n_relax=8, tau_min=5.0):
        """
        Parameters:
        -----------
        nx, ny : int
            Nested domain grid dimensions
        n_relax : int
            Number of cells in relaxation zone (default 8)
        tau_min : float
            Minimum relaxation timescale at boundary (seconds)
        """
        self.nx = nx
        self.ny = ny
        self.n_relax = n_relax
        self.tau_min = tau_min

        # Precompute relaxation timescale mask
        self.tau_mask = self._compute_tau_mask()

    def _compute_tau_mask(self):
        """
        Compute 2D array of relaxation timescales τ(i,j).

        Interior (dist >= n_relax): τ = ∞ (no relaxation)
        Boundary (dist = 0): τ = tau_min
        Transition: quadratic ramp

        Returns (nx, ny) array.
        """
        tau = np.full((self.nx, self.ny), np.inf)

        for i in range(self.nx):
            for j in range(self.ny):
                # Distance from nearest boundary (in cells)
                dist_x = min(i, self.nx - 1 - i)
                dist_y = min(j, self.ny - 1 - j)
                dist = min(dist_x, dist_y)

                if dist < self.n_relax:
                    # Quadratic ramping: τ = τ_min / (1 - d/n)²
                    # At boundary (d=0): τ = τ_min
                    # At d=n_relax: τ → ∞
                    frac = dist / self.n_relax
                    tau[i, j] = self.tau_min / max(0.01, (1 - frac) ** 2)

        return tau

    def apply(self, state_nest, parent_fields, dt):
        """
        Apply relaxation to nested domain toward parent fields.

        Parameters:
        -----------
        state_nest : State
            Nested domain state object
        parent_fields : dict
            Dictionary with keys 'u', 'v', 'w', 'theta', 'qv', 'qc', 'qr'
            Each value is (nx, ny, nz) array interpolated to nest grid
        dt : float
            Timestep (seconds)
        """
        # Get grid dimensions
        nz = state_nest.theta.shape[2]

        # Extend tau to 3D: (nx, ny, nz) by broadcasting
        tau_3d = np.broadcast_to(self.tau_mask[:, :, np.newaxis], (self.nx, self.ny, nz))

        # Relaxation factor: α = dt / τ (capped for stability)
        alpha = dt / tau_3d
        alpha = np.minimum(alpha, 0.5)  # Never relax more than 50% per step

        # Identify relaxation zone (where tau < inf)
        relax_zone = np.isfinite(tau_3d)

        # Apply to dynamic fields (u, v, theta, qv)
        for field_name in ['u', 'v', 'theta', 'qv']:
            phi_nest = getattr(state_nest, field_name)
            phi_parent = parent_fields[field_name]

            # φ_nest += α · (φ_parent - φ_nest)
            phi_nest[relax_zone] += alpha[relax_zone] * (
                phi_parent[relax_zone] - phi_nest[relax_zone]
            )

        # For w (face-staggered, shape nx×ny×(nz+1))
        # Apply same logic but with nz+1 levels
        nz_w = state_nest.w.shape[2]
        tau_3d_w = np.broadcast_to(self.tau_mask[:, :, np.newaxis], (self.nx, self.ny, nz_w))
        alpha_w = np.minimum(dt / tau_3d_w, 0.5)
        relax_zone_w = np.isfinite(tau_3d_w)

        w_nest = state_nest.w
        w_parent = parent_fields['w']

        w_nest[relax_zone_w] += alpha_w[relax_zone_w] * (
            w_parent[relax_zone_w] - w_nest[relax_zone_w]
        )

        # For cloud fields (qc, qr): only relax outgoing clouds
        # If cloud is leaving domain, nudge to parent (usually zero outside)
        # This prevents artificial cloud accumulation at boundaries
        for field_name in ['qc', 'qr']:
            phi_nest = getattr(state_nest, field_name)
            phi_parent = parent_fields.get(field_name, np.zeros_like(phi_nest))

            # Only relax where parent value is small (cloud exiting domain)
            outgoing = (phi_parent < 0.1e-3) & relax_zone
            phi_nest[outgoing] += alpha[outgoing] * (
                phi_parent[outgoing] - phi_nest[outgoing]
            )

    def info(self):
        """Return string describing BC configuration."""
        n_relax_cells = np.isfinite(self.tau_mask).sum()
        total_cells = self.nx * self.ny
        pct = 100 * n_relax_cells / total_cells

        return (
            f"RelaxationBC: {self.n_relax}-cell zone, τ_min={self.tau_min}s, "
            f"{n_relax_cells} cells ({pct:.1f}% of domain)"
        )


def create_boundary_handler(config_module):
    """
    Factory function to create BC handler from config.

    Parameters:
    -----------
    config_module : module
        Configuration module (e.g., config_nested)

    Returns:
    --------
    RelaxationBC or None
    """
    if not getattr(config_module, 'USE_RELAXATION_BC', False):
        return None

    bc = RelaxationBC(
        nx=config_module.NX,
        ny=config_module.NY,
        n_relax=getattr(config_module, 'N_RELAX', 8),
        tau_min=getattr(config_module, 'TAU_RELAX_MIN', 5.0),
    )

    print(f"  {bc.info()}")
    return bc

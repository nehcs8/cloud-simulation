"""
Terrain representation for immersed boundary method.

Supports:
  - Synthetic terrain (Gaussian hills, sine ridges) for testing
  - Real DEM loading and regridding (future: Phase 2b)
  - 3D terrain mask generation for periodic domains
"""

import numpy as np
from typing import Optional
from grid import Grid


class Terrain:
    """
    Immersed boundary terrain for cloud simulations.

    Attributes:
        h (np.ndarray): Terrain height field, shape (nx, ny) [m]
        mask (np.ndarray): 3D boolean mask, shape (nx, ny, nz).
                          True where z < h(x,y) (inside terrain)
        k_sfc (np.ndarray): Surface level index for each column, shape (nx, ny).
                           Lowest grid level k where z[k] >= h[i,j]
    """

    def __init__(self, h_2d: np.ndarray, grid: Grid):
        """
        Initialize terrain from a 2D height field.

        Parameters:
            h_2d: Terrain elevation [m], shape (nx, ny)
            grid: Grid instance defining the computational domain
        """
        if h_2d.shape != (grid.nx, grid.ny):
            raise ValueError(
                f"Terrain shape {h_2d.shape} does not match grid ({grid.nx}, {grid.ny})"
            )

        # Store terrain height
        self.h = h_2d.copy()

        # Create 3D mask: True where z < h(x,y)
        # Broadcasting: z is (nz,), h is (nx,ny) → need (nx,ny,nz) comparison
        Z_3d = grid.z[np.newaxis, np.newaxis, :]  # shape (1, 1, nz)
        H_3d = h_2d[:, :, np.newaxis]              # shape (nx, ny, 1)
        self.mask = Z_3d < H_3d                     # shape (nx, ny, nz)

        # Surface level index: first k where z[k] >= h[i,j]
        # If terrain is above all grid levels, k_sfc = nz-1 (top level)
        above = grid.z[np.newaxis, np.newaxis, :] >= h_2d[:, :, np.newaxis]
        self.k_sfc = np.argmax(above, axis=2)      # shape (nx, ny)

        # Handle edge case: terrain higher than domain top
        # (in this case argmax returns 0, which is wrong)
        too_high = h_2d >= grid.z[-1]
        if too_high.any():
            print(f"Warning: {too_high.sum()} grid columns have terrain above domain top.")
            print(f"  Max terrain height: {h_2d.max():.1f} m, domain top: {grid.z[-1]:.1f} m")
            print("  These columns will be treated as solid (all levels masked).")
            # Set k_sfc to last level for these columns
            self.k_sfc[too_high] = grid.nz - 1

    # ── Class methods for creating synthetic terrains ────────────────────────

    @classmethod
    def from_gaussian_hill(
        cls,
        grid: Grid,
        h_max: float,
        x_center: Optional[float] = None,
        y_center: Optional[float] = None,
        sigma: float = 2000.0,
    ) -> "Terrain":
        """
        Create a Gaussian hill for testing.

        h(x,y) = h_max * exp(-r² / (2σ²))

        where r² = (x - x_c)² + (y - y_c)²

        Parameters:
            grid: Grid instance
            h_max: Maximum height at center [m]
            x_center: Hill center x-coordinate [m]. Default: domain center
            y_center: Hill center y-coordinate [m]. Default: domain center
            sigma: Gaussian width [m]

        Returns:
            Terrain instance
        """
        if x_center is None:
            x_center = grid.x.mean()
        if y_center is None:
            y_center = grid.y.mean()

        X, Y = np.meshgrid(grid.x, grid.y, indexing='ij')
        r_squared = (X - x_center)**2 + (Y - y_center)**2
        h = h_max * np.exp(-r_squared / (2 * sigma**2))

        return cls(h, grid)

    @classmethod
    def from_sine_ridge(
        cls,
        grid: Grid,
        h_max: float,
        wavelength: Optional[float] = None,
        axis: str = 'x',
    ) -> "Terrain":
        """
        Create a sinusoidal ridge (2D mountain range).

        axis='x': h(x,y) = h_max * sin(2π x / λ)  (ridge along y-direction)
        axis='y': h(x,y) = h_max * sin(2π y / λ)  (ridge along x-direction)

        Parameters:
            grid: Grid instance
            h_max: Ridge amplitude [m]
            wavelength: Ridge wavelength [m]. Default: domain size
            axis: Direction perpendicular to ridge ('x' or 'y')

        Returns:
            Terrain instance
        """
        if wavelength is None:
            wavelength = grid.nx * grid.dx if axis == 'x' else grid.ny * grid.dy

        X, Y = np.meshgrid(grid.x, grid.y, indexing='ij')

        if axis == 'x':
            h = h_max * np.sin(2 * np.pi * X / wavelength)
        elif axis == 'y':
            h = h_max * np.sin(2 * np.pi * Y / wavelength)
        else:
            raise ValueError(f"axis must be 'x' or 'y', got '{axis}'")

        # Shift to ensure h >= 0
        h = h_max * (1 + np.sin(2 * np.pi * (X if axis == 'x' else Y) / wavelength)) / 2

        return cls(h, grid)

    @classmethod
    def from_random_field(
        cls,
        grid: Grid,
        h_mean: float,
        h_std: float,
        correlation_length: float,
        seed: int = 42,
    ) -> "Terrain":
        """
        Create terrain from a correlated random field.

        Useful for testing flow response to rough terrain.

        Parameters:
            grid: Grid instance
            h_mean: Mean terrain elevation [m]
            h_std: Standard deviation of elevation [m]
            correlation_length: Spatial correlation scale [m]
            seed: Random seed for reproducibility

        Returns:
            Terrain instance
        """
        from scipy.ndimage import gaussian_filter

        rng = np.random.default_rng(seed)
        sigma_pixels = correlation_length / grid.dx  # correlation in grid cells

        # Generate white noise
        noise = rng.standard_normal((grid.nx, grid.ny))

        # Apply Gaussian filter to create spatial correlation
        h_corr = gaussian_filter(noise, sigma=sigma_pixels)

        # Normalize to unit variance, then scale and shift
        h_corr = h_corr / h_corr.std()
        h = h_mean + h_std * h_corr

        # Ensure non-negative terrain
        h = np.maximum(h, 0.0)

        return cls(h, grid)

    # ── Future: DEM loading ───────────────────────────────────────────────────

    @classmethod
    def from_dem_file(
        cls,
        grid: Grid,
        dem_path: str,
        smooth_sigma: float = 100.0,
    ) -> "Terrain":
        """
        Load terrain from a DEM file (GeoTIFF or NetCDF).

        **Not yet implemented** - placeholder for Phase 2b.

        Will:
          1. Load DEM using rasterio (GeoTIFF) or xarray (NetCDF)
          2. Crop to domain extent
          3. Regrid to model resolution using scipy.interpolate
          4. Optionally smooth with Gaussian filter

        Parameters:
            grid: Grid instance
            dem_path: Path to DEM file
            smooth_sigma: Smoothing scale [m]. Set to 0 for no smoothing.

        Returns:
            Terrain instance
        """
        raise NotImplementedError(
            "DEM loading not yet implemented. Use synthetic terrain for now:\n"
            "  - Terrain.from_gaussian_hill()\n"
            "  - Terrain.from_sine_ridge()\n"
            "  - Terrain.from_random_field()"
        )

    # ── Utility methods ───────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Terrain(h_min={self.h.min():.1f}m, h_max={self.h.max():.1f}m, "
            f"h_mean={self.h.mean():.1f}m, filled={self.mask.mean()*100:.1f}%)"
        )

    def stats(self) -> dict:
        """
        Compute terrain statistics.

        Returns:
            Dictionary with terrain metrics
        """
        # Compute slopes (finite differences)
        dh_dx = np.gradient(self.h, axis=0)
        dh_dy = np.gradient(self.h, axis=1)
        slope = np.sqrt(dh_dx**2 + dh_dy**2)

        return {
            'h_min': float(self.h.min()),
            'h_max': float(self.h.max()),
            'h_mean': float(self.h.mean()),
            'h_std': float(self.h.std()),
            'slope_mean': float(slope.mean()),
            'slope_max': float(slope.max()),
            'volume_fraction': float(self.mask.mean()),  # fraction of domain filled
        }

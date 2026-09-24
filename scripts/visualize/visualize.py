"""
Live matplotlib visualisation: four panels updated every output step.

  Top-left:    qc cross-section at y = NY/2  (cloud water, x-z plane)
  Top-right:   qc plan view at z ≈ 1500 m     (cloud fraction map)
  Bottom-left: w cross-section at y = NY/2    (vertical velocity)
  Bottom-right: domain-mean vertical profiles  (θ′, qv, qc)
"""

import numpy as np
import matplotlib
import os
# Use Agg backend if no DISPLAY (headless/background mode)
if os.environ.get('DISPLAY') is None:
    matplotlib.use("Agg")
else:
    matplotlib.use("TkAgg")          # change to "Qt5Agg" if TkAgg not available
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from grid import Grid
from state import State


class Visualizer:
    def __init__(self, grid: Grid) -> None:
        self.grid = grid
        g = grid

        self.fig, self.axes = plt.subplots(2, 2, figsize=(13, 8))
        self.fig.suptitle("Cloud simulation", fontsize=12, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.94])

        ax = self.axes
        # x-z extents in km
        xkm = g.x / 1e3
        zkm = g.z / 1e3

        # ── Panel 0: qc x-z ───────────────────────────────────────────────────
        self.im0 = ax[0, 0].imshow(
            np.zeros((g.nz, g.nx)),
            origin="lower", aspect="auto",
            extent=[xkm[0], xkm[-1], zkm[0], zkm[-1]],
            cmap="Blues", vmin=0, vmax=0.5,
        )
        ax[0, 0].set_xlabel("x [km]"); ax[0, 0].set_ylabel("z [km]")
        ax[0, 0].set_title("qc [g/kg]  (x-z slice)")
        self.fig.colorbar(self.im0, ax=ax[0, 0], fraction=0.03)

        # ── Panel 1: qc plan view ──────────────────────────────────────────────
        ykm = g.y / 1e3
        self.im1 = ax[0, 1].imshow(
            np.zeros((g.ny, g.nx)),
            origin="lower", aspect="equal",
            extent=[xkm[0], xkm[-1], ykm[0], ykm[-1]],
            cmap="Blues", vmin=0, vmax=0.5,
        )
        ax[0, 1].set_xlabel("x [km]"); ax[0, 1].set_ylabel("y [km]")
        ax[0, 1].set_title("qc [g/kg]  (plan view, z≈2.5 km)")
        self.fig.colorbar(self.im1, ax=ax[0, 1], fraction=0.03)

        # ── Panel 2: w x-z ────────────────────────────────────────────────────
        self.im2 = ax[1, 0].imshow(
            np.zeros((g.nz, g.nx)),
            origin="lower", aspect="auto",
            extent=[xkm[0], xkm[-1], zkm[0], zkm[-1]],
            cmap="RdBu_r", vmin=-5, vmax=5,
        )
        ax[1, 0].set_xlabel("x [km]"); ax[1, 0].set_ylabel("z [km]")
        ax[1, 0].set_title("w [m/s]  (x-z slice)")
        self.fig.colorbar(self.im2, ax=ax[1, 0], fraction=0.03)

        # ── Panel 3: vertical profiles ────────────────────────────────────────
        ax[1, 1].set_xlabel("Value"); ax[1, 1].set_ylabel("z [km]")
        ax[1, 1].set_title("Domain-mean profiles")
        self.ln_theta, = ax[1, 1].plot([], [], "r-",  label="θ′ [K]")
        self.ln_qv,    = ax[1, 1].plot([], [], "b--", label="qv [g/kg]")
        self.ln_qc,    = ax[1, 1].plot([], [], "c-",  label="qc [g/kg]×10")
        ax[1, 1].set_xlim(-2, 16)
        ax[1, 1].set_ylim(0, grid.z_face[-1] / 1e3)
        ax[1, 1].legend(fontsize=8)
        ax[1, 1].grid(True, alpha=0.3)

        self._plan_level = self._find_level(2500.0)
        self.t_text = self.fig.text(0.5, 0.955, "", ha="center", fontsize=10)
        plt.ion()
        plt.show()

    def _find_level(self, z_target: float) -> int:
        return int(np.argmin(np.abs(self.grid.z - z_target)))

    def update(self, s: State, t: float) -> None:
        g = self.grid
        jmid = g.ny // 2

        qc_xz = s.qc[:, jmid, :].T * 1e3          # (nz, nx), g/kg
        qc_xy = s.qc[:, :, self._plan_level].T * 1e3  # (ny, nx), g/kg
        w_xz  = 0.5 * (s.w[:, jmid, :-1] + s.w[:, jmid, 1:]).T  # (nz, nx)

        self.im0.set_data(qc_xz)
        self.im1.set_data(qc_xy)
        self.im2.set_data(w_xz)

        zkm = g.z / 1e3
        theta_mean = s.theta.mean(axis=(0, 1))
        qv_mean    = s.qv.mean(axis=(0, 1)) * 1e3
        qc_mean    = s.qc.mean(axis=(0, 1)) * 1e3 * 10

        self.ln_theta.set_data(theta_mean, zkm)
        self.ln_qv.set_data(qv_mean, zkm)
        self.ln_qc.set_data(qc_mean, zkm)

        self.t_text.set_text(f"t = {t/60:.1f} min")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

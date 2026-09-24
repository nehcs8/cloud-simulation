"""Configuration for nested cloud simulation (ultra-lightweight test)."""

# ── Parent simulation reference ────────────────────────────────────────────────
PARENT_SNAPSHOT = "experiments/phase2_nudging/exp13_tau3600/snapshots/snap_0001200.npz"
CLOUD_CENTER = (44, 56, 44)  # (i, j, k) in parent grid - Cloud #16 (nice cumulus)
CLOUD_ID = 16  # Cloud identifier from find_cloud.py

# ── Nested domain ──────────────────────────────────────────────────────────────
NX: int   = 60           # cells in x (3.0 km / 50 m)
NY: int   = 60           # cells in y
NZ: int   = 60           # cells in z (matches available data from setup)
DX: float = 50.0         # m
DY: float = 50.0         # m
Z_TOP: float = 3000.0    # m - reduced domain top (most clouds < 3 km)

# ── Time ───────────────────────────────────────────────────────────────────────
DT: float = 0.4          # s
T_END: float = 1800.0    # s - 30 minutes
OUTPUT_EVERY: float = 60.0  # s - output every 60s (30 snapshots)

# ── Reference atmosphere (inherit from parent) ─────────────────────────────────
THETA_S: float = 300.0
Z_BL: float   = 1500.0
GAMMA_BL: float = 1.0
GAMMA_FT: float = 6.0
QV_SURF: float = 0.016    # Same as parent exp08
QV_SCALE: float = 4000.0

# ── Surface forcing ────────────────────────────────────────────────────────────
# Disable surface fluxes - cloud already exists from parent
USE_SURFACE_FLUXES: bool = False
SHF: float = 0.0          # W/m² - turned off
LHF: float = 0.0          # W/m² - turned off
SHF_SIGMA: float = 0.0
SHF_CORR:  float = 500.0
RANDOM_SEED: int = 42

# ── SGS diffusion (same as parent initially) ───────────────────────────────────
CS:   float = 0.18        # Smagorinsky constant - same as parent
PR_T: float = 0.33        # turbulent Prandtl number

# ── Large-scale forcing (inherit from parent) ──────────────────────────────────
W_SUBS: float = 0.005     # m/s - same as parent
TAU_NUDGE_T: float = 3600.0  # s - nudge to parent profiles, not own mean
TAU_NUDGE_Q: float = 3600.0  # s

# ── Sponge layer (same physics) ────────────────────────────────────────────────
Z_SPONGE: float   = 2500.0  # m - adjusted for lower domain top
TAU_SPONGE: float = 60.0    # s

# ── Physical constants (unchanged) ─────────────────────────────────────────────
G:    float = 9.81
CP:   float = 1004.0
LV:   float = 2.5e6
RD:   float = 287.0
RV:   float = 461.5
EPS:  float = RD / RV
P00:  float = 1.0e5

# ── Relaxation boundary conditions ─────────────────────────────────────────────
USE_RELAXATION_BC: bool = True
N_RELAX: int = 4          # cells in relaxation zone (~200m at dx=50m)
TAU_RELAX_MIN: float = 5.0  # s - minimum relaxation timescale at boundary

# ── Nested-specific settings ───────────────────────────────────────────────────
RESTART_FROM: str | None = "nested_ic_50m_withwind.npz"  # Created by setup_nested.py
TERRAIN_TYPE: str = "flat"
TERRAIN_H_MAX: float = 0.0
TERRAIN_SIGMA: float = 0.0
TERRAIN_FILE: str | None = None

# ── Initial boundary-layer perturbations ───────────────────────────────────────
# Disable for nested - use parent interpolated fields
THETA_NOISE: float = 0.0
QV_NOISE:    float = 0.0
NOISE_CORR:  float = 0.0

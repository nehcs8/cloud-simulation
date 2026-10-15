"""Central configuration for the cloud simulation."""

# ── Grid ──────────────────────────────────────────────────────────────────────
NX: int   = 120          # cells in x  (100m resolution for baseline)
NY: int   = 120          # cells in y  (12×12 km domain)
NZ: int   = 80           # cells in z
DX: float = 100.0  # m
DY: float = 100.0  # m
Z_TOP: float = 5000.0    # m, domain top
# Vertical grid: "exp" = exponential stretching over NZ levels to Z_TOP
#                "uniform_stretched" = uniform DZ_UNIFORM up to Z_UNIFORM_TOP, then
#                geometric stretching (N_STRETCH levels) to Z_TOP; NZ is derived
GRID_TYPE: str = "exp"
DZ_UNIFORM: float = 40.0       # m
Z_UNIFORM_TOP: float = 3000.0  # m
N_STRETCH: int = 12            # levels in the stretched layer

# ── Time ──────────────────────────────────────────────────────────────────────
DT: float = 2.0          # s, maximum time step (adaptive dt never exceeds this)
CFL_MAX: float = 1.0     # max Courant number; checked on max(|u|,|v|,|w|) or their sum
CFL_SUM: bool = False    # True: limit (|u|/dx + |v|/dy + |w|/dz)·dt (safer for 3-D RK3)
T_END: float = 14400.0   # s (4 hours)
OUTPUT_EVERY: float = 60.0  # s
SNAPSHOT_EVERY: float = 60.0   # s, 3-D snapshot interval (independent of OUTPUT_EVERY)
PROFILE_EVERY: float = 600.0   # s, profile output interval (10 min)

# ── Case selection ─────────────────────────────────────────────────────────────
# "default": sounding below (THETA_S, Z_BL, QV_SURF, ...) + correlated BL noise
# "land_cu": cases/land_cu.py sounding + white noise in lowest 200 m
# "single_cu": cases/single_cu.py, one congestus from a hot spot or a warm bubble
CASE: str = "default"

# Warm bubble at t = 0 (applied by cases/single_cu.py):
# θ′ += BUBBLE_DTHETA·cos²(π/2·r), r = √((Δx² + Δy²)/BUBBLE_RX² + Δz²/BUBBLE_RZ²) < 1,
# and qv raised toward BUBBLE_RH·q_sat with the same weight. 0 = off.
BUBBLE_DTHETA: float = 0.0          # K
BUBBLE_RH:     float = 0.0          # target RH at the bubble centre (0 = keep qv)
BUBBLE_X0: float | None = None      # m, None = domain centre
BUBBLE_Y0: float | None = None      # m
BUBBLE_Z0:     float = 500.0        # m
BUBBLE_RX:     float = 1500.0       # m, horizontal radius
BUBBLE_RZ:     float = 500.0        # m, vertical radius

# ── Reference atmosphere ───────────────────────────────────────────────────────
THETA_S: float = 300.0   # K, surface potential temperature
Z_BL: float   = 1350.0   # m, well-mixed BL top (constant θ and qv up to this height)
GAMMA_BL: float = 0.0    # K/km, well-mixed BL (zero gradient)
GAMMA_FT: float = 6.0    # K/km, dθ/dz in free troposphere (above inversion)

# Inversion parameters (capping inversion above well-mixed BL)
Z_INVERSION: float = 1500.0  # m, inversion center height (1.2-1.5 km range)
INVERSION_STRENGTH: float = 3.0  # K, theta jump across inversion (~200m)
INVERSION_WIDTH: float = 200.0   # m, transition width

# Moisture profile (well-mixed BL with 14-15 g/kg, then 40-50% RH above)
QV_SURF: float = 0.0145      # kg/kg, surface moisture (14.5 g/kg, well-mixed in BL)
QV_SCALE: float = 1200.0     # m, moisture scale height (tuned for ~40-50% RH above inversion)

# ── Surface forcing ────────────────────────────────────────────────────────────
SHF: float = 150.0       # W/m², domain-mean sensible heat flux (moderate)
LHF: float = 300.0       # W/m², domain-mean latent heat flux (moisture-driven convection)
SHF_SIGMA: float = 0.0   # relative std of the FIXED SHF/LHF map (0 = homogeneous)
SHF_CORR:  float = 2000.0  # m, spatial correlation length of flux patterns
# Warm stripe: SHF + SHF_STRIPE_AMP over x in [SHF_STRIPE_X0, SHF_STRIPE_X0 + SHF_STRIPE_WIDTH],
# uniform in y (cos² edges of SHF_STRIPE_TAPER inside that width). Added on top of the map
# above; the domain mean then rises. LHF is unchanged. 0 = off.
SHF_STRIPE_AMP:   float = 0.0      # W/m²
SHF_STRIPE_X0:    float = 0.0      # m, upstream edge
SHF_STRIPE_WIDTH: float = 2000.0   # m
SHF_STRIPE_TAPER: float = 250.0    # m
# Hot spot: SHF + SHF_SPOT_AMP inside a disc of diameter SHF_SPOT_DIAM centred at
# (SHF_SPOT_X0, SHF_SPOT_Y0) (None = domain centre), cos² edge of SHF_SPOT_TAPER inside
# that diameter, distance measured periodically. Added on top of the map; LHF unchanged. 0 = off.
SHF_SPOT_AMP:   float = 0.0         # W/m²
SHF_SPOT_DIAM:  float = 3000.0      # m
SHF_SPOT_TAPER: float = 500.0       # m
SHF_SPOT_X0: float | None = None    # m
SHF_SPOT_Y0: float | None = None    # m

# ── Lateral boundaries in x ────────────────────────────────────────────────────
# "periodic": doubly periodic domain (default).
# "fringe":   inflow/outflow in x for a mean wind along +x (y stays periodic). The last
#             FRINGE_WIDTH of the domain is relaxed toward a copy of the strip just upstream
#             of RECYCLE_X, so what wraps into x = 0 is turbulent upstream air, not recirculated
#             air. [0, RECYCLE_X) is a homogeneous fetch (its own precursor); flux patterns
#             (SHF_SIGMA > 0) are confined to [RECYCLE_X + HETERO_BUFFER, Lx − FRINGE_WIDTH].
LATERAL_BC: str = "periodic"
FRINGE_WIDTH: float = 1600.0   # m, relaxation zone at the downstream end of x
FRINGE_TAU: float = 20.0       # s, relaxation time where the fringe is at full strength
RECYCLE_X: float = 4800.0      # m, recycling plane = length of the homogeneous fetch
HETERO_BUFFER: float = 1000.0  # m, homogeneous gap between recycling plane and flux patterns
HETERO_TAPER: float = 500.0    # m, cos² taper of the flux patterns at the test-region edges

# Surface flux ramp-up (avoid startup shock): linear from 0 to full over this time
FLUX_RAMP_TIME: float = 2700.0  # s (45 minutes)

# Optional evolving flux pattern (AR(1) in time, updated every FLUX_PATTERN_DT):
#   pattern ← a·pattern + sqrt(1−a²)·new_field,  a = exp(−FLUX_PATTERN_DT / FLUX_PATTERN_TAU)
#   SHF, LHF = mean · (1 + FLUX_PATTERN_AMP · pattern)   (pattern has unit std)
ENABLE_FLUX_PATTERN: bool = False
FLUX_PATTERN_TAU: float = 1800.0  # s
FLUX_PATTERN_AMP: float = 0.1
FLUX_PATTERN_DT:  float = 60.0    # s

# ── SGS diffusion (Smagorinsky-Lilly) ─────────────────────────────────────────
CS:   float = 0.18       # Smagorinsky constant
PR_T: float = 0.33       # turbulent Prandtl number  (K_H = K_M / PR_T)

# ── Performance ───────────────────────────────────────────────────────────────
# Numba kernels (kernels.py) for advection, SGS diffusion and K_M; False = numpy
# reference path in dynamics.py. Falls back to numpy if numba is not installed.
USE_NUMBA: bool = True

# ── Large-scale forcing ───────────────────────────────────────────────────────
# Subsidence w_ls(z): linear 0 → −W_SUBS at Z_BL, then constant above
# (SUBS_Z_ZERO = None) or linear back to 0 at SUBS_Z_ZERO (constant 0 above).
# SUBS_Z_MAX overrides Z_BL as the height of the maximum when set.
W_SUBS: float = 0.005
SUBS_Z_MAX: float | None = None
SUBS_Z_ZERO: float | None = None
# Subsidence advects: "reference" (θ0, qv0 gradients) or "mean" (current horizontal mean)
SUBS_GRADIENT: str = "reference"

# Radiative cooling (K/day, negative = cooling): uniform below RAD_Z_TOP,
# linear taper to 0 at RAD_Z_ZERO
RAD_COOLING: float = 0.0
RAD_Z_TOP: float = 1900.0
RAD_Z_ZERO: float = 2500.0

TAU_NUDGE_T: float = 3600.0  # s, relaxation timescale for domain-mean θ′ (DISABLED in BL)
TAU_NUDGE_Q: float = 5400.0  # s, relaxation timescale for domain-mean qv′ (DISABLED in BL)
NUDGE_Z_TOP: float = 1500.0  # m, only nudge ABOVE this height (disable in BL)

# ── Sponge layer ──────────────────────────────────────────────────────────────
Z_SPONGE: float   = 4000.0  # m, base of sponge
TAU_SPONGE: float = 60.0    # s, e-folding damping timescale

# ── Physical constants ─────────────────────────────────────────────────────────
G:    float = 9.81        # m/s²
CP:   float = 1004.0      # J/kg/K
LV:   float = 2.5e6       # J/kg
RD:   float = 287.0       # J/kg/K
RV:   float = 461.5       # J/kg/K
EPS:  float = RD / RV     # ≈ 0.622
P00:  float = 1.0e5       # Pa, reference pressure

# ── Initial boundary-layer perturbations (replaces single bubble) ─────────────
THETA_NOISE: float = 0.2    # K,      amplitude of random θ′ perturbations in BL
QV_NOISE:    float = 3e-4   # kg/kg,  amplitude of random qv perturbations in BL
NOISE_CORR:  float = 800.0  # m,      horizontal correlation length of noise
RANDOM_SEED: int   = 42

# ── Background wind profile ────────────────────────────────────────────────────
U_GEO: float = 2.0         # m/s, geostrophic wind (E-W component)
V_GEO: float = 0.0         # m/s, geostrophic wind (N-S component)
# Optional height-dependent u_g(z) = U_GEO_PROFILE(z) (callable on a z array);
# None = uniform U_GEO. Used for initial u, the Coriolis term and the sponge.
U_GEO_PROFILE = None
# Wind initialized uniform in vertical, will adjust via turbulence/shear

# ── Rotation fix experiments ───────────────────────────────────────────────────
ENABLE_CORIOLIS: bool = True         # Add Coriolis force (f-plane approximation) - PROVEN EFFECTIVE
CORIOLIS_F: float = 1.0e-4           # s^-1, Coriolis parameter (mid-latitudes ~45°N)
ENABLE_WENO_MOMENTUM: bool = False   # Use WENO-3 for u,v advection (causes instability - keep OFF)

# ── Entrainment parameterization ──────────────────────────────────────────────
# DISABLED: LES should resolve mixing, not parameterize it (double-counting issue)
ENABLE_ENTRAINMENT: bool = False
ENTRAINMENT_RATE: float = 0.001  # s^-1, weak
ENABLE_DYNAMIC_ENTRAINMENT: bool = False
DYNAMIC_ENTR_W_SCALE: float = 5.0    # m/s, reference updraft velocity for scaling

# ── Cold pool (evaporative cooling from rain) ─────────────────────────────────
ENABLE_COLD_POOL_HACK: bool = False  # Instant rain evaporation in sub-cloud layer (now redundant - rain sedimentation handles this properly)

# ── Cloud-radiative feedback ───────────────────────────────────────────────────
ENABLE_CLOUD_SHADING: bool = False  # reduce surface fluxes under clouds (Beer's law on LWP)
CLOUD_EXTINCTION: float = 200.0  # m²/kg, strong
MIN_TRANSMITTANCE: float = 0.1

# ── Microphysics parameters ────────────────────────────────────────────────────
AUTOCONV_RATE: float = 1e-3      # s^-1, autoconversion rate (K1 in Kessler)
AUTOCONV_THRESHOLD: float = 0.5e-3  # kg/kg, cloud water threshold for autoconversion (QC0)

# ── Terrain (Phase 2) ──────────────────────────────────────────────────────────
TERRAIN_TYPE: str = "flat"       # "flat", "gaussian_hill", "sine_ridge", "random", "dem"
TERRAIN_H_MAX: float = 300.0     # m, max height for synthetic terrain
TERRAIN_SIGMA: float = 2000.0    # m, Gaussian hill width or correlation length
TERRAIN_FILE: str | None = None  # Path to DEM file (if TERRAIN_TYPE="dem")

# ── Initial conditions ─────────────────────────────────────────────────────────
RESTART_FROM: str | None = None
                                  # If None, use warm-start from noise perturbations

# ── Run infrastructure ─────────────────────────────────────────────────────────
CASE_NAME: str = "default"       # Case name for run folder (runs/YYYYMMDD_HHMM_<case>/)
ENABLE_NAN_GUARD: bool = True    # Save crash state and halt on NaN/Inf detection

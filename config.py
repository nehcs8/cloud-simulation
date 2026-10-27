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
# that diameter, distance measured periodically. Added on top of the map. 0 = off.
# SHF_SPOT_LHF_AMP adds a latent-heat excess with the same footprint (moist hot spot).
SHF_SPOT_AMP:   float = 0.0         # W/m²
SHF_SPOT_LHF_AMP: float = 0.0       # W/m²
SHF_SPOT_DIAM:  float = 3000.0      # m
SHF_SPOT_TAPER: float = 500.0       # m
# "disc" (flat top, cos² edge) or "gaussian": exp(−r²/2σ²) peaked at the centre, σ = DIAM/(2√2)
# so the area integral equals that of an untapered disc of diameter SHF_SPOT_DIAM
SHF_SPOT_SHAPE: str = "disc"
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
# Pressure-solver boundary type per direction (stage 1 of docs/PLAN_open_boundaries.md):
# "periodic": FFT, wrap-around (default).
# "open":     the normal velocity on the two boundary faces (u[0], u[nx] / v[:, 0], v[:, ny])
#             is taken as given and kept by the projection (∂φ/∂n = 0, cosine transform);
#             the net boundary mass flux is first balanced to zero by a uniform correction.
#             Ghost cells, radiation and inflow relaxation: open_bc.py.
#             LATERAL_BC = "fringe" needs BC_X = "periodic".
BC_X: str = "periodic"
BC_Y: str = "periodic"
# Open-boundary parameters (open_bc.py): radiation phase speed for the boundary-face normal
# velocity (Klemp–Wilhelmson; also limits dt: c*·dt/Δ ≤ CFL_MAX), its weak relaxation toward
# u_g / V_GEO, and the inflow relaxation zone (cells, e-folding time at the boundary cell).
OPEN_C_STAR: float = 10.0        # m/s (10 reflected least in the dry-bubble test, 2026-10-18)
OPEN_TAU_NORMAL: float = 1800.0  # s (0 = off)
OPEN_RELAX_WIDTH: int = 4        # cells
OPEN_RELAX_TAU: float = 60.0     # s
# Inflow reference state of θ_l′, q_t (ghost cells + relaxation zone; winds stay u_g / V_GEO):
# "sounding": the initial sounding for the whole run (before 2026-10-27; the inflow BL is never heated).
# "mean":     horizontal mean over clear columns (no hydrometeor > OPEN_REF_QMIN anywhere, lowest-level θ′ not
#             more than OPEN_REF_COLD below the median: no cold pools), filtered with time scale OPEN_REF_TAU.
#             The anomaly from the sounding is kept in full below OPEN_REF_Z1 and tapered linearly to 0 at
#             OPEN_REF_Z2, so the heated BL flows in but convective subsidence warming aloft is not recycled.
OPEN_INFLOW_REF: str = "sounding"
OPEN_REF_TAU: float = 600.0      # s
OPEN_REF_Z1: float = 2000.0      # m
OPEN_REF_Z2: float = 3000.0      # m
OPEN_REF_QMIN: float = 1e-6      # kg/kg
OPEN_REF_COLD: float = 0.3       # K
# Cell perturbation method (Muñoz-Esparza et al. 2014, 2015) at inflow faces, so the inflow is turbulent: every
# OPEN_CPM_PERIOD s (0 = one cell length / mean wind speed below OPEN_CPM_ZTOP) θ in cells of OPEN_CPM_CELL ×
# OPEN_CPM_CELL columns × 1 level gets a uniform random offset in [−OPEN_CPM_AMP, OPEN_CPM_AMP], in the first
# OPEN_CPM_BANDS rows of cells along each inflow face, below OPEN_CPM_ZTOP. Only θ: no water or momentum added.
OPEN_CPM_AMP: float = 0.0        # K (0 = off)
OPEN_CPM_CELL: int = 8           # grid columns per cell side
OPEN_CPM_BANDS: int = 3          # rows of cells from the boundary
OPEN_CPM_ZTOP: float = 1000.0    # m
OPEN_CPM_PERIOD: float = 0.0     # s
OPEN_CPM_SEED: int = 12345
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
# RAIN_SCHEME: "sb2001" = Seifert & Beheng (2001) autoconversion + accretion, which depend on the cloud
# droplet number CLOUD_NC (continental ~200–1000 cm⁻³ rains late, maritime ~50–100 cm⁻³ early);
# "kessler" = the threshold scheme used before 2026-10-19 (rain within ~1 min of qc > AUTOCONV_THRESHOLD).
RAIN_SCHEME: str = "sb2001"
CLOUD_NC: float = 300.0e6        # m⁻³, cloud droplet number (sb2001)
AUTOCONV_RATE: float = 1e-3      # s^-1, autoconversion rate (K1 in Kessler)
AUTOCONV_THRESHOLD: float = 0.5e-3  # kg/kg, cloud water threshold for autoconversion (QC0)
# Rain evaporation: "kw78" = Klemp & Wilhelmson (1978) as in CM1/WRF Kessler; "legacy" = the
# K_EVP·deficit·qr^0.525 form used before 2026-10-19 (70–100× weaker: almost no cold pools)
RAIN_EVAP: str = "kw78"
# Ice (2026-10-22, step 1): "none" = warm only; "mixed" = cloud condensate qc split into liquid/ice by
# temperature (liquid fraction 1 at ≥ T_ICE_0, 0 at ≤ T_ICE_00, linear between), saturation over that mix
# (Buck water/ice), latent heat L_v + (1 − α)·L_f, so θ_l is the ice-liquid potential temperature. Rain stays
# liquid: the ice part of cloud → rain conversion takes L_f back (no precipitating ice yet).
ICE_SCHEME: str = "none"
LF: float = 3.34e5               # J/kg, latent heat of fusion
T_ICE_0: float = 273.15          # K, all condensate liquid at and above
T_ICE_00: float = 253.15         # K, all condensate ice at and below
# "graupel" (step 2) = "mixed" + prognostic precipitating ice q_g (one graupel/snow category, Marshall–Palmer,
# V = GR_A·D^GR_B): cloud-ice autoconversion (Lin 1983, rate 1e-3·exp(0.025 T_c) s⁻¹ above ICE_AUTO_QI0),
# collection of cloud ice (GR_E_ICE), cloud liquid (riming) and rain (T < 0 °C) by graupel, Bigg rain freezing,
# melting (T > 0 °C), sublimation; warm rain acts on the liquid part of q_c only.
RAIN_MOMENTS: int = 1            # 1 = rain mass only (fixed Marshall–Palmer N0); 2 = + prognostic drop number s.nr
# (2026-10-26, Seifert & Beheng 2006: autoconversion drops of mass x*, self-collection/breakup toward D_eq 1.1 mm,
# evaporation from the exponential size distribution of (q_r, N_r), two-moment sedimentation with Atlas V(D))
RAIN_KRR: float = 4.33           # m³ kg⁻¹ s⁻¹, rain self-collection (SB2006)
RAIN_DEQ: float = 1.1e-3         # m, breakup equilibrium diameter
RAIN_XMIN: float = 2.6e-10       # kg, mean drop mass bounds (D ≈ 0.08–1.8 mm)
RAIN_XMAX: float = 3.0e-6
GR_N0: float = 4.0e6            # m⁻⁴, intercept (Rutledge & Hobbs 1984 graupel)
GR_RHO: float = 400.0            # kg/m³, particle density
GR_A: float = 19.3               # m^(1−b)/s, fall speed V = a D^b (Locatelli & Hobbs lump graupel)
GR_B: float = 0.37
GR_E_ICE: float = 0.1            # collection efficiency graupel–cloud ice
ICE_AUTO_QI0: float = 1.0e-4     # kg/kg, cloud-ice threshold for autoconversion to graupel

# ── Terrain (Phase 2) ──────────────────────────────────────────────────────────
TERRAIN_TYPE: str = "flat"       # "flat", "gaussian_hill", "sine_ridge", "agnesi_ridge", "random", "dem"
TERRAIN_H_MAX: float = 300.0     # m, max height for synthetic terrain
TERRAIN_SIGMA: float = 2000.0    # m, Gaussian hill σ, Agnesi half-width a, or correlation length
TERRAIN_FILE: str | None = None  # Path to DEM file (if TERRAIN_TYPE="dem")
# Partial bottom cells (fluid fraction hc of the cell containing the ground) instead of a staircase
# of whole cells; fractions below TERRAIN_HFAC_MIN are rounded to 0 or HFAC_MIN (stability).
TERRAIN_PARTIAL_CELLS: bool = False
TERRAIN_HFAC_MIN: float = 0.3
# Pressure solve with terrain (pressure._project_terrain): preconditioned CG on the fluid cells
TERRAIN_PCG_TOL: float = 1e-8    # relative residual (∇·(ρ₀u) ~1e-7 of its pre-projection value)
TERRAIN_PCG_MAXIT: int = 200

# ── Initial conditions ─────────────────────────────────────────────────────────
RESTART_FROM: str | None = None
                                  # If None, use warm-start from noise perturbations

# ── Run infrastructure ─────────────────────────────────────────────────────────
CASE_NAME: str = "default"       # Case name for run folder (runs/YYYYMMDD_HHMM_<case>/)
ENABLE_NAN_GUARD: bool = True    # Save crash state and halt on NaN/Inf detection

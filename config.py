"""Central configuration for the cloud simulation."""

# ── Grid ──────────────────────────────────────────────────────────────────────
NX: int   = 240          # cells in x
NY: int   = 240          # cells in y
NZ: int   = 80           # cells in z
DX: float = 50.0         # m
DY: float = 50.0         # m
Z_TOP: float = 5000.0    # m, domain top

# ── Time ──────────────────────────────────────────────────────────────────────
DT: float = 2.0          # s, time step (kept small for stability)
T_END: float = 7200.0    # s, 2 hours
OUTPUT_EVERY: float = 60.0  # s

# ── Reference atmosphere ───────────────────────────────────────────────────────
THETA_S: float = 300.0   # K, surface potential temperature
Z_BL: float   = 1500.0   # m, nominal BL top
GAMMA_BL: float = 2.0    # K/km, dθ/dz in BL  (slightly stable)
GAMMA_FT: float = 6.0    # K/km, dθ/dz in free troposphere
QV_SURF: float  = 0.011  # kg/kg, surface water-vapour mixing ratio
QV_SCALE: float = 2500.0 # m, exponential decay scale height for qv

# ── Surface forcing ────────────────────────────────────────────────────────────
SHF: float = 150.0       # W/m², domain-mean sensible heat flux (summer afternoon)
LHF: float = 5.0         # W/m², low LHF — dry continental conditions
SHF_SIGMA: float = 0.4   # relative std of spatial SHF variation (±40% of mean)
SHF_CORR:  float = 2000.0  # m, spatial correlation length of SHF pattern

# ── SGS diffusion (Smagorinsky-Lilly) ─────────────────────────────────────────
CS:   float = 0.18       # Smagorinsky constant
PR_T: float = 0.33       # turbulent Prandtl number  (K_H = K_M / PR_T)

# ── Large-scale forcing ───────────────────────────────────────────────────────
W_SUBS: float    = 0.005   # m/s, subsidence at and above BL top
TAU_NUDGE_T: float = 3600.0  # s, relaxation timescale for domain-mean θ′
TAU_NUDGE_Q: float = 1800.0  # s, relaxation timescale for domain-mean qv

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

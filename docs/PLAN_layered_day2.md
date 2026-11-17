# Plan: layered_day, second round — turbulent, patchy multi-layer clouds (2026-11-04)

## What went wrong in `runs/20261101_1311_layered_day_trough_360min_first`

Re-rendered with height axis and translucent ice: `cloud_gifs/20261101_layered_day_trough_0-360min_zaxis_5min.mp4` (5-min frames, snapshots thinned).

| Layer | Set up as | What happened (cf > 5 %) |
|---|---|---|
| Cu | LCL 1.35 km, LFC 2.1 km | weak until ~5 h (cover ≤ 7 %), then deep towers 1.6–8.4 km, w_max up to 29 m/s, tops 10.3 km |
| A (altocumulus) | 3.8–4.7 km, RH 0.92 | from 3.5 h, 4.3 → 6.0 km, cover 13–42 %, w std 0.2–0.7 m/s |
| C (cirrus) | 8.5–10 km, RH 0.92 over ice | from 4 h, **9.1–10.2 km, 100 % cover until the end** |

The cirrus deck (snapshots at 270 / 330 / 360 min):
- ice water path 7 / 11 / 14 g/m² → optical depth ≈ 0.4–0.7 (r_eff 30 µm): **physically thin, see-through cirrus**,
  but the renderer's q_c = 0.01 g/kg iso-surface draws it as an opaque white slab;
- horizontally uniform (CV of q_c at 9.5 km 0.08–0.17) and laminar (w std at 9.5 km 0.05–0.18 m/s);
- **cloud ice never leaves**: q_c does not sediment, and ice → graupel needs q_i > `ICE_AUTO_QI0` = 0.1 g/kg while the
  deck holds 0.025–0.044 g/kg. Real cirrus ice falls at 0.3–1 m/s (fall streaks, ragged base, self-limiting);
- saturation adjustment at RH_ice = 100 % (no ice supersaturation), and the trough lifts a uniform 1.5 km deep RH-0.92
  layer → the whole layer saturates at once. (1 km thickness itself is normal for cirrus; the uniformity is not.)

## Model changes (implemented 2026-11-04, all off by default → earlier runs unchanged; `tests/test_cloud_rad_ice.py`)

1. **Cloud LW radiation** `"RAD_SCHEME='cloud_lw'"` (`forcing.apply_cloud_radiation`, after `apply_radiative_cooling`, so the
   precursor gets it too). DYCOMS-II-type grey fluxes on the cloud optical path, κ 85 m²/kg liquid / 50 ice (α(T) split):
   cell loses `RAD_F0` (70 W/m²) · e^(−Q above) · (1 − e^(−b)), gains F1 · e^(−Q below) · (1 − e^(−b)) with
   F1 = `RAD_F1_EPS` (0.35) · σ(T_sfc⁴ − T⁴) — ~24 W/m² for BL Cu, ~49 for Ac at 4 km, ~94 for cirrus at 9.5 km.
   Test slabs: Ac 0.3 g/kg at 4–4.5 km top −3.1 K/h, base +2.3 K/h; Ci 0.03 g/kg at 9.1–10.1 km top −2.2 K/day, base
   +10.2 K/day (thin cirrus warms on the whole, as observed). No SW, no clear-sky cooling (RAD_COOLING stays 0).
2. **Falling cloud ice** `ICE_SEDIMENT=True` (`microphysics._sediment_cloud_ice`, in `sedimentation`): ice part (1 − α) q_c
   falls with the implicit sweep at V = `ICE_VT_A` (ρq)^`ICE_VT_B` √(ρ_sfc/ρ) ≈ Heymsfield–Donner: 0.52 m/s for 0.03 g/kg at
   9.5 km (test), mass conserved, numba = numpy. θ kept, so ice falling into sub-saturated air sublimates at the next
   adjustment (fall streaks). Ice reaching the ground counts as graupel_sfc. Known simplification: ice falling into a
   warmer cell is relabelled by that cell's α(T) without latent heat of fusion (small: cirrus ice sublimates long before).
3. **Variants** `layered_day:turb` / `turb_jet` (`cases/layered_day.py`): layer C 8.8–9.6 km at RH 0.88 (`RH_TURB`);
   zero-mean smooth RH patches σ 0.04 (`LD_RH_PATCH`), ~2 km across (`LD_PATCH_SCALE`), independent below / above 7 km,
   RH ≤ 0.98, in the moist layers; ±0.1 K θ noise there (`LD_THETA_NOISE_UP`); SHF 200 / LHF 320; `LD_W_MAX` 3 cm/s.
   `turb_jet` adds a jet +8 m/s, cos² over 4.6–5.6 km (`LD_JET_*`), shear ~0.025 s⁻¹ on its flanks (Ri ≈ 0.2 in the
   dry slot above layer A) for Kelvin–Helmholtz billows. Precursor gets the same perturbations.
4. **Renderer**: done 2026-11-04 (height axis, per-layer caption with phase and optical depth, ice as extinction volume).

## The 4-day campaign — `scripts/campaign_layered_day2.py`

One command, then it runs by itself (systemd user unit `ld2-campaign`, survives closing VS Code; resumes after a stop or
reboot with `start` again):

    ./venv/bin/python3 scripts/campaign_layered_day2.py start
    ./venv/bin/python3 scripts/campaign_layered_day2.py status

Every run: open BCs + precursor, graupel + 2-moment rain, **1-min snapshots → mp4 from all of them** (`finish_run.py`),
then thinning to 5 min. mp4s: `cloud_gifs/YYYYMMDD_ld2_<id>_<desc>_0-<min>min.mp4`; logs: `outputs/logs/campaign_ld2/`.
Stages run in order (radiation → falling ice → main → resolution); within a stage two runs side by side (2 threads each).

| id | stage | variant | grid | sim | LW | ice fall | question |
|---|---|---|---|---|---|---|---|
| R0 | 1 | trough | 64² / 150 m | 8 h | – | – | control on the small domain (vs the 128² first run) |
| R1 | 1 | trough | 64² | 8 h | ✓ | – | radiation alone on the old sounding: does the cirrus slab break up? |
| R2 | 1 | turb | 64² | 8 h | – | – | effect of the patchy / more turbulent setup alone |
| R3 | 1 | turb | 64² | 8 h | ✓ | – | setup + radiation |
| R4 | 1 | turb | 64² | 8 h | ✓ strong (F0 100, ε 0.5) | – | sensitivity to the radiation strength |
| R5 | 1 | turb_jet | 64² | 8 h | ✓ | – | shear layer: KH billows at the top of layer A? |
| I1 | 2 | trough | 64² | 8 h | ✓ | ✓ | old sounding, both processes (vs R1) |
| I2 | 2 | turb | 64² | 8 h | – | ✓ | ice fall alone (vs R2) |
| I3 | 2 | turb | 64² | 8 h | ✓ | ✓ | both (vs R3, I2) |
| I4 | 2 | turb_jet | 64² | 8 h | ✓ | ✓ | both + shear (vs R5) |
| M1 | 3 | turb | 128² / 150 m | 8 h | ✓ | ✓ | main run |
| M2 | 3 | turb_jet | 128² | 8 h | ✓ | ✓ | main run with shear |
| M3 | 3 | trough | 128² | 6 h | ✓ | ✓ | the first run (20261101 trough) with the new physics only |
| H1 | 4 | turb | 128² / 100 m | 8 h | ✓ | ✓ | resolution (cells of Ac / Ci are ~1 km) |

64² = 9.6 km with a 32² precursor (pilots); 128² with a 64² precursor as before. 8 h because in the 4-h check
(`runs/20261104_1116_layered_day_turb_jet_240min_check64`, 64², LW + ice fall, stable, budget residual 0, 15.3 wall-min per
simulated hour) only Cu had formed by 4 h (cover ≤ 10 %, tops ≤ 2.9 km): with 3 cm/s the layers saturate later. Estimated wall
time: stage 1 ≈ 10.5 h, stage 2 ≈ 7 h, stage 3 ≈ 16 h (M1 ∥ M2) + 8 h (M3), stage 4 ≈ 14 h → ~56 h. Deadline 96 h after the first start: a job whose estimate (× the measured speed factor) would end
later is skipped, not started. Disk: peak ~25 GB per 128² run before thinning (free 2026-11-04: ~100 GB); a job waits if
free space is short. Failed runs are logged and skipped.

Interpretation after everything has finished (not run by run). Success criteria: cirrus cover < 100 % with visible fall
streaks below its base, w std aloft > 0.3 m/s, altocumulus as cells (cover 30–70 %, not one sheet), Cu cover 10–20 % from
~2 h; three separable layers in the caption.

## Results (campaign 2026-11-04 12:18 → 11-06 11:20, all 14 runs done, ~77 h of CPU-wall in 47 h)

All runs stable, water-budget residual 0, mp4 with one frame per 1-min snapshot. Statistics: `scripts/analyze_layered_day.py`
(5-min snapshots; bands low < 3.3 km, mid 3.3–7.5 km, high ≥ 7.5 km; τ with κ 150 liquid / 55 ice), outputs
`outputs/logs/analyze_ld2_{300-480,trough_240-360,series_turb}.txt`.

**Two regimes in every run.** Stratiform layers first, then deep convection: w_max 15–44 m/s, tops 10–11.8 km, rain from
~6 h (trough runs: convection from ~5.5–6 h; turb runs, with SHF 200 / LHF 320: from ~4.5–5.5 h). From ~6.5 h anvils dominate
the mid and high bands and reach 85–100 % cover with τ 5–50 by 7.5–8 h, so the 5–8 h means are anvil statistics. The clean
window for the layers is 4–6 h (trough) / 3.5–5 h (turb).

**Falling ice (clear, robust).** Trough runs, 4–6 h, cirrus band:

| run | physics | cover | visible (τ>0.1) | median τ | depth | ice path |
|---|---|---|---|---|---|---|
| first (20261101, 128²) | – | 96 % | 100 % | 0.52 | 1.1 km | 11.3 g/m² |
| R0 (64²) | – | 100 % | 100 % | 0.61 | 1.2 km | 11.6 |
| R1 (64²) | LW | 100 % | 100 % | 0.61 | 1.2 km | 16.1 |
| I1 (64²) | LW + ice fall | 87 % | 97 % | 0.32 | 0.9 km | 13.6 |
| M3 (128²) | LW + ice fall | 64 % | 85 % | 0.27 | 0.9 km | 7.5 |

Ice fall halves the cirrus optical depth, thins the layer and breaks the deck into wind-parallel streaks (M3 frame at 300 min);
visible cover > cloud-mask cover because fallen ice spreads as a thin veil below the q_c 0.01 g/kg mask. In the turb pilots
it also cuts the upper-band ice path at 5–8 h in 2 of 3 pairs (I2 vs R2 67 vs 176 g/m², I4 vs R5 38 vs 143; I3 vs R3 92 vs 100).

**Cloud LW radiation (small).** Cirrus: no measurable effect in 2 h (R1 = R0: τ 0.61, w′ at 9.3 km 0.11 m/s in both).
Altocumulus: w′ at 4.5 km 0.32 → 0.39 m/s, τ 4.8 → 5.2. In the turb pilots the LW differences (R3 vs R2, strong R4) are not
monotonic in the forcing strength → within the run-to-run variability of single 64² realisations.

**Turbulence aloft not reached.** Before convection, w′ at 9.3 km is 0.04–0.07 m/s in every run (criterion > 0.3): the
cirrus stays laminar; its texture comes from ice fall + shear, not from overturning. Layer A gets w′ 0.3–0.7 m/s only once
saturated (turb: patchy Ac 20–45 % cover, τ 1–2 at 4–5 h — the intended look).

**Jet (`turb_jet`) did not test Kelvin–Helmholtz.** Ri_min on the jet flanks at 3 h is 0.8 (not ~0.2 as estimated): no
billows, w′ at 4.6–5.8 km identical with and without jet (0.09–0.12 m/s). In 128² (M2 vs M1) the shear organises the deep
convection: w′ mid 1.38 vs 0.98 m/s, rain 0.27 vs 0.14 mm/h (5–8 h), tops to 11.8 km.

**turb setup.** Patchy RH gives patchy Ac early (from 20 min where patches reach RH 0.98, cover 20–45 % at 4–5 h), but the
in-situ cirrus layer (RH 0.88, 3 cm/s) hardly saturates before the towers reach it; high cloud appears together with the
convective tops (5–6 h). Only H1 (100 m) shows a thin in-situ cirrus at 6 h (77 %, τ 0.35). The stronger surface fluxes
bring deep convection ~1 h earlier.

**Resolution (H1, 100 m, 12.8 km vs M1, 150 m, 19.2 km).** Cu cover the same (9 %), w_max 24 vs 34 m/s, upper ice path 46 vs
98 g/m² at 5–8 h — but the domains differ in size, so not conclusive.

**Domain size.** 64² vs 128² with the same physics (I1 vs M3): cirrus cover 87 vs 64 %, mid cover 49 vs 40 % — 64² pilots
tend to more cover; use them for signs, not amounts.

**Caveat found 2026-11-07 — the mid/high clouds are tiled copies of the precursor.** The main domain sits on the periodic tiling
of the precursor (64² = 9.6 km under the 128² runs, 32² = 4.8 km under the 64² pilots), and winds of 10–20 m/s aloft cross the
domain in 15–30 min while the layers form over hours, so nearly all mid/high cloud in the main domain was made in the precursor
and enters through the west/south faces twice per edge. Correlation of the cloud path with itself shifted by one tile in y
(4–7 h): mid 0.7–0.98, high 0.8–0.99 (I1, R3, H1, M1 at 360 min); low cloud mostly 0.0–0.6 (formed locally by surface heating);
w at 1 km in the westernmost quarter 0.7–0.93. The mp4s look periodic for that reason. Physics signs (ice fall halves τ) hold —
the precursor has the same physics — but cover/patchiness numbers of the mid/high layers are those of a 9.6 km (pilots: 4.8 km)
periodic box. Fix before the next campaign: precursor as large as the main domain (no repetition) and/or BL-only precursor.

### Next steps
1. Make ICE_SEDIMENT the layered_day default (clear improvement; keep LW on, it is physical, just slow).
2. For multilayer stratiform studies suppress or delay deep convection (back to SHF 150 / LHF 250, or weaker trough at low
   levels); analyse the anvil regime separately.
3. Cirrus turbulence / patchy formation: ice supersaturation (nucleation threshold RH_i ~ 1.2–1.4) instead of adjustment at
   RH_i = 1, and longer runs for LW to act.
4. Jet with Ri < 0.25 (ΔU ~15 m/s over 500 m, or thinner) on 64² if KH billows are wanted.
5. LW/turb sensitivities need 2–3 seeds (`RANDOM_SEED`) per setup; single 64² runs differ by ±30 %.

## Next: full-size precursor (ld3, planned 2026-11-07)

Precursor 128² under the 128² domain (no tiled inflow). Benchmark (10 min, side by side, 2 threads): 6.59 vs 4.60 s/step with
the 64² precursor → ×1.43, i.e. ~143 wall-min per simulated hour paired (M1 was 100), ~87 alone. 12 h of wall time → two runs
side by side, 4.5 h each (~10.7 h + rendering). Launcher `outputs/logs/launch/ld3_fullpre.sh`.

| id | repeats | variant | physics | compare at | question |
|---|---|---|---|---|---|
| P1 | M3 | trough | LW + ice fall | 3.5–4.5 h (cirrus from ~3.5–4 h, Ac from ~3.4 h, before convection) | cirrus / Ac cover, τ, streaks without tiling |
| P2 | M1 | turb | LW + ice fall | 3.5–4.5 h (patchy Ac, before convection at ~4.5–5.5 h) | is the patchy Ac real or inherited from the 9.6 km tile? |

Check: tile-shift correlation (should drop to ~0), then `scripts/analyze_layered_day.py --t0 210 --t1 270` vs M3 / M1.

### Results (analysed 2026-11-17)
Runs `runs/20261110_1114_layered_day_{trough,turb}_270min_ld3P{1,2}`, mp4 `cloud_gifs/20261110_ld3_P{1,2}_*_fullpre_*.mp4`,
stats `outputs/logs/analyze_ld3_fullpre.txt`. Cost 156–158 wall-min per simulated hour side by side (11.8 h incl. rendering).
- Tiling gone: correlation of the field with itself shifted by 64 cells (old tile) — mid-level condensate path and w at 4.5 km
  0.5–0.87 in M3/M1 → −0.03…0.12 in P1/P2. The cirrus path keeps 0.27–0.53 at both 32- and 64-cell shifts: large bands of
  domain size, not repetition. Altocumulus in P1 = irregular wind-parallel streaks instead of M3's identical parallel rolls.
- Less mid/high cloud than the tiled runs (210–270 min): P1 mid 21 % (M3 41 %), high 4 % (31 %), high cover 22 % at 270 min vs 84 %.
  Not caused by the precursor: the precursor means (open_ref) of P1 and M3 are identical to 0.001 in RH. The main domain of P1
  develops a domain-mean subsidence of 2–5 cm/s at 3–10 km from 185 min (≈ the imposed 4 cm/s ascent), warming it +0.13 K over its
  precursor and lowering RH at 4.3 / 9.2 km by ~2 %, which delays both layers. The same mode exists in the old runs (M1 −6.5 cm/s,
  +0.17 K at 240 min; M3 a ±2–3 cm/s, ~50-min oscillation; P2 −3 cm/s at 175 min): it is a domain-mean vertical-motion mode of the
  open domain (the face-mean normal velocity is set only by radiation, nothing ties it to the precursor, whose w̄ = 0), whose phase
  differs run to run. With layers that saturate at RH ≈ 1 it decides when the cloud appears.
- P2 (turb): no altocumulus layer — Cu field 1.6–4.8 km at 4 h, congestus to 8.8 km at 4.5 h; the "mid" cloud of M1/P2 is convective.
- Next: tie the face-mean normal velocity to the precursor (relax the face mean toward u_g / precursor mean per level, or remove the
  domain-mean w each step) and repeat P1 for 4.5 h.

# SAGE26 Parameter Reference

This document is the canonical reference for every parameter accepted by SAGE26
parameter files (`input/*.par`). Parameter files are parsed by
[`src/core_read_parameter_file.c`](../src/core_read_parameter_file.c).

**Syntax:** `ParameterName  value  % optional comment`

Lines beginning with `%` are comments. Required parameters must be present;
optional parameters take the listed default if omitted.

---

## Output

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `FileNameGalaxies` | string | yes | — | Base name for output files (e.g. `model` → `model_0.hdf5`). |
| `OutputDir` | string | yes | — | Directory for galaxy output. Created if absent. |
| `OutputFormat` | string | no | `sage_hdf5` | `sage_hdf5`, `sage_binary`, or `lhalo_binary_output`. The last converts any supported input tree format to lhalo-binary and writes no galaxy catalogue. |
| `NumOutputs` | int | no | `-1` | Number of snapshot outputs; `-1` = all snapshots. |
| `SaveFullSFH` | 0/1 | no | `0` | Store per-snapshot SFR history arrays (`SFHMassDisk`, `SFHMassBulge`). |

---

## Simulation

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `TreeType` | string | yes | — | Merger tree format: `lhalo_binary`, `mimic_lhalo_binary` (L-Halo header with Mimic's 88-byte record), `lhalo_hdf5`, `consistent_trees_ascii`, `consistent_trees_hdf5`, `genesis_hdf5`, `gadget4_hdf5`. |
| `TreeName` | string | yes | — | Tree file basename (files are named `TreeName.N`). |
| `SimulationDir` | string | yes | — | Directory containing tree files. |
| `FileWithSnapList` | string | yes | — | File listing snapshot scale factors, one per line. |
| `FirstFile` | int | yes | — | First tree file index to process. |
| `LastFile` | int | yes | — | Last tree file index to process (inclusive). |
| `NumSimulationTreeFiles` | int | yes | — | Total number of tree files (may differ from FirstFile–LastFile range). |
| `LastSnapshotNr` | int | yes | — | Index of the final snapshot in the tree files. |
| `Omega` | double | yes | — | Matter density parameter Ω_m. |
| `OmegaLambda` | double | yes | — | Dark energy density parameter Ω_Λ. |
| `BaryonFrac` | double | yes | — | Universal baryon fraction f_b = Ω_b / Ω_m. |
| `Hubble_h` | double | yes | — | Dimensionless Hubble parameter h (H₀ = 100 h km/s/Mpc). |
| `PartMass` | double | yes | — | N-body particle mass in 10¹⁰ M_sun/h. |
| `BoxSize` | double | yes | — | Simulation box side length in Mpc/h. |

---

## Units

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `UnitLength_in_cm` | double | yes | — | 1 internal length unit in cm. Typically `3.08568e+24` (= Mpc/h). |
| `UnitMass_in_g` | double | yes | — | 1 internal mass unit in g. Typically `1.989e+43` (= 10¹⁰ M_sun). |
| `UnitVelocity_in_cm_per_s` | double | yes | — | 1 internal velocity unit in cm/s. Typically `100000` (= km/s). |

---

## Physics switches

| Parameter | Type | No | Default | Values and meaning |
|-----------|------|----|---------|-------------------|
| `SFprescription` | int | no | `1` | Star formation prescription: 0=Croton+06; 1=Blitz & Rosolowsky 06 H₂; 2=Somerville+25 SFR; 3=Somerville+25 SFR+H₂; 4=Krumholz & Dekel 12; 5=KMT 09; 6=Krumholz 13; 7=Gnedin & Draine 14. |
| `AGNrecipeOn` | int | no | `2` | AGN feedback: 0=off; 1=empirical; 2=Bondi-Hoyle; 3=cold cloud accretion. |
| `SupernovaRecipeOn` | 0/1 | no | `1` | SN feedback: 0=off; 1=Croton+16 reheating/ejection. |
| `ReionizationOn` | 0/1 | no | `1` | Reionization suppression of infall: 0=off; 1=Kravtsov+04 analytic fit. |
| `DiskInstabilityOn` | 0/1 | no | `1` | Disk instability: 0=off; 1=Toomre criterion drives bulge and BH growth. |
| `CGMrecipeOn` | 0/1 | no | `1` | Two-regime CGM model: 0=off (classical C16 cooling only, including its rapid cold-accretion branch at `r_cool > R_vir`); 1=on. |
| `KarpovModeOn` | 0/1 | no | `0` | Metallicity of SN-reheated and ejected gas: 0=the full Karpov+23 recipe; 1=a low-metallicity floor at `Z/Z_sun = 0.01`. Only acts when the reheating/ejection path computes a metallicity for the outflow. |
| `FIREmodeOn` | 0/1 | no | `1` | FIRE stellar feedback: 0=off; 1=on. |
| `EnhancedStarFormationOn` | int | no | `1` | Feedback-free burst (FFB) galaxies: 0=off; 1=Li+24 mass threshold with their eq. 3 sigmoid, so a halo is FFB with probability `f_ffb(Mvir, z)`; 2=Boylan-Kolchin (2025) acceleration threshold `g_max > g_crit`, with log-normal scatter applied to the halo concentration. Both apply to centrals regardless of the halo's CGM/hot regime, and both draw fresh from the shared `rand()` stream each snapshot. See [Star formation and feedback](physics/starformation_and_feedback.md). |
| `ConcentrationOn` | int | no | `3` | Halo concentration method: 0=off; 1=Ishiyama+21 table; 2=V_max/V_vir; 3=V_max/V_vir with infall freeze for satellites. |
| `BulgeSizeOn` | int | no | `3` | Bulge radius model: 0=off; 1=Shen+2003 eq.33; 2=Shen+2003 eq.32; 3=Tonini+2016 (separate merger and instability channels, mass-weighted average). |
| `StarburstColdGasOn` | 0/1 | no | `1` | Include cold gas contribution during merger starbursts. |

---


## FFB parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `EnhancedSFEfficiency` | double | no | `0.2` | Maximum star formation efficiency during FFB bursts. `0.2` matches observations; `1.0` is the theoretical maximum. |

---

## H₂ star formation parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `H2DiskAreaOption` | int | no | `1` | Disk area for H₂ surface density: 0=π r_s²; 1=π (3 r_s)²; 2=2π r_s² (central Σ₀). |
| `H2RadialIntegrationOn` | 0/1 | no | `1` | Use radial ring integration for H₂ fraction (more accurate, slower). |
| `H2RadialNBins` | int | no | `25` | Number of radial bins for the ring integration. |
| `H2RadialRMaxFactor` | double | no | `5.0` | Outer integration radius as a multiple of the disk scale radius. |

---

## Model parameters

### Star formation

| Parameter | Units | Default | Description |
|-----------|-------|---------|-------------|
| `SfrEfficiency` | dimensionless | `0.05` | Cold/H2 gas consumption efficiency per dynamical time. Used by SFprescription 0, 1, 4, 5, 7 unconditionally, and by 6 (K13) only in the single-slab path (`H2RadialIntegrationOn=0`). Unused by 2 and 3 (Somerville+25 use their own density-modulated `epsilon_cl`) and by 6 in the radial path (uses the K13 local depletion time natively). |
| `RecycleFraction` | dimensionless | `0.43` | Fraction of stellar mass instantaneously recycled to cold gas. |
| `Yield` | dimensionless | `0.025` | Fraction of stellar mass returned as metals. |
| `FracZleaveDisk` | dimensionless | `0.0` | Fraction of newly produced metals transferred directly to hot gas. |

### Supernova feedback

| Parameter | Units | Default | Description |
|-----------|-------|---------|-------------|
| `FeedbackReheatingEpsilon` | dimensionless | `2.9` | Mass of cold gas reheated per unit of stellar mass formed (Martin 1999). |
| `FeedbackEjectionEfficiency` | dimensionless | `0.3` | Fraction of SN energy deposited into hot gas for ejection. |
| `EnergySN` | erg | `1.0e51` | Energy per supernova event. |
| `EtaSN` | M_sun⁻¹ | `5.0e-3` | Number of supernovae per solar mass of stars formed. |

### AGN feedback

| Parameter | Units | Default | Description |
|-----------|-------|---------|-------------|
| `RadioModeEfficiency` | dimensionless | `0.08` | AGN radio-mode heating efficiency (AGNrecipeOn=2). |
| `QuasarModeEfficiency` | dimensionless | `0.005` | AGN quasar-mode wind heating efficiency (AGNrecipeOn > 0). |
| `BlackHoleGrowthRate` | dimensionless | `0.015` | Fraction of cold gas accreted onto the BH during mergers (AGNrecipeOn > 0). |

### Mergers

| Parameter | Units | Default | Description |
|-----------|-------|---------|-------------|
| `ThreshMajorMerger` | dimensionless | `0.3` | Mass ratio above which a merger is classified as major. |
| `ThresholdSatDisruption` | dimensionless | `1.0` | M_vir-to-baryonic mass ratio below which a satellite is disrupted rather than merged. |
| `MergerTimeFactor` | dimensionless | `2.0` | Scales the Binney & Tremaine (1987) dynamical-friction merger timescale. It also sets how accreted stellar mass splits between the intracluster component and the BCG: when a satellite's subhalo is lost from the tree, its stars go to the ICS if the clock is still running (`MergTime > 0`) and onto the central if it has expired. |

### Gas cycling

| Parameter | Units | Default | Description |
|-----------|-------|---------|-------------|
| `ReIncorporationFactor` | dimensionless | `0.19` | Scales the reincorporation velocity threshold: `Vcrit = 354.26 km/s * ReIncorporationFactor`, so it is not a mass fraction despite the name. Reincorporation runs only where `Vvir > Vcrit`, at a rate `(Vvir/Vcrit - 1) * EjectedMass / t_dyn`. The `354.26 = V_SN/sqrt(2)` follows from `V_SN = 501 km/s` (Paper I eq. 11, from `eta_SN = 5e-3 Msun^-1` and `E_SN = 1e51 erg`). An earlier version used `V_SN = 630 km/s`, giving `445.48`, and `0.15` was fitted against that; `0.19` restores the same effective threshold (`445.48*0.15 = 66.8` vs `354.26*0.19 = 67.3 km/s`) with the correct `V_SN`. |

### Cooling and cold streams

| Parameter | Units | Default | Description |
|-----------|-------|---------|-------------|
| `MShockMsun` | Msun | `6.0e11` | Dekel & Birnboim (2006) virial-shock stability mass. Used twice: it sets the CGM/hot regime classification in `determine_and_store_regime()`, and it sets the cold-stream criterion in `cooling_recipe_hot()`. Both must see the same value, so change it here rather than in either site. |

### Reionization

| Parameter | — | Default | Description |
|-----------|---|---------|-------------|
| `Reionization_z0` | — | `8.0` | Characteristic redshift for reionization suppression (Kravtsov+04). |
| `Reionization_zr` | — | `7.0` | Width parameter for reionization suppression. |

See the **FFB parameters** section above for `EnhancedSFEfficiency`.

---

## Numerical time resolution

The per-snapshot physics loop (cooling, star formation, feedback) is integrated
with an *adaptive* number of sub-timesteps: the count scales with `deltaT / t_dyn`,
so a snapshot interval spanning several halo dynamical times is resolved with more
substeps (bounded by a `STEPS` floor and a `MAX_STEPS` cap). Tying the effective
resolution to `t_dyn` rather than the raw snapshot cadence is what lets one
calibration transfer across simulations with different output spacing.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `SubstepResolution` | double | `1.0` | Runtime multiplier on the adaptive-substep floor **and** cap. **Calibration-locked numerical knob, not a physics choice** — the model is calibrated at `1.0`; do not change it for science runs without recalibrating. Coarse steps over-cool (cooling outruns the AGN `r_heat` response before it can react), so raising the resolution lowers the massive-end SMF and total stellar mass. The shift from `1.0` to fully converged is only **~0.1 dex** at the massive end (within typical observational SMF scatter), but runtime grows **~linearly** with the substep count. Use higher values only for deliberate convergence / resolution studies. Note that the reported `SfrDisk`/`SfrBulge` are normalised by the substep count actually integrated (`SubstepsUsed`), not by `STEPS`; before September 2026 they were divided by `STEPS`, which scaled the reported SFR by `effective_steps/STEPS` and made this knob useless for convergence testing. **Run convergence sweeps with `FFBRandomMode=1` and `RegimeRandomMode=1`.** With the default per-snapshot draws, changing the substep count shifts the global `rand()` sequence, and the resulting scatter is non-monotonic and larger than the convergence signal: total stellar mass on mini-Millennium goes 1528 / 962 / 1255 / 1191 for N = 5 / 10 / 20 / 30. With persistent per-galaxy draws the same sweep is monotonic and converging -- 1529 / 1349 / 1262 / 1211, i.e. -12%, -6.4%, -4%. Note the two modes agree to within 2% at N = 5, 20 and 30 but differ by 30% at the default N = 10, which is worth understanding before quoting either number. |

**Convergence note.** The substep dependence is a long-standing property of SAGE's
cooling/feedback operator-splitting (it is present, and slightly *stronger*, with
the classic `CGMrecipeOn=0` cooling), not something introduced by the two-regime
CGM physics. Because the effect is only ~0.1 dex on the SMF, the calibrated `1.0` model
sits within observational constraints; a converged model would need at most a light
retune and would show more simulation-consistent behaviour, at higher compute cost.

---

## MPI forest distribution

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `ForestDistributionScheme` | string | no | `generic_power_in_nhalos` | How forests are distributed over MPI tasks: `uniform_in_forests`, `linear_in_nhalos`, `quadratic_in_nhalos`, `exponent_in_nhalos`, `generic_power_in_nhalos`. |
| `ExponentForestDistributionScheme` | double | no | `0.7` | Exponent for `exponent_in_nhalos` or `generic_power_in_nhalos` schemes. |

---

## Removed parameters

SAGE26 was developed with a wide set of switches so alternative prescriptions
could be compared. For release, the prescriptions the papers use are the only
ones the code carries, and their parameters are gone.

**An unrecognised tag is a startup error, not a warning.** A parameter file
written against an earlier version will abort with
`Tag 'X' not allowed or multiply defined` until the lines below are deleted.

Values that are still meaningful physical constants now live as named
constants in the source, cited at their definition. They are listed here so a
reader who goes looking for a knob can find where it went.

| Removed parameter | Fate |
|---|---|
| `ColdStreamCeilingOn` | Dekel & Birnboim (2006) eq. 39 (the former mode 1) is the only cold-stream prescription. |
| `StreamMassFactor` | `DB06_STREAM_MASS_FACTOR = 3.0` in [`src/model_cooling_heating.c`](../src/model_cooling_heating.c). |
| `StreamThresholdWidthDex` | `STREAM_THRESHOLD_WIDTH_DEX = 0.15` in [`src/model_cooling_heating.c`](../src/model_cooling_heating.c). |
| `StreamZCritWidth` | Removed with the smoothed-gate modes. |
| `TrackICSAssembly` | Always on: `ICS_disrupt`, `ICS_accrete` and `ICS_sum_mt` are always accumulated. |
| `GasDiskRadiusFactor` | `GAS_DISK_RADIUS_FACTOR = 1.0` in [`src/model_starformation_and_feedback.c`](../src/model_starformation_and_feedback.c). |
| `FeedbackFreeModeOn` | Renamed `EnhancedStarFormationOn`, with modes reduced to 0/1/2. The old mode 4 (BK25 with log-normal concentration scatter) is now mode 2; old modes 2, 3, 5, 6, 7 and 8-11 are gone. |
| `FFBMaxEfficiency` | Renamed `EnhancedSFEfficiency`. Still tunable. |
| `FFBConcSigma` | `FFB_CONC_SIGMA = 0.2` in [`src/model_regimes.c`](../src/model_regimes.c). |
| `FFBThresholdSlope` | `FFB_THRESHOLD_SLOPE = -6.2` in [`src/model_regimes.c`](../src/model_regimes.c). |
| `FFBIgnoreRegime` | Always on: the FFB criteria apply whatever the halo's CGM/hot regime. |
| `FFBRandomMode` | Always a fresh draw each snapshot. |
| `RegimeRandomMode` | Always a fresh draw each snapshot. |
| `FFBFeedbackDelayMyr`, `FFBCloudClumping`, `FFBCloudClumpingDisk`, `FFBStreamRadiusFraction`, `FFBShellSoundSpeedKms`, `FFBToomreQ`, `FFBSigmaCritMsunPc2` | Removed with the Dekel+23 shell and disc criteria (old FFB modes 8-11). |
| `SNEnergyConservationOn` | Always on: both the reheating and the ejection term are bounded by the supernova energy available. |
| `MaxSNEnergyCoupling` | `MAX_SN_ENERGY_COUPLING = 2.0` in [`src/model_misc.h`](../src/model_misc.h). |
| `RedshiftPowerLawExponent` | `FIRE_REDSHIFT_EXPONENT = 1.25`, defined in both [`src/model_starformation_and_feedback.c`](../src/model_starformation_and_feedback.c) and [`src/model_mergers.c`](../src/model_mergers.c). |

The same names are also gone from the `Header/Runtime` attributes of the HDF5
output, which records tunable parameters. Analysis scripts that read them
should use the constants above instead.

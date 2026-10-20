# Changelog

Notable changes to SAGE26. This file starts at the pre-release cleanup;
earlier history is in `git log`.

The cleanup pass is bound by a bit-identical output guarantee: every entry
below reproduces the committed regression baseline dataset-for-dataset unless
it says otherwise. See
[docs/developer/REGRESSION_BASELINE.md](docs/developer/REGRESSION_BASELINE.md).

## Unreleased

### Added

- `TreeType mimic_lhalo_binary`: a reader for the L-Halo binary trees written by Mimic's tree converter (`src/io/read_tree_mimic_lhalo_binary.c`). The files carry the classic L-Halo header followed by Mimic's 88-byte halo record, which stores `M_Crit200` in Msun/h and already-specific `Spin`, and lacks `M_Mean200`, `M_TopHat`, `SubhaloIndex` and `SubHalfMass`. The reader converts each record into `struct halo_data`, scaling the mass to 10^10 Msun/h and setting the absent fields to -1. The existing `lhalo_binary` setup pass is shared through `setup_forests_io_lht_binary_recsize()`, and both readers now refuse a file whose size does not equal header plus one record per halo, or whose per-forest counts do not sum to the header total. Output of the existing readers is unchanged.

### Removed and renamed parameters (breaking)

The release build carries only the prescriptions the papers use. Switches that
selected alternatives are gone, along with the parameters that only those
alternatives read.

**An unrecognised tag is a startup error.** A parameter file written against
an earlier version aborts with `Tag 'X' not allowed or multiply defined` until
the removed lines are deleted. The same names are also gone from the
`Header/Runtime` attributes of the HDF5 output.

[docs/parameters.md](docs/parameters.md#removed-parameters) has the full table
of what each one became.

Renamed:

- `FeedbackFreeModeOn` -> `EnhancedStarFormationOn`, reduced from twelve modes
  to three: `0` off, `1` Li+24 mass threshold with their eq. 3 sigmoid, `2`
  Boylan-Kolchin 2025 acceleration threshold with log-normal concentration
  scatter. Old mode 4 is the new mode 2; old modes 2, 3, 5, 6, 7 and 8-11 are
  gone.
- `FFBMaxEfficiency` -> `EnhancedSFEfficiency` (still tunable).

Hardcoded to their published values, and the switch removed:

- `ColdStreamCeilingOn` -- Dekel & Birnboim (2006) eq. 39 is the only
  cold-stream prescription.
- `TrackICSAssembly` -- ICS assembly history is always accumulated.
- `FFBIgnoreRegime` -- the FFB criteria always apply, whatever the halo regime.
- `RegimeRandomMode`, `FFBRandomMode` -- both draws are always fresh each
  snapshot.
- `SNEnergyConservationOn` -- the supernova energy bound is always applied to
  both the reheating and the ejection term.

Moved to named constants in the source, cited at their definition:

- `StreamMassFactor`, `StreamThresholdWidthDex`, `GasDiskRadiusFactor`,
  `MaxSNEnergyCoupling`, `FFBConcSigma`, `FFBThresholdSlope`,
  `RedshiftPowerLawExponent`.

Removed outright with the Dekel+23 shell and disc FFB criteria:

- `FFBFeedbackDelayMyr`, `FFBCloudClumping`, `FFBCloudClumpingDisk`,
  `FFBStreamRadiusFraction`, `FFBShellSoundSpeedKms`, `FFBToomreQ`,
  `FFBSigmaCritMsunPc2`, `StreamZCritWidth`.

### Removed API

- `calculate_gmax_BK25()` (the old FFB mode 2/3 implementation) is gone from
  `model_regimes.h`.
- `determine_and_store_ffb_regime()` no longer takes `infallingGas` and `dt`;
  they were only read by the removed Dekel+23 criteria.

### Removed galaxy fields

- `FFBRandom` and `RegimeRandom` are gone from `struct GALAXY`. They were never
  written to output. Two `rand()` calls are still consumed and discarded at
  galaxy creation: the stream is shared with the regime and FFB
  classifications, so dropping them would re-phase every later draw.

### Fixed

- `docs/parameters.md` and `README.md` documented `H2DiskAreaOption` as taking
  1/2/3; the code accepts 0/1/2, so the documented meaning of the default was
  wrong by one.
- Two merger-timescale unit tests asserted a positive merger time against a
  value that was identically zero, because their fixtures never set
  `MergerTimeFactor`.
- `MergerTimeFactor` was missing from the parameter reference.
- This file did not exist, so the `docs/changelog.md` Sphinx include was broken.

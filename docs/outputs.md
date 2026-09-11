# Outputs

This page documents the galaxy catalogues SAGE26 writes to disk: file
layout, per-field units and descriptions, flag conventions, and the
optional auxiliary datasets controlled by run parameters.

Sources:
[`src/io/save_gals_hdf5.h`](https://github.com/MBradley1985/SAGE26/blob/main/src/io/save_gals_hdf5.h),
[`src/io/save_gals_hdf5.c`](https://github.com/MBradley1985/SAGE26/blob/main/src/io/save_gals_hdf5.c),
[`src/io/save_gals_binary.h`](https://github.com/MBradley1985/SAGE26/blob/main/src/io/save_gals_binary.h),
[`src/io/save_gals_binary.c`](https://github.com/MBradley1985/SAGE26/blob/main/src/io/save_gals_binary.c).

## Output formats

The `OutputFormat` parameter selects one of two writers:

| Value | Format | Notes |
|-------|--------|-------|
| `sage_binary` | Raw little-endian C struct dump | One file per task (`<FileNameGalaxies>_<filenr>`). |
| `sage_hdf5` | HDF5 column store with metadata | One file per task plus a master file linking them. |

Both writers emit one file per output snapshot listed in
`FileWithOutputSnaps`. Per-galaxy field content is the same in both
formats, but field *names* and the on-disk layout differ -- see the
field reference below.

Code units used internally are:

- Length: `Mpc / h`
- Mass: `1.0e10 Msun / h`
- Velocity: `km / s`
- Time: derived from length / velocity (UnitTime_in_Megayears stored in
  the header)

Where the writer converts to a more convenient unit (SFR, outflow rate,
cooling/heating power, merger times) it is noted explicitly in the
field reference. Everything else is in code units.

## HDF5 layout

```
<FileNameGalaxies>_<N>.hdf5
+- Header/
|  +- Simulation/   (attributes: SimulationDir, FileWithSnapList,
|  |                              LastSnapshotNr, SimMaxSnaps,
|  |                              omega_matter, omega_lambda, particle_mass,
|  |                              hubble_h, num_simulation_tree_files,
|  |                              box_size, num_trees_this_file)
|  +- Runtime/      (every recipe flag and model parameter from the .par
|  |                 file, plus UnitLength_in_cm/UnitMass_in_g/
|  |                 UnitVelocity_in_cm_per_s and frac_volume_processed)
|  +- Misc/         (attributes: num_cores, sage_data_version,
|  |                              sage_version, git_SHA_reference)
|  +- snapshot_redshifts   (1-D dataset, length SimMaxSnaps)
|  +- output_snapshots     (1-D dataset, length NumOutputs)
+- TreeInfo/<snap>/  (per-tree galaxy counts for each output snapshot)
+- Snap_<N>/         (one group per output snapshot)
   +- SnapNum, Type, GalaxyIndex, ...   (one dataset per field)
   +- SFHMassDisk, SFHMassBulge          (only if SaveFullSFH = 1)
```

Each galaxy-property dataset has two HDF5 attributes:

- `Description` -- one-line text description.
- `Units` -- unit string.

The cumulative SFH datasets additionally carry a `NumSnapshots`
attribute equal to `SimMaxSnaps`.

A separate master file (`<FileNameGalaxies>.hdf5`) is written at the
end of the run; it contains the same `Header` group and uses HDF5
external links to expose each per-task file's snapshot data under a
single tree.

## Binary layout

```
[int32  ntrees]
[int32  tot_ngals]
[int32  forest_ngals[ntrees]]
[GALAXY_OUTPUT  galaxies[tot_ngals]]
```

Galaxies are stored in snapshot order; the i-th block of
`forest_ngals[i]` galaxies belongs to tree `i`. The `GALAXY_OUTPUT`
struct layout is fixed at compile time -- see
`src/io/save_gals_binary.h` for the canonical definition. No header
metadata, descriptions, or units are written; downstream tools must
know the struct layout in advance.

The binary writer does **not** emit the cumulative SFH arrays; use HDF5
output if you need `SFHMassDisk` / `SFHMassBulge`.

## Flag conventions

### `Type` -- galaxy hierarchy position

| Value | Meaning |
|-------|---------|
| 0 | Central galaxy of the main FoF halo. |
| 1 | Central of a sub-halo (Type 1 satellite, still has its own subhalo). |
| 2 | Orphan satellite (subhalo lost; will merge or disrupt within the current timestep). |

For Type 0 galaxies, all infall fields (`infallMvir`, `infallVvir`,
`infallVmax`, `infallStellarMass`, `TimeOfInfall`) are zeroed on
output -- they are only meaningful for satellites.

### `mergeType` -- merger/disruption channel

| Value | Meaning |
|-------|---------|
| 0 | None (galaxy still evolving). |
| 1 | Minor merger (mass ratio below `ThreshMajorMerger`). |
| 2 | Major merger (mass ratio above `ThreshMajorMerger`). |
| 3 | Reserved for disk instability (header documents it, but no code path sets this value). |
| 4 | Disrupted to intra-cluster stars. |

### `Regime` -- CGM regime classification

| Value | Meaning |
|-------|---------|
| 0 | CGM regime (cool flow / precipitation; cooling driven by `cooling_recipe_cgm()`). |
| 1 | Hot-halo regime (Mvir above the Dekel & Birnboim 2006 M_shock; cooling driven by `cooling_recipe_hot()`). |

Set every snapshot in `determine_and_store_regime()` when
`CGMrecipeOn = 1`. Left at zero (and unused) when `CGMrecipeOn = 0`.

### `FFBRegime` -- feedback-free burst classification

| Value | Meaning |
|-------|---------|
| 0 | Normal halo (standard star formation + SN feedback). |
| 1 | FFB halo (Li+2024 / Boylan-Kolchin+2025 starburst path; disk-instability check skipped, SN feedback still applied). |

Only set when `FeedbackFreeModeOn = 1`.

## Galaxy property reference

The tables below list every per-galaxy field written to disk, with its
HDF5 name (the binary writer uses the corresponding `GALAXY_OUTPUT`
struct field name -- see the "Binary vs HDF5 name differences"
section below) and the unit actually written.

### Identity and tree linkage

| Field | Units | Description |
|-------|-------|-------------|
| `SnapNum` | -- | Snapshot the galaxy is located at. |
| `Type` | -- | Galaxy hierarchy position (see flag table above). |
| `GalaxyIndex` | -- | Galaxy ID unique across all trees and files. Computed from local galaxy number, tree number, and file number. |
| `CentralGalaxyIndex` | -- | `GalaxyIndex` of the central in this galaxy's FoF group. |
| `SAGEHaloIndex` | -- | Halo index from SAGE's restructured tree (not the input tree's). |
| `SAGETreeIndex` | -- | Tree number (within file) this galaxy belongs to. |
| `SimulationHaloIndex` | -- | `\|MostBoundID\|` from the input tree files. |
| `TaskForestNr` | -- | HDF5 only. Task-local forest number used for parallel bookkeeping. |
| `mergeType` | -- | Merger/disruption channel (see flag table above). |
| `mergeIntoID` | -- | `GalaxyIndex` of the galaxy this one is merging into. |
| `mergeIntoSnapNum` | -- | Snapshot of the merge target. |
| `dT` | Myr | Time since this galaxy was last evolved (snapshot interval / sub-step). |

### Halo properties

| Field | Units | Description |
|-------|-------|-------------|
| `Posx`, `Posy`, `Posz` | Mpc / h | Galaxy spatial position. HDF5 splits into three 1-D datasets; binary stores as `Pos[3]`. |
| `Velx`, `Vely`, `Velz` | km / s | Galaxy peculiar velocity. HDF5 splits; binary stores as `Vel[3]`. |
| `Spinx`, `Spiny`, `Spinz` | Mpc * km / s | Halo specific angular momentum vector. HDF5 splits; binary stores as `Spin[3]`. |
| `Len` | -- | Number of particles in the galaxy's halo. |
| `Mvir` | 1.0e10 Msun / h | Virial mass of this galaxy's halo. |
| `CentralMvir` | 1.0e10 Msun / h | Virial mass of the main FoF halo (central). |
| `Rvir` | Mpc / h | Virial radius of this galaxy's halo. |
| `Vvir` | km / s | Virial velocity of this galaxy's halo. |
| `Vmax` | km / s | Maximum circular velocity of this galaxy's halo. |
| `VelDisp` | km / s | Velocity dispersion of this galaxy's halo. |
| `Concentration` | -- | NFW halo concentration from the Ishiyama+21 c-M relation (set when `ConcentrationOn = 1`). |
| `g_max` | -- | Running maximum of the precipitation g-parameter across all snapshots for this halo (HDF5 dtype: float64). |

### Baryonic reservoirs

| Field | Units | Description |
|-------|-------|-------------|
| `ColdGas` | 1.0e10 Msun / h | Mass of gas in the cold reservoir (disk + bulge). |
| `StellarMass` | 1.0e10 Msun / h | Total stellar mass (disk + bulge). |
| `BulgeMass` | 1.0e10 Msun / h | Stellar mass in the bulge (= `MergerBulgeMass` + `InstabilityBulgeMass`). |
| `HotGas` | 1.0e10 Msun / h | Mass of gas in the hot halo reservoir. |
| `EjectedMass` | 1.0e10 Msun / h | Mass of gas ejected from the halo by SN feedback. |
| `BlackHoleMass` | 1.0e10 Msun / h | Mass of the central black hole. |
| `IntraClusterStars` / `ICS` | 1.0e10 Msun / h | Total ICS mass (HDF5 name `IntraClusterStars`; binary field `ICS`). |
| `CGMgas` | 1.0e10 Msun / h | Mass of gas in the circum-galactic medium (Regime 0 reservoir). |
| `H2gas` | 1.0e10 Msun / h | Mass of molecular hydrogen in the cold gas. |
| `H1gas` | 1.0e10 Msun / h | Mass of atomic hydrogen in the cold gas. |
| `MergerBulgeMass` | 1.0e10 Msun / h | Stellar mass in the merger-built classical bulge (Tonini+2016 channel A). |
| `InstabilityBulgeMass` | 1.0e10 Msun / h | Stellar mass in the disk-instability pseudo-bulge (Tonini+2016 channel B). |

### Metals

| Field | Units | Description |
|-------|-------|-------------|
| `MetalsColdGas` | 1.0e10 Msun / h | Metals in the cold reservoir. |
| `MetalsStellarMass` | 1.0e10 Msun / h | Metals locked in stars. |
| `MetalsBulgeMass` | 1.0e10 Msun / h | Metals locked in bulge stars. |
| `MetalsHotGas` | 1.0e10 Msun / h | Metals in the hot reservoir. |
| `MetalsEjectedMass` | 1.0e10 Msun / h | Metals in the ejected reservoir. |
| `MetalsIntraClusterStars` / `MetalsICS` | 1.0e10 Msun / h | Metals in intra-cluster stars (HDF5 name `MetalsIntraClusterStars`; binary field `MetalsICS`). |
| `MetalsCGMgas` | 1.0e10 Msun / h | Metals in the CGM reservoir. |

### Star formation, feedback, and BH accretion

| Field | Units | Description |
|-------|-------|-------------|
| `SfrDisk` | Msun / yr | Disk SFR averaged over the snapshot interval (mean over the STEPS sub-steps). |
| `SfrBulge` | Msun / yr | Bulge SFR averaged over the snapshot interval (starburst component). |
| `SfrDiskZ` | dimensionless | Mass-weighted metallicity of the star-forming disk gas (`MetalsColdGas / ColdGas` averaged over sub-steps). Despite the "Units" attribute saying `Msun/yr`, this field is a metallicity ratio, not a rate. |
| `SfrBulgeZ` | dimensionless | Mass-weighted metallicity of the star-forming bulge gas. Same caveat as `SfrDiskZ`. |
| `OutflowRate` | Msun / yr | Cold-gas reheating rate (cold -> hot) integrated over the substep. |
| `MassLoading` | dimensionless | SN mass-loading factor (outflow rate / SFR). |
| `Cooling` | log10(erg / s) | log10 of cooling energy rate. Stored as 0.0 when no cooling occurred (i.e. zero, not "1 erg/s"). |
| `Heating` | log10(erg / s) | log10 of AGN radio-mode heating energy rate. Stored as 0.0 when no heating occurred. |
| `QuasarModeBHaccretionMass` | 1.0e10 Msun / h | Mass accreted by the black hole during the last sub-step via the quasar (merger-driven) mode. |

### Sizes

| Field | Units | Description |
|-------|-------|-------------|
| `DiskRadius` / `DiskScaleRadius` | Mpc / h | Disk exponential scale length from Mo, Mao & White (1998). HDF5 name `DiskRadius`; binary field `DiskScaleRadius`. |
| `BulgeRadius` | Mpc / h | Bulge half-mass radius (Lange et al. 2015 / Shen et al. 2003 normalisation; updated by mergers and instabilities). |
| `MergerBulgeRadius` | Mpc / h | Sub-component bulge radius from the merger channel. |
| `InstabilityBulgeRadius` | Mpc / h | Sub-component bulge radius from the disk-instability channel (Tonini+2016 Eq. 15 incremental update). |

### Times

| Field | Units | Description |
|-------|-------|-------------|
| `TimeOfLastMajorMerger` | Myr | Code-time of the last major merger, converted to Myr. |
| `TimeOfLastMinorMerger` | Myr | Code-time of the last minor merger, converted to Myr. |
| `TimeOfInfall` | Myr | Code-time at which this galaxy became a satellite (zero for Type 0). |

### Two-regime cooling (CGM model)

| Field | Units | Description |
|-------|-------|-------------|
| `Regime` | -- | CGM regime flag (see flag table above). |
| `tcool` | Msun / Gyr | Cooled gas rate, `coolingGas / dt`, including the active regime's final suppression. |
| `tff` | Gyr | Free-fall time of the CGM gas at the cooling radius. |
| `tcool_over_tff` | dimensionless | Voit (2015) precipitation criterion ratio. |
| `tdeplete` | Myr | Depletion time of the CGM reservoir under the current cooling rate. |
| `H2DepletionTime_Gyr` | Gyr | H2 depletion time from the K13 prescription. Set to -1 when not applicable. |
| `RcoolToRvir` | dimensionless | Ratio of the cooling radius to the virial radius. |
| `mdot_cool` | Msun / yr | Hot-halo cooling rate (mass flowing from hot to cold). |
| `mdot_stream` | Msun / yr | Cold-stream accretion rate (CGM-regime mass flowing from CGM to cold). |

### Infall properties (Type > 0 only)

For Type 0 (central) galaxies, all five infall fields are zeroed on
output. For satellites, they record the halo state at the snapshot
just before infall:

| Field | Units | Description |
|-------|-------|-------------|
| `infallMvir` | 1.0e10 Msun / h | Virial mass of this halo at the previous timestep (the "infall snapshot"). |
| `infallVvir` | km / s | Virial velocity at the infall snapshot. |
| `infallVmax` | km / s | Maximum circular velocity at the infall snapshot. |
| `infallStellarMass` | 1.0e10 Msun / h | Stellar mass at the moment of becoming a satellite. |

### Feedback-free burst diagnostic

| Field | Units | Description |
|-------|-------|-------------|
| `FFBRegime` | -- | FFB classification flag (see flag table above). |

### ICS assembly tracking (`TrackICSAssembly = 1`)

These three fields accumulate per-channel deposits into the ICS so
that the mean assembly time can be reconstructed downstream:

| Field | Units | Description |
|-------|-------|-------------|
| `ICS_disrupt` | 1.0e10 Msun / h | Cumulative stellar mass disrupted to ICS via satellite disruption. |
| `ICS_accrete` | 1.0e10 Msun / h | Cumulative ICS accreted from satellites via per-snapshot inheritance. |
| `ICS_sum_mt` | 1.0e10 Msun / h * code_time | Mass-weighted sum of `mass * time_of_deposition` (code-time units). |

Mean ICS assembly lookback for a given galaxy:

```
<t_assembly> = ICS_sum_mt / (ICS_disrupt + ICS_accrete)
```

The result is in code-time units; multiply by
`UnitTime_in_Megayears` (read from the HDF5 header) to convert to Myr.

These fields are always present in both writers, but they only
accumulate when `TrackICSAssembly = 1`. With the parameter off, they
remain at zero.

## Cumulative star formation history (`SaveFullSFH = 1`)

HDF5 output only. When `SaveFullSFH = 1`, two extra 2-D datasets are
written per output snapshot:

| Dataset | Shape | Units | Description |
|---------|-------|-------|-------------|
| `SFHMassDisk` | (ngalaxies, SimMaxSnaps) | 1.0e10 Msun / h | Stellar mass formed in the disk at each snapshot up to the current one. |
| `SFHMassBulge` | (ngalaxies, SimMaxSnaps) | 1.0e10 Msun / h | Stellar mass formed in the bulge (starbursts) at each snapshot up to the current one. |

Each dataset carries `Description`, `Units`, and `NumSnapshots`
attributes. Entries for snapshots beyond the galaxy's current
`SnapNum` are zero. Reconstructing the SFR(z) curve for a galaxy
amounts to differencing consecutive snapshot bins and dividing by
the snapshot interval.

This option roughly doubles the per-galaxy footprint of the output
file, so leave it off unless you need the full SFH.

## Binary vs HDF5 name differences

A handful of fields use different names between the two writers.
Otherwise the field set is identical:

| Binary (GALAXY_OUTPUT) | HDF5 (HDF5_GALAXY_OUTPUT) |
|------------------------|----------------------------|
| `Pos[3]` | `Posx`, `Posy`, `Posz` |
| `Vel[3]` | `Velx`, `Vely`, `Velz` |
| `Spin[3]` | `Spinx`, `Spiny`, `Spinz` |
| `DiskScaleRadius` | `DiskRadius` |
| `ICS` | `IntraClusterStars` |
| `MetalsICS` | `MetalsIntraClusterStars` |
| -- | `TaskForestNr` (HDF5 only) |

The HDF5 writer also lays out fields as struct-of-arrays (one
dataset per field), whereas the binary writer lays them out as
array-of-structs.

## Parameter -> output mapping

| Parameter | Effect on output |
|-----------|------------------|
| `OutputFormat` | Selects `sage_binary` or `sage_hdf5`. |
| `OutputDir` | Directory for the output files. |
| `FileNameGalaxies` | Filename stem; final names are `<stem>_<filenr>` (binary) or `<stem>_<filenr>.hdf5` (HDF5), plus `<stem>.hdf5` master. |
| `FileWithOutputSnaps` | Snapshot list to write. Each listed snap becomes either its own binary file or a `Snap_<N>` HDF5 group. |
| `SaveFullSFH` | Enables the cumulative `SFHMassDisk` / `SFHMassBulge` 2-D HDF5 datasets. |
| `TrackICSAssembly` | Activates accumulation into `ICS_disrupt`, `ICS_accrete`, `ICS_sum_mt`. |
| `ConcentrationOn` | Populates the `Concentration` field (otherwise 0). |
| `CGMrecipeOn` | Populates `Regime`, `CGMgas`, `MetalsCGMgas`, `tcool`, `tff`, `tcool_over_tff`, `tdeplete`, `RcoolToRvir`, `mdot_cool`, `mdot_stream`. With it off, these stay at their initialised values. |
| `FeedbackFreeModeOn` | Populates `FFBRegime`. |

See [`parameters.md`](parameters.md) for full parameter descriptions
and defaults.

## Where to go next

- [`getting_started.md`](getting_started.md) -- run SAGE26 and produce
  these outputs.
- [`parameters.md`](parameters.md) -- the parameter file that controls
  which fields are populated.
- [`physics/ics.md`](physics/ics.md) -- detail on the ICS assembly
  channels that feed `ICS_disrupt` / `ICS_accrete` / `ICS_sum_mt`.
- [`physics/cooling_and_heating.md`](physics/cooling_and_heating.md) --
  what `tcool`, `tff`, `tcool_over_tff`, `Regime`, and `Cooling` /
  `Heating` actually measure.

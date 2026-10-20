<p align="center">
  <img src="SAGElogo.png" width="300" alt="SAGE26 logo"/>
</p>

# SAGE26 — Semi-Analytic Galaxy Evolution

[![Documentation Status](https://readthedocs.org/projects/sage26/badge/?version=latest)](https://sage26.readthedocs.io/en/latest/)

SAGE26 is a C99 semi-analytic code for modelling galaxy formation in a cosmological
context. It is a major update to [Croton et al. (2016)](https://arxiv.org/abs/1601.04709),
adding a two-regime CGM model, FIRE stellar feedback, feedback-free burst (FFB) galaxies,
multiple H2-based star formation prescriptions, and extended bulge/ICS tracking.

SAGE reads N-body merger trees, evolves galaxies through cosmic time using
semi-analytic prescriptions, and writes galaxy catalogues in HDF5 or binary format.
It runs on any simulation whose trees are in a supported format and contain a minimum set
of halo properties. Test trees for the
[Mini-Millennium Simulation](http://arxiv.org/abs/astro-ph/0504097) are provided.

---

## What is new in SAGE26

| Feature | Parameter | Reference |
|---------|-----------|-----------|
| Two-regime CGM model with self-regulating precipitation | `CGMrecipeOn` | Dekel & Birnboim (2006), Voit (2015) |
| FIRE stellar feedback | `FIREmodeOn` | Muratov et al. (2015) |
| Feedback-free burst galaxies | `EnhancedStarFormationOn` | Li et al. (2024), Boylan-Kolchin (2025) |
| 8 star formation prescriptions, 6 of them H2-based | `SFprescription` | BR06, S25, KD12, KMT09, K13, GD14 |
| Separate merger/instability bulge tracking | `BulgeSizeOn` | Tonini et al. (2016) |
| ICS assembly tracking | always on | — |
| Full star formation history arrays | `SaveFullSFH` | — |
| Concentration | `ConcentrationOn` | Ishiyama+21 lookup table, Vmax/Vvir, Vmax/Vvir with subhalo infall freeze |
| ConsistentTrees, Genesis, Gadget-4, Mimic L-Halo tree readers | `TreeType` | — |
| HDF5 output + libsage.so for Python/PSO | `OutputFormat` | — |

---

## Install

### Dependencies

| Package | Required | Notes |
|---------|----------|-------|
| C99 compiler (gcc or clang) | Yes | |
| [GSL](https://www.gnu.org/software/gsl/) | Yes | required for tests |
| [HDF5](https://www.hdfgroup.org/) | Optional | HDF5 tree reading and output |
| MPI | Optional | parallel execution |

### Build

```bash
git clone https://github.com/MBradley1985/SAGE26.git
cd SAGE26
make                   # default build -- produces ./sage and libsage.so
make USE-MPI=          # serial build (no mpicc needed)
make USE-HDF5=         # build without HDF5 tree reading and output
make MEM-CHECK=yes     # address/UB sanitizers for debugging (gcc only)
make clean             # remove all build artefacts
```

**MPI and HDF5 are on by default** (`USE-MPI` and `USE-HDF5` at the top of the
Makefile), so a plain `make` needs `mpicc` and the HDF5 libraries on your path.
Set a switch to empty to turn it off, as above -- `USE-MPI=yes` is already the
default and does nothing. The regression baseline requires the serial build.

---

## Quickstart

```bash
./first_run.sh                          # create output dirs, download Mini-Millennium trees
./sage input/millennium.par             # run the model (serial)
mpirun -np 4 ./sage input/millennium.par  # run in parallel
python plotting/allresults-local.py     # z=0 diagnostic plots
python plotting/allresults-history.py   # multi-redshift diagnostics
```

Full details on the parameter file format, output format, and all physics switches are in
[`docs/parameters.md`](docs/parameters.md).

---

## Physics options

Every switch below is parsed by [`src/core_read_parameter_file.c`](src/core_read_parameter_file.c).
Full descriptions, defaults, and the model-parameter (non-switch) list are in
[`docs/parameters.md`](docs/parameters.md).

### Star formation (`SFprescription`)

| Value | Prescription |
|-------|-------------|
| 0 | Croton et al. (2006) original |
| 1 | Blitz & Rosolowsky (2006) H2 |
| 2 | Somerville et al. (2025) SFR |
| 3 | Somerville et al. (2025) SFR + H2 |
| 4 | Krumholz & Dekel (2012) |
| 5 | Krumholz, McKee & Tumlinson (2009) |
| 6 | Krumholz (2013) |
| 7 | Gnedin & Draine (2014) |

### AGN feedback (`AGNrecipeOn`)

| Value | Mode |
|-------|------|
| 0 | Off |
| 1 | Empirical (Croton+2016 radio mode) |
| 2 | Bondi-Hoyle accretion |
| 3 | Cold-cloud accretion |

### Supernova feedback (`SupernovaRecipeOn`)

| Value | Mode |
|-------|------|
| 0 | Off |
| 1 | Croton+2016 reheating/ejection |

### Reionization (`ReionizationOn`)

| Value | Mode |
|-------|------|
| 0 | Off |
| 1 | Kravtsov+2004 analytic suppression of infall |

### Disk instability (`DiskInstabilityOn`)

| Value | Mode |
|-------|------|
| 0 | Off |
| 1 | Toomre criterion drives bulge growth, instability starbursts, BH growth |

### Two-regime CGM model (`CGMrecipeOn`)

Galaxies below the Dekel & Birnboim (2006) shock mass are placed in Regime 0 (CGM /
precipitation regime); those above are in Regime 1 (hot-halo / classical cooling regime).
Each regime uses a dedicated cooling recipe.

| Parameter | Values | Effect |
|-----------|--------|--------|
| `CGMrecipeOn` | 0/1 | 0=off (classical C16 cooling only); 1=on, following Carr et al. (2023). |


### Adaptive time integration (`SubstepResolution`)

The snapshot interval is integrated with a substep count that scales with
`deltaT / t_dyn`, so high-redshift snapshots spanning several dynamical times
are resolved with more substeps (bounded by a `STEPS` floor and `MAX_STEPS`
cap) rather than a fixed count.

| Parameter | Values | Effect |
|-----------|--------|--------|
| `SubstepResolution` | double | Runtime multiplier on both the adaptive-substep floor and cap; default 1.0. Sweep for convergence / N-invariance testing without recompiling |

### FIRE stellar feedback (`FIREmodeOn`)

| Value | Mode |
|-------|------|
| 0 | Off |
| 1 | Muratov+2015 FIRE mass loading |

### Feedback-free burst galaxies (`EnhancedStarFormationOn`)

| Value | Mode |
|-------|------|
| 0 | Off |
| 1 | Li+2024 mass threshold with their eq. 3 sigmoid |
| 2 | Boylan-Kolchin+2025 acceleration threshold, log-normal concentration scatter |

### Halo concentration (`ConcentrationOn`)

| Value | Method |
|-------|--------|
| 0 | Off |
| 1 | Ishiyama+2021 lookup table |
| 2 | V_max / V_vir from the simulation |
| 3 | V_max / V_vir with subhalo infall freeze for satellites |

### H2 star formation (auxiliary switches)

| Parameter | Values | Effect |
|-----------|--------|--------|
| `H2DiskAreaOption` | 0–2 | Disk area for H2 surface density: 0=π r_s²; 1=π (3 r_s)²; 2=2π r_s² |
| `H2RadialIntegrationOn` | 0/1 | Radial ring integration for H2 fraction (slower, more accurate) |
| `H2RadialNBins` | int | Number of radial bins for ring integration |

### Mergers and ICS tracking

| Parameter | Values | Effect |
|-----------|--------|--------|
| `MergerTimeFactor` | double | Scales the dynamical-friction merger timescale, and with it the split of accreted stellar mass between the ICS and the BCG |

ICS assembly tracking (`ICS_disrupt`, `ICS_accrete`, `ICS_sum_mt`) is always on.

### Output

| Parameter | Values | Effect |
|-----------|--------|--------|
| `OutputFormat` | string | `sage_hdf5`, `sage_binary`, or `lhalo_binary_output` to convert the input trees to lhalo-binary instead of running the model |
| `SaveFullSFH` | 0/1 | Track per-snapshot SFR history arrays |
| `NumOutputs` | int | Number of snapshot outputs; `-1` = all snapshots |

### Supported tree formats (`TreeType`)

`lhalo_binary`, `mimic_lhalo_binary`, `lhalo_hdf5`, `consistent_trees_ascii`,
`consistent_trees_hdf5`, `genesis_hdf5`, `gadget4_hdf5`

`mimic_lhalo_binary` reads the L-Halo binary files that Mimic's tree converter writes (an L-Halo header followed by Mimic's 88-byte halo record, with masses in Msun/h). The reader scales masses to SAGE26's code units and sets the record fields the format lacks to -1; see `src/io/read_tree_mimic_lhalo_binary.c`.

---

## Tests

```bash
cd tests && make test               # build and run all suites
cd tests && make test_conservation  # conservation tests only (fastest)
cd tests && make quick              # single fastest check
bash tests/run_integration_tests.sh # full integration test (slower)
```

The regression baseline checks that every one of the 5252 output datasets is
bit-identical to the committed reference. It needs the serial build, and note
that `make tests` relinks `sage` through its `$(EXEC)` dependency, so build in
this order:

```bash
make clean && make USE-MPI=      # serial build
bash tests/regression_baseline.sh
```

After an intentional physics change, recapture the baseline in the same commit
and say so in the message (see
[docs/developer/STYLE_COMMITS.md](docs/developer/STYLE_COMMITS.md)):

```bash
python3 tests/regression_baseline.py capture input/millennium.par
```

See [docs/developer/REGRESSION_BASELINE.md](docs/developer/REGRESSION_BASELINE.md)
for the full policy.

---

## Parameter calibration (SAGE-PSO)

Automated parameter calibration via Particle Swarm Optimization is available as a separate
package: [SAGE-PSO](https://github.com/MBradley1985/SAGE-PSO). It drives `libsage.so` to
evaluate parameter samples against observational constraints (stellar mass functions,
star formation rates, etc.) and supports machine-learning emulators to accelerate
optimization.

---

## Citation

If you use SAGE26 in a publication, please cite:

```bibtex
@article{bradley2026a,
  title = {SAGE26 Paper I: Modelling the Baryon Cycle from Cosmic Dawn to the Present Day},
  author = {Bradley, Michael and Croton, Darren J. and Paun, Robert A. Mostoghiu and Chowdhury, Dhruba Dutta and Willingham, Jayde},
  year = 2026,
  journal = {The Astrophysical Journal Supplement Series},
  publisher = {(In prep.)}
}
```


and the original SAGE paper:

```bibtex
@article{croton2016sage,
  author  = {Croton, D.~J. and Stevens, A.~R.~H. and Tonini, C. and
             Garel, T. and Bernyk, M. and Bibiano, A. and Hodkinson, L. and
             Mutch, S.~J. and Poole, G.~B. and Shattow, G.~M.},
  title   = {Semi-Analytic Galaxy Evolution (SAGE): Model Calibration and Basic Results},
  journal = {ApJS},
  year    = {2016},
  volume  = {222},
  pages   = {22},
  doi     = {10.3847/0067-0049/222/2/22},
  eprint  = {1601.04709},
}
```

Original SAGE is also available on [ascl.net](http://ascl.net/1601.006).

---

## Links

- [Documentation](https://sage26.readthedocs.io/en/latest/)
- [Parameter reference](https://sage26.readthedocs.io/en/latest/parameters.html)
- [Developer guide](docs/developer/README.md)
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
- [SAGE-PSO](https://github.com/MBradley1985/SAGE-PSO) — automated parameter calibration

---

## Authors and maintainers

- Michael Bradley ([@MBradley1985](https://github.com/MBradley1985)) — mbradley@swin.edu.au
- Darren Croton ([@darrencroton](https://github.com/darrencroton))

Questions and comments welcome via GitHub Issues or email.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Acknowledgement

Claude Code (Anthropic) was used during development for documentation and style-guide work, linting, code-error fixes, building the test suite, and some code restructuring. All model design, physics choices, and results remain the authors' own, and all such changes were reviewed and tested before inclusion.

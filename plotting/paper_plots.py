#!/usr/bin/env python
"""
SAGE26 Paper Plots
==================
Publication-quality figures for the SAGE26 paper.

Usage:
    python paper_plots.py              # Generate all plots
    python paper_plots.py 1            # Generate plot 1 only
    python paper_plots.py 1 3 5        # Generate plots 1, 3, 5
"""

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, ScalarFormatter
import os
import numpy as np
import sys
from scipy import interpolate
from scipy import stats
from scipy.integrate import quad
from scipy.ndimage import gaussian_filter
from random import sample, seed
import matplotlib.cm as cm
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
try:
    from astropy.table import Table
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False
    print("Warning: astropy not available, observational data will not be loaded")



# ========================== CONFIGURATION ==========================

# File paths
PRIMARY_DIR = './output/millennium/'
VANILLA_DIR = './output/millennium_vanilla/'
NOFFB_DIR = './output/millennium_noffb/'
NOCGM_DIR = './output/millennium_nocgm/'
C16_FEEDBACK_DIR = './output/millennium_c16feedback/'
GD14_DIR = './output/millennium_gd14/'
KD12_DIR = './output/millennium_kd12/'
KMT09_DIR = './output/millennium_kmt09/'
K13_DIR = './output/millennium_k13/'
FFB_BK25_DIR        = './output/millennium_mbk/'
FFB_BK25_SMOOTH_DIR = './output/millennium_mbk_smooth/'
FFB100_DIR          = './output/millennium_ffb100/'
FFB_BK25_FFB100_DIR = './output/millennium_ffb100_mbk/'
FFB_NOSIGMOID_DIR = './output/millennium_nosigmoid/'
# CGM_DYN_DIR = './output/millennium_cgmdyn/'
# DISK_SMOOTH_DIR = './output/millennium_disk2/'
MINIUCHUU_DIR = './output/microuchuu/'
MODEL_FILE = 'model_0.hdf5'
OBS_DIR = './data/'

# Plotting (analysis choices — not simulation parameters)
OUTPUT_FORMAT = '.pdf'
DILUTE = 7500
SEED = 2222

# Draw order: model 1-sigma bands sit beneath the observations (so they tint
# rather than hide the markers), while the model lines sit on top of them.
Z_MODEL_BAND = 2       # primary model band
Z_MODEL_BAND_ALT = 3   # comparison model band
Z_OBS = 5              # observational markers, error bars and fitted relations
Z_MODEL_LINE = 10      # primary model line
Z_MODEL_LINE_ALT = 11  # comparison model line

# Analysis thresholds (not simulation parameters)
MIN_PARTICLES = 20     # minimum DM particles for a resolved halo (applied at load time)
SSFR_CUT = -11.0       # log10(sSFR/yr^-1) dividing quiescent from star-forming

# Solar metallicity (Asplund et al. 2009)
Z_SUN = 0.0134

# IMF convention.
#
# SAGE26's RecycleFraction of 0.43 is a Chabrier instantaneous return fraction
# (Salpeter gives ~0.3), so every model SFR and stellar mass in this module is on
# a Chabrier scale.  Observational compilations quoted for a Salpeter IMF must
# therefore be shifted DOWN by this amount before being compared with the model,
# never up: a Salpeter fit converts the same light into ~1.7x more stellar mass.
#
# Getting the sign wrong on one dataset and not another puts two observational
# curves on the same axes 0.2 dex apart, which is how a real ~0.2 dex model
# excess at cosmic noon came to look like agreement with Madau & Dickinson and a
# disagreement with COSMOS-Web (which is natively Chabrier and needs no shift).
SALPETER_TO_CHABRIER_DEX = -0.24

# Solar mass in grams (for MASS_CONVERT derivation)
_MSUN_CGS = 1.989e33


# --------------- HDF5 header reader ---------------

import glob as _glob_early


def _find_model_files_early(directory):
    """Minimal file discovery used during module init (before full I/O helpers)."""
    pattern = os.path.join(directory, 'model_*.hdf5')
    files = sorted(_glob_early.glob(pattern))
    if not files:
        single = os.path.join(directory, MODEL_FILE)
        if os.path.exists(single):
            files = [single]
    return files


def _read_sim_header(directory):
    """
    Read simulation parameters from the HDF5 header of the first model
    file found in *directory*.

    Returns a dict of parameters, or ``None`` if no model files exist.
    The ``volume_fraction`` key is the *total* fraction across all MPI
    files (summed ``frac_volume_processed``).
    """
    files = _find_model_files_early(directory)
    if not files:
        return None

    try:
        with h5.File(files[0], 'r') as f:
            sim = f['Header/Simulation']
            runtime = f['Header/Runtime']

            header = {
                'hubble_h':       float(sim.attrs['hubble_h']),
                'box_size':       float(sim.attrs['box_size']),
                'omega_matter':   float(sim.attrs['omega_matter']),
                'omega_lambda':   float(sim.attrs['omega_lambda']),
                'last_snap_nr':   int(sim.attrs['LastSnapshotNr']),
                'unit_mass_in_g': float(runtime.attrs['UnitMass_in_g']),
                'baryon_frac':    float(runtime.attrs.get('BaryonFrac', 0.17)),
                'redshifts':      list(f['Header/snapshot_redshifts'][:]),
                'output_snaps':   list(f['Header/output_snapshots'][:]),
            }

        # Sum frac_volume_processed across all MPI files to get the total
        total_fvp = 0.0
        for fp in files:
            with h5.File(fp, 'r') as f:
                total_fvp += float(f['Header/Runtime'].attrs['frac_volume_processed'])
        header['volume_fraction'] = total_fvp
    except Exception as e:
        print(f"Warning: could not read header from {directory}: {e}")
        return None

    return header


def _snap_for_z(redshifts, target_z):
    """
    Return the snapshot index of the last output snapshot whose redshift
    is >= *target_z*.  This reproduces the standard convention of choosing
    the snapshot just above the target redshift (e.g. z=4.179 for target 4).
    """
    neg_z = -np.array(redshifts)          # make increasing for searchsorted
    idx = int(np.searchsorted(neg_z, -target_z, side='right')) - 1
    return max(idx, 0)


def _snap_nearest_z(redshifts, target_z):
    """
    Return the snapshot index whose redshift is *closest* to *target_z*,
    from either side.  Use this where panels are labelled by the round
    target redshift, so the snapshot sits as near to it as the output
    table allows.
    """
    return int(np.argmin(np.abs(np.array(redshifts) - target_z)))


# --------------- Primary simulation parameters (from HDF5) ---------------

_primary_hdr = _read_sim_header(PRIMARY_DIR)
if _primary_hdr is not None:
    HUBBLE_H         = _primary_hdr['hubble_h']
    BOX_SIZE         = _primary_hdr['box_size']
    VOLUME_FRACTION  = _primary_hdr['volume_fraction']
    VOLUME           = (BOX_SIZE / HUBBLE_H)**3 * VOLUME_FRACTION  # Mpc^3
    MASS_CONVERT     = _primary_hdr['unit_mass_in_g'] / _MSUN_CGS / HUBBLE_H
    OMEGA_M          = _primary_hdr['omega_matter']
    OMEGA_L          = _primary_hdr['omega_lambda']
    BARYON_FRAC      = _primary_hdr['baryon_frac']
    OMEGA_B          = BARYON_FRAC * OMEGA_M
    SNAPSHOT         = f"Snap_{_primary_hdr['last_snap_nr']}"
    REDSHIFTS        = _primary_hdr['redshifts']
    OUTPUT_DIR       = os.path.join(PRIMARY_DIR, 'plots/')

    # Snapshot aliases for key redshifts (derived from the redshift table)
    SNAP_Z0  = _snap_for_z(REDSHIFTS, 0.0)
    SNAP_Z1  = _snap_for_z(REDSHIFTS, 1.0)
    SNAP_Z2  = _snap_for_z(REDSHIFTS, 2.0)
    SNAP_Z3  = _snap_for_z(REDSHIFTS, 3.0)
    SNAP_Z4  = _snap_for_z(REDSHIFTS, 4.0)
    SNAP_Z5  = _snap_for_z(REDSHIFTS, 5.0)
    SNAP_Z7  = _snap_for_z(REDSHIFTS, 7.0)
    SNAP_Z10 = _snap_for_z(REDSHIFTS, 10.0)
else:
    # Fallback if primary HDF5 files are not available
    print("Warning: could not read primary model header — using hardcoded defaults")
    HUBBLE_H         = 0.73
    BOX_SIZE         = 62.5
    VOLUME_FRACTION  = 1.0
    VOLUME           = (BOX_SIZE / HUBBLE_H)**3 * VOLUME_FRACTION
    MASS_CONVERT     = 1.0e10 / HUBBLE_H
    OMEGA_M          = 0.25
    OMEGA_L          = 0.75
    BARYON_FRAC      = 0.17
    OMEGA_B          = 0.045
    SNAPSHOT         = 'Snap_63'
    REDSHIFTS        = [
        127.000, 79.998, 50.000, 30.000, 19.916, 18.244, 16.725, 15.343,
         14.086, 12.941, 11.897, 10.944, 10.073,  9.278,  8.550,  7.883,
          7.272,  6.712,  6.197,  5.724,  5.289,  4.888,  4.520,  4.179,
          3.866,  3.576,  3.308,  3.060,  2.831,  2.619,  2.422,  2.239,
          2.070,  1.913,  1.766,  1.630,  1.504,  1.386,  1.276,  1.173,
          1.078,  0.989,  0.905,  0.828,  0.755,  0.687,  0.624,  0.564,
          0.509,  0.457,  0.408,  0.362,  0.320,  0.280,  0.242,  0.208,
          0.175,  0.144,  0.116,  0.089,  0.064,  0.041,  0.020,  0.000,
    ]
    OUTPUT_DIR = './output/millennium/plots/'
    SNAP_Z0  = 63
    SNAP_Z1  = 39
    SNAP_Z2  = 32
    SNAP_Z3  = 27
    SNAP_Z4  = 23
    SNAP_Z5  = 20
    SNAP_Z7  = 16
    SNAP_Z10 = 12


# --------------- miniUchuu simulation parameters (from HDF5) ---------------

_miniuchuu_hdr = _read_sim_header(MINIUCHUU_DIR)
if _miniuchuu_hdr is not None:
    MINIUCHUU_HUBBLE_H        = _miniuchuu_hdr['hubble_h']
    MINIUCHUU_BOX_SIZE        = _miniuchuu_hdr['box_size']
    MINIUCHUU_VOLUME_FRACTION = _miniuchuu_hdr['volume_fraction']
    MINIUCHUU_VOLUME          = (MINIUCHUU_BOX_SIZE / MINIUCHUU_HUBBLE_H)**3 * MINIUCHUU_VOLUME_FRACTION
    MINIUCHUU_MASS_CONVERT    = _miniuchuu_hdr['unit_mass_in_g'] / _MSUN_CGS / MINIUCHUU_HUBBLE_H
    MINIUCHUU_FIRST_SNAP      = min(_miniuchuu_hdr['output_snaps'])
    MINIUCHUU_LAST_SNAP       = max(_miniuchuu_hdr['output_snaps'])
    MINIUCHUU_REDSHIFTS       = _miniuchuu_hdr['redshifts']
else:
    # Fallback if miniUchuu HDF5 files are not available
    MINIUCHUU_HUBBLE_H        = 0.677
    MINIUCHUU_BOX_SIZE        = 400.0
    MINIUCHUU_VOLUME_FRACTION = 0.3
    MINIUCHUU_VOLUME          = (MINIUCHUU_BOX_SIZE / MINIUCHUU_HUBBLE_H)**3 * MINIUCHUU_VOLUME_FRACTION
    MINIUCHUU_MASS_CONVERT    = 1.0e10 / MINIUCHUU_HUBBLE_H
    MINIUCHUU_FIRST_SNAP      = 0
    MINIUCHUU_LAST_SNAP       = 49
    MINIUCHUU_REDSHIFTS       = [
        13.9334, 12.67409, 11.50797, 10.44649, 9.480752, 8.58543, 7.77447,
        7.032387, 6.344409, 5.721695, 5.153127, 4.629078, 4.26715, 3.929071,
        3.610462, 3.314082, 3.128427, 2.951226, 2.77809, 2.616166, 2.458114,
        2.309724, 2.16592, 2.027963, 1.8962, 1.770958, 1.65124, 1.535928,
        1.426272, 1.321656, 1.220303, 1.124166, 1.031983, 0.9441787, 0.8597281,
        0.779046, 0.7020205, 0.6282588, 0.5575475, 0.4899777, 0.4253644,
        0.3640053, 0.3047063, 0.2483865, 0.1939743, 0.1425568, 0.09296665,
        0.0455745, 0.02265383, 0.0001130128,
    ]

# FFB model variants (different max star-formation efficiencies)
FFB_MODELS = [
    {'name': r'FFB 10\%',  'dir': './output/millennium_ffb10/',  'sfe': 0.10},
    {'name': r'FFB 20\%',  'dir': './output/millennium_ffb20/',  'sfe': 0.20},
    {'name': r'FFB 30\%',  'dir': './output/millennium_ffb30/',  'sfe': 0.30},
    {'name': r'FFB 40\%',  'dir': './output/millennium_ffb40/',  'sfe': 0.40},
    {'name': r'FFB 50\%',  'dir': './output/millennium_ffb50/',  'sfe': 0.50},
    {'name': r'FFB 60\%',  'dir': './output/millennium_ffb60/',  'sfe': 0.60},
    {'name': r'FFB 70\%',  'dir': './output/millennium_ffb70/',  'sfe': 0.70},
    {'name': r'FFB 80\%',  'dir': './output/millennium_ffb80/',  'sfe': 0.80},
    {'name': r'FFB 90\%',  'dir': './output/millennium_ffb90/',  'sfe': 0.90},
    {'name': r'FFB 100\%', 'dir': './output/millennium_ffb100/', 'sfe': 1.00},
]

# Properties stored in HDF5 mass units (need MASS_CONVERT)
_MASS_PROPS = frozenset({
    'CentralMvir', 'Mvir', 'StellarMass', 'BulgeMass', 'BlackHoleMass',
    'MetalsStellarMass', 'MetalsColdGas', 'MetalsEjectedMass',
    'MetalsHotGas', 'MetalsCGMgas', 'ColdGas', 'HotGas', 'CGMgas',
    'EjectedMass', 'H2gas', 'H1gas', 'IntraClusterStars',
    'MergerBulgeMass', 'InstabilityBulgeMass',
})

# Default properties to load for the primary model
_DEFAULT_PROPERTIES = [
    'StellarMass', 'BulgeMass', 'ColdGas', 'HotGas', 'CGMgas',
    'EjectedMass', 'H2gas', 'H1gas', 'BlackHoleMass',
    'IntraClusterStars', 'CentralMvir', 'Mvir',
    'MergerBulgeMass', 'InstabilityBulgeMass',
    'MetalsStellarMass', 'MetalsColdGas', 'MetalsHotGas',
    'MetalsEjectedMass', 'MetalsCGMgas',
    'SfrDisk', 'SfrBulge', 'Vvir', 'Vmax', 'Rvir',
    'DiskRadius', 'BulgeRadius',
    'Type', 'CentralGalaxyIndex',
    'Posx', 'Posy', 'Posz',
    'OutflowRate', 'MassLoading', 'Cooling', 'Regime', 'CoolingRate'
]

# Properties to load for evolution (multi-snapshot) plots
_EVOLUTION_PROPERTIES = [
    'StellarMass', 'SfrDisk', 'SfrBulge', 'Mvir', 'Rvir',
    'CGMgas', 'HotGas', 'MetalsStellarMass', 'DiskRadius', 'BulgeRadius',
    'CoolingRate',
    'FFBRegime', 'Regime', 'tcool_over_tff', 'tdeplete', 'tff',
    'GalaxyIndex', 'Type',
]


# ========================== PLOTTING STYLE ==========================

def setup_style():
    """Configure matplotlib for publication-quality white-background plots."""
    plt.style.use("./plotting/kieren_cohare_palatino_sty.mplstyle")


def _tex_safe(s):
    """Make label strings safe for both usetex and non-usetex modes."""
    if not plt.rcParams.get('text.usetex', False):
        s = s.replace(r"\'{e}", "\u00e9")   # é
        s = s.replace(r'\&', '&')
    return s


# ========================== DATA I/O ==========================


def find_model_files(directory):
    """
    Find all model_*.hdf5 files in *directory*.

    Returns a sorted list of absolute paths.  Falls back to the single
    ``model_0.hdf5`` if no files match (backward-compatible).
    """
    return _find_model_files_early(directory)


def model_files_exist(directory):
    """Return True if at least one model HDF5 file exists in *directory*."""
    return len(find_model_files(directory)) > 0


def read_snap_from_files(filepaths, snap_key, properties, mass_convert=MASS_CONVERT):
    """
    Read *properties* from *snap_key* across multiple HDF5 files and
    concatenate the results.

    Parameters
    ----------
    filepaths : list of str
        HDF5 file paths (e.g. from ``find_model_files``).
    snap_key : str
        Snapshot group name, e.g. ``'Snap_63'``.
    properties : list of str
        Dataset names to read.
    mass_convert : float
        Multiplicative factor applied to properties in ``_MASS_PROPS``.

    Returns
    -------
    dict : property name -> numpy array (concatenated across files).
           Empty dict if no file contains *snap_key*.
    """
    caller_wants_len = 'Len' in properties
    load_props = list(properties) if caller_wants_len else list(properties) + ['Len']

    chunks = {prop: [] for prop in load_props}
    found_snap = False

    for fp in filepaths:
        try:
            with h5.File(fp, 'r') as f:
                if snap_key not in f:
                    continue
                found_snap = True
                grp = f[snap_key]
                for prop in load_props:
                    if prop in grp:
                        chunks[prop].append(np.array(grp[prop]))
        except Exception as e:
            print(f"  Warning: could not read {fp}: {e}")
            continue

    if not found_snap:
        return {}

    data = {}
    for prop in load_props:
        if chunks[prop]:
            arr = np.concatenate(chunks[prop])
            if prop in _MASS_PROPS:
                arr = arr * mass_convert
            data[prop] = arr

    if 'Len' in data:
        mask = data['Len'] >= MIN_PARTICLES
        data = {p: arr[mask] for p, arr in data.items()}

    if not caller_wants_len:
        data.pop('Len', None)

    return data


def load_model(directory, filename=None, snapshot=SNAPSHOT,
               properties=None):
    """
    Load galaxy properties from one or more model HDF5 files.

    When SAGE is run with MPI each rank writes its own file
    (``model_0.hdf5``, ``model_1.hdf5``, …).  This function automatically
    discovers all such files and concatenates their datasets.

    Parameters
    ----------
    directory : str
        Path to the model output directory.
    filename : str, optional
        Kept for backward compatibility.  If given, only that single file
        is read; otherwise every ``model_*.hdf5`` in *directory* is used.
    snapshot : str
        Snapshot key (e.g. ``'Snap_63'``).
    properties : list of str, optional
        Properties to load.  If *None*, loads ``_DEFAULT_PROPERTIES``.

    Returns
    -------
    dict : property name -> numpy array (converted where applicable).
    """
    if properties is None:
        properties = _DEFAULT_PROPERTIES

    if filename is not None:
        filepaths = [os.path.join(directory, filename)]
    else:
        filepaths = find_model_files(directory)

    if not filepaths:
        print(f"  Warning: no model files found in {directory}")
        return {}

    data = read_snap_from_files(filepaths, snapshot, properties)
    if not data:
        print(f"  Warning: {snapshot} not found in any file in {directory}")
    return data


def load_snapshots(directory, snaps, properties=None, filename=None):
    """
    Load multiple snapshots from one or more HDF5 files.

    Parameters
    ----------
    directory : str
        Path to model output directory.
    snaps : list of int
        Snapshot numbers to load.
    properties : list of str, optional
        Properties to load.  Defaults to ``_EVOLUTION_PROPERTIES``.
    filename : str, optional
        If given, only that single file is read; otherwise every
        ``model_*.hdf5`` in *directory* is used.

    Returns
    -------
    dict : {snap_num: {prop_name: numpy array}}
    """
    if properties is None:
        properties = _EVOLUTION_PROPERTIES

    if filename is not None:
        filepaths = [os.path.join(directory, filename)]
    else:
        filepaths = find_model_files(directory)

    if not filepaths:
        print(f"  Warning: no model files found in {directory}")
        return {}

    snapdata = {}
    for snap in snaps:
        snap_key = f'Snap_{snap}'
        data = read_snap_from_files(filepaths, snap_key, properties)
        if data:
            snapdata[snap] = data
        else:
            print(f"  Warning: {snap_key} not found, skipping.")

    return snapdata


# ========================== COMPUTATION UTILITIES ==========================

def calculate_muratov_mass_loading(vvir, z=0.0):
    """
    Calculate mass loading factor using Muratov et al. (2015) formulation
    Vectorized for better performance
    """
    # Constants from Muratov et al. (2015) and SAGE implementation
    V_CRIT = 60.0      # Critical velocity where the power law breaks
    NORM = 2.9         # Normalization factor
    Z_EXP = 1.3        # Redshift power-law exponent
    LOW_V_EXP = -3.2   # Low velocity power-law exponent
    HIGH_V_EXP = -1.0  # High velocity power-law exponent
    
    # Vectorized calculation for better performance
    z_term = np.power(1.0 + z, Z_EXP)
    v_ratio = vvir / V_CRIT
    
    # Vectorized broken power law
    v_term = np.where(vvir < V_CRIT, 
                      np.power(v_ratio, LOW_V_EXP),
                      np.power(v_ratio, HIGH_V_EXP))
    
    # Calculate final mass loading factor
    eta = NORM * z_term * v_term
    
    # Vectorized capping and finite value handling
    eta = np.clip(eta, 0.0, 100.0)
    eta = np.where(np.isfinite(eta), eta, 0.0)
    
    return eta

def mass_function(log_masses, volume, binwidth=0.1, mass_range=None):
    """
    Compute a mass function (log10 number density per dex per Mpc^3).

    Parameters
    ----------
    log_masses : array
        log10 masses.
    volume : float
        Comoving volume in Mpc^3.
    binwidth : float
        Bin width in dex.
    mass_range : tuple of (float, float), optional
        (min, max) for histogram. Auto-determined if None.

    Returns
    -------
    centers : array
        Bin centres.
    phi : array
        log10(number density). NaN where counts == 0.
    mrange : tuple
        (min, max) used, so subsets can reuse the same bins.
    """
    if mass_range is None:
        mi = np.floor(np.min(log_masses)) - 2
        ma = np.floor(np.max(log_masses)) + 2
    else:
        mi, ma = mass_range

    nbins = int(round((ma - mi) / binwidth))
    counts, edges = np.histogram(log_masses, range=(mi, ma), bins=nbins)
    centers = edges[:-1] + 0.5 * binwidth

    with np.errstate(divide='ignore'):
        phi = np.log10(counts / volume / binwidth)
    phi[~np.isfinite(phi)] = np.nan

    return centers, phi, (mi, ma)


def mass_function_bootstrap(log_masses, volume, binwidth=0.1, mass_range=None,
                            n_boot=100):
    """
    Compute a mass function with bootstrap confidence intervals.

    Returns
    -------
    centers : array
        Bin centres.
    phi : array
        log10(number density).
    phi_lo : array
        16th percentile (lower bound).
    phi_hi : array
        84th percentile (upper bound).
    mrange : tuple
        (min, max) used.
    """
    # First compute the main mass function to get bin edges
    centers, phi, mrange = mass_function(log_masses, volume, binwidth, mass_range)

    n_gal = len(log_masses)
    if n_gal == 0:
        return centers, phi, phi, phi, mrange

    mi, ma = mrange
    nbins = int(round((ma - mi) / binwidth))

    # Bootstrap resampling
    boot_phi = np.full((n_boot, len(centers)), np.nan)
    for b in range(n_boot):
        idx = np.random.randint(0, n_gal, n_gal)
        boot_masses = log_masses[idx]
        counts, _ = np.histogram(boot_masses, range=(mi, ma), bins=nbins)
        with np.errstate(divide='ignore'):
            vals = np.log10(counts / volume / binwidth)
        # Replace -inf (empty bins) with nan so percentiles ignore them
        boot_phi[b, :] = np.where(np.isfinite(vals), vals, np.nan)

    # Compute percentiles (nanpercentile skips nan, so empty-bin samples are excluded)
    with np.errstate(invalid='ignore'):
        phi_lo = np.nanpercentile(boot_phi, 16, axis=0)
        phi_hi = np.nanpercentile(boot_phi, 84, axis=0)

    return centers, phi, phi_lo, phi_hi, mrange


def metallicity_12logOH(metals_cold_gas, cold_gas):
    """
    Gas-phase metallicity in 12 + log10(O/H).

    Uses Z_cold = MetalsColdGas / ColdGas, solar reference Z_sun = 0.02,
    and 12 + log10(O/H)_sun = 9.0.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log10((metals_cold_gas / cold_gas) / 0.02) + 9.0


def stellar_metallicity(metals_stellar_mass, stellar_mass):
    """
    Stellar metallicity log10(Z/Z_sun).
    Uses Z_star = MetalsStellarMass / StellarMass, solar reference Z_sun = 0.02.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log10((metals_stellar_mass / stellar_mass) / 0.02)


def log_ssfr(sfr_disk, sfr_bulge, stellar_mass):
    """Compute log10(sSFR / yr^-1)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log10((sfr_disk + sfr_bulge) / stellar_mass)


def binned_median(x, y, bins, min_count=5):
    """Binned median with 25th/75th percentiles."""
    centers = 0.5 * (bins[:-1] + bins[1:])
    n = len(bins) - 1
    med = np.full(n, np.nan)
    p25 = np.full(n, np.nan)
    p75 = np.full(n, np.nan)

    for i in range(n):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        count = np.sum(mask)
        if count >= min_count:
            vals = y[mask]
            med[i] = np.median(vals)
            p25[i] = np.percentile(vals, 25)
            p75[i] = np.percentile(vals, 75)

    return centers, med, p25, p75


def binned_percentiles(x, y, bins, percentiles=(16, 50, 84), min_count=20):
    """Compute binned percentiles of *y* as a function of *x*.

    Parameters
    ----------
    x, y : array-like
        Data arrays.
    bins : array-like
        Bin edges in x.
    percentiles : tuple
        Percentiles to compute (e.g. (16, 50, 84)).
    min_count : int
        Minimum number of points required in a bin.

    Returns
    -------
    centers : array
        Bin centers.
    pct : array, shape (len(percentiles), nbins)
        Percentiles per bin; NaN for bins with insufficient counts.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    centers = 0.5 * (bins[:-1] + bins[1:])
    nbins = len(bins) - 1
    pct = np.full((len(percentiles), nbins), np.nan)

    for i in range(nbins):
        m = (x >= bins[i]) & (x < bins[i + 1])
        if np.sum(m) >= min_count:
            pct[:, i] = np.percentile(y[m], percentiles)

    return centers, pct


def plot_binned_median_1sigma(
    ax,
    x,
    y,
    bins,
    *,
    color,
    label,
    alpha=0.25,
    lw=3.0,
    ls='-',
    min_count=20,
    zorder_fill=3,
    zorder_line=4,
):
    """Plot a median line with a 16--84% (1\u03c3) shaded band."""
    centers, pct = binned_percentiles(x, y, bins, percentiles=(16, 50, 84), min_count=min_count)
    p16, p50, p84 = pct
    valid = np.isfinite(p50) & np.isfinite(p16) & np.isfinite(p84)
    if not np.any(valid):
        return None

    ax.fill_between(centers[valid], p16[valid], p84[valid],
                    color=color, alpha=alpha, lw=0.0, zorder=zorder_fill)
    (line,) = ax.plot(centers[valid], p50[valid],
                      color=color, lw=lw, ls=ls, label=label, zorder=zorder_line)
    return line


def density_contour(x, y, bins=100, weights=None, smooth=1.5):
    """
    Generate a 2D density map for contour plotting.

    Parameters
    ----------
    x, y : array-like
        The x and y coordinates of the data points.
    bins : int or [int, int]
        The number of bins in each dimension.
    weights : array-like, optional
        An array of weights for each point.
    smooth : float or None
        Gaussian smoothing sigma in bin units. None to disable.

    Returns
    -------
    X, Y : array-like
        The coordinates of the bin centers.
    Z : array-like
        The 2D density map (or weighted counts).
    """
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=weights)

    if smooth:
        H = gaussian_filter(H, sigma=smooth)

    # Convert to bin centers
    X = 0.5 * (xedges[:-1] + xedges[1:])
    Y = 0.5 * (yedges[:-1] + yedges[1:])

    # The histogram needs to be transposed for contour plotting
    return X, Y, H.T


def sigma_contour_levels(Z):
    """
    Compute density thresholds enclosing 1-3 sigma of a 2D distribution.

    For a 2D distribution the fraction enclosed within N sigma is
    f(N) = 1 - exp(-N^2 / 2).

    Returns levels ordered [3sigma, 2sigma, 1sigma, Z_max]
    suitable for direct use in contourf (ascending density).
    """
    flat = Z.flatten()
    flat = flat[flat > 0]
    if len(flat) == 0:
        return None
    sorted_Z = np.sort(flat)[::-1]
    cumsum = np.cumsum(sorted_Z) / np.sum(sorted_Z)

    fractions = [1 - np.exp(-0.5 * n**2) for n in [3, 2, 1]]
    levels = []
    for f in fractions:
        idx = np.searchsorted(cumsum, f)
        idx = min(idx, len(sorted_Z) - 1)
        levels.append(sorted_Z[idx])
    levels.append(Z.max())
    return levels


def baryon_fractions_by_halo_mass(primary, halo_bins=None):
    """
    Compute mean baryon component fractions binned by halo mass.

    Uses np.bincount to sum components per halo in O(N), avoiding
    per-halo Python loops.

    Returns
    -------
    mass_centers : array
        Mean log10(Mvir) in each occupied bin.
    results : dict
        {component_name: {'mean': array, 'upper': array, 'lower': array}}
    """
    if halo_bins is None:
        halo_bins = np.arange(11.0, 16.1, 0.1)

    cgi = primary['CentralGalaxyIndex'].astype(np.int64)

    # Remap CentralGalaxyIndex IDs to compact 0-based group indices
    unique_ids, compact_idx = np.unique(cgi, return_inverse=True)
    ngroups = len(unique_ids)

    # Components to track
    comp_keys = ['StellarMass', 'ColdGas', 'HotGas', 'CGMgas',
                 'IntraClusterStars', 'BlackHoleMass', 'EjectedMass']

    # Sum each component by halo using bincount — O(N), fully vectorized
    halo_sums = {}
    for key in comp_keys:
        halo_sums[key] = np.bincount(compact_idx, weights=primary[key],
                                     minlength=ngroups)
    halo_sums['Total'] = sum(halo_sums[k] for k in comp_keys)

    # Central galaxies define halos
    central_mask = primary['Type'] == 0
    central_compact = compact_idx[central_mask]
    mvir = primary['Mvir'][central_mask]
    log_mvir = np.log10(mvir)

    # Fractions: component_sum / Mvir for each halo
    fractions = {}
    all_keys = ['Total'] + comp_keys
    for key in all_keys:
        fractions[key] = halo_sums[key][central_compact] / mvir

    # Bin by halo mass and compute mean +/- stderr
    bin_idx = np.digitize(log_mvir, halo_bins) - 1
    results = {k: {'mean': [], 'upper': [], 'lower': []} for k in all_keys}
    mass_centers = []

    for i in range(len(halo_bins) - 1):
        w = bin_idx == i
        n_halos = np.sum(w)
        if n_halos < 3:
            continue

        mass_centers.append(np.mean(log_mvir[w]))
        sqrt_n = np.sqrt(n_halos)

        for key in all_keys:
            vals = fractions[key][w]
            mean = np.mean(vals)
            err = np.std(vals) / sqrt_n
            results[key]['mean'].append(mean)
            results[key]['upper'].append(mean + err)
            results[key]['lower'].append(max(mean - err, 1e-6))

    # Convert to arrays
    mass_centers = np.array(mass_centers)
    for key in results:
        for stat in results[key]:
            results[key][stat] = np.array(results[key][stat])

    return mass_centers, results


def baryon_fractions_by_halo_mass_vanilla(vanilla, halo_bins=None):
    """
    Compute mean baryon component fractions binned by halo mass.

    Uses np.bincount to sum components per halo in O(N), avoiding
    per-halo Python loops.

    Returns
    -------
    mass_centers : array
        Mean log10(Mvir) in each occupied bin.
    results : dict
        {component_name: {'mean': array, 'upper': array, 'lower': array}}
    """
    if halo_bins is None:
        halo_bins = np.arange(11.0, 16.1, 0.1)

    cgi = vanilla['CentralGalaxyIndex'].astype(np.int64)

    # Remap CentralGalaxyIndex IDs to compact 0-based group indices
    unique_ids, compact_idx = np.unique(cgi, return_inverse=True)
    ngroups = len(unique_ids)

    # Components to track
    comp_keys = ['StellarMass', 'ColdGas', 'HotGas',
                 'IntraClusterStars', 'BlackHoleMass', 'EjectedMass']

    # Sum each component by halo using bincount — O(N), fully vectorized
    halo_sums = {}
    for key in comp_keys:
        halo_sums[key] = np.bincount(compact_idx, weights=vanilla[key],
                                     minlength=ngroups)
    halo_sums['Total'] = sum(halo_sums[k] for k in comp_keys)

    # Central galaxies define halos
    central_mask = vanilla['Type'] == 0
    central_compact = compact_idx[central_mask]
    mvir = vanilla['Mvir'][central_mask]
    log_mvir = np.log10(mvir)

    # Fractions: component_sum / Mvir for each halo
    fractions = {}
    all_keys = ['Total'] + comp_keys
    for key in all_keys:
        fractions[key] = halo_sums[key][central_compact] / mvir

    # Bin by halo mass and compute mean +/- stderr
    bin_idx = np.digitize(log_mvir, halo_bins) - 1
    results = {k: {'mean': [], 'upper': [], 'lower': []} for k in all_keys}
    mass_centers = []

    for i in range(len(halo_bins) - 1):
        w = bin_idx == i
        n_halos = np.sum(w)
        if n_halos < 3:
            continue

        mass_centers.append(np.mean(log_mvir[w]))
        sqrt_n = np.sqrt(n_halos)

        for key in all_keys:
            vals = fractions[key][w]
            mean = np.mean(vals)
            err = np.std(vals) / sqrt_n
            results[key]['mean'].append(mean)
            results[key]['upper'].append(mean + err)
            results[key]['lower'].append(max(mean - err, 1e-6))

    # Convert to arrays
    mass_centers = np.array(mass_centers)
    for key in results:
        for stat in results[key]:
            results[key][stat] = np.array(results[key][stat])

    return mass_centers, results


def snap_to_redshift(snap):
    """Return the redshift for a given snapshot number."""
    return REDSHIFTS[snap]


def cosmic_time_gyr(z):
    """Age of the universe at redshift z, in Gyr."""
    t_H = 977.8 / (HUBBLE_H * 100)  # Hubble time in Gyr

    def integrand(zp):
        return 1.0 / ((1 + zp) * np.sqrt(OMEGA_M * (1 + zp)**3 + OMEGA_L))

    result, _ = quad(integrand, z, 1000.0)
    return t_H * result


PRECIP_THRESHOLD = 10.0     # matches PRECIP_THRESHOLD in model_cooling_heating.c
PRECIP_WIDTH     = 2.0      # matches PRECIP_TRANSITION_WIDTH


def precipitation_fraction(tcool_over_tff, include_condensation=True):
    """Effective inflow fraction f_inflow as implemented in SAGE26.

    The model condenses only the CGM mass in excess of the marginally stable
    reservoir m_eq = m_CGM (t_cool/t_ff) / threshold, so the rate is

        mdot = S((threshold - r)/width) * (m_CGM - m_eq) / t_ff
             = [ S((threshold - r)/width) * max(0, 1 - r/threshold) ] * m_CGM / t_ff

    and the bracketed quantity is the effective inflow fraction.  It reaches
    0.9 at r = 0.90 and is identically zero for r >= threshold.

    include_condensation=False returns the bare sigmoid only.  That is the form
    printed as Eq. 5 in the first submission, which omitted the condensation
    term; it is retained so the two can be shown side by side.
    """
    ratio = np.atleast_1d(np.array(tcool_over_tff, dtype=float))
    x_sig = (PRECIP_THRESHOLD - ratio) / PRECIP_WIDTH
    f = 1.0 / (1.0 + np.exp(-np.clip(x_sig, -700.0, 700.0)))
    if include_condensation:
        f = f * np.clip(1.0 - ratio / PRECIP_THRESHOLD, 0.0, None)
    return f.squeeze()


def ffb_threshold_mass_msun(z):
    """FFB threshold mass from Li et al. (2024) Eq. 2."""
    z_norm = (1.0 + z) / 10.0
    log_M_code = 0.8 + np.log10(HUBBLE_H) - 6.2 * np.log10(z_norm)
    return 10.0**log_M_code * 1.0e10 / HUBBLE_H


def ffb_fraction(Mvir_msun, z, delta_log_M=0.15):
    """
    Theoretical FFB fraction as a logistic sigmoid at the threshold mass.

    Matches the C implementation: f = 1 / (1 + exp(-x))
    where x = log10(Mvir / Mvir_ffb) / delta_log_M.
    """
    M_thresh = ffb_threshold_mass_msun(z)
    x = (np.log10(Mvir_msun) - np.log10(M_thresh)) / delta_log_M
    return 1.0 / (1.0 + np.exp(-x))


# -------- MBK25 (Boylan-Kolchin 2025) theoretical FFB fraction --------

try:
    from colossus.cosmology import cosmology as _colossus_cosmology
    from colossus.halo import concentration as _colossus_conc
    _colossus_cosmology.setCosmology('custom_millennium', flat=True,
                                     H0=73.0, Om0=OMEGA_M, Ob0=0.045,
                                     sigma8=0.90, ns=1.0, relspecies=False)
    _HAS_COLOSSUS = True
except Exception:
    _HAS_COLOSSUS = False


def _delta_vir_bn98(z):
    """Halo overdensity relative to rho_crit(z), matching the model.

    SAGE defines R_vir with DELTA_VIRT = 200 rho_crit (model_halo_properties.c),
    and the Ishiyama+21 concentration table it reads is mdef=200c.  The MBK25
    criterion combines R_vir with c through c^2/(2 mu(c)), so both must use the
    same definition; a Bryan & Norman virial overdensity here (the previous
    behaviour) mixed a BN98 R_vir with a 200c concentration and biased the
    threshold mass.  Kept under the original name so existing callers are
    unaffected.
    """
    return 200.0 + 0.0 * np.asarray(z, dtype=float)


def _rvir_m(Mvir_msun, z):
    """Virial radius [m] from M_vir [M_sun] using Bryan & Norman overdensity."""
    H0_si = HUBBLE_H * 1.0e5 / 3.085678e22   # H_0 in s^-1
    Ez = np.sqrt(OMEGA_M * (1.0 + z)**3 + OMEGA_L)
    rho_crit = 3.0 * (H0_si * Ez)**2 / (8.0 * np.pi * 6.674e-11)
    delta = _delta_vir_bn98(z)
    return (3.0 * Mvir_msun * 1.989e30 / (4.0 * np.pi * delta * rho_crit))**(1.0 / 3.0)


# g_crit = G * 3100 M_sun / pc^2 (BK25 Table 1)
_G_CRIT_SI = 6.674e-11 * 3100.0 * 1.989e30 / (3.085678e16)**2


def _c_ishiyama21(Mvir_msun, z):
    """
    Mean Ishiyama+21 concentration (200c) for an array of M_vir [M_sun] at redshift z.
    Falls back to a Bullock+01 power-law if colossus is not available.
    """
    M_h = np.asarray(Mvir_msun) * HUBBLE_H   # M_sun/h, as colossus expects
    if _HAS_COLOSSUS:
        try:
            c = _colossus_conc.concentration(M_h, '200c', z, model='ishiyama21')
            return np.maximum(np.atleast_1d(np.asarray(c, dtype=float)), 1.0)
        except Exception:
            pass
    # Fallback power-law approximation (Bullock+01 style)
    c = 9.0 / (1.0 + z) * (M_h / 1.0e12)**(-0.13)
    return np.maximum(c, 1.0)


def ffb_fraction_mbk25(Mvir_msun, z, sigma_c=0.2):
    """
    MBK25 FFB fraction via the Boylan-Kolchin 2025 maximum-acceleration criterion.

    g_max = G M_vir / R_vir^2 * c^2 / (2 mu(c)),  mu(c) = ln(1+c) - c/(1+c)
    FFB when g_max > g_crit = G * 3100 M_sun / pc^2  (BK25 Table 1).

    With sigma_c > 0, concentration scatters log-normally around the Ishiyama+21
    mean (matching FeedbackFreeModeOn=4 in the C code):
        f_ffb(M, z) = P(c > c_thresh) = norm.sf((ln c_thresh - ln c_mean) / sigma_c)
    With sigma_c = 0, returns a sharp step function (FeedbackFreeModeOn=2).

    Parameters
    ----------
    Mvir_msun : array_like  Halo virial mass [M_sun].
    z         : float       Redshift.
    sigma_c   : float       Log-normal scatter in ln(c); 0.2 matches BK25 mode 4.
    """
    from scipy.optimize import brentq
    from scipy.stats import norm as _snorm

    Mvir_msun = np.atleast_1d(np.asarray(Mvir_msun, dtype=float))
    c_mean = _c_ishiyama21(Mvir_msun, z)
    Rvir = _rvir_m(Mvir_msun, z)
    g_vir = 6.674e-11 * Mvir_msun * 1.989e30 / Rvir**2

    if sigma_c == 0.0:
        mu = np.log(1.0 + c_mean) - c_mean / (1.0 + c_mean)
        g_max = g_vir * c_mean**2 / (2.0 * mu)
        return (g_max > _G_CRIT_SI).astype(float)

    f = np.zeros(len(Mvir_msun))
    for i in range(len(Mvir_msun)):
        gv = float(g_vir[i])

        def _obj(c_val):
            mu = np.log(1.0 + c_val) - c_val / (1.0 + c_val)
            return gv * c_val**2 / (2.0 * mu) - _G_CRIT_SI

        if _obj(1.0) > 0.0:       # even c=1 exceeds g_crit
            f[i] = 1.0
            continue
        if _obj(200.0) < 0.0:     # even c=200 is below g_crit
            f[i] = 0.0
            continue
        try:
            c_thresh = brentq(_obj, 1.0, 200.0, xtol=1e-3, rtol=1e-4)
            f[i] = _snorm.sf((np.log(c_thresh) - np.log(float(c_mean[i]))) / sigma_c)
        except ValueError:
            f[i] = 0.0

    return f


def mbk25_threshold_mass_msun(z, c):
    """
    MBK25 FFB threshold virial mass [M_sun] at redshift *z* for a fixed
    halo concentration *c*.

    Inverts the maximum-acceleration criterion g_max(M, z, c) = g_crit.  With
        g_max = G M / R_vir^2 * c^2 / (2 mu(c)),   mu(c) = ln(1+c) - c/(1+c)
    and R_vir^3 = 3 M / (4 pi Delta_vir(z) rho_crit(z)), the virial acceleration
    scales as g_vir = G M^{1/3} (4 pi Delta rho_crit / 3)^{2/3}, so the criterion
    is linear in M^{1/3} and inverts in closed form.

    A more concentrated halo (larger c) reaches g_crit at lower mass, so the
    threshold line drops with increasing c.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    G = 6.674e-11
    H0_si = HUBBLE_H * 1.0e5 / 3.085678e22
    Ez2 = OMEGA_M * (1.0 + z)**3 + OMEGA_L
    rho_crit = 3.0 * (H0_si**2) * Ez2 / (8.0 * np.pi * G)
    delta = _delta_vir_bn98(z)
    A = 4.0 * np.pi * delta * rho_crit / 3.0          # R_vir^3 = M_kg / A

    mu = np.log(1.0 + c) - c / (1.0 + c)
    shape = c**2 / (2.0 * mu)                          # g_max / g_vir
    # g_max = G * A^{2/3} * shape * M_kg^{1/3} = g_crit
    M_kg_cbrt = _G_CRIT_SI / (G * A**(2.0 / 3.0) * shape)
    M_kg = M_kg_cbrt**3
    return (M_kg / 1.989e30).squeeze()


def mbk25_threshold_concentration(Mvir_msun, z):
    """
    Threshold concentration c_thresh such that a halo of mass *Mvir_msun* at
    redshift *z* exactly satisfies the MBK25 criterion g_max = g_crit.

    A halo is FFB (mode-4) when its drawn concentration exceeds this value, so a
    selected galaxy lies on the fixed-c threshold line for c = c_thresh.  Returns
    NaN where even c = 200 fails to reach g_crit (never FFB at any concentration).
    """
    from scipy.optimize import brentq

    Mvir_msun = np.atleast_1d(np.asarray(Mvir_msun, dtype=float))
    G = 6.674e-11
    Rvir = _rvir_m(Mvir_msun, z)
    g_vir = G * Mvir_msun * 1.989e30 / Rvir**2

    out = np.full(len(Mvir_msun), np.nan)
    for i, gv in enumerate(g_vir):
        def _obj(c):
            mu = np.log(1.0 + c) - c / (1.0 + c)
            return gv * c**2 / (2.0 * mu) - _G_CRIT_SI
        if _obj(1.0) > 0.0:
            out[i] = 1.0
        elif _obj(200.0) < 0.0:
            out[i] = np.nan
        else:
            out[i] = brentq(_obj, 1.0, 200.0, xtol=1e-3, rtol=1e-4)
    return out.squeeze()


# ========================== FIGURE UTILITIES ==========================

def save_figure(fig, filepath):
    """Save figure to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath)
    print(f'  Saved: {filepath}')
    plt.close(fig)


def _scaled_font_rc(scale):
    """rcParams overrides with every font size scaled by *scale*.

    Used by figures whose canvas is smaller than the stylesheet default, so
    the text renders at the same size *relative to the axes* as elsewhere.
    """
    from matplotlib.font_manager import font_scalings

    base = plt.rcParams['font.size']
    keys = ('font.size', 'axes.labelsize', 'axes.titlesize',
            'xtick.labelsize', 'ytick.labelsize',
            'legend.fontsize', 'legend.title_fontsize')
    out = {}
    for k in keys:
        v = plt.rcParams[k]
        if isinstance(v, str):
            v = base * font_scalings.get(v, 1.0)
        out[k] = v * scale
    return out


def _standard_legend(ax, loc='lower left', handles=None, labels=None, **kwargs):
    """Apply consistent legend formatting with fully opaque handles."""
    kwargs.setdefault('frameon', False)
    if handles is not None and labels is not None:
        leg = ax.legend(handles, labels, loc=loc, numpoints=1,
                        labelspacing=0.1, **kwargs)
    else:
        leg = ax.legend(loc=loc, numpoints=1, labelspacing=0.1, **kwargs)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    return leg


# ========================== OBSERVATIONAL DATA ==========================

def load_gama_smf_morph():
    """Load GAMA morphological SMF (Moffett et al. 2016)."""
    path = os.path.join(OBS_DIR, 'smf/gama_smf_morph.ecsv')
    data = np.genfromtxt(path, comments='#', skip_header=1)
    return {
        'mass': data[:, 0],
        'E_HE': data[:, 1],
        'E_HE_err': data[:, 2],
        'D': data[:, 7],
        'D_err': data[:, 8],
    }


def load_baldry_blue_red():
    """Load Baldry et al. blue/red SMF data."""
    path = os.path.join(OBS_DIR, 'morphology/baldry_blue_red.csv')
    data = np.genfromtxt(path, delimiter=',', skip_header=2)
    return {
        'sf_mass': data[:, 0],
        'sf_phi': data[:, 1],
        'q_mass': data[:, 2],
        'q_phi': data[:, 3],
    }


def load_mzr_observations():
    """
    Load mass-metallicity relation observational data.

    Returns a list of dicts, each with keys:
        'mass', 'Z', 'yerr' (optional), 'fmt', 'label'
    """
    obs = []

    # Tremonti et al. 2004
    path = os.path.join(OBS_DIR, 'metallicity/Tremonti04.dat')
    if os.path.exists(path):
        d = np.loadtxt(path)
        obs.append({
            'mass': d[:, 0], 'Z': d[:, 1],
            'yerr': [d[:, 1] - d[:, 2], d[:, 3] - d[:, 1]],
            'fmt': 'o', 'color': 'k', 'label': 'Tremonti+04',
        })
    else:
        # Polynomial fallback
        m = np.arange(7.0, 13.0, 0.1)
        z = -1.492 + 1.847 * m - 0.08026 * m * m
        obs.append({
            'mass': m, 'Z': z, 'yerr': None,
            'fmt': 'o', 'color': 'k', 'label': 'Tremonti+04 (fit)',
        })

    # Curti et al. 2020
    path = os.path.join(OBS_DIR, 'metallicity/Curti2020.dat')
    if os.path.exists(path):
        d = np.loadtxt(path)
        obs.append({
            'mass': d[:, 0], 'Z': d[:, 1],
            'yerr': [d[:, 1] - d[:, 2], d[:, 3] - d[:, 1]],
            'fmt': 's', 'color': 'k', 'label': 'Curti+20',
        })

    # Andrews & Martini 2013
    path = os.path.join(OBS_DIR, 'metallicity/MMAdrews13.dat')
    if os.path.exists(path):
        d = np.loadtxt(path)
        obs.append({
            'mass': d[:, 0], 'Z': d[:, 1],
            'yerr': [d[:, 1] - d[:, 2], d[:, 3] - d[:, 1]],
            'fmt': '^', 'color': 'k',
            'label': _tex_safe(r'Andrews \& Martini 2013'),
        })

    # Kewley & Ellison 2008 - T04 calibration
    path = os.path.join(OBS_DIR, 'metallicity/MMR-Kewley08.dat')
    if os.path.exists(path):
        d = np.loadtxt(path)
        obs.append({
            'mass': d[59:74, 0], 'Z': d[59:74, 1], 'yerr': None,
            'fmt': 'd', 'color': 'k',
            'label': _tex_safe(r'Kewley \& Ellison 2008'),
        })

    # Gallazzi et al. 2005 (stellar -> gas-phase conversion)
    path = os.path.join(OBS_DIR, 'metallicity/MSZR-Gallazzi05.dat')
    if os.path.exists(path):
        d = np.loadtxt(path)
        m = d[7:, 0]
        z_gas = d[7:, 1] + 8.69
        z_lo = d[7:, 2] + 8.69
        z_hi = d[7:, 3] + 8.69
        obs.append({
            'mass': m, 'Z': z_gas,
            'yerr': [z_gas - z_lo, z_hi - z_gas],
            'fmt': 'v', 'color': 'k', 'label': 'Gallazzi+05 (conv.)',
        })

    return obs


def load_bh_bulge_observations():
    """
    Load black hole - bulge mass observational data.

    Returns
    -------
    dict with keys:
        'M_sph', 'M_BH', 'xerr', 'yerr', 'core' (boolean mask),
        'haring_rix_x', 'haring_rix_y' (relation line).
    """
    h_ratio = (0.7 / HUBBLE_H)**2

    M_BH_obs = h_ratio * 1e8 * np.array([
        39, 11, 0.45, 25, 24, 0.044, 1.4, 0.73, 9.0, 58, 0.10, 8.3, 0.39,
        0.42, 0.084, 0.66, 0.73, 15, 4.7, 0.083, 0.14, 0.15, 0.4, 0.12,
        1.7, 0.024, 8.8, 0.14, 2.0, 0.073, 0.77, 4.0, 0.17, 0.34, 2.4,
        0.058, 3.1, 1.3, 2.0, 97, 8.1, 1.8, 0.65, 0.39, 5.0, 3.3, 4.5,
        0.075, 0.68, 1.2, 0.13, 4.7, 0.59, 6.4, 0.79, 3.9, 47, 1.8, 0.06,
        0.016, 210, 0.014, 7.4, 1.6, 6.8, 2.6, 11, 37, 5.9, 0.31, 0.10,
        3.7, 0.55, 13, 0.11])
    M_BH_hi = h_ratio * 1e8 * np.array([
        4, 2, 0.17, 7, 10, 0.044, 0.9, 0.0, 0.9, 3.5, 0.10, 2.7, 0.26,
        0.04, 0.003, 0.03, 0.69, 2, 0.6, 0.004, 0.02, 0.09, 0.04, 0.005,
        0.2, 0.024, 10, 0.1, 0.5, 0.015, 0.04, 1.0, 0.01, 0.02, 0.3,
        0.008, 1.4, 0.5, 1.1, 30, 2.0, 0.6, 0.07, 0.01, 1.0, 0.9, 2.3,
        0.002, 0.13, 0.4, 0.08, 0.5, 0.03, 0.4, 0.38, 0.4, 10, 0.2,
        0.014, 0.004, 160, 0.014, 4.7, 0.3, 0.7, 0.4, 1, 18, 2.0, 0.004,
        0.001, 2.6, 0.26, 5, 0.005])
    M_BH_lo = h_ratio * 1e8 * np.array([
        5, 2, 0.10, 7, 10, 0.022, 0.3, 0.0, 0.8, 3.5, 0.05, 1.3, 0.09,
        0.04, 0.003, 0.03, 0.35, 2, 0.6, 0.004, 0.13, 0.1, 0.05, 0.005,
        0.2, 0.012, 2.7, 0.06, 0.5, 0.015, 0.06, 1.0, 0.02, 0.02, 0.3,
        0.008, 0.6, 0.5, 0.6, 26, 1.9, 0.3, 0.07, 0.01, 1.0, 2.5, 1.5,
        0.002, 0.13, 0.9, 0.08, 0.5, 0.09, 0.4, 0.33, 0.4, 10, 0.1,
        0.014, 0.004, 160, 0.007, 3.0, 0.4, 0.7, 1.5, 1, 11, 2.0, 0.004,
        0.001, 1.5, 0.19, 4, 0.005])
    M_sph_obs = h_ratio * 1e10 * np.array([
        69, 37, 1.4, 55, 27, 2.4, 0.46, 1.0, 19, 23, 0.61, 4.6, 11, 1.9,
        4.5, 1.4, 0.66, 4.7, 26, 2.0, 0.39, 0.35, 0.30, 3.5, 6.7, 0.88,
        1.9, 0.93, 1.24, 0.86, 2.0, 5.4, 1.2, 4.9, 2.0, 0.66, 5.1, 2.6,
        3.2, 100, 1.4, 0.88, 1.3, 0.56, 29, 6.1, 0.65, 3.3, 2.0, 6.9,
        1.4, 7.7, 0.9, 3.9, 1.8, 8.4, 27, 6.0, 0.43, 1.0, 122, 0.30, 29,
        11, 20, 2.8, 24, 78, 96, 3.6, 2.6, 55, 1.4, 64, 1.2])
    M_sph_hi = h_ratio * 1e10 * np.array([
        59, 32, 2.0, 80, 23, 3.5, 0.68, 1.5, 16, 19, 0.89, 6.6, 9, 2.7,
        6.6, 2.1, 0.91, 6.9, 22, 2.9, 0.57, 0.52, 0.45, 5.1, 5.7, 1.28,
        2.7, 1.37, 1.8, 1.26, 1.7, 4.7, 1.7, 7.1, 2.9, 0.97, 7.4, 3.8,
        2.7, 86, 2.1, 1.30, 1.9, 0.82, 25, 5.2, 0.96, 4.9, 3.0, 5.9, 1.2,
        6.6, 1.3, 5.7, 2.7, 7.2, 23, 5.2, 0.64, 1.5, 105, 0.45, 25, 10,
        17, 2.4, 20, 67, 83, 5.2, 3.8, 48, 2.0, 55, 1.8])
    M_sph_lo = h_ratio * 1e10 * np.array([
        32, 17, 0.8, 33, 12, 1.4, 0.28, 0.6, 9, 10, 0.39, 2.7, 5, 1.1,
        2.7, 0.8, 0.40, 2.8, 12, 1.2, 0.23, 0.21, 0.18, 2.1, 3.1, 0.52,
        1.1, 0.56, 0.7, 0.51, 0.9, 2.5, 0.7, 2.9, 1.2, 0.40, 3.0, 1.5,
        1.5, 46, 0.9, 0.53, 0.8, 0.34, 13, 2.8, 0.39, 2.0, 1.2, 3.2, 0.6,
        3.6, 0.5, 2.3, 1.1, 3.9, 12, 2.8, 0.26, 0.6, 57, 0.18, 13, 5, 9,
        1.3, 11, 36, 44, 2.1, 1.5, 26, 0.8, 30, 0.7])
    core = np.array([
        1,1,0,1,1,0,0,0,1,1,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,
        0,0,0,0,0,0,1,1,1,0,0,0,1,1,0,0,0,0,0,1,0,1,0,0,1,0,0,0,1,0,1,0,
        1,0,1,1,1,0,0,1,0,1,0], dtype=bool)

    # Log-space error bars
    yerr_hi = np.log10((M_BH_obs + M_BH_hi) / M_BH_obs)
    yerr_lo = -np.log10((M_BH_obs - M_BH_lo) / M_BH_obs)
    xerr_hi = np.log10((M_sph_obs + M_sph_hi) / M_sph_obs)
    xerr_lo = -np.log10((M_sph_obs - M_sph_lo) / M_sph_obs)

    # Haring & Rix 2004 relation
    hr_x = np.linspace(8, 13, 100)
    hr_y = 8.2 + 1.12 * (hr_x - 11.0)

    return {
        'log_M_sph': np.log10(M_sph_obs),
        'log_M_BH': np.log10(M_BH_obs),
        'xerr': [xerr_lo, xerr_hi],
        'yerr': [yerr_lo, yerr_hi],
        'core': core,
        'haring_rix_x': hr_x,
        'haring_rix_y': hr_y,
    }

def load_shmr_observations():
    """
    Load stellar-to-halo mass relation observational data.

    Returns a dict with keys:
        'moster'   : dict with 'mvir', 'mstar' (best-fit line)
        'romeo'    : dict with 'mvir', 'mstar' (combined all + ETGs)
        'kravtsov' : dict with 'mvir', 'mstar', 'xerr_lo', 'xerr_hi',
                     'has_xerr' (combined ETGs + LTGs + Sat.)
        'taylor'   : dict with 'mvir', 'mstar', 'xerr', 'yerr'
    """
    obs = {}

    # Moster et al. 2013 — best-fit relation (first pair of columns)
    path = os.path.join(OBS_DIR, 'smhm/Moster_2013.csv')
    if os.path.exists(path):
        d = np.genfromtxt(path)
        valid = ~np.isnan(d[:, 0])
        obs['moster'] = {
            'mvir': d[valid, 0],
            'mstar': d[valid, 1],
        }

    # Romeo et al. 2020 — combined (all galaxies + ETGs)
    # Format: (log_Mvir, log_M*/Mvir)
    mvir_parts, mstar_parts = [], []
    for fname in ['smhm/Romeo20_SMHM.dat', 'smhm/Romeo20_SMHM_ETGs.dat']:
        path = os.path.join(OBS_DIR, fname)
        if os.path.exists(path):
            d = np.loadtxt(path)
            mvir_parts.append(d[:, 0])
            mstar_parts.append(d[:, 0] + d[:, 1])
    if mvir_parts:
        obs['romeo'] = {
            'mvir': np.concatenate(mvir_parts),
            'mstar': np.concatenate(mstar_parts),
        }

    # Kravtsov et al. 2018 — combined (ETGs + LTGs + Sat/Clusters)
    k_mvir, k_mstar, k_xerr_lo, k_xerr_hi, k_has_xerr = [], [], [], [], []
    for fname in ['morphology/ETGs_Kravtsov18.dat', 'morphology/LTGs_Kravtsov18.dat']:
        path = os.path.join(OBS_DIR, fname)
        if os.path.exists(path):
            d = np.loadtxt(path)
            k_mvir.append(d[:, 0])
            k_mstar.append(d[:, 1])
            k_xerr_lo.append(d[:, 0] - d[:, 2])
            k_xerr_hi.append(d[:, 3] - d[:, 0])
            k_has_xerr.append(np.ones(len(d), dtype=bool))
    path = os.path.join(OBS_DIR, 'morphology/SatKinsAndClusters_Kravtsov18.dat')
    if os.path.exists(path):
        d = np.loadtxt(path)
        k_mvir.append(d[:, 0])
        k_mstar.append(d[:, 1])
        k_xerr_lo.append(np.zeros(len(d)))
        k_xerr_hi.append(np.zeros(len(d)))
        k_has_xerr.append(np.zeros(len(d), dtype=bool))
    if k_mvir:
        obs['kravtsov'] = {
            'mvir': np.concatenate(k_mvir),
            'mstar': np.concatenate(k_mstar),
            'xerr_lo': np.concatenate(k_xerr_lo),
            'xerr_hi': np.concatenate(k_xerr_hi),
            'has_xerr': np.concatenate(k_has_xerr),
        }

    # Taylor et al. 2020
    # File columns, per its own header:
    #   log10(m*/Msun)  log10(m*)_lo  log10(m*)_hi  Mhalo/1e12Msun  Mhalo_lo  Mhalo_hi
    # Note the stellar mass is logarithmic and the halo mass is LINEAR in units of
    # 1e12 Msun. This was previously read as (log_Mhalo, lo, hi, ratio, lo, hi), which
    # swapped the two axes and log10'd a halo mass as though it were a ratio -- putting
    # the points at log_Mvir ~ 10.3-10.6 with m*/Mvir up to 0.7, above the cosmic baryon
    # fraction and so unphysical.
    path = os.path.join(OBS_DIR, 'morphology/Taylor20.dat')
    if os.path.exists(path):
        d = np.atleast_2d(np.loadtxt(path))
        log_mstar = d[:, 0]
        log_mstar_lo = d[:, 1]
        log_mstar_hi = d[:, 2]
        log_mvir = np.log10(d[:, 3]) + 12.0
        log_mvir_lo = np.log10(d[:, 4]) + 12.0
        log_mvir_hi = np.log10(d[:, 5]) + 12.0
        log_ratio = log_mstar - log_mvir
        obs['taylor'] = {
            'mvir': log_mvir,
            'mstar': log_mstar,
            'xerr': [log_mvir - log_mvir_lo, log_mvir_hi - log_mvir],
            'yerr': [log_mstar - log_mstar_lo, log_mstar_hi - log_mstar],
            # Ratio uncertainty takes the outer corners of both intervals, so the bar
            # spans the full range the two independent measurements allow.
            'ratio': log_ratio,
            'ratio_err': [log_ratio - (log_mstar_lo - log_mvir_hi),
                          (log_mstar_hi - log_mvir_lo) - log_ratio],
        }

    return obs

def load_madau_dickinson_2014_data():
    """Load Madau and Dickinson 2014 SFRD data."""
    if not HAS_ASTROPY:
        return None, None, None, None
    filename = './data/sfrd/MandD_sfrd_2014.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None

    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z_min']
        # Madau & Dickinson (2014) quote their compilation for a Salpeter IMF;
        # shift it onto the model's Chabrier scale. The errors are in dex and so
        # are unchanged by the shift.
        re = table['log_psi'] + SALPETER_TO_CHABRIER_DEX
        re_err_plus = table['e_log_psi_up']
        re_err_minus = table['e_log_psi_lo']
        return z, re, re_err_plus, re_err_minus
    except Exception as e:
        print(f"Error loading Madau and Dickinson 2014 SFRD data: {e}")
    return None, None, None, None

def load_madau_dickinson_smd_2014_data():
    """Load Madau and Dickinson 2014 SMD data."""
    if not HAS_ASTROPY:
        return None, None, None, None
    filename = './data/sfrd/MandD_smd_2014.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None

    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z_min']
        # Salpeter -> Chabrier, as for the SFRD compilation above. Keeps this
        # curve on the same footing as the COSMOS-Web SMD, which is Chabrier.
        re = table['log_rho'] + SALPETER_TO_CHABRIER_DEX
        re_err_plus = table['e_log_rho_up']
        re_err_minus = table['e_log_rho_lo']
        return z, re, re_err_plus, re_err_minus
    except Exception as e:
        print(f"Error loading Madau and Dickinson 2014 SMD data: {e}")
    return None, None, None, None

def load_kikuchihara_smd_2020_data():
    """Load Kikuchihara et al. 2020 SMD data."""
    if not HAS_ASTROPY:
        return None, None, None, None
    filename = './data/sfrd/kikuchihara_smd_2020.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None

    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z']
        re = table['log_rho_star']
        re_err_plus = table['e_log_rho_star_upper']
        re_err_minus = table['e_log_rho_star_lower']
        return z, re, re_err_plus, re_err_minus
    except Exception as e:
        print(f"Error loading Kikuchihara 2020 SMD data: {e}")
    return None, None, None, None

def load_papovich_smd_2023_data():
    """Load Papovich et al. 2023 SMD data."""
    if not HAS_ASTROPY:
        return None, None, None, None
    filename = './data/sfrd/papovich_smd_2023.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None

    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z']
        re = table['log_rho_star']
        re_err_plus = table['e_log_rho_star_upper']
        re_err_minus = table['e_log_rho_star_lower']
        return z, re, re_err_plus, re_err_minus
    except Exception as e:
        print(f"Error loading Papovich 2023 SMD data: {e}")
    return None, None, None, None

def load_oesch_sfrd_2018_data():
    """Load Oesch et al. 2018 SFRD data."""
    if not HAS_ASTROPY:
        return None, None, None, None
    filename = './data/sfrd/oesch_sfrd_2018.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None

    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z']
        re = table['log_rho_sfr']
        re_err_plus = table['e_log_rho_sfr_upper']
        re_err_minus = table['e_log_rho_sfr_lower']
        return z, re, re_err_plus, re_err_minus
    except Exception as e:
        print(f"Error loading Oesch 2018 SFRD data: {e}")
    return None, None, None, None

def load_mcleod_rho_sfr_2024_data():
    """Load McLeod et al. 2024 SFR density data."""
    if not HAS_ASTROPY:
        return None, None, None, None
    filename = './data/sfrd/mcleod_rhouv_2024.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None

    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z']
        re = table['log_rho_sfr']
        re_err_plus = np.zeros_like(re)
        re_err_minus = np.zeros_like(re)
        return z, re, re_err_plus, re_err_minus
    except Exception as e:
        print(f"Error loading McLeod 2024 SFRD data: {e}")
    return None, None, None, None

def load_harikane_sfr_density_2023_data():
    """Load Harikane et al. 2023 SFR density data."""
    if not HAS_ASTROPY:
        return None, None, None, None
    filename = './data/sfrd/harikane_density_2023.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None

    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z']
        re = table['log_rho_SFR_UV']
        re_err_plus = table['e_log_rho_SFR_UV_upper']
        re_err_minus = table['e_log_rho_SFR_UV_lower']
        return z, re, re_err_plus, re_err_minus
    except Exception as e:
        print(f"Error loading Harikane 2023 SFRD data: {e}")
    return None, None, None, None

def load_brinchmann_sfr_mass_2004_data():
    """Load Brinchmann et al. 2004 SFR vs Stellar Mass data."""
    if not HAS_ASTROPY:
        return None, None
    filename = './data/sfr/Brinchmann04.dat'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None

    try:
        # Read lines up to the stop marker
        data_lines = []
        with open(filename, 'r') as f:
            for line in f:
                if line.strip().startswith('#low boundary of 0.02 probability'):
                    break
                if line.strip().startswith('#') or not line.strip():
                    continue
                data_lines.append(line)
        # Use astropy Table to parse the collected lines
        from io import StringIO
        # Pass the list of lines directly as an iterable
        table = Table.read(
            data_lines,
            format='ascii.no_header',
            names=['log_mass', 'log_sfr'],
            delimiter=' ',  # whitespace
            guess=False,
            fast_reader=False
        )
        mass = table['log_mass']
        sfr = table['log_sfr']
        return mass, sfr
    except Exception as e:
        print(f"Error loading Brinchmann 2004 SFR-Mass data: {e}")
    return None, None

#
# Load Terrazas+17 MBH host galaxy SFR data
def load_terrazas17_mbh_host_sfr_data():
    """Load Terrazas et al. 2017 MBH host galaxy SFR data."""
    import numpy as np
    filename = './data/bh/MBH_host_gals_Terrazas17.dat'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None
    try:
        data = np.loadtxt(filename, comments='#', usecols=(0,1))
        log_mstar = data[:,0]
        sfr = data[:,1]
        return log_mstar, sfr
    except Exception as e:
        print(f"Error loading Terrazas+17 MBH host SFR data: {e}")
        return None, None
    
# Load and process GAMA ProSpect Claudia data for SFR vs stellar mass
def load_gama_prospect_claudia(obsdir=None):
    """Load GAMA ProSpect Claudia data, apply SFR floor, and return log10(mass), log10(SFR)."""
    import numpy as np
    # If obsdir is given, use it; else assume data/ subdir
    filename = os.path.join(obsdir, 'GAMA/ProSpect_Claudia.txt') if obsdir else './data/morphology/ProSpect_Claudia.txt'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None
    try:
        data = np.genfromtxt(filename, comments='#', usecols=(1,5))
        ms_gama = data[:,0]
        sfr_gama = data[:,1]
        sfr_gama[sfr_gama < 1e-3] = 1e-3
        log_ms = np.log10(ms_gama)
        log_sfr = np.log10(sfr_gama)
        return log_ms, log_sfr
    except Exception as e:
        print(f"Error loading GAMA ProSpect Claudia data: {e}")
        return None, None
    
# Load Bell+03 SMF starforming data
def load_bell_smf_sf_data():
    """Load Bell+03 SMF starforming data."""
    import numpy as np
    filename = './data/morphology/Bell_z0pt0_blue.dat'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None
    try:
        data = np.loadtxt(filename, comments='#', usecols=(0,1,2,3))
        log_mstar = data[:,0]
        sfr = data[:,1]
        error_high = data[:,2]
        error_low = data[:,3]
        return log_mstar, sfr, error_high, error_low
    except Exception as e:
        print(f"Error loading Bell+03 SMF starforming data: {e}")
        return None, None, None, None
    
# Load Bell+03 SMF quiescent data
def load_bell_smf_q_data():
    """Load Bell+03 SMF quiescent data."""
    import numpy as np
    filename = './data/morphology/Bell_z0pt0_red.dat'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None
    try:
        data = np.loadtxt(filename, comments='#', usecols=(0,1,2,3))
        log_mstar = data[:,0]
        sfr = data[:,1]
        error_high = data[:,2]
        error_low = data[:,3]
        return log_mstar, sfr, error_high, error_low
    except Exception as e:
        print(f"Error loading Bell+03 SMF starforming data: {e}")
        return None, None, None, None


def load_himf_observations():
    """
    Load HI mass function observations from Jones+18 and Zwaan+05.

    Returns a list of dicts, each with:
        'label': str
        'mass': array of log10(M_HI/Msun)
        'phi': array of log10(phi / Mpc^-3 dex^-1)
        'phi_lo': lower error (absolute phi values or error bars)
        'phi_hi': upper error
        'marker': plot marker style
        'color': plot color
    """
    observations = []

    # Jones et al. (2018) - ALFALFA 100
    jones_path = os.path.join(OBS_DIR, 'Gas/HIMF_Jones18.dat')
    if os.path.exists(jones_path):
        try:
            data = np.loadtxt(jones_path, comments='#')
            # Columns: log(MHI), log(phi), lower_bound, upper_bound
            # h=0.7 assumed in Jones+18, same as our simulation
            observations.append({
                'label': 'Jones+18 (ALFALFA)',
                'mass': data[:, 0],
                'phi': data[:, 1],
                'phi_lo': data[:, 2],  # These are absolute values, not errors
                'phi_hi': data[:, 3],
                'marker': 'o',
                'color': 'k',
            })
        except Exception as e:
            print(f"Warning: Could not load Jones+18 HIMF: {e}")

    # Zwaan et al. (2005) - HIPASS
    zwaan_path = os.path.join(OBS_DIR, 'Gas/HIMF_Zwaan2005.dat')
    if os.path.exists(zwaan_path):
        try:
            data = np.loadtxt(zwaan_path, comments='#')
            # Columns: log(MHI), log(Theta), lower_1sigma, upper_1sigma
            # h=0.75 assumed, need to convert to h=0.7
            # M_HI scales as h^-2, so log(M) shifts by -2*log(h_new/h_old)
            # phi scales as h^3, so log(phi) shifts by 3*log(h_new/h_old)
            h_zwaan = 0.75
            h_ours = 0.7  # HUBBLE_H from simulation
            h_ratio = h_ours / h_zwaan
            mass_shift = -2.0 * np.log10(h_ratio)
            phi_shift = 3.0 * np.log10(h_ratio)

            observations.append({
                'label': 'Zwaan+05 (HIPASS)',
                'mass': data[:, 0] + mass_shift,
                'phi': data[:, 1] + phi_shift,
                # Errors are relative (sigma values to add/subtract)
                'phi_err_lo': data[:, 2],  # These are error magnitudes
                'phi_err_hi': data[:, 3],
                'marker': 's',
                'color': 'gray',
            })
        except Exception as e:
            print(f"Warning: Could not load Zwaan+05 HIMF: {e}")

    return observations


def _gasmf_obs_yerr(obs, mask):
    """Normalise the two error conventions used by the gas MF observation loaders.

    Jones+18 and the H2 sets store absolute phi bounds in 'phi_lo'/'phi_hi';
    Zwaan+05 stores error magnitudes in 'phi_err_lo'/'phi_err_hi'.
    """
    phi = obs['phi'][mask]
    if 'phi_err_lo' in obs:
        return [obs['phi_err_lo'][mask], obs['phi_err_hi'][mask]]
    return [phi - obs['phi_lo'][mask], obs['phi_hi'][mask] - phi]


# ========================== PLOT 1: STELLAR MASS FUNCTION (SF/Q) ==========================

def plot_1_stellar_mass_function_ssfr_s(primary, vanilla):
    """
    Stellar mass function divided by sSFR into star-forming
    and quiescent populations.

    Compares SAGE26 (primary) with C16 (vanilla) and observations
    (GAMA morphological SMF + Baldry blue/red).
    Includes bootstrap error shading for SAGE26.
    """
    print('Plot 1: Stellar mass function (SF/Q split) with Bootstrap Errors')

    binwidth = 0.1
    N_BOOT = 100  # Number of bootstrap samples

    # --- Primary model ---
    w = primary['StellarMass'] > 0
    mass = np.log10(primary['StellarMass'][w])
    ssfr = log_ssfr(primary['SfrDisk'][w], primary['SfrBulge'][w],
                     primary['StellarMass'][w])

    # 1. Calculate main lines (and establish common bins)
    # We calculate the total MF first just to get the 'mrange' covering all galaxies
    x, _, mrange = mass_function(mass, VOLUME, binwidth)
    
    # Split populations
    mass_q = mass[ssfr < SSFR_CUT]
    mass_sf = mass[ssfr > SSFR_CUT]

    _, phi_q, _ = mass_function(mass_q, VOLUME, binwidth, mass_range=mrange)
    _, phi_sf, _ = mass_function(mass_sf, VOLUME, binwidth, mass_range=mrange)

    # 2. Bootstrap Error Calculation
    def calc_bootstrap_errors(data_mass, m_range, vol, bw, n_boot=100):
        if len(data_mass) == 0:
            return np.nan, np.nan
        
        # Reconstruct bin edges from mrange (same logic as mass_function)
        mi, ma = m_range
        nbins = int(round((ma - mi) / bw))
        edges = np.linspace(mi, ma, nbins + 1)
        
        boot_phis = []
        n_obj = len(data_mass)
        
        for _ in range(n_boot):
            # Resample with replacement
            sample = data_mass[np.random.randint(0, n_obj, n_obj)]
            counts, _ = np.histogram(sample, bins=edges)
            
            # Convert to log density (phi)
            with np.errstate(divide='ignore'):
                phi = np.log10(counts / vol / bw)
            # Treat empty bins as NaN for percentile calculation
            phi[~np.isfinite(phi)] = np.nan
            boot_phis.append(phi)
            
        boot_phis = np.array(boot_phis)
        # Calculate 16th and 84th percentiles ignoring NaNs
        lo = np.nanpercentile(boot_phis, 16, axis=0)
        hi = np.nanpercentile(boot_phis, 84, axis=0)
        return lo, hi

    print(f'  Bootstrapping SAGE26 data ({N_BOOT} iterations)...')
    phi_q_lo, phi_q_hi = calc_bootstrap_errors(mass_q, mrange, VOLUME, binwidth, N_BOOT)
    phi_sf_lo, phi_sf_hi = calc_bootstrap_errors(mass_sf, mrange, VOLUME, binwidth, N_BOOT)

    # --- Vanilla model ---
    w2 = vanilla['StellarMass'] > 0
    mass_v = np.log10(vanilla['StellarMass'][w2])
    ssfr_v = log_ssfr(vanilla['SfrDisk'][w2], vanilla['SfrBulge'][w2],
                       vanilla['StellarMass'][w2])

    x_v, _, mrange_v = mass_function(mass_v, VOLUME, binwidth)
    _, phi_q_v, _ = mass_function(mass_v[ssfr_v < SSFR_CUT], VOLUME, binwidth,
                                  mass_range=mrange_v)
    _, phi_sf_v, _ = mass_function(mass_v[ssfr_v > SSFR_CUT], VOLUME, binwidth,
                                   mass_range=mrange_v)

    # --- QUANTITATIVE COMPARISON: SAGE26 vs SAGE16 (Star-forming SMF, z=0) ---
    print("\n" + "="*60)
    print("QUANTITATIVE COMPARISON: SAGE26 vs SAGE16 (Star-forming SMF, z=0)")
    print("="*60)
    _x_com = np.round(np.intersect1d(np.round(x, 4), np.round(x_v, 4)), 4)
    if len(_x_com) > 0:
        _ip = np.array([np.argmin(np.abs(x - xc)) for xc in _x_com])
        _iv = np.array([np.argmin(np.abs(x_v - xc)) for xc in _x_com])
        _pp = phi_sf[_ip];  _pv = phi_sf_v[_iv]
        _ok = np.isfinite(_pp) & np.isfinite(_pv)
        if np.sum(_ok) > 0:
            _d = _pp[_ok] - _pv[_ok];  _xok = _x_com[_ok]
            print(f"\n  Over log10(M*/Msun) = {_xok.min():.1f} to {_xok.max():.1f}")
            print(f"  Mean difference (SAGE26 - SAGE16):  {np.mean(_d):+.3f} dex")
            print(f"  Median difference:                  {np.median(_d):+.3f} dex")
            print(f"  Std of difference:                  {np.std(_d):.3f} dex")
            print(f"  Max offset at log10(M*)={_xok[np.argmax(_d)]:.2f}: {np.max(_d):+.3f} dex")
            print(f"  Min offset at log10(M*)={_xok[np.argmin(_d)]:.2f}: {np.min(_d):+.3f} dex")
            print(f"\n  phi(SF) at specific stellar masses:")
            for _tm in [9.0, 9.5, 10.0, 10.5, 11.0, 11.5]:
                _ti = np.argmin(np.abs(_xok - _tm))
                if np.abs(_xok[_ti] - _tm) < binwidth:
                    print(f"    log10(M*)={_xok[_ti]:.2f}: SAGE26={_pp[_ok][_ti]:.2f}, "
                          f"SAGE16={_pv[_ok][_ti]:.2f}, Δ={_d[_ti]:+.2f} dex")
    print("="*60 + "\n")

    # --- Observations ---
    gama = load_gama_smf_morph()
    baldry = load_baldry_blue_red()

    # --- Plot ---
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # SAGE26 (Lines + Shading)
    # Quiescent
    # ax.plot(x, phi_q, color='firebrick', lw=3, label='SAGE26 Quiescent')
    # ax.fill_between(x, phi_q_lo, phi_q_hi, color='firebrick', alpha=0.3, edgecolor='none', zorder=10)

    # Star-forming
    ax.plot(x, phi_sf, color='steelblue', lw=4, label='SAGE26 Star-forming', zorder=10)
    ax.fill_between(x, phi_sf_lo, phi_sf_hi, color='steelblue', alpha=0.3, edgecolor='none', zorder=10)
    # C16 (vanilla)
    # ax.plot(x_v, phi_q_v, color='firebrick', lw=2, ls='--', label='C16 Quiescent')
    ax.plot(x_v, phi_sf_v, color='steelblue', lw=2, ls='--', label='SAGE16 Star-forming')

    # Observational data: GAMA (Moffett+16) with 'd' markers
    valid_D = ~np.isnan(gama['D'])
    valid_E = ~np.isnan(gama['E_HE'])
    ax.errorbar(gama['mass'][valid_D], gama['D'][valid_D],
                yerr=gama['D_err'][valid_D],
                fmt='d', color='k',markeredgecolor='k', markeredgewidth=1.0, linewidth=1.0,
                markerfacecolor = 'gray', ms=8,
                alpha=0.6, zorder=9,
                label='Moffett+16')
    # ax.errorbar(gama['mass'][valid_E], gama['E_HE'][valid_E],
    #             yerr=gama['E_HE_err'][valid_E],
    #             fmt='d', color='r', ms=10, lw=1.5, capsize=2)

    # Observational data: Baldry+12 with 'o' markers
    ax.scatter(baldry['sf_mass'], baldry['sf_phi'], edgecolor='k', facecolor='gray', marker=
            'o', color='k', s=50, label='Baldry+12', alpha=0.6, zorder=9)
    # ax.plot(baldry['q_mass'], baldry['q_phi'],
    #         'o', color='r', ms=10)

    # Load Bell+03 SMF starforming data
    bell_mass, bell_phi, bell_err_hi, bell_err_lo = load_bell_smf_sf_data()
    if bell_mass is not None:
        ax.errorbar(bell_mass, bell_phi,
                    yerr=[bell_err_lo, bell_err_hi],
                    markeredgecolor='k', markerfacecolor='gray',
                    fmt='s', color='k', ms=8, lw=1.0, alpha=0.6, zorder=9,
                    label='Bell+03')

    ax.set_xlim(8, 12)
    ax.set_ylim(-6, -1)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.set_ylabel(r'$\log_{10}\ \phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$')
    ax.set_xlabel(r'$\log_{10}\ m_{\mathrm{*}}\ [M_{\odot}]$')

    handles, labels = ax.get_legend_handles_labels()
    sim_h = [h for h, l in zip(handles, labels) if l.startswith(('SAGE26', 'SAGE16'))]
    sim_l = [l for l in labels if l.startswith(('SAGE26', 'SAGE16'))]
    obs_h = [h for h, l in zip(handles, labels) if l.startswith('Baldry') or l.startswith('Moffett') or l.startswith('Bell')]
    obs_l = [l for l in labels if l.startswith('Baldry') or l.startswith('Moffett') or l.startswith('Bell')]
    leg1 = _standard_legend(ax, loc='lower left', handles=sim_h, labels=sim_l)
    ax.add_artist(leg1)
    _standard_legend(ax, loc='upper right', handles=obs_h, labels=obs_l)
    fig.tight_layout()

    save_figure(fig, os.path.join(OUTPUT_DIR,
                'StellarMassFunction_SF' + OUTPUT_FORMAT))
    
def plot_1_stellar_mass_function_ssfr_q(primary, vanilla):
    """
    Stellar mass function divided by sSFR into star-forming
    and quiescent populations.

    Compares SAGE26 (primary) with C16 (vanilla) and observations
    (GAMA morphological SMF + Baldry blue/red).
    Includes bootstrap error shading for SAGE26.
    """
    print('Plot 1: Stellar mass function (SF/Q split) with Bootstrap Errors')

    binwidth = 0.1
    N_BOOT = 100  # Number of bootstrap samples

    # --- Primary model ---
    w = primary['StellarMass'] > 0
    mass = np.log10(primary['StellarMass'][w])
    ssfr = log_ssfr(primary['SfrDisk'][w], primary['SfrBulge'][w],
                     primary['StellarMass'][w])

    # 1. Calculate main lines (and establish common bins)
    # We calculate the total MF first just to get the 'mrange' covering all galaxies
    x, _, mrange = mass_function(mass, VOLUME, binwidth)
    
    # Split populations
    mass_q = mass[ssfr < SSFR_CUT]
    mass_sf = mass[ssfr > SSFR_CUT]

    _, phi_q, _ = mass_function(mass_q, VOLUME, binwidth, mass_range=mrange)
    _, phi_sf, _ = mass_function(mass_sf, VOLUME, binwidth, mass_range=mrange)

    # 2. Bootstrap Error Calculation
    def calc_bootstrap_errors(data_mass, m_range, vol, bw, n_boot=100):
        if len(data_mass) == 0:
            return np.nan, np.nan
        
        # Reconstruct bin edges from mrange (same logic as mass_function)
        mi, ma = m_range
        nbins = int(round((ma - mi) / bw))
        edges = np.linspace(mi, ma, nbins + 1)
        
        boot_phis = []
        n_obj = len(data_mass)
        
        for _ in range(n_boot):
            # Resample with replacement
            sample = data_mass[np.random.randint(0, n_obj, n_obj)]
            counts, _ = np.histogram(sample, bins=edges)
            
            # Convert to log density (phi)
            with np.errstate(divide='ignore'):
                phi = np.log10(counts / vol / bw)
            # Treat empty bins as NaN for percentile calculation
            phi[~np.isfinite(phi)] = np.nan
            boot_phis.append(phi)
            
        boot_phis = np.array(boot_phis)
        # Calculate 16th and 84th percentiles ignoring NaNs
        lo = np.nanpercentile(boot_phis, 16, axis=0)
        hi = np.nanpercentile(boot_phis, 84, axis=0)
        return lo, hi

    print(f'  Bootstrapping SAGE26 data ({N_BOOT} iterations)...')
    phi_q_lo, phi_q_hi = calc_bootstrap_errors(mass_q, mrange, VOLUME, binwidth, N_BOOT)
    phi_sf_lo, phi_sf_hi = calc_bootstrap_errors(mass_sf, mrange, VOLUME, binwidth, N_BOOT)

    # --- Vanilla model ---
    w2 = vanilla['StellarMass'] > 0
    mass_v = np.log10(vanilla['StellarMass'][w2])
    ssfr_v = log_ssfr(vanilla['SfrDisk'][w2], vanilla['SfrBulge'][w2],
                       vanilla['StellarMass'][w2])

    x_v, _, mrange_v = mass_function(mass_v, VOLUME, binwidth)
    _, phi_q_v, _ = mass_function(mass_v[ssfr_v < SSFR_CUT], VOLUME, binwidth,
                                  mass_range=mrange_v)
    _, phi_sf_v, _ = mass_function(mass_v[ssfr_v > SSFR_CUT], VOLUME, binwidth,
                                   mass_range=mrange_v)

    # --- QUANTITATIVE COMPARISON: SAGE26 vs SAGE16 (Quiescent SMF, z=0) ---
    print("\n" + "="*60)
    print("QUANTITATIVE COMPARISON: SAGE26 vs SAGE16 (Quiescent SMF, z=0)")
    print("="*60)
    _x_com = np.round(np.intersect1d(np.round(x, 4), np.round(x_v, 4)), 4)
    if len(_x_com) > 0:
        _ip = np.array([np.argmin(np.abs(x - xc)) for xc in _x_com])
        _iv = np.array([np.argmin(np.abs(x_v - xc)) for xc in _x_com])
        _pp = phi_q[_ip];  _pv = phi_q_v[_iv]
        _ok = np.isfinite(_pp) & np.isfinite(_pv)
        if np.sum(_ok) > 0:
            _d = _pp[_ok] - _pv[_ok];  _xok = _x_com[_ok]
            print(f"\n  Over log10(M*/Msun) = {_xok.min():.1f} to {_xok.max():.1f}")
            print(f"  Mean difference (SAGE26 - SAGE16):  {np.mean(_d):+.3f} dex")
            print(f"  Median difference:                  {np.median(_d):+.3f} dex")
            print(f"  Std of difference:                  {np.std(_d):.3f} dex")
            print(f"  Max offset at log10(M*)={_xok[np.argmax(_d)]:.2f}: {np.max(_d):+.3f} dex")
            print(f"  Min offset at log10(M*)={_xok[np.argmin(_d)]:.2f}: {np.min(_d):+.3f} dex")
            print(f"\n  phi(Q) at specific stellar masses:")
            for _tm in [9.0, 9.5, 10.0, 10.5, 11.0, 11.5]:
                _ti = np.argmin(np.abs(_xok - _tm))
                if np.abs(_xok[_ti] - _tm) < binwidth:
                    print(f"    log10(M*)={_xok[_ti]:.2f}: SAGE26={_pp[_ok][_ti]:.2f}, "
                          f"SAGE16={_pv[_ok][_ti]:.2f}, Δ={_d[_ti]:+.2f} dex")
    print("="*60 + "\n")

    # --- Observations ---
    gama = load_gama_smf_morph()
    baldry = load_baldry_blue_red()

    # --- Plot ---
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # SAGE26 (Lines + Shading)
    # Quiescent
    ax.plot(x, phi_q, color='firebrick', lw=3, label='SAGE26 Quiescent', zorder=10)
    ax.fill_between(x, phi_q_lo, phi_q_hi, color='firebrick', alpha=0.3, edgecolor='none', zorder=10)
    
    # Star-forming
    # ax.plot(x, phi_sf, color='steelblue', lw=3, label='SAGE26 Star-forming')
    # ax.fill_between(x, phi_sf_lo, phi_sf_hi, color='steelblue', alpha=0.3, edgecolor='none', zorder=10)
    # C16 (vanilla)
    ax.plot(x_v, phi_q_v, color='firebrick', lw=2, ls='--', label='SAGE16 Quiescent')
    # ax.plot(x_v, phi_sf_v, color='steelblue', lw=2, ls='--', label='C16 Star-forming')

    # Observational data: GAMA (Moffett+16) with 'd' markers
    valid_D = ~np.isnan(gama['D'])
    valid_E = ~np.isnan(gama['E_HE'])
    # ax.errorbar(gama['mass'][valid_D], gama['D'][valid_D],
    #             yerr=gama['D_err'][valid_D],
    #             fmt='d', color='b', ms=10, lw=1.5, capsize=2,
    #             label='Moffett+16')
    ax.errorbar(gama['mass'][valid_E], gama['E_HE'][valid_E],
                yerr=gama['E_HE_err'][valid_E], markeredgecolor='k', markerfacecolor='gray',
                fmt='d', color='k', ms=8, lw=1,label='Moffett+16', alpha=0.6, zorder=9)

    # Observational data: Baldry+12 with 'o' markers
    # ax.plot(baldry['sf_mass'], baldry['sf_phi'],
    #         'o', color='b', ms=10, label='Baldry+12')
    ax.scatter(baldry['q_mass'], baldry['q_phi'], edgecolor='k', facecolor='gray', marker=
            'o', color='k', s=50, label='Baldry+12', alpha=0.6, zorder=9)
    
    # Load Bell+03 SMF quiescent data
    bell_mass, bell_phi, bell_err_hi, bell_err_lo = load_bell_smf_q_data()
    if bell_mass is not None:
        ax.errorbar(bell_mass, bell_phi,
                    yerr=[bell_err_lo, bell_err_hi],
                    markeredgecolor='k', markerfacecolor='gray',
                    fmt='s', color='k', ms=8, lw=1, alpha=0.6, zorder=9,
                    label='Bell+03')

    ax.set_xlim(8, 12)
    ax.set_ylim(-6, -1)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.set_ylabel(r'$\log_{10}\ \phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$')
    ax.set_xlabel(r'$\log_{10}\ m_{\mathrm{*}}\ [M_{\odot}]$')

    handles, labels = ax.get_legend_handles_labels()
    sim_h = [h for h, l in zip(handles, labels) if l.startswith(('SAGE26', 'SAGE16'))]
    sim_l = [l for l in labels if l.startswith(('SAGE26', 'SAGE16'))]
    obs_h = [h for h, l in zip(handles, labels) if l.startswith('Baldry') or l.startswith('Moffett') or l.startswith('Bell')]
    obs_l = [l for l in labels if l.startswith('Baldry') or l.startswith('Moffett') or l.startswith('Bell')]
    leg1 = _standard_legend(ax, loc='lower left', handles=sim_h, labels=sim_l)
    ax.add_artist(leg1)
    _standard_legend(ax, loc='upper right', handles=obs_h, labels=obs_l)
    fig.tight_layout()

    save_figure(fig, os.path.join(OUTPUT_DIR,
                'StellarMassFunction_Q' + OUTPUT_FORMAT))


def plot_1_stellar_mass_function_ssfr_combined(primary, vanilla):
    """
    Star-forming and quiescent stellar mass functions side by side.

    Combines StellarMassFunction_SF and StellarMassFunction_Q into a single
    1x2 figure. Each panel keeps the same style and formatting as the
    standalone plots (colours, markers, bootstrap shading, observations).
    """
    print('Plot 1 (combined): Stellar mass function (SF | Q) with Bootstrap Errors')

    binwidth = 0.1
    N_BOOT = 100  # Number of bootstrap samples

    # --- Primary model ---
    w = primary['StellarMass'] > 0
    mass = np.log10(primary['StellarMass'][w])
    ssfr = log_ssfr(primary['SfrDisk'][w], primary['SfrBulge'][w],
                     primary['StellarMass'][w])

    # Establish common bins from the total MF, then split populations
    x, _, mrange = mass_function(mass, VOLUME, binwidth)
    mass_q = mass[ssfr < SSFR_CUT]
    mass_sf = mass[ssfr > SSFR_CUT]
    _, phi_q, _ = mass_function(mass_q, VOLUME, binwidth, mass_range=mrange)
    _, phi_sf, _ = mass_function(mass_sf, VOLUME, binwidth, mass_range=mrange)

    # Bootstrap error calculation (identical to the standalone plots)
    def calc_bootstrap_errors(data_mass, m_range, vol, bw, n_boot=100):
        if len(data_mass) == 0:
            return np.nan, np.nan
        mi, ma = m_range
        nbins = int(round((ma - mi) / bw))
        edges = np.linspace(mi, ma, nbins + 1)
        boot_phis = []
        n_obj = len(data_mass)
        for _ in range(n_boot):
            sample = data_mass[np.random.randint(0, n_obj, n_obj)]
            counts, _ = np.histogram(sample, bins=edges)
            with np.errstate(divide='ignore'):
                phi = np.log10(counts / vol / bw)
            phi[~np.isfinite(phi)] = np.nan
            boot_phis.append(phi)
        boot_phis = np.array(boot_phis)
        lo = np.nanpercentile(boot_phis, 16, axis=0)
        hi = np.nanpercentile(boot_phis, 84, axis=0)
        return lo, hi

    print(f'  Bootstrapping SAGE26 data ({N_BOOT} iterations)...')
    phi_q_lo, phi_q_hi = calc_bootstrap_errors(mass_q, mrange, VOLUME, binwidth, N_BOOT)
    phi_sf_lo, phi_sf_hi = calc_bootstrap_errors(mass_sf, mrange, VOLUME, binwidth, N_BOOT)

    # --- Vanilla model ---
    w2 = vanilla['StellarMass'] > 0
    mass_v = np.log10(vanilla['StellarMass'][w2])
    ssfr_v = log_ssfr(vanilla['SfrDisk'][w2], vanilla['SfrBulge'][w2],
                       vanilla['StellarMass'][w2])
    x_v, _, mrange_v = mass_function(mass_v, VOLUME, binwidth)
    _, phi_q_v, _ = mass_function(mass_v[ssfr_v < SSFR_CUT], VOLUME, binwidth,
                                  mass_range=mrange_v)
    _, phi_sf_v, _ = mass_function(mass_v[ssfr_v > SSFR_CUT], VOLUME, binwidth,
                                   mass_range=mrange_v)

    # --- Observations ---
    gama = load_gama_smf_morph()
    baldry = load_baldry_blue_red()

    # --- Plot ---
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # SAGE26 star-forming (steelblue) and quiescent (firebrick), with bootstrap shading
    ax.plot(x, phi_sf, color='steelblue', lw=4, label='SAGE26 Star-forming', zorder=10)
    ax.fill_between(x, phi_sf_lo, phi_sf_hi, color='steelblue', alpha=0.3, edgecolor='none', zorder=10)
    ax.plot(x, phi_q, color='firebrick', lw=3, label='SAGE26 Quiescent', zorder=10)
    ax.fill_between(x, phi_q_lo, phi_q_hi, color='firebrick', alpha=0.3, edgecolor='none', zorder=10)

    # SAGE16 (vanilla)
    ax.plot(x_v, phi_sf_v, color='steelblue', lw=2, ls='--', label='SAGE16 Star-forming')
    ax.plot(x_v, phi_q_v, color='firebrick', lw=2, ls='--', label='SAGE16 Quiescent')

    # Observations -- grey markers keep their per-source shapes (Moffett=diamond,
    # Baldry=circle, Bell=square); outlined by population (blue=SF, red=Q) so the
    # two are distinguishable on a single panel.
    valid_D = ~np.isnan(gama['D'])
    valid_E = ~np.isnan(gama['E_HE'])
    # Star-forming observations (blue outline)
    ax.errorbar(gama['mass'][valid_D], gama['D'][valid_D], yerr=gama['D_err'][valid_D],
                fmt='d', color='gray', markeredgecolor='steelblue', markeredgewidth=1.0,
                linewidth=1.0, markerfacecolor='gray', ms=8, alpha=0.6, zorder=9)
    ax.scatter(baldry['sf_mass'], baldry['sf_phi'], edgecolor='steelblue', facecolor='gray',
               marker='o', s=50, alpha=0.6, zorder=9)
    bell_mass, bell_phi, bell_err_hi, bell_err_lo = load_bell_smf_sf_data()
    if bell_mass is not None:
        ax.errorbar(bell_mass, bell_phi, yerr=[bell_err_lo, bell_err_hi],
                    markeredgecolor='steelblue', markerfacecolor='gray',
                    fmt='s', color='gray', ms=8, lw=1.0, alpha=0.6, zorder=9)
    # Quiescent observations (red outline)
    ax.errorbar(gama['mass'][valid_E], gama['E_HE'][valid_E], yerr=gama['E_HE_err'][valid_E],
                fmt='d', color='gray', markeredgecolor='firebrick', markerfacecolor='gray',
                ms=8, lw=1, alpha=0.6, zorder=9)
    ax.scatter(baldry['q_mass'], baldry['q_phi'], edgecolor='firebrick', facecolor='gray',
               marker='o', s=50, alpha=0.6, zorder=9)
    bell_mass, bell_phi, bell_err_hi, bell_err_lo = load_bell_smf_q_data()
    if bell_mass is not None:
        ax.errorbar(bell_mass, bell_phi, yerr=[bell_err_lo, bell_err_hi],
                    markeredgecolor='firebrick', markerfacecolor='gray',
                    fmt='s', color='gray', ms=8, lw=1, alpha=0.6, zorder=9)

    # Combined observations legend: one grey marker per source (SF and Q share an entry)
    obs_h = [
        plt.Line2D([], [], marker='d', color='gray', markerfacecolor='gray',
                   markeredgecolor='k', ms=8, lw=0, alpha=0.6),
        plt.Line2D([], [], marker='o', color='gray', markerfacecolor='gray',
                   markeredgecolor='k', ms=8, lw=0, alpha=0.6),
        plt.Line2D([], [], marker='s', color='gray', markerfacecolor='gray',
                   markeredgecolor='k', ms=8, lw=0, alpha=0.6),
    ]
    obs_l = ['Moffett+16', 'Baldry+12', 'Bell+03']

    ax.set_xlim(8, 12)
    ax.set_ylim(-6, -1)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.set_ylabel(r'$\log_{10}\ \phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$')
    ax.set_xlabel(r'$\log_{10}\ m_{\mathrm{*}}\ [M_{\odot}]$')

    handles, labels = ax.get_legend_handles_labels()
    sim_h = [h for h, l in zip(handles, labels) if l.startswith(('SAGE26', 'SAGE16'))]
    sim_l = [l for l in labels if l.startswith(('SAGE26', 'SAGE16'))]
    leg1 = _standard_legend(ax, loc='lower left', handles=sim_h, labels=sim_l)
    ax.add_artist(leg1)
    _standard_legend(ax, loc='upper right', handles=obs_h, labels=obs_l)
    fig.tight_layout()

    save_figure(fig, os.path.join(OUTPUT_DIR,
                'StellarMassFunction_SF_Q' + OUTPUT_FORMAT))

# ========================== PLOT 2: BARYON FRACTION vs HALO MASS ==========================

def plot_2_baryon_fraction(primary, vanilla):
    """
    Mean baryon component fractions vs halo mass.

    Shows how baryons are partitioned into stars, cold gas, hot gas,
    CGM, intracluster stars, black holes, and ejected gas as a
    function of halo virial mass.
    """
    print('Plot 2: Baryon fraction vs halo mass')

    mass_centers, bf = baryon_fractions_by_halo_mass(primary)

    # Component plotting config: (key, label, color, linestyle)
    components = [
        ('Total',             'Total',          'black',     '-'),
        ('StellarMass',       'Stars',          'magenta',   '--'),
        ('ColdGas',           'Cold gas',       'blue',      ':'),
        ('HotGas',            'Hot gas',        'red',       '-'),
        ('CGMgas',            'CGM',            'green',     '-.'),
        ('IntraClusterStars', 'ICS',            'orange',    '-.'),
        ('BlackHoleMass',     'Black holes',    'purple',    ':'),
        ('EjectedMass',       'Ejected gas',    'goldenrod', '--'),
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Universal baryon fraction reference line
    ax.axhline(y=BARYON_FRAC, color='grey', ls='--', lw=1.0,
               label=rf'$f_{{b}}$ = {BARYON_FRAC:.2f}')

    # Plot each component with shading
    for key, label, color, ls in components:
        ax.fill_between(mass_centers,
                        bf[key]['lower'], bf[key]['upper'],
                        color=color, alpha=0.3)
        ax.plot(mass_centers, bf[key]['mean'],
                color=color, ls=ls, lw=2, label=label)

    ax.set_xlim(11.1, 15.0)
    ax.set_ylim(0.0, 0.20)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.01))
    ax.set_xlabel(r'$\log_{10}\ M_{\mathrm{vir}}\ [M_{\odot}]$')
    ax.set_ylabel(r'Baryon Fraction')

    _standard_legend(ax, loc='center right')
    fig.tight_layout()

    save_figure(fig, os.path.join(OUTPUT_DIR,
                'BaryonFraction' + OUTPUT_FORMAT))


# ========================== PLOT 3: GAS METALLICITY vs STELLAR MASS ==========================

def plot_3_gas_metallicity_vs_stellar_mass(primary, vanilla):
    """
    Gas-phase metallicity vs. stellar mass distribution.

    Shows the distribution of galaxies in the metallicity-mass plane
    as a KDE contour plot, with observational data overplotted.
    """
    print('Plot 3: Gas metallicity vs stellar mass')

    # --- Primary model ---
    w = ((primary['StellarMass'] > 1e8)
         & (primary['ColdGas'] / (primary['StellarMass'] + primary['ColdGas']) > 0.1)
         & (primary['MetalsColdGas'] > 0))
    log_mass = np.log10(primary['StellarMass'][w])
    gas_Z = metallicity_12logOH(primary['MetalsColdGas'][w],
                                primary['ColdGas'][w])

    # --- Plot ---
    fig = plt.figure()
    ax = fig.add_subplot(111)

    mass_bins = np.arange(8.0, 12.0 + 0.1, 0.1)
    plot_binned_median_1sigma(
        ax, log_mass, gas_Z, mass_bins,
        color='steelblue', label='SAGE26',
        alpha=0.25, lw=3.5, min_count=50,
        zorder_fill=Z_MODEL_BAND, zorder_line=Z_MODEL_LINE,
    )

    # --- C16 (Vanilla) model ---
    w_v = ((vanilla['StellarMass'] > 1e8)
           & (vanilla['ColdGas'] > 0)
           & (vanilla['MetalsColdGas'] > 0))
    if np.any(w_v):
        log_mass_v = np.log10(vanilla['StellarMass'][w_v])
        gas_Z_v = metallicity_12logOH(vanilla['MetalsColdGas'][w_v],
                                      vanilla['ColdGas'][w_v])
        plot_binned_median_1sigma(
            ax, log_mass_v, gas_Z_v, mass_bins,
            color='purple', label='SAGE16', ls='--',
            alpha=0.20, lw=3.0, min_count=50,
            zorder_fill=Z_MODEL_BAND_ALT, zorder_line=Z_MODEL_LINE_ALT,
        )

    # --- Observational data ---
    for obs in load_mzr_observations():
        if obs['yerr'] is not None:
            ax.errorbar(obs['mass'], obs['Z'], yerr=obs['yerr'],
                        fmt=obs['fmt'], color=obs['color'],
                        markeredgecolor='k', markeredgewidth=1.0, linewidth=1.0,
                        markerfacecolor = 'gray', ms=8,
                        label=obs['label'], alpha=0.6, zorder=Z_OBS)
        else:
            ax.plot(obs['mass'], obs['Z'], obs['fmt'],
                    markeredgecolor='k', markeredgewidth=1.0, linewidth=1.0,
                    markerfacecolor = 'gray', ms=8,
                    color=obs['color'], label=obs['label'], alpha=0.6,
                    zorder=Z_OBS)

    ax.set_xlim(8.0, 12.0)
    ax.set_ylim(7.5, 9.5)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.set_xlabel(r'$\log_{10}\ m_{\mathrm{*}}\ [M_{\odot}]$')
    ax.set_ylabel(r'$12\ +\ \log_{10}\ (\mathrm{O/H})$')

    handles, labels = ax.get_legend_handles_labels()
    sim_set = {'SAGE26', 'SAGE16'}
    sim_h = [h for h, l in zip(handles, labels) if l in sim_set]
    sim_l = [l for l in labels if l in sim_set]
    obs_h = [h for h, l in zip(handles, labels) if l not in sim_set]
    obs_l = [l for l in labels if l not in sim_set]
    leg1 = _standard_legend(ax, loc='lower right', handles=sim_h, labels=sim_l)
    ax.add_artist(leg1)
    _standard_legend(ax, loc='upper left', handles=obs_h, labels=obs_l)
    fig.tight_layout()

    save_figure(fig, os.path.join(OUTPUT_DIR,
                'MetallicityStellarMass' + OUTPUT_FORMAT))


# ========================== PLOT 4: BLACK HOLE - BULGE MASS RELATION ==========================

def plot_4_bh_bulge_mass(primary, vanilla):
    """
    Black hole mass vs. bulge mass relation.

    Shows the distribution of galaxies in the BH-bulge mass plane
    as a KDE contour plot, with observational data overplotted.
    """
    print('Plot 4: Black hole - bulge mass relation')

    # --- Primary model ---
    w = (primary['BlackHoleMass'] > 0) & (primary['BulgeMass'] > 0)
    log_bulge = np.log10(primary['BulgeMass'][w])
    log_bh = np.log10(primary['BlackHoleMass'][w])

    # --- Plot ---
    fig = plt.figure()
    ax = fig.add_subplot(111)

    bulge_bins = np.arange(8.0, 12.0 + 0.1, 0.1)
    plot_binned_median_1sigma(
        ax, log_bulge, log_bh, bulge_bins,
        color='steelblue', label='SAGE26',
        alpha=0.25, lw=3.5, min_count=50,
        zorder_fill=Z_MODEL_BAND, zorder_line=Z_MODEL_LINE,
    )

    # --- C16 (Vanilla) model ---
    w_v = (vanilla['BlackHoleMass'] > 0) & (vanilla['BulgeMass'] > 0)
    if np.any(w_v):
        log_bulge_v = np.log10(vanilla['BulgeMass'][w_v])
        log_bh_v = np.log10(vanilla['BlackHoleMass'][w_v])
        plot_binned_median_1sigma(
            ax, log_bulge_v, log_bh_v, bulge_bins,
            color='purple', label='SAGE16', ls='--',
            alpha=0.20, lw=3.0, min_count=50,
            zorder_fill=Z_MODEL_BAND_ALT, zorder_line=Z_MODEL_LINE_ALT,
        )

    # --- Observational data ---
    obs = load_bh_bulge_observations()
    sersic = ~obs['core']

    ax.errorbar(obs['log_M_sph'][sersic], obs['log_M_BH'][sersic],
                yerr=[obs['yerr'][0][sersic], obs['yerr'][1][sersic]],
                xerr=[obs['xerr'][0][sersic], obs['xerr'][1][sersic]],
                color='k', ls='none', lw=1, marker='d', ms=8, alpha=0.6,
                zorder=Z_OBS,
                markeredgecolor='k', markeredgewidth=0.8,
                        markerfacecolor = 'gray',
                label='S13 core')
    ax.errorbar(obs['log_M_sph'][obs['core']], obs['log_M_BH'][obs['core']],
                yerr=[obs['yerr'][0][obs['core']], obs['yerr'][1][obs['core']]],
                xerr=[obs['xerr'][0][obs['core']], obs['xerr'][1][obs['core']]],
                color='k', ls='none', lw=1, marker='o', ms=8,
                markeredgecolor='k', markeredgewidth=0.8, alpha=0.6,
                zorder=Z_OBS,
                        markerfacecolor = 'gray',
                label=_tex_safe(r'S13 S\'{e}rsic'))

    ax.plot(obs['haring_rix_x'], obs['haring_rix_y'], 'k--', zorder=Z_OBS,
            label=_tex_safe(r'Haring \& Rix 2004'))

    ax.set_xlim(8.0, 12.0)
    ax.set_ylim(6.0, 10.0)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.set_xlabel(r'$\log_{10}\ m_{\mathrm{bulge}}\ [M_{\odot}]$')
    ax.set_ylabel(r'$\log_{10}\ m_{\mathrm{BH}}\ [M_{\odot}]$')

    handles, labels = ax.get_legend_handles_labels()
    sim_set = {'SAGE26', 'SAGE16'}
    sim_h = [h for h, l in zip(handles, labels) if l in sim_set]
    sim_l = [l for l in labels if l in sim_set]
    obs_h = [h for h, l in zip(handles, labels) if l not in sim_set]
    obs_l = [l for l in labels if l not in sim_set]
    leg1 = _standard_legend(ax, loc='upper left', handles=obs_h, labels=obs_l)
    ax.add_artist(leg1)
    _standard_legend(ax, loc='lower right', handles=sim_h, labels=sim_l)
    fig.tight_layout()

    save_figure(fig, os.path.join(OUTPUT_DIR,
                'BlackHoleBulgeMass' + OUTPUT_FORMAT))


# ========================== PLOT 5: STELLAR-TO-HALO MASS RELATION ==========================

def plot_5_stellar_halo_mass(primary, vanilla):
    """
    Stellar mass vs. halo virial mass relation.

    Shows the SAGE26 distribution as a KDE contour plot
    with C16 as a scatter overlay and observational data.
    """
    print('Plot 5: Stellar-to-halo mass relation')

    # --- Primary model ---
    w = (primary['StellarMass'] > 0) & (primary['Mvir'] > 0)
    log_mvir = np.log10(primary['Mvir'][w])
    log_mstar = np.log10(primary['StellarMass'][w])

    # --- Plot ---
    fig = plt.figure()
    ax = fig.add_subplot(111)

    mvir_bins = np.arange(10.0, 15.0 + 0.1, 0.1)
    plot_binned_median_1sigma(
        ax, log_mvir, log_mstar, mvir_bins,
        color='steelblue', label='SAGE26',
        alpha=0.25, lw=3.5, min_count=50,
        zorder_fill=Z_MODEL_BAND, zorder_line=Z_MODEL_LINE,
    )

    # --- C16 (Vanilla) model ---
    w_v = (vanilla['StellarMass'] > 0) & (vanilla['Mvir'] > 0)
    if np.any(w_v):
        log_mvir_v = np.log10(vanilla['Mvir'][w_v])
        log_mstar_v = np.log10(vanilla['StellarMass'][w_v])
        plot_binned_median_1sigma(
            ax, log_mvir_v, log_mstar_v, mvir_bins,
            color='purple', label='SAGE16', ls='--',
            alpha=0.20, lw=3.0, min_count=50,
            zorder_fill=Z_MODEL_BAND_ALT, zorder_line=Z_MODEL_LINE_ALT,
        )

    # --- Observational data ---
    obs = load_shmr_observations()

    if 'moster' in obs:
        ax.plot(obs['moster']['mvir'], obs['moster']['mstar'],
                'k-', lw=2, label='Moster+13', zorder=Z_OBS)

    if 'romeo' in obs:
        ax.scatter(obs['romeo']['mvir'], obs['romeo']['mstar'],
                   marker='o', s=50, c='gray', label='Romeo+20',
                   edgecolor='k', linewidth=0.8, alpha=0.6, zorder=Z_OBS)

    if 'kravtsov' in obs:
        k = obs['kravtsov']
        xerr = [k['xerr_lo'], k['xerr_hi']]
        ax.errorbar(k['mvir'], k['mstar'], xerr=xerr,
                    fmt='s', color='k', ms=8, lw=1,
                    markeredgecolor='k', markeredgewidth=0.8,
                    markerfacecolor = 'gray', alpha=0.6, zorder=Z_OBS,
                    label='Kravtsov+18')

    if 'taylor' in obs:
        t = obs['taylor']
        ax.errorbar(t['mvir'], t['mstar'],
                    xerr=t['xerr'], yerr=t['yerr'],
                    fmt='d', color='k', ms=8, lw=1,
                    markeredgecolor='k', markeredgewidth=0.8,
                    markerfacecolor = 'gray', alpha=0.6, zorder=Z_OBS,
                    label='Taylor+20')

    ax.set_xlim(10.0, 15.0)
    ax.set_ylim(8.0, 12.0)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.set_xlabel(r'$\log_{10}\ M_{\mathrm{vir}}\ [M_{\odot}]$')
    ax.set_ylabel(r'$\log_{10}\ m_{\mathrm{*}}\ [M_{\odot}]$')

    handles, labels = ax.get_legend_handles_labels()
    sim_set = {'SAGE26', 'SAGE16'}
    sim_h = [h for h, l in zip(handles, labels) if l in sim_set]
    sim_l = [l for l in labels if l in sim_set]
    obs_h = [h for h, l in zip(handles, labels) if l not in sim_set]
    obs_l = [l for l in labels if l not in sim_set]
    leg1 = _standard_legend(ax, loc='upper left', handles=sim_h, labels=sim_l)
    ax.add_artist(leg1)
    _standard_legend(ax, loc='lower right', handles=obs_h, labels=obs_l)
    fig.tight_layout()

    save_figure(fig, os.path.join(OUTPUT_DIR,
                'StellarHaloMass' + OUTPUT_FORMAT))


# ========================== PLOT 5b: STELLAR-TO-HALO MASS RATIO ==========================

def plot_5b_stellar_halo_mass_ratio(primary, vanilla):
    """
    Stellar-to-halo mass ratio m*/M_vir against halo virial mass at z = 0.

    The ratio form of plot 5. Dividing out M_vir removes the near-unit slope that
    dominates the m*-M_vir plane and leaves the peak and the two falling wings, which is
    where the models actually differ from each other and from the data.

    Three curves: SAGE26 on Millennium and on miniUchuu, plus SAGE16. The two SAGE26
    curves are the same physics on different resolutions -- miniUchuu resolves haloes
    roughly 2.6x lighter -- so the low-mass end of the pair is a resolution check rather
    than a physics comparison, and is drawn down to whatever each run resolves.

    miniUchuu is read with its own mass conversion (its h differs from Millennium's);
    read_snap_from_files() applies the MIN_PARTICLES cut to both.
    """
    print('Plot 5b: Stellar-to-halo mass ratio')

    mvir_bins = np.arange(10.0, 15.0 + 0.1, 0.1)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    def _ratio(d):
        """log10(M_vir) and log10(m*/M_vir) for the resolved, star-forming-or-not set."""
        if not d:
            return None, None
        w = (d['StellarMass'] > 0) & (d['Mvir'] > 0)
        if not np.any(w):
            return None, None
        lm = np.log10(d['Mvir'][w])
        return lm, np.log10(d['StellarMass'][w]) - lm

    # --- SAGE26, Millennium ---
    x, y = _ratio(primary)
    if x is not None:
        plot_binned_median_1sigma(
            ax, x, y, mvir_bins,
            color='steelblue', label='SAGE26 (Millennium)',
            alpha=0.25, lw=3.5, min_count=50,
            zorder_fill=Z_MODEL_BAND, zorder_line=Z_MODEL_LINE,
        )

    # --- SAGE26, miniUchuu ---
    if os.path.exists(MINIUCHUU_DIR):
        mu_files = find_model_files(MINIUCHUU_DIR)
        mu = read_snap_from_files(mu_files, f'Snap_{MINIUCHUU_LAST_SNAP}',
                                  ['StellarMass', 'Mvir'],
                                  mass_convert=MINIUCHUU_MASS_CONVERT) if mu_files else {}
        x, y = _ratio(mu)
        if x is not None:
            plot_binned_median_1sigma(
                ax, x, y, mvir_bins,
                color='darkorange', label='SAGE26 (miniUchuu)', ls='-.',
                alpha=0.18, lw=3.0, min_count=50,
                zorder_fill=Z_MODEL_BAND, zorder_line=Z_MODEL_LINE,
            )
        else:
            print('  miniUchuu: no usable z = 0 snapshot -- curve omitted')

    # --- SAGE16 ---
    x, y = _ratio(vanilla)
    if x is not None:
        plot_binned_median_1sigma(
            ax, x, y, mvir_bins,
            color='purple', label='SAGE16', ls='--',
            alpha=0.20, lw=3.0, min_count=50,
            zorder_fill=Z_MODEL_BAND_ALT, zorder_line=Z_MODEL_LINE_ALT,
        )

    # --- Observations ---
    obs = load_shmr_observations()

    if 'moster' in obs:
        ax.plot(obs['moster']['mvir'],
                obs['moster']['mstar'] - obs['moster']['mvir'],
                'k-', lw=2, label='Moster+13', zorder=Z_OBS)

    if 'romeo' in obs:
        ax.scatter(obs['romeo']['mvir'],
                   obs['romeo']['mstar'] - obs['romeo']['mvir'],
                   marker='o', s=50, c='gray', label='Romeo+20',
                   edgecolor='k', linewidth=0.8, alpha=0.6, zorder=Z_OBS)

    if 'kravtsov' in obs:
        k = obs['kravtsov']
        ax.errorbar(k['mvir'], k['mstar'] - k['mvir'],
                    xerr=[k['xerr_lo'], k['xerr_hi']],
                    fmt='s', color='k', ms=8, lw=1,
                    markeredgecolor='k', markeredgewidth=0.8,
                    markerfacecolor='gray', alpha=0.6, zorder=Z_OBS,
                    label='Kravtsov+18')

    if 'taylor' in obs:
        t = obs['taylor']
        ax.errorbar(t['mvir'], t['ratio'], xerr=t['xerr'], yerr=t['ratio_err'],
                    fmt='d', color='k', ms=8, lw=1,
                    markeredgecolor='k', markeredgewidth=0.8,
                    markerfacecolor='gray', alpha=0.6, zorder=Z_OBS,
                    label='Taylor+20')

    # Cosmic baryon fraction: the ceiling m*/M_vir cannot exceed if every accreted
    # baryon turned into a star, so it bounds the plot from above.
    ax.axhline(np.log10(BARYON_FRAC), color='0.45', ls=':', lw=1.4, zorder=Z_OBS - 1)
    ax.text(0.015, np.log10(BARYON_FRAC) + 0.06,
            r'$f_{\mathrm{b}}$: every accreted baryon into stars',
            transform=ax.get_yaxis_transform(), ha='left', va='bottom',
            fontsize=10, color='0.35')

    ax.set_xlim(10.0, 15.0)
    ax.set_ylim(-4.0, 0.0)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.set_xlabel(r'$\log_{10}\ M_{\mathrm{vir}}\ [M_{\odot}]$')
    ax.set_ylabel(r'$\log_{10}\ (m_{\mathrm{*}} / M_{\mathrm{vir}})$')

    handles, labels = ax.get_legend_handles_labels()
    sim_set = {'SAGE26 (Millennium)', 'SAGE26 (miniUchuu)', 'SAGE16'}
    sim_h = [h for h, l in zip(handles, labels) if l in sim_set]
    sim_l = [l for l in labels if l in sim_set]
    obs_h = [h for h, l in zip(handles, labels) if l not in sim_set]
    obs_l = [l for l in labels if l not in sim_set]
    leg1 = _standard_legend(ax, loc='lower right', handles=sim_h, labels=sim_l)
    ax.add_artist(leg1)
    _standard_legend(ax, loc='upper right', handles=obs_h, labels=obs_l)
    fig.tight_layout()

    save_figure(fig, os.path.join(OUTPUT_DIR,
                'StellarHaloMassRatio' + OUTPUT_FORMAT))

# ========================== PLOT 6: BULGE MASS-SIZE BY FORMATION TYPE ==========================

def plot_6_bulge_mass_size(primary, vanilla):
    """
    Bulge mass vs. effective radius, coloured by formation channel.

    Merger-dominated bulges (InstabilityBulgeMass/BulgeMass < 0.1),
    instability-dominated (ratio > 0.9), and mixed (0.1-0.9) are
    shown separately, with Shen+2003 and pseudo-bulge scaling relations.
    """
    print('Plot 6: Bulge mass-size by formation type')

    w = (primary['BulgeMass'] > 0) & (primary['BulgeRadius'] > 0)
    bulge_mass = primary['BulgeMass'][w]
    bulge_radius = primary['BulgeRadius'][w] / HUBBLE_H / 0.001  # kpc
    inst_ratio = primary['InstabilityBulgeMass'][w] / bulge_mass

    merger_mask = inst_ratio < 0.1
    inst_mask = inst_ratio > 0.9
    mixed_mask = (inst_ratio >= 0.1) & (inst_ratio <= 0.9)

    n_tot = np.sum(w)
    print(f'  Total galaxies with bulges: {n_tot}')
    print(f'  Merger-dominated (ratio<0.1): {np.sum(merger_mask)}'
          f' ({100*np.sum(merger_mask)/n_tot:.1f}%)')
    print(f'  Instability-dominated (ratio>0.9): {np.sum(inst_mask)}'
          f' ({100*np.sum(inst_mask)/n_tot:.1f}%)')
    print(f'  Mixed (0.1-0.9): {np.sum(mixed_mask)}'
          f' ({100*np.sum(mixed_mask)/n_tot:.1f}%)')

    # Subsample for plotting
    def _subsample(mask, n):
        idx = np.where(mask)[0]
        if len(idx) > n:
            idx = np.random.choice(idx, n, replace=False)
        return idx

    merger_idx = _subsample(merger_mask, DILUTE)
    inst_idx = _subsample(inst_mask, DILUTE)
    mixed_idx = _subsample(mixed_mask, DILUTE // 2)

    log_mass_m = np.log10(bulge_mass[merger_idx])
    log_rad_m = np.log10(bulge_radius[merger_idx])
    log_mass_i = np.log10(bulge_mass[inst_idx])
    log_rad_i = np.log10(bulge_radius[inst_idx])
    log_mass_x = np.log10(bulge_mass[mixed_idx])
    log_rad_x = np.log10(bulge_radius[mixed_idx])

    # --- Plot ---
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(log_mass_m, log_rad_m, c='orangered', s=20, alpha=0.6,
               edgecolors='darkred', linewidths=0.3, label='Merger-driven',
               rasterized=True)
    ax.scatter(log_mass_i, log_rad_i, c='steelblue', s=20, alpha=0.6,
               edgecolors='darkblue', linewidths=0.3, label='Instability-driven',
               rasterized=True)
    ax.scatter(log_mass_x, log_rad_x, c='mediumorchid', s=15, alpha=0.4,
               edgecolors='purple', linewidths=0.2, label='Mixed',
               rasterized=True)

    # Theoretical relations
    log_M = np.linspace(8, 12, 100)
    ax.plot(log_M, 0.56 * log_M - 5.54, 'k--', lw=2,
            label='Shen+2003 (classical)', zorder=10)
    ax.plot(log_M, 0.25 * log_M - 2.5, 'g--', lw=2, alpha=0.6,
            label='Pseudo-bulge (shallow)', zorder=10)

    ax.set_xlim(8.0, 12.0)
    ax.set_ylim(-2.0, 3.0)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.set_xlabel(r'$\log_{10}\ m_{\mathrm{bulge}}\ [M_{\odot}]$')
    ax.set_ylabel(r'$\log_{10}\ R_{\mathrm{bulge}}\ [\mathrm{kpc}]$')

    handles, labels = ax.get_legend_handles_labels()
    scatter_names = {'Merger-driven', 'Instability-driven', 'Mixed'}
    scat_h = [h for h, l in zip(handles, labels) if l in scatter_names]
    scat_l = [l for l in labels if l in scatter_names]
    line_h = [h for h, l in zip(handles, labels) if l not in scatter_names]
    line_l = [l for l in labels if l not in scatter_names]
    leg1 = _standard_legend(ax, loc='upper left', handles=scat_h, labels=scat_l)
    ax.add_artist(leg1)
    _standard_legend(ax, loc='lower right', handles=line_h, labels=line_l)
    fig.tight_layout()

    save_figure(fig, os.path.join(OUTPUT_DIR,
                'BulgeMassSize' + OUTPUT_FORMAT))


# ================= PLOT 6b: BULGE MASS-SIZE MEDIANS BY FORMATION TYPE =================

def plot_6b_bulge_mass_size_median(primary, vanilla):
    """
    As plot 6, but with binned medians and 16--84% (1sigma) bands
    instead of scatter points, one for each formation channel.
    """
    print('Plot 6b: Bulge mass-size medians by formation type')

    w = (primary['BulgeMass'] > 0) & (primary['BulgeRadius'] > 0)
    bulge_mass = primary['BulgeMass'][w]
    bulge_radius = primary['BulgeRadius'][w] / HUBBLE_H / 0.001  # kpc
    inst_ratio = primary['InstabilityBulgeMass'][w] / bulge_mass

    log_mass = np.log10(bulge_mass)
    log_rad = np.log10(bulge_radius)

    merger_mask = inst_ratio < 0.1
    inst_mask = inst_ratio > 0.9
    mixed_mask = (inst_ratio >= 0.1) & (inst_ratio <= 0.9)

    n_tot = np.sum(w)
    print(f'  Total galaxies with bulges: {n_tot}')
    print(f'  Merger-dominated (ratio<0.1): {np.sum(merger_mask)}'
          f' ({100*np.sum(merger_mask)/n_tot:.1f}%)')
    print(f'  Instability-dominated (ratio>0.9): {np.sum(inst_mask)}'
          f' ({100*np.sum(inst_mask)/n_tot:.1f}%)')
    print(f'  Mixed (0.1-0.9): {np.sum(mixed_mask)}'
          f' ({100*np.sum(mixed_mask)/n_tot:.1f}%)')

    # --- Plot ---
    fig = plt.figure()
    ax = fig.add_subplot(111)

    bins = np.arange(8.0, 12.01, 0.2)
    cmap = plt.cm.plasma
    colors = [cmap(x) for x in (0.05, 0.45, 0.8)]
    channels = [
        (merger_mask, colors[0], 'Merger-driven'),
        (inst_mask, colors[1], 'Instability-driven'),
        (mixed_mask, colors[2], 'Mixed'),
    ]
    for mask, color, label in channels:
        plot_binned_median_1sigma(ax, log_mass[mask], log_rad[mask], bins,
                                  color=color, label=label, alpha=0.25,
                                  min_count=10)

    # Theoretical relations
    log_M = np.linspace(8, 12, 100)
    ax.plot(log_M, 0.56 * log_M - 5.54, 'k--', lw=2,
            label='Shen+2003 (classical)', zorder=10)
    ax.plot(log_M, 0.25 * log_M - 2.5, ls='--', color='dimgray', lw=2,
            alpha=0.8, label='Pseudo-bulge (shallow)', zorder=10)

    ax.set_xlim(8.0, 12.0)
    ax.set_ylim(-2.0, 3.0)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.set_xlabel(r'$\log_{10}\ m_{\mathrm{bulge}}\ [M_{\odot}]$')
    ax.set_ylabel(r'$\log_{10}\ R_{\mathrm{bulge}}\ [\mathrm{kpc}]$')

    handles, labels = ax.get_legend_handles_labels()
    median_names = {'Merger-driven', 'Instability-driven', 'Mixed'}
    med_h = [h for h, l in zip(handles, labels) if l in median_names]
    med_l = [l for l in labels if l in median_names]
    line_h = [h for h, l in zip(handles, labels) if l not in median_names]
    line_l = [l for l in labels if l not in median_names]
    leg1 = _standard_legend(ax, loc='upper left', handles=med_h, labels=med_l)
    ax.add_artist(leg1)
    _standard_legend(ax, loc='lower right', handles=line_h, labels=line_l)
    fig.tight_layout()

    save_figure(fig, os.path.join(OUTPUT_DIR,
                'BulgeMassSize_Median' + OUTPUT_FORMAT))


# ========================== PLOT 7: t_cool/t_ff DISTRIBUTION ==========================

def plot_7_tcool_tff_distribution(snapdata):
    """
    Violin plot of log10(t_cool/t_ff) for CGM-regime haloes
    at z=4.2, 2.1, 1.2, 0.
    """
    print('Plot 7: t_cool/t_ff distribution')

    snap_info = [
        (SNAP_Z0, f'z = {REDSHIFTS[SNAP_Z0]:.1f}'),
        (SNAP_Z1, f'z = {REDSHIFTS[SNAP_Z1]:.1f}'),
        (SNAP_Z2, f'z = {REDSHIFTS[SNAP_Z2]:.1f}'),
        (SNAP_Z3, f'z = {REDSHIFTS[SNAP_Z3]:.1f}'),
        (SNAP_Z4, f'z = {REDSHIFTS[SNAP_Z4]:.1f}'),
    ]
    cmap_violin = plt.cm.plasma
    colors_violin = [cmap_violin(x) for x in np.linspace(0.0, 0.85, len(snap_info))]

    violin_data = []
    violin_positions = []
    violin_labels = []
    valid_colors = []

    for i, (snap, label) in enumerate(snap_info):
        if snap not in snapdata:
            continue
        d = snapdata[snap]
        ratio = d['tcool_over_tff']
        w = np.where(
            (d['Regime'] == 0) &
            (ratio > 0) & np.isfinite(ratio) &
            (d['Type'] == 0) &
            (d['Mvir'] > 1e10)
        )[0]

        if len(w) > 10:
            data = np.log10(ratio[w])
            data = data[np.isfinite(data)]
            data = np.clip(data, -2, 5)
            violin_data.append(data)
            violin_positions.append(i)
            violin_labels.append(label)
            valid_colors.append(colors_violin[i])

    if len(violin_data) == 0:
        print('  No valid CGM-regime data found. Skipping.')
        return

    fig, ax = plt.subplots()

    parts = ax.violinplot(violin_data, positions=violin_positions,
                          showmedians=True, showextrema=True, widths=0.7)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(valid_colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.6)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2)
    parts['cmins'].set_color('gray')
    parts['cmaxes'].set_color('gray')
    parts['cbars'].set_color('gray')

    # Precipitation threshold
    ax.axhline(y=np.log10(10), color='black', ls='--', lw=2,
               label=r'$t_{\rm cool}/t_{\rm ff} = 10$ (inflow threshold)')

    # Shaded precipitation zone
    ax.axhspan(np.log10(5), np.log10(20), alpha=0.12, color='gray',
               label='Inflow zone (5--20)')

    ax.set_xticks(violin_positions)
    ax.set_xticklabels(violin_labels)
    ax.set_ylabel(r'$\log_{10}(t_{\rm cool}/t_{\rm ff})$')
    ax.set_ylim(-3.5, 6.0)

    _standard_legend(ax, loc='lower left')
    fig.tight_layout()

    save_figure(fig, os.path.join(OUTPUT_DIR,
                'TcoolTffDistribution' + OUTPUT_FORMAT))


def plot_7b_inflow_transition_fraction(snapdata):
    """
    Two-panel figure answering the referee's question on how much work the
    precipitation criterion actually does.

    Left  -- f_inflow as a function of t_cool/t_ff, for the implemented form
             and for the bare sigmoid, with the transition band 0.1 < f < 0.9
             shaded and the z=0 halo distribution overlaid.
    Right -- fraction of the CGM-regime central population inside that
             transition band as a function of redshift, weighted by number and
             by CGM mass, together with the saturated fraction f >= 0.9.
    """
    print('Plot 7b: inflow transition fraction')

    def _cgm_selection(d):
        ratio = d['tcool_over_tff']
        return np.where(
            (d['Regime'] == 0) &
            (d['Type'] == 0) &
            (ratio > 0) & np.isfinite(ratio) &
            (d['CGMgas'] > 0) &
            (d['Mvir'] > 1e10)
        )[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # ---------------- Panel 1: the inflow fraction curve ----------------
    r_curve = np.logspace(-2.5, 1.6, 800)
    f_impl  = precipitation_fraction(r_curve, include_condensation=True)
    f_sig   = precipitation_fraction(r_curve, include_condensation=False)

    ax1.axhspan(0.1, 0.9, alpha=0.10, color='gray', zorder=0)
    ax1.plot(r_curve, f_sig, color='0.55', ls='--', lw=2.0,
             label=r'sigmoid only, $S\left(\frac{10-t_{\rm cool}/t_{\rm ff}}{2}\right)$')
    ax1.plot(r_curve, f_impl, color='#1f77b4', lw=3.0,
             label=r'$f_{\rm inflow}$ as implemented')
    ax1.axvline(PRECIP_THRESHOLD, color='black', ls=':', lw=1.8,
                label=r'$t_{\rm cool}/t_{\rm ff}=10$')

    # overlay the z=0 population so the reader sees where it sits
    if SNAP_Z0 in snapdata:
        d = snapdata[SNAP_Z0]
        w = _cgm_selection(d)
        if len(w) > 10:
            axh = ax1.twinx()
            axh.hist(d['tcool_over_tff'][w], bins=np.logspace(-2.5, 1.6, 60),
                     color='#d62728', alpha=0.22, density=True)
            axh.set_ylabel(r'$z=0$ halo density', color='#d62728')
            axh.tick_params(axis='y', colors='#d62728')
            axh.set_zorder(0)
            ax1.set_zorder(1)
            ax1.patch.set_visible(False)

    ax1.set_xscale('log')
    ax1.set_xlabel(r'$t_{\rm cool}/t_{\rm ff}$')
    ax1.set_ylabel(r'$f_{\rm inflow}$')
    ax1.set_ylim(-0.02, 1.05)
    _standard_legend(ax1, loc='lower left')

    # ---------------- Panel 2: transition fraction vs redshift ----------------
    zs, frac_n, frac_m, frac_sat = [], [], [], []
    for snap in sorted(snapdata.keys()):
        d = snapdata[snap]
        if 'tcool_over_tff' not in d or 'CGMgas' not in d:
            continue
        w = _cgm_selection(d)
        if len(w) < 100:
            continue
        r = np.asarray(d['tcool_over_tff'][w], dtype=float)
        m = np.asarray(d['CGMgas'][w], dtype=float)
        f = precipitation_fraction(r)
        band = (f > 0.1) & (f < 0.9)
        zs.append(REDSHIFTS[snap])
        frac_n.append(band.mean())
        frac_m.append(m[band].sum() / m.sum() if m.sum() > 0 else np.nan)
        frac_sat.append((f >= 0.9).mean())

    if len(zs) == 0:
        print('  No valid CGM-regime data found. Skipping.')
        plt.close(fig)
        return

    order = np.argsort(zs)
    zs = np.array(zs)[order]
    frac_n = np.array(frac_n)[order]
    frac_m = np.array(frac_m)[order]
    frac_sat = np.array(frac_sat)[order]

    ax2.plot(zs, frac_sat, 'o-', color='#2ca02c', lw=2.5, ms=5,
             label=r'saturated, $f_{\rm inflow}\geq0.9$')
    ax2.plot(zs, frac_n, 'o-', color='#1f77b4', lw=2.5, ms=5,
             label=r'transition, $0.1<f_{\rm inflow}<0.9$ (by number)')
    ax2.plot(zs, frac_m, 's--', color='#ff7f0e', lw=2.5, ms=5,
             label=r'transition (weighted by $m_{\rm CGM}$)')

    ax2.set_xlabel(r'$z$')
    ax2.set_ylabel('fraction of CGM-regime centrals')
    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(left=0)
    _standard_legend(ax2, loc='center right')

    fig.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR,
                'InflowTransitionFraction' + OUTPUT_FORMAT))


# ========================== PLOT 8: PRECIPITATION FRACTION MODEL ==========================

def plot_8_precipitation_fraction(snapdata):
    """
    Figure 8a: Precipitation fraction vs t_cool/t_ff: theoretical model curve
    with mass-stratified galaxy scatter at z=0 and z=2.

    Figure 8b: Median t_cool/t_ff vs halo mass at z=0 and z=2, showing
    the mass dependence of thermal stability directly.
    """
    print('Plot 8: Precipitation fraction model')

    snap_list = [(SNAP_Z0, 'z=0'), (SNAP_Z2, 'z=2'), (SNAP_Z3, 'z=3'), (SNAP_Z4, 'z=4')]
    markers   = ['x', 'o', '^', 's']
    zcolors   = ['steelblue', 'firebrick', 'darkorange', 'purple']
    mass_bins = np.arange(10.0, 15.5, 0.25)   # log10(Mvir/Msun) bin edges

    # ------------------------------------------------------------------ #
    # Figure 8a: f_precip vs t_cool/t_ff                                  #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots()

    ratio_arr = np.logspace(np.log10(0.5), 4.0, 2000)
    f_curve   = precipitation_fraction(ratio_arr)
    ax.plot(ratio_arr, f_curve, 'teal', lw=3,
            label='SAGE26 inflow model', zorder=5)

    ax.axvline(x=10, color='goldenrod', ls='--', lw=1.5, alpha=0.8,
               label=r'$t_{\rm cool}/t_{\rm ff} = 10$')
    ax.axvspan(0.5, 10,  alpha=0.06, color='red')
    ax.axvspan(10,  15,  alpha=0.06, color='goldenrod')
    ax.axvspan(12,  1e4, alpha=0.06, color='steelblue')
    ax.text(2.5,  0.55, 'Thermally\nUnstable', fontsize=12, ha='center',
            va='center', color='firebrick', fontweight='bold')
    ax.text(200,  0.55, 'Thermally\nStable',   fontsize=12, ha='center',
            va='center', color='steelblue',  fontweight='bold')

    # Mass-stratified sampling: N_per_bin points from each 0.5-dex mass bin
    N_PER_BIN = 80
    sc = None
    for (snap, label), mark, zcol in zip(snap_list, markers, zcolors):
        if snap not in snapdata:
            continue
        d     = snapdata[snap]
        ratio = d['tcool_over_tff']
        logm  = np.log10(d['Mvir'])   # Mvir already in 1e10 Msun units
        base  = np.where(
            (d['Regime'] == 0) & (ratio > 0) & np.isfinite(ratio) &
            (d['Type'] == 0)  & (d['Mvir'] > 1e10)
        )[0]

        sel = []
        for mlo, mhi in zip(mass_bins[:-1], mass_bins[1:]):
            inbin = base[(logm[base] > mlo) & (logm[base] <= mhi)]
            if len(inbin) > N_PER_BIN:
                inbin = np.random.choice(inbin, N_PER_BIN, replace=False)
            sel.append(inbin)
        if not sel:
            continue
        w      = np.concatenate(sel)
        r_vals = ratio[w]
        f_vals = precipitation_fraction(r_vals)
        sc = ax.scatter(r_vals, f_vals, s=150, alpha=0.6,
                        c=np.log10(d['Mvir'][w]), cmap='plasma',
                        vmin=10, vmax=14,
                        marker=mark, edgecolors='none', zorder=10,
                        label=label)

    if sc is not None:
        cbar = plt.colorbar(sc, ax=ax, pad=0.02, aspect=30)
        cbar.set_label(r'$\log_{10}(M_{\rm vir}/M_{\odot})$')

    ax.set_xscale('log')
    ax.set_xlim(0.5, 1e4)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r'$t_{\rm cool}/t_{\rm ff}$')
    ax.set_ylabel(r'$f_{\rm inflow}$')
    _standard_legend(ax, loc='upper right')
    fig.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR,
                'PrecipitationFraction' + OUTPUT_FORMAT))

    # ------------------------------------------------------------------ #
    # Figure 8b: Median t_cool/t_ff vs halo mass                          #
    # ------------------------------------------------------------------ #
    fig2, ax2 = plt.subplots()

    ax2.axhline(y=np.log10(10), color='goldenrod', ls='--', lw=2,
                label=r'$t_{\rm cool}/t_{\rm ff} = 10$ (threshold)')
    ax2.axhspan(np.log10(5), np.log10(20), alpha=0.10, color='goldenrod')


    for (snap, label), zcol, mark in zip(snap_list, zcolors, markers):
        if snap not in snapdata:
            continue
        d     = snapdata[snap]
        ratio = d['tcool_over_tff']
        logm  = np.log10(d['Mvir'])
        base  = np.where(
            (d['Regime'] == 0) & (ratio > 0) & np.isfinite(ratio) &
            (d['Type'] == 0)  & (d['Mvir'] > 1e10)
        )[0]

        meds, p16, p84, xc = [], [], [], []
        for mlo, mhi in zip(mass_bins[:-1], mass_bins[1:]):
            inbin = base[(logm[base] > mlo) & (logm[base] <= mhi)]
            if len(inbin) < 2:
                continue
            lr = np.log10(ratio[inbin])
            lr = lr[np.isfinite(lr)]
            if len(lr) < 5:
                continue
            meds.append(np.median(lr))
            p16.append(np.percentile(lr, 16))
            p84.append(np.percentile(lr, 84))
            xc.append(0.5 * (mlo + mhi))   # convert to log10(M/Msun)

        if not meds:
            continue
        xc   = np.array(xc)
        meds = np.array(meds)
        p16  = np.array(p16)
        p84  = np.array(p84)

        ax2.fill_between(xc, p16, p84, alpha=0.25, color=zcol)
        ax2.plot(xc, meds, color=zcol, lw=2, marker=mark,
                 ms=10, label=label)
    
    ax2.set_xlim(10.0, 12.5)

    ax2.set_xlabel(r'$\log_{10}(M_{\rm vir}/M_{\odot})$')
    ax2.set_ylabel(r'$\log_{10}(t_{\rm cool}/t_{\rm ff})$')
    _standard_legend(ax2, loc='best')
    fig2.tight_layout()
    save_figure(fig2, os.path.join(OUTPUT_DIR,
                'PrecipitationFractionMassTrend' + OUTPUT_FORMAT))


# ========================== PLOT 9: CGM FRACTIONS & DEPLETION ==========================

def plot_9_cgm_fractions_depletion(snapdata):
    """
    Two-panel: (left) CGM/hot gas fraction vs halo mass,
    (right) depletion timescale vs halo mass, at z=0, 2, 4.
    """
    print('Plot 9: CGM fractions and depletion timescales')

    snap_list = [
        (SNAP_Z0, f'z={REDSHIFTS[SNAP_Z0]:.0f}', '#1f77b4'),
        (SNAP_Z1, f'z={REDSHIFTS[SNAP_Z1]:.1f}', '#17becf'),
        (SNAP_Z2, f'z={REDSHIFTS[SNAP_Z2]:.1f}', '#2ca02c'),
        (SNAP_Z3, f'z={REDSHIFTS[SNAP_Z3]:.1f}', '#ff7f0e'),
        (SNAP_Z4, f'z={REDSHIFTS[SNAP_Z4]:.1f}', '#d62728'),
        (SNAP_Z5, f'z={REDSHIFTS[SNAP_Z5]:.1f}', '#9467bd'),
    ]

    mass_bins = np.arange(10.0, 15.0, 0.3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Detect CGM recipe
    cgm_active = (SNAP_Z0 in snapdata
                  and np.any(snapdata[SNAP_Z0]['tcool_over_tff'] > 0))

    # --- Panel 1: gas fractions ---
    if cgm_active:
        gas_label = r'$m_{\rm CGM}/M_{\rm vir}$'
        for snap, label, color in snap_list:
            if snap not in snapdata:
                continue
            d = snapdata[snap]
            w_cgm = np.where(
                (d['Regime'] == 0) & (d['Mvir'] > 1e10) &
                (d['CGMgas'] > 0) & (d['Type'] == 0)
            )[0]
            w_hot = np.where(
                (d['Regime'] == 1) & (d['Mvir'] > 1e10) &
                (d['HotGas'] > 0) & (d['Type'] == 0)
            )[0]

            if len(w_cgm) > 0:
                log_mv = np.log10(d['Mvir'][w_cgm])
                frac = d['CGMgas'][w_cgm] / d['Mvir'][w_cgm]
                bc, med, _, _ = binned_median(log_mv, frac, mass_bins)
                valid = ~np.isnan(med)
                ax1.plot(bc[valid], med[valid], '-o', color=color, lw=2,
                         markersize=5, label=f'CGM ({label})')

            if len(w_hot) > 0:
                log_mv = np.log10(d['Mvir'][w_hot])
                frac = d['HotGas'][w_hot] / d['Mvir'][w_hot]
                bc, med, _, _ = binned_median(log_mv, frac, mass_bins)
                valid = ~np.isnan(med)
                ax1.plot(bc[valid], med[valid], '--s', color=color, lw=2,
                         alpha=0.6, markersize=5, label=f'Hot ({label})')
    else:
        gas_label = r'$M_{\rm hot}/M_{\rm vir}$'
        for snap, label, color in snap_list:
            if snap not in snapdata:
                continue
            d = snapdata[snap]
            w = np.where(
                (d['Mvir'] > 1e10) & (d['HotGas'] > 0) & (d['Type'] == 0)
            )[0]
            if len(w) > 0:
                log_mv = np.log10(d['Mvir'][w])
                frac = d['HotGas'][w] / d['Mvir'][w]
                bc, med, _, _ = binned_median(log_mv, frac, mass_bins)
                valid = ~np.isnan(med)
                ax1.plot(bc[valid], med[valid], '-o', color=color, lw=2,
                         markersize=5, label=label)

    ax1.axhline(y=BARYON_FRAC, color='gray', ls=':', lw=1.5, alpha=1.0,
                label=r'$f_b = \Omega_b/\Omega_m$')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$\log_{10}(M_{\rm vir}/M_{\odot})$')
    ax1.set_ylabel(gas_label)
    ax1.set_xlim(10.2, 14.5)
    ax1.set_ylim(1e-4, 0.5)
    ax1.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    # ax1.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    _standard_legend(ax1, loc='lower right')

    # --- Panel 2: depletion timescales (bootstrap errors on the median) ---
    N_BOOT = 200
    for snap, label, color in snap_list:
        if snap not in snapdata:
            continue
        d = snapdata[snap]
        w = np.where(
            (d['Mvir'] > 1e10) &
            (d['tdeplete'] > 0) & np.isfinite(d['tdeplete']) &
            (d['Type'] == 0)
        )[0]
        if len(w) > 0:
            log_mv = np.log10(d['Mvir'][w])
            td = d['tdeplete'][w] * (977.8 / HUBBLE_H)  # code units -> Gyr

            bc, med, _, _ = binned_median(log_mv, td, mass_bins)

            # Bootstrap confidence intervals on the median
            n_bins = len(bc)
            boot_lo = np.full(n_bins, np.nan)
            boot_hi = np.full(n_bins, np.nan)
            for i in range(n_bins):
                mask = (log_mv >= mass_bins[i]) & (log_mv < mass_bins[i + 1])
                vals = td[mask]
                if len(vals) >= 5:
                    boot_meds = np.array([
                        np.median(vals[np.random.randint(0, len(vals), len(vals))])
                        for _ in range(N_BOOT)
                    ])
                    boot_lo[i] = np.percentile(boot_meds, 16)
                    boot_hi[i] = np.percentile(boot_meds, 84)

            valid = ~np.isnan(med)
            ax2.plot(bc[valid], med[valid], '-o', color=color, lw=2,
                     markersize=5, label=label)
            valid_boot = valid & ~np.isnan(boot_lo)
            if np.any(valid_boot):
                ax2.fill_between(bc[valid_boot], boot_lo[valid_boot],
                                 boot_hi[valid_boot],
                                 color=color, alpha=0.15)

    ax2.set_yscale('log')
    ax2.set_xlabel(r'$\log_{10}(M_{\rm vir}/M_{\odot})$')
    ax2.set_ylabel(r'$t_{\rm deplete}$ [Gyr]')
    ax2.set_xlim(10.2, 14.5)
    ax2.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    # ax2.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    _standard_legend(ax2, loc='upper right')

    fig.tight_layout()

    save_figure(fig, os.path.join(OUTPUT_DIR,
                'CGMFractionsDepletion' + OUTPUT_FORMAT))


# ========================== PLOT 9b: CGM FRACTIONS REDSHIFT GRID ==========================

def plot_9b_cgm_fractions_grid(snapdata):
    """
    1x3 redshift grid: CGM/hot gas fraction vs halo mass.
    Each panel shows one redshift with solid line for CGM and dashed for HotGas.
    """
    print('Plot 9b: CGM fractions redshift grid')

    snap_list = [
        (SNAP_Z0, f'z={REDSHIFTS[SNAP_Z0]:.0f}'),
        (SNAP_Z1, f'z={REDSHIFTS[SNAP_Z1]:.1f}'),
        (SNAP_Z2, f'z={REDSHIFTS[SNAP_Z2]:.1f}'),
        # (SNAP_Z3, f'z={REDSHIFTS[SNAP_Z3]:.1f}'),
        # (SNAP_Z4, f'z={REDSHIFTS[SNAP_Z4]:.1f}'),
        # (SNAP_Z5, f'z={REDSHIFTS[SNAP_Z5]:.1f}'),
    ]

    mass_bins = np.arange(10.0, 15.0, 0.3)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    # Detect CGM recipe
    cgm_active = (SNAP_Z0 in snapdata
                  and np.any(snapdata[SNAP_Z0]['tcool_over_tff'] > 0))

    for idx, (snap, label) in enumerate(snap_list):
        ax = axes_flat[idx]

        if snap not in snapdata:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center')
            ax.text(0.95, 0.95, label, transform=ax.transAxes,
                    ha='right', va='top')
            continue

        d = snapdata[snap]

        if cgm_active:
            # CGM regime galaxies - solid line
            w_cgm = np.where(
                (d['Regime'] == 0) & (d['Mvir'] > 1e10) &
                (d['CGMgas'] > 0) & (d['Type'] == 0)
            )[0]
            # Hot regime galaxies - dashed line
            w_hot = np.where(
                (d['Regime'] == 1) & (d['Mvir'] > 1e10) &
                (d['HotGas'] > 0) & (d['Type'] == 0)
            )[0]

            if len(w_cgm) > 0:
                log_mv = np.log10(d['Mvir'][w_cgm])
                frac = d['CGMgas'][w_cgm] / (BARYON_FRAC * d['Mvir'][w_cgm])
                bc, med, _, _ = binned_median(log_mv, frac, mass_bins)
                valid = ~np.isnan(med)
                ax.plot(bc[valid], med[valid], '-', color='black', lw=2,
                        label='CGM' if idx == 0 else None)

            if len(w_hot) > 0:
                log_mv = np.log10(d['Mvir'][w_hot])
                frac = d['HotGas'][w_hot] / (BARYON_FRAC * d['Mvir'][w_hot])
                bc, med, _, _ = binned_median(log_mv, frac, mass_bins)
                valid = ~np.isnan(med)
                ax.plot(bc[valid], med[valid], '--', color='black', lw=2,
                        label='HotGas' if idx == 0 else None)
        else:
            # No CGM recipe - just plot HotGas
            w = np.where(
                (d['Mvir'] > 1e10) & (d['HotGas'] > 0) & (d['Type'] == 0)
            )[0]
            if len(w) > 0:
                log_mv = np.log10(d['Mvir'][w])
                frac = d['HotGas'][w] / (BARYON_FRAC * d['Mvir'][w])
                bc, med, _, _ = binned_median(log_mv, frac, mass_bins)
                valid = ~np.isnan(med)
                ax.plot(bc[valid], med[valid], '-', color='black', lw=2,
                        label='HotGas' if idx == 0 else None)

        # Reference line at unity (full baryon retention)
        ax.axhline(y=1.0, color='gray', ls=':', lw=1.5, alpha=1.0)

        # Redshift label
        ax.text(0.95, 0.95, label, transform=ax.transAxes,
                ha='right', va='top')

    # Common formatting
    for ax in axes_flat:
        ax.set_yscale('log')
        ax.set_xlim(10.2, 14.5)
        ax.set_ylim(1e-3, 3.0)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.tick_params(axis='both', which='both', direction='in',
                       top=True, bottom=True, left=True, right=True)

    # Axis labels
    for ax in axes_flat:
        ax.set_xlabel(r'$\log_{10}(M_{\rm vir}/M_{\odot})$')
    axes[0].set_ylabel(r'$m_{\rm CGM,\ Hot}/(f_b\ M_{\rm vir})$')

    # Legend in first panel only
    _standard_legend(axes_flat[0], loc='lower right')

    fig.tight_layout()

    save_figure(fig, os.path.join(OUTPUT_DIR,
                'CGMFractionsGrid' + OUTPUT_FORMAT))


# ========================== PLOT 9c: DEPLETION TIMESCALE REDSHIFT GRID ==========================

def plot_9c_depletion_grid(snapdata):
    """
    1x3 redshift grid: depletion timescale vs halo mass.
    Each panel shows one redshift with solid line for CGM and dashed for HotGas.
    """
    print('Plot 9c: Depletion timescale redshift grid')

    snap_list = [
        (SNAP_Z0, f'z={REDSHIFTS[SNAP_Z0]:.0f}'),
        (SNAP_Z1, f'z={REDSHIFTS[SNAP_Z1]:.1f}'),
        (SNAP_Z2, f'z={REDSHIFTS[SNAP_Z2]:.1f}'),
        # (SNAP_Z3, f'z={REDSHIFTS[SNAP_Z3]:.1f}'),
        # (SNAP_Z4, f'z={REDSHIFTS[SNAP_Z4]:.1f}'),
        # (SNAP_Z5, f'z={REDSHIFTS[SNAP_Z5]:.1f}'),
    ]

    mass_bins = np.arange(10.0, 15.0, 0.3)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    # Detect CGM recipe
    cgm_active = (SNAP_Z0 in snapdata
                  and np.any(snapdata[SNAP_Z0]['tcool_over_tff'] > 0))

    for idx, (snap, label) in enumerate(snap_list):
        ax = axes_flat[idx]

        if snap not in snapdata:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center')
            ax.text(0.95, 0.95, label, transform=ax.transAxes,
                    ha='right', va='top')
            continue

        d = snapdata[snap]

        if cgm_active:
            # CGM regime galaxies - solid line
            w_cgm = np.where(
                (d['Regime'] == 0) & (d['Mvir'] > 1e10) &
                (d['tdeplete'] > 0) & np.isfinite(d['tdeplete']) &
                (d['Type'] == 0)
            )[0]
            # Hot regime galaxies - dashed line
            w_hot = np.where(
                (d['Regime'] == 1) & (d['Mvir'] > 1e10) &
                (d['tdeplete'] > 0) & np.isfinite(d['tdeplete']) &
                (d['Type'] == 0)
            )[0]

            if len(w_cgm) > 0:
                log_mv = np.log10(d['Mvir'][w_cgm])
                td = d['tdeplete'][w_cgm] * (977.8 / HUBBLE_H)  # code units -> Gyr
                bc, med, _, _ = binned_median(log_mv, td, mass_bins)
                valid = ~np.isnan(med)
                ax.plot(bc[valid], med[valid], '-', color='black', lw=2,
                        label='CGM' if idx == 0 else None)

            if len(w_hot) > 0:
                log_mv = np.log10(d['Mvir'][w_hot])
                td = d['tdeplete'][w_hot] * (977.8 / HUBBLE_H)  # code units -> Gyr
                bc, med, _, _ = binned_median(log_mv, td, mass_bins)
                valid = ~np.isnan(med)
                ax.plot(bc[valid], med[valid], '--', color='black', lw=2,
                        label='HotGas' if idx == 0 else None)
        else:
            # No CGM recipe - just plot all galaxies
            w = np.where(
                (d['Mvir'] > 1e10) &
                (d['tdeplete'] > 0) & np.isfinite(d['tdeplete']) &
                (d['Type'] == 0)
            )[0]
            if len(w) > 0:
                log_mv = np.log10(d['Mvir'][w])
                td = d['tdeplete'][w] * (977.8 / HUBBLE_H)  # code units -> Gyr
                bc, med, _, _ = binned_median(log_mv, td, mass_bins)
                valid = ~np.isnan(med)
                ax.plot(bc[valid], med[valid], '-', color='black', lw=2,
                        label='All' if idx == 0 else None)

        # Redshift label
        ax.text(0.95, 0.95, label, transform=ax.transAxes,
                ha='right', va='top')

    # Common formatting
    for ax in axes_flat:
        ax.set_yscale('log')
        ax.set_xlim(10.2, 14.5)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.tick_params(axis='both', which='both', direction='in',
                       top=True, bottom=True, left=True, right=True)

    # Axis labels
    for ax in axes_flat:
        ax.set_xlabel(r'$\log_{10}(M_{\rm vir}/M_{\odot})$')
    axes[0].set_ylabel(r'$t_{\rm deplete}$ [Gyr]')

    # Legend in first panel only
    _standard_legend(axes_flat[0], loc='upper right')

    fig.tight_layout()

    save_figure(fig, os.path.join(OUTPUT_DIR,
                'DepletionGrid' + OUTPUT_FORMAT))


# ========================== PLOT 10: STAR FORMATION EFFICIENCY ==========================

def plot_10_sfe_ffb(snapdata):
    """
    Star formation efficiency (epsilon = M_* / f_b M_vir) at z~10:
    FFB vs normal galaxies.
    """
    print('Plot 10: Star formation efficiency at z~10')

    snap = SNAP_Z10
    if snap not in snapdata:
        print('  Snapshot not available. Skipping.')
        return

    d = snapdata[snap]
    z_snap = REDSHIFTS[snap]

    fig, ax = plt.subplots()

    w_ffb = np.where(
        (d['StellarMass'] > 0) & (d['Mvir'] > 0) &
        (d['FFBRegime'] == 1) & (d['Type'] == 0)
    )[0]
    w_normal = np.where(
        (d['StellarMass'] > 0) & (d['Mvir'] > 0) &
        (d['FFBRegime'] == 0) & (d['Type'] == 0)
    )[0]

    def _median_percentile(log_mvir, eps, nbins=20):
        """Return bin centres, median, 16th and 84th percentiles."""
        bins = np.linspace(log_mvir.min(), log_mvir.max(), nbins + 1)
        centres = 0.5 * (bins[:-1] + bins[1:])
        median = np.full(nbins, np.nan)
        lo = np.full(nbins, np.nan)
        hi = np.full(nbins, np.nan)
        for i in range(nbins):
            mask = (log_mvir >= bins[i]) & (log_mvir < bins[i + 1])
            if np.sum(mask) >= 10:
                median[i] = np.median(eps[mask])
                lo[i] = np.percentile(eps[mask], 16)
                hi[i] = np.percentile(eps[mask], 84)
        good = ~np.isnan(median)
        return centres[good], median[good], lo[good], hi[good]

    # Compute epsilon for both populations
    eps_normal, log_mvir_normal = None, None
    eps_ffb, log_mvir_ffb = None, None

    if len(w_normal) > 0:
        eps_normal = d['StellarMass'][w_normal] / (BARYON_FRAC * d['Mvir'][w_normal])
        log_mvir_normal = np.log10(d['Mvir'][w_normal])

    if len(w_ffb) > 0:
        eps_ffb = d['StellarMass'][w_ffb] / (BARYON_FRAC * d['Mvir'][w_ffb])
        log_mvir_ffb = np.log10(d['Mvir'][w_ffb])

    # Median lines with percentile bands
    if log_mvir_normal is not None:
        x, med, lo, hi = _median_percentile(log_mvir_normal, eps_normal)
        ax.plot(x, med, color='firebrick', lw=2, label='No FFB/MBK25 model', zorder=3)
        ax.fill_between(x, lo, hi, color='darkred', alpha=0.2, zorder=2)

    if log_mvir_ffb is not None:
        x, med, lo, hi = _median_percentile(log_mvir_ffb, eps_ffb)
        ax.plot(x, med, color='black', lw=2, label='Li+24 FFB galaxies', zorder=5)
        ax.fill_between(x, lo, hi, color='grey', alpha=0.2, zorder=4)

    bk25_snap = load_snapshots(FFB_BK25_SMOOTH_DIR, [snap])
    if snap in bk25_snap:
        d_bk25 = bk25_snap[snap]
        w_bk25 = np.where(
            (d_bk25['StellarMass'] > 0) & (d_bk25['Mvir'] > 0) &
            (d_bk25['FFBRegime'] == 1) & (d_bk25['Type'] == 0)
        )[0]
        if len(w_bk25) > 0:
            eps_bk25 = d_bk25['StellarMass'][w_bk25] / (BARYON_FRAC * d_bk25['Mvir'][w_bk25])
            log_mvir_bk25 = np.log10(d_bk25['Mvir'][w_bk25])
            x, med, lo, hi = _median_percentile(log_mvir_bk25, eps_bk25, nbins=6)
            ax.plot(x, med, color='mediumpurple', lw=2, label='MBK25 galaxies', zorder=6)
            ax.fill_between(x, lo, hi, color='mediumpurple', alpha=0.2, zorder=5)

    M_ffb = ffb_threshold_mass_msun(z_snap)
    ax.axvline(x=np.log10(M_ffb), color='goldenrod', ls=':', lw=2,
               alpha=0.6,
               label=fr'Li+24: $M_{{\rm vir, FFB}}$ = {M_ffb:.1e} $M_\odot$')

    # MBK25 threshold: mass where ffb_fraction_mbk25 = 0.5
    try:
        from scipy.optimize import brentq as _brentq
        _f = lambda m: ffb_fraction_mbk25(np.array([m]), z_snap)[0] - 0.5
        if _f(1e8) * _f(1e13) < 0:
            M_ffb_mbk25 = _brentq(_f, 1e8, 1e13, xtol=1e6, rtol=1e-4)
            ax.axvline(x=np.log10(M_ffb_mbk25), color='goldenrod', ls='--', lw=2,
                       alpha=0.7,
                       label=fr'MBK25: $M_{{\rm vir, FFB}}$ = {M_ffb_mbk25:.1e} $M_\odot$')
    except Exception:
        pass

    ax.set_yscale('log')
    ax.set_xlabel(r'$\log_{10}(M_{\rm vir}/M_{\odot})$')
    ax.set_ylabel(r'$\varepsilon_{\mathrm{SFE}} \equiv m_*/(\,f_b \, M_{\rm vir})$')
    ax.set_ylim(1e-4, 2.0)

    # Snapshot redshift in the bottom-right corner.
    ax.text(0.98, 0.04, f'$z = {z_snap:.1f}$', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=20, zorder=10)

    _standard_legend(ax, loc='upper left')
    fig.tight_layout()

    save_figure(fig, os.path.join(OUTPUT_DIR,
                'SFE_FFB' + OUTPUT_FORMAT))


# ========================== PLOT 11: FFB GALAXY PROPERTIES ==========================

def plot_11_ffb_properties(snapdata):
    """
    Three-panel FFB galaxy properties at z~10:
    (a) size-mass, (b) mass-metallicity, (c) SFR-mass.
    """
    print('Plot 11: FFB galaxy properties at z~10')

    snap = SNAP_Z10
    if snap not in snapdata:
        print('  Snapshot not available. Skipping.')
        return

    d = snapdata[snap]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    w_ffb = np.where(
        (d['StellarMass'] > 0) & (d['FFBRegime'] == 1) & (d['Type'] == 0)
    )[0]
    if len(w_ffb) >= DILUTE:
        w_ffb = np.random.choice(w_ffb, DILUTE, replace=False)
    w_normal = np.where(
        (d['StellarMass'] > 0) & (d['FFBRegime'] == 0) & (d['Type'] == 0)
    )[0]
    if len(w_normal) >= DILUTE:
        w_normal = np.random.choice(w_normal, DILUTE, replace=False)

    log_ms_ffb = np.log10(d['StellarMass'][w_ffb]) if len(w_ffb) > 0 else np.array([])
    log_ms_norm = np.log10(d['StellarMass'][w_normal]) if len(w_normal) > 0 else np.array([])

    # ----- Panel (a): Size-mass relation -----
    if len(w_ffb) > 0:
        Re_ffb = 1.678 * (d['DiskRadius'][w_ffb] / HUBBLE_H) * 1e3  # kpc
        ok = Re_ffb > 0
        if np.sum(ok) > 0:
            ax1.scatter(log_ms_ffb[ok], Re_ffb[ok], s=50, c='firebrick',
                        alpha=0.8, edgecolors='darkred', linewidths=0.8,
                        label='FFB galaxies', zorder=2, rasterized=True)

    if len(w_normal) > 0:
        Re_norm = 1.678 * (d['DiskRadius'][w_normal] / HUBBLE_H) * 1e3
        ok = Re_norm > 0
        if np.sum(ok) > 0:
            ax1.scatter(log_ms_norm[ok], Re_norm[ok], s=50, c='steelblue',
                        alpha=0.1, edgecolors='navy', linewidths=0.8,
                        label='non FFB galaxies', zorder=3, rasterized=True)

    ax1.axhline(y=0.3, color='goldenrod', ls='--', lw=1.5, alpha=1.0,
                label='0.3 kpc (compact)')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$\log_{10}(m_*/M_{\odot})$')
    ax1.set_ylabel(r'$R_e$ [kpc]')

    # Add redshift text in upper right corner
    ax1.text(0.05, 0.95, f'z={REDSHIFTS[snap]:.1f}', transform=ax1.transAxes,
             ha='left', va='top')
    _standard_legend(ax1, loc='lower left')

    # ----- Panel (b): Mass-metallicity relation -----
    if len(w_ffb) > 0:
        ms = d['StellarMass'][w_ffb]
        mz = d['MetalsStellarMass'][w_ffb]
        Z_ratio = (mz / ms) / Z_SUN
        ok = Z_ratio > 0
        if np.sum(ok) > 0:
            ax2.scatter(log_ms_ffb[ok], np.log10(Z_ratio[ok]), s=50,
                        c='firebrick', alpha=0.8, edgecolors='darkred',
                        linewidths=0.8, label='FFB galaxies', zorder=2, rasterized=True)

    if len(w_normal) > 0:
        ms = d['StellarMass'][w_normal]
        mz = d['MetalsStellarMass'][w_normal]
        Z_ratio = (mz / ms) / Z_SUN
        ok = Z_ratio > 0
        if np.sum(ok) > 0:
            ax2.scatter(log_ms_norm[ok], np.log10(Z_ratio[ok]), s=50,
                        c='steelblue', alpha=0.1, edgecolors='navy',
                        linewidths=0.8, label='non FFB galaxies', zorder=3, rasterized=True)

    ax2.axhline(y=np.log10(0.1), color='goldenrod', ls='--', lw=1.5,
                alpha=1.0, label=r'$0.1\,Z_{\odot}$')
    ax2.set_xlabel(r'$\log_{10}(m_*/M_{\odot})$')
    ax2.set_ylabel(r'$\log_{10}(Z_*/Z_{\odot})$')
    # Add redshift text in upper right corner
    ax2.text(0.05, 0.95, f'z={REDSHIFTS[snap]:.1f}', transform=ax2.transAxes,
             ha='left', va='top')
    _standard_legend(ax2, loc='lower right')

    # ----- Panel (c): SFR vs M_* -----
    if len(w_ffb) > 0:
        sfr = d['SfrDisk'][w_ffb] + d['SfrBulge'][w_ffb]
        ok = sfr > 0
        if np.sum(ok) > 0:
            ax3.scatter(log_ms_ffb[ok], np.log10(sfr[ok]), s=50,
                        c='firebrick', alpha=0.8, edgecolors='darkred',
                        linewidths=0.8, label='FFB galaxies', zorder=2, rasterized=True)

    if len(w_normal) > 0:
        sfr = d['SfrDisk'][w_normal] + d['SfrBulge'][w_normal]
        ok = sfr > 0
        if np.sum(ok) > 0:
            ax3.scatter(log_ms_norm[ok], np.log10(sfr[ok]), s=50,
                        c='steelblue', alpha=0.1, edgecolors='navy',
                        linewidths=0.8, label='non FFB galaxies', zorder=3, rasterized=True)
    ax3.set_xlabel(r'$\log_{10}(m_*/M_{\odot})$')
    ax3.set_ylabel(r'$\log_{10}(\mathrm{SFR}\ [M_{\odot}\,\mathrm{yr}^{-1}])$')
    # Add redshift text in upper right corner
    ax3.text(0.05, 0.95, f'z={REDSHIFTS[snap]:.1f}', transform=ax3.transAxes,
             ha='left', va='top')
    _standard_legend(ax3, loc='lower left')

    fig.tight_layout()

    save_figure(fig, os.path.join(OUTPUT_DIR,
                'FFBProperties' + OUTPUT_FORMAT))


# ========================== PLOT 11b: FFB PROPERTY HISTOGRAMS ==========================

def plot_11b_ffb_histograms(snapdata):
    """
    Histogram comparison of galaxy properties for FFB vs non-FFB galaxies at z~7:
    (a) Effective Radius, (b) Metallicity, (c) SFR.
    """
    print('Plot 11b: FFB property histograms at z~10')

    snap = SNAP_Z10
    if snap not in snapdata:
        print('  Snapshot not available. Skipping.')
        return

    d = snapdata[snap]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    w_ffb = np.where((d['FFBRegime'] == 1) & (d['Type'] == 0))[0]
    w_normal = np.where((d['FFBRegime'] == 0) & (d['Type'] == 0))[0]
    print(f'  FFB galaxies (resolved): {len(w_ffb)}')
    print(f'  Non-FFB galaxies (resolved): {len(w_normal)}')

    hist_kwargs_ffb = dict(bins=30, alpha=0.7, color='firebrick',
                           edgecolor='darkred', linewidth=1.2,
                           label='Li+24 FFB galaxies', density=True)
    hist_kwargs_norm = dict(bins=30, alpha=0.5, color='steelblue',
                            edgecolor='navy', linewidth=1.2,
                            label='Li+24 non-FFB galaxies', density=True)

    # ----- Panel (a): Effective Radius -----
    if len(w_ffb) > 0:
        Re_ffb = 1.678 * (d['DiskRadius'][w_ffb] / HUBBLE_H) * 1e3  # kpc
        ok = Re_ffb > 0
        if np.sum(ok) > 0:
            ax1.hist(np.log10(Re_ffb[ok]), **hist_kwargs_ffb)
    if len(w_normal) > 0:
        Re_norm = 1.678 * (d['DiskRadius'][w_normal] / HUBBLE_H) * 1e3
        ok = Re_norm > 0
        if np.sum(ok) > 0:
            ax1.hist(np.log10(Re_norm[ok]), **hist_kwargs_norm)
    ax1.set_xlabel(r'$\log_{10}(R_e\ [\mathrm{kpc}])$')
    ax1.set_ylabel('Normalized Count')
    ax1.text(0.95, 0.95, f'z={REDSHIFTS[snap]:.1f}', transform=ax1.transAxes,
             ha='right', va='top')
    _standard_legend(ax1, loc='upper left')
    ax1.set_ylim(0, ax1.get_ylim()[1] * 1.35)

    # ----- Panel (b): Metallicity -----
    if len(w_ffb) > 0:
        ms = d['StellarMass'][w_ffb]
        mz = d['MetalsStellarMass'][w_ffb]
        Z_ratio = (mz / ms) / Z_SUN
        ok = Z_ratio > 0
        if np.sum(ok) > 0:
            ax2.hist(np.log10(Z_ratio[ok]), **hist_kwargs_ffb)
    if len(w_normal) > 0:
        ms = d['StellarMass'][w_normal]
        mz = d['MetalsStellarMass'][w_normal]
        Z_ratio = (mz / ms) / Z_SUN
        ok = Z_ratio > 0
        if np.sum(ok) > 0:
            ax2.hist(np.log10(Z_ratio[ok]), **hist_kwargs_norm)
    ax2.set_xlabel(r'$\log_{10}(Z_*/Z_{\odot})$')
    ax2.set_ylabel('Normalized Count')
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.35)

    # ----- Panel (c): Star Formation Rate -----
    if len(w_ffb) > 0:
        sfr_ffb = d['SfrDisk'][w_ffb] + d['SfrBulge'][w_ffb]
        ok = sfr_ffb > 0
        if np.sum(ok) > 0:
            ax3.hist(np.log10(sfr_ffb[ok]), **hist_kwargs_ffb)
    if len(w_normal) > 0:
        sfr_norm = d['SfrDisk'][w_normal] + d['SfrBulge'][w_normal]
        ok = sfr_norm > 0
        if np.sum(ok) > 0:
            ax3.hist(np.log10(sfr_norm[ok]), **hist_kwargs_norm)
    ax3.set_xlabel(r'$\log_{10}(\mathrm{SFR}\ [M_{\odot}\,\mathrm{yr}^{-1}])$')
    ax3.set_ylabel('Normalized Count')
    ax3.set_ylim(0, ax3.get_ylim()[1] * 1.35)

    fig.tight_layout()

    save_figure(fig, os.path.join(OUTPUT_DIR,
                'FFBPropertiesHistograms' + OUTPUT_FORMAT))


# ========================== PLOT 11c: FFB PROPERTY HISTOGRAMS (MBK25) ==========================

def plot_11c_ffb_histograms_mbk25(snapdata):
    """
    Histogram comparison of galaxy properties for FFB vs non-FFB galaxies at z~10,
    using the MBK25 smooth model output.
    (a) Effective Radius, (b) Metallicity, (c) SFR.
    """
    print('Plot 11c: MBK25 FFB property histograms at z~10')

    snap = SNAP_Z10

    bk25_snap = load_snapshots(FFB_BK25_SMOOTH_DIR, [snap])
    if snap not in bk25_snap:
        print('  MBK25 snapshot not available. Skipping.')
        return

    d = bk25_snap[snap]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    w_ffb = np.where((d['FFBRegime'] == 1) & (d['Type'] == 0))[0]
    w_normal = np.where((d['FFBRegime'] == 0) & (d['Type'] == 0))[0]
    print(f'  MBK25  galaxies (resolved): {len(w_ffb)}')
    print(f'  non-MBK25 galaxies (resolved): {len(w_normal)}')

    hist_kwargs_ffb = dict(bins=30, alpha=0.7, color='firebrick',
                           edgecolor='darkred', linewidth=1.2,
                           label='MBK25 galaxies', density=True)
    hist_kwargs_norm = dict(bins=30, alpha=0.5, color='steelblue',
                            edgecolor='navy', linewidth=1.2,
                            label='non-MBK25 galaxies', density=True)

    # ----- Panel (a): Effective Radius -----
    if len(w_ffb) > 0:
        Re_ffb = 1.678 * (d['DiskRadius'][w_ffb] / HUBBLE_H) * 1e3
        ok = Re_ffb > 0
        if np.sum(ok) > 0:
            ax1.hist(np.log10(Re_ffb[ok]), **hist_kwargs_ffb)
    if len(w_normal) > 0:
        Re_norm = 1.678 * (d['DiskRadius'][w_normal] / HUBBLE_H) * 1e3
        ok = Re_norm > 0
        if np.sum(ok) > 0:
            ax1.hist(np.log10(Re_norm[ok]), **hist_kwargs_norm)
    ax1.set_xlabel(r'$\log_{10}(R_e\ [\mathrm{kpc}])$')
    ax1.set_ylabel('Normalized Count')
    ax1.text(0.95, 0.95, f'z={REDSHIFTS[snap]:.1f}', transform=ax1.transAxes,
             ha='right', va='top')
    _standard_legend(ax1, loc='upper left')
    ax1.set_ylim(0, ax1.get_ylim()[1] * 1.35)

    # ----- Panel (b): Metallicity -----
    if len(w_ffb) > 0:
        ms = d['StellarMass'][w_ffb]
        mz = d['MetalsStellarMass'][w_ffb]
        Z_ratio = (mz / ms) / Z_SUN
        ok = Z_ratio > 0
        if np.sum(ok) > 0:
            ax2.hist(np.log10(Z_ratio[ok]), **hist_kwargs_ffb)
    if len(w_normal) > 0:
        ms = d['StellarMass'][w_normal]
        mz = d['MetalsStellarMass'][w_normal]
        Z_ratio = (mz / ms) / Z_SUN
        ok = Z_ratio > 0
        if np.sum(ok) > 0:
            ax2.hist(np.log10(Z_ratio[ok]), **hist_kwargs_norm)
    ax2.set_xlabel(r'$\log_{10}(Z_*/Z_{\odot})$')
    ax2.set_ylabel('Normalized Count')
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.35)

    # ----- Panel (c): Star Formation Rate -----
    if len(w_ffb) > 0:
        sfr_ffb = d['SfrDisk'][w_ffb] + d['SfrBulge'][w_ffb]
        ok = sfr_ffb > 0
        if np.sum(ok) > 0:
            ax3.hist(np.log10(sfr_ffb[ok]), **hist_kwargs_ffb)
    if len(w_normal) > 0:
        sfr_norm = d['SfrDisk'][w_normal] + d['SfrBulge'][w_normal]
        ok = sfr_norm > 0
        if np.sum(ok) > 0:
            ax3.hist(np.log10(sfr_norm[ok]), **hist_kwargs_norm)
    ax3.set_xlabel(r'$\log_{10}(\mathrm{SFR}\ [M_{\odot}\,\mathrm{yr}^{-1}])$')
    ax3.set_ylabel('Normalized Count')
    ax3.set_ylim(0, ax3.get_ylim()[1] * 1.35)

    fig.tight_layout()

    save_figure(fig, os.path.join(OUTPUT_DIR,
                'FFBPropertiesHistograms_MBK25' + OUTPUT_FORMAT))


# ========================== PLOT 11d: FFB PROPERTY HISTOGRAMS (COMBINED) ==========================

def _print_ffb_hist_stats(title, panels):
    """
    Print per-panel, per-model summary statistics to the terminal.

    Parameters
    ----------
    title : str
        Heading printed above the table.
    panels : list of (str, dict)
        Each entry is (quantity_label, {model_label: values_array}).
        Arrays are already the plotted (finite, log10) values.
    """
    print(f'\n=== Stats: {title} ===')
    header = f"{'quantity':<20} {'model':<12} {'N':>7} {'median':>9} " \
             f"{'mean':>9} {'std':>9} {'min':>9} {'max':>9}"
    print(header)
    print('-' * len(header))
    for label, models in panels:
        for model_label, vals in models.items():
            v = np.asarray(vals, dtype=float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                print(f"{label:<20} {model_label:<12} {0:>7} "
                      f"{'--':>9} {'--':>9} {'--':>9} {'--':>9} {'--':>9}")
                continue
            print(f"{label:<20} {model_label:<12} {v.size:>7} "
                  f"{np.median(v):>9.3f} {np.mean(v):>9.3f} {np.std(v):>9.3f} "
                  f"{np.min(v):>9.3f} {np.max(v):>9.3f}")
        print('-' * len(header))


def plot_11d_ffb_histograms_combined(snapdata):
    """
    Combined histogram comparison at z~10:
      - Li+24 non-FFB galaxies (firebrick) — baseline population
      - Li+24 FFB galaxies (black)
      - MBK25 FFB galaxies (mediumpurple)
    (a) SFR, (b) Metallicity, (c) Disk Radius.
    """
    print('Plot 11d: Combined FFB property histograms at z~10')

    snap = SNAP_Z10

    # --- Li+24 FFB data ---
    if snap not in snapdata:
        print('  Li+24 snapshot not available. Skipping.')
        return
    d_li = snapdata[snap]

    w_ffb_li = np.where((d_li['FFBRegime'] == 1) & (d_li['Type'] == 0))[0]

    # --- No-FFB model: full central population ---
    noffb_snap = load_snapshots(NOFFB_DIR, [snap])
    d_noffb = noffb_snap.get(snap, None)
    if d_noffb is not None:
        w_normal = np.where(d_noffb['Type'] == 0)[0]
    else:
        w_normal = np.array([], dtype=int)
        d_noffb = {}

    # --- MBK25 data ---
    bk25_snap = load_snapshots(FFB_BK25_SMOOTH_DIR, [snap])
    d_bk = bk25_snap.get(snap, None)
    if d_bk is not None:
        w_ffb_bk = np.where((d_bk['FFBRegime'] == 1) & (d_bk['Type'] == 0))[0]
    else:
        w_ffb_bk = np.array([], dtype=int)

    hist_kwargs_norm = dict(alpha=0.5, color='firebrick',
                            edgecolor='darkred', linewidth=1.2,
                            label='No FFB/MBK25 model', density=True)
    hist_kwargs_li = dict(alpha=0.7, color='black',
                          edgecolor='black', linewidth=1.2,
                          label='Li+24 FFB galaxies', density=True)
    hist_kwargs_bk = dict(alpha=0.7, color='mediumpurple',
                          edgecolor='indigo', linewidth=1.2,
                          label='MBK25 galaxies', density=True)

    def common_edges(arrays, nbins=30):
        """Shared bin edges spanning every model's data in a panel."""
        finite = [a[np.isfinite(a)] for a in arrays if len(a) > 0]
        finite = [a for a in finite if len(a) > 0]
        if not finite:
            return nbins
        allvals = np.concatenate(finite)
        lo, hi = allvals.min(), allvals.max()
        if lo == hi:
            return nbins
        return np.linspace(lo, hi, nbins + 1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # ----- Panel (a): Star Formation Rate -----
    def _logSFR(d, w):
        if len(w) == 0:
            return np.array([])
        sfr = d['SfrDisk'][w] + d['SfrBulge'][w]
        return np.log10(sfr[sfr > 0])
    sfr_norm = _logSFR(d_noffb, w_normal)
    sfr_li = _logSFR(d_li, w_ffb_li)
    sfr_bk = _logSFR(d_bk, w_ffb_bk)
    edges_a = common_edges([sfr_norm, sfr_li, sfr_bk])
    if len(sfr_norm) > 0:
        ax1.hist(sfr_norm, bins=edges_a, **hist_kwargs_norm)
    if len(sfr_li) > 0:
        ax1.hist(sfr_li, bins=edges_a, **hist_kwargs_li)
    if len(sfr_bk) > 0:
        ax1.hist(sfr_bk, bins=edges_a, **hist_kwargs_bk)
    ax1.set_xlabel(r'$\log_{10}(\mathrm{SFR}\ [M_{\odot}\,\mathrm{yr}^{-1}])$')
    ax1.set_ylabel('Normalized Count')
    ax1.text(0.95, 0.95, f'z={REDSHIFTS[snap]:.1f}', transform=ax1.transAxes,
             ha='right', va='top')
    _standard_legend(ax1, loc='upper left')
    ax1.set_ylim(0, ax1.get_ylim()[1] * 1.35)
    ax1.set_xlim(-2, 2.7)

    # ----- Panel (b): Metallicity -----
    def _logZ(d, w):
        if len(w) == 0:
            return np.array([])
        Z_ratio = (d['MetalsStellarMass'][w] / d['StellarMass'][w]) / Z_SUN
        return np.log10(Z_ratio[Z_ratio > 0])
    Z_norm = _logZ(d_noffb, w_normal)
    Z_li = _logZ(d_li, w_ffb_li)
    Z_bk = _logZ(d_bk, w_ffb_bk)
    edges_b = common_edges([Z_norm, Z_li, Z_bk])
    if len(Z_norm) > 0:
        ax2.hist(Z_norm, bins=edges_b, **hist_kwargs_norm)
    if len(Z_li) > 0:
        ax2.hist(Z_li, bins=edges_b, **hist_kwargs_li)
    if len(Z_bk) > 0:
        ax2.hist(Z_bk, bins=edges_b, **hist_kwargs_bk)
    ax2.set_xlabel(r'$\log_{10}(Z_*/Z_{\odot})$')
    ax2.set_ylabel('Normalized Count')
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.35)
    ax2.set_xlim(-2, -0.5)

    # ----- Panel (c): Disk Radius (raw DiskRadius in kpc) -----
    def _logRdisk(d, w):
        if len(w) == 0:
            return np.array([])
        rd = d['DiskRadius'][w]
        rd = rd[rd > 0]
        return np.log10((rd / HUBBLE_H) * 1e3)  # kpc
    Rd_norm = _logRdisk(d_noffb, w_normal)
    Rd_li = _logRdisk(d_li, w_ffb_li)
    Rd_bk = _logRdisk(d_bk, w_ffb_bk)
    edges_c = common_edges([Rd_norm, Rd_li, Rd_bk])
    if len(Rd_norm) > 0:
        ax3.hist(Rd_norm, bins=edges_c, **hist_kwargs_norm)
    if len(Rd_li) > 0:
        ax3.hist(Rd_li, bins=edges_c, **hist_kwargs_li)
    if len(Rd_bk) > 0:
        ax3.hist(Rd_bk, bins=edges_c, **hist_kwargs_bk)
    ax3.set_xlabel(r'$\log_{10}(R_{\mathrm{disk}}\ [\mathrm{kpc}])$')
    ax3.set_ylabel('Normalized Count')
    ax3.set_ylim(0, ax3.get_ylim()[1] * 1.35)

    # ----- Terminal stats for all panels and models -----
    _print_ffb_hist_stats(
        f'FFBPropertiesHistograms_Combined (z={REDSHIFTS[snap]:.1f})',
        panels=[
            (r'log10 SFR [Msun/yr]',   {'No FFB': sfr_norm, 'Li+24 FFB': sfr_li, 'MBK25 galaxies': sfr_bk}),
            (r'log10 Z*/Zsun',          {'No FFB': Z_norm,   'Li+24 FFB': Z_li,   'MBK25 galaxies': Z_bk}),
            (r'log10 Rdisk [kpc]',      {'No FFB': Rd_norm,  'Li+24 FFB': Rd_li,  'MBK25 galaxies': Rd_bk}),
        ])

    fig.tight_layout()

    save_figure(fig, os.path.join(OUTPUT_DIR,
                'FFBPropertiesHistograms_Combined' + OUTPUT_FORMAT))


# ========================== PLOT 11e: FFB PROPERTY HISTOGRAMS (COMBINED + BULGE RADIUS) ==========================

def plot_11e_ffb_histograms_combined_bulge(snapdata):
    """
    Same as plot_11d but with an added fourth panel for bulge radius:
      - Li+24 non-FFB galaxies (firebrick) — baseline population
      - Li+24 FFB galaxies (black)
      - MBK25 FFB galaxies (mediumpurple)
    (a) SFR, (b) Metallicity, (c) Effective Radius, (d) Bulge Radius.
    """
    print('Plot 11e: Combined FFB property histograms (with bulge radius) at z~10')

    snap = SNAP_Z10

    # --- Li+24 FFB data ---
    if snap not in snapdata:
        print('  Li+24 snapshot not available. Skipping.')
        return
    d_li = snapdata[snap]

    w_ffb_li = np.where((d_li['FFBRegime'] == 1) & (d_li['Type'] == 0))[0]

    # --- No-FFB model: full central population ---
    noffb_snap = load_snapshots(NOFFB_DIR, [snap])
    d_noffb = noffb_snap.get(snap, None)
    if d_noffb is not None:
        w_normal = np.where(d_noffb['Type'] == 0)[0]
    else:
        w_normal = np.array([], dtype=int)
        d_noffb = {}

    # --- MBK25 data ---
    bk25_snap = load_snapshots(FFB_BK25_SMOOTH_DIR, [snap])
    d_bk = bk25_snap.get(snap, None)
    if d_bk is not None:
        w_ffb_bk = np.where((d_bk['FFBRegime'] == 1) & (d_bk['Type'] == 0))[0]
    else:
        w_ffb_bk = np.array([], dtype=int)

    hist_kwargs_norm = dict(alpha=0.5, color='firebrick',
                            edgecolor='darkred', linewidth=1.2,
                            label='No FFB/MBK25 model', density=True)
    hist_kwargs_li = dict(alpha=0.7, color='black',
                          edgecolor='black', linewidth=1.2,
                          label='Li+24 FFB galaxies', density=True)
    hist_kwargs_bk = dict(alpha=0.7, color='mediumpurple',
                          edgecolor='indigo', linewidth=1.2,
                          label='MBK25 galaxies', density=True)

    def common_edges(arrays, nbins=30):
        """Shared bin edges spanning every model's data in a panel."""
        finite = [a[np.isfinite(a)] for a in arrays if len(a) > 0]
        finite = [a for a in finite if len(a) > 0]
        if not finite:
            return nbins
        allvals = np.concatenate(finite)
        lo, hi = allvals.min(), allvals.max()
        if lo == hi:
            return nbins
        return np.linspace(lo, hi, nbins + 1)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))

    # ----- Panel (a): Star Formation Rate -----
    def _logSFR(d, w):
        if len(w) == 0:
            return np.array([])
        sfr = d['SfrDisk'][w] + d['SfrBulge'][w]
        return np.log10(sfr[sfr > 0])
    sfr_norm = _logSFR(d_noffb, w_normal)
    sfr_li = _logSFR(d_li, w_ffb_li)
    sfr_bk = _logSFR(d_bk, w_ffb_bk)
    edges_a = common_edges([sfr_norm, sfr_li, sfr_bk])
    if len(sfr_norm) > 0:
        ax1.hist(sfr_norm, bins=edges_a, **hist_kwargs_norm)
    if len(sfr_li) > 0:
        ax1.hist(sfr_li, bins=edges_a, **hist_kwargs_li)
    if len(sfr_bk) > 0:
        ax1.hist(sfr_bk, bins=edges_a, **hist_kwargs_bk)
    ax1.set_xlabel(r'$\log_{10}(\mathrm{SFR}\ [M_{\odot}\,\mathrm{yr}^{-1}])$')
    ax1.set_ylabel('Normalized Count')
    ax1.text(0.95, 0.95, f'z={REDSHIFTS[snap]:.1f}', transform=ax1.transAxes,
             ha='right', va='top')
    _standard_legend(ax1, loc='upper left')
    ax1.set_ylim(0, ax1.get_ylim()[1] * 1.35)
    ax1.set_xlim(-2, 2.7)

    # ----- Panel (b): Metallicity -----
    def _logZ(d, w):
        if len(w) == 0:
            return np.array([])
        Z_ratio = (d['MetalsStellarMass'][w] / d['StellarMass'][w]) / Z_SUN
        return np.log10(Z_ratio[Z_ratio > 0])
    Z_norm = _logZ(d_noffb, w_normal)
    Z_li = _logZ(d_li, w_ffb_li)
    Z_bk = _logZ(d_bk, w_ffb_bk)
    edges_b = common_edges([Z_norm, Z_li, Z_bk])
    if len(Z_norm) > 0:
        ax2.hist(Z_norm, bins=edges_b, **hist_kwargs_norm)
    if len(Z_li) > 0:
        ax2.hist(Z_li, bins=edges_b, **hist_kwargs_li)
    if len(Z_bk) > 0:
        ax2.hist(Z_bk, bins=edges_b, **hist_kwargs_bk)
    ax2.set_xlabel(r'$\log_{10}(Z_*/Z_{\odot})$')
    ax2.set_ylabel('Normalized Count')
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.35)
    ax2.set_xlim(-2, -0.5)

    # ----- Panel (c): Effective Radius -----
    Re_norm = (np.log10(1.678 * (d_noffb['DiskRadius'][w_normal] / HUBBLE_H) * 1e3)
               if len(w_normal) > 0 else np.array([]))
    Re_li = (np.log10(1.678 * (d_li['DiskRadius'][w_ffb_li] / HUBBLE_H) * 1e3)
             if len(w_ffb_li) > 0 else np.array([]))
    Re_bk = (np.log10(1.678 * (d_bk['DiskRadius'][w_ffb_bk] / HUBBLE_H) * 1e3)
             if len(w_ffb_bk) > 0 else np.array([]))
    edges_c = common_edges([Re_norm, Re_li, Re_bk])
    if len(Re_norm) > 0:
        ax3.hist(Re_norm, bins=edges_c, **hist_kwargs_norm)
    if len(Re_li) > 0:
        ax3.hist(Re_li, bins=edges_c, **hist_kwargs_li)
    if len(Re_bk) > 0:
        ax3.hist(Re_bk, bins=edges_c, **hist_kwargs_bk)
    ax3.set_xlabel(r'$\log_{10}(R_e\ [\mathrm{kpc}])$')
    ax3.set_ylabel('Normalized Count')
    ax3.set_ylim(0, ax3.get_ylim()[1] * 1.35)

    # ----- Panel (d): Bulge Radius -----
    def _logRbulge(d, w):
        if len(w) == 0:
            return np.array([])
        rb = d['BulgeRadius'][w]
        rb = rb[rb > 0]
        return np.log10((rb / HUBBLE_H) * 1e3)  # kpc
    Rb_norm = _logRbulge(d_noffb, w_normal)
    Rb_li = _logRbulge(d_li, w_ffb_li)
    Rb_bk = _logRbulge(d_bk, w_ffb_bk)
    edges_d = common_edges([Rb_norm, Rb_li, Rb_bk])
    if len(Rb_norm) > 0:
        ax4.hist(Rb_norm, bins=edges_d, **hist_kwargs_norm)
    if len(Rb_li) > 0:
        ax4.hist(Rb_li, bins=edges_d, **hist_kwargs_li)
    if len(Rb_bk) > 0:
        ax4.hist(Rb_bk, bins=edges_d, **hist_kwargs_bk)
    ax4.set_xlabel(r'$\log_{10}(R_{\mathrm{bulge}}\ [\mathrm{kpc}])$')
    ax4.set_ylabel('Normalized Count')
    ax4.set_ylim(0, ax4.get_ylim()[1] * 1.35)

    # ----- Terminal stats for all panels and models -----
    _print_ffb_hist_stats(
        f'FFBPropertiesHistograms_Combined_Bulge (z={REDSHIFTS[snap]:.1f})',
        panels=[
            (r'log10 SFR [Msun/yr]',   {'No FFB': sfr_norm, 'Li+24 FFB': sfr_li, 'MBK25 galaxies': sfr_bk}),
            (r'log10 Z*/Zsun',          {'No FFB': Z_norm,   'Li+24 FFB': Z_li,   'MBK25 galaxies': Z_bk}),
            (r'log10 Re [kpc]',         {'No FFB': Re_norm,  'Li+24 FFB': Re_li,  'MBK25 galaxies': Re_bk}),
            (r'log10 Rbulge [kpc]',     {'No FFB': Rb_norm,  'Li+24 FFB': Rb_li,  'MBK25 galaxies': Rb_bk}),
        ])

    fig.tight_layout()

    save_figure(fig, os.path.join(OUTPUT_DIR,
                'FFBPropertiesHistograms_Combined_Bulge' + OUTPUT_FORMAT))


# ========================== PLOT 11f: FFB PROPERTY HISTOGRAMS (COMBINED + DISK & BULGE RADIUS) ==========================

def plot_11f_ffb_histograms_combined_diskbulge(snapdata):
    """
    Same as plot_11e but the radius panel shows the raw disk scale radius
    (DiskRadius converted to kpc) rather than an effective radius:
      - Li+24 non-FFB galaxies (firebrick) — baseline population
      - Li+24 FFB galaxies (black)
      - MBK25 FFB galaxies (mediumpurple)
    (a) SFR, (b) Metallicity, (c) Disk Radius, (d) Bulge Radius.
    """
    print('Plot 11f: Combined FFB property histograms (with disk & bulge radius) at z~10')

    snap = SNAP_Z10

    # --- Li+24 FFB data ---
    if snap not in snapdata:
        print('  Li+24 snapshot not available. Skipping.')
        return
    d_li = snapdata[snap]

    w_ffb_li = np.where((d_li['FFBRegime'] == 1) & (d_li['Type'] == 0))[0]

    # --- No-FFB model: full central population ---
    noffb_snap = load_snapshots(NOFFB_DIR, [snap])
    d_noffb = noffb_snap.get(snap, None)
    if d_noffb is not None:
        w_normal = np.where(d_noffb['Type'] == 0)[0]
    else:
        w_normal = np.array([], dtype=int)
        d_noffb = {}

    # --- MBK25 data ---
    bk25_snap = load_snapshots(FFB_BK25_SMOOTH_DIR, [snap])
    d_bk = bk25_snap.get(snap, None)
    if d_bk is not None:
        w_ffb_bk = np.where((d_bk['FFBRegime'] == 1) & (d_bk['Type'] == 0))[0]
    else:
        w_ffb_bk = np.array([], dtype=int)

    hist_kwargs_norm = dict(alpha=0.5, color='firebrick',
                            edgecolor='darkred', linewidth=1.2,
                            label='No FFB/MBK25 model', density=True)
    hist_kwargs_li = dict(alpha=0.7, color='black',
                          edgecolor='black', linewidth=1.2,
                          label='Li+24 FFB galaxies', density=True)
    hist_kwargs_bk = dict(alpha=0.7, color='mediumpurple',
                          edgecolor='indigo', linewidth=1.2,
                          label='MBK25 galaxies', density=True)

    def common_edges(arrays, nbins=30):
        """Shared bin edges spanning every model's data in a panel."""
        finite = [a[np.isfinite(a)] for a in arrays if len(a) > 0]
        finite = [a for a in finite if len(a) > 0]
        if not finite:
            return nbins
        allvals = np.concatenate(finite)
        lo, hi = allvals.min(), allvals.max()
        if lo == hi:
            return nbins
        return np.linspace(lo, hi, nbins + 1)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))

    # ----- Panel (a): Star Formation Rate -----
    def _logSFR(d, w):
        if len(w) == 0:
            return np.array([])
        sfr = d['SfrDisk'][w] + d['SfrBulge'][w]
        return np.log10(sfr[sfr > 0])
    sfr_norm = _logSFR(d_noffb, w_normal)
    sfr_li = _logSFR(d_li, w_ffb_li)
    sfr_bk = _logSFR(d_bk, w_ffb_bk)
    edges_a = common_edges([sfr_norm, sfr_li, sfr_bk])
    if len(sfr_norm) > 0:
        ax1.hist(sfr_norm, bins=edges_a, **hist_kwargs_norm)
    if len(sfr_li) > 0:
        ax1.hist(sfr_li, bins=edges_a, **hist_kwargs_li)
    if len(sfr_bk) > 0:
        ax1.hist(sfr_bk, bins=edges_a, **hist_kwargs_bk)
    ax1.set_xlabel(r'$\log_{10}(\mathrm{SFR}\ [M_{\odot}\,\mathrm{yr}^{-1}])$')
    ax1.set_ylabel('Normalized Count')
    ax1.text(0.95, 0.95, f'z={REDSHIFTS[snap]:.1f}', transform=ax1.transAxes,
             ha='right', va='top')
    _standard_legend(ax1, loc='upper left')
    ax1.set_ylim(0, ax1.get_ylim()[1] * 1.35)
    ax1.set_xlim(-2, 2.7)

    # ----- Panel (b): Metallicity -----
    def _logZ(d, w):
        if len(w) == 0:
            return np.array([])
        Z_ratio = (d['MetalsStellarMass'][w] / d['StellarMass'][w]) / Z_SUN
        return np.log10(Z_ratio[Z_ratio > 0])
    Z_norm = _logZ(d_noffb, w_normal)
    Z_li = _logZ(d_li, w_ffb_li)
    Z_bk = _logZ(d_bk, w_ffb_bk)
    edges_b = common_edges([Z_norm, Z_li, Z_bk])
    if len(Z_norm) > 0:
        ax2.hist(Z_norm, bins=edges_b, **hist_kwargs_norm)
    if len(Z_li) > 0:
        ax2.hist(Z_li, bins=edges_b, **hist_kwargs_li)
    if len(Z_bk) > 0:
        ax2.hist(Z_bk, bins=edges_b, **hist_kwargs_bk)
    ax2.set_xlabel(r'$\log_{10}(Z_*/Z_{\odot})$')
    ax2.set_ylabel('Normalized Count')
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.35)
    ax2.set_xlim(-2, -0.5)

    # ----- Panel (c): Disk Radius (raw DiskRadius in kpc) -----
    def _logRdisk(d, w):
        if len(w) == 0:
            return np.array([])
        rd = d['DiskRadius'][w]
        rd = rd[rd > 0]
        return np.log10((rd / HUBBLE_H) * 1e3)  # kpc
    Rd_norm = _logRdisk(d_noffb, w_normal)
    Rd_li = _logRdisk(d_li, w_ffb_li)
    Rd_bk = _logRdisk(d_bk, w_ffb_bk)
    edges_c = common_edges([Rd_norm, Rd_li, Rd_bk])
    if len(Rd_norm) > 0:
        ax3.hist(Rd_norm, bins=edges_c, **hist_kwargs_norm)
    if len(Rd_li) > 0:
        ax3.hist(Rd_li, bins=edges_c, **hist_kwargs_li)
    if len(Rd_bk) > 0:
        ax3.hist(Rd_bk, bins=edges_c, **hist_kwargs_bk)
    ax3.set_xlabel(r'$\log_{10}(R_{\mathrm{disk}}\ [\mathrm{kpc}])$')
    ax3.set_ylabel('Normalized Count')
    ax3.set_ylim(0, ax3.get_ylim()[1] * 1.35)

    # ----- Panel (d): Bulge Radius -----
    def _logRbulge(d, w):
        if len(w) == 0:
            return np.array([])
        rb = d['BulgeRadius'][w]
        rb = rb[rb > 0]
        return np.log10((rb / HUBBLE_H) * 1e3)  # kpc
    Rb_norm = _logRbulge(d_noffb, w_normal)
    Rb_li = _logRbulge(d_li, w_ffb_li)
    Rb_bk = _logRbulge(d_bk, w_ffb_bk)
    edges_d = common_edges([Rb_norm, Rb_li, Rb_bk])
    if len(Rb_norm) > 0:
        ax4.hist(Rb_norm, bins=edges_d, **hist_kwargs_norm)
    if len(Rb_li) > 0:
        ax4.hist(Rb_li, bins=edges_d, **hist_kwargs_li)
    if len(Rb_bk) > 0:
        ax4.hist(Rb_bk, bins=edges_d, **hist_kwargs_bk)
    ax4.set_xlabel(r'$\log_{10}(R_{\mathrm{bulge}}\ [\mathrm{kpc}])$')
    ax4.set_ylabel('Normalized Count')
    ax4.set_ylim(0, ax4.get_ylim()[1] * 1.35)

    # ----- Terminal stats for all panels and models -----
    _print_ffb_hist_stats(
        f'FFBPropertiesHistograms_Combined_DiskBulge (z={REDSHIFTS[snap]:.1f})',
        panels=[
            (r'log10 SFR [Msun/yr]',   {'No FFB': sfr_norm, 'Li+24 FFB': sfr_li, 'MBK25 galaxies': sfr_bk}),
            (r'log10 Z*/Zsun',          {'No FFB': Z_norm,   'Li+24 FFB': Z_li,   'MBK25 galaxies': Z_bk}),
            (r'log10 Rdisk [kpc]',      {'No FFB': Rd_norm,  'Li+24 FFB': Rd_li,  'MBK25 galaxies': Rd_bk}),
            (r'log10 Rbulge [kpc]',     {'No FFB': Rb_norm,  'Li+24 FFB': Rb_li,  'MBK25 galaxies': Rb_bk}),
        ])

    fig.tight_layout()

    save_figure(fig, os.path.join(OUTPUT_DIR,
                'FFBPropertiesHistograms_Combined_DiskBulge' + OUTPUT_FORMAT))


# ========================== PLOT 12: STAR FORMATION HISTORIES ==========================

def plot_12_sfh_ffb(snapdata):
    """
    Star formation histories of the most massive FFB galaxies
    tracked across snapshots 8-63, with a dual x-axis
    (cosmic time + redshift).
    """
    print('Plot 12: Star formation histories of FFB galaxies')

    snap = SNAP_Z10
    if snap not in snapdata:
        print('  Snapshot not available. Skipping.')
        return

    d = snapdata[snap]

    w_ffb = np.where(
        (d['StellarMass'] > 0) & (d['FFBRegime'] == 1) & (d['Type'] == 0)
    )[0]
    w_normal = np.where(
        (d['StellarMass'] > 0) & (d['FFBRegime'] == 0) & (d['Type'] == 0)
    )[0]

    if len(w_ffb) == 0:
        print('  No FFB galaxies found at z~10. Skipping.')
        return

    # Select top FFB galaxies by mass
    N_track = min(5, len(w_ffb))
    mass_order = np.argsort(d['StellarMass'][w_ffb])[::-1]
    ffb_idx = w_ffb[mass_order[:N_track]]
    ffb_gal_ids = d['GalaxyIndex'][ffb_idx]

    # Snapshots to track through
    fig_g_snaps = [s for s in range(8, 64) if s in snapdata]

    # First, find all GalaxyIndex values that are EVER in FFB regime
    print('  Building set of galaxies that are ever FFB...')
    ever_ffb_gids = set()
    for s in fig_g_snaps:
        sd = snapdata[s]
        w_ffb_snap = np.where(sd['FFBRegime'] == 1)[0]
        ever_ffb_gids.update(sd['GalaxyIndex'][w_ffb_snap].astype(int))
    print(f'  Found {len(ever_ffb_gids)} galaxies that are FFB at some snapshot')

    # Filter non-FFB candidates to only those NEVER in FFB regime
    never_ffb_mask = np.array([int(d['GalaxyIndex'][i]) not in ever_ffb_gids
                               for i in w_normal])
    w_never_ffb = w_normal[never_ffb_mask]
    print(f'  Non-FFB candidates at z~10: {len(w_normal)}, never-FFB: {len(w_never_ffb)}')

    # Mass-match from the never-FFB pool
    norm_gal_ids = np.array([], dtype=np.int64)
    if len(w_never_ffb) > 0:
        norm_masses = d['StellarMass'][w_never_ffb]
        matched_norm_idx = []
        used = set()
        for fi in ffb_idx:
            ffb_mass = d['StellarMass'][fi]
            diffs = np.abs(norm_masses - ffb_mass)
            # Pick closest unused match
            order = np.argsort(diffs)
            for j in order:
                if j not in used:
                    matched_norm_idx.append(w_never_ffb[j])
                    used.add(j)
                    break
        if matched_norm_idx:
            norm_idx = np.array(matched_norm_idx)
            norm_gal_ids = d['GalaxyIndex'][norm_idx]

    print(f'  Selected {len(norm_gal_ids)} never-FFB galaxies for comparison')

    # Diagnostic: print FFB history for selected galaxies
    print('  --- FFB galaxies ---')
    for gid in ffb_gal_ids:
        ffb_snaps = []
        for s in fig_g_snaps:
            sd = snapdata[s]
            match = np.where(sd['GalaxyIndex'] == gid)[0]
            if len(match) > 0 and sd['FFBRegime'][match[0]] == 1:
                ffb_snaps.append(s)
        print(f'    GalaxyIndex {int(gid)}: FFB at snapshots {ffb_snaps}')

    if len(norm_gal_ids) > 0:
        print('  --- Non-FFB galaxies (verified never-FFB) ---')
        for gid in norm_gal_ids:
            print(f'    GalaxyIndex {int(gid)}: never FFB (verified)')

    cosmic_times = {s: cosmic_time_gyr(REDSHIFTS[s]) for s in fig_g_snaps}

    ffb_tracks = {int(gid): ([], []) for gid in ffb_gal_ids}
    norm_tracks = {int(gid): ([], []) for gid in norm_gal_ids}

    for s in fig_g_snaps:
        sd = snapdata[s]
        gids = sd['GalaxyIndex']
        sfr_total = sd['SfrDisk'] + sd['SfrBulge']
        t = cosmic_times[s]

        for gid in ffb_gal_ids:
            match = np.where(gids == gid)[0]
            if len(match) > 0:
                ffb_tracks[int(gid)][0].append(t)
                ffb_tracks[int(gid)][1].append(sfr_total[match[0]])

        for gid in norm_gal_ids:
            match = np.where(gids == gid)[0]
            if len(match) > 0:
                norm_tracks[int(gid)][0].append(t)
                norm_tracks[int(gid)][1].append(sfr_total[match[0]])

    fig, ax = plt.subplots()

    for i, gid in enumerate(ffb_gal_ids):
        times, sfrs = ffb_tracks[int(gid)]
        if len(times) > 1:
            sfrs = np.array(sfrs)
            lbl = 'FFB galaxies' if i == 0 else None
            ax.plot(times, sfrs, '-', color='firebrick', alpha=1.0,
                    lw=2.5, label=lbl, zorder=2)

    for i, gid in enumerate(norm_gal_ids):
        times, sfrs = norm_tracks[int(gid)]
        if len(times) > 1:
            sfrs = np.array(sfrs)
            lbl = 'non FFB galaxies' if i == 0 else None
            ax.plot(times, sfrs, '--', color='steelblue', alpha=1.0,
                    lw=1.5, label=lbl, zorder=3)

    ax.set_xlabel('Cosmic time [Gyr]')
    ax.set_ylabel(r'SFR [$M_{\odot}\,\mathrm{yr}^{-1}$]')
    ax.set_xlim(0, 4)

    # Top axis: redshift
    ax_top = ax.twiny()

    # Redshift ticks corresponding to cosmic times within 0-4 Gyr
    z_ticks = [10, 8, 6, 5, 4, 3, 2.5, 2]
    t_ticks = [cosmic_time_gyr(z) for z in z_ticks]
    # Only keep ticks within the x-axis limits
    xlim = ax.get_xlim()
    z_ticks_filtered = [z for z, t in zip(z_ticks, t_ticks) if xlim[0] <= t <= xlim[1]]
    t_ticks_filtered = [t for t in t_ticks if xlim[0] <= t <= xlim[1]]
    ax_top.set_xlim(xlim)
    ax_top.set_xticks(t_ticks_filtered)
    ax_top.set_xticklabels([str(z) for z in z_ticks_filtered])
    ax_top.set_xlabel('Redshift')

    _standard_legend(ax, loc='upper right')
    fig.tight_layout()

    save_figure(fig, os.path.join(OUTPUT_DIR,
                'SFH_FFB' + OUTPUT_FORMAT))


# ========================== PLOT 12b: FFB REGIME HISTORY ==========================

def plot_12b_ffb_regime_history(snapdata):
    """
    Timeline plot showing when each tracked galaxy is in the FFB regime.

    For each galaxy selected in plot_12 (most massive FFB galaxies at z~10 and
    mass-matched non-FFB galaxies), draws a horizontal bar coloured red when
    FFBRegime==1 and blue when FFBRegime==0.  The FFB→non-FFB transition for
    each red galaxy is marked with a vertical dashed line labelled with the
    transition redshift.

    Also verifies:
      - FFB galaxies are continuously FFB up to the transition and non-FFB
        afterwards (any violations are printed to console).
      - Non-FFB galaxies never enter the FFB regime (violations printed).
    """
    print('Plot 12b: FFB regime history of tracked galaxies')

    snap = SNAP_Z10
    if snap not in snapdata:
        print('  Snapshot not available. Skipping.')
        return

    d = snapdata[snap]

    w_ffb = np.where(
        (d['StellarMass'] > 0) & (d['FFBRegime'] == 1) & (d['Type'] == 0)
    )[0]
    w_normal = np.where(
        (d['StellarMass'] > 0) & (d['FFBRegime'] == 0) & (d['Type'] == 0)
    )[0]

    if len(w_ffb) == 0:
        print('  No FFB galaxies found at z~10. Skipping.')
        return

    # Select top FFB galaxies by stellar mass
    N_track = min(10, len(w_ffb))
    mass_order = np.argsort(d['StellarMass'][w_ffb])[::-1]
    ffb_idx = w_ffb[mass_order[:N_track]]
    ffb_gal_ids = d['GalaxyIndex'][ffb_idx]

    # Build set of galaxies ever in FFB regime (to exclude from normal pool)
    fig_g_snaps = [s for s in range(8, 64) if s in snapdata]
    ever_ffb_gids = set()
    for s in fig_g_snaps:
        sd = snapdata[s]
        w_ffb_snap = np.where(sd['FFBRegime'] == 1)[0]
        ever_ffb_gids.update(sd['GalaxyIndex'][w_ffb_snap].astype(int))

    # Mass-match non-FFB galaxies from the never-FFB pool
    never_ffb_mask = np.array([int(d['GalaxyIndex'][i]) not in ever_ffb_gids
                               for i in w_normal])
    w_never_ffb = w_normal[never_ffb_mask]

    norm_gal_ids = np.array([], dtype=np.int64)
    if len(w_never_ffb) > 0:
        norm_masses = d['StellarMass'][w_never_ffb]
        matched_norm_idx = []
        used = set()
        for fi in ffb_idx:
            ffb_mass = d['StellarMass'][fi]
            diffs = np.abs(norm_masses - ffb_mass)
            for j in np.argsort(diffs):
                if j not in used:
                    matched_norm_idx.append(w_never_ffb[j])
                    used.add(j)
                    break
        if matched_norm_idx:
            norm_idx = np.array(matched_norm_idx)
            norm_gal_ids = d['GalaxyIndex'][norm_idx]

    cosmic_times = {s: cosmic_time_gyr(REDSHIFTS[s]) for s in fig_g_snaps}

    # Collect (cosmic_time, FFBRegime) per galaxy
    # ffb_regime_tracks[gid] = list of (t, regime, snap)
    all_gal_ids = list(ffb_gal_ids.astype(int)) + list(norm_gal_ids.astype(int))
    regime_tracks = {gid: [] for gid in all_gal_ids}

    for s in fig_g_snaps:
        sd = snapdata[s]
        gids = sd['GalaxyIndex']
        t = cosmic_times[s]
        for gid in all_gal_ids:
            match = np.where(gids == gid)[0]
            if len(match) > 0:
                regime_tracks[gid].append((t, int(sd['FFBRegime'][match[0]]), s))

    # ---- Diagnostics ----
    print('  --- FFB galaxy regime history ---')
    ffb_transition_times = {}   # gid -> (t_first_nonffb_after_ffb, z_transition)
    for gid in ffb_gal_ids.astype(int):
        track = regime_tracks[gid]
        if not track:
            continue
        track_sorted = sorted(track, key=lambda x: x[0])
        regimes = [(t, r, s) for t, r, s in track_sorted]

        transition_idx = None
        for k in range(len(regimes) - 1):
            if regimes[k][1] == 1 and regimes[k + 1][1] == 0:
                transition_idx = k + 1
                break

        if transition_idx is None:
            print(f'    GalaxyIndex {gid}: no FFB→non-FFB transition in tracked range')
            continue

        t_trans, _, s_trans = regimes[transition_idx]
        z_trans = REDSHIFTS[s_trans]
        ffb_transition_times[gid] = (t_trans, z_trans)
        print(f'    GalaxyIndex {gid}: first non-FFB after FFB at z={z_trans:.2f} '
              f'(t={t_trans:.3f} Gyr)')

        # Clean-transition check: any non-FFB point before first 1->0 crossing?
        for t, r, s in regimes[:transition_idx]:
            if r == 0:
                z_viol = REDSHIFTS[s]
                print(f'    WARNING: GalaxyIndex {gid} has FFBRegime=0 at '
                      f'z={z_viol:.2f} (snap {s}) before first 1→0 crossing')

    print('  --- Non-FFB galaxy verification ---')
    for gid in norm_gal_ids.astype(int):
        track = regime_tracks[gid]
        ffb_violations = [(t, s) for t, r, s in track if r == 1]
        if ffb_violations:
            for t_v, s_v in ffb_violations:
                print(f'    WARNING: Non-FFB galaxy {gid} has FFBRegime=1 at '
                      f'z={REDSHIFTS[s_v]:.2f} (snap {s_v})')
        else:
            print(f'    GalaxyIndex {gid}: confirmed never-FFB throughout')

    # ---- Oscillation check (all tracked galaxies) ----
    print('  --- Oscillation check ---')
    any_oscillation = False
    for gid in all_gal_ids:
        track = sorted(regime_tracks[gid], key=lambda x: x[0])
        regime_seq = [r for t, r, s in track]
        # Count transitions: consecutive pairs that differ
        transitions = [(track[k], track[k + 1])
                       for k in range(len(regime_seq) - 1)
                       if regime_seq[k] != regime_seq[k + 1]]
        n_transitions = len(transitions)
        if n_transitions > 1:
            any_oscillation = True
            tag = 'FFB' if gid in ffb_gal_ids.astype(int) else 'non-FFB'
            print(f'    OSCILLATION: GalaxyIndex {gid} ({tag}) switches '
                  f'{n_transitions} times:')
            for (t0, r0, s0), (t1, r1, s1) in transitions:
                print(f'      z={REDSHIFTS[s0]:.2f} → z={REDSHIFTS[s1]:.2f}  '
                      f'FFBRegime {r0} → {r1}')
        elif n_transitions == 1:
            (t0, r0, s0), (t1, r1, s1) = transitions[0]
            tag = 'FFB' if gid in ffb_gal_ids.astype(int) else 'non-FFB'
            print(f'    GalaxyIndex {gid} ({tag}): single clean transition '
                  f'FFBRegime {r0}→{r1} at z={REDSHIFTS[s1]:.2f}')
        else:
            tag = 'FFB' if gid in ffb_gal_ids.astype(int) else 'non-FFB'
            regime_val = regime_seq[0] if regime_seq else '?'
            print(f'    GalaxyIndex {gid} ({tag}): no transitions — '
                  f'always FFBRegime={regime_val}')
    if not any_oscillation:
        print('  No oscillating galaxies found.')

    # ---- Plot ----
    n_ffb  = len(ffb_gal_ids)
    n_norm = len(norm_gal_ids)
    n_total = n_ffb + n_norm

    fig, ax = plt.subplots(figsize=(8, 0.7 * n_total + 1.5))
    x_min, x_max = 0.0, 2.5

    row_labels = []
    transition_marked = False   # for legend deduplication

    for row_idx, gid in enumerate(list(ffb_gal_ids.astype(int)) +
                                   list(norm_gal_ids.astype(int))):
        is_ffb_gal = gid in ffb_gal_ids.astype(int)
        track = sorted(regime_tracks[gid], key=lambda x: x[0])
        if not track:
            row_labels.append(str(gid))
            continue

        times   = [t for t, r, s in track]
        regimes = [r for t, r, s in track]

        # Draw segments between consecutive snapshots
        for k in range(len(times) - 1):
            t0, t1 = times[k], times[k + 1]
            r = regimes[k]
            color = 'firebrick' if r == 1 else 'steelblue'
            lw = 3.5
            ax.plot([t0, t1], [row_idx, row_idx], '-', color=color, lw=lw,
                    solid_capstyle='butt', zorder=2)

        # Final segment (last snap → extend half a step for visibility)
        if len(times) >= 2:
            dt = times[-1] - times[-2]
        else:
            dt = 0.05
        r_last = regimes[-1]
        color_last = 'firebrick' if r_last == 1 else 'steelblue'
        ax.plot([times[-1], times[-1] + 0.5 * dt], [row_idx, row_idx],
                '-', color=color_last, lw=3.5, solid_capstyle='butt', zorder=2)

        # Mark transition for FFB galaxies
        if is_ffb_gal and gid in ffb_transition_times:
            t_trans, z_trans = ffb_transition_times[gid]
            if x_min <= t_trans <= x_max:
                vline_lbl = 'FFB → non-FFB transition' if not transition_marked else None
                ax.axvline(t_trans, color='goldenrod', ls='--', lw=1.2,
                           alpha=0.85, zorder=3, label=vline_lbl)
                transition_marked = True
                ax.annotate(
                    fr'$z={z_trans:.1f}$',
                    xy=(t_trans, row_idx),
                    xytext=(4, 3), textcoords='offset points',
                    fontsize=7, color='goldenrod', va='bottom',
                )

        mstar = d['StellarMass'][
            np.where(d['GalaxyIndex'] == gid)[0][0]] * 1e10
        tag = 'FFB' if is_ffb_gal else 'non-FFB'
        row_labels.append(fr'{tag}  $\log M_*={np.log10(mstar):.1f}$')

    # Y-axis: one row per galaxy
    ax.set_yticks(range(n_total))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_ylim(-0.6, n_total - 0.4)

    ax.set_xlabel('Cosmic time [Gyr]')
    ax.set_xlim(x_min, x_max)

    # Top axis: redshift
    ax_top = ax.twiny()
    z_ticks = [10, 8, 6, 5, 4, 3, 2.5, 2]
    t_ticks = [cosmic_time_gyr(z) for z in z_ticks]
    xlim = ax.get_xlim()
    z_ticks_f = [z for z, t in zip(z_ticks, t_ticks) if xlim[0] <= t <= xlim[1]]
    t_ticks_f = [t for t in t_ticks if xlim[0] <= t <= xlim[1]]
    ax_top.set_xlim(xlim)
    ax_top.set_xticks(t_ticks_f)
    ax_top.set_xticklabels([str(z) for z in z_ticks_f])
    ax_top.set_xlabel('Redshift')

    # Custom legend patches
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color='firebrick',   label='FFB regime (FFBRegime=1)'),
        mpatches.Patch(color='steelblue',  label='Non-FFB regime (FFBRegime=0)'),
        plt.Line2D([0], [0], color='goldenrod', ls='--', lw=1.5,
                   label='FFB → non-FFB transition'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=8,
              framealpha=0.9)

    fig.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR, 'SFH_FFB_regime_history' + OUTPUT_FORMAT))


# ========================== PLOT 12c: FFB REGIME HEATMAP (large sample) ==========================

def plot_12c_ffb_regime_heatmap(snapdata):
    """
    Heatmap of FFBRegime over time for a random sample of 100 FFB and 100
    non-FFB central galaxies selected at z~10.

    Rows = galaxies (FFB on top, non-FFB below, separated by a gap).
    Columns = snapshots ordered by cosmic time.
    Colour = red (FFBRegime=1) / blue (FFBRegime=0) / grey (galaxy not present).

    FFB galaxies are sorted by their last-FFB snapshot so any transition
    front shows as a diagonal edge.  Oscillations appear as isolated red/blue
    specks against the dominant colour.

    Console output summarises oscillation counts for both groups.
    """
    print('Plot 12c: FFB regime heatmap (100+100 sample)')

    snap = SNAP_Z10
    if snap not in snapdata:
        print('  Snapshot not available. Skipping.')
        return

    d = snapdata[snap]

    w_ffb = np.where(
        (d['StellarMass'] > 0) & (d['FFBRegime'] == 1) & (d['Type'] == 0)
    )[0]
    w_normal = np.where(
        (d['StellarMass'] > 0) & (d['FFBRegime'] == 0) & (d['Type'] == 0)
    )[0]

    if len(w_ffb) == 0:
        print('  No FFB galaxies found at z~10. Skipping.')
        return

    N_sample = 100
    rng = np.random.default_rng(seed=42)

    # Random sample (or all if fewer than N_sample)
    ffb_sample_idx  = rng.choice(w_ffb,   size=min(N_sample, len(w_ffb)),   replace=False)
    norm_sample_idx = rng.choice(w_normal, size=min(N_sample, len(w_normal)), replace=False)

    ffb_gal_ids  = d['GalaxyIndex'][ffb_sample_idx].astype(int)
    norm_gal_ids = d['GalaxyIndex'][norm_sample_idx].astype(int)
    all_gal_ids  = list(ffb_gal_ids) + list(norm_gal_ids)

    fig_g_snaps   = [s for s in range(8, 64) if s in snapdata]
    cosmic_times  = {s: cosmic_time_gyr(REDSHIFTS[s]) for s in fig_g_snaps}
    snap_times    = [cosmic_times[s] for s in fig_g_snaps]
    snap_redshifts = [REDSHIFTS[s] for s in fig_g_snaps]

    # Build regime matrix: shape (n_gal, n_snap), NaN = not present
    n_gal  = len(all_gal_ids)
    n_snap = len(fig_g_snaps)
    regime_matrix = np.full((n_gal, n_snap), np.nan)

    for j, s in enumerate(fig_g_snaps):
        sd   = snapdata[s]
        gids = sd['GalaxyIndex']
        for i, gid in enumerate(all_gal_ids):
            match = np.where(gids == gid)[0]
            if len(match) > 0:
                regime_matrix[i, j] = sd['FFBRegime'][match[0]]

    # ---- Oscillation check ----
    def count_transitions(row):
        present = ~np.isnan(row)
        vals = row[present].astype(int)
        return int(np.sum(np.diff(vals) != 0))

    print('  --- Oscillation summary ---')
    for label, indices in [('FFB', range(len(ffb_gal_ids))),
                           ('non-FFB', range(len(ffb_gal_ids),
                                             len(ffb_gal_ids) + len(norm_gal_ids)))]:
        n_trans = [count_transitions(regime_matrix[i]) for i in indices]
        n_osc   = sum(1 for n in n_trans if n > 1)
        n_clean = sum(1 for n in n_trans if n == 1)
        n_stable = sum(1 for n in n_trans if n == 0)
        print(f'  {label} ({len(list(indices))} galaxies):')
        print(f'    No transitions (stable):  {n_stable}')
        print(f'    Single clean transition:   {n_clean}')
        print(f'    Oscillating (>1 transition): {n_osc}')
        if n_osc > 0:
            osc_counts = sorted([n for n in n_trans if n > 1], reverse=True)
            print(f'    Transition counts: {osc_counts}')

    # ---- Sort FFB rows by last-FFB snapshot for a clean transition front ----
    def last_ffb_snap_idx(row):
        ffb_cols = np.where(row == 1)[0]
        return int(ffb_cols.max()) if len(ffb_cols) > 0 else -1

    ffb_sort_order  = sorted(range(len(ffb_gal_ids)),
                             key=lambda i: last_ffb_snap_idx(regime_matrix[i]))
    norm_sort_order = sorted(range(len(norm_gal_ids)),
                             key=lambda i: last_ffb_snap_idx(
                                 regime_matrix[len(ffb_gal_ids) + i]))

    sorted_ffb_rows  = regime_matrix[ffb_sort_order]
    sorted_norm_rows = regime_matrix[[len(ffb_gal_ids) + i for i in norm_sort_order]]

    # Gap row of NaNs between the two groups
    gap_rows = np.full((3, n_snap), np.nan)
    plot_matrix = np.vstack([sorted_ffb_rows, gap_rows, sorted_norm_rows])

    # ---- Build custom colormap ----
    import matplotlib.colors as mcolors
    cmap = mcolors.ListedColormap(['steelblue', 'firebrick'])
    cmap.set_bad(color='lightgrey')   # NaN = not present / gap
    norm_cmap = mcolors.BoundaryNorm([0, 0.5, 1.0], cmap.N)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(plot_matrix, aspect='auto', cmap=cmap, norm=norm_cmap,
                   interpolation='nearest',
                   extent=[snap_times[0], snap_times[-1],
                           plot_matrix.shape[0], 0])

    # Horizontal separator between FFB and non-FFB groups
    sep_y = len(ffb_gal_ids) + len(gap_rows) / 2
    ax.axhline(sep_y, color='black', lw=1.0, ls='-')

    # Y-axis labels
    ax.set_ylabel('Galaxy index (sorted)')
    n_ffb_shown  = len(ffb_gal_ids)
    n_norm_shown = len(norm_gal_ids)
    mid_ffb  = n_ffb_shown / 2
    mid_norm = n_ffb_shown + len(gap_rows) + n_norm_shown / 2
    ax.set_yticks([mid_ffb, mid_norm])
    ax.set_yticklabels([f'FFB at z~10\n(n={n_ffb_shown})',
                        f'non-FFB at z~10\n(n={n_norm_shown})'])

    ax.set_xlabel('Cosmic time [Gyr]')
    ax.set_xlim(snap_times[0], snap_times[-1])

    # Top axis: redshift
    ax_top = ax.twiny()
    z_ticks = [10, 8, 6, 5, 4, 3, 2.5, 2]
    t_ticks = [cosmic_time_gyr(z) for z in z_ticks]
    xlim = (snap_times[0], snap_times[-1])
    z_ticks_f = [z for z, t in zip(z_ticks, t_ticks) if xlim[0] <= t <= xlim[1]]
    t_ticks_f = [t for t in t_ticks if xlim[0] <= t <= xlim[1]]
    ax_top.set_xlim(xlim)
    ax_top.set_xticks(t_ticks_f)
    ax_top.set_xticklabels([str(z) for z in z_ticks_f])
    ax_top.set_xlabel('Redshift')

    # Legend
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color='firebrick',  label='FFB regime'),
        mpatches.Patch(color='steelblue', label='Non-FFB regime'),
        mpatches.Patch(color='lightgrey',  label='Not present / gap'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=9,
              framealpha=0.9)

    fig.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR,
                'SFH_FFB_regime_heatmap' + OUTPUT_FORMAT))


# ========================== PLOT 12d: SFH FFB WITH TRANSITION MARKERS ==========================

def plot_12d_sfh_ffb_transitions(snapdata):
    """
    Like plot_12_sfh_ffb but additionally:
      - Tracks FFBRegime at every snapshot for each plotted galaxy.
      - For FFB galaxies (red): marks the last snapshot where FFBRegime==1
        with a vertical dashed line and annotates the transition redshift.
        Prints a warning if FFBRegime is not continuously 1 up to that point.
      - For non-FFB galaxies (blue): verifies FFBRegime==0 throughout and
        prints a warning for any snapshot where it is 1.
    """
    print('Plot 12d: SFH of FFB galaxies with transition redshift markers')

    snap = SNAP_Z10
    if snap not in snapdata:
        print('  Snapshot not available. Skipping.')
        return

    d = snapdata[snap]

    w_ffb = np.where(
        (d['StellarMass'] > 0) & (d['FFBRegime'] == 1) & (d['Type'] == 0)
    )[0]
    w_normal = np.where(
        (d['StellarMass'] > 0) & (d['FFBRegime'] == 0) & (d['Type'] == 0)
    )[0]

    if len(w_ffb) == 0:
        print('  No FFB galaxies found at z~10. Skipping.')
        return

    N_track    = min(10, len(w_ffb))
    mass_order = np.argsort(d['StellarMass'][w_ffb])[::-1]
    ffb_idx    = w_ffb[mass_order[:N_track]]
    ffb_gal_ids = d['GalaxyIndex'][ffb_idx]

    fig_g_snaps = [s for s in range(8, 64) if s in snapdata]
    ever_ffb_gids = set()
    for s in fig_g_snaps:
        sd = snapdata[s]
        w_e = np.where(sd['FFBRegime'] == 1)[0]
        ever_ffb_gids.update(sd['GalaxyIndex'][w_e].astype(int))

    never_ffb_mask = np.array([int(d['GalaxyIndex'][i]) not in ever_ffb_gids
                                for i in w_normal])
    w_never_ffb = w_normal[never_ffb_mask]

    norm_gal_ids = np.array([], dtype=np.int64)
    if len(w_never_ffb) > 0:
        norm_masses = d['StellarMass'][w_never_ffb]
        matched_norm_idx, used = [], set()
        for fi in ffb_idx:
            for j in np.argsort(np.abs(norm_masses - d['StellarMass'][fi])):
                if j not in used:
                    matched_norm_idx.append(w_never_ffb[j])
                    used.add(j)
                    break
        if matched_norm_idx:
            norm_gal_ids = d['GalaxyIndex'][np.array(matched_norm_idx)]

    cosmic_times = {s: cosmic_time_gyr(REDSHIFTS[s]) for s in fig_g_snaps}

    # Track SFR and FFBRegime per galaxy
    ffb_tracks  = {int(gid): {'t': [], 'sfr': [], 'regime': [], 'snap': []}
                   for gid in ffb_gal_ids}
    norm_tracks = {int(gid): {'t': [], 'sfr': [], 'regime': [], 'snap': []}
                   for gid in norm_gal_ids}

    for s in fig_g_snaps:
        sd   = snapdata[s]
        gids = sd['GalaxyIndex']
        sfr_total = sd['SfrDisk'] + sd['SfrBulge']
        t = cosmic_times[s]

        for gid in ffb_gal_ids:
            match = np.where(gids == gid)[0]
            if len(match) > 0:
                m = match[0]
                ffb_tracks[int(gid)]['t'].append(t)
                ffb_tracks[int(gid)]['sfr'].append(sfr_total[m])
                ffb_tracks[int(gid)]['regime'].append(int(sd['FFBRegime'][m]))
                ffb_tracks[int(gid)]['snap'].append(s)

        for gid in norm_gal_ids:
            match = np.where(gids == gid)[0]
            if len(match) > 0:
                m = match[0]
                norm_tracks[int(gid)]['t'].append(t)
                norm_tracks[int(gid)]['sfr'].append(sfr_total[m])
                norm_tracks[int(gid)]['regime'].append(int(sd['FFBRegime'][m]))
                norm_tracks[int(gid)]['snap'].append(s)

    # ---- Diagnostics ----
    print('  --- FFB galaxy transition analysis ---')
    ffb_transition = {}   # gid -> (t_first_nonffb_after_ffb, z_at_transition)
    for gid in ffb_gal_ids.astype(int):
        tr = ffb_tracks[gid]
        if not tr['t']:
            continue
        pairs = sorted(zip(tr['t'], tr['regime'], tr['snap']))
        t_vals, r_vals, s_vals = zip(*pairs)

        ffb_indices = [k for k, r in enumerate(r_vals) if r == 1]
        if not ffb_indices:
            print(f'    GalaxyIndex {gid}: no FFB snaps in tracked range')
            continue

        transition_idx = None
        for k in range(len(r_vals) - 1):
            if r_vals[k] == 1 and r_vals[k + 1] == 0:
                transition_idx = k + 1
                break

        if transition_idx is None:
            print(f'    GalaxyIndex {gid}: no FFB→non-FFB transition in tracked range')
            continue

        t_trans = t_vals[transition_idx]
        z_trans = REDSHIFTS[s_vals[transition_idx]]
        ffb_transition[gid] = (t_trans, z_trans)
        print(f'    GalaxyIndex {gid}: first non-FFB after FFB at z={z_trans:.2f} '
              f'(t={t_trans:.3f} Gyr)')

        # Clean-transition check: any non-FFB point before the first 1->0 crossing?
        for k in range(transition_idx):
            if r_vals[k] == 0:
                print(f'      WARNING: FFBRegime=0 at z={REDSHIFTS[s_vals[k]]:.2f} '
                      f'(snap {s_vals[k]}) before first 1→0 crossing — not a clean transition')

    print('  --- Non-FFB galaxy verification ---')
    for gid in norm_gal_ids.astype(int):
        tr = norm_tracks[gid]
        violations = [(t, s) for t, r, s in zip(tr['t'], tr['regime'], tr['snap'])
                      if r == 1]
        if violations:
            for t_v, s_v in violations:
                print(f'    WARNING: GalaxyIndex {gid} has FFBRegime=1 at '
                      f'z={REDSHIFTS[s_v]:.2f} (snap {s_v})')
        else:
            print(f'    GalaxyIndex {gid}: confirmed never-FFB throughout')

    # ---- Plot ----
    fig, ax = plt.subplots()

    ffb_regime_label = False
    nonffb_regime_label = False

    all_plot_ids = list(ffb_gal_ids.astype(int)) + list(norm_gal_ids.astype(int))
    for gid in all_plot_ids:
        tr = ffb_tracks.get(gid, norm_tracks.get(gid, None))
        if tr is None or len(tr['t']) <= 1:
            continue

        pairs = sorted(zip(tr['t'], tr['sfr'], tr['regime'], tr['snap']),
                       key=lambda x: x[0])

        for k in range(len(pairs) - 1):
            t0, sfr0, r0, s0 = pairs[k]
            t1, sfr1, r1, s1 = pairs[k + 1]

            color = 'k' if r0 == 1 else 'firebrick'
            ls = '-' if r0 == 1 else '--'
            lbl = None
            if r0 == 1 and not ffb_regime_label:
                lbl = 'Li+24 FFB galaxies'
                ffb_regime_label = True
            elif r0 == 0 and not nonffb_regime_label:
                lbl = 'Non-FFB galaxies'
                nonffb_regime_label = True

            ax.plot([t0, t1], [sfr0, sfr1], ls, color=color,
                    alpha=1.0, lw=2.2, label=lbl, zorder=2)

    ax.set_xlabel('Cosmic time [Gyr]')
    ax.set_ylabel(r'$\log_{10}\,\mathrm{SFR}\;[M_{\odot}\,\mathrm{yr}^{-1}]$')
    ax.set_yscale('log')
    ax.set_ylim(1e-3, 1e5)

    # x-axis fixed to requested range: min snapshot time to 1.0 Gyr
    t_min = min(cosmic_times[s] for s in fig_g_snaps)
    t_max = 1.0
    print(f'  t_min = {t_min:.2f} Gyr, t_max = {t_max:.2f} Gyr')
    print(f'  Transition times: ' +
          ', '.join(f'z={_z:.1f} (t={t:.2f} Gyr)'
                    for t, _z in sorted(ffb_transition.values())))
    ax.set_xlim(t_min, t_max)

    # Mark FFB -> non-FFB transitions
    for gid in ffb_gal_ids.astype(int):
        if gid in ffb_transition:
            t_trans, z_trans = ffb_transition[gid]
            if not (t_min <= t_trans <= t_max):
                continue
            ax.axvline(t_trans, color='goldenrod', ls='--', lw=1.2,
                       alpha=0.85, zorder=4)
            # ax.annotate(fr'$z={z_trans:.1f}$',
            #             xy=(t_trans, ax.get_ylim()[0]),
            #             xytext=(3, 6), textcoords='offset points',
            #             fontsize=7, color='goldenrod', va='bottom',
            #             rotation=90)

    # Top axis: redshift
    ax_top = ax.twiny()
    z_ticks = [10, 8, 6, 5, 4, 3, 2.5, 2, 1.5, 1]
    t_ticks = [cosmic_time_gyr(z) for z in z_ticks]
    xlim = ax.get_xlim()
    z_ticks_f = [z for z, t in zip(z_ticks, t_ticks) if xlim[0] <= t <= xlim[1]]
    t_ticks_f = [t for t in t_ticks if xlim[0] <= t <= xlim[1]]
    ax_top.set_xlim(xlim)
    ax_top.set_xticks(t_ticks_f)
    ax_top.set_xticklabels([str(z) for z in z_ticks_f])
    ax_top.set_xlabel('Redshift')

    # Add transition marker to legend
    import matplotlib.lines as mlines
    trans_handle = mlines.Line2D([], [], color='goldenrod', ls='--', lw=1.5,
                                 label='FFB → non-FFB transition')
    handles, labels = ax.get_legend_handles_labels()
    _standard_legend(ax, loc='upper left',
                     handles=handles + [trans_handle],
                     labels=labels + ['FFB → non-FFB transition'])

    fig.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR,
                'SFH_FFB_transitions' + OUTPUT_FORMAT))


# ========================== PLOT 12e: SFH FFB TRANSITIONS (MBK25) ==========================

def plot_12e_sfh_ffb_transitions_mbk25(snapdata):
    """
    MBK25 version of plot_12d_sfh_ffb_transitions.
    Loads data from FFB_BK25_SMOOTH_DIR instead of the primary model.
    """
    print('Plot 12e: SFH of MBK25 FFB galaxies with transition redshift markers')

    needed_snaps = list(range(8, 64))
    mbk25_snapdata = load_snapshots(FFB_BK25_SMOOTH_DIR, needed_snaps)
    if not mbk25_snapdata:
        print('  MBK25 smooth data not available. Skipping.')
        return

    if SNAP_Z10 not in mbk25_snapdata:
        print('  Snapshot z~10 not available in MBK25 data. Skipping.')
        return

    d = mbk25_snapdata[SNAP_Z10]

    w_ffb = np.where(
        (d['StellarMass'] > 0) & (d['FFBRegime'] == 1) & (d['Type'] == 0)
    )[0]
    w_normal = np.where(
        (d['StellarMass'] > 0) & (d['FFBRegime'] == 0) & (d['Type'] == 0)
    )[0]

    if len(w_ffb) == 0:
        print('  No MBK25 galaxies found at z~10. Skipping.')
        return

    N_track    = min(10, len(w_ffb))
    mass_order = np.argsort(d['StellarMass'][w_ffb])[::-1]
    ffb_idx    = w_ffb[mass_order[:N_track]]
    ffb_gal_ids = d['GalaxyIndex'][ffb_idx]

    fig_g_snaps = [s for s in needed_snaps if s in mbk25_snapdata]
    ever_ffb_gids = set()
    for s in fig_g_snaps:
        sd = mbk25_snapdata[s]
        w_e = np.where(sd['FFBRegime'] == 1)[0]
        ever_ffb_gids.update(sd['GalaxyIndex'][w_e].astype(int))

    never_ffb_mask = np.array([int(d['GalaxyIndex'][i]) not in ever_ffb_gids
                                for i in w_normal])
    w_never_ffb = w_normal[never_ffb_mask]

    norm_gal_ids = np.array([], dtype=np.int64)
    if len(w_never_ffb) > 0:
        norm_masses = d['StellarMass'][w_never_ffb]
        matched_norm_idx, used = [], set()
        for fi in ffb_idx:
            for j in np.argsort(np.abs(norm_masses - d['StellarMass'][fi])):
                if j not in used:
                    matched_norm_idx.append(w_never_ffb[j])
                    used.add(j)
                    break
        if matched_norm_idx:
            norm_gal_ids = d['GalaxyIndex'][np.array(matched_norm_idx)]

    cosmic_times = {s: cosmic_time_gyr(REDSHIFTS[s]) for s in fig_g_snaps}

    ffb_tracks  = {int(gid): {'t': [], 'sfr': [], 'regime': [], 'snap': []}
                   for gid in ffb_gal_ids}
    norm_tracks = {int(gid): {'t': [], 'sfr': [], 'regime': [], 'snap': []}
                   for gid in norm_gal_ids}

    for s in fig_g_snaps:
        sd   = mbk25_snapdata[s]
        gids = sd['GalaxyIndex']
        sfr_total = sd['SfrDisk'] + sd['SfrBulge']
        t = cosmic_times[s]

        for gid in ffb_gal_ids:
            match = np.where(gids == gid)[0]
            if len(match) > 0:
                m = match[0]
                ffb_tracks[int(gid)]['t'].append(t)
                ffb_tracks[int(gid)]['sfr'].append(sfr_total[m])
                ffb_tracks[int(gid)]['regime'].append(int(sd['FFBRegime'][m]))
                ffb_tracks[int(gid)]['snap'].append(s)

        for gid in norm_gal_ids:
            match = np.where(gids == gid)[0]
            if len(match) > 0:
                m = match[0]
                norm_tracks[int(gid)]['t'].append(t)
                norm_tracks[int(gid)]['sfr'].append(sfr_total[m])
                norm_tracks[int(gid)]['regime'].append(int(sd['FFBRegime'][m]))
                norm_tracks[int(gid)]['snap'].append(s)

    ffb_transition = {}
    for gid in ffb_gal_ids.astype(int):
        tr = ffb_tracks[gid]
        if not tr['t']:
            continue
        pairs = sorted(zip(tr['t'], tr['regime'], tr['snap']))
        t_vals, r_vals, s_vals = zip(*pairs)

        transition_idx = None
        for k in range(len(r_vals) - 1):
            if r_vals[k] == 1 and r_vals[k + 1] == 0:
                transition_idx = k + 1
                break

        if transition_idx is None:
            continue

        t_trans = t_vals[transition_idx]
        z_trans = REDSHIFTS[s_vals[transition_idx]]
        ffb_transition[gid] = (t_trans, z_trans)

    fig, ax = plt.subplots()

    ffb_regime_label = False
    nonffb_regime_label = False

    all_plot_ids = list(ffb_gal_ids.astype(int)) + list(norm_gal_ids.astype(int))
    for gid in all_plot_ids:
        tr = ffb_tracks.get(gid, norm_tracks.get(gid, None))
        if tr is None or len(tr['t']) <= 1:
            continue

        pairs = sorted(zip(tr['t'], tr['sfr'], tr['regime'], tr['snap']),
                       key=lambda x: x[0])

        for k in range(len(pairs) - 1):
            t0, sfr0, r0, s0 = pairs[k]
            t1, sfr1, r1, s1 = pairs[k + 1]

            color = 'mediumpurple' if r0 == 1 else 'firebrick'
            ls = '-' if r0 == 1 else '--'
            lbl = None
            if r0 == 1 and not ffb_regime_label:
                lbl = 'MBK25 galaxies'
                ffb_regime_label = True
            elif r0 == 0 and not nonffb_regime_label:
                lbl = 'Non-MBK25 regime'
                nonffb_regime_label = True

            ax.plot([t0, t1], [sfr0, sfr1], ls, color=color,
                    alpha=1.0, lw=2.2, label=lbl, zorder=2)

    ax.set_xlabel('Cosmic time [Gyr]')
    ax.set_ylabel(r'$\log_{10}\,\mathrm{SFR}\;[M_{\odot}\,\mathrm{yr}^{-1}]$')
    ax.set_yscale('log')
    ax.set_ylim(1e-3, 1e5)

    t_min = min(cosmic_times[s] for s in fig_g_snaps)
    t_max = 1.0
    ax.set_xlim(t_min, t_max)

    for gid in ffb_gal_ids.astype(int):
        if gid in ffb_transition:
            t_trans, z_trans = ffb_transition[gid]
            if not (t_min <= t_trans <= t_max):
                continue
            ax.axvline(t_trans, color='goldenrod', ls='--', lw=1.2,
                       alpha=0.85, zorder=4)

    ax_top = ax.twiny()
    z_ticks = [10, 8, 6, 5, 4, 3, 2.5, 2, 1.5, 1]
    t_ticks = [cosmic_time_gyr(z) for z in z_ticks]
    xlim = ax.get_xlim()
    z_ticks_f = [z for z, t in zip(z_ticks, t_ticks) if xlim[0] <= t <= xlim[1]]
    t_ticks_f = [t for t in t_ticks if xlim[0] <= t <= xlim[1]]
    ax_top.set_xlim(xlim)
    ax_top.set_xticks(t_ticks_f)
    ax_top.set_xticklabels([str(z) for z in z_ticks_f])
    ax_top.set_xlabel('Redshift')

    import matplotlib.lines as mlines
    trans_handle = mlines.Line2D([], [], color='goldenrod', ls='--', lw=1.5,
                                 label='MBK25 → non-MBK25 transition')
    handles, labels = ax.get_legend_handles_labels()
    _standard_legend(ax, loc='upper left',
                     handles=handles + [trans_handle],
                     labels=labels + ['MBK25 → non-MBK25 transition'])

    fig.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR,
                'SFH_FFB_transitions_MBK25' + OUTPUT_FORMAT))


# ============ PLOT 12f: SFH TRANSITIONS, FFB + MBK25 STACKED ============

def _sfh_transition_tracks(source_snapdata, snaps):
    """
    Shared sample selection and history tracking for the stacked transition
    figure.  Mirrors plot_12d/plot_12e: take the N most massive FFB centrals
    at z~10, mass-match a never-FFB control for each, then follow the SFR and
    FFBRegime of both sets across `snaps`.

    Returns None if the source has no usable galaxies, otherwise a dict with
    the tracks, the plotting order, the FFB->non-FFB transition times and the
    cosmic-time lookup.
    """
    if SNAP_Z10 not in source_snapdata:
        return None

    d = source_snapdata[SNAP_Z10]

    w_ffb = np.where(
        (d['StellarMass'] > 0) & (d['FFBRegime'] == 1) & (d['Type'] == 0)
    )[0]
    w_normal = np.where(
        (d['StellarMass'] > 0) & (d['FFBRegime'] == 0) & (d['Type'] == 0)
    )[0]

    if len(w_ffb) == 0:
        return None

    N_track    = min(10, len(w_ffb))
    mass_order = np.argsort(d['StellarMass'][w_ffb])[::-1]
    ffb_idx    = w_ffb[mass_order[:N_track]]
    ffb_gal_ids = d['GalaxyIndex'][ffb_idx]

    fig_g_snaps = [s for s in snaps if s in source_snapdata]
    ever_ffb_gids = set()
    for s in fig_g_snaps:
        sd = source_snapdata[s]
        w_e = np.where(sd['FFBRegime'] == 1)[0]
        ever_ffb_gids.update(sd['GalaxyIndex'][w_e].astype(int))

    never_ffb_mask = np.array([int(d['GalaxyIndex'][i]) not in ever_ffb_gids
                                for i in w_normal])
    w_never_ffb = w_normal[never_ffb_mask]

    norm_gal_ids = np.array([], dtype=np.int64)
    if len(w_never_ffb) > 0:
        norm_masses = d['StellarMass'][w_never_ffb]
        matched_norm_idx, used = [], set()
        for fi in ffb_idx:
            for j in np.argsort(np.abs(norm_masses - d['StellarMass'][fi])):
                if j not in used:
                    matched_norm_idx.append(w_never_ffb[j])
                    used.add(j)
                    break
        if matched_norm_idx:
            norm_gal_ids = d['GalaxyIndex'][np.array(matched_norm_idx)]

    cosmic_times = {s: cosmic_time_gyr(REDSHIFTS[s]) for s in fig_g_snaps}

    ffb_tracks  = {int(gid): {'t': [], 'sfr': [], 'regime': [], 'snap': []}
                   for gid in ffb_gal_ids}
    norm_tracks = {int(gid): {'t': [], 'sfr': [], 'regime': [], 'snap': []}
                   for gid in norm_gal_ids}

    for s in fig_g_snaps:
        sd   = source_snapdata[s]
        gids = sd['GalaxyIndex']
        sfr_total = sd['SfrDisk'] + sd['SfrBulge']
        t = cosmic_times[s]

        for gid in ffb_gal_ids:
            match = np.where(gids == gid)[0]
            if len(match) > 0:
                m = match[0]
                ffb_tracks[int(gid)]['t'].append(t)
                ffb_tracks[int(gid)]['sfr'].append(sfr_total[m])
                ffb_tracks[int(gid)]['regime'].append(int(sd['FFBRegime'][m]))
                ffb_tracks[int(gid)]['snap'].append(s)

        for gid in norm_gal_ids:
            match = np.where(gids == gid)[0]
            if len(match) > 0:
                m = match[0]
                norm_tracks[int(gid)]['t'].append(t)
                norm_tracks[int(gid)]['sfr'].append(sfr_total[m])
                norm_tracks[int(gid)]['regime'].append(int(sd['FFBRegime'][m]))
                norm_tracks[int(gid)]['snap'].append(s)

    ffb_transition = {}
    for gid in ffb_gal_ids.astype(int):
        tr = ffb_tracks[gid]
        if not tr['t']:
            continue
        pairs = sorted(zip(tr['t'], tr['regime'], tr['snap']))
        t_vals, r_vals, s_vals = zip(*pairs)

        transition_idx = None
        for k in range(len(r_vals) - 1):
            if r_vals[k] == 1 and r_vals[k + 1] == 0:
                transition_idx = k + 1
                break

        if transition_idx is None:
            continue

        ffb_transition[gid] = (t_vals[transition_idx],
                               REDSHIFTS[s_vals[transition_idx]])

    return {
        'ffb_tracks':  ffb_tracks,
        'norm_tracks': norm_tracks,
        'plot_ids':    list(ffb_gal_ids.astype(int)) + list(norm_gal_ids.astype(int)),
        'ffb_ids':     list(ffb_gal_ids.astype(int)),
        'transition':  ffb_transition,
        'cosmic_times': cosmic_times,
    }


def plot_12f_sfh_ffb_transitions_stacked(snapdata):
    """
    plot_12d and plot_12e stacked into a single figure sharing one x-axis:
    the Li+24 FFB model on top, MBK25 below.

    Each panel is annotated with its model name in the bottom-right corner.
    Line style encodes the burst regime rather than the galaxy sample: solid
    where FFBRegime==1 ("With bursts"), dashed where it is 0 ("No bursts").
    The transition-marker legend entry is only drawn for panels that actually
    contain an FFB -> non-FFB crossing inside the plotted time range.
    """
    print('Plot 12f: stacked SFH transition figure (FFB + MBK25)')

    needed_snaps = list(range(8, 64))

    mbk25_snapdata = load_snapshots(FFB_BK25_SMOOTH_DIR, needed_snaps)
    if not mbk25_snapdata:
        print('  MBK25 smooth data not available; bottom panel will be empty.')

    panels = [
        {'data': snapdata,       'tag': 'FFB',   'color': 'k',
         'trans_label': 'FFB → non-FFB transition'},
        {'data': mbk25_snapdata, 'tag': 'MBK25', 'color': 'mediumpurple',
         'trans_label': 'MBK25 → non-MBK25 transition'},
    ]

    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    t_min_all = []
    t_max = 1.0

    for ax, cfg in zip(axes, panels):
        tracks = _sfh_transition_tracks(cfg['data'] or {}, needed_snaps)
        if tracks is None:
            print(f"  No usable galaxies for the {cfg['tag']} panel.")
            ax.text(0.5, 0.5, f"No {cfg['tag']} data", transform=ax.transAxes,
                    ha='center', va='center', color='0.5')
            continue

        t_min_all.append(min(tracks['cosmic_times'].values()))

        burst_label_done = quiet_label_done = False
        for gid in tracks['plot_ids']:
            tr = tracks['ffb_tracks'].get(gid, tracks['norm_tracks'].get(gid))
            if tr is None or len(tr['t']) <= 1:
                continue

            pairs = sorted(zip(tr['t'], tr['sfr'], tr['regime'], tr['snap']),
                           key=lambda x: x[0])

            for k in range(len(pairs) - 1):
                t0, sfr0, r0, _ = pairs[k]
                t1, sfr1, _, _  = pairs[k + 1]

                lbl = None
                if r0 == 1 and not burst_label_done:
                    lbl = 'With bursts'
                    burst_label_done = True
                elif r0 == 0 and not quiet_label_done:
                    lbl = 'No bursts'
                    quiet_label_done = True

                ax.plot([t0, t1], [sfr0, sfr1],
                        '-' if r0 == 1 else '--',
                        color=cfg['color'] if r0 == 1 else 'firebrick',
                        alpha=1.0, lw=2.2, label=lbl, zorder=2)

        # Transition markers, and the matching legend entry only if any land
        # inside the plotted range.
        drew_transition = False
        for gid in tracks['ffb_ids']:
            if gid not in tracks['transition']:
                continue
            t_trans, _ = tracks['transition'][gid]
            if not (min(tracks['cosmic_times'].values()) <= t_trans <= t_max):
                continue
            ax.axvline(t_trans, color='goldenrod', ls='--', lw=1.2,
                       alpha=0.85, zorder=4)
            drew_transition = True

        ax.set_ylabel(r'$\log_{10}\,\mathrm{SFR}\;[M_{\odot}\,\mathrm{yr}^{-1}]$')
        ax.set_yscale('log')
        ax.set_ylim(1e-3, 1e5)

        # Model name in the bottom-right corner of the panel.
        ax.text(0.98, 0.04, cfg['tag'], transform=ax.transAxes,
                ha='right', va='bottom', fontsize=20, zorder=10)

        import matplotlib.lines as mlines

        handles, labels = ax.get_legend_handles_labels()
        if drew_transition:
            handles = handles + [mlines.Line2D([], [], color='goldenrod',
                                               ls='--', lw=1.5)]
            labels  = labels + [cfg['trans_label']]
        if labels:
            _standard_legend(ax, loc='upper left',
                             handles=handles, labels=labels)

    t_min = min(t_min_all) if t_min_all else 0.0
    axes[0].set_xlim(t_min, t_max)
    axes[1].set_xlabel('Cosmic time [Gyr]')

    # The panels butt together, so the top panel's lowest decade label and the
    # bottom panel's highest would print on top of each other.  Keep every tick
    # mark but blank those two labels.  set_yticks widens the view to span the
    # list it is given, so re-assert the limits afterwards.
    _decades = list(range(-3, 6))
    _ticks   = [10.0 ** e for e in _decades]

    def _decade_labels(blank_exp):
        return ['' if e == blank_exp else rf'$10^{{{e}}}$' for e in _decades]

    for _ax, _blank in ((axes[0], -3), (axes[1], 5)):
        _ax.set_yticks(_ticks)
        _ax.set_yticklabels(_decade_labels(_blank))
        _ax.set_ylim(1e-3, 1e5)

    # Redshift axis on the top panel only.
    ax_top = axes[0].twiny()
    z_ticks = [10, 8, 6, 5, 4, 3, 2.5, 2, 1.5, 1]
    t_ticks = [cosmic_time_gyr(z) for z in z_ticks]
    xlim = axes[0].get_xlim()
    z_ticks_f = [z for z, t in zip(z_ticks, t_ticks) if xlim[0] <= t <= xlim[1]]
    t_ticks_f = [t for t in t_ticks if xlim[0] <= t <= xlim[1]]
    ax_top.set_xlim(xlim)
    ax_top.set_xticks(t_ticks_f)
    ax_top.set_xticklabels([str(z) for z in z_ticks_f])
    ax_top.set_xlabel('Redshift')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.0)
    save_figure(fig, os.path.join(OUTPUT_DIR,
                'SFH_FFB_transitions_stacked' + OUTPUT_FORMAT))


# ========================== PLOT 13: FFB FRACTION vs HALO MASS ==========================

def plot_13_ffb_vs_redshift(snapdata):
    """
    FFB fraction as a function of halo mass at different redshifts.

    Shows theoretical sigmoid curves (from the SAGE26 model) overlaid
    with binned simulation data with bootstrap error bars.
    """
    print('Plot 13: FFB fraction vs redshift')

    # Target redshifts (defined early so we can pre-load only the needed snaps)
    redshift_targets = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    needed_snaps = sorted({np.argmin(np.abs(np.array(REDSHIFTS) - z_t))
                           for z_t in redshift_targets})

    # Load MBK25 smooth simulation data (square markers)
    snapdata_bk25 = {}
    if os.path.exists(FFB_BK25_SMOOTH_DIR):
        print('  Loading MBK25 smooth snapshots...')
        snapdata_bk25 = load_snapshots(FFB_BK25_SMOOTH_DIR, needed_snaps)

    def _bin_ffb(d):
        """Bin FFB fraction vs halo mass. Returns (centres, fracs, ffb_errs, mass_errs)."""
        central = d['Type'] == 0
        Mvir_d = d['Mvir'][central]
        ffb_d = d['FFBRegime'][central].astype(float)
        pos = Mvir_d > 0
        lM = np.log10(Mvir_d[pos])
        ffb_d = ffb_d[pos]
        bin_edges = np.linspace(8, 14, 17)
        fracs, ferrs, merrs, centres = [], [], [], []
        for j in range(len(bin_edges) - 1):
            mask = (lM >= bin_edges[j]) & (lM < bin_edges[j + 1])
            n = np.sum(mask)
            if n < 10:
                continue
            vals = ffb_d[mask]
            frac = np.mean(vals)
            if frac == 0.0 or frac == 1.0:
                zs = 1.0
                denom = 1 + zs**2 / n
                centre_w = (frac + zs**2 / (2 * n)) / denom
                margin = zs * np.sqrt((frac * (1 - frac) + zs**2 / (4 * n)) / n) / denom
                el = max(0, frac - (centre_w - margin))
                eh = max(0, (centre_w + margin) - frac)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = stats.bootstrap((vals,), np.mean, n_resamples=1000,
                                          confidence_level=0.6827, method='percentile')
                el = max(0, frac - res.confidence_interval.low)
                eh = max(0, res.confidence_interval.high - frac)
            fracs.append(frac)
            ferrs.append([el, eh])
            masses = lM[mask]
            mm = np.mean(masses)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mres = stats.bootstrap((masses,), np.mean, n_resamples=1000,
                                        confidence_level=0.6827, method='percentile')
            merrs.append([max(0, mm - mres.confidence_interval.low),
                          max(0, mres.confidence_interval.high - mm)])
            centres.append(mm)
        return centres, fracs, ferrs, merrs

    fig, ax = plt.subplots()

    # Halo mass range (log10 M_sun)
    log_Mvir = np.linspace(8, 14, 500)
    Mvir = 10.0**log_Mvir

    cmap = plt.cm.plasma
    # Truncate colormap to avoid lightest yellow
    colors = [cmap(i / (len(redshift_targets) - 1) * 0.85)
              for i in range(len(redshift_targets))]

    delta_log_M = 0.15  # model smoothing width

    for z_target, color in zip(redshift_targets, colors):
        # Find closest snapshot
        snap_idx = np.argmin(np.abs(np.array(REDSHIFTS) - z_target))
        actual_z = REDSHIFTS[snap_idx]

        # Theoretical curve at actual snapshot redshift
        f_theory = ffb_fraction(Mvir, actual_z, delta_log_M)
        M_thresh = ffb_threshold_mass_msun(actual_z)
        ax.plot(log_Mvir, f_theory, color=color, lw=2,
                label=f'z = {actual_z:.2f}')
        ax.axvline(np.log10(M_thresh), color=color, ls=':', alpha=1.0, lw=1)

        # MBK25 theoretical curve (log-normal concentration scatter, sigma_c=0.2)
        f_mbk25 = ffb_fraction_mbk25(Mvir, actual_z, sigma_c=0.2)
        ax.plot(log_Mvir, f_mbk25, color=color, lw=2, ls='--', alpha=0.8)

        # Li+24 simulation data — circles
        if snap_idx in snapdata:
            centres, fracs, ferrs, merrs = _bin_ffb(snapdata[snap_idx])
            if len(centres) > 0:
                ax.errorbar(centres, fracs,
                            xerr=np.array(merrs).T, yerr=np.array(ferrs).T,
                            fmt='o', color=color, markersize=8, capsize=3,
                            alpha=0.6, markeredgecolor='k', markeredgewidth=0.3)

        # MBK25 smooth simulation data — squares
        if snap_idx in snapdata_bk25:
            centres, fracs, ferrs, merrs = _bin_ffb(snapdata_bk25[snap_idx])
            if len(centres) > 0:
                ax.errorbar(centres, fracs,
                            xerr=np.array(merrs).T, yerr=np.array(ferrs).T,
                            fmt='s', color=color, markersize=8, capsize=3,
                            alpha=0.6, markeredgecolor='k', markeredgewidth=0.3)

    ax.axhline(0.5, color='gray', ls='--', alpha=1.0, lw=1)

    ax.set_xlabel(r'$\log_{10}\ M_{\mathrm{vir}}\ [M_{\odot}]$')
    ax.set_ylabel(r'$f_{\mathrm{FFB}}$')
    ax.set_xlim(8.25, 13)
    ax.set_ylim(0, 1)

    # Proxy artists: line styles (theory) and marker shapes (simulation)
    from matplotlib.lines import Line2D
    kw_mk = dict(lw=0, markersize=8, markeredgecolor='k', markeredgewidth=0.3, alpha=0.6)
    # proxy_li24_line  = Line2D([0], [0], color='gray', lw=2, ls='-',
    #                           label='Li+24')
    # proxy_mbk25_line = Line2D([0], [0], color='gray', lw=2, ls='--',
    #                           label=r'MBK25 (theory, log-normal $c$)')
    proxy_li24_pts   = Line2D([0], [0], color='gray', marker='o',
                              label='Li+24', **kw_mk)
    proxy_mbk25_pts  = Line2D([0], [0], color='gray', marker='s',
                              label='MBK25', **kw_mk)
    style_handles = [
                     proxy_li24_pts, proxy_mbk25_pts]
    z_handles, z_labels = ax.get_legend_handles_labels()
    _standard_legend(ax, loc='upper left',
                     handles=style_handles + z_handles,
                     labels=[h.get_label() for h in style_handles] + z_labels)
    fig.tight_layout()

    save_figure(fig, os.path.join(OUTPUT_DIR,
                'FFBvsRedshift' + OUTPUT_FORMAT))


# ========================== PLOT 14: FFB MODEL COMPARISON ==========================

def plot_14_density_evolution():
    """
    Create 2x1 figure with SFRD and SMD vs redshift (stacked vertically).

    Top panel: SFRD vs redshift
    Bottom panel: SMD vs redshift

    Shows entire galaxy populations from FFB and no-FFB models, plus additional
    FFB models with different star formation efficiencies.
    """
    print('Plot 14: Density evolution (SFRD & SMD)')
    seed(SEED)

    # Output directory setup
    output_dir = OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create 2x1 figure (stacked vertically)
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    # Volume for density calculations
    # Note: VOLUME global is defined as (BOX_SIZE/h)^3 * Fraction
    volume = VOLUME

    # Define redshift range of interest (e.g. z=5 to z=16)
    # Filter global REDSHIFTS and get corresponding snapshot indices
    target_snaps = []
    for snap_idx, z in enumerate(REDSHIFTS):
        if 4.8 <= z <= 20.0:
            target_snaps.append(f'Snap_{snap_idx}')

    # Sort snapshots naturally if needed, though REDSHIFTS is ordered
    # Ensure we process in order

    print("Loading data for density evolution plots...")

    # Arrays to store density evolution data
    redshifts_density = []
    # Main values
    sfrd_ffb_list, sfrd_noffb_list, sfrd_ffb100_list, sfrd_ffb_bk25_list = [], [], [], []
    smd_ffb_list, smd_noffb_list, smd_ffb100_list, smd_ffb_bk25_list = [], [], [], []
    # Bootstrap errors (16th and 84th percentiles)
    sfrd_ffb_lo, sfrd_ffb_hi = [], []
    sfrd_noffb_lo, sfrd_noffb_hi = [], []
    sfrd_ffb100_lo, sfrd_ffb100_hi = [], []
    sfrd_ffb_bk25_lo, sfrd_ffb_bk25_hi = [], []
    smd_ffb_lo, smd_ffb_hi = [], []
    smd_noffb_lo, smd_noffb_hi = [], []
    smd_ffb100_lo, smd_ffb100_hi = [], []
    smd_ffb_bk25_lo, smd_ffb_bk25_hi = [], []

    N_BOOT = 100
    rng = np.random.default_rng(SEED)

    def bootstrap_density(values, n_boot=N_BOOT):
        """Bootstrap resampling for density (sum of values / volume)."""
        if len(values) == 0:
            return np.nan, np.nan, np.nan
        total = np.sum(values)
        n = len(values)
        boot_sums = np.array([np.sum(rng.choice(values, size=n, replace=True))
                              for _ in range(n_boot)])
        lo = np.log10(np.percentile(boot_sums, 16) / volume) if np.percentile(boot_sums, 16) > 0 else np.nan
        hi = np.log10(np.percentile(boot_sums, 84) / volume) if np.percentile(boot_sums, 84) > 0 else np.nan
        med = np.log10(total / volume) if total > 0 else np.nan
        return med, lo, hi

    for Snapshot in target_snaps:
        snapnum = int(Snapshot.split('_')[1])
        z = REDSHIFTS[snapnum]
        print(f'  Processing {Snapshot} (z = {z:.2f})')

        # Load data using existing load_model function
        props = ['StellarMass', 'SfrDisk', 'SfrBulge']

        # Load Primary (FFB default, sfe=0.2)
        data_FFB = load_model(PRIMARY_DIR,
                              snapshot=Snapshot, properties=props)

        # Load No FFB
        data_noFFB = load_model(NOFFB_DIR,
                                snapshot=Snapshot, properties=props)

        # Load FFB 100% (sfe=1.0)
        data_FFB100 = load_model(FFB100_DIR,
                                 snapshot=Snapshot, properties=props)

        # Load FFB BK25
        data_FFB_BK25 = load_model(FFB_BK25_SMOOTH_DIR,
                                   snapshot=Snapshot, properties=props)

        if not data_FFB and not data_noFFB and not data_FFB100 and not data_FFB_BK25:
            continue

        redshifts_density.append(z)

        # FFB (default)
        if data_FFB:
            sfr_vals = data_FFB['SfrDisk'] + data_FFB['SfrBulge']
            sm_vals = data_FFB['StellarMass']
            sfrd_med, sfrd_l, sfrd_h = bootstrap_density(sfr_vals)
            smd_med, smd_l, smd_h = bootstrap_density(sm_vals)
        else:
            sfrd_med, sfrd_l, sfrd_h = np.nan, np.nan, np.nan
            smd_med, smd_l, smd_h = np.nan, np.nan, np.nan
        sfrd_ffb_list.append(sfrd_med)
        sfrd_ffb_lo.append(sfrd_l)
        sfrd_ffb_hi.append(sfrd_h)
        smd_ffb_list.append(smd_med)
        smd_ffb_lo.append(smd_l)
        smd_ffb_hi.append(smd_h)

        # No FFB
        if data_noFFB:
            sfr_vals = data_noFFB['SfrDisk'] + data_noFFB['SfrBulge']
            sm_vals = data_noFFB['StellarMass']
            sfrd_med, sfrd_l, sfrd_h = bootstrap_density(sfr_vals)
            smd_med, smd_l, smd_h = bootstrap_density(sm_vals)
        else:
            sfrd_med, sfrd_l, sfrd_h = np.nan, np.nan, np.nan
            smd_med, smd_l, smd_h = np.nan, np.nan, np.nan
        sfrd_noffb_list.append(sfrd_med)
        sfrd_noffb_lo.append(sfrd_l)
        sfrd_noffb_hi.append(sfrd_h)
        smd_noffb_list.append(smd_med)
        smd_noffb_lo.append(smd_l)
        smd_noffb_hi.append(smd_h)

        # FFB 100%
        if data_FFB100:
            sfr_vals = data_FFB100['SfrDisk'] + data_FFB100['SfrBulge']
            sm_vals = data_FFB100['StellarMass']
            sfrd_med, sfrd_l, sfrd_h = bootstrap_density(sfr_vals)
            smd_med, smd_l, smd_h = bootstrap_density(sm_vals)
        else:
            sfrd_med, sfrd_l, sfrd_h = np.nan, np.nan, np.nan
            smd_med, smd_l, smd_h = np.nan, np.nan, np.nan
        sfrd_ffb100_list.append(sfrd_med)
        sfrd_ffb100_lo.append(sfrd_l)
        sfrd_ffb100_hi.append(sfrd_h)
        smd_ffb100_list.append(smd_med)
        smd_ffb100_lo.append(smd_l)
        smd_ffb100_hi.append(smd_h)

        # FFB BK25
        if data_FFB_BK25:
            sfr_vals = data_FFB_BK25['SfrDisk'] + data_FFB_BK25['SfrBulge']
            sm_vals = data_FFB_BK25['StellarMass']
            sfrd_med, sfrd_l, sfrd_h = bootstrap_density(sfr_vals)
            smd_med, smd_l, smd_h = bootstrap_density(sm_vals)
        else:
            sfrd_med, sfrd_l, sfrd_h = np.nan, np.nan, np.nan
            smd_med, smd_l, smd_h = np.nan, np.nan, np.nan
        sfrd_ffb_bk25_list.append(sfrd_med)
        sfrd_ffb_bk25_lo.append(sfrd_l)
        sfrd_ffb_bk25_hi.append(sfrd_h)
        smd_ffb_bk25_list.append(smd_med)
        smd_ffb_bk25_lo.append(smd_l)
        smd_ffb_bk25_hi.append(smd_h)

    # Convert to arrays and sort by redshift
    redshifts_density = np.array(redshifts_density)
    sort_idx = np.argsort(redshifts_density)
    z_sorted = redshifts_density[sort_idx]

    sfrd_ffb_sorted = np.array(sfrd_ffb_list)[sort_idx]
    sfrd_ffb_lo_sorted = np.array(sfrd_ffb_lo)[sort_idx]
    sfrd_ffb_hi_sorted = np.array(sfrd_ffb_hi)[sort_idx]
    sfrd_noffb_sorted = np.array(sfrd_noffb_list)[sort_idx]
    sfrd_noffb_lo_sorted = np.array(sfrd_noffb_lo)[sort_idx]
    sfrd_noffb_hi_sorted = np.array(sfrd_noffb_hi)[sort_idx]
    sfrd_ffb100_sorted = np.array(sfrd_ffb100_list)[sort_idx]
    sfrd_ffb100_lo_sorted = np.array(sfrd_ffb100_lo)[sort_idx]
    sfrd_ffb100_hi_sorted = np.array(sfrd_ffb100_hi)[sort_idx]
    sfrd_ffb_bk25_sorted = np.array(sfrd_ffb_bk25_list)[sort_idx]
    sfrd_ffb_bk25_lo_sorted = np.array(sfrd_ffb_bk25_lo)[sort_idx]
    sfrd_ffb_bk25_hi_sorted = np.array(sfrd_ffb_bk25_hi)[sort_idx]


    smd_ffb_sorted = np.array(smd_ffb_list)[sort_idx]
    smd_ffb_lo_sorted = np.array(smd_ffb_lo)[sort_idx]
    smd_ffb_hi_sorted = np.array(smd_ffb_hi)[sort_idx]
    smd_noffb_sorted = np.array(smd_noffb_list)[sort_idx]
    smd_noffb_lo_sorted = np.array(smd_noffb_lo)[sort_idx]
    smd_noffb_hi_sorted = np.array(smd_noffb_hi)[sort_idx]
    smd_ffb100_sorted = np.array(smd_ffb100_list)[sort_idx]
    smd_ffb100_lo_sorted = np.array(smd_ffb100_lo)[sort_idx]
    smd_ffb100_hi_sorted = np.array(smd_ffb100_hi)[sort_idx]
    smd_ffb_bk25_sorted = np.array(smd_ffb_bk25_list)[sort_idx]
    smd_ffb_bk25_lo_sorted = np.array(smd_ffb_bk25_lo)[sort_idx]
    smd_ffb_bk25_hi_sorted = np.array(smd_ffb_bk25_hi)[sort_idx]

    # ===== QUANTITATIVE COMPARISON: SAGE26 vs No FFB =====
    print("\n" + "="*60)
    print("QUANTITATIVE COMPARISON: SAGE26 vs No FFB")
    print("="*60)

    # Find common valid indices
    valid_both_sfrd = ~np.isnan(sfrd_ffb_sorted) & ~np.isnan(sfrd_noffb_sorted)
    valid_both_smd = ~np.isnan(smd_ffb_sorted) & ~np.isnan(smd_noffb_sorted)

    # --- SFRD Comparison ---
    sfrd_diff = sfrd_ffb_sorted[valid_both_sfrd] - sfrd_noffb_sorted[valid_both_sfrd]
    z_sfrd = z_sorted[valid_both_sfrd]

    print("\n--- COSMIC STAR FORMATION RATE DENSITY (SFRD) ---")
    print(f"  Mean difference (SAGE26 - No FFB):  {np.mean(sfrd_diff):+.3f} dex")
    print(f"  Median difference:                  {np.median(sfrd_diff):+.3f} dex")
    print(f"  Std of difference:                  {np.std(sfrd_diff):.3f} dex")
    print(f"  Max enhancement at z={z_sfrd[np.argmax(sfrd_diff)]:.1f}: {np.max(sfrd_diff):+.3f} dex ({10**np.max(sfrd_diff):.1f}x)")
    print(f"  Min enhancement at z={z_sfrd[np.argmin(sfrd_diff)]:.1f}: {np.min(sfrd_diff):+.3f} dex ({10**np.min(sfrd_diff):.1f}x)")

    print("\n  SFRD at specific redshifts:")
    for target_z in [6, 8, 10, 12, 14]:
        idx = np.argmin(np.abs(z_sorted - target_z))
        if valid_both_sfrd[idx]:
            diff = sfrd_ffb_sorted[idx] - sfrd_noffb_sorted[idx]
            print(f"    z~{z_sorted[idx]:.1f}: SAGE26={sfrd_ffb_sorted[idx]:.2f}, NoFFB={sfrd_noffb_sorted[idx]:.2f}, Δ={diff:+.2f} dex ({10**diff:.1f}x)")

    # --- SMD Comparison ---
    smd_diff = smd_ffb_sorted[valid_both_smd] - smd_noffb_sorted[valid_both_smd]
    z_smd = z_sorted[valid_both_smd]

    print("\n--- STELLAR MASS DENSITY (SMD) ---")
    print(f"  Mean difference (SAGE26 - No FFB):  {np.mean(smd_diff):+.3f} dex")
    print(f"  Median difference:                  {np.median(smd_diff):+.3f} dex")
    print(f"  Std of difference:                  {np.std(smd_diff):.3f} dex")
    print(f"  Max enhancement at z={z_smd[np.argmax(smd_diff)]:.1f}: {np.max(smd_diff):+.3f} dex ({10**np.max(smd_diff):.1f}x)")
    print(f"  Min enhancement at z={z_smd[np.argmin(smd_diff)]:.1f}: {np.min(smd_diff):+.3f} dex ({10**np.min(smd_diff):.1f}x)")

    print("\n  SMD at specific redshifts:")
    for target_z in [6, 8, 10, 12, 14]:
        idx = np.argmin(np.abs(z_sorted - target_z))
        if valid_both_smd[idx]:
            diff = smd_ffb_sorted[idx] - smd_noffb_sorted[idx]
            print(f"    z~{z_sorted[idx]:.1f}: SAGE26={smd_ffb_sorted[idx]:.2f}, NoFFB={smd_noffb_sorted[idx]:.2f}, Δ={diff:+.2f} dex ({10**diff:.1f}x)")

    print("="*60 + "\n")

    # --- Additional model comparisons: SAGE26 vs FFB100 and FFB_BK25 ---
    print("\n" + "="*60)
    print("QUANTITATIVE COMPARISONS: SAGE26 vs Other FFB Models")
    print("="*60)
    for _lbl, _sfrd_b, _smd_b in [
        ("FFB100 (alpha=1.0)", sfrd_ffb100_sorted, smd_ffb100_sorted),
        ("FFB BK25 (MBK25, alpha=0.2)", sfrd_ffb_bk25_sorted, smd_ffb_bk25_sorted),
    ]:
        _vs = ~np.isnan(sfrd_ffb_sorted) & ~np.isnan(_sfrd_b)
        _vm = ~np.isnan(smd_ffb_sorted) & ~np.isnan(_smd_b)
        print(f"\n  --- SAGE26 vs {_lbl} ---")
        if np.sum(_vs) > 0:
            _ds = sfrd_ffb_sorted[_vs] - _sfrd_b[_vs];  _zs = z_sorted[_vs]
            print(f"  CSFRD: Mean Δ={np.mean(_ds):+.3f} dex, Median={np.median(_ds):+.3f} dex, "
                  f"Max at z={_zs[np.argmax(_ds)]:.1f}: {np.max(_ds):+.3f} dex ({10**np.max(_ds):.1f}x)")
            print(f"         At specific z:")
            for _tz in [6, 8, 10, 12, 14]:
                _ti = np.argmin(np.abs(_zs - _tz))
                if np.abs(_zs[_ti] - _tz) < 1.0:
                    print(f"           z~{_zs[_ti]:.1f}: SAGE26={sfrd_ffb_sorted[_vs][_ti]:.2f}, "
                          f"{_lbl}={_sfrd_b[_vs][_ti]:.2f}, Δ={_ds[_ti]:+.2f} dex ({10**_ds[_ti]:.1f}x)")
        if np.sum(_vm) > 0:
            _dm = smd_ffb_sorted[_vm] - _smd_b[_vm];  _zm = z_sorted[_vm]
            print(f"  SMD:   Mean Δ={np.mean(_dm):+.3f} dex, Median={np.median(_dm):+.3f} dex, "
                  f"Max at z={_zm[np.argmax(_dm)]:.1f}: {np.max(_dm):+.3f} dex ({10**np.max(_dm):.1f}x)")
            print(f"         At specific z:")
            for _tz in [6, 8, 10, 12, 14]:
                _ti = np.argmin(np.abs(_zm - _tz))
                if np.abs(_zm[_ti] - _tz) < 1.0:
                    print(f"           z~{_zm[_ti]:.1f}: SAGE26={smd_ffb_sorted[_vm][_ti]:.2f}, "
                          f"{_lbl}={_smd_b[_vm][_ti]:.2f}, Δ={_dm[_ti]:+.2f} dex ({10**_dm[_ti]:.1f}x)")
    print("="*60 + "\n")

    # --- miniUchuu (if available) ---
    mu_z_list, mu_sfrd_list, mu_smd_list = [], [], []
    if model_files_exist(MINIUCHUU_DIR):
        mu_redshifts = np.array(MINIUCHUU_REDSHIFTS)
        mu_volume = MINIUCHUU_VOLUME
        for snap_idx in range(MINIUCHUU_FIRST_SNAP, MINIUCHUU_LAST_SNAP + 1):
            z = mu_redshifts[snap_idx]
            if not (4.8 <= z <= 20.0):
                continue
            snap_key = f'Snap_{snap_idx}'
            try:
                d = load_model(MINIUCHUU_DIR,
                               snapshot=snap_key,
                               properties=['StellarMass', 'SfrDisk', 'SfrBulge'])
                if not d:
                    continue
                mstar_mu = d['StellarMass'] / MASS_CONVERT * MINIUCHUU_MASS_CONVERT
                sfr_mu = d['SfrDisk'] + d['SfrBulge']
                tot_sfr = np.sum(sfr_mu)
                tot_sm = np.sum(mstar_mu)
                mu_z_list.append(z)
                mu_sfrd_list.append(np.log10(tot_sfr / mu_volume) if tot_sfr > 0 else np.nan)
                mu_smd_list.append(np.log10(tot_sm / mu_volume) if tot_sm > 0 else np.nan)
            except Exception:
                continue

    if len(mu_z_list) > 1:
        mu_z_arr = np.array(mu_z_list)
        mu_si = np.argsort(mu_z_arr)
        mu_z_sorted = mu_z_arr[mu_si]
        mu_sfrd_sorted = np.array(mu_sfrd_list)[mu_si]
        mu_smd_sorted = np.array(mu_smd_list)[mu_si]

    print("Generating density evolution plots...")

    # ----- Top Panel: SFRD vs Redshift -----
    valid_ffb = ~np.isnan(sfrd_ffb_sorted)
    valid_noffb = ~np.isnan(sfrd_noffb_sorted)
    valid_ffb100 = ~np.isnan(sfrd_ffb100_sorted)
    if np.sum(valid_noffb) > 1:
        axes[0].plot(z_sorted[valid_noffb], sfrd_noffb_sorted[valid_noffb], '-',
                    color='firebrick', linewidth=3.0, label='No FFB/MBK25 model')
        boot_valid = valid_noffb & ~np.isnan(sfrd_noffb_lo_sorted) & ~np.isnan(sfrd_noffb_hi_sorted)
        if np.sum(boot_valid) > 1:
            axes[0].fill_between(z_sorted[boot_valid], sfrd_noffb_lo_sorted[boot_valid],
                                sfrd_noffb_hi_sorted[boot_valid], color='firebrick', alpha=0.2)
    if np.sum(valid_ffb) > 1:
        axes[0].plot(z_sorted[valid_ffb], sfrd_ffb_sorted[valid_ffb], '-',
                    color='black', linewidth=3.5, label=r'$\alpha_{\rm FFB}=0.2$')
        boot_valid = valid_ffb & ~np.isnan(sfrd_ffb_lo_sorted) & ~np.isnan(sfrd_ffb_hi_sorted)
        if np.sum(boot_valid) > 1:
            axes[0].fill_between(z_sorted[boot_valid], sfrd_ffb_lo_sorted[boot_valid],
                                sfrd_ffb_hi_sorted[boot_valid], color='black', alpha=0.2)
    if np.sum(valid_ffb100) > 1:
        axes[0].plot(z_sorted[valid_ffb100], sfrd_ffb100_sorted[valid_ffb100], '-',
                    color='steelblue', linewidth=3.0, label=r'$\alpha_{\rm FFB}=1.0$')
        boot_valid = valid_ffb100 & ~np.isnan(sfrd_ffb100_lo_sorted) & ~np.isnan(sfrd_ffb100_hi_sorted)
        if np.sum(boot_valid) > 1:
            axes[0].fill_between(z_sorted[boot_valid], sfrd_ffb100_lo_sorted[boot_valid],
                                sfrd_ffb100_hi_sorted[boot_valid], color='steelblue', alpha=0.2)
    if len(mu_z_list) > 1:
        valid_mu = ~np.isnan(mu_sfrd_sorted)
        if np.sum(valid_mu) > 1:
            axes[0].plot(mu_z_sorted[valid_mu], mu_sfrd_sorted[valid_mu], '--',
                        color='steelblue', linewidth=2.5, label='miniUchuu')

    # Add SFRD observational data (only if loaded)
    z_madau, re_madau, re_err_plus_madau, re_err_minus_madau = load_madau_dickinson_2014_data()
    if z_madau is not None:
        mask = (z_madau >= 5) & (z_madau <= 16)
        if np.sum(mask) > 0:
            axes[0].errorbar(z_madau[mask], re_madau[mask],
                            yerr=[re_err_minus_madau[mask], re_err_plus_madau[mask]],
                            fmt='o', color='black', markersize=8, alpha=0.6,
                            markeredgecolor='k', markeredgewidth=0.8,
                            markerfacecolor = 'gray',
                            label=_tex_safe(r'Madau \& Dickinson 14'), linewidth=1.0, zorder=5)

    z_oesch, re_oesch, re_err_plus_oesch, re_err_minus_oesch = load_oesch_sfrd_2018_data()
    if z_oesch is not None:
        mask = (z_oesch >= 5) & (z_oesch <= 16)
        if np.sum(mask) > 0:
            axes[0].errorbar(z_oesch[mask], re_oesch[mask],
                            yerr=[re_err_minus_oesch[mask], re_err_plus_oesch[mask]],
                            fmt='*', color='black', markersize=8, alpha=0.6,
                            markeredgecolor='k', markeredgewidth=0.8,
                            markerfacecolor = 'gray',
                            label='Oesch+18', linewidth=1.0, zorder=5)

    z_mcleod, re_mcleod, re_err_plus_mcleod, re_err_minus_mcleod = load_mcleod_rho_sfr_2024_data()
    if z_mcleod is not None:
        mask = (z_mcleod >= 5) & (z_mcleod <= 16)
        if np.sum(mask) > 0:
            axes[0].errorbar(z_mcleod[mask], re_mcleod[mask],
                            yerr=[re_err_minus_mcleod[mask], re_err_plus_mcleod[mask]],
                            fmt='v', color='black', markersize=8, alpha=0.6,
                            markeredgecolor='k', markeredgewidth=0.8,
                            markerfacecolor = 'gray',
                            label='McLeod+24', linewidth=1.0, zorder=5)

    z_harikane, re_harikane, re_err_plus_harikane, re_err_minus_harikane = load_harikane_sfr_density_2023_data()
    if z_harikane is not None:
        mask = (z_harikane >= 5) & (z_harikane <= 16)
        if np.sum(mask) > 0:
            axes[0].errorbar(z_harikane[mask], re_harikane[mask],
                            yerr=[re_err_minus_harikane[mask], re_err_plus_harikane[mask]],
                            fmt='D', color='black', markersize=8, alpha=0.6,
                            markeredgecolor='k', markeredgewidth=0.8,
                            markerfacecolor = 'gray',
                            label='Harikane+23', linewidth=1.0, zorder=5)

    # ----- Bottom Panel: SMD vs Redshift -----
    valid_smd_noffb = ~np.isnan(smd_noffb_sorted)
    valid_smd_ffb = ~np.isnan(smd_ffb_sorted)
    valid_smd_ffb100 = ~np.isnan(smd_ffb100_sorted)
    if np.sum(valid_smd_noffb) > 1:
        axes[1].plot(z_sorted[valid_smd_noffb], smd_noffb_sorted[valid_smd_noffb], '-',
                    color='firebrick', linewidth=3.0, label='No FFB/MBK25 model')
        boot_valid = valid_smd_noffb & ~np.isnan(smd_noffb_lo_sorted) & ~np.isnan(smd_noffb_hi_sorted)
        if np.sum(boot_valid) > 1:
            axes[1].fill_between(z_sorted[boot_valid], smd_noffb_lo_sorted[boot_valid],
                                smd_noffb_hi_sorted[boot_valid], color='firebrick', alpha=0.2)
    if np.sum(valid_smd_ffb) > 1:
        axes[1].plot(z_sorted[valid_smd_ffb], smd_ffb_sorted[valid_smd_ffb], '-',
                    color='black', linewidth=3.5, label=r'$\alpha_{\rm FFB}=0.2$')
        boot_valid = valid_smd_ffb & ~np.isnan(smd_ffb_lo_sorted) & ~np.isnan(smd_ffb_hi_sorted)
        if np.sum(boot_valid) > 1:
            axes[1].fill_between(z_sorted[boot_valid], smd_ffb_lo_sorted[boot_valid],
                                smd_ffb_hi_sorted[boot_valid], color='black', alpha=0.2)
    if np.sum(valid_smd_ffb100) > 1:
        axes[1].plot(z_sorted[valid_smd_ffb100], smd_ffb100_sorted[valid_smd_ffb100], '-',
                    color='steelblue', linewidth=3.0, label=r'$\alpha_{\rm FFB}=1.0$')
        boot_valid = valid_smd_ffb100 & ~np.isnan(smd_ffb100_lo_sorted) & ~np.isnan(smd_ffb100_hi_sorted)
        if np.sum(boot_valid) > 1:
            axes[1].fill_between(z_sorted[boot_valid], smd_ffb100_lo_sorted[boot_valid],
                                smd_ffb100_hi_sorted[boot_valid], color='steelblue', alpha=0.2)
    if len(mu_z_list) > 1:
        valid_mu_smd = ~np.isnan(mu_smd_sorted)
        if np.sum(valid_mu_smd) > 1:
            axes[1].plot(mu_z_sorted[valid_mu_smd], mu_smd_sorted[valid_mu_smd], '--',
                        color='steelblue', linewidth=2.5, label='SAGE26 (miniUchuu)')

    # Add SMD observational data (only if loaded)
    z_madau_smd, re_madau_smd, re_err_plus_madau_smd, re_err_minus_madau_smd = load_madau_dickinson_smd_2014_data()
    if z_madau_smd is not None:
        mask = (z_madau_smd >= 5) & (z_madau_smd <= 16)
        if np.sum(mask) > 0:
            axes[1].errorbar(z_madau_smd[mask], re_madau_smd[mask],
                            yerr=[re_err_minus_madau_smd[mask], re_err_plus_madau_smd[mask]],
                            fmt='o', color='black', markersize=8, alpha=0.6,
                            markeredgecolor='k', markeredgewidth=0.8,
                            markerfacecolor = 'gray',
                            label=_tex_safe(r'Madau \& Dickinson 14'), linewidth=1.0, zorder=5)

    z_kiku, re_kiku, re_err_plus_kiku, re_err_minus_kiku = load_kikuchihara_smd_2020_data()
    if z_kiku is not None:
        mask = (z_kiku >= 5) & (z_kiku <= 16)
        if np.sum(mask) > 0:
            axes[1].errorbar(z_kiku[mask], re_kiku[mask],
                            yerr=[re_err_minus_kiku[mask], re_err_plus_kiku[mask]],
                            fmt='d', color='black', markersize=8, alpha=0.6,
                            markeredgecolor='k', markeredgewidth=0.8,
                            markerfacecolor = 'gray',
                            label='Kikuchihara+20', linewidth=1.0, zorder=5)

    z_papovich, re_papovich, re_err_plus_papovich, re_err_minus_papovich = load_papovich_smd_2023_data()
    if z_papovich is not None:
        mask = (z_papovich >= 5) & (z_papovich <= 16)
        if np.sum(mask) > 0:
            axes[1].errorbar(z_papovich[mask], re_papovich[mask],
                            yerr=[re_err_minus_papovich[mask], re_err_plus_papovich[mask]],
                            fmt='s', color='black', markersize=8, alpha=0.6,
                            markeredgecolor='k', markeredgewidth=0.8,
                            markerfacecolor = 'gray',
                            label='Papovich+23', linewidth=1.0, zorder=5)

    # Configure axes and split legends
    def is_sim_label(l):
        return 'SAGE26' in l or 'FFB' in l or 'miniUchuu' in l or 'epsilon' in l
    for panel in axes:
        handles, labels = panel.get_legend_handles_labels()
        sim_h = [h for h, l in zip(handles, labels) if is_sim_label(l)]
        sim_l = [l for l in labels if is_sim_label(l)]
        obs_h = [h for h, l in zip(handles, labels) if not is_sim_label(l)]
        obs_l = [l for l in labels if not is_sim_label(l)]
        leg1 = _standard_legend(panel, loc='upper right', handles=sim_h, labels=sim_l)
        panel.add_artist(leg1)
        _standard_legend(panel, loc='lower left', handles=obs_h, labels=obs_l)

    # Top panel: SFRD
    axes[0].set_ylabel(r'$\log_{10} \rho_{\mathrm{SFR}}\ (M_\odot\,\mathrm{yr}^{-1}\,\mathrm{Mpc}^{-3})$')
    axes[0].set_xlim(5, 16)
    axes[0].set_ylim(-5, -1)
    axes[0].xaxis.set_major_locator(plt.MultipleLocator(2.0))
    axes[0].yaxis.set_major_locator(plt.MultipleLocator(1.0))
    axes[0].xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    axes[0].yaxis.set_minor_locator(plt.MultipleLocator(0.2))

    # Bottom panel: SMD
    axes[1].set_xlabel(r'Redshift')
    axes[1].set_ylabel(r'$\log_{10} \rho_\star\ [M_\odot\,\mathrm{Mpc}^{-3}]$')
    axes[1].set_xlim(5, 16)
    axes[1].set_ylim(3, 8)
    axes[1].xaxis.set_major_locator(plt.MultipleLocator(2.0))
    axes[1].yaxis.set_major_locator(plt.MultipleLocator(1.0))
    axes[1].xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    axes[1].yaxis.set_minor_locator(plt.MultipleLocator(0.2))


    fig.tight_layout()

    output_file = os.path.join(output_dir, 'FFB_Density_Evolution' + OUTPUT_FORMAT)
    save_figure(fig, output_file)

# ========================== PLOT 14c: DENSITY EVOLUTION WITH MBK25 ==========================

def plot_14c_density_evolution_mbk25():
    """
    Create 2x1 figure with SFRD and SMD vs redshift (stacked vertically).
    Same as plot_14 but with an additional MBK25 (smooth) green line and
    updated legend labels with (Li+24)/(MBK25) suffixes.
    """
    print('Plot 14c: Density evolution (SFRD & SMD) with MBK25')
    seed(SEED)

    output_dir = OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    volume = VOLUME

    target_snaps = []
    for snap_idx, z in enumerate(REDSHIFTS):
        if 4.8 <= z <= 20.0:
            target_snaps.append(f'Snap_{snap_idx}')

    print("Loading data for density evolution plots...")

    redshifts_density = []
    sfrd_ffb_list, sfrd_noffb_list, sfrd_ffb100_list, sfrd_ffb_bk25_list = [], [], [], []
    smd_ffb_list, smd_noffb_list, smd_ffb100_list, smd_ffb_bk25_list = [], [], [], []
    sfrd_ffb_lo, sfrd_ffb_hi = [], []
    sfrd_noffb_lo, sfrd_noffb_hi = [], []
    sfrd_ffb100_lo, sfrd_ffb100_hi = [], []
    sfrd_ffb_bk25_lo, sfrd_ffb_bk25_hi = [], []
    smd_ffb_lo, smd_ffb_hi = [], []
    smd_noffb_lo, smd_noffb_hi = [], []
    smd_ffb100_lo, smd_ffb100_hi = [], []
    smd_ffb_bk25_lo, smd_ffb_bk25_hi = [], []
    sfrd_ffb_bk25_ffb100_list = []
    smd_ffb_bk25_ffb100_list = []
    sfrd_ffb_bk25_ffb100_lo, sfrd_ffb_bk25_ffb100_hi = [], []
    smd_ffb_bk25_ffb100_lo, smd_ffb_bk25_ffb100_hi = [], []

    N_BOOT = 100
    rng = np.random.default_rng(SEED)

    def bootstrap_density(values, n_boot=N_BOOT):
        """Bootstrap resampling for density (sum of values / volume)."""
        if len(values) == 0:
            return np.nan, np.nan, np.nan
        total = np.sum(values)
        n = len(values)
        boot_sums = np.array([np.sum(rng.choice(values, size=n, replace=True))
                              for _ in range(n_boot)])
        lo = np.log10(np.percentile(boot_sums, 16) / volume) if np.percentile(boot_sums, 16) > 0 else np.nan
        hi = np.log10(np.percentile(boot_sums, 84) / volume) if np.percentile(boot_sums, 84) > 0 else np.nan
        med = np.log10(total / volume) if total > 0 else np.nan
        return med, lo, hi

    for Snapshot in target_snaps:
        snapnum = int(Snapshot.split('_')[1])
        z = REDSHIFTS[snapnum]
        print(f'  Processing {Snapshot} (z = {z:.2f})')

        props = ['StellarMass', 'SfrDisk', 'SfrBulge']

        data_FFB = load_model(PRIMARY_DIR,
                              snapshot=Snapshot, properties=props)
        data_noFFB = load_model(NOFFB_DIR,
                                snapshot=Snapshot, properties=props)
        data_FFB100 = load_model(FFB100_DIR,
                                 snapshot=Snapshot, properties=props)
        data_FFB_BK25 = load_model(FFB_BK25_SMOOTH_DIR,
                                   snapshot=Snapshot, properties=props)
        data_FFB_BK25_FFB100 = load_model(FFB_BK25_FFB100_DIR,
                                          snapshot=Snapshot, properties=props)

        if not data_FFB and not data_noFFB and not data_FFB100 and not data_FFB_BK25:
            continue

        redshifts_density.append(z)

        if data_FFB:
            sfr_vals = data_FFB['SfrDisk'] + data_FFB['SfrBulge']
            sm_vals = data_FFB['StellarMass']
            sfrd_med, sfrd_l, sfrd_h = bootstrap_density(sfr_vals)
            smd_med, smd_l, smd_h = bootstrap_density(sm_vals)
        else:
            sfrd_med, sfrd_l, sfrd_h = np.nan, np.nan, np.nan
            smd_med, smd_l, smd_h = np.nan, np.nan, np.nan
        sfrd_ffb_list.append(sfrd_med)
        sfrd_ffb_lo.append(sfrd_l)
        sfrd_ffb_hi.append(sfrd_h)
        smd_ffb_list.append(smd_med)
        smd_ffb_lo.append(smd_l)
        smd_ffb_hi.append(smd_h)

        if data_noFFB:
            sfr_vals = data_noFFB['SfrDisk'] + data_noFFB['SfrBulge']
            sm_vals = data_noFFB['StellarMass']
            sfrd_med, sfrd_l, sfrd_h = bootstrap_density(sfr_vals)
            smd_med, smd_l, smd_h = bootstrap_density(sm_vals)
        else:
            sfrd_med, sfrd_l, sfrd_h = np.nan, np.nan, np.nan
            smd_med, smd_l, smd_h = np.nan, np.nan, np.nan
        sfrd_noffb_list.append(sfrd_med)
        sfrd_noffb_lo.append(sfrd_l)
        sfrd_noffb_hi.append(sfrd_h)
        smd_noffb_list.append(smd_med)
        smd_noffb_lo.append(smd_l)
        smd_noffb_hi.append(smd_h)

        if data_FFB100:
            sfr_vals = data_FFB100['SfrDisk'] + data_FFB100['SfrBulge']
            sm_vals = data_FFB100['StellarMass']
            sfrd_med, sfrd_l, sfrd_h = bootstrap_density(sfr_vals)
            smd_med, smd_l, smd_h = bootstrap_density(sm_vals)
        else:
            sfrd_med, sfrd_l, sfrd_h = np.nan, np.nan, np.nan
            smd_med, smd_l, smd_h = np.nan, np.nan, np.nan
        sfrd_ffb100_list.append(sfrd_med)
        sfrd_ffb100_lo.append(sfrd_l)
        sfrd_ffb100_hi.append(sfrd_h)
        smd_ffb100_list.append(smd_med)
        smd_ffb100_lo.append(smd_l)
        smd_ffb100_hi.append(smd_h)

        if data_FFB_BK25:
            sfr_vals = data_FFB_BK25['SfrDisk'] + data_FFB_BK25['SfrBulge']
            sm_vals = data_FFB_BK25['StellarMass']
            sfrd_med, sfrd_l, sfrd_h = bootstrap_density(sfr_vals)
            smd_med, smd_l, smd_h = bootstrap_density(sm_vals)
        else:
            sfrd_med, sfrd_l, sfrd_h = np.nan, np.nan, np.nan
            smd_med, smd_l, smd_h = np.nan, np.nan, np.nan
        sfrd_ffb_bk25_list.append(sfrd_med)
        sfrd_ffb_bk25_lo.append(sfrd_l)
        sfrd_ffb_bk25_hi.append(sfrd_h)
        smd_ffb_bk25_list.append(smd_med)
        smd_ffb_bk25_lo.append(smd_l)
        smd_ffb_bk25_hi.append(smd_h)

        if data_FFB_BK25_FFB100:
            sfr_vals = data_FFB_BK25_FFB100['SfrDisk'] + data_FFB_BK25_FFB100['SfrBulge']
            sm_vals = data_FFB_BK25_FFB100['StellarMass']
            sfrd_med, sfrd_l, sfrd_h = bootstrap_density(sfr_vals)
            smd_med, smd_l, smd_h = bootstrap_density(sm_vals)
        else:
            sfrd_med, sfrd_l, sfrd_h = np.nan, np.nan, np.nan
            smd_med, smd_l, smd_h = np.nan, np.nan, np.nan
        sfrd_ffb_bk25_ffb100_list.append(sfrd_med)
        sfrd_ffb_bk25_ffb100_lo.append(sfrd_l)
        sfrd_ffb_bk25_ffb100_hi.append(sfrd_h)
        smd_ffb_bk25_ffb100_list.append(smd_med)
        smd_ffb_bk25_ffb100_lo.append(smd_l)
        smd_ffb_bk25_ffb100_hi.append(smd_h)

    redshifts_density = np.array(redshifts_density)
    sort_idx = np.argsort(redshifts_density)
    z_sorted = redshifts_density[sort_idx]

    sfrd_ffb_sorted = np.array(sfrd_ffb_list)[sort_idx]
    sfrd_ffb_lo_sorted = np.array(sfrd_ffb_lo)[sort_idx]
    sfrd_ffb_hi_sorted = np.array(sfrd_ffb_hi)[sort_idx]
    sfrd_noffb_sorted = np.array(sfrd_noffb_list)[sort_idx]
    sfrd_noffb_lo_sorted = np.array(sfrd_noffb_lo)[sort_idx]
    sfrd_noffb_hi_sorted = np.array(sfrd_noffb_hi)[sort_idx]
    sfrd_ffb100_sorted = np.array(sfrd_ffb100_list)[sort_idx]
    sfrd_ffb100_lo_sorted = np.array(sfrd_ffb100_lo)[sort_idx]
    sfrd_ffb100_hi_sorted = np.array(sfrd_ffb100_hi)[sort_idx]
    sfrd_ffb_bk25_sorted = np.array(sfrd_ffb_bk25_list)[sort_idx]
    sfrd_ffb_bk25_lo_sorted = np.array(sfrd_ffb_bk25_lo)[sort_idx]
    sfrd_ffb_bk25_hi_sorted = np.array(sfrd_ffb_bk25_hi)[sort_idx]

    smd_ffb_sorted = np.array(smd_ffb_list)[sort_idx]
    smd_ffb_lo_sorted = np.array(smd_ffb_lo)[sort_idx]
    smd_ffb_hi_sorted = np.array(smd_ffb_hi)[sort_idx]
    smd_noffb_sorted = np.array(smd_noffb_list)[sort_idx]
    smd_noffb_lo_sorted = np.array(smd_noffb_lo)[sort_idx]
    smd_noffb_hi_sorted = np.array(smd_noffb_hi)[sort_idx]
    smd_ffb100_sorted = np.array(smd_ffb100_list)[sort_idx]
    smd_ffb100_lo_sorted = np.array(smd_ffb100_lo)[sort_idx]
    smd_ffb100_hi_sorted = np.array(smd_ffb100_hi)[sort_idx]
    smd_ffb_bk25_sorted = np.array(smd_ffb_bk25_list)[sort_idx]
    smd_ffb_bk25_lo_sorted = np.array(smd_ffb_bk25_lo)[sort_idx]
    smd_ffb_bk25_hi_sorted = np.array(smd_ffb_bk25_hi)[sort_idx]
    sfrd_ffb_bk25_ffb100_sorted = np.array(sfrd_ffb_bk25_ffb100_list)[sort_idx]
    sfrd_ffb_bk25_ffb100_lo_sorted = np.array(sfrd_ffb_bk25_ffb100_lo)[sort_idx]
    sfrd_ffb_bk25_ffb100_hi_sorted = np.array(sfrd_ffb_bk25_ffb100_hi)[sort_idx]
    smd_ffb_bk25_ffb100_sorted = np.array(smd_ffb_bk25_ffb100_list)[sort_idx]
    smd_ffb_bk25_ffb100_lo_sorted = np.array(smd_ffb_bk25_ffb100_lo)[sort_idx]
    smd_ffb_bk25_ffb100_hi_sorted = np.array(smd_ffb_bk25_ffb100_hi)[sort_idx]

    # ===== QUANTITATIVE COMPARISONS: all model pairs =====
    print("\n" + "="*60)
    print("QUANTITATIVE COMPARISONS: Model Differences (CSFRD & SMD)")
    print("="*60)
    _cmp14c = [
        ("Li+24 (a=0.2) vs No FFB",          sfrd_ffb_sorted,         sfrd_noffb_sorted,         smd_ffb_sorted,         smd_noffb_sorted),
        ("Li+24 (a=0.2) vs Li+24 (a=1.0)",   sfrd_ffb_sorted,         sfrd_ffb100_sorted,        smd_ffb_sorted,         smd_ffb100_sorted),
        ("Li+24 (a=0.2) vs MBK25 (a=0.2)",   sfrd_ffb_sorted,         sfrd_ffb_bk25_sorted,      smd_ffb_sorted,         smd_ffb_bk25_sorted),
        ("Li+24 (a=0.2) vs MBK25 (a=1.0)",   sfrd_ffb_sorted,         sfrd_ffb_bk25_ffb100_sorted, smd_ffb_sorted,       smd_ffb_bk25_ffb100_sorted),
        ("MBK25 (a=0.2) vs No FFB",           sfrd_ffb_bk25_sorted,   sfrd_noffb_sorted,         smd_ffb_bk25_sorted,   smd_noffb_sorted),
        ("MBK25 (a=0.2) vs MBK25 (a=1.0)",   sfrd_ffb_bk25_sorted,   sfrd_ffb_bk25_ffb100_sorted, smd_ffb_bk25_sorted, smd_ffb_bk25_ffb100_sorted),
    ]
    for _lbl, _sa, _sb, _ma, _mb in _cmp14c:
        _vs = ~np.isnan(_sa) & ~np.isnan(_sb)
        _vm = ~np.isnan(_ma) & ~np.isnan(_mb)
        print(f"\n  --- {_lbl} ---")
        if np.sum(_vs) > 0:
            _ds = _sa[_vs] - _sb[_vs];  _zs = z_sorted[_vs]
            print(f"  CSFRD: Mean Δ={np.mean(_ds):+.3f} dex, Median={np.median(_ds):+.3f} dex, "
                  f"Max at z={_zs[np.argmax(_ds)]:.1f}: {np.max(_ds):+.3f} dex ({10**np.max(_ds):.1f}x)")
            print(f"         At specific z:")
            for _tz in [6, 8, 10, 12, 14]:
                _ti = np.argmin(np.abs(_zs - _tz))
                if np.abs(_zs[_ti] - _tz) < 1.0:
                    print(f"           z~{_zs[_ti]:.1f}: A={_sa[_vs][_ti]:.2f}, B={_sb[_vs][_ti]:.2f}, Δ={_ds[_ti]:+.2f} dex ({10**_ds[_ti]:.1f}x)")
        if np.sum(_vm) > 0:
            _dm = _ma[_vm] - _mb[_vm];  _zm = z_sorted[_vm]
            print(f"  SMD:   Mean Δ={np.mean(_dm):+.3f} dex, Median={np.median(_dm):+.3f} dex, "
                  f"Max at z={_zm[np.argmax(_dm)]:.1f}: {np.max(_dm):+.3f} dex ({10**np.max(_dm):.1f}x)")
            print(f"         At specific z:")
            for _tz in [6, 8, 10, 12, 14]:
                _ti = np.argmin(np.abs(_zm - _tz))
                if np.abs(_zm[_ti] - _tz) < 1.0:
                    print(f"           z~{_zm[_ti]:.1f}: A={_ma[_vm][_ti]:.2f}, B={_mb[_vm][_ti]:.2f}, Δ={_dm[_ti]:+.2f} dex ({10**_dm[_ti]:.1f}x)")
    print("="*60 + "\n")

    print("Generating density evolution plots...")

    # ----- Top Panel: SFRD vs Redshift -----
    valid_ffb = ~np.isnan(sfrd_ffb_sorted)
    valid_noffb = ~np.isnan(sfrd_noffb_sorted)
    valid_ffb100 = ~np.isnan(sfrd_ffb100_sorted)
    valid_ffb_bk25 = ~np.isnan(sfrd_ffb_bk25_sorted)
    if np.sum(valid_noffb) > 1:
        axes[0].plot(z_sorted[valid_noffb], sfrd_noffb_sorted[valid_noffb], '-',
                    color='firebrick', linewidth=3.0, label='No FFB/MBK25 model')
        boot_valid = valid_noffb & ~np.isnan(sfrd_noffb_lo_sorted) & ~np.isnan(sfrd_noffb_hi_sorted)
        if np.sum(boot_valid) > 1:
            axes[0].fill_between(z_sorted[boot_valid], sfrd_noffb_lo_sorted[boot_valid],
                                sfrd_noffb_hi_sorted[boot_valid], color='firebrick', alpha=0.2)
    if np.sum(valid_ffb) > 1:
        axes[0].plot(z_sorted[valid_ffb], sfrd_ffb_sorted[valid_ffb], '-',
                    color='black', linewidth=3.5, label=r'$\alpha_{\rm FFB}=0.2$ (Li+24)')
        boot_valid = valid_ffb & ~np.isnan(sfrd_ffb_lo_sorted) & ~np.isnan(sfrd_ffb_hi_sorted)
        if np.sum(boot_valid) > 1:
            axes[0].fill_between(z_sorted[boot_valid], sfrd_ffb_lo_sorted[boot_valid],
                                sfrd_ffb_hi_sorted[boot_valid], color='black', alpha=0.2)
    if np.sum(valid_ffb_bk25) > 1:
        axes[0].plot(z_sorted[valid_ffb_bk25], sfrd_ffb_bk25_sorted[valid_ffb_bk25], '-',
                    color='mediumpurple', linewidth=3.0, label=r'$\alpha_{\rm FFB}=0.2$ (MBK25)')
        boot_valid = valid_ffb_bk25 & ~np.isnan(sfrd_ffb_bk25_lo_sorted) & ~np.isnan(sfrd_ffb_bk25_hi_sorted)
        if np.sum(boot_valid) > 1:
            axes[0].fill_between(z_sorted[boot_valid], sfrd_ffb_bk25_lo_sorted[boot_valid],
                                sfrd_ffb_bk25_hi_sorted[boot_valid], color='mediumpurple', alpha=0.2)
    if np.sum(valid_ffb100) > 1:
        axes[0].plot(z_sorted[valid_ffb100], sfrd_ffb100_sorted[valid_ffb100], '--',
                    color='steelblue', linewidth=3.0, label=r'$\alpha_{\rm FFB}=1.0$ (Li+24)')
        boot_valid = valid_ffb100 & ~np.isnan(sfrd_ffb100_lo_sorted) & ~np.isnan(sfrd_ffb100_hi_sorted)
        if np.sum(boot_valid) > 1:
            axes[0].fill_between(z_sorted[boot_valid], sfrd_ffb100_lo_sorted[boot_valid],
                                sfrd_ffb100_hi_sorted[boot_valid], color='steelblue', alpha=0.2)
    valid_ffb_bk25_ffb100 = ~np.isnan(sfrd_ffb_bk25_ffb100_sorted)
    if np.sum(valid_ffb_bk25_ffb100) > 1:
        axes[0].plot(z_sorted[valid_ffb_bk25_ffb100], sfrd_ffb_bk25_ffb100_sorted[valid_ffb_bk25_ffb100], '--',
                    color='magenta', linewidth=3.0, label=r'$\alpha_{\rm FFB}=1.0$ (MBK25)')
        boot_valid = valid_ffb_bk25_ffb100 & ~np.isnan(sfrd_ffb_bk25_ffb100_lo_sorted) & ~np.isnan(sfrd_ffb_bk25_ffb100_hi_sorted)
        if np.sum(boot_valid) > 1:
            axes[0].fill_between(z_sorted[boot_valid], sfrd_ffb_bk25_ffb100_lo_sorted[boot_valid],
                                sfrd_ffb_bk25_ffb100_hi_sorted[boot_valid], color='magenta', alpha=0.2)

    # Add SFRD observational data
    z_madau, re_madau, re_err_plus_madau, re_err_minus_madau = load_madau_dickinson_2014_data()
    if z_madau is not None:
        mask = (z_madau >= 5) & (z_madau <= 16)
        if np.sum(mask) > 0:
            axes[0].errorbar(z_madau[mask], re_madau[mask],
                            yerr=[re_err_minus_madau[mask], re_err_plus_madau[mask]],
                            fmt='o', color='black', markersize=8, alpha=0.6,
                            markeredgecolor='k', markeredgewidth=0.8,
                            markerfacecolor = 'gray',
                            label=_tex_safe(r'Madau \& Dickinson 14'), linewidth=1.0, zorder=5)

    z_oesch, re_oesch, re_err_plus_oesch, re_err_minus_oesch = load_oesch_sfrd_2018_data()
    if z_oesch is not None:
        mask = (z_oesch >= 5) & (z_oesch <= 16)
        if np.sum(mask) > 0:
            axes[0].errorbar(z_oesch[mask], re_oesch[mask],
                            yerr=[re_err_minus_oesch[mask], re_err_plus_oesch[mask]],
                            fmt='*', color='black', markersize=8, alpha=0.6,
                            markeredgecolor='k', markeredgewidth=0.8,
                            markerfacecolor = 'gray',
                            label='Oesch+18', linewidth=1.0, zorder=5)

    z_mcleod, re_mcleod, re_err_plus_mcleod, re_err_minus_mcleod = load_mcleod_rho_sfr_2024_data()
    if z_mcleod is not None:
        mask = (z_mcleod >= 5) & (z_mcleod <= 16)
        if np.sum(mask) > 0:
            axes[0].errorbar(z_mcleod[mask], re_mcleod[mask],
                            yerr=[re_err_minus_mcleod[mask], re_err_plus_mcleod[mask]],
                            fmt='v', color='black', markersize=8, alpha=0.6,
                            markeredgecolor='k', markeredgewidth=0.8,
                            markerfacecolor = 'gray',
                            label='McLeod+24', linewidth=1.0, zorder=5)

    z_harikane, re_harikane, re_err_plus_harikane, re_err_minus_harikane = load_harikane_sfr_density_2023_data()
    if z_harikane is not None:
        mask = (z_harikane >= 5) & (z_harikane <= 16)
        if np.sum(mask) > 0:
            axes[0].errorbar(z_harikane[mask], re_harikane[mask],
                            yerr=[re_err_minus_harikane[mask], re_err_plus_harikane[mask]],
                            fmt='D', color='black', markersize=8, alpha=0.6,
                            markeredgecolor='k', markeredgewidth=0.8,
                            markerfacecolor = 'gray',
                            label='Harikane+23', linewidth=1.0, zorder=5)

    # ----- Bottom Panel: SMD vs Redshift -----
    valid_smd_noffb = ~np.isnan(smd_noffb_sorted)
    valid_smd_ffb = ~np.isnan(smd_ffb_sorted)
    valid_smd_ffb100 = ~np.isnan(smd_ffb100_sorted)
    valid_smd_bk25 = ~np.isnan(smd_ffb_bk25_sorted)
    if np.sum(valid_smd_noffb) > 1:
        axes[1].plot(z_sorted[valid_smd_noffb], smd_noffb_sorted[valid_smd_noffb], '-',
                    color='firebrick', linewidth=3.0, label='No FFB/MBK25 model')
        boot_valid = valid_smd_noffb & ~np.isnan(smd_noffb_lo_sorted) & ~np.isnan(smd_noffb_hi_sorted)
        if np.sum(boot_valid) > 1:
            axes[1].fill_between(z_sorted[boot_valid], smd_noffb_lo_sorted[boot_valid],
                                smd_noffb_hi_sorted[boot_valid], color='firebrick', alpha=0.2)
    if np.sum(valid_smd_ffb) > 1:
        axes[1].plot(z_sorted[valid_smd_ffb], smd_ffb_sorted[valid_smd_ffb], '-',
                    color='black', linewidth=3.5, label=r'$\alpha_{\rm FFB}=0.2$ (Li+24)')
        boot_valid = valid_smd_ffb & ~np.isnan(smd_ffb_lo_sorted) & ~np.isnan(smd_ffb_hi_sorted)
        if np.sum(boot_valid) > 1:
            axes[1].fill_between(z_sorted[boot_valid], smd_ffb_lo_sorted[boot_valid],
                                smd_ffb_hi_sorted[boot_valid], color='black', alpha=0.2)
    if np.sum(valid_smd_bk25) > 1:
        axes[1].plot(z_sorted[valid_smd_bk25], smd_ffb_bk25_sorted[valid_smd_bk25], '-',
                    color='mediumpurple', linewidth=3.0, label=r'$\alpha_{\rm FFB}=0.2$ (MBK25)')
        boot_valid = valid_smd_bk25 & ~np.isnan(smd_ffb_bk25_lo_sorted) & ~np.isnan(smd_ffb_bk25_hi_sorted)
        if np.sum(boot_valid) > 1:
            axes[1].fill_between(z_sorted[boot_valid], smd_ffb_bk25_lo_sorted[boot_valid],
                                smd_ffb_bk25_hi_sorted[boot_valid], color='mediumpurple', alpha=0.2)
    if np.sum(valid_smd_ffb100) > 1:
        axes[1].plot(z_sorted[valid_smd_ffb100], smd_ffb100_sorted[valid_smd_ffb100], '--',
                    color='steelblue', linewidth=3.0, label=r'$\alpha_{\rm FFB}=1.0$ (Li+24)')
        boot_valid = valid_smd_ffb100 & ~np.isnan(smd_ffb100_lo_sorted) & ~np.isnan(smd_ffb100_hi_sorted)
        if np.sum(boot_valid) > 1:
            axes[1].fill_between(z_sorted[boot_valid], smd_ffb100_lo_sorted[boot_valid],
                                smd_ffb100_hi_sorted[boot_valid], color='steelblue', alpha=0.2)
    valid_smd_bk25_ffb100 = ~np.isnan(smd_ffb_bk25_ffb100_sorted)
    if np.sum(valid_smd_bk25_ffb100) > 1:
        axes[1].plot(z_sorted[valid_smd_bk25_ffb100], smd_ffb_bk25_ffb100_sorted[valid_smd_bk25_ffb100], '--',
                    color='magenta', linewidth=3.0, label=r'$\alpha_{\rm FFB}=1.0$ (MBK25)')
        boot_valid = valid_smd_bk25_ffb100 & ~np.isnan(smd_ffb_bk25_ffb100_lo_sorted) & ~np.isnan(smd_ffb_bk25_ffb100_hi_sorted)
        if np.sum(boot_valid) > 1:
            axes[1].fill_between(z_sorted[boot_valid], smd_ffb_bk25_ffb100_lo_sorted[boot_valid],
                                smd_ffb_bk25_ffb100_hi_sorted[boot_valid], color='magenta', alpha=0.2)

    # Add SMD observational data
    z_madau_smd, re_madau_smd, re_err_plus_madau_smd, re_err_minus_madau_smd = load_madau_dickinson_smd_2014_data()
    if z_madau_smd is not None:
        mask = (z_madau_smd >= 5) & (z_madau_smd <= 16)
        if np.sum(mask) > 0:
            axes[1].errorbar(z_madau_smd[mask], re_madau_smd[mask],
                            yerr=[re_err_minus_madau_smd[mask], re_err_plus_madau_smd[mask]],
                            fmt='o', color='black', markersize=8, alpha=0.6,
                            markeredgecolor='k', markeredgewidth=0.8,
                            markerfacecolor = 'gray',
                            label=_tex_safe(r'Madau \& Dickinson 14'), linewidth=1.0, zorder=5)

    z_kiku, re_kiku, re_err_plus_kiku, re_err_minus_kiku = load_kikuchihara_smd_2020_data()
    if z_kiku is not None:
        mask = (z_kiku >= 5) & (z_kiku <= 16)
        if np.sum(mask) > 0:
            axes[1].errorbar(z_kiku[mask], re_kiku[mask],
                            yerr=[re_err_minus_kiku[mask], re_err_plus_kiku[mask]],
                            fmt='d', color='black', markersize=8, alpha=0.6,
                            markeredgecolor='k', markeredgewidth=0.8,
                            markerfacecolor = 'gray',
                            label='Kikuchihara+20', linewidth=1.0, zorder=5)

    z_papovich, re_papovich, re_err_plus_papovich, re_err_minus_papovich = load_papovich_smd_2023_data()
    if z_papovich is not None:
        mask = (z_papovich >= 5) & (z_papovich <= 16)
        if np.sum(mask) > 0:
            axes[1].errorbar(z_papovich[mask], re_papovich[mask],
                            yerr=[re_err_minus_papovich[mask], re_err_plus_papovich[mask]],
                            fmt='s', color='black', markersize=8, alpha=0.6,
                            markeredgecolor='k', markeredgewidth=0.8,
                            markerfacecolor = 'gray',
                            label='Papovich+23', linewidth=1.0, zorder=5)

    # Configure axes and split legends
    def is_sim_label(l):
        return 'SAGE26' in l or 'FFB' in l or 'epsilon' in l or 'MBK25' in l
    for idx, panel in enumerate(axes):
        handles, labels = panel.get_legend_handles_labels()
        sim_h = [h for h, l in zip(handles, labels) if is_sim_label(l)]
        sim_l = [l for l in labels if is_sim_label(l)]
        obs_h = [h for h, l in zip(handles, labels) if not is_sim_label(l)]
        obs_l = [l for l in labels if not is_sim_label(l)]
        if idx == 0 and sim_l:
            leg1 = _standard_legend(panel, loc='upper right', handles=sim_h, labels=sim_l)
            panel.add_artist(leg1)
        if obs_l:
            _standard_legend(panel, loc='lower left', handles=obs_h, labels=obs_l)

    axes[0].set_ylabel(r'$\log_{10} \rho_{\mathrm{SFR}}\ (M_\odot\,\mathrm{yr}^{-1}\,\mathrm{Mpc}^{-3})$')
    axes[0].set_xlim(5, 16)
    axes[0].set_ylim(-5, -1)
    axes[0].xaxis.set_major_locator(plt.MultipleLocator(2.0))
    axes[0].yaxis.set_major_locator(plt.MultipleLocator(1.0))
    axes[0].xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    axes[0].yaxis.set_minor_locator(plt.MultipleLocator(0.2))
    # Panels are joined, so drop the lowest y tick label on the top panel: it
    # would otherwise collide with the top label of the panel below.  set_yticks
    # widens the view limits to span the list it is given, so re-assert them.
    axes[0].set_yticks([-4, -3, -2, -1])
    axes[0].set_ylim(-5, -1)

    axes[1].set_xlabel(r'Redshift')
    axes[1].set_ylabel(r'$\log_{10} \rho_\star\ [M_\odot\,\mathrm{Mpc}^{-3}]$')
    axes[1].set_xlim(5, 16)
    axes[1].set_ylim(3, 8)
    axes[1].xaxis.set_major_locator(plt.MultipleLocator(2.0))
    axes[1].yaxis.set_major_locator(plt.MultipleLocator(1.0))
    axes[1].xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    axes[1].yaxis.set_minor_locator(plt.MultipleLocator(0.2))
    # Likewise drop the highest y tick label on the bottom panel, leaving the
    # shared boundary between the two panels unlabelled on both sides.
    axes[1].set_yticks([3, 4, 5, 6, 7])
    axes[1].set_ylim(3, 8)

    fig.tight_layout()
    # Butt the shared-x panels together after tight_layout has sized them.
    fig.subplots_adjust(hspace=0.0)

    output_file = os.path.join(output_dir, 'FFB_Density_Evolution_MBK25' + OUTPUT_FORMAT)
    save_figure(fig, output_file)

# ========================== PLOT 14b: FFB METHOD COMPARISON ==========================

def plot_14b_density_evolution_methods():
    """
    Create 4x1 figure:
      1. Main SFRD vs redshift
      2. SFRD Ratio (Smooth / Cutoff) with normalized observations
      3. Main SMD vs redshift
      4. SMD Ratio (Smooth / Cutoff) with normalized observations
    Handles missing models/data gracefully without crashing.
    """
    print('Plot 14b: FFB method comparison (Main + Ratios)')
    seed(SEED)

    output_dir = OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, axes = plt.subplots(4, 1, figsize=(9, 14), 
                             gridspec_kw={'height_ratios': [3, 1.2, 3, 1.2]})
    
    ax_sfrd = axes[0]
    ax_sfrd_res = axes[1]
    ax_smd = axes[2]
    ax_smd_res = axes[3]

    volume = VOLUME

    target_snaps = []
    for snap_idx, z in enumerate(REDSHIFTS):
        if 4.8 <= z <= 20.0:
            target_snaps.append(f'Snap_{snap_idx}')

    print("Loading data for FFB method plots...")

    model_keys = ['li_sigmoid', 'MBK25_smooth', 'li_nosig', 'MBK25_sharp']
    model_dirs = {
        'li_sigmoid':  PRIMARY_DIR,
        'MBK25_smooth': FFB_BK25_SMOOTH_DIR,
        'li_nosig':    FFB_NOSIGMOID_DIR,
        'MBK25_sharp':  FFB_BK25_DIR,
    }
    model_labels = {
        'li_sigmoid':  r'Li+24 (sigmoid)',
        'MBK25_smooth': r'MBK25 (log-normal scatter)',
        'li_nosig':    r'Li+24 (no sigmoid)',
        'MBK25_sharp':  r'MBK25 (no log-normal scatter))',
    }
    model_colors = {
        'li_sigmoid':  'black',
        'MBK25_smooth': 'steelblue',
        'li_nosig':    'firebrick',
        'MBK25_sharp':  'darkgreen',
    }
    model_ls = {
        'li_sigmoid':  '-',
        'MBK25_smooth': '-',
        'li_nosig':    '--',
        'MBK25_sharp':  '--',
    }

    # Warn if directories are completely missing, but keep them in the keys to prevent KeyErrors
    for key in model_keys:
        if not os.path.exists(model_dirs[key]):
            print(f"  --> WARNING: Directory for {key} not found: {model_dirs[key]}")

    # Storage arrays
    sfrd = {k: [] for k in model_keys}
    smd = {k: [] for k in model_keys}
    redshifts_density = []

    N_BOOT = 100
    rng = np.random.default_rng(SEED)

    def bootstrap_density(values):
        if len(values) == 0: return np.nan
        total = np.sum(values)
        return np.log10(total / volume) if total > 0 else np.nan

    for Snapshot in target_snaps:
        snapnum = int(Snapshot.split('_')[1])
        z = REDSHIFTS[snapnum]

        props = ['StellarMass', 'SfrDisk', 'SfrBulge']
        any_loaded = False
        
        # Temporary storage to ensure array lengths stay perfectly aligned
        temp_sfrd = {}
        temp_smd = {}

        for key in model_keys:
            data = None
            try:
                if os.path.exists(model_dirs[key]):
                    data = load_model(model_dirs[key], snapshot=Snapshot, properties=props)
            except Exception as e:
                print(f"    Error loading {key} at {Snapshot}: {e}")
                
            if data:
                any_loaded = True
                sfr_vals = data['SfrDisk'] + data['SfrBulge']
                sm_vals = data['StellarMass']
                temp_sfrd[key] = bootstrap_density(sfr_vals)
                temp_smd[key] = bootstrap_density(sm_vals)
            else:
                temp_sfrd[key] = np.nan
                temp_smd[key] = np.nan

        # Only append to main arrays if at least one model loaded data for this snapshot
        if any_loaded:
            redshifts_density.append(z)
            for key in model_keys:
                sfrd[key].append(temp_sfrd[key])
                smd[key].append(temp_smd[key])

    redshifts_density = np.array(redshifts_density)
    
    if len(redshifts_density) == 0:
        print("CRITICAL: No data loaded for any model across any snapshot. Exiting plot generation.")
        plt.close(fig)
        return

    sort_idx = np.argsort(redshifts_density)
    z_sorted = redshifts_density[sort_idx]

    sfrd_sorted = {k: np.array(sfrd[k])[sort_idx] for k in model_keys}
    smd_sorted = {k: np.array(smd[k])[sort_idx] for k in model_keys}

    print("Generating plots...")

    # ==========================================
    # 1. PLOT MAIN PANELS (Absolute Values)
    # ==========================================
    for key in model_keys:
        v_sfrd = ~np.isnan(sfrd_sorted[key])
        if np.sum(v_sfrd) > 1:
            ax_sfrd.plot(z_sorted[v_sfrd], sfrd_sorted[key][v_sfrd],
                         model_ls[key], color=model_colors[key],
                         linewidth=3, label=model_labels[key])
            
        v_smd = ~np.isnan(smd_sorted[key])
        if np.sum(v_smd) > 1:
            ax_smd.plot(z_sorted[v_smd], smd_sorted[key][v_smd],
                        model_ls[key], color=model_colors[key],
                        linewidth=3, label=model_labels[key])

    # Add Absolute Observations (SFRD)
    try:
        z_madau, r_madau, r_ep_madau, r_em_madau = load_madau_dickinson_2014_data()
        if z_madau is not None:
            mask = (z_madau >= 5) & (z_madau <= 16)
            ax_sfrd.errorbar(z_madau[mask], r_madau[mask], yerr=[r_em_madau[mask], r_ep_madau[mask]],
                             fmt='o', color='gray', markeredgecolor='k', label=_tex_safe(r'Madau \& Dickinson 14'))
    except Exception as e: print(f"Could not plot Madau SFRD: {e}")

    try:
        z_harikane, r_hari, r_ep_hari, r_em_hari = load_harikane_sfr_density_2023_data()
        if z_harikane is not None:
            mask = (z_harikane >= 5) & (z_harikane <= 16)
            ax_sfrd.errorbar(z_harikane[mask], r_hari[mask], yerr=[r_em_hari[mask], r_ep_hari[mask]],
                             fmt='D', color='gray', markeredgecolor='k', label='Harikane+23')
    except Exception as e: print(f"Could not plot Harikane SFRD: {e}")

    # Add Absolute Observations (SMD)
    try:
        z_m_smd, r_m_smd, r_ep_m_smd, r_em_m_smd = load_madau_dickinson_smd_2014_data()
        if z_m_smd is not None:
            mask = (z_m_smd >= 5) & (z_m_smd <= 16)
            ax_smd.errorbar(z_m_smd[mask], r_m_smd[mask], yerr=[r_em_m_smd[mask], r_ep_m_smd[mask]],
                            fmt='o', color='gray', markeredgecolor='k', label=_tex_safe(r'Madau \& Dickinson 14'))
    except Exception as e: print(f"Could not plot Madau SMD: {e}")

    try:
        z_papo, r_papo, r_ep_papo, r_em_papo = load_papovich_smd_2023_data()
        if z_papo is not None:
            mask = (z_papo >= 5) & (z_papo <= 16)
            ax_smd.errorbar(z_papo[mask], r_papo[mask], yerr=[r_em_papo[mask], r_ep_papo[mask]],
                            fmt='s', color='gray', markeredgecolor='k', label='Papovich+23')
    except Exception as e: print(f"Could not plot Papovich SMD: {e}")

    # ==========================================
    # 2. PLOT RESIDUAL PANELS (Ratios vs Cutoff)
    # ==========================================
    for ax in [ax_sfrd_res, ax_smd_res]:
        ax.axhline(0, color='black', linestyle='-', linewidth=1.5, zorder=1)
        ax.axhspan(-0.3, 0.3, color='gray', alpha=0.15, zorder=0, label=r'$\pm 0.3$ dex')

    # SFRD Ratios
    v_sfrd_li = ~np.isnan(sfrd_sorted['li_sigmoid']) & ~np.isnan(sfrd_sorted['li_nosig'])
    if np.sum(v_sfrd_li) > 0:
        ax_sfrd_res.plot(z_sorted[v_sfrd_li], sfrd_sorted['li_sigmoid'][v_sfrd_li] - sfrd_sorted['li_nosig'][v_sfrd_li],
                         '-', color='black', linewidth=3)
    
    v_sfrd_mbk = ~np.isnan(sfrd_sorted['MBK25_smooth']) & ~np.isnan(sfrd_sorted['MBK25_sharp'])
    if np.sum(v_sfrd_mbk) > 0:
        ax_sfrd_res.plot(z_sorted[v_sfrd_mbk], sfrd_sorted['MBK25_smooth'][v_sfrd_mbk] - sfrd_sorted['MBK25_sharp'][v_sfrd_mbk],
                         '-', color='steelblue', linewidth=3)

    # SMD Ratios
    v_smd_li = ~np.isnan(smd_sorted['li_sigmoid']) & ~np.isnan(smd_sorted['li_nosig'])
    if np.sum(v_smd_li) > 0:
        ax_smd_res.plot(z_sorted[v_smd_li], smd_sorted['li_sigmoid'][v_smd_li] - smd_sorted['li_nosig'][v_smd_li],
                        '-', color='black', linewidth=3)

    v_smd_mbk = ~np.isnan(smd_sorted['MBK25_smooth']) & ~np.isnan(smd_sorted['MBK25_sharp'])
    if np.sum(v_smd_mbk) > 0:
        ax_smd_res.plot(z_sorted[v_smd_mbk], smd_sorted['MBK25_smooth'][v_smd_mbk] - smd_sorted['MBK25_sharp'][v_smd_mbk],
                        '-', color='steelblue', linewidth=3)

    # Helper function for normalized observations
    def plot_obs_ratio(ax, z_obs, y_obs, err_minus, err_plus, base_z, base_y, fmt):
        if z_obs is None: return
        mask = (z_obs >= 5) & (z_obs <= 16) & (z_obs >= np.min(base_z)) & (z_obs <= np.max(base_z))
        if np.sum(mask) > 0:
            zm, ym = z_obs[mask], y_obs[mask]
            em, ep = err_minus[mask], err_plus[mask]
            interp_base = np.interp(zm, base_z, base_y)
            y_ratio = ym - interp_base 
            ax.errorbar(zm, y_ratio, yerr=[em, ep], fmt=fmt, color='gray', 
                        markeredgecolor='k', zorder=5)

    base_z_sfrd = z_sorted[~np.isnan(sfrd_sorted['li_nosig'])]
    base_y_sfrd = sfrd_sorted['li_nosig'][~np.isnan(sfrd_sorted['li_nosig'])]
    
    if len(base_z_sfrd) > 1:
        try: plot_obs_ratio(ax_sfrd_res, z_madau, r_madau, r_em_madau, r_ep_madau, base_z_sfrd, base_y_sfrd, 'o')
        except NameError: pass
        try: plot_obs_ratio(ax_sfrd_res, z_harikane, r_hari, r_em_hari, r_ep_hari, base_z_sfrd, base_y_sfrd, 'D')
        except NameError: pass

    base_z_smd = z_sorted[~np.isnan(smd_sorted['li_nosig'])]
    base_y_smd = smd_sorted['li_nosig'][~np.isnan(smd_sorted['li_nosig'])]
    
    if len(base_z_smd) > 1:
        try: plot_obs_ratio(ax_smd_res, z_m_smd, r_m_smd, r_em_m_smd, r_ep_m_smd, base_z_smd, base_y_smd, 'o')
        except NameError: pass
        try: plot_obs_ratio(ax_smd_res, z_papo, r_papo, r_em_papo, r_ep_papo, base_z_smd, base_y_smd, 's')
        except NameError: pass

    # ==========================================
    # 3. FORMATTING AND CLEANUP
    # ==========================================
    for ax in [ax_sfrd, ax_sfrd_res, ax_smd, ax_smd_res]:
        ax.set_xlim(5, 16)
        ax.xaxis.set_major_locator(plt.MultipleLocator(2.0))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))

    ax_sfrd.set_xticklabels([])
    ax_smd.set_xticklabels([])

    ax_sfrd.set_ylabel(r'$\log_{10} \rho_{\mathrm{SFR}}$')
    ax_sfrd.set_ylim(-5, -1)
    
    ax_sfrd_res.set_ylabel(r'$\Delta$ (dex)')
    ax_sfrd_res.set_ylim(-0.5, 2.5) 

    ax_smd.set_ylabel(r'$\log_{10} \rho_\star$')
    ax_smd.set_ylim(3, 8)
    
    ax_smd_res.set_xlabel('Redshift')
    ax_smd_res.set_ylabel(r'$\Delta$ (dex)')
    ax_smd_res.set_ylim(-0.5, 2.5)

    # ------------------------------------------
    # 4. SPLIT LEGENDS
    # ------------------------------------------
    sim_label_set = set(model_labels.values())

    for ax in [ax_sfrd, ax_smd]:
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            continue
            
        # Separate simulation handles from observation handles
        sim_h = [h for h, l in zip(handles, labels) if l in sim_label_set]
        sim_l = [l for l in labels if l in sim_label_set]
        obs_h = [h for h, l in zip(handles, labels) if l not in sim_label_set]
        obs_l = [l for l in labels if l not in sim_label_set]
        
        # Models -> Lower Left
        if sim_h:
            leg_sim = ax.legend(sim_h, sim_l, loc='lower left', frameon=False, title='SAGE26')
            leg_sim.get_title().set_fontweight('bold')
            ax.add_artist(leg_sim)
            
        # Observations -> Upper Right (If the data drops down to the right, this space is clear)
        if obs_h:
            ax.legend(obs_h, obs_l, loc='upper right', frameon=False)

    # Residual legends (just the shaded +/- 0.3 dex box)
    if ax_sfrd_res.get_legend_handles_labels()[0]:
        ax_sfrd_res.legend(loc='upper right', frameon=False)
    if ax_smd_res.get_legend_handles_labels()[0]:
        ax_smd_res.legend(loc='upper right', frameon=False)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1)

    output_file = os.path.join(output_dir, 'FFB_Density_Evolution_Methods' + OUTPUT_FORMAT)
    plt.savefig(output_file)
    print(f'Saved file to {output_file}\n')
    plt.close()

# ========================== PLOT 15: sSFR vs STELLAR MASS (DENSITY) ==========================

def plot_15_sfr_vs_stellar_mass(primary, vanilla):
    """
    Star formation rate vs. stellar mass distribution.

    Shows the distribution of galaxies in the SFR-mass plane
    as a KDE contour plot, with C16 as a scatter overlay.
    Includes median lines exclusively for the star-forming populations.
    """
    print('Plot 15: SFR vs stellar mass (All + SF Medians)')

    # --- Primary model ---
    sfr = primary['SfrDisk'] + primary['SfrBulge']
    w = (primary['StellarMass'] > 1e8) & (sfr > 0)
    log_mass = np.log10(primary['StellarMass'][w])
    log_sfr = np.log10(sfr[w])

    # Safely calculate the SF mask on the already-filtered arrays
    # log(SFR/Mass) = log(SFR) - log(Mass)
    starforming_sage26 = (log_sfr - log_mass) > SSFR_CUT

    # --- Plot ---
    fig = plt.figure()
    ax = fig.add_subplot(111)

    mass_bins = np.arange(8.0, 12.0 + 0.1, 0.1)
    
    # Plot ALL SAGE26 galaxies
    plot_binned_median_1sigma(
        ax, log_mass, log_sfr, mass_bins,
        color='steelblue', label='SAGE26 (All)',
        alpha=0.25, lw=3.5, min_count=50,
        zorder_fill=Z_MODEL_BAND, zorder_line=Z_MODEL_LINE,
    )

    # Plot SAGE26 median for SF only
    centers, med_sfr, _, _ = binned_median(log_mass[starforming_sage26], log_sfr[starforming_sage26], mass_bins)
    ax.plot(centers, med_sfr, color='steelblue', lw=1.5, ls=':', label='SAGE26 (SF)', zorder=Z_MODEL_LINE+1)


    # --- C16 (Vanilla) model ---
    sfr_v = vanilla['SfrDisk'] + vanilla['SfrBulge']
    w_v = (vanilla['StellarMass'] > 1e8) & (sfr_v > 0)
    if np.any(w_v):
        log_mass_v = np.log10(vanilla['StellarMass'][w_v])
        log_sfr_v = np.log10(sfr_v[w_v])
        
        starforming_vanilla = (log_sfr_v - log_mass_v) > SSFR_CUT

        # Plot ALL SAGE16 galaxies
        plot_binned_median_1sigma(
            ax, log_mass_v, log_sfr_v, mass_bins,
            color='purple', label='SAGE16 (All)', ls='--',
            alpha=0.20, lw=3.5, min_count=50,
            zorder_fill=Z_MODEL_BAND_ALT, zorder_line=Z_MODEL_LINE_ALT,
        )

        # Plot SAGE16 median for SF only
        centers_v, med_sfr_v, _, _ = binned_median(log_mass_v[starforming_vanilla], log_sfr_v[starforming_vanilla], mass_bins)
        ax.plot(centers_v, med_sfr_v, color='purple', lw=1.5, ls=':', label='SAGE16 (SF)', zorder=Z_MODEL_LINE_ALT+1)

    # --- Load Brinchmann et al. (2004) data ---
    bz04_mass, bz04_sfr = load_brinchmann_sfr_mass_2004_data()
    if bz04_mass is not None and bz04_sfr is not None:
        ax.scatter(bz04_mass, bz04_sfr, marker='d', alpha=0.6, zorder=Z_OBS,
                facecolors='gray', edgecolors='black', s=50,
                label='Brinchmann+04')

    # --- Load Terrazas et al. (2017) data ---
    ter_mass, ter_sfr = load_terrazas17_mbh_host_sfr_data()
    if ter_mass is not None and ter_sfr is not None:
        # Plot with error bars
        ax.errorbar(ter_mass, ter_sfr, xerr=0.2, yerr=0.3, fmt='o', ecolor='black', alpha=0.6, zorder=Z_OBS,
                   mfc='gray', mec='black', ms=8, mew=1.0, elinewidth=1.0, label='Terrazas+17')
        
    # --- Load and plot GAMA ProSpect Claudia data ---
    log_ms, log_sfr = load_gama_prospect_claudia()
    if log_ms is not None and log_sfr is not None:
        # Plot binned medians/errors
        bins = np.linspace(8, 12, 13)
        centers_gama, med_gama, p25_gama, p75_gama = binned_median(log_ms, log_sfr, bins)
        valid = ~np.isnan(med_gama)
        ax.errorbar(centers_gama[valid], med_gama[valid], 
                    yerr=[med_gama[valid] - p25_gama[valid], p75_gama[valid] - med_gama[valid]],
                    fmt='s', color='black', label='Bellstedt+20', markersize=8, alpha=0.6, zorder=Z_OBS,
                    markeredgewidth=0.8, markerfacecolor='gray', markeredgecolor='black')
        

    ax.set_xlim(8.0, 12.0)
    ax.set_ylim(-4.0, 2.0)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.set_xlabel(r'$\log_{10}\ m_{\mathrm{*}}\ [M_{\odot}]$')
    ax.set_ylabel(r'$\log_{10}\ \mathrm{SFR}\ [M_{\odot}\,\mathrm{yr}^{-1}]$')

    # Use startswith to cleanly capture the newly named 'All' and 'SF' simulation labels
    handles, labels = ax.get_legend_handles_labels()
    sim_h = [h for h, l in zip(handles, labels) if l.startswith(('SAGE26', 'SAGE16'))]
    sim_l = [l for l in labels if l.startswith(('SAGE26', 'SAGE16'))]
    obs_h = [h for h, l in zip(handles, labels) if not l.startswith(('SAGE26', 'SAGE16'))]
    obs_l = [l for l in labels if not l.startswith(('SAGE26', 'SAGE16'))]
    
    leg1 = _standard_legend(ax, loc='lower left', handles=sim_h, labels=sim_l)
    ax.add_artist(leg1)
    _standard_legend(ax, loc='upper left', handles=obs_h, labels=obs_l)
    fig.tight_layout()

    save_figure(fig, os.path.join(OUTPUT_DIR,
                'StarFormationRate' + OUTPUT_FORMAT))

# ========================== PLOT 16: COSMIC SFR DENSITY HISTORY (CSRDH) ==========================

def plot_16_sfrd_history():
    """
    Plot G: Cosmic SFR Density History.
    Replicates the structure of the uploaded paper_plots.py.
    Iterates snapshots 0-63 explicitly to capture full history.
    """
    print('Plot 16: SFR Density History (CSRDH)')
    
    # --- 1. SETUP & DEFINITIONS ---
    # Exact redshift list from the uploaded script
    redshifts = np.array([
        127.000, 79.998, 50.000, 30.000, 19.916, 18.244, 16.725, 15.343, 14.086, 12.941, 
        11.897, 10.944, 10.073, 9.278, 8.550, 7.883, 7.272, 6.712, 6.197, 5.724, 
        5.289, 4.888, 4.520, 4.179, 3.866, 3.576, 3.308, 3.060, 2.831, 2.619, 
        2.422, 2.239, 2.070, 1.913, 1.766, 1.630, 1.504, 1.386, 1.276, 1.173, 
        1.078, 0.989, 0.905, 0.828, 0.755, 0.687, 0.624, 0.564, 0.509, 0.457, 
        0.408, 0.362, 0.320, 0.280, 0.242, 0.208, 0.175, 0.144, 0.116, 0.089, 
        0.064, 0.041, 0.020, 0.000
    ])

    FirstSnap = 0
    LastSnap = 63
    
    # Define models to process
    # We construct a list similar to SFR_SimDirs
    redshifts_mu = np.array(MINIUCHUU_REDSHIFTS)
    sim_dirs = []

    # 1. Primary Model (SAGE26 Millennium)
    if os.path.exists(PRIMARY_DIR):
        sim_dirs.append({
            'path': PRIMARY_DIR, 'label': 'SAGE26 (Millennium)',
            'color': 'steelblue', 'ls': '-', 'lw': 3.5,
            'redshifts': redshifts, 'first_snap': FirstSnap, 'last_snap': LastSnap,
            'volume': VOLUME,
        })

    # 2. Vanilla Model (C16)
    if os.path.exists(VANILLA_DIR):
        sim_dirs.append({
            'path': VANILLA_DIR, 'label': 'SAGE16',
            'color': 'purple', 'ls': '--', 'lw': 3.5,
            'redshifts': redshifts, 'first_snap': FirstSnap, 'last_snap': LastSnap,
            'volume': VOLUME,
        })

    # 3. miniUchuu Model
    if os.path.exists(MINIUCHUU_DIR):
        sim_dirs.append({
            'path': MINIUCHUU_DIR, 'label': 'SAGE26 (miniUchuu)',
            'color': 'steelblue', 'ls': '--', 'lw': 3.5,
            'redshifts': redshifts_mu, 'first_snap': MINIUCHUU_FIRST_SNAP, 'last_snap': MINIUCHUU_LAST_SNAP,
            'volume': MINIUCHUU_VOLUME,
        })

    # # 4. CGM Dynamical Time Model
    # if os.path.exists(CGM_DYN_DIR):
    #     sim_dirs.append({
    #         'path': CGM_DYN_DIR, 'label': 'SAGE26 (CGM Dyn Time)',
    #         'color': 'darkorange', 'ls': '-.', 'lw': 3.5,
    #         'redshifts': redshifts, 'first_snap': FirstSnap, 'last_snap': LastSnap,
    #         'volume': VOLUME,
    #     })

    # # 5. Simple CGM with disk smoothing
    # if os.path.exists(DISK_SMOOTH_DIR):
    #     sim_dirs.append({
    #         'path': DISK_SMOOTH_DIR, 'label': 'SAGE26 (Disk Smooth)',
    #         'color': 'darkgreen', 'ls': ':', 'lw': 3.5,
    #         'redshifts': redshifts, 'first_snap': FirstSnap, 'last_snap': LastSnap,
    #         'volume': VOLUME,
    #     })

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # --- 2. PLOT OBSERVATIONAL DATA (Croton et al. 2006 Compilation) ---
    # Exact array from uploaded file
    ObsSFRdensity = np.array([
        [0, 0.0158489, 0, 0, 0.0251189, 0.01000000],
        [0.150000, 0.0173780, 0, 0.300000, 0.0181970, 0.0165959],
        [0.0425000, 0.0239883, 0.0425000, 0.0425000, 0.0269153, 0.0213796],
        [0.200000, 0.0295121, 0.100000, 0.300000, 0.0323594, 0.0269154],
        [0.350000, 0.0147911, 0.200000, 0.500000, 0.0173780, 0.0125893],
        [0.625000, 0.0275423, 0.500000, 0.750000, 0.0331131, 0.0229087],
        [0.825000, 0.0549541, 0.750000, 1.00000, 0.0776247, 0.0389045],
        [0.625000, 0.0794328, 0.500000, 0.750000, 0.0954993, 0.0660693],
        [0.700000, 0.0323594, 0.575000, 0.825000, 0.0371535, 0.0281838],
        [1.25000, 0.0467735, 1.50000, 1.00000, 0.0660693, 0.0331131],
        [0.750000, 0.0549541, 0.500000, 1.00000, 0.0389045, 0.0776247],
        [1.25000, 0.0741310, 1.00000, 1.50000, 0.0524807, 0.104713],
        [1.75000, 0.0562341, 1.50000, 2.00000, 0.0398107, 0.0794328],
        [2.75000, 0.0794328, 2.00000, 3.50000, 0.0562341, 0.112202],
        [4.00000, 0.0309030, 3.50000, 4.50000, 0.0489779, 0.0194984],
        [0.250000, 0.0398107, 0.00000, 0.500000, 0.0239883, 0.0812831],
        [0.750000, 0.0446684, 0.500000, 1.00000, 0.0323594, 0.0776247],
        [1.25000, 0.0630957, 1.00000, 1.50000, 0.0478630, 0.109648],
        [1.75000, 0.0645654, 1.50000, 2.00000, 0.0489779, 0.112202],
        [2.50000, 0.0831764, 2.00000, 3.00000, 0.0512861, 0.158489],
        [3.50000, 0.0776247, 3.00000, 4.00000, 0.0416869, 0.169824],
        [4.50000, 0.0977237, 4.00000, 5.00000, 0.0416869, 0.269153],
        [5.50000, 0.0426580, 5.00000, 6.00000, 0.0177828, 0.165959],
        [3.00000, 0.120226, 2.00000, 4.00000, 0.173780, 0.0831764],
        [3.04000, 0.128825, 2.69000, 3.39000, 0.151356, 0.109648],
        [4.13000, 0.114815, 3.78000, 4.48000, 0.144544, 0.0912011],
        [0.350000, 0.0346737, 0.200000, 0.500000, 0.0537032, 0.0165959],
        [0.750000, 0.0512861, 0.500000, 1.00000, 0.0575440, 0.0436516],
        [1.50000, 0.0691831, 1.00000, 2.00000, 0.0758578, 0.0630957],
        [2.50000, 0.147911, 2.00000, 3.00000, 0.169824, 0.128825],
        [3.50000, 0.0645654, 3.00000, 4.00000, 0.0776247, 0.0512861],
    ], dtype=np.float32)

    ObsRedshift = ObsSFRdensity[:, 0]
    xErrLo = np.abs(ObsSFRdensity[:, 0]-ObsSFRdensity[:, 2])
    xErrHi = np.abs(ObsSFRdensity[:, 3]-ObsSFRdensity[:, 0])
    ObsSFR = np.log10(ObsSFRdensity[:, 1])
    yErrLo = np.abs(np.log10(ObsSFRdensity[:, 1])-np.log10(ObsSFRdensity[:, 4]))
    yErrHi = np.abs(np.log10(ObsSFRdensity[:, 5])-np.log10(ObsSFRdensity[:, 1]))

    ax.errorbar(ObsRedshift, ObsSFR, yerr=[yErrLo, yErrHi], xerr=[xErrLo, xErrHi],
                fmt='o', markerfacecolor='gray', markeredgecolor='k', markeredgewidth=1.0,
                ecolor='k', color='k', ms=8, lw=1.0, alpha=0.6, ls='none',
                zorder=Z_OBS, label='Somerville+01 compilation')

    # Madau & Dickinson 2014 Fit
    def MD14_sfrd(z):
        psi = 0.015 * (1+z)**2.7 / (1 + ((1+z)/2.9)**5.6)
        return psi

    # Their eq. 15 is a Salpeter-IMF fit, so it comes DOWN onto the model's
    # Chabrier scale. This previously multiplied by 1/0.63, raising it by 0.2 dex
    # instead of lowering it by 0.24 -- a 0.44 dex error in the wrong direction,
    # which made the model look ~0.2 dex low against this curve while looking
    # ~0.2 dex high against the natively-Chabrier COSMOS-Web curve beside it.
    z_values = np.linspace(0, 8, 200)
    md14 = np.log10(MD14_sfrd(z_values)) + SALPETER_TO_CHABRIER_DEX
    ax.plot(z_values, md14, color='gray', lw=1.5, alpha=0.6, zorder=Z_OBS,
            label=_tex_safe(r'Madau \& Dickinson 2014'))

    # --- 3. PROCESS & PLOT MODELS ---
    N_BOOT = 100
    model_results = {}  # Store results for comparison
    for sim in sim_dirs:
        sim_path = sim['path']
        sim_label = sim['label']
        sim_redshifts = sim['redshifts']
        sim_first = sim['first_snap']
        sim_last = sim['last_snap']
        sim_volume = sim['volume']
        do_bootstrap = True  # bootstrap error shading for all three models
        n_snaps = sim_last - sim_first + 1
        sfr_density = np.zeros(n_snaps)
        sfr_density_lo = np.zeros(n_snaps)
        sfr_density_hi = np.zeros(n_snaps)

        # Loop strictly from first_snap to last_snap
        for snap in range(sim_first, sim_last + 1):
            snap_name = f'Snap_{snap}'
            idx = snap - sim_first

            try:
                model_files = find_model_files(sim_path)
                d = read_snap_from_files(model_files, snap_name,
                                         ['SfrDisk', 'SfrBulge'])
                if d:
                    sfr_disk = d['SfrDisk']
                    sfr_bulge = d['SfrBulge']
                    sfr_total = sfr_disk + sfr_bulge
                    sfr_density[idx] = np.sum(sfr_total) / sim_volume

                    if do_bootstrap and len(sfr_total) > 0:
                        n_gal = len(sfr_total)
                        boot = np.array([
                            np.sum(sfr_total[np.random.randint(0, n_gal, n_gal)])
                            for _ in range(N_BOOT)
                        ]) / sim_volume
                        sfr_density_lo[idx] = np.percentile(boot, 16)
                        sfr_density_hi[idx] = np.percentile(boot, 84)
            except Exception:
                continue

        # Plot
        nonzero = np.where(sfr_density > 0.0)[0]
        if len(nonzero) > 0:
            z_vals = sim_redshifts[sim_first:sim_last+1]
            ax.plot(z_vals[nonzero], np.log10(sfr_density[nonzero]),
                    lw=sim['lw'], color=sim['color'], linestyle=sim['ls'],
                    zorder=Z_MODEL_LINE, label=sim_label)

            if do_bootstrap:
                valid = nonzero[sfr_density_lo[nonzero] > 0]
                if len(valid) > 0:
                    ax.fill_between(z_vals[valid],
                                    np.log10(sfr_density_lo[valid]),
                                    np.log10(sfr_density_hi[valid]),
                                    color=sim['color'], alpha=0.2,
                                    zorder=Z_MODEL_BAND)

            # Store results for comparison
            model_results[sim_label] = {
                'z': z_vals[nonzero],
                'sfrd': np.log10(sfr_density[nonzero])
            }

    # ===== QUANTITATIVE COMPARISON: SAGE26 vs SAGE16 =====
    if 'SAGE26 (Millennium)' in model_results and 'SAGE16' in model_results:
        print("\n" + "="*60)
        print("QUANTITATIVE COMPARISON: SAGE26 vs SAGE16 (SFRD)")
        print("="*60)

        z_sage = model_results['SAGE26 (Millennium)']['z']
        sfrd_sage = model_results['SAGE26 (Millennium)']['sfrd']
        z_c16 = model_results['SAGE16']['z']
        sfrd_c16 = model_results['SAGE16']['sfrd']

        # Filter to plot range (z <= 7.5)
        z_mask = z_sage <= 7.5
        z_sage = z_sage[z_mask]
        sfrd_sage = sfrd_sage[z_mask]

        # Interpolate C16 to SAGE26 redshifts for direct comparison
        from scipy.interpolate import interp1d
        c16_interp = interp1d(z_c16, sfrd_c16, bounds_error=False, fill_value=np.nan)
        sfrd_c16_matched = c16_interp(z_sage)

        # Find valid comparison points
        valid = ~np.isnan(sfrd_c16_matched)
        z_valid = z_sage[valid]
        sfrd_diff = sfrd_c16_matched[valid] - sfrd_sage[valid]

        print(f"\n  Comparison over z = {z_valid.min():.1f} to {z_valid.max():.1f}")
        print(f"  Mean difference (SAGE16 - SAGE26):  {np.mean(sfrd_diff):+.3f} dex")
        print(f"  Median difference:               {np.median(sfrd_diff):+.3f} dex")
        print(f"  Std of difference:               {np.std(sfrd_diff):.3f} dex")
        print(f"  Max difference at z={z_valid[np.argmax(sfrd_diff)]:.1f}: {np.max(sfrd_diff):+.3f} dex ({10**np.max(sfrd_diff):.1f}x)")
        print(f"  Min difference at z={z_valid[np.argmin(sfrd_diff)]:.1f}: {np.min(sfrd_diff):+.3f} dex ({10**np.min(sfrd_diff):.1f}x)")

        print("\n  SFRD at specific redshifts:")
        for target_z in [0, 1, 2, 3, 4, 5, 6]:
            idx = np.argmin(np.abs(z_valid - target_z))
            if np.abs(z_valid[idx] - target_z) < 0.5:
                sage_val = sfrd_sage[valid][idx]
                c16_val = sfrd_c16_matched[valid][idx]
                diff = c16_val - sage_val
                print(f"    z~{z_valid[idx]:.1f}: SAGE16={c16_val:.2f}, SAGE26={sage_val:.2f}, Δ={diff:+.2f} dex ({10**diff:.1f}x)")

        print("="*60 + "\n")

    # --- COSMOS-Web ---
    if HAS_ASTROPY:
        csfrd_file = './data/sfrd/CSFRD_inferred_from_SMD.ecsv'
        if os.path.exists(csfrd_file):
            try:
                csfrd_table = Table.read(csfrd_file, format='ascii.ecsv')
                z_csfrd = np.array(csfrd_table['Redshift'])
                sfrd_50 = np.log10(np.array(csfrd_table['sfrd_50']))
                sfrd_16 = np.log10(np.array(csfrd_table['sfrd_16']))
                sfrd_84 = np.log10(np.array(csfrd_table['sfrd_84']))
                # Densely sampled (~4000 pts) inferred curve -> subsample to
                # discrete markers spaced ~0.25 in redshift so they read as points.
                dz = np.median(np.diff(z_csfrd)) if len(z_csfrd) > 1 else 0.25
                stride = max(1, int(round(0.25 / dz))) if dz > 0 else 1
                sel = slice(None, None, stride)
                ax.errorbar(z_csfrd[sel], sfrd_50[sel],
                            yerr=[(sfrd_50 - sfrd_16)[sel], (sfrd_84 - sfrd_50)[sel]],
                            fmt='s', color='k', markerfacecolor='gray',
                            markeredgecolor='k', markeredgewidth=1.0, ecolor='k',
                            ms=8, lw=1.0, alpha=0.6, ls='none',
                            zorder=Z_OBS, label='COSMOS-Web')
            except Exception as e:
                print(f"Error loading CSFRD inferred from SMD: {e}")

    # --- 4. FORMATTING ---
    ax.set_ylabel(r'$\log_{10}\ {\rho_{\rm SFR}}\ (M_{\odot}\ \mathrm{yr}^{-1}\ \mathrm{Mpc}^{-3})$')
    ax.set_xlabel(r'$\mathrm{Redshift}$')
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.set_xlim(0.0, 7.5)
    ax.set_ylim(-3.0, -0.5)

    sim_names = {'SAGE26 (Millennium)', 'SAGE26 (miniUchuu)', 'SAGE16', 'SAGE26 (CGM Dyn Time)'}
    handles, labels = ax.get_legend_handles_labels()
    sim_h = [h for h, l in zip(handles, labels) if l in sim_names]
    sim_l = [l for l in labels if l in sim_names]
    obs_h = [h for h, l in zip(handles, labels) if l not in sim_names]
    obs_l = [l for l in labels if l not in sim_names]
    # Legend order: SAGE16 and SAGE26 (Millennium) swapped
    sim_order = {'SAGE16': 0, 'SAGE26 (Millennium)': 1, 'SAGE26 (miniUchuu)': 2, 'SAGE26 (CGM Dyn Time)': 3}
    _sim_pairs = sorted(zip(sim_l, sim_h), key=lambda p: sim_order.get(p[0], 99))
    sim_l = [p[0] for p in _sim_pairs]
    sim_h = [p[1] for p in _sim_pairs]
    leg1 = _standard_legend(ax, loc='lower right', handles=sim_h, labels=sim_l)
    ax.add_artist(leg1)
    _standard_legend(ax, loc='lower left', handles=obs_h, labels=obs_l)

    fig.tight_layout()

    outputFile = os.path.join(OUTPUT_DIR, 'SFR_Density_History_Comparison' + OUTPUT_FORMAT)
    save_figure(fig, outputFile)


# ========================== PLOT 17: STELLAR MASS DENSITY HISTORY (SMDH) ==========================

def plot_17_smd_history():
    """
    Plot H: Stellar Mass Density History.
    Replicates structure of uploaded paper_plots.py using explicit snap loops 0-63.
    """
    print('Plot 17: Stellar Mass Density History (SMDH)')
    
    # --- 1. SETUP ---
    redshifts = np.array([
        127.000, 79.998, 50.000, 30.000, 19.916, 18.244, 16.725, 15.343, 14.086, 12.941, 
        11.897, 10.944, 10.073, 9.278, 8.550, 7.883, 7.272, 6.712, 6.197, 5.724, 
        5.289, 4.888, 4.520, 4.179, 3.866, 3.576, 3.308, 3.060, 2.831, 2.619, 
        2.422, 2.239, 2.070, 1.913, 1.766, 1.630, 1.504, 1.386, 1.276, 1.173, 
        1.078, 0.989, 0.905, 0.828, 0.755, 0.687, 0.624, 0.564, 0.509, 0.457, 
        0.408, 0.362, 0.320, 0.280, 0.242, 0.208, 0.175, 0.144, 0.116, 0.089, 
        0.064, 0.041, 0.020, 0.000
    ])

    FirstSnap = 0
    LastSnap = 63
    
    redshifts_mu = np.array(MINIUCHUU_REDSHIFTS)
    sim_dirs = []
    if os.path.exists(PRIMARY_DIR):
        sim_dirs.append({
            'path': PRIMARY_DIR, 'label': 'SAGE26 (Millennium)', 'color': 'black', 'ls': '-', 'lw': 3.5,
            'redshifts': redshifts, 'first_snap': FirstSnap, 'last_snap': LastSnap,
            'volume': VOLUME, 'mass_convert': MASS_CONVERT,
        })
    if os.path.exists(VANILLA_DIR):
        sim_dirs.append({
            'path': VANILLA_DIR, 'label': 'SAGE16', 'color': 'firebrick', 'ls': '--', 'lw': 2.0,
            'redshifts': redshifts, 'first_snap': FirstSnap, 'last_snap': LastSnap,
            'volume': VOLUME, 'mass_convert': MASS_CONVERT,
        })
    if os.path.exists(MINIUCHUU_DIR):
        sim_dirs.append({
            'path': MINIUCHUU_DIR, 'label': 'SAGE26 (miniUchuu)', 'color': 'steelblue', 'ls': '--', 'lw': 3.5,
            'redshifts': redshifts_mu, 'first_snap': MINIUCHUU_FIRST_SNAP, 'last_snap': MINIUCHUU_LAST_SNAP,
            'volume': MINIUCHUU_VOLUME, 'mass_convert': MINIUCHUU_MASS_CONVERT,
        })
    # for m in FFB_MODELS:
    #     if os.path.exists(m['dir']):
    #         sim_dirs.append({'path': m['dir'], 'label': m['name'], 'color': 'blue', 'ls': ':'})

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # --- 2. PLOT OBSERVATIONAL DATA (Marchesini et al. 2009 Compilation) ---
    # Values are (minz, maxz, rho,-err,+err)
    dickenson2003 = np.array(((0.6,1.4,8.26,0.08,0.08),(1.4,2.0,7.86,0.22,0.33),
                     (2.0,2.5,7.58,0.29,0.54),(2.5,3.0,7.52,0.51,0.48)),float)
    drory2005 = np.array(((0.25,0.75,8.3,0.15,0.15),(0.75,1.25,8.16,0.15,0.15),
                (1.25,1.75,8.0,0.16,0.16),(1.75,2.25,7.85,0.2,0.2),
                (2.25,3.0,7.75,0.2,0.2),(3.0,4.0,7.58,0.2,0.2)),float)
    PerezGonzalez2008 = np.array(((0.2,0.4,8.41,0.06,0.06),(0.4,0.6,8.37,0.04,0.04),
             (0.6,0.8,8.32,0.05,0.05),(0.8,1.0,8.24,0.05,0.05),
             (1.0,1.3,8.15,0.05,0.05),(1.3,1.6,7.95,0.07,0.07),
             (1.6,2.0,7.82,0.07,0.07),(2.0,2.5,7.67,0.08,0.08),
             (2.5,3.0,7.56,0.18,0.18),(3.0,3.5,7.43,0.14,0.14),
             (3.5,4.0,7.29,0.13,0.13)),float)
    glazebrook2004 = np.array(((0.8,1.1,7.98,0.14,0.1),(1.1,1.3,7.62,0.14,0.11),
                     (1.3,1.6,7.9,0.14,0.14),(1.6,2.0,7.49,0.14,0.12)),float)
    fontana2006 = np.array(((0.4,0.6,8.26,0.03,0.03),(0.6,0.8,8.17,0.02,0.02),
                  (0.8,1.0,8.09,0.03,0.03),(1.0,1.3,7.98,0.02,0.02),
                  (1.3,1.6,7.87,0.05,0.05),(1.6,2.0,7.74,0.04,0.04),
                  (2.0,3.0,7.48,0.04,0.04),(3.0,4.0,7.07,0.15,0.11)),float)
    rudnick2006 = np.array(((0.0,1.0,8.17,0.27,0.05),(1.0,1.6,7.99,0.32,0.05),
                  (1.6,2.4,7.88,0.34,0.09),(2.4,3.2,7.71,0.43,0.08)),float)
    elsner2008 = np.array(((0.25,0.75,8.37,0.03,0.03),(0.75,1.25,8.17,0.02,0.02),
                 (1.25,1.75,8.02,0.03,0.03),(1.75,2.25,7.9,0.04,0.04),
                 (2.25,3.0,7.73,0.04,0.04),(3.0,4.0,7.39,0.05,0.05)),float)
    
    obs = (dickenson2003,drory2005,PerezGonzalez2008,glazebrook2004,fontana2006,rudnick2006,elsner2008)
    whichimf = 1  # 1 = Chabrier
    
    # Define your colors list
    obs_colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'orange', 'purple']
    
    label_added = False
    
    # Use enumerate(obs) to get both the index (i) and the data (o)
    for i, o in enumerate(obs):
        xval = ((o[:,1]-o[:,0])/2.)+o[:,0]
        if(whichimf == 0):
            yval = np.log10(10**o[:,2] *1.6)
        elif(whichimf == 1):
            yval = np.log10(10**o[:,2] *1.6/1.8)

        # Select color safely
        current_color = obs_colors[i % len(obs_colors)]
            
        lbl = 'Observations' if not label_added else None
        
        ax.errorbar(xval, yval, xerr=(xval-o[:,0], o[:,1]-xval), yerr=(o[:,3], o[:,4]), 
                    alpha=0.4, lw=1.0, marker='o', ls='none', label=lbl, 
                    markeredgecolor='k', markeredgewidth=0.8,
                            markerfacecolor = 'gray',
                    color=current_color)
        
        if not label_added: label_added = True

    # --- 3. PROCESS & PLOT MODELS ---
    N_BOOT = 100
    model_results_17 = {}  # Store for quantitative comparison
    for sim in sim_dirs:
        sim_path = sim['path']
        sim_label = sim['label']
        sim_redshifts = sim['redshifts']
        sim_first = sim['first_snap']
        sim_last = sim['last_snap']
        sim_volume = sim['volume']
        sim_mass_convert = sim['mass_convert']
        do_bootstrap = (sim_label == 'SAGE26 (Millennium)')
        n_snaps = sim_last - sim_first + 1
        smd = np.zeros(n_snaps)
        smd_lo = np.zeros(n_snaps)
        smd_hi = np.zeros(n_snaps)

        for snap in range(sim_first, sim_last + 1):
            snap_name = f'Snap_{snap}'
            idx = snap - sim_first

            try:
                model_files = find_model_files(sim_path)
                d = read_snap_from_files(model_files, snap_name,
                                         ['StellarMass'],
                                         mass_convert=sim_mass_convert)
                if d:
                    m_stars = d['StellarMass']

                    # Apply limits 1e8 < M < 1e13 (from uploaded script)
                    w = np.where((m_stars > 1.0e8) & (m_stars < 1.0e13))[0]
                    if len(w) > 0:
                        m_sel = m_stars[w]
                        smd[idx] = np.sum(m_sel) / sim_volume

                        if do_bootstrap:
                            n_gal = len(m_sel)
                            boot = np.array([
                                np.sum(m_sel[np.random.randint(0, n_gal, n_gal)])
                                for _ in range(N_BOOT)
                            ]) / sim_volume
                            smd_lo[idx] = np.percentile(boot, 16)
                            smd_hi[idx] = np.percentile(boot, 84)
            except Exception:
                continue

        # Plot
        nonzero = np.where(smd > 0.0)[0]
        if len(nonzero) > 0:
            z_vals = sim_redshifts[sim_first:sim_last+1]
            ax.plot(z_vals[nonzero], np.log10(smd[nonzero]),
                    lw=sim['lw'], color=sim['color'], linestyle=sim['ls'], label=sim_label)

            if do_bootstrap:
                valid = nonzero[smd_lo[nonzero] > 0]
                if len(valid) > 0:
                    ax.fill_between(z_vals[valid],
                                    np.log10(smd_lo[valid]),
                                    np.log10(smd_hi[valid]),
                                    color=sim['color'], alpha=0.2)

            model_results_17[sim_label] = {
                'z': z_vals[nonzero],
                'smd': np.log10(smd[nonzero]),
            }

    # ===== QUANTITATIVE COMPARISON: SAGE26 vs SAGE16 (SMD) =====
    if 'SAGE26 (Millennium)' in model_results_17 and 'SAGE16' in model_results_17:
        from scipy.interpolate import interp1d as _interp1d
        print("\n" + "="*60)
        print("QUANTITATIVE COMPARISON: SAGE26 vs SAGE16 (SMD)")
        print("="*60)
        _z26 = model_results_17['SAGE26 (Millennium)']['z']
        _s26 = model_results_17['SAGE26 (Millennium)']['smd']
        _zc16 = model_results_17['SAGE16']['z']
        _sc16 = model_results_17['SAGE16']['smd']
        _z_mask = _z26 <= 7.5
        _z26 = _z26[_z_mask];  _s26 = _s26[_z_mask]
        _c16_interp = _interp1d(_zc16, _sc16, bounds_error=False, fill_value=np.nan)
        _sc16_m = _c16_interp(_z26)
        _ok = ~np.isnan(_sc16_m)
        if np.sum(_ok) > 0:
            _zv = _z26[_ok];  _diff = _sc16_m[_ok] - _s26[_ok]
            print(f"\n  Comparison over z = {_zv.min():.1f} to {_zv.max():.1f}")
            print(f"  Mean difference (SAGE16 - SAGE26):  {np.mean(_diff):+.3f} dex")
            print(f"  Median difference:               {np.median(_diff):+.3f} dex")
            print(f"  Std of difference:               {np.std(_diff):.3f} dex")
            print(f"  Max at z={_zv[np.argmax(_diff)]:.1f}: {np.max(_diff):+.3f} dex ({10**np.max(_diff):.1f}x)")
            print(f"  Min at z={_zv[np.argmin(_diff)]:.1f}: {np.min(_diff):+.3f} dex ({10**np.min(_diff):.1f}x)")
            print(f"\n  SMD at specific redshifts:")
            for _tz in [0, 1, 2, 3, 4, 5, 6]:
                _ti = np.argmin(np.abs(_zv - _tz))
                if np.abs(_zv[_ti] - _tz) < 0.5:
                    print(f"    z~{_zv[_ti]:.1f}: SAGE16={_sc16_m[_ok][_ti]:.2f}, SAGE26={_s26[_ok][_ti]:.2f}, "
                          f"Δ={_diff[_ti]:+.2f} dex ({10**_diff[_ti]:.1f}x)")
        print("="*60 + "\n")

    # --- COSMOS-Web ---
    if HAS_ASTROPY:
        smd_file = './data/sfrd/SMD.ecsv'
        if os.path.exists(smd_file):
            try:
                smd_table = Table.read(smd_file, format='ascii.ecsv')
                z_smd = np.array(smd_table['z'])
                rho_50 = np.log10(np.array(smd_table['rho_50']))
                rho_16 = np.log10(np.array(smd_table['rho_16']))
                rho_84 = np.log10(np.array(smd_table['rho_84']))
                ax.plot(z_smd, rho_50, color='darkorange', lw=2,
                        label='COSMOS-Web')
                ax.fill_between(z_smd, rho_16, rho_84,
                                color='orange', alpha=0.3)
            except Exception as e:
                print(f"Error loading SMD data: {e}")

    # --- 4. FORMATTING ---
    ax.set_ylabel(r'$\log_{10}\ \rho_{*}\ (M_{\odot}\ \mathrm{Mpc}^{-3})$')
    ax.set_xlabel(r'$\mathrm{Redshift}$')
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.set_xlim(0.0, 5.0)
    ax.set_ylim(6.0, 9.0)

    fig.tight_layout()

    sim_names = {'SAGE26 (Millennium)', 'SAGE26 (miniUchuu)', 'SAGE16'}
    handles, labels = ax.get_legend_handles_labels()
    sim_h = [h for h, l in zip(handles, labels) if l in sim_names]
    sim_l = [l for l in labels if l in sim_names]
    obs_h = [h for h, l in zip(handles, labels) if l not in sim_names]
    obs_l = [l for l in labels if l not in sim_names]
    leg1 = _standard_legend(ax, loc='upper right', handles=sim_h, labels=sim_l)
    ax.add_artist(leg1)
    _standard_legend(ax, loc='lower left', handles=obs_h, labels=obs_l)

    outputFile = os.path.join(OUTPUT_DIR, 'Stellar_Mass_Density_History_Comparison' + OUTPUT_FORMAT)
    save_figure(fig, outputFile)


# ========================== SMF OBSERVATIONAL DATA LOADER ==========================

def _select_grid_snapshot(redshifts, first_snap, last_snap, z_lo, z_hi,
                          z_ref=None, tol=0.05):
    """
    Snapshot representing the redshift bin [z_lo, z_hi] for one model.

    Without *z_ref*, returns the in-bin snapshot nearest the bin centre.

    With *z_ref* -- the redshift the reference model actually landed on -- returns the
    in-bin snapshot nearest *z_ref* instead. Selecting each model independently against
    the bin centre looks equivalent but is not: the simulations have different snapshot
    tables (Millennium 64 snapshots, miniUchuu 50), so at high redshift the two curves
    end up at genuinely different epochs while the panel is labelled with a single bin.
    Measured on the shipped runs, that put them dz = 0.70 apart in 8.5 < z < 9.5 and
    dz = 0.47 apart in 5.5 < z < 6.5, on opposite sides of the bin centre -- comparable
    to 0.2-0.3 dex of SMF evolution at the massive end, i.e. an offset between the model
    curves that is an artefact of snapshot sampling rather than physics.

    Alignment is not pursued at any cost: candidates within *tol* of the best |z - z_ref|
    count as ties and are broken toward the bin centre, so a snapshot is not dragged away
    from the epoch the panel advertises in exchange for a negligible gain in alignment.
    In 4.5 < z < 5.5 that guard matters -- matching would have improved alignment by 0.01
    while nearly tripling the distance from the bin centre.

    Returns (snap_num, snap_redshift), or (None, None) if the model has no snapshot in
    the bin. Residual offsets that no choice can remove are reported by
    _report_grid_alignment().
    """
    sub = np.asarray(redshifts)[first_snap:last_snap + 1]
    idx = np.where((sub >= z_lo) & (sub <= z_hi))[0]
    if idx.size == 0:
        return None, None
    z_mid = 0.5 * (z_lo + z_hi)
    if z_ref is None:
        pick = idx[np.argmin(np.abs(sub[idx] - z_mid))]
    else:
        d = np.abs(sub[idx] - z_ref)
        near = idx[d <= d.min() + tol]
        pick = near[np.argmin(np.abs(sub[near] - z_mid))]
    return int(pick + first_snap), float(sub[pick])


def _report_grid_alignment(chosen, threshold=0.10):
    """
    Print the redshift each model was drawn at per bin, and flag bins where the models
    are further apart than *threshold* in z.

    Those bins are limited by the coarser snapshot sampling of one simulation, not by the
    selection: the offset cannot be removed without a snapshot that does not exist. They
    are printed so the residual is visible in the log rather than hidden inside a panel
    labelled with a single redshift range.

    *chosen* maps bin label -> list of (model_label, snap_num, snap_redshift).
    """
    print('  snapshot alignment across models:')
    worst = []
    for bin_label, entries in chosen.items():
        if len(entries) < 2:
            continue
        zs = [e[2] for e in entries]
        spread = max(zs) - min(zs)
        detail = ',  '.join(f'{lab}: Snap_{sn} z={zz:.3f}' for lab, sn, zz in entries)
        flag = '   <-- limited by snapshot sampling' if spread > threshold else ''
        print(f'    {bin_label:<14s} spread dz={spread:.3f}   {detail}{flag}')
        if spread > threshold:
            worst.append((bin_label, spread))
    if worst:
        print(f'    {len(worst)} of {len(chosen)} bins exceed dz = {threshold:g}: '
              + ', '.join(f'{b} ({d:.2f})' for b, d in worst))
    else:
        print(f'    all bins aligned to within dz = {threshold:g}')



def _load_smf_grid_observations():
    """
    Load all observational SMF datasets for the redshift grid plot.

    Returns list of dicts with keys:
        z, log_mass, log_phi, err_lo, err_hi, label, marker, ms
    err_lo / err_hi are positive offsets in dex (for errorbar yerr).
    They may be None when errors are unavailable.
    """
    obs = []
    h = HUBBLE_H  # 0.73

    # ------------------------------------------------------------------
    # 1. Baldry+08  (z ~ 0, inline data)
    # ------------------------------------------------------------------
    try:
        _baldry = np.array([
            [7.05,1.3531e-01,6.0741e-02],[7.15,1.3474e-01,6.0109e-02],
            [7.25,2.0971e-01,7.7965e-02],[7.35,1.7161e-01,3.1841e-02],
            [7.45,2.1648e-01,5.7832e-02],[7.55,2.1645e-01,3.9988e-02],
            [7.65,2.0837e-01,4.8713e-02],[7.75,2.0402e-01,7.0061e-02],
            [7.85,1.5536e-01,3.9182e-02],[7.95,1.5232e-01,2.6824e-02],
            [8.05,1.5067e-01,4.8824e-02],[8.15,1.3032e-01,2.1892e-02],
            [8.25,1.2545e-01,3.5526e-02],[8.35,9.8472e-02,2.7181e-02],
            [8.45,8.7194e-02,2.8345e-02],[8.55,7.0758e-02,2.0808e-02],
            [8.65,5.8190e-02,1.3359e-02],[8.75,5.6057e-02,1.3512e-02],
            [8.85,5.1380e-02,1.2815e-02],[8.95,4.4206e-02,9.6866e-03],
            [9.05,4.1149e-02,1.0169e-02],[9.15,3.4959e-02,6.7898e-03],
            [9.25,3.3111e-02,8.3704e-03],[9.35,3.0138e-02,4.7741e-03],
            [9.45,2.6692e-02,5.5029e-03],[9.55,2.4656e-02,4.4359e-03],
            [9.65,2.2885e-02,3.7915e-03],[9.75,2.1849e-02,3.9812e-03],
            [9.85,2.0383e-02,3.2930e-03],[9.95,1.9929e-02,2.9370e-03],
            [10.05,1.8865e-02,2.4624e-03],[10.15,1.8136e-02,2.5208e-03],
            [10.25,1.7657e-02,2.4217e-03],[10.35,1.6616e-02,2.2784e-03],
            [10.45,1.6114e-02,2.1783e-03],[10.55,1.4366e-02,1.8819e-03],
            [10.65,1.2588e-02,1.8249e-03],[10.75,1.1372e-02,1.4436e-03],
            [10.85,9.1213e-03,1.5816e-03],[10.95,6.1125e-03,9.6735e-04],
            [11.05,4.3923e-03,9.6254e-04],[11.15,2.5463e-03,5.0038e-04],
            [11.25,1.4298e-03,4.2816e-04],[11.35,6.4867e-04,1.6439e-04],
            [11.45,2.8294e-04,9.9799e-05],[11.55,1.0617e-04,4.9085e-05],
            [11.65,3.2702e-05,2.4546e-05],[11.75,1.2571e-05,1.2571e-05],
            [11.85,8.4589e-06,8.4589e-06],[11.95,7.4764e-06,7.4764e-06],
        ], dtype=np.float32)
        log_m = np.log10(10**_baldry[:, 0] / h / h) - 0.26  # h^-2 + Chabrier
        phi_c = _baldry[:, 1] * h**3
        phi_u = (_baldry[:, 1] + _baldry[:, 2]) * h**3
        phi_l = (_baldry[:, 1] - _baldry[:, 2]) * h**3
        ok = phi_l > 0
        lp = np.log10(phi_c[ok])
        obs.append({'z': 0.05, 'log_mass': log_m[ok], 'log_phi': lp,
                     'err_lo': lp - np.log10(phi_l[ok]),
                     'err_hi': np.log10(phi_u[ok]) - lp,
                     'label': 'Baldry+08', 'marker': 'o', 'ms': 8})
    except Exception as e:
        print(f"  Baldry+08 load error: {e}")

    # ------------------------------------------------------------------
    # 2. Thorne+21  (GAMA/DEVILS; CSV: logM, phi, phi_16, phi_84 — linear)
    #    Loaded for z=1.6–4.0 only; lower-z bins are already well-covered
    #    by Baldry+08, Muzzin+13, and SMF_data_points.
    # ------------------------------------------------------------------
    _thorne = [
        ('./data/smf/Thorne21/SMFvals_z1.6.csv', 1.6),
        ('./data/smf/Thorne21/SMFvals_z2.csv',   2.0),
        ('./data/smf/Thorne21/SMFvals_z2.4.csv', 2.4),
        ('./data/smf/Thorne21/SMFvals_z3.csv',   3.0),
        ('./data/smf/Thorne21/SMFvals_z3.5.csv', 3.5),
        ('./data/smf/Thorne21/SMFvals_z4.csv',   4.0),
    ]
    h_t = 0.7
    for fpath, z_val in _thorne:
        try:
            if not os.path.exists(fpath):
                continue
            d = np.genfromtxt(fpath, delimiter=',', skip_header=1)
            if d.ndim != 2 or d.shape[1] < 4:
                continue
            m, phi, p16, p84 = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
            ok = np.isfinite(m) & (phi > 0) & (p16 > 0) & (p84 > 0)
            if not np.any(ok):
                continue
            phi_c  = phi[ok]  * (h_t / h)**3
            p16_c  = p16[ok]  * (h_t / h)**3
            p84_c  = p84[ok]  * (h_t / h)**3
            lp = np.log10(phi_c)
            obs.append({'z': z_val, 'log_mass': m[ok], 'log_phi': lp,
                         'err_lo': lp - np.log10(p16_c),
                         'err_hi': np.log10(p84_c) - lp,
                         'label': 'Thorne+21', 'marker': 's', 'ms': 6})
        except Exception as e:
            print(f"  Thorne+21 z={z_val} load error: {e}")

    # ------------------------------------------------------------------
    # 3. Weaver+23  (farmer TXT: logM, bw, phi, phi_lo, phi_hi — linear)
    # ------------------------------------------------------------------
    _weaver = [
        ('./data/smf/COSMOS2020/SMF_Farmer_v2.1_1.5z2.0_total.txt', 1.75),
        ('./data/smf/COSMOS2020/SMF_Farmer_v2.1_2.0z2.5_total.txt', 2.25),
        ('./data/smf/COSMOS2020/SMF_Farmer_v2.1_2.5z3.0_total.txt', 2.75),
        ('./data/smf/COSMOS2020/SMF_Farmer_v2.1_3.0z3.5_total.txt', 3.25),
        ('./data/smf/COSMOS2020/SMF_Farmer_v2.1_3.5z4.5_total.txt', 4.0),
        ('./data/smf/COSMOS2020/SMF_Farmer_v2.1_4.5z5.5_total.txt', 5.0),
        ('./data/smf/COSMOS2020/SMF_Farmer_v2.1_5.5z6.5_total.txt', 6.0),
        ('./data/smf/COSMOS2020/SMF_Farmer_v2.1_6.5z7.5_total.txt', 7.0),
    ]
    for fpath, z_val in _weaver:
        try:
            if not os.path.exists(fpath):
                continue
            d = None
            for delim in [None, ',', '\t', ' ']:
                for skip in [0, 1, 2]:
                    try:
                        d = np.genfromtxt(fpath, delimiter=delim, skip_header=skip)
                        if d.ndim == 2 and d.shape[1] >= 5:
                            break
                    except Exception:
                        d = None
                if d is not None and d.ndim == 2:
                    break
            if d is None or d.ndim != 2 or d.shape[1] < 5:
                continue
            m, phi, plo, phi_hi = d[:, 0], d[:, 2], d[:, 3], d[:, 4]
            ok = np.isfinite(m) & (phi > 0) & (plo > 0) & (phi_hi > 0)
            if np.any(ok):
                lp = np.log10(phi[ok])
                obs.append({'z': z_val, 'log_mass': m[ok], 'log_phi': lp,
                             'err_lo': lp - np.log10(plo[ok]),
                             'err_hi': np.log10(phi_hi[ok]) - lp,
                             'label': 'Weaver+23', 'marker': 'D', 'ms': 8})
        except Exception as e:
            print(f"  Weaver+23 z={z_val} load error: {e}")

    # ------------------------------------------------------------------
    # 4. Muzzin+13  (dat: z_lo z_hi M_star E_M logPhi)
    # ------------------------------------------------------------------
    try:
        _muz_file = './data/smf/SMF_Muzzin2013.dat'
        if os.path.exists(_muz_file):
            h_m = 0.7
            bins = {}
            with open(_muz_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    p = line.split()
                    if len(p) < 5:
                        continue
                    zl, zh, ms, _, lp = float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
                    if lp < -10:
                        continue
                    key = (zl, zh)
                    if key not in bins:
                        bins[key] = {'m': [], 'lp': []}
                    phi_c = 10**lp * (h_m / h)**3
                    bins[key]['m'].append(ms - 0.04)  # Kroupa→Chabrier
                    bins[key]['lp'].append(np.log10(phi_c))
            for (zl, zh), v in bins.items():
                m_arr = np.array(v['m'])
                lp_arr = np.array(v['lp'])
                obs.append({'z': 0.5*(zl+zh), 'log_mass': m_arr, 'log_phi': lp_arr,
                             'err_lo': None, 'err_hi': None,
                             'label': 'Muzzin+13', 'marker': '^', 'ms': 8})
    except Exception as e:
        print(f"  Muzzin+13 load error: {e}")

    # ------------------------------------------------------------------
    # 5. Santini+12  (dat: z_lo z_hi lg_mass lg_phi err_hi err_lo ...)
    # ------------------------------------------------------------------
    try:
        _san_file = './data/smf/SMF_Santini2012.dat'
        if os.path.exists(_san_file):
            h_s = 0.7
            bins = {}
            with open(_san_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    p = line.split()
                    if len(p) < 6:
                        continue
                    zl, zh = float(p[0]), float(p[1])
                    lg_m, lg_p = float(p[2]), float(p[3])
                    ehi, elo = float(p[4]), float(p[5])
                    if lg_p < -10 or not np.isfinite(lg_p):
                        continue
                    key = (zl, zh)
                    if key not in bins:
                        bins[key] = {'m': [], 'lp': [], 'ehi': [], 'elo': []}
                    phi_c = 10**lg_p * (h_s / h)**3
                    bins[key]['m'].append(lg_m + SALPETER_TO_CHABRIER_DEX)  # Salpeter→Chabrier
                    bins[key]['lp'].append(np.log10(phi_c))
                    bins[key]['ehi'].append(ehi)
                    bins[key]['elo'].append(elo)
            for (zl, zh), v in bins.items():
                obs.append({'z': 0.5*(zl+zh), 'log_mass': np.array(v['m']),
                             'log_phi': np.array(v['lp']),
                             'err_lo': np.array(v['elo']),
                             'err_hi': np.array(v['ehi']),
                             'label': 'Santini+12', 'marker': 'v', 'ms': 8})
    except Exception as e:
        print(f"  Santini+12 load error: {e}")

    # ------------------------------------------------------------------
    # 6. Wright+18  (dat: med_z mass log_y dlog_yu dlog_yd ycv)
    # ------------------------------------------------------------------
    try:
        _wr_file = './data/smf/Wright18_CombinedSMF.dat'
        if os.path.exists(_wr_file):
            h_w = 0.7
            bins = {}
            with open(_wr_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    p = line.split()
                    if len(p) < 6:
                        continue
                    mz, sm, ly = float(p[0]), float(p[1]), float(p[2])
                    dyu, dyd = float(p[3]), float(p[4])
                    if ly < -10 or not np.isfinite(ly):
                        continue
                    ly_corr = ly + np.log10(1.0 / 0.25)  # bin-width correction
                    phi_c = 10**ly_corr * (h_w / h)**3
                    if mz not in bins:
                        bins[mz] = {'m': [], 'lp': [], 'ehi': [], 'elo': []}
                    bins[mz]['m'].append(sm)
                    bins[mz]['lp'].append(np.log10(phi_c))
                    bins[mz]['ehi'].append(dyu)
                    bins[mz]['elo'].append(dyd)
            for z_val, v in bins.items():
                obs.append({'z': z_val, 'log_mass': np.array(v['m']),
                             'log_phi': np.array(v['lp']),
                             'err_lo': np.array(v['elo']),
                             'err_hi': np.array(v['ehi']),
                             'label': 'Wright+18', 'marker': 'p', 'ms': 8})
    except Exception as e:
        print(f"  Wright+18 load error: {e}")

    # ------------------------------------------------------------------
    # 7. Observational compilation  (SMF_data_points.ecsv)
    # ------------------------------------------------------------------
    if HAS_ASTROPY:
        try:
            _obs_file = './data/smf/SMF_data_points.ecsv'
            if os.path.exists(_obs_file):
                with open(_obs_file, 'r') as f:
                    lines = f.readlines()
                data_lines = [l.strip() for l in lines
                              if not l.startswith('#') and l.strip()]
                bins = {}
                for line in data_lines[1:]:
                    if '"' not in line:
                        continue
                    parts = line.split('"')
                    zbin_str = parts[1]
                    dp = parts[2].strip().split()
                    if len(dp) < 3:
                        continue
                    ms, phi, dphi = float(dp[0]), float(dp[1]), float(dp[2])
                    # Parse z_center from string like "0.2 < z < 0.5"
                    nums = [float(s) for s in zbin_str.replace('<', ' ').replace('z', ' ').split()
                            if s.replace('.', '', 1).replace('-', '', 1).isdigit()]
                    if len(nums) >= 2:
                        zc = 0.5 * (nums[0] + nums[-1])
                    else:
                        continue
                    if zc not in bins:
                        bins[zc] = {'m': [], 'phi': [], 'dphi': []}
                    bins[zc]['m'].append(ms)
                    bins[zc]['phi'].append(phi)
                    bins[zc]['dphi'].append(dphi)
                for zc, v in bins.items():
                    m_arr = np.array(v['m'])
                    phi_arr = np.array(v['phi'])
                    dphi_arr = np.array(v['dphi'])
                    ok = phi_arr > 0
                    if np.any(ok):
                        lp = np.log10(phi_arr[ok])
                        upper = phi_arr[ok] + dphi_arr[ok]
                        lower = phi_arr[ok] - dphi_arr[ok]
                        ehi = np.where(upper > 0, np.log10(upper) - lp, 0.0)
                        elo = np.where(lower > 0, lp - np.log10(lower), 0.0)
                        obs.append({'z': zc, 'log_mass': m_arr[ok], 'log_phi': lp,
                                     'err_lo': elo, 'err_hi': ehi,
                                     'label': 'COSMOS-Web', 'marker': 'h', 'ms': 8})
        except Exception as e:
            print(f"  COSMOS-Web load error: {e}")

    # ------------------------------------------------------------------
    # 8. Harvey+25  (ECSV: z, log10Mstar, phi, phi_error_low, phi_error_upp)
    # ------------------------------------------------------------------
    if HAS_ASTROPY:
        try:
            _har_file = './data/smf/FiducialBagpipesGSMF.ecsv'
            if os.path.exists(_har_file):
                t = Table.read(_har_file, format='ascii.ecsv')
                # Handle possible column name variations
                _pcol_lo = ([c for c in t.colnames if 'low' in c and 'phi' in c] + ['phi_error_low'])[0]
                _pcol_hi = ([c for c in t.colnames if 'upp' in c and 'phi' in c] + ['phi_error_upp'])[0]
                for z_val in np.unique(t['z']):
                    mask = t['z'] == z_val
                    s = t[mask]
                    log_m = np.array(s['log10Mstar'])
                    phi = np.array(s['phi'])
                    phi_elo = np.array(s[_pcol_lo])
                    phi_ehi = np.array(s[_pcol_hi])
                    ok = phi > 0
                    if np.any(ok):
                        lp = np.log10(phi[ok])
                        upper = phi[ok] + phi_ehi[ok]
                        lower = phi[ok] - phi_elo[ok]
                        ehi = np.where(upper > 0, np.log10(upper) - lp, 0.0)
                        elo = np.where(lower > 0, lp - np.log10(lower), 0.0)
                        obs.append({'z': float(z_val), 'log_mass': log_m[ok], 'log_phi': lp,
                                     'err_lo': elo, 'err_hi': ehi,
                                     'label': 'Harvey+25', 'marker': 'H', 'ms': 8})
        except Exception as e:
            print(f"  Harvey+25 load error: {e}")

    # ------------------------------------------------------------------
    # 9–12.  High-z ECSV datasets (Stefanon+21, Navarro-Carrera+23,
    #         Weibel+24, Kikuchihara+20)
    # ------------------------------------------------------------------
    _highz_ecsv = [
        {'file': './data/smf/stefanon_smf_2021.ecsv',
         'label': 'Stefanon+21', 'marker': '*', 'ms': 8,
         'zcol': 'redshift_bin', 'mcol': 'log_M',
         'phi_col': 'phi', 'phi_eu': 'phi_err_up', 'phi_el': 'phi_err_low',
         'phi_scale': 1e-4, 'phi_log': False, 'zbins': [6, 7, 8, 9, 10]},
        {'file': './data/smf/navarro_carrera_smf_2023.ecsv',
         'label': 'Navarro-Carrera+23', 'marker': 'X', 'ms': 8,
         'zcol': 'redshift_bin', 'mcol': 'log_M',
         'phi_col': 'phi', 'phi_eu': 'phi_err_up', 'phi_el': 'phi_err_low',
         'phi_scale': 1e-4, 'phi_log': False, 'zbins': [6, 7, 8]},
        {'file': './data/smf/weibel_smf_2024.ecsv',
         'label': 'Weibel+24', 'marker': 'P', 'ms': 8,
         'zcol': 'redshift_bin', 'mcol': 'log_M',
         'phi_col': 'log_phi', 'phi_eu': 'log_phi_err_up', 'phi_el': 'log_phi_err_low',
         'phi_scale': 1.0, 'phi_log': True, 'zbins': [6, 7, 8, 9]},
        {'file': './data/smf/kikuchihara_smf_2020.ecsv',
         'label': 'Kikuchihara+20', 'marker': 'd', 'ms': 8,
         'zcol': 'redshift_approx', 'mcol': 'log_M_star',
         'phi_col': 'phi_star', 'phi_eu': 'phi_star_err_up', 'phi_el': 'phi_star_err_low',
         'phi_scale': 1e-5, 'phi_log': False, 'zbins': [6, 7, 8, 9]},
    ]
    if HAS_ASTROPY:
        for cfg in _highz_ecsv:
            try:
                if not os.path.exists(cfg['file']):
                    continue
                t = Table.read(cfg['file'], format='ascii.ecsv')
                for zb in cfg['zbins']:
                    zm = t[cfg['zcol']] == zb
                    if not np.any(zm):
                        continue
                    s = t[zm]
                    log_m = np.array(s[cfg['mcol']])
                    if cfg['phi_log']:
                        lp = np.array(s[cfg['phi_col']])
                        eu = np.array(s[cfg['phi_eu']])
                        el = np.array(s[cfg['phi_el']])
                        lp_hi = lp + eu
                        lp_lo = lp - el
                        lp_lo[el == 0] = np.nan
                        ok = np.isfinite(lp)
                        obs.append({'z': float(zb), 'log_mass': log_m[ok],
                                     'log_phi': lp[ok],
                                     'err_lo': np.where(np.isfinite(lp_lo[ok]),
                                                        lp[ok] - lp_lo[ok], 0.0),
                                     'err_hi': np.where(np.isfinite(lp_hi[ok]),
                                                        lp_hi[ok] - lp[ok], 0.0),
                                     'label': cfg['label'], 'marker': cfg['marker'],
                                     'ms': cfg['ms']})
                    else:
                        phi_lin = np.array(s[cfg['phi_col']], dtype=float) * cfg['phi_scale']
                        eu_lin = np.array(s[cfg['phi_eu']], dtype=float) * cfg['phi_scale']
                        el_lin = np.array(s[cfg['phi_el']], dtype=float) * cfg['phi_scale']
                        ok = phi_lin > 0
                        if not np.any(ok):
                            continue
                        lp = np.log10(phi_lin[ok])
                        upper = phi_lin[ok] + eu_lin[ok]
                        lower = phi_lin[ok] - el_lin[ok]
                        ehi = np.where(upper > 0, np.log10(upper) - lp, 0.0)
                        elo = np.where(lower > 0, lp - np.log10(lower), 0.0)
                        obs.append({'z': float(zb), 'log_mass': log_m[ok],
                                     'log_phi': lp, 'err_lo': elo, 'err_hi': ehi,
                                     'label': cfg['label'], 'marker': cfg['marker'],
                                     'ms': cfg['ms']})
            except Exception as e:
                print(f"  {cfg['label']} load error: {e}")

    # ------------------------------------------------------------------
    # 13. Song+16  (CANDELS; wide ECSV: log_M + log_phi columns per z)
    #     Covers z=4–8; h=0.7, Chabrier IMF.
    # ------------------------------------------------------------------
    if HAS_ASTROPY:
        try:
            _song_file = './data/smf/song_smf_2016.ecsv'
            if os.path.exists(_song_file):
                t = Table.read(_song_file, format='ascii.ecsv')
                h_s = 0.7
                log_phi_corr = 3.0 * np.log10(h_s / h)
                log_m_song = np.array(t['log_M'], dtype=float)
                _song_bins = [
                    (4, 'phi_z4', 'phi_z4_err_up', 'phi_z4_err_lo'),
                    (5, 'phi_z5', 'phi_z5_err_up', 'phi_z5_err_lo'),
                    (6, 'phi_z6', 'phi_z6_err_up', 'phi_z6_err_lo'),
                    (7, 'phi_z7', 'phi_z7_err_up', 'phi_z7_err_lo'),
                    (8, 'phi_z8', 'phi_z8_err_up', 'phi_z8_err_lo'),
                ]
                for z_val, pcol, eu_col, el_col in _song_bins:
                    lp  = np.array(t[pcol],  dtype=float)
                    eu  = np.array(t[eu_col], dtype=float)
                    el  = np.array(t[el_col], dtype=float)
                    ok  = np.isfinite(lp) & np.isfinite(eu) & np.isfinite(el)
                    if not np.any(ok):
                        continue
                    obs.append({'z': float(z_val),
                                'log_mass': log_m_song[ok],
                                'log_phi':  lp[ok] + log_phi_corr,
                                'err_lo': el[ok], 'err_hi': eu[ok],
                                'label': 'Song+16', 'marker': '<', 'ms': 6})
        except Exception as e:
            print(f"  Song+16 load error: {e}")

    print(f"  Loaded {len(obs)} observational SMF datasets")
    return obs


# ========================== PLOT 18: SMF REDSHIFT GRID ==========================

def plot_18_smf_redshift_grid():
    """
    Plot: 3x5 grid of Stellar Mass Functions at 15 redshift bins.
    Each panel shows the SMF for SAGE26 (Millennium) and SAGE26 (miniUchuu).
    """
    print('Plot 18: SMF Redshift Grid')

    # Redshift bins: (z_lo, z_hi)
    z_bins = [
        (0.0, 0.5),   (0.5, 0.8),   (0.8, 1.1),
        (1.1, 1.5),   (1.5, 2.0),   (2.0, 2.5),
        (2.5, 3.0),   (3.0, 3.5),   (3.5, 4.5),
        (4.5, 5.5),   (5.5, 6.5),   (6.5, 7.5),
        (7.5, 8.5),   (8.5, 9.5),   (9.5, 12.0),
    ]

    # Models to plot
    mill_redshifts = np.array(REDSHIFTS)
    mu_redshifts = np.array(MINIUCHUU_REDSHIFTS)

    models = []
    if os.path.exists(PRIMARY_DIR):
        models.append({
            'path': PRIMARY_DIR, 'label': 'SAGE26 (Millennium)',
            'color': 'black', 'ls': '-', 'lw': 3.5,
            'redshifts': mill_redshifts, 'first_snap': 0, 'last_snap': 63,
            'volume': VOLUME, 'mass_convert': MASS_CONVERT,
        })
    if os.path.exists(MINIUCHUU_DIR):
        models.append({
            'path': MINIUCHUU_DIR, 'label': 'SAGE26 (miniUchuu)',
            'color': 'steelblue', 'ls': '--', 'lw': 2.5,
            'redshifts': mu_redshifts, 'first_snap': MINIUCHUU_FIRST_SNAP, 'last_snap': MINIUCHUU_LAST_SNAP,
            'volume': MINIUCHUU_VOLUME, 'mass_convert': MINIUCHUU_MASS_CONVERT,
        })
    if os.path.exists(VANILLA_DIR):
        models.append({
            'path': VANILLA_DIR, 'label': 'SAGE16',
            'color': 'firebrick', 'ls': '--', 'lw': 2.5,
            'redshifts': mill_redshifts, 'first_snap': 0, 'last_snap': 63,
            'volume': VOLUME, 'mass_convert': MASS_CONVERT,
        })

    # Load observational data
    all_obs = _load_smf_grid_observations()
    labels_used = set()  # track legend entries to avoid duplicates

    fig, axes = plt.subplots(5, 3, figsize=(15, 25), sharex=True, sharey=True)
    fig.set_tight_layout(False)
    axes_flat = axes.flatten()
    binwidth = 0.1
    _bin18_smf = {}  # store per-bin SMF for comparison printout
    _alignment = {}  # bin -> [(model, snap, z)] for the alignment report

    for i, (z_lo, z_hi) in enumerate(z_bins):
        ax = axes_flat[i]
        z_mid = 0.5 * (z_lo + z_hi)

        # The first model listed sets the epoch for the panel; the rest are matched to
        # the redshift it actually landed on, not to the bin centre, so overlaid curves
        # are compared at the same epoch. See _select_grid_snapshot().
        z_ref_panel = None
        for model in models:
            snap_num, snap_z = _select_grid_snapshot(
                model['redshifts'], model['first_snap'], model['last_snap'],
                z_lo, z_hi, z_ref=z_ref_panel)
            if snap_num is None:
                continue
            if z_ref_panel is None:
                z_ref_panel = snap_z
            _alignment.setdefault(f'{z_lo:.1f}-{z_hi:.1f}', []).append(
                (model['label'], snap_num, snap_z))
            snap_name = f'Snap_{snap_num}'

            try:
                model_files = find_model_files(model['path'])
                d = read_snap_from_files(model_files, snap_name,
                                         ['StellarMass'],
                                         mass_convert=model['mass_convert'])
                if not d:
                    continue
                m_stars = d['StellarMass']
                w = m_stars > 0
                if np.sum(w) == 0:
                    continue
                log_m = np.log10(m_stars[w])

                    # Use bootstrap for SAGE26 models
                if model['label'].startswith('SAGE26'):
                    x, phi, phi_lo, phi_hi, _ = mass_function_bootstrap(
                        log_m, model['volume'], binwidth, n_boot=100)
                    valid = np.isfinite(phi)
                    ax.plot(x[valid], phi[valid],
                            lw=model['lw'], color=model['color'],
                            ls=model['ls'],
                            label=model['label'] if i == 0 else None)
                    # Bootstrap shading
                    boot_valid = np.isfinite(phi_lo) & np.isfinite(phi_hi)
                    if np.any(boot_valid):
                        ax.fill_between(x[boot_valid], phi_lo[boot_valid], phi_hi[boot_valid],
                                        color=model['color'], alpha=0.2, linewidth=0)
                else:
                    x, phi, _ = mass_function(log_m, model['volume'], binwidth)
                    valid = np.isfinite(phi)
                    ax.plot(x[valid], phi[valid],
                            lw=model['lw'], color=model['color'],
                            ls=model['ls'],
                            label=model['label'] if i == 0 else None)
                _bin18_smf.setdefault(i, {})[model['label']] = (x[valid], phi[valid])
            except Exception as e:
                print(f"  Error loading {snap_name} from {model['path']}: {e}")
                continue

        # Print model comparisons for this z-bin
        _b18 = _bin18_smf.get(i, {})
        _ref18 = 'SAGE26 (Millennium)'
        if _ref18 in _b18 and len(_b18) > 1:
            _x26, _p26 = _b18[_ref18]
            print(f"\n  SMF z=[{z_lo:.1f},{z_hi:.1f}] (SAGE26 Millennium vs others):")
            for _olab, (_xo, _po) in _b18.items():
                if _olab == _ref18 or len(_xo) == 0 or len(_x26) == 0:
                    continue
                _xc = np.round(np.intersect1d(np.round(_x26, 4), np.round(_xo, 4)), 4)
                if len(_xc) == 0:
                    continue
                _ip18 = np.array([np.argmin(np.abs(_x26 - xc)) for xc in _xc])
                _io18 = np.array([np.argmin(np.abs(_xo - xc)) for xc in _xc])
                _dv = _p26[_ip18] - _po[_io18]
                _ok18 = np.isfinite(_dv)
                if np.sum(_ok18) == 0:
                    continue
                _dv_ok = _dv[_ok18];  _xv18 = _xc[_ok18]
                print(f"    vs {_olab}: Mean Δ={np.mean(_dv_ok):+.3f} dex, "
                      f"Median={np.median(_dv_ok):+.3f} dex, "
                      f"Max at log10(M*)={_xv18[np.argmax(_dv_ok)]:.2f}: {np.max(_dv_ok):+.3f} dex")
                for _tm in [9.0, 10.0, 10.5, 11.0]:
                    _ti = np.argmin(np.abs(_xv18 - _tm))
                    if np.abs(_xv18[_ti] - _tm) < binwidth:
                        print(f"      log10(M*)={_xv18[_ti]:.2f}: SAGE26={_p26[_ip18][_ok18][_ti]:.2f}, "
                              f"{_olab}={_po[_io18][_ok18][_ti]:.2f}, Δ={_dv_ok[_ti]:+.2f} dex")

        # Plot observational data for this redshift bin
        for od in all_obs:
            z_obs = od['z']
            # Match obs to bin: inclusive lower, exclusive upper (last bin inclusive)
            if i == len(z_bins) - 1:
                in_bin = z_lo <= z_obs <= z_hi
            else:
                in_bin = z_lo <= z_obs < z_hi
            if not in_bin:
                continue
            lbl = od['label'] if od['label'] not in labels_used else None
            if lbl is not None:
                labels_used.add(od['label'])
            yerr = None
            if od['err_lo'] is not None and od['err_hi'] is not None:
                yerr = [od['err_lo'], od['err_hi']]
            ax.errorbar(od['log_mass'], od['log_phi'], yerr=yerr,
                        fmt=od['marker'], color='grey', ms=od['ms'],
                        markeredgecolor='k', markeredgewidth=0.8,
                            markerfacecolor = 'gray',
                        alpha=0.6, lw=1.5, capsize=1.5, label=lbl, zorder=1)

        # Redshift label in each panel
        ax.text(0.95, 0.95, rf'${z_lo:.1f} < z < {z_hi:.1f}$',
                transform=ax.transAxes, ha='right', va='top')

    # Axis limits and labels
    axes_flat[0].set_xlim(8.001, 12.2)
    axes_flat[0].set_ylim(-6, -0.8)

    for i, ax in enumerate(axes_flat):
        row, col = divmod(i, 3)
        if col == 0:
            ax.set_ylabel(r'$\log_{10}\ \phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$')
        if row == 4:
            ax.set_xlabel(r'$\log_{10}\ m_{\mathrm{*}}\ [M_{\odot}]$')
        # Re-apply tick style from stylesheet (sharex/sharey overrides these)
        ax.tick_params(axis='both', which='both', direction='in',
                       top=True, bottom=True, left=True, right=True)

    # Per-panel legends (only panels with new labelled entries get a legend)
    for ax in axes_flat:
        _, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc='lower left', frameon=False)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.001, wspace=0.001)

    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(2.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))

    _report_grid_alignment(_alignment)

    outputFile = os.path.join(OUTPUT_DIR, 'SMF_Redshift_Grid' + OUTPUT_FORMAT)
    save_figure(fig, outputFile)


# ======================== PLOT 18b: SMF REDSHIFT GRID (5-col) ========================

def plot_18b_smf_redshift_grid_wide():
    """
    Plot: 3x5 (rows x cols) grid of Stellar Mass Functions at 15 redshift bins.
    Wide version of Plot 18 with x-axis labels on every panel.
    """
    print('Plot 18b: SMF Redshift Grid (wide)')

    # Redshift bins: (z_lo, z_hi)
    z_bins = [
        (0.0, 0.5),   (0.5, 0.8),   (0.8, 1.1),
        (1.1, 1.5),   (1.5, 2.0),   (2.0, 2.5),
        (2.5, 3.0),   (3.0, 3.5),   (3.5, 4.5),
        (4.5, 5.5),   (5.5, 6.5),   (6.5, 7.5),
        (7.5, 8.5),   (8.5, 9.5),   (9.5, 12.0),
    ]

    # Models to plot
    mill_redshifts = np.array(REDSHIFTS)
    mu_redshifts = np.array(MINIUCHUU_REDSHIFTS)

    models = []
    if os.path.exists(PRIMARY_DIR):
        models.append({
            'path': PRIMARY_DIR, 'label': 'SAGE26 (Millennium)',
            'color': 'black', 'ls': '-', 'lw': 4.5,
            'redshifts': mill_redshifts, 'first_snap': 0, 'last_snap': 63,
            'volume': VOLUME, 'mass_convert': MASS_CONVERT,
        })
    if os.path.exists(MINIUCHUU_DIR):
        models.append({
            'path': MINIUCHUU_DIR, 'label': 'SAGE26 (miniUchuu)',
            'color': 'steelblue', 'ls': '--', 'lw': 2.5,
            'redshifts': mu_redshifts, 'first_snap': MINIUCHUU_FIRST_SNAP, 'last_snap': MINIUCHUU_LAST_SNAP,
            'volume': MINIUCHUU_VOLUME, 'mass_convert': MINIUCHUU_MASS_CONVERT,
        })
    if os.path.exists(VANILLA_DIR):
        models.append({
            'path': VANILLA_DIR, 'label': 'SAGE16',
            'color': 'firebrick', 'ls': '--', 'lw': 4.5,
            'redshifts': mill_redshifts, 'first_snap': 0, 'last_snap': 63,
            'volume': VOLUME, 'mass_convert': MASS_CONVERT,
        })
    if os.path.exists(NOFFB_DIR):
        models.append({
            'path': NOFFB_DIR, 'label': 'SAGE26 (no FFBs)',
            'color': 'darkorange', 'ls': ':', 'lw': 2.5,
            'redshifts': mill_redshifts, 'first_snap': 0, 'last_snap': 63,
            'volume': VOLUME, 'mass_convert': MASS_CONVERT,
        })
    # if os.path.exists(C16_FEEDBACK_DIR):
    #     models.append({
    #         'path': C16_FEEDBACK_DIR, 'label': 'SAGE26 (no FIRE)',
    #         'color': 'purple', 'ls': '-.', 'lw': 2.5,
    #         'redshifts': mill_redshifts, 'first_snap': 0, 'last_snap': 63,
    #         'volume': VOLUME, 'mass_convert': MASS_CONVERT,
    #     })
    # if os.path.exists(NOCGM_DIR):
    #     models.append({
    #         'path': NOCGM_DIR, 'label': 'SAGE26 (no CGM)',
    #         'color': 'seagreen', 'ls': ':', 'lw': 2.5,
    #         'redshifts': mill_redshifts, 'first_snap': 0, 'last_snap': 63,
    #         'volume': VOLUME, 'mass_convert': MASS_CONVERT,
    #     })
    # if os.path.exists(CGM_DYN_DIR):
    #     models.append({
    #         'path': CGM_DYN_DIR, 'label': 'SAGE26 (CGM Dyn Time)',
    #         'color': 'darkorange', 'ls': '-.', 'lw': 3.5,
    #         'redshifts': mill_redshifts, 'first_snap': 0, 'last_snap': 63,
    #         'volume': VOLUME, 'mass_convert': MASS_CONVERT,
    #     })

    # Load observational data
    all_obs = _load_smf_grid_observations()
    labels_used = set()  # track legend entries to avoid duplicates

    nrows, ncols = 3, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(25, 15), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    binwidth = 0.1
    _bin18b_smf = {}  # store per-bin SMF for comparison printout
    _alignment = {}  # bin -> [(model, snap, z)] for the alignment report

    for i, (z_lo, z_hi) in enumerate(z_bins):
        ax = axes_flat[i]
        z_mid = 0.5 * (z_lo + z_hi)

        # The first model listed sets the epoch for the panel; the rest are matched to
        # the redshift it actually landed on, not to the bin centre, so overlaid curves
        # are compared at the same epoch. See _select_grid_snapshot().
        z_ref_panel = None
        for model in models:
            snap_num, snap_z = _select_grid_snapshot(
                model['redshifts'], model['first_snap'], model['last_snap'],
                z_lo, z_hi, z_ref=z_ref_panel)
            if snap_num is None:
                continue
            if z_ref_panel is None:
                z_ref_panel = snap_z
            _alignment.setdefault(f'{z_lo:.1f}-{z_hi:.1f}', []).append(
                (model['label'], snap_num, snap_z))
            snap_name = f'Snap_{snap_num}'

            try:
                model_files = find_model_files(model['path'])
                d = read_snap_from_files(model_files, snap_name,
                                         ['StellarMass'],
                                         mass_convert=model['mass_convert'])
                if not d:
                    continue
                m_stars = d['StellarMass']
                w = m_stars > 0
                if np.sum(w) == 0:
                    continue
                log_m = np.log10(m_stars[w])

                # Use bootstrap for SAGE26 models
                if model['label'].startswith('SAGE26'):
                    x, phi, phi_lo, phi_hi, _ = mass_function_bootstrap(
                        log_m, model['volume'], binwidth, n_boot=100)
                    valid = np.isfinite(phi)
                    ax.plot(x[valid], phi[valid],
                            lw=model['lw'], color=model['color'],
                            ls=model['ls'],
                            label=model['label'] if i == 0 else None)
                    # Bootstrap shading
                    boot_valid = np.isfinite(phi_lo) & np.isfinite(phi_hi)
                    if np.any(boot_valid):
                        ax.fill_between(x[boot_valid], phi_lo[boot_valid], phi_hi[boot_valid],
                                        color=model['color'], alpha=0.2, linewidth=0)
                else:
                    x, phi, _ = mass_function(log_m, model['volume'], binwidth)
                    valid = np.isfinite(phi)
                    ax.plot(x[valid], phi[valid],
                            lw=model['lw'], color=model['color'],
                            ls=model['ls'],
                            label=model['label'] if i == 0 else None)
                _bin18b_smf.setdefault(i, {})[model['label']] = (x[valid], phi[valid])
            except Exception as e:
                print(f"  Error loading {snap_name} from {model['path']}: {e}")
                continue

        # Print model comparisons for this z-bin
        _b18b = _bin18b_smf.get(i, {})
        _ref18b = 'SAGE26 (Millennium)'
        if _ref18b in _b18b and len(_b18b) > 1:
            _x26b, _p26b = _b18b[_ref18b]
            print(f"\n  SMF z=[{z_lo:.1f},{z_hi:.1f}] (SAGE26 Millennium vs others):")
            for _olab, (_xo, _po) in _b18b.items():
                if _olab == _ref18b or len(_xo) == 0 or len(_x26b) == 0:
                    continue
                _xc = np.round(np.intersect1d(np.round(_x26b, 4), np.round(_xo, 4)), 4)
                if len(_xc) == 0:
                    continue
                _ip18b = np.array([np.argmin(np.abs(_x26b - xc)) for xc in _xc])
                _io18b = np.array([np.argmin(np.abs(_xo - xc)) for xc in _xc])
                _dv = _p26b[_ip18b] - _po[_io18b]
                _ok18b = np.isfinite(_dv)
                if np.sum(_ok18b) == 0:
                    continue
                _dv_ok = _dv[_ok18b];  _xv18b = _xc[_ok18b]
                print(f"    vs {_olab}: Mean Δ={np.mean(_dv_ok):+.3f} dex, "
                      f"Median={np.median(_dv_ok):+.3f} dex, "
                      f"Max at log10(M*)={_xv18b[np.argmax(_dv_ok)]:.2f}: {np.max(_dv_ok):+.3f} dex")
                for _tm in [9.0, 10.0, 10.5, 11.0]:
                    _ti = np.argmin(np.abs(_xv18b - _tm))
                    if np.abs(_xv18b[_ti] - _tm) < binwidth:
                        print(f"      log10(M*)={_xv18b[_ti]:.2f}: SAGE26={_p26b[_ip18b][_ok18b][_ti]:.2f}, "
                              f"{_olab}={_po[_io18b][_ok18b][_ti]:.2f}, Δ={_dv_ok[_ti]:+.2f} dex")

        # Plot observational data for this redshift bin
        for od in all_obs:
            z_obs = od['z']
            # Match obs to bin: inclusive lower, exclusive upper (last bin inclusive)
            if i == len(z_bins) - 1:
                in_bin = z_lo <= z_obs <= z_hi
            else:
                in_bin = z_lo <= z_obs < z_hi
            if not in_bin:
                continue
            lbl = od['label'] if od['label'] not in labels_used else None
            if lbl is not None:
                labels_used.add(od['label'])
            yerr = None
            if od['err_lo'] is not None and od['err_hi'] is not None:
                yerr = [od['err_lo'], od['err_hi']]
            ax.errorbar(od['log_mass'], od['log_phi'], yerr=yerr,
                        fmt=od['marker'], color='grey', ms=od['ms'],
                        markeredgecolor='k', markeredgewidth=0.8,
                            markerfacecolor = 'gray',
                        alpha=0.6, lw=1.5, capsize=1.5, label=lbl, zorder=1)

        # Redshift label in each panel
        ax.text(0.95, 0.95, rf'${z_lo:.1f} < z < {z_hi:.1f}$',
                transform=ax.transAxes, ha='right', va='top')

    # Axis limits and labels
    axes_flat[0].set_xlim(8.001, 12.2)
    axes_flat[0].set_ylim(-6, -0.8)
    for i, ax in enumerate(axes_flat):
        row, col = divmod(i, ncols)
        if row == nrows - 1:
            ax.set_xlabel(r'$\log_{10}\ m_{\mathrm{*}}\ [M_{\odot}]$')
        if col == 0:
            ax.set_ylabel(r'$\log_{10}\ \phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$')
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.yaxis.set_major_locator(plt.MultipleLocator(2.0))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
        ax.tick_params(axis='both', which='both', direction='in',
                       top=True, bottom=True, left=True, right=True)

    # Per-panel legends (only panels with new labelled entries get a legend)
    for ax in axes_flat:
        _, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc='lower left', frameon=False)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.001, wspace=0.001)

    _report_grid_alignment(_alignment)

    outputFile = os.path.join(OUTPUT_DIR, 'SMF_Redshift_Grid_Wide' + OUTPUT_FORMAT)
    save_figure(fig, outputFile)


# ========================== PLOT 19: SMF FFB GRID ==========================

def plot_19_smf_ffb_grid():
    """
    Plot: 2x3 grid of Stellar Mass Functions at high-z bins.
    Shows SAGE26 (no FFB), SAGE26 (default, sfe=0.2), and SAGE26 (FFB 100%)
    with bootstrap errors.
    """
    print('Plot 19: SMF FFB Grid')

    # Redshift bins: (z_lo, z_hi) - 2 rows x 2 cols
    z_bins = [
        (5.0, 6.0), (6.0, 7.0),
        (7.0, 9.0), (9.0, 11.0),
    ]

    # Redshift arrays
    mill_redshifts = np.array(REDSHIFTS)

    # FFB models to compare: no FFB, default (0.2), and 100%
    models = []
    if os.path.exists(NOFFB_DIR):
        models.append({
            'path': NOFFB_DIR, 'label': r'No FFB/MBK25 model',
            'color': 'firebrick', 'ls': '-', 'lw': 3.0,
            'redshifts': mill_redshifts, 'first_snap': 0, 'last_snap': 63,
            'volume': VOLUME, 'mass_convert': MASS_CONVERT,
        })
    if os.path.exists(PRIMARY_DIR):
        models.append({
            'path': PRIMARY_DIR, 'label': r'$\alpha_{\rm FFB}=0.2$',
            'color': 'black', 'ls': '-', 'lw': 3.5,
            'redshifts': mill_redshifts, 'first_snap': 0, 'last_snap': 63,
            'volume': VOLUME, 'mass_convert': MASS_CONVERT,
        })
    if os.path.exists(FFB100_DIR):
        models.append({
            'path': FFB100_DIR, 'label': r'$\alpha_{\rm FFB}=1.0$',
            'color': 'steelblue', 'ls': '-', 'lw': 3.0,
            'redshifts': mill_redshifts, 'first_snap': 0, 'last_snap': 63,
            'volume': VOLUME, 'mass_convert': MASS_CONVERT,
        })
    # Load observational data
    all_obs = _load_smf_grid_observations()
    labels_used = set()

    nrows, ncols = 1, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 6), sharex=True, sharey=True)
    fig.set_tight_layout(False)
    axes_flat = axes.flatten()
    binwidth = 0.2

    for i, (z_lo, z_hi) in enumerate(z_bins):
        ax = axes_flat[i]
        z_mid = 0.5 * (z_lo + z_hi)

        # --- Model lines with bootstrap errors ---
        for model in models:
            mod_redshifts = model['redshifts']
            first_snap = model['first_snap']
            last_snap = model['last_snap']

            # Find snapshot closest to bin centre that falls within the bin
            snap_redshifts = mod_redshifts[first_snap:last_snap + 1]
            in_bin = np.where((snap_redshifts >= z_lo) & (snap_redshifts <= z_hi))[0]
            if len(in_bin) == 0:
                continue
            snap_idx = in_bin[np.argmin(np.abs(snap_redshifts[in_bin] - z_mid))]
            snap_num = snap_idx + first_snap
            snap_name = f'Snap_{snap_num}'

            try:
                data = load_model(model['path'], snapshot=snap_name, properties=['StellarMass'])
                m_stars = data['StellarMass']
                w = m_stars > 0
                if np.sum(w) == 0:
                    continue
                log_m = np.log10(m_stars[w])
                # Bootstrap SMF
                x, phi, phi_lo, phi_hi, _ = mass_function_bootstrap(
                    log_m, model['volume'], binwidth, n_boot=100)
                valid = np.isfinite(phi)
                ax.plot(x[valid], phi[valid],
                        lw=model['lw'], color=model['color'],
                        ls=model['ls'],
                        label=model['label'] if i == 0 else None)
                # Bootstrap shading
                boot_valid = np.isfinite(phi_lo) & np.isfinite(phi_hi)
                if np.any(boot_valid):
                    ax.fill_between(x[boot_valid], phi_lo[boot_valid], phi_hi[boot_valid],
                                    color=model['color'], alpha=0.2, linewidth=0)
            except Exception as e:
                print(f"  Error loading {snap_name} from {model['path']}: {e}")
                continue

        # Plot observational data for this redshift bin
        for od in all_obs:
            z_obs = od['z']
            if i == len(z_bins) - 1:
                in_bin = z_lo <= z_obs <= z_hi
            else:
                in_bin = z_lo <= z_obs < z_hi
            if not in_bin:
                continue
            lbl = od['label'] if od['label'] not in labels_used else None
            if lbl is not None:
                labels_used.add(od['label'])
            yerr = None
            if od['err_lo'] is not None and od['err_hi'] is not None:
                yerr = [od['err_lo'], od['err_hi']]
            ax.errorbar(od['log_mass'], od['log_phi'], yerr=yerr,
                        fmt=od['marker'], color='grey', ms=od['ms'],
                        markeredgecolor='k', markeredgewidth=0.8,
                        markerfacecolor='gray',
                        alpha=0.6, lw=1.0, label=lbl, zorder=1)

        # Redshift label in each panel
        ax.text(0.95, 0.95, rf'${z_lo:.0f} < z < {z_hi:.0f}$',
                transform=ax.transAxes, ha='right', va='top')

    # Axis limits and labels
    axes_flat[0].set_xlim(9, 12.3)
    axes_flat[0].set_ylim(-6, -1.5)

    for i, ax in enumerate(axes_flat):
        row, col = divmod(i, ncols)
        if col == 0:
            ax.set_ylabel(r'$\log_{10}\ \phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$')
        if row == nrows - 1:
            ax.set_xlabel(r'$\log_{10}\ m_{\mathrm{*}}\ [M_{\odot}]$')
        ax.tick_params(axis='both', which='both', direction='in',
                       top=True, bottom=True, left=True, right=True)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))

    # Legend in first panel
    handles, labels = axes_flat[0].get_legend_handles_labels()
    model_labels_set = {m['label'] for m in models}
    sim_h = [h for h, l in zip(handles, labels) if l in model_labels_set]
    sim_l = [l for l in labels if l in model_labels_set]
    obs_h = [h for h, l in zip(handles, labels) if l not in model_labels_set]
    obs_l = [l for l in labels if l not in model_labels_set]
    if sim_l:
        leg1 = axes_flat[0].legend(sim_h, sim_l, loc='lower left', frameon=False,
                                   title='SAGE26')
        leg1.get_title().set_fontweight('bold')
        axes_flat[0].add_artist(leg1)
    if obs_l:
        axes_flat[0].legend(obs_h, obs_l, loc='upper right', frameon=False,
                            bbox_to_anchor=(1.0, 0.88))

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.001, wspace=0.001)

    outputFile = os.path.join(OUTPUT_DIR, 'SMF_FFB_Grid' + OUTPUT_FORMAT)
    save_figure(fig, outputFile)

# ========================== PLOT 19c: SMF FFB GRID WITH MBK25 ==========================

def plot_19c_smf_ffb_grid_mbk25():
    """
    Plot: 2x2 grid of Stellar Mass Functions at high-z bins.
    Same as plot_19 but with an additional MBK25 (smooth) line in green.
    """
    print('Plot 19c: SMF FFB Grid with MBK25')

    z_bins = [
        (5.0, 6.0), (6.0, 7.0),
        (7.0, 9.0), (9.0, 11.0),
    ]

    mill_redshifts = np.array(REDSHIFTS)

    models = []
    if os.path.exists(NOFFB_DIR):
        models.append({
            'path': NOFFB_DIR, 'label': r'No FFB/MBK25 model',
            'color': 'firebrick', 'ls': '-', 'lw': 3.0,
            'redshifts': mill_redshifts, 'first_snap': 0, 'last_snap': 63,
            'volume': VOLUME, 'mass_convert': MASS_CONVERT,
        })
    if os.path.exists(PRIMARY_DIR):
        models.append({
            'path': PRIMARY_DIR, 'label': r'$\alpha_{\rm FFB}=0.2$ (Li+24)',
            'color': 'black', 'ls': '-', 'lw': 3.5,
            'redshifts': mill_redshifts, 'first_snap': 0, 'last_snap': 63,
            'volume': VOLUME, 'mass_convert': MASS_CONVERT,
        })
    if os.path.exists(FFB_BK25_SMOOTH_DIR):
        models.append({
            'path': FFB_BK25_SMOOTH_DIR, 'label': r'$\alpha_{\rm FFB}=0.2$ (MBK25)',
            'color': 'mediumpurple', 'ls': '-', 'lw': 3.0,
            'redshifts': mill_redshifts, 'first_snap': 0, 'last_snap': 63,
            'volume': VOLUME, 'mass_convert': MASS_CONVERT,
        })
    if os.path.exists(FFB100_DIR):
        models.append({
            'path': FFB100_DIR, 'label': r'$\alpha_{\rm FFB}=1.0$ (Li+24)',
            'color': 'steelblue', 'ls': '--', 'lw': 3.0,
            'redshifts': mill_redshifts, 'first_snap': 0, 'last_snap': 63,
            'volume': VOLUME, 'mass_convert': MASS_CONVERT,
        })
    if os.path.exists(FFB_BK25_FFB100_DIR):
        models.append({
            'path': FFB_BK25_FFB100_DIR, 'label': r'$\alpha_{\rm FFB}=1.0$ (MBK25)',
            'color': 'magenta', 'ls': '--', 'lw': 3.0,
            'redshifts': mill_redshifts, 'first_snap': 0, 'last_snap': 63,
            'volume': VOLUME, 'mass_convert': MASS_CONVERT,
        })

    all_obs = _load_smf_grid_observations()
    labels_used = set()

    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 12), sharex=True, sharey=True)
    fig.set_tight_layout(False)
    axes_flat = axes.flatten()
    binwidth = 0.2

    for i, (z_lo, z_hi) in enumerate(z_bins):
        ax = axes_flat[i]
        z_mid = 0.5 * (z_lo + z_hi)

        for model in models:
            mod_redshifts = model['redshifts']
            first_snap = model['first_snap']
            last_snap = model['last_snap']

            snap_redshifts = mod_redshifts[first_snap:last_snap + 1]
            in_bin = np.where((snap_redshifts >= z_lo) & (snap_redshifts <= z_hi))[0]
            if len(in_bin) == 0:
                continue
            snap_idx = in_bin[np.argmin(np.abs(snap_redshifts[in_bin] - z_mid))]
            snap_num = snap_idx + first_snap
            snap_name = f'Snap_{snap_num}'

            try:
                data = load_model(model['path'], snapshot=snap_name, properties=['StellarMass'])
                m_stars = data['StellarMass']
                w = m_stars > 0
                if np.sum(w) == 0:
                    continue
                log_m = np.log10(m_stars[w])
                x, phi, phi_lo, phi_hi, _ = mass_function_bootstrap(
                    log_m, model['volume'], binwidth, n_boot=100)
                valid = np.isfinite(phi)
                ax.plot(x[valid], phi[valid],
                        lw=model['lw'], color=model['color'],
                        ls=model['ls'],
                        label=model['label'] if i == 0 else None)
                boot_valid = np.isfinite(phi_lo) & np.isfinite(phi_hi)
                if np.any(boot_valid):
                    ax.fill_between(x[boot_valid], phi_lo[boot_valid], phi_hi[boot_valid],
                                    color=model['color'], alpha=0.2, linewidth=0)
            except Exception as e:
                print(f"  Error loading {snap_name} from {model['path']}: {e}")
                continue

        for od in all_obs:
            z_obs = od['z']
            if i == len(z_bins) - 1:
                in_bin = z_lo <= z_obs <= z_hi
            else:
                in_bin = z_lo <= z_obs < z_hi
            if not in_bin:
                continue
            lbl = od['label'] if od['label'] not in labels_used else None
            if lbl is not None:
                labels_used.add(od['label'])
            yerr = None
            if od['err_lo'] is not None and od['err_hi'] is not None:
                yerr = [od['err_lo'], od['err_hi']]
            ax.errorbar(od['log_mass'], od['log_phi'], yerr=yerr,
                        fmt=od['marker'], color='grey', ms=od['ms'],
                        markeredgecolor='k', markeredgewidth=0.8,
                        markerfacecolor='gray',
                        alpha=0.6, lw=1.0, label=lbl, zorder=1)

        ax.text(0.95, 0.95, rf'${z_lo:.0f} < z < {z_hi:.0f}$',
                transform=ax.transAxes, ha='right', va='top')

    axes_flat[0].set_xlim(9, 12.3)
    axes_flat[0].set_ylim(-6, -1.5)

    for i, ax in enumerate(axes_flat):
        row, col = divmod(i, ncols)
        if col == 0:
            ax.set_ylabel(r'$\log_{10}\ \phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$')
        if row == nrows - 1:
            ax.set_xlabel(r'$\log_{10}\ m_{\mathrm{*}}\ [M_{\odot}]$')
        ax.tick_params(axis='both', which='both', direction='in',
                       top=True, bottom=True, left=True, right=True)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))

    handles, labels = axes_flat[0].get_legend_handles_labels()
    model_labels_set = {m['label'] for m in models}
    sim_h = [h for h, l in zip(handles, labels) if l in model_labels_set]
    sim_l = [l for l in labels if l in model_labels_set]
    obs_h = [h for h, l in zip(handles, labels) if l not in model_labels_set]
    obs_l = [l for l in labels if l not in model_labels_set]
    if sim_l:
        leg1 = axes_flat[0].legend(sim_h, sim_l, loc='lower left', frameon=False,
                                   title='SAGE26')
        leg1.get_title().set_fontweight('bold')
        axes_flat[0].add_artist(leg1)
    if obs_l:
        axes_flat[0].legend(obs_h, obs_l, loc='upper right', frameon=False,
                            bbox_to_anchor=(1.0, 0.88))

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.001, wspace=0.001)

    outputFile = os.path.join(OUTPUT_DIR, 'SMF_FFB_Grid_MBK25' + OUTPUT_FORMAT)
    save_figure(fig, outputFile)

# ========================== PLOT 19b: SMF FFB METHOD COMPARISON GRID ==========================

def plot_19b_smf_ffb_methods_grid():
    """
    Plot: 2x4 grid of Stellar Mass Functions at high-z bins comparing 4 FFB methods.
      - Top row: Absolute SMF
      - Bottom row: Residual ratios (Model - Cutoff Baseline)
    Handles missing models/data gracefully.
    """
    print('Plot 19b: SMF FFB Methods Grid (Main + Ratios)')

    z_bins = [
        (5.0, 6.0), (6.0, 7.0),
        (7.0, 9.0), (9.0, 11.0),
    ]

    mill_redshifts = np.array(REDSHIFTS)

    # Dictionary definition for robust lookups
    model_keys = ['li_sigmoid', 'MBK25_smooth', 'li_nosig', 'MBK25_sharp']
    model_dirs = {
        'li_sigmoid':   PRIMARY_DIR,
        'MBK25_smooth': FFB_BK25_SMOOTH_DIR,
        'li_nosig':     FFB_NOSIGMOID_DIR,
        'MBK25_sharp':  FFB_BK25_DIR,
    }
    model_labels = {
        'li_sigmoid':   r'Li+24 (sigmoid)',
        'MBK25_smooth': r'MBK25 (log-normal scatter)',
        'li_nosig':     r'Li+24 (no sigmoid)',
        'MBK25_sharp':  r'MBK25 (no log-normal scatter))',
    }
    model_colors = {
        'li_sigmoid':   'black',
        'MBK25_smooth': 'steelblue',
        'li_nosig':     'firebrick',
        'MBK25_sharp':  'darkgreen',
    }
    model_ls = {
        'li_sigmoid':   '-',
        'MBK25_smooth': '-',
        'li_nosig':     '--',
        'MBK25_sharp':  '--',
    }
    model_lw = {
        'li_sigmoid':   3.5,
        'MBK25_smooth': 3.0,
        'li_nosig':     3.0,
        'MBK25_sharp':  3.0,
    }

    # Warn if directories are missing
    for key in model_keys:
        if not os.path.exists(model_dirs[key]):
            print(f"  --> WARNING: Directory for {key} not found: {model_dirs[key]}")

    all_obs = _load_smf_grid_observations()
    labels_used = set()

    # 2 rows (Main, Res), 4 cols (z-bins)
    nrows, ncols = 2, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 8), 
                             sharex=True, sharey='row',
                             gridspec_kw={'height_ratios': [3, 1.2]})
    fig.set_tight_layout(False)
    binwidth = 0.2

    for col, (z_lo, z_hi) in enumerate(z_bins):
        ax_main = axes[0, col]
        ax_res = axes[1, col]
        z_mid = 0.5 * (z_lo + z_hi)

        # Draw deviation region in residual panel
        ax_res.axhline(0, color='black', linestyle='-', linewidth=1.5, zorder=1)
        ax_res.axhspan(-0.3, 0.3, color='gray', alpha=0.15, zorder=0)

        # Temporary storage for interpolation
        col_data = {}

        # ==========================================
        # 1. LOAD AND PLOT MAIN MODELS
        # ==========================================
        for key in model_keys:
            path = model_dirs[key]
            if not os.path.exists(path):
                continue
            
            in_bin = np.where((mill_redshifts >= z_lo) & (mill_redshifts <= z_hi))[0]
            if len(in_bin) == 0:
                continue
            
            snap_idx = in_bin[np.argmin(np.abs(mill_redshifts[in_bin] - z_mid))]
            snap_name = f'Snap_{snap_idx}' # Assuming first_snap = 0

            try:
                data = load_model(path, snapshot=snap_name, properties=['StellarMass'])
                m_stars = data['StellarMass']
                w = m_stars > 0
                if np.sum(w) == 0:
                    continue
                
                log_m = np.log10(m_stars[w])
                x, phi, _, _, _ = mass_function_bootstrap(log_m, VOLUME, binwidth, n_boot=100)
                valid = np.isfinite(phi)
                
                x_val, phi_val = x[valid], phi[valid]
                col_data[key] = {'x': x_val, 'phi': phi_val}

                # Plot absolute SMF
                ax_main.plot(x_val, phi_val, lw=model_lw[key], color=model_colors[key],
                             ls=model_ls[key], label=model_labels[key] if col == 0 else None)
                             
            except Exception as e:
                print(f"  Error loading {snap_name} for {key}: {e}")
                continue

        # ==========================================
        # 2. PLOT RESIDUAL MODELS (Ratios)
        # ==========================================
        # Li+24 Ratios
        if 'li_sigmoid' in col_data and 'li_nosig' in col_data:
            x_sig, phi_sig = col_data['li_sigmoid']['x'], col_data['li_sigmoid']['phi']
            x_base, phi_base = col_data['li_nosig']['x'], col_data['li_nosig']['phi']
            
            # Interpolate baseline onto sigmoid mass bins. Use NaN outside bounds to avoid flatlines.
            interp_base = np.interp(x_sig, x_base, phi_base, left=np.nan, right=np.nan)
            ax_res.plot(x_sig, phi_sig - interp_base, '-', color='black', lw=3.5)

        # MBK25 Ratios
        if 'MBK25_smooth' in col_data and 'MBK25_sharp' in col_data:
            x_sm, phi_sm = col_data['MBK25_smooth']['x'], col_data['MBK25_smooth']['phi']
            x_base, phi_base = col_data['MBK25_sharp']['x'], col_data['MBK25_sharp']['phi']
            
            interp_base = np.interp(x_sm, x_base, phi_base, left=np.nan, right=np.nan)
            ax_res.plot(x_sm, phi_sm - interp_base, '-', color='steelblue', lw=3.0)

        # ==========================================
        # 3. LOAD AND PLOT OBSERVATIONS
        # ==========================================
        for od in all_obs:
            z_obs = od['z']
            if col == len(z_bins) - 1:
                in_bin = z_lo <= z_obs <= z_hi
            else:
                in_bin = z_lo <= z_obs < z_hi
            if not in_bin:
                continue

            # LaTeX safe replacement
            raw_lbl = od['label']
            safe_lbl = raw_lbl.replace('&', r'\&') if raw_lbl else None
            
            lbl = safe_lbl if safe_lbl not in labels_used else None
            if lbl is not None:
                labels_used.add(safe_lbl)
                
            yerr = [od['err_lo'], od['err_hi']] if od['err_lo'] is not None else None

            # Main absolute observation plot
            ax_main.errorbar(od['log_mass'], od['log_phi'], yerr=yerr,
                             fmt=od['marker'], color='grey', ms=od['ms'],
                             markeredgecolor='k', markeredgewidth=0.8,
                             markerfacecolor='gray', alpha=0.6, lw=1.0, label=lbl, zorder=5)

            # Residual observation plot (normalized by Li+24 Cutoff)
            if 'li_nosig' in col_data:
                x_base, phi_base = col_data['li_nosig']['x'], col_data['li_nosig']['phi']
                interp_base = np.interp(od['log_mass'], x_base, phi_base, left=np.nan, right=np.nan)
                ratio_phi = od['log_phi'] - interp_base
                
                ax_res.errorbar(od['log_mass'], ratio_phi, yerr=yerr,
                                fmt=od['marker'], color='grey', ms=od['ms'],
                                markeredgecolor='k', markeredgewidth=0.8,
                                markerfacecolor='gray', alpha=0.6, lw=1.0, zorder=5)

        # Redshift text box
        ax_main.text(0.95, 0.95, rf'${z_lo:.0f} < z < {z_hi:.0f}$',
                     transform=ax_main.transAxes, ha='right', va='top', fontsize=12)

    # ==========================================
    # 4. FORMATTING AND CLEANUP
    # ==========================================
    axes[0, 0].set_xlim(7, 12.3)
    axes[0, 0].set_ylim(-6, -1.5)
    axes[1, 0].set_ylim(-0.75, 2.5) # Residual y-bounds

    axes[0, 0].set_ylabel(r'$\log_{10}\ \phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$')
    axes[1, 0].set_ylabel(r'$\Delta$ (dex)')

    for i, ax in enumerate(axes.flatten()):
        row, col = divmod(i, ncols)
        
        # Only bottom row gets X-labels
        if row == 1:
            ax.set_xlabel(r'$\log_{10}\ m_{\mathrm{*}}\ [M_{\odot}]$')
        
        ax.tick_params(axis='both', which='both', direction='in',
                       top=True, bottom=True, left=True, right=True)
                       
        if row == 0:
            ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
        else:
            ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
            
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))

    # Legends in the first panel column
    handles, labels = axes[0, 0].get_legend_handles_labels()
    model_labels_set = set(model_labels.values())
    
    sim_h = [h for h, l in zip(handles, labels) if l in model_labels_set]
    sim_l = [l for l in labels if l in model_labels_set]
    obs_h = [h for h, l in zip(handles, labels) if l not in model_labels_set]
    obs_l = [l for l in labels if l not in model_labels_set]
    
    if sim_l:
        leg1 = axes[0, 0].legend(sim_h, sim_l, loc='lower left', frameon=False, title='SAGE26')
        leg1.get_title().set_fontweight('bold')
        axes[0, 0].add_artist(leg1)
        
    if obs_l:
        axes[0, 0].legend(obs_h, obs_l, loc='upper right', frameon=False, bbox_to_anchor=(1.0, 0.88))

    # Optional: Add legend for the gray shaded region in the bottom left panel
    axes[1, 0].legend([plt.Rectangle((0,0),1,1, color='gray', alpha=0.15)], [r'$\pm 0.3$ dex'], 
                      loc='upper left', frameon=False)

    # Tighten spacing to make it visually cohesive
    fig.subplots_adjust(hspace=0.05, wspace=0.001)

    outputFile = os.path.join(OUTPUT_DIR, 'SMF_FFB_Methods_Grid' + OUTPUT_FORMAT)
    save_figure(fig, outputFile)

# ========================== PLOT 20: SMF LOW-Z GRID ==========================

def plot_20_smf_lowz_grid():
    """
    Plot: 1x3 grid of Stellar Mass Functions at low-z bins (0-1, 1-2, 2-3).
    Shows SAGE26 (no FFB), SAGE26 (no CGM), and C16.
    """
    print('Plot 20: SMF Low-z Grid')

    # Redshift bins: (z_lo, z_hi)
    z_bins = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]

    # Redshift arrays
    mill_redshifts = np.array(REDSHIFTS)

    # Model lines: SAGE26 (with CGM), SAGE26 (no CGM), C16
    models = []
    if os.path.exists(NOFFB_DIR):
        models.append({
            'path': NOFFB_DIR, 'label': 'SAGE26 (with CGM)',
            'color': 'green', 'ls': '-', 'lw': 3.5,
            'redshifts': mill_redshifts, 'first_snap': 0, 'last_snap': 63,
            'volume': VOLUME, 'mass_convert': MASS_CONVERT,
        })
    if os.path.exists(NOCGM_DIR):
        models.append({
            'path': NOCGM_DIR, 'label': 'SAGE26 (no CGM)',
            'color': 'purple', 'ls': '-', 'lw': 3.5,
            'redshifts': mill_redshifts, 'first_snap': 0, 'last_snap': 63,
            'volume': VOLUME, 'mass_convert': MASS_CONVERT,
        })
    if os.path.exists(VANILLA_DIR):
        models.append({
            'path': VANILLA_DIR, 'label': 'SAGE16',
            'color': 'firebrick', 'ls': '--', 'lw': 2.5,
            'redshifts': mill_redshifts, 'first_snap': 0, 'last_snap': 63,
            'volume': VOLUME, 'mass_convert': MASS_CONVERT,
        })

    # Load observational data
    all_obs = _load_smf_grid_observations()
    labels_used = set()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    fig.set_tight_layout(False)
    binwidth = 0.2

    for i, (z_lo, z_hi) in enumerate(z_bins):
        ax = axes[i]
        z_mid = 0.5 * (z_lo + z_hi)

        # --- Model lines ---
        for model in models:
            mod_redshifts = model['redshifts']
            first_snap = model['first_snap']
            last_snap = model['last_snap']

            # Find snapshot closest to bin centre that falls within the bin
            snap_redshifts = mod_redshifts[first_snap:last_snap + 1]
            in_bin = np.where((snap_redshifts >= z_lo) & (snap_redshifts <= z_hi))[0]
            if len(in_bin) == 0:
                continue
            snap_idx = in_bin[np.argmin(np.abs(snap_redshifts[in_bin] - z_mid))]
            snap_num = snap_idx + first_snap
            snap_name = f'Snap_{snap_num}'

            try:
                model_files = find_model_files(model['path'])
                d = read_snap_from_files(model_files, snap_name,
                                         ['StellarMass'],
                                         mass_convert=model['mass_convert'])
                if not d:
                    continue
                m_stars = d['StellarMass']
                w = m_stars > 0
                if np.sum(w) == 0:
                    continue
                log_m = np.log10(m_stars[w])

                # Use bootstrap for SAGE26 models
                if model['label'].startswith('SAGE26'):
                    x, phi, phi_lo, phi_hi, _ = mass_function_bootstrap(
                        log_m, model['volume'], binwidth, n_boot=100)
                    valid = np.isfinite(phi)
                    ax.plot(x[valid], phi[valid],
                            lw=model['lw'], color=model['color'],
                            ls=model['ls'],
                            label=model['label'] if i == 0 else None)
                    # Bootstrap shading
                    boot_valid = np.isfinite(phi_lo) & np.isfinite(phi_hi)
                    if np.any(boot_valid):
                        ax.fill_between(x[boot_valid], phi_lo[boot_valid], phi_hi[boot_valid],
                                        color=model['color'], alpha=0.2, linewidth=0)
                else:
                    x, phi, _ = mass_function(log_m, model['volume'], binwidth)
                    valid = np.isfinite(phi)
                    ax.plot(x[valid], phi[valid],
                            lw=model['lw'], color=model['color'],
                            ls=model['ls'],
                            label=model['label'] if i == 0 else None)
            except Exception as e:
                print(f"  Error loading {snap_name} from {model['path']}: {e}")
                continue

        # Plot observational data for this redshift bin
        for od in all_obs:
            z_obs = od['z']
            if i == len(z_bins) - 1:
                in_bin = z_lo <= z_obs <= z_hi
            else:
                in_bin = z_lo <= z_obs < z_hi
            if not in_bin:
                continue
            lbl = od['label'] if od['label'] not in labels_used else None
            if lbl is not None:
                labels_used.add(od['label'])
            yerr = None
            if od['err_lo'] is not None and od['err_hi'] is not None:
                yerr = [od['err_lo'], od['err_hi']]
            ax.errorbar(od['log_mass'], od['log_phi'], yerr=yerr,
                        fmt=od['marker'], color='grey', ms=od['ms'],
                        markeredgecolor='k', markeredgewidth=0.8,
                            markerfacecolor = 'gray',
                        alpha=0.6, lw=1.0, label=lbl, zorder=1)

        # Redshift label in each panel
        ax.text(0.95, 0.95, rf'${z_lo:.0f} < z < {z_hi:.0f}$',
                transform=ax.transAxes, ha='right', va='top')

    # Axis limits and labels
    axes[0].set_xlim(10.5, 12.5)
    axes[0].set_ylim(-6, -1.5)

    axes[0].set_ylabel(r'$\log_{10}\ \phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$')
    for i, ax in enumerate(axes):
        ax.set_xlabel(r'$\log_{10}\ m_{\mathrm{*}}\ [M_{\odot}]$')
        ax.tick_params(axis='both', which='both', direction='in',
                       top=True, bottom=True, left=True, right=True)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))

    # Split legends: SAGE26 models lower left, observations upper right
    handles, labels = axes[0].get_legend_handles_labels()
    sim_h = [h for h, l in zip(handles, labels) if l.startswith('SAGE26') or l == 'SAGE16']
    sim_l = [l for l in labels if l.startswith('SAGE26') or l == 'SAGE16']
    obs_h = [h for h, l in zip(handles, labels) if not (l.startswith('SAGE26') or l == 'SAGE16')]
    obs_l = [l for l in labels if not (l.startswith('SAGE26') or l == 'SAGE16')]
    if sim_l:
        leg1 = axes[0].legend(sim_h, sim_l, loc='lower left', frameon=False)
        axes[0].add_artist(leg1)
    if obs_l:
        axes[0].legend(obs_h, obs_l, loc='upper right', frameon=False,
                       bbox_to_anchor=(1.0, 0.88))

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.001)

    outputFile = os.path.join(OUTPUT_DIR, 'SMF_LowZ_Grid' + OUTPUT_FORMAT)
    save_figure(fig, outputFile)


# ========================== PLOT 21: SMF LOW-Z LOW-MASS GRID ==========================

def plot_21_smf_lowz_lowmass_grid():
    """
    Plot: 1x3 grid of Stellar Mass Functions at low-z bins (0-1, 1-2, 2-3).
    Shows SAGE26 (Millennium), SAGE26 (C16 Feedback), and C16.
    Low-mass x-axis range.
    """
    print('Plot 21: SMF Low-z Low-mass Grid')

    # Redshift bins: (z_lo, z_hi)
    z_bins = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]

    # Redshift arrays
    mill_redshifts = np.array(REDSHIFTS)

    # Model lines: SAGE26 (Millennium), SAGE26 (C16 Feedback), C16
    models = []
    if os.path.exists(PRIMARY_DIR):
        models.append({
            'path': PRIMARY_DIR, 'label': 'SAGE26 (Millennium)',
            'color': 'black', 'ls': '-', 'lw': 3.5,
            'redshifts': mill_redshifts, 'first_snap': 0, 'last_snap': 63,
            'volume': VOLUME, 'mass_convert': MASS_CONVERT,
        })
    if os.path.exists(C16_FEEDBACK_DIR):
        models.append({
            'path': C16_FEEDBACK_DIR, 'label': 'SAGE26 (C16 Feedback)',
            'color': 'steelblue', 'ls': '-', 'lw': 3.5,
            'redshifts': mill_redshifts, 'first_snap': 0, 'last_snap': 63,
            'volume': VOLUME, 'mass_convert': MASS_CONVERT,
        })
    if os.path.exists(VANILLA_DIR):
        models.append({
            'path': VANILLA_DIR, 'label': 'SAGE16',
            'color': 'firebrick', 'ls': '--', 'lw': 2.5,
            'redshifts': mill_redshifts, 'first_snap': 0, 'last_snap': 63,
            'volume': VOLUME, 'mass_convert': MASS_CONVERT,
        })

    # Load observational data
    all_obs = _load_smf_grid_observations()
    labels_used = set()

    fig, axes = plt.subplots(3, 1, figsize=(8, 18), sharex=True, sharey=True)
    fig.set_tight_layout(False)
    binwidth = 0.2

    for i, (z_lo, z_hi) in enumerate(z_bins):
        ax = axes[i]
        z_mid = 0.5 * (z_lo + z_hi)

        # --- Model lines ---
        for model in models:
            mod_redshifts = model['redshifts']
            first_snap = model['first_snap']
            last_snap = model['last_snap']

            # Find snapshot closest to bin centre that falls within the bin
            snap_redshifts = mod_redshifts[first_snap:last_snap + 1]
            in_bin = np.where((snap_redshifts >= z_lo) & (snap_redshifts <= z_hi))[0]
            if len(in_bin) == 0:
                continue
            snap_idx = in_bin[np.argmin(np.abs(snap_redshifts[in_bin] - z_mid))]
            snap_num = snap_idx + first_snap
            snap_name = f'Snap_{snap_num}'

            try:
                model_files = find_model_files(model['path'])
                d = read_snap_from_files(model_files, snap_name,
                                         ['StellarMass'],
                                         mass_convert=model['mass_convert'])
                if not d:
                    continue
                m_stars = d['StellarMass']
                w = m_stars > 0
                if np.sum(w) == 0:
                    continue
                log_m = np.log10(m_stars[w])

                # Use bootstrap for SAGE26 models
                if model['label'].startswith('SAGE26'):
                    x, phi, phi_lo, phi_hi, _ = mass_function_bootstrap(
                        log_m, model['volume'], binwidth, n_boot=100)
                    valid = np.isfinite(phi)
                    ax.plot(x[valid], phi[valid],
                            lw=model['lw'], color=model['color'],
                            ls=model['ls'],
                            label=model['label'] if i == 0 else None)
                    # Bootstrap shading
                    boot_valid = np.isfinite(phi_lo) & np.isfinite(phi_hi)
                    if np.any(boot_valid):
                        ax.fill_between(x[boot_valid], phi_lo[boot_valid], phi_hi[boot_valid],
                                        color=model['color'], alpha=0.2, linewidth=0)
                else:
                    x, phi, _ = mass_function(log_m, model['volume'], binwidth)
                    valid = np.isfinite(phi)
                    ax.plot(x[valid], phi[valid],
                            lw=model['lw'], color=model['color'],
                            ls=model['ls'],
                            label=model['label'] if i == 0 else None)
            except Exception as e:
                print(f"  Error loading {snap_name} from {model['path']}: {e}")
                continue

        # Plot observational data for this redshift bin
        for od in all_obs:
            z_obs = od['z']
            if i == len(z_bins) - 1:
                in_bin = z_lo <= z_obs <= z_hi
            else:
                in_bin = z_lo <= z_obs < z_hi
            if not in_bin:
                continue
            lbl = od['label'] if od['label'] not in labels_used else None
            if lbl is not None:
                labels_used.add(od['label'])
            yerr = None
            if od['err_lo'] is not None and od['err_hi'] is not None:
                yerr = [od['err_lo'], od['err_hi']]
            ax.errorbar(od['log_mass'], od['log_phi'], yerr=yerr,
                        fmt=od['marker'], color='grey', ms=od['ms'],
                        markeredgecolor='k', markeredgewidth=0.8,
                            markerfacecolor = 'gray',
                        alpha=0.6, lw=1.0, label=lbl, zorder=1)

        # Redshift label in each panel
        ax.text(0.95, 0.95, rf'${z_lo:.0f} < z < {z_hi:.0f}$',
                transform=ax.transAxes, ha='right', va='top')

    # Axis limits and labels (low-mass range)
    axes[0].set_xlim(8, 10.5)
    axes[0].set_ylim(-4, -0.5)

    for i, ax in enumerate(axes):
        ax.set_ylabel(r'$\log_{10}\ \phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$')
        ax.tick_params(axis='both', which='both', direction='in',
                       top=True, bottom=True, left=True, right=True)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    axes[-1].set_xlabel(r'$\log_{10}\ m_{\mathrm{*}}\ [M_{\odot}]$')

    # Split legends: SAGE26 models lower left, observations lower right
    handles, labels = axes[0].get_legend_handles_labels()
    sim_h = [h for h, l in zip(handles, labels) if l.startswith('SAGE26') or l == 'SAGE16']
    sim_l = [l for l in labels if l.startswith('SAGE26') or l == 'SAGE16']
    obs_h = [h for h, l in zip(handles, labels) if not (l.startswith('SAGE26') or l == 'SAGE16')]
    obs_l = [l for l in labels if not (l.startswith('SAGE26') or l == 'SAGE16')]
    if sim_l:
        leg1 = axes[0].legend(sim_h, sim_l, loc='lower left', frameon=False)
        axes[0].add_artist(leg1)
    if obs_l:
        axes[0].legend(obs_h, obs_l, loc='lower right', frameon=False)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.001)

    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))

    outputFile = os.path.join(OUTPUT_DIR, 'SMF_LowZ_LowMass_Grid' + OUTPUT_FORMAT)
    save_figure(fig, outputFile)


# ========================== PLOT 22: REGIME HISTOGRAM (EVOLUTION) ==========================

def plot_22_regime_histogram():
    """
    Plot: Histogram of galaxy counts for Hot-regime vs CGM-regime as a function of redshift.
    CGM Galaxies: Blues colormap
    Hot Galaxies: Gist Heat (Reverse) colormap
    """
    print('Plot 22: Regime Histogram (Evolution)')

    num_hot_per_snap = []
    num_cgm_per_snap = []
    redshifts_list = []

    model_files = find_model_files(PRIMARY_DIR)
    if not model_files:
        print(f"  No model files found in {PRIMARY_DIR}")
        return

    for snap in range(64):
        snap_key = f'Snap_{snap}'
        d = read_snap_from_files(model_files, snap_key, ['Regime'])
        if d and 'Regime' in d:
            regime = d['Regime']
            num_hot = np.sum(regime == 1)
            num_cgm = np.sum(regime == 0)
        else:
            num_hot = 0
            num_cgm = 0

        num_hot_per_snap.append(num_hot)
        num_cgm_per_snap.append(num_cgm)
        redshifts_list.append(REDSHIFTS[snap])

    z = np.array(redshifts_list)
    num_hot_plot = np.array(num_hot_per_snap)
    num_cgm_plot = np.array(num_cgm_per_snap)

    fig = plt.figure()
    ax = plt.subplot(111)

    # Filter for z <= 15
    z_mask = z <= 15
    z_filtered = z[z_mask]
    num_hot_filtered = num_hot_plot[z_mask]
    num_cgm_filtered = num_cgm_plot[z_mask]

    # Define bin edges in log10(1+z) space
    z_edges = [15.0]
    for i in range(len(z_filtered) - 1):
        mid_point = (z_filtered[i] + z_filtered[i+1]) / 2.0
        z_edges.append(mid_point)
    z_edges.append(0.0)
    z_edges = np.array(z_edges)

    # Convert to log10(1+z)
    log1pz_edges = np.log10(1 + z_edges)
    widths = log1pz_edges[:-1] - log1pz_edges[1:]

    # Colormaps - normalize on log10(1+z) scale for even color distribution
    log1pz_values = np.log10(1 + z_edges[:-1])
    norm = plt.Normalize(vmin=np.min(log1pz_values), vmax=np.max(log1pz_values))

    # Hot Galaxies (Red/Heat gradient)
    cmap_hot = plt.get_cmap('Reds')
    colors_hot = cmap_hot(norm(log1pz_values))

    # CGM Galaxies (Blues gradient)
    cmap_cgm = plt.get_cmap('Greens')
    colors_cgm = cmap_cgm(norm(log1pz_values))

    ax.bar(log1pz_edges[:-1], num_cgm_filtered, width=widths, align='edge',
           label='CGM Galaxies', edgecolor='black', color=colors_cgm)
    ax.bar(log1pz_edges[:-1], num_hot_filtered, width=widths, align='edge',
           label='Hot Galaxies', edgecolor='black', color=colors_hot)

    ax.set_yscale('log')
    ax.set_ylabel('Number of Galaxies')
    ax.set_xlabel(r'$\log_{10}(1+z)$')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=2)

    ax.set_xlim(np.log10(1+15), 0)

    # Add top x-axis for redshift
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    z_ticks = [0, 1, 2, 3, 5, 7, 10, 15]
    log1pz_ticks = [np.log10(1 + zt) for zt in z_ticks]
    ax2.set_xticks(log1pz_ticks)
    ax2.set_xticklabels([str(zt) for zt in z_ticks])
    ax2.set_xlabel(r'$z$')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    outputFile = os.path.join(OUTPUT_DIR, 'Regime_Histogram_Evolution' + OUTPUT_FORMAT)
    plt.savefig(outputFile)
    print(f'Saved file to {outputFile}\n')
    plt.close()


# ========================== PLOT 23: FFB HISTOGRAM (EVOLUTION) ==========================

def plot_23_ffb_histogram():
    """
    Plot: Stacked bar chart of FFB vs Non-FFB Galaxies as a function of redshift.
    Non-FFB Galaxies: Blues colormap
    FFB Galaxies: Reds colormap
    """
    print('Plot 23: FFB Histogram (Evolution)')

    num_non_ffb_per_snap = []
    num_ffb_per_snap = []
    redshifts_list = []

    model_files = find_model_files(PRIMARY_DIR)
    if not model_files:
        print(f"  No model files found in {PRIMARY_DIR}")
        return

    for snap in range(64):
        snap_key = f'Snap_{snap}'
        d = read_snap_from_files(model_files, snap_key, ['FFBRegime'])
        if d and 'FFBRegime' in d:
            ffb_regime = d['FFBRegime']
            num_ffb = np.sum(ffb_regime == 1)
            num_non_ffb = np.sum(ffb_regime == 0)
        else:
            num_ffb = 0
            num_non_ffb = 0

        num_non_ffb_per_snap.append(num_non_ffb)
        num_ffb_per_snap.append(num_ffb)
        redshifts_list.append(REDSHIFTS[snap])

    z = np.array(redshifts_list)
    num_non_ffb_plot = np.array(num_non_ffb_per_snap)
    num_ffb_plot = np.array(num_ffb_per_snap)

    fig = plt.figure()
    ax = plt.subplot(111)

    # Filter for z <= 15
    z_mask = z <= 15
    z_filtered = z[z_mask]
    num_non_ffb_filtered = num_non_ffb_plot[z_mask]
    num_ffb_filtered = num_ffb_plot[z_mask]

    # Define bin edges in log10(1+z) space
    z_edges = [15.0]
    for i in range(len(z_filtered) - 1):
        mid_point = (z_filtered[i] + z_filtered[i+1]) / 2.0
        z_edges.append(mid_point)
    z_edges.append(0.0)
    z_edges = np.array(z_edges)

    # Convert to log10(1+z)
    log1pz_edges = np.log10(1 + z_edges)
    widths = log1pz_edges[:-1] - log1pz_edges[1:]

    # Colormaps - normalize on log10(1+z) scale for even color distribution
    log1pz_values = np.log10(1 + z_edges[:-1])
    norm = plt.Normalize(vmin=np.min(log1pz_values), vmax=np.max(log1pz_values))

    # FFB Galaxies (Reds gradient)
    cmap_ffb = plt.get_cmap('RdPu')
    colors_ffb = cmap_ffb(norm(log1pz_values))

    # Non-FFB Galaxies (Blues gradient)
    cmap_non_ffb = plt.get_cmap('Greys')
    colors_non_ffb = cmap_non_ffb(norm(log1pz_values))

    ax.bar(log1pz_edges[:-1], num_non_ffb_filtered, width=widths, align='edge',
           label='Non-FFB Galaxies', edgecolor='black', color=colors_non_ffb)
    ax.bar(log1pz_edges[:-1], num_ffb_filtered, width=widths, align='edge',
           label='FFB Galaxies', edgecolor='black', color=colors_ffb)

    ax.set_yscale('log')
    ax.set_ylabel('Number of Galaxies')
    ax.set_xlabel(r'$\log_{10}(1+z)$')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=2)

    ax.set_xlim(np.log10(1+15), 0)

    # Add top x-axis for redshift
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    z_ticks = [0, 1, 2, 3, 5, 7, 10, 15]
    log1pz_ticks = [np.log10(1 + zt) for zt in z_ticks]
    ax2.set_xticks(log1pz_ticks)
    ax2.set_xticklabels([str(zt) for zt in z_ticks])
    ax2.set_xlabel(r'$z$')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    outputFile = os.path.join(OUTPUT_DIR, 'FFB_Histogram_Evolution' + OUTPUT_FORMAT)
    plt.savefig(outputFile)
    print(f'Saved file to {outputFile}\n')
    plt.close()


def plot_23b_ffb_histogram_bk25():
    """
    Plot: Step plot of FFB Galaxies as a function of redshift,
    comparing Li+24 and MBK25 FFB implementations.
    """
    print('Plot 23b: FFB Histogram - Li+24 vs MBK25')

    # Updated to use simple colors for step plots
    models = [
        {'dir': PRIMARY_DIR,  'label': 'Li+24',  'color': 'black'},
        {'dir': FFB_BK25_SMOOTH_DIR, 'label': 'MBK25',   'color': 'mediumpurple'},
    ]

    fig = plt.figure()
    ax = plt.subplot(111)

    for model in models:
        model_files = find_model_files(model['dir'])
        if not model_files:
            print(f"  No model files found in {model['dir']}")
            continue

        num_ffb_per_snap = []
        redshifts_list = []

        for snap in range(64):
            snap_key = f'Snap_{snap}'
            d = read_snap_from_files(model_files, snap_key, ['FFBRegime'])

            # Count only FFB galaxies now
            if d and 'FFBRegime' in d:
                ffb_regime = d['FFBRegime']
                num_ffb = np.sum(ffb_regime == 1)
            else:
                num_ffb = 0

            num_ffb_per_snap.append(num_ffb)
            redshifts_list.append(REDSHIFTS[snap])

        z = np.array(redshifts_list)
        num_ffb_plot = np.array(num_ffb_per_snap)

        # Filter for z <= 15
        z_mask = z <= 15
        z_filtered = z[z_mask]
        num_ffb_filtered = num_ffb_plot[z_mask]

        # Define bin edges in log10(1+z) space
        z_edges = [15.0]
        for i in range(len(z_filtered) - 1):
            mid_point = (z_filtered[i] + z_filtered[i+1]) / 2.0
            z_edges.append(mid_point)
        z_edges.append(0.0)
        z_edges = np.array(z_edges)

        log1pz_edges = np.log10(1 + z_edges)

        # Draw FFB as an unfilled step plot using ax.stairs
        ax.stairs(num_ffb_filtered, log1pz_edges, fill=False,
                  label=model['label'], edgecolor=model['color'], linewidth=2)

    ax.set_yscale('log')
    ax.set_ylabel('Number of Galaxies')
    # Left, not right: reversing the x-axis puts the peak under a right-hand legend.
    ax.legend(loc='upper left', frameon=False)

    _ffb_histogram_x_axes(ax)

    fig.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    outputFile = os.path.join(OUTPUT_DIR, 'FFB_Histogram_Li24_vs_MBK25' + OUTPUT_FORMAT)
    plt.savefig(outputFile)
    print(f'Saved file to {outputFile}\n')
    plt.close()


def _ffb_histogram_x_axes(ax):
    """
    Shared x-axis setup for the Li+24 vs MBK25 histogram figures.

    The data are binned in log10(1+z), which stays the plotting coordinate.
    The bottom axis is labelled with redshift and the top axis carries the
    log10(1+z) values.  The direction is reversed relative to the original
    figure, so both quantities now increase left to right.
    """
    ax.set_xlim(0, np.log10(1 + 15))
    ax.set_xlabel(r'$z$')

    z_ticks = [0, 1, 2, 3, 5, 7, 10, 15]
    ax.set_xticks([np.log10(1 + zt) for zt in z_ticks])
    ax.set_xticklabels([str(zt) for zt in z_ticks])

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    log1pz_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    ax_top.set_xticks(log1pz_ticks)
    ax_top.set_xticklabels([f'{t:g}' for t in log1pz_ticks])
    ax_top.set_xlabel(r'$\log_{10}(1+z)$', labelpad=10)
    return ax_top


def plot_23c_ffb_fraction_bk25():
    """
    Plot 23b as a fraction rather than a count: the share of all galaxies in
    the FFB regime at each snapshot, for Li+24 and MBK25.

    Dividing by the total galaxy count at the same snapshot removes the
    simulation volume, so the curves can be compared across boxes.
    """
    print('Plot 23c: FFB fraction - Li+24 vs MBK25')

    models = [
        {'dir': PRIMARY_DIR,  'label': 'Li+24',  'color': 'black'},
        {'dir': FFB_BK25_SMOOTH_DIR, 'label': 'MBK25',   'color': 'mediumpurple'},
    ]

    fig = plt.figure()
    ax = plt.subplot(111)

    for model in models:
        model_files = find_model_files(model['dir'])
        if not model_files:
            print(f"  No model files found in {model['dir']}")
            continue

        frac_per_snap = []
        redshifts_list = []

        for snap in range(64):
            snap_key = f'Snap_{snap}'
            d = read_snap_from_files(model_files, snap_key, ['FFBRegime'])

            if d and 'FFBRegime' in d and len(d['FFBRegime']) > 0:
                ffb_regime = d['FFBRegime']
                frac = np.sum(ffb_regime == 1) / len(ffb_regime)
            else:
                # No galaxies at this snapshot: undefined rather than zero.
                frac = np.nan

            frac_per_snap.append(frac)
            redshifts_list.append(REDSHIFTS[snap])

        z = np.array(redshifts_list)
        frac_plot = np.array(frac_per_snap, dtype=float)

        z_mask = z <= 15
        z_filtered = z[z_mask]
        frac_filtered = frac_plot[z_mask]

        z_edges = [15.0]
        for i in range(len(z_filtered) - 1):
            mid_point = (z_filtered[i] + z_filtered[i+1]) / 2.0
            z_edges.append(mid_point)
        z_edges.append(0.0)
        z_edges = np.array(z_edges)

        log1pz_edges = np.log10(1 + z_edges)

        ax.stairs(frac_filtered, log1pz_edges, fill=False,
                  label=model['label'], edgecolor=model['color'], linewidth=2)

        finite = np.isfinite(frac_filtered) & (frac_filtered > 0)
        if finite.any():
            print(f"    {model['label']}: peak fraction {np.nanmax(frac_filtered):.3f} "
                  f"at z={z_filtered[np.nanargmax(frac_filtered)]:.2f}")

    ax.set_yscale('log')
    ax.set_ylabel('FFB/MBK25 fraction')
    ax.legend(loc='upper left', frameon=False)

    _ffb_histogram_x_axes(ax)

    fig.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    outputFile = os.path.join(OUTPUT_DIR, 'FFB_Fraction_Li24_vs_MBK25' + OUTPUT_FORMAT)
    plt.savefig(outputFile)
    print(f'Saved file to {outputFile}\n')
    plt.close()


# ========================== PLOT 24: MASS LOADING VS VELOCITY  ==========================

def _feedback_params(directory=PRIMARY_DIR):
    """Feedback parameters straight from the run header, so the analytic curves
    cannot drift from the model that produced the points."""
    fallback = dict(eps_disk=2.9, eps_halo=0.3, alpha_z=1.25, eta_sn=5.0e-3,
                    energy_sn=1.0e51, eps_max=2.0, sn_bound=1, reheat_bound=1)
    files = _find_model_files_early(directory)
    if not files:
        return fallback
    with h5.File(files[0], 'r') as f:
        r = f['Header/Runtime'].attrs
        get = lambda k, d: (float(r[k]) if k in r else d)
        return dict(eps_disk=get('FeedbackReheatingEpsilon', 2.9),
                    eps_halo=get('FeedbackEjectionEfficiency', 0.3),
                    alpha_z=get('RedshiftPowerLawExponent', 1.25),
                    eta_sn=get('EtaSN', 5.0e-3),
                    energy_sn=get('EnergySN', 1.0e51),
                    eps_max=get('MaxSNEnergyCoupling', 2.0),
                    sn_bound=int(get('SNEnergyConservationOn', 1)),
                    # capped_eta_reheat() in src/model_misc.h gates the mass-loading
                    # cap on SNEnergyConservationOn, the same switch as the ejection
                    # coupling -- there is no separate ReheatEnergyConservationOn.
                    # Reading one would silently default to 0 and draw analytic
                    # curves without a cap the model does apply, which shows up as
                    # a spurious offset at high V_vir and high z.
                    reheat_bound=int(get('SNEnergyConservationOn', 1)))


# FIRE (Muratov et al. 2015) critical circular velocity separating the two
# power-law slopes of the wind loading factor. Matches FIRE_V_CRIT_KMS in
# src/model_starformation_and_feedback.c.
FIRE_V_CRIT = 60.0
FIRE_BETA_LOW = -3.2
FIRE_BETA_HIGH = -1.0


def _fire_scaling(vvir, z, p):
    """f(V_vir, z) = (1+z)^alpha (V_vir/60)^beta, the shared factor in both
    the mass loading and the ejection energy."""
    v = np.maximum(np.asarray(vvir, dtype=float), 1.0)
    beta = np.where(v < FIRE_V_CRIT, FIRE_BETA_LOW, FIRE_BETA_HIGH)
    return (1.0 + z) ** p['alpha_z'] * (v / FIRE_V_CRIT) ** beta


def _sn_energy_per_mass_kms2(p):
    """eta_SN * E_SN in (km/s)^2 per unit mass -- the combination the code
    carries as EtaSNcode * EnergySNcode (the Hubble_h factors cancel)."""
    return p['eta_sn'] * p['energy_sn'] / _MSUN_CGS / 1.0e10


def _eta_reheat(vvir, z, p):
    eta = p['eps_disk'] * _fire_scaling(vvir, z, p)
    if p['reheat_bound']:
        esn = _sn_energy_per_mass_kms2(p)
        eta = np.minimum(eta, p['eps_max'] * esn / np.asarray(vvir, float) ** 2)
    return eta


def _eject_per_star(vvir, z, p):
    """mdot_eject / mdot_*, exactly as compute_sn_feedback() evaluates it."""
    esn = _sn_energy_per_mass_kms2(p)
    v = np.asarray(vvir, dtype=float)
    coupling = p['eps_halo'] * _fire_scaling(v, z, p)
    if p['sn_bound']:
        coupling = np.minimum(coupling, p['eps_max'])
    eta = _eta_reheat(v, z, p)
    e_fb = coupling * 0.5 * esn
    e_lift = 0.5 * eta * v ** 2
    
    # Calculate ejection and apply a 1e-5 floor so zeroes aren't dropped
    ej = (e_fb - e_lift) / (0.5 * v ** 2)
    return np.maximum(ej, 1e-5)


def _log10_tick_formatter(decimals=1):
    """Label a log-scaled axis with the log10 of the tick position.

    The axes are log-scaled and the labels say log10(...), so the numbers
    printed must be the exponents (-1, 0, 1, 2), not the values (0.1, 1, 10).
    *decimals* is fixed rather than per-tick so a set like 1.2, 1.4, 1.6, 1.8,
    2.0 does not render its one integral member as a bare "2".
    """
    def fmt(x, _pos):
        return '' if x <= 0 else f'{np.log10(x):.{decimals}f}'
    return FuncFormatter(fmt)


def plot_24_mass_loading_vs_velocity(primary, vanilla):
    """
    Supernova mass loading and ejection against halo virial velocity.

    Both panels show quantities MEASURED from SAGE26 at several redshifts, with
    the analytic expressions as thin reference lines.  Measured rather than
    analytic is the point: Major Comment 2 was "I cannot reproduce the model's
    stated behaviour from the equations", so a curve that satisfies the algebra
    by construction answers nothing.  What the panels demonstrate is that the
    code reproduces the equations, including where the energy bound departs
    from them.

    Panel (a): eta_reheat, the stored MassLoading, which is the value actually
    applied after the reheating bound.  It rises with redshift as (1+z)^1.25.

    Panel (b): mdot_eject / mdot_*, reconstructed per galaxy from the stored
    MassLoading, V_vir and the snapshot redshift, exactly as
    compute_sn_feedback() evaluates it.  It falls to zero at the SAME
    V_vir = V_SN sqrt(eps_halo/eps_disk) at every redshift, because f cancels
    between the feedback energy and the lifting energy.

    The contrast between the two panels is the figure's argument: mass loading
    is strongly redshift dependent, the ejection threshold is not.

    Not captured in panel (b): ejection is additionally limited to the gas
    present in the CGM or hot reservoir, which depends on reservoir state at the
    timestep and cannot be reconstructed from a snapshot.  The measured curve is
    therefore an upper bound at low V_vir.
    """
    print('Plot 24: Mass loading and ejection vs virial velocity')

    p = _feedback_params()
    esn = _sn_energy_per_mass_kms2(p)
    v_eject = np.sqrt(p['eps_halo'] * esn / p['eps_disk'])
    z_targets = [0.0, 1.0, 2.0, 4.0, 6.0]
    # Lower edge of the measured curves.  load_model() applies the MIN_PARTICLES
    # cut, but between 20 and ~30 particles the recorded V_vir and Mvir cease to
    # be mutually consistent and the measured eta falls away from the scaling by
    # up to an order of magnitude.  Truncate where the measurement is
    # trustworthy rather than where galaxies merely exist.
    V_MEASURED_MIN = 24.0
    vbins = np.logspace(np.log10(V_MEASURED_MIN), np.log10(600.0), 24)

    cmap = plt.get_cmap('plasma')
    colours = [cmap(x) for x in np.linspace(0.05, 0.85, len(z_targets))]

    # figsize and fonts match the other 1x2 figures (plot_9, plot_7b), which
    # take the stylesheet raw at this canvas size.  A smaller canvas with the
    # same absolute font sizes renders the text larger relative to the axes.
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))

    # Each simulation has its own snapshot grid, so a target redshift maps to a
    # different actual z in each.  Two targets can also collide on one snapshot
    # if the grid is coarse at high z, which would draw two identical curves
    # under different labels.  Report the mapping and drop duplicates, so a
    # coincidence in the snapshot grid is never mistaken for a physical result.
    _drawn_snaps = {}
    _sat_marks = []
    print('  Fig. 8 redshift mapping (target -> snapshot -> actual z):')

    # The secondary simulation picks the redshift and the primary is matched to
    # it, rather than both snapping independently to the nominal target.  Taking
    # the nearest snapshot to the target in each simulation separately put the
    # two curves labelled z=6 at z=6.197 (Millennium) and z=5.722 (miniUchuu):
    # eta goes as (1+z)^alpha, so that Dz=0.475 alone offsets them by 8%, which
    # reads as a resolution difference and is nothing of the kind.  Millennium
    # Snap_19 sits at z=5.724, so matching on the secondary's redshift brings
    # the pair to Dz=0.002.  The analytic curve is drawn at the matched primary
    # redshift for the same reason.
    _have_secondary = model_files_exist(MINIUCHUU_DIR)

    for z_t, c in zip(z_targets, colours):
        if _have_secondary:
            snap2 = _snap_nearest_z(MINIUCHUU_REDSHIFTS, z_t)
            z_ref = MINIUCHUU_REDSHIFTS[snap2]
        else:
            snap2, z_ref = None, z_t
        snap = _snap_nearest_z(REDSHIFTS, z_ref)
        z = REDSHIFTS[snap]
        if snap in _drawn_snaps:
            print(f'    z={z_t:.0f} -> Snap_{snap} -> z={z:.3f}  SKIPPED, same'
                  f' snapshot as target z={_drawn_snaps[snap]:.0f}')
            continue
        _drawn_snaps[snap] = z_t
        # Label the redshift actually plotted, not the target it was chosen
        # from, so a matched pair at z=5.72 is never printed as "z = 6".
        _zlab = (rf'$z = {z:.0f}$' if abs(z - round(z)) < 0.1
                 else rf'$z = {z:.1f}$')
        print(f'    z={z_t:.0f} -> Snap_{snap} -> z={z:.3f} (primary)')

        # Where the ejection coupling saturates.  sn_energy_coupling() caps
        # eps_halo*f at eps_max, so E_FB never exceeds the whole supernova
        # budget -- but Eq. 14 as printed is unbounded, and left of these ticks
        # the plotted curve is the capped value, not what the equation gives.
        # Collected here and drawn after set_xlim(): tested against the axis
        # limits in-loop, the first redshift is compared with matplotlib's
        # default (0, 1) and silently dropped.
        if p['sn_bound']:
            _fc = p['eps_max'] / p['eps_halo']
            _fz = (1.0 + z) ** p['alpha_z']
            _sat_marks.append(
                ((60.0 * (_fz / _fc)) if _fz >= _fc
                 else 60.0 * (_fc / _fz) ** (-1.0 / 3.2), c))
        # No analytic reference curves in either panel.  Both quantities follow
        # their scalings by construction -- capped_eta_reheat() stores exactly
        # eps_disk*f, and the ejected mass is reconstructed from it -- so an
        # overlaid analytic curve restates the input rather than testing it.
        try:
            d = load_model(PRIMARY_DIR, snapshot=f'Snap_{snap}',
                           properties=['MassLoading', 'Vvir'])
        except Exception:                                       # noqa: BLE001
            continue
        vv = np.asarray(d['Vvir'], float)
        eta = np.asarray(d['MassLoading'], float)
        w = (eta > 0) & (vv > 0)
        if w.sum() < 100:
            continue
        plot_binned_median_1sigma(
            axL, vv[w], eta[w], vbins, color=c,
            label=_zlab, alpha=0.18, lw=2.2, min_count=50,
            zorder_fill=Z_MODEL_BAND, zorder_line=Z_MODEL_LINE)

        # Second simulation, dashed and without a shaded band so that ten
        # curves per panel stay readable.  MINIUCHUU_DIR must point at the
        # miniUchuu production output for the label below to be accurate.
        if snap2 is not None:
            z2 = z_ref
            # Residual mismatch after matching.  eta goes as (1+z)^alpha, so
            # report what the leftover Dz is worth: anything above a couple of
            # per cent is a redshift effect, not resolution.
            _off = 100.0 * (((1.0 + z2) / (1.0 + z)) ** p['alpha_z'] - 1.0)
            print(f'    z={z_t:.0f} -> Snap_{snap2} -> z={z2:.3f} (miniUchuu)'
                  f'   dz={z2 - z:+.3f} vs primary'
                  + (f'  ==> {_off:+.1f}% offset in eta from redshift alone'
                     if abs(_off) > 2.0 else '  (matched)'))
            try:
                d2 = load_model(MINIUCHUU_DIR, snapshot=f'Snap_{snap2}',
                                properties=['MassLoading', 'Vvir'])
                v2 = np.asarray(d2['Vvir'], float)
                e2 = np.asarray(d2['MassLoading'], float)
                w2 = (e2 > 0) & (v2 > 0)
                if w2.sum() >= 100:
                    plot_binned_median_1sigma(
                        axL, v2[w2], e2[w2], vbins, color=c, label=None,
                        ls='--', alpha=0.0, lw=1.8, min_count=50,
                        zorder_fill=Z_MODEL_BAND, zorder_line=Z_MODEL_LINE)
                    c2 = p['eps_halo'] * _fire_scaling(v2[w2], MINIUCHUU_REDSHIFTS[snap2], p)
                    if p['sn_bound']:
                        c2 = np.minimum(c2, p['eps_max'])
                    ef2 = c2 * 0.5 * esn
                    el2 = 0.5 * e2[w2] * v2[w2] ** 2
                    
                    # Retain all galaxies, setting zero/negative ejection to 1e-5
                    j2 = (ef2 - el2) / (0.5 * v2[w2] ** 2)
                    j2 = np.maximum(j2, 1e-5)
                    k2 = np.isfinite(j2)
                    
                    if k2.sum() > 100:
                        plot_binned_median_1sigma(
                            axR, v2[w2][k2], j2[k2], vbins, color=c, label=None,
                            ls='--', alpha=0.0, lw=1.8, min_count=50,
                            zorder_fill=Z_MODEL_BAND, zorder_line=Z_MODEL_LINE)
            except Exception:                                   # noqa: BLE001
                pass

        # mdot_eject / mdot_*, reconstructed exactly as the code evaluates it,
        # but using the STORED eta rather than the analytic one, so the
        # reheating bound is included as applied.
        coupling = p['eps_halo'] * _fire_scaling(vv[w], z, p)
        if p['sn_bound']:
            coupling = np.minimum(coupling, p['eps_max'])
        e_fb = coupling * 0.5 * esn
        e_lift = 0.5 * eta[w] * vv[w] ** 2
        
        # Retain all galaxies, setting zero/negative ejection to 1e-5
        ej = (e_fb - e_lift) / (0.5 * vv[w] ** 2)
        ej = np.maximum(ej, 1e-5)
        ok = np.isfinite(ej)
        
        if ok.sum() > 100:
            plot_binned_median_1sigma(
                axR, vv[w][ok], ej[ok], vbins, color=c, label=None,
                alpha=0.18, lw=2.2, min_count=50,
                zorder_fill=Z_MODEL_BAND, zorder_line=Z_MODEL_LINE)

    # observations, left panel only
    obs_handles = []
    for fname, marker, lbl in [('Chisholm_17_ml.csv', 'o', 'Chisholm+17'),
                               ('Heckman_15_ml.csv', 'X', 'Heckman+15'),
                               ('Rupke_05_ml.csv', 's', 'Rupke+05'),
                               ('Sugahara_17_ml.csv', 'd', 'Sugahara+17')]:
        fp = os.path.join('./data/outflows', fname)
        if not os.path.exists(fp):
            continue
        try:
            # these .csv files are whitespace/tab separated despite the suffix
            dd = np.loadtxt(fp, unpack=True)
        except Exception:                                       # noqa: BLE001
            continue
        h = axL.scatter(dd[0], dd[1], marker=marker, s=45, color='k',
                        alpha=0.6, label=lbl, zorder=Z_OBS)
        obs_handles.append(h)

    axL.axvline(FIRE_V_CRIT, color='0.55', ls=':', lw=1.4, zorder=1)
    # Keep this inside axL's y-range: it was pinned at 1.6e3 and disappeared
    # when the limit came down to 500.
    # In-panel annotations take the legend size, not font.size: at the
    # stylesheet default (20) they render as large as the axis labels and
    # dominate the panel.  Still stylesheet-driven, just the smaller of the
    # two values it defines.
    _ann_fs = plt.rcParams['legend.fontsize']
    axL.annotate(rf'$V_{{\rm vir}}={FIRE_V_CRIT:.0f}$ km s$^{{-1}}$',
                 xy=(FIRE_V_CRIT, 260), xytext=(3, 0), fontsize=_ann_fs,
                 textcoords='offset points', color='0.4')
    axR.axvline(v_eject, color='crimson', ls='--', lw=1.4, zorder=1)
    axR.annotate(rf'$\dot{{E}}_{{\rm FB}}=\dot{{E}}_{{\rm lift}}$'
                 '\n' rf'$V_{{\rm vir}}={v_eject:.0f}$ km s$^{{-1}}$',
                 xy=(v_eject, 9.0e2), xytext=(-6, 0), ha='right',
                 fontsize=_ann_fs,
                 textcoords='offset points', color='crimson')

    for a in (axL, axR):
        a.set_xscale('log'); a.set_yscale('log')
        a.set_xlabel(r'$\rm \log_{10}\,V_{vir}\,[km\,s^{-1}]$')
        # Plain values on both axes rather than 10^n, matching the rest of the
        # paper.  The explicit tick lists below control which ones appear.
        a.xaxis.set_major_formatter(_log10_tick_formatter(1))
        a.yaxis.set_major_formatter(_log10_tick_formatter(0))
        a.set_xticks([], minor=True)
        a.set_yticks([], minor=True)
    axL.set_xlim(15, 600); axL.set_ylim(0.05, 500)
    axL.set_xticks([10**e for e in (1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6)])
    axL.set_yticks([10**e for e in (-1, 0, 1, 2)])
    axR.set_xlim(20, 200); axR.set_ylim(1e-2, 1e4)
    # for _v, _c in _sat_marks:
    #     if 20 <= _v <= 200:
    #         axR.axvline(_v, ymin=0.0, ymax=0.05, color=_c, lw=2.4,
    #                     solid_capstyle='butt', zorder=Z_MODEL_LINE)
    axR.set_xticks([10**e for e in (1.4, 1.6, 1.8, 2.0, 2.2)])
    axR.set_yticks([10**e for e in (-2, -1, 0, 1, 2, 3, 4)])
    # Both are mass ratios -- Msun of gas per Msun of stars formed -- so they
    # are dimensionless and carry no unit in brackets.
    # \eta must stay OUTSIDE \rm: mathtext has no Greek glyph in the roman
    # font, so "\rm \eta" renders as a fallback box.  Roman subscript only.
    # \rm goes after the \eta so the Greek stays italic and everything from the
    # "=" onwards is upright, matching axR.
    axL.set_ylabel(r'$\rm \log_{10}\,\dot{m}_{reheat}/\dot{m}_{*}$')
    axR.set_ylabel(r'$\rm \log_{10}\,\dot{m}_{eject}/\dot{m}_{*}$')
    # No hardcoded fontsize anywhere in this figure: sizes come from
    # kieren_cohare_palatino_sty.mplstyle (legend.fontsize 14, font.size 20)
    # so this figure matches every other panel in the paper.
    # Simulation key lives in the ejection panel, lower left, where nothing is
    # drawn.  The left panel already carries the redshift key (upper right) and
    # the observations key (lower left), and a third block there is one too
    # many.  Proxy artists, since every measured curve is drawn label=None.
    if model_files_exist(MINIUCHUU_DIR):
        axR.plot([], [], '-',  color='0.3', lw=2.2, label='Millennium')
        axR.plot([], [], '--', color='0.3', lw=1.8, label='miniUchuu')
    model_handles = [h for h in axL.get_legend_handles_labels()[0]
                     if h not in obs_handles]
    model_labels = [l for h, l in zip(*axL.get_legend_handles_labels())
                    if h not in obs_handles]
    leg1 = _standard_legend(axL, loc='upper right',
                           handles=model_handles, labels=model_labels)
    axL.add_artist(leg1)
    if obs_handles:
        _standard_legend(axL, loc='lower left', handles=obs_handles,
                         labels=[h.get_label() for h in obs_handles])
    _standard_legend(axR, loc='lower left')

    fig.tight_layout()
    outputFile = os.path.join(OUTPUT_DIR, 'MassLoading_vs_Velocity' + OUTPUT_FORMAT)
    save_figure(fig, outputFile)

# ======================== GAS RATIO PLOTS =========================

_GAS_MODELS = [
    {'dir': PRIMARY_DIR,  'label': 'SAGE26 (BR06)', 'color': 'black'},
    {'dir': GD14_DIR,     'label': 'GD14',          'color': 'goldenrod'},
    {'dir': KD12_DIR,     'label': 'KD12',          'color': 'steelblue'},
    {'dir': KMT09_DIR,    'label': 'KMT09',         'color': 'limegreen'},
    {'dir': K13_DIR,      'label': 'K13',            'color': 'firebrick'},
]

GAS_OBS_DIR = os.path.join(OBS_DIR, 'Gas')


def _gas_ratio_plot(gas_prop, obs_file, obs_label, ylabel, output_name):
    """
    Generic gas-mass-ratio comparison plot.

    Plots log10(gas_prop / M_*) vs log10(M_*) for each H2 model,
    with a 2D density contour for the primary model (SAGE26) and
    median lines with bootstrap error bands for all models.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i, model in enumerate(_GAS_MODELS):
        dirpath = model['dir']
        if not model_files_exist(dirpath):
            print(f"  Skipping {model['label']}: directory not found")
            continue

        data = load_model(dirpath, properties=['StellarMass', gas_prop])
        mstar = data['StellarMass']
        gas = data[gas_prop]

        valid = (mstar > 1e8) & (gas > 0)
        mstar = mstar[valid]
        gas = gas[valid]

        log_mstar = np.log10(mstar)
        log_ratio = np.log10(gas / mstar)

        # Sigma contour for primary model only
        if i == 0:
            X, Y, Z = density_contour(log_mstar, log_ratio,
                                      bins=[np.linspace(8.0, 12.0, 101),
                                            np.linspace(-3.0, 1.0, 101)])
            if Z.max() > 0:
                lvls = sigma_contour_levels(Z)
                if lvls is not None:
                    ax.contourf(X, Y, Z, levels=lvls, cmap='Blues_r', alpha=0.6)
                    ax.contour(X, Y, Z, levels=lvls, colors='steelblue',
                               linestyles='-', alpha=1.0, linewidths=1.5)

        # Median line with bootstrap errors
        bin_width = 0.2
        mass_bins = np.arange(8.0, 12.0 + bin_width, bin_width)
        mass_centers = mass_bins[:-1] + bin_width / 2

        median_ratio = np.full_like(mass_centers, np.nan)
        p16 = np.full_like(mass_centers, np.nan)
        p84 = np.full_like(mass_centers, np.nan)
        n_bootstrap = 1000
        rng = np.random.default_rng(42)

        for j in range(len(mass_bins) - 1):
            mask = (log_mstar >= mass_bins[j]) & (log_mstar < mass_bins[j + 1])
            bindata = log_ratio[mask]
            if bindata.size > 0:
                median_ratio[j] = np.median(bindata)
                boot_meds = np.array([
                    np.median(rng.choice(bindata, size=bindata.size, replace=True))
                    for _ in range(n_bootstrap)
                ])
                p16[j] = np.percentile(boot_meds, 16)
                p84[j] = np.percentile(boot_meds, 84)

        good = ~np.isnan(median_ratio)
        lw = 3.5 if i == 0 else 2
        ax.plot(mass_centers[good], median_ratio[good],
                label=model['label'], color=model['color'], lw=lw, zorder=5)
        ax.fill_between(mass_centers[good], p16[good], p84[good],
                        color=model['color'], alpha=0.2, zorder=4)

    # Observational data
    obs_path = os.path.join(GAS_OBS_DIR, obs_file)
    if os.path.exists(obs_path):
        obs = np.loadtxt(obs_path)
        log_ms = obs[:, 0]
        med = obs[:, 1]
        op16 = obs[:, 2]
        op84 = obs[:, 3]
        omask = (med > -10) & (med < 2) & (op16 > -10) & (op84 > -10)
        yerr_lo = np.abs(med[omask] - op16[omask])
        yerr_hi = np.abs(op84[omask] - med[omask])
        ax.errorbar(log_ms[omask], med[omask], yerr=[yerr_lo, yerr_hi],
                    fmt='o', color='k', markersize=8,
                    label=obs_label, zorder=10, linewidth=1.0,
                    markerfacecolor='gray', markeredgecolor='k',
                    markeredgewidth=1.0, alpha=0.6)

    ax.set_xlim(8, 12)
    ax.set_ylim(-3, 1)
    ax.set_xlabel(r'$\log_{10}\ m_{\mathrm{*}}\ [M_{\odot}]$')
    ax.set_ylabel(ylabel)

    handles, labels = ax.get_legend_handles_labels()
    n_items = len(handles)
    ax.legend(handles, labels, loc='upper center',
              bbox_to_anchor=(0.5, -0.18), ncol=n_items/2, frameon=False)
    
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))

    fig.subplots_adjust(bottom=0.22)
    outputFile = os.path.join(OUTPUT_DIR, output_name + OUTPUT_FORMAT)
    save_figure(fig, outputFile)


def plot_25_hi_mass_ratio():
    """HI-to-stellar mass ratio comparison across H2 models."""
    print('Plot 25: HI Mass Ratio Comparison')
    _gas_ratio_plot(
        gas_prop='H1gas',
        obs_file='HIGasRatio_NonDetEQZero.dat',
        obs_label='xGASS',
        ylabel=r'$\log_{10}\ (m_{\mathrm{HI}} / m_{\mathrm{*}})$',
        output_name='HI_Mass_Ratio',
    )


def plot_26_h2_mass_ratio():
    """H2-to-stellar mass ratio comparison across H2 models."""
    print('Plot 26: H2 Mass Ratio Comparison')
    _gas_ratio_plot(
        gas_prop='H2gas',
        obs_file='MolecularGasRatio_NonDetEQZero.dat',
        obs_label='xCOLDGASS',
        ylabel=r'$\log_{10}\ (m_{\mathrm{H2}} / m_{\mathrm{*}})$',
        output_name='H2_Mass_Ratio',
    )


def plot_27_cold_gas_mass_ratio():
    """Cold-gas-to-stellar mass ratio comparison across H2 models."""
    print('Plot 27: Cold Gas Mass Ratio Comparison')
    _gas_ratio_plot(
        gas_prop='ColdGas',
        obs_file='NeutralGasRatio_NonDetEQZero.dat',
        obs_label='xGASS',
        ylabel=r'$\log_{10}\ (m_{\mathrm{cold\ gas}} / m_{\mathrm{*}})$',
        output_name='Cold_Gas_Mass_Ratio',
    )


# ==================== MDOT PLOTS ====================

# Panels at z = 0, 1, 2, 3, 4.  The output table has no snapshot exactly at
# these redshifts, so take the nearest one and label it by the round target.
_MDOT_Z_TARGETS = [0.0, 1.0, 2.0, 3.0, 4.0]

_MDOT_SNAP_PANELS = [
    (_snap_nearest_z(REDSHIFTS, z), f'z = {z:.0f}')
    for z in _MDOT_Z_TARGETS
]

_MDOT_PROPS = ['Mvir', 'Vvir', 'Type', 'mdot_cool', 'mdot_stream']


def _plot_mdot_panels(x_prop, x_label, xlim, xbins, output_name,
                      upper_axis=None):
    """
    Generic multi-panel mdot_cool / mdot_stream plot.

    Parameters
    ----------
    x_prop : str
        Property for x-axis ('Mvir' or 'Vvir').
    x_label : str
        LaTeX x-axis label.
    xlim : tuple
        (xmin, xmax) for x-axis.
    xbins : array
        Bin edges for binned_median.
    output_name : str
        Output filename stem.
    upper_axis : callable or None
        If given, called as upper_axis(ax) to add a twin top axis.
    """
    snap_nums = [s for s, _ in _MDOT_SNAP_PANELS]
    snapdata = load_snapshots(PRIMARY_DIR, snap_nums, _MDOT_PROPS)

    print('  panels: ' + ', '.join(
        f'{lbl} (Snap_{s}, z={REDSHIFTS[s]:.3f})'
        for s, lbl in _MDOT_SNAP_PANELS))

    nrows = len(_MDOT_SNAP_PANELS)

    # This figure keeps a compact canvas (7 in wide, 3.5 in per panel) rather
    # than the 8 x 6-in-per-panel stacked geometry used elsewhere, so the
    # stylesheet's absolute font sizes are scaled down by the corresponding
    # linear factor -- sqrt((7 * 3.5) / (8 * 6)) -- to render text at the same
    # size relative to the axes as in the other figures.
    with plt.rc_context(_scaled_font_rc(0.71)):
        fig, axes = plt.subplots(nrows, 1, figsize=(7, 3.5 * nrows),
                                 sharex=True)
        if nrows == 1:
            axes = [axes]

        for idx, (snap, zlabel) in enumerate(_MDOT_SNAP_PANELS):
            ax = axes[idx]

            if snap not in snapdata:
                ax.text(0.5, 0.5, f'{zlabel}: no data', transform=ax.transAxes,
                        ha='center', va='center')
                continue

            d = snapdata[snap]
            xval = d[x_prop]
            mdot_cool = d.get('mdot_cool')
            mdot_stream = d.get('mdot_stream')

            central = (d.get('Type', np.zeros_like(xval)) == 0) & (xval > 0)
            log_x = np.log10(xval[central])

            # mdot_cool
            if mdot_cool is not None:
                mc = mdot_cool[central]
                pos = mc > 0
                if np.sum(pos) > 0:
                    log_mc = np.log10(mc[pos])
                    c, med, p25, p75 = binned_median(log_x[pos], log_mc, xbins)
                    valid = np.isfinite(med)
                    ax.plot(c[valid], med[valid], color='C3', lw=2.2,
                            label=r'$\dot{M}_{\rm cool}$')
                    ax.fill_between(c[valid], p25[valid], p75[valid],
                                    color='C3', alpha=0.2)

            # mdot_stream
            if mdot_stream is not None:
                ms = mdot_stream[central]
                pos = ms > 0
                if np.sum(pos) > 0:
                    log_ms = np.log10(ms[pos])
                    c, med, p25, p75 = binned_median(log_x[pos], log_ms, xbins)
                    valid = np.isfinite(med)
                    ax.plot(c[valid], med[valid], color='C0', lw=2.2,
                            label=r'$\dot{M}_{\rm stream}$')
                    ax.fill_between(c[valid], p25[valid], p75[valid],
                                    color='C0', alpha=0.2)

            ax.set_ylabel(r'$\log_{10}\,\dot{m}_{\mathrm{cool}}\ [M_{\odot}\,\mathrm{yr}^{-1}]$')
            ax.set_xlim(*xlim)
            ax.text(0.05, 0.92, zlabel, transform=ax.transAxes, va='top')
            # ax.tick_params(axis='y')  # Use style sheet for y-axis ticks
            ax.set_ylim(-1, 3.5)

            if idx == 0:
                _standard_legend(ax, loc='lower right')

        axes[-1].set_xlabel(x_label)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))

        if x_prop == 'Vvir':
            ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))

        # Optional upper axis on top panel
        if upper_axis is not None:
            upper_axis(axes[0])

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.001)

        outputFile = os.path.join(OUTPUT_DIR, output_name + OUTPUT_FORMAT)
        save_figure(fig, outputFile)


def plot_28_mdot_vs_mvir():
    """Multi-panel mdot_cool and mdot_stream vs Mvir."""
    print('Plot 28: mdot vs Mvir')
    _plot_mdot_panels(
        x_prop='Mvir',
        x_label=r'$\log_{10}\ (M_{\rm vir}\ /\ M_{\odot})$',
        xlim=(9.5, 14.5),
        xbins=np.arange(9.5, 14.5, 0.2),
        output_name='Mdot_vs_Mvir',
    )


def _add_tvir_axis(ax):
    """Add a log10(Tvir) upper axis given a log10(Vvir) lower axis."""
    ax_top = ax.twiny()
    vmin, vmax = ax.get_xlim()
    # Tvir = 35.9 * Vvir^2  =>  log10(Tvir) = log10(35.9) + 2*log10(Vvir)
    tmin = np.log10(35.9) + 2 * vmin
    tmax = np.log10(35.9) + 2 * vmax
    ax_top.set_xlim(tmin, tmax)
    ax_top.set_xlabel(r'$\log_{10}\ T_{\rm vir}\ [\mathrm{K}]$', labelpad=20)


def plot_29_mdot_vs_vvir():
    """Multi-panel mdot_cool and mdot_stream vs Vvir with Tvir upper axis."""
    print('Plot 29: mdot vs Vvir')
    _plot_mdot_panels(
        x_prop='Vvir',
        x_label=r'$\log_{10}\ V_{\rm vir}\ [\mathrm{km\,s}^{-1}]$',
        xlim=(1.6, 3.0),
        xbins=np.arange(1.6, 3.0, 0.1),
        output_name='Mdot_vs_Vvir',
        upper_axis=_add_tvir_axis,
    )
    print_mdot_panel_stats(x_prop='Vvir')

def _read_runtime_attrs(directory):
    """Runtime parameters from the HDF5 header of the first model file.

    Returns an empty dict if the directory or the group is missing, so callers
    can fall back for runs written before a parameter existed.
    """
    files = _find_model_files_early(directory)
    if not files:
        return {}
    try:
        with h5.File(files[0], 'r') as f:
            out = {}
            for k, v in f['Header/Runtime'].attrs.items():
                out[k] = v.decode() if isinstance(v, bytes) else v
            return out
    except Exception:
        return {}


# Press-Schechter clustering mass M_*(z), log10 in Msun, on the same grid and
# for the same two cosmologies as interpolate_clustering_mass() in
# src/model_halo_properties.c.  Mirrored here so the diagnostics can report the
# Dekel & Birnboim (2006) stream ceiling without re-deriving sigma(M).
_MSTAR_Z = np.arange(0.0, 15.5, 0.5)
_MSTAR_MILL = np.array([
    12.8964, 12.2501, 11.5685, 10.9046, 10.2717, 9.6699, 9.0969, 8.5486,
    8.0214, 7.5126, 7.0206, 6.5435, 6.0794, 5.6266, 5.1842, 4.7514,
    4.3276, 3.9122, 3.5045, 3.1039, 2.7098, 2.3217, 1.9393, 1.5626,
    1.1912, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])
_MSTAR_UCHUU = np.array([
    12.8028, 12.0672, 11.3101, 10.5795, 9.8831, 9.2194, 8.5834, 7.9701,
    7.3764, 6.8002, 6.2389, 5.6901, 5.1521, 4.6241, 4.1054, 3.5950,
    3.0915, 2.5943, 2.1027, 1.6166, 1.1359, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])


def interpolate_clustering_mass_py(z):
    """log10 M_*(z) in Msun, selected by Omega as the model code does."""
    table = _MSTAR_UCHUU if abs(OMEGA_M - 0.3089) < 0.01 else _MSTAR_MILL
    return float(np.interp(z, _MSTAR_Z, table))


def print_mdot_panel_stats(x_prop='Vvir'):
    """Per-panel cold-stream diagnostics for the mdot figures.

    Reports, for the hot-regime centrals that actually carry a stream channel:
      N_hot        hot-regime centrals in the panel
      f_str>0      fraction with a non-zero stream rate
      stream frac  stream share of total reservoir-fed accretion
      M_stream     Dekel & Birnboim (2006) ceiling Mshock^2/(f Mstar), and
                   whether the stream window above Mshock is open

    The header echoes ColdStreamCeilingOn and StreamMassFactor from the run so
    it is unambiguous which prescription produced the figure.
    """
    props = _MDOT_PROPS + ['Regime']
    snap_nums = [s for s, _ in _MDOT_SNAP_PANELS]
    snapdata = load_snapshots(PRIMARY_DIR, snap_nums, props)

    rt = _read_runtime_attrs(PRIMARY_DIR)
    ceiling = rt.get('ColdStreamCeilingOn', '?')
    fstream_f = rt.get('StreamMassFactor', '?')
    mshock = rt.get('MShockMsun', 6.0e11)
    print('\n  cold-stream diagnostics  (ColdStreamCeilingOn=%s, StreamMassFactor=%s, '
          'Mshock=%.2e)' % (ceiling, fstream_f, mshock))

    xlabel = 'log Vvir' if x_prop == 'Vvir' else 'log Mvir'
    for snap, zlabel in _MDOT_SNAP_PANELS:
        d = snapdata.get(snap)
        if d is None:
            print('    %-9s no data' % zlabel)
            continue
        z = REDSHIFTS[snap]
        mvir = d['Mvir'] * MASS_CONVERT
        hot = (d.get('Type', np.zeros_like(mvir)) == 0) & (mvir > 0)
        if 'Regime' in d:
            hot &= (d['Regime'] == 1)
        ms = d.get('mdot_stream')
        mc = d.get('mdot_cool')
        if ms is None or mc is None or hot.sum() < 5:
            print('    %-9s too few hot-regime centrals' % zlabel)
            continue

        # DB06 ceiling for context, whichever prescription is running.
        window = ''
        try:
            mstar = 10.0 ** interpolate_clustering_mass_py(z)
            f_fac = float(fstream_f) if fstream_f != '?' else 3.0
            m_ceil = mshock * mshock / (f_fac * mstar)
            window = ('M_stream=%.2e (%.2f Mshock, %s)'
                      % (m_ceil, m_ceil / mshock,
                         'open' if m_ceil > mshock else 'closed'))
        except Exception:
            pass

        st, co = ms[hot], mc[hot]
        tot = st.sum() + co.sum()
        print('    %-9s N_hot=%6d  f_str>0=%4.0f%%  stream frac=%4.0f%%   %s'
              % (zlabel, hot.sum(), 100.0 * np.mean(st > 0),
                 100.0 * st.sum() / tot if tot > 0 else 0.0, window))

        # Trend across the panel's x-axis, which is what the curves show.
        xv = d[x_prop][hot]
        with np.errstate(divide='ignore', invalid='ignore'):
            lx = np.log10(xv)
        edges = np.percentile(lx[np.isfinite(lx)], [0, 20, 40, 60, 80, 100])
        cells = []
        for i in range(len(edges) - 1):
            m = (lx >= edges[i]) & (lx < edges[i + 1] if i < len(edges) - 2
                                    else lx <= edges[i + 1])
            if m.sum() < 5:
                cells.append('  --  ')
                continue
            t = st[m].sum() + co[m].sum()
            cells.append('%3.0f%%  ' % (100.0 * st[m].sum() / t if t > 0 else 0.0))
        print('              %s quintiles %s'
              % (xlabel, ' '.join('%.2f' % e for e in edges[:-1])))
        print('              stream frac %s' % ' '.join(cells))
    print()


# ========================== MDOT RATIO STATISTICS ==========================

def print_mdot_stream_cool_stats():
    print("\n==== mdot_stream / mdot_cool statistics by halo mass ====")
    mass_bins = np.arange(10.0, 16.0, 0.2)
    snap_nums = [SNAP_Z0, SNAP_Z1, SNAP_Z2, SNAP_Z3, SNAP_Z4, SNAP_Z5, SNAP_Z10]
    snap_labels = [f"z = {REDSHIFTS[s]:.1f}" for s in snap_nums]
    snapdata = load_snapshots(PRIMARY_DIR, snap_nums, _MDOT_PROPS)

    for snap, zlabel in zip(snap_nums, snap_labels):
        d = snapdata.get(snap)
        if d is None:
            print(f"{zlabel}: No data.")
            continue
        mvir = d['Mvir']
        mdot_cool = d.get('mdot_cool')
        mdot_stream = d.get('mdot_stream')
        types = d.get('Type', np.zeros_like(mvir))
        central = (types == 0) & (mvir > 0)
        mvir = mvir[central]
        mc = mdot_cool[central]
        ms = mdot_stream[central]
        log_mvir = np.log10(mvir)
        print(f"\n--- {zlabel} ---")
        for i in range(len(mass_bins) - 1):
            mask = (log_mvir >= mass_bins[i]) & (log_mvir < mass_bins[i+1])
            N = np.sum(mask)
            if N < 5:
                continue
            mc_bin = mc[mask]
            ms_bin = ms[mask]
            mean_mass = np.mean(np.log10(mvir[mask])) if N > 0 else np.nan
            sum_stream = np.sum(ms_bin)
            sum_cool = np.sum(mc_bin)
            pop_norm_ratio = sum_stream / (sum_stream + sum_cool) if (sum_stream + sum_cool) > 0 else np.nan
            # For all centrals in the bin, percent where streaming dominates, percent where cooling dominates
            # (streaming dominates: ms_bin > mc_bin, cooling dominates: mc_bin > ms_bin, ignore cases where both are zero)
            valid = (mc_bin > 0) | (ms_bin > 0)
            n_valid = np.sum(valid)
            pct_stream_dom = 100.0 * np.sum((ms_bin > mc_bin) & valid) / n_valid if n_valid > 0 else np.nan
            pct_cool_dom = 100.0 * np.sum((mc_bin > ms_bin) & valid) / n_valid if n_valid > 0 else np.nan
            print(f"z={zlabel}  mean_logM={mean_mass:.2f}  pop_norm_ratio={pop_norm_ratio:.3f}  %stream_dom={pct_stream_dom:5.1f}%  %cool_dom={pct_cool_dom:5.1f}%")

# ========================== HIGH-Z MASSIVE GALAXY STATS ==========================

def print_massive_galaxy_stats():
    """Print properties of massive galaxies (M* > 10^9.5) at z = 4-6."""
    print("\n==== Massive galaxy properties at z = 4-6 (M* > 10^9.5 Msun) ====\n")

    props = ['StellarMass', 'Mvir', 'ColdGas', 'H2gas',
             'MassLoading', 'MetalsColdGas', 'BlackHoleMass', 'Type',
             'SfrDisk', 'SfrBulge', 'Vvir', 'Regime', 'EjectedMass']
    mass_cut = 10**9.5

    model_files = find_model_files(PRIMARY_DIR)
    if not model_files:
        print(f"  No model files found in {PRIMARY_DIR}")
        return

    for snap in range(len(REDSHIFTS)):
        z = REDSHIFTS[snap]
        if z < 4.0 or z > 6.0:
            continue

        snap_key = f'Snap_{snap}'
        data = read_snap_from_files(model_files, snap_key, props)
        if not data:
            continue

        mstar = data.get('StellarMass')
        if mstar is None:
            continue

        gal_type = data.get('Type', np.zeros_like(mstar))
        mask = (mstar > mass_cut) & (gal_type == 0)
        n_gal = np.sum(mask)
        if n_gal == 0:
            print(f"  Snap {snap} (z = {z:.3f}): 0 galaxies above cut\n")
            continue

        print(f"  Snap {snap} (z = {z:.3f}): {n_gal} galaxies with M* > 10^9.5 Msun")
        print(f"  {'#':>3s}  {'log M*':>8s}  {'log Mhalo':>9s}  {'Vvir':>7s}  {'log Mcold':>9s}  "
              f"{'log MH2':>8s}  {'log Meject':>10s}  {'SFR':>8s}  {'eta_rh':>7s}  {'12+log(O/H)':>11s}  {'log MBH':>8s}  {'Regime':>6s}")
        print(f"  {'':->3s}  {'':->8s}  {'':->9s}  {'':->7s}  {'':->9s}  "
              f"{'':->8s}  {'':->10s}  {'':->8s}  {'':->7s}  {'':->11s}  {'':->8s}  {'':->6s}")

        # Top 10 most massive CGM regime galaxies
        reg = data.get('Regime')
        idx = np.where(mask)[0]
        if reg is not None:
            idx = idx[reg[idx] == 0]
        idx = idx[np.argsort(-mstar[idx])][:10]

        for i, gi in enumerate(idx):
            log_ms = np.log10(mstar[gi])
            mvir = data.get('Mvir')
            log_mh = np.log10(mvir[gi]) if mvir is not None and mvir[gi] > 0 else np.nan
            cg = data.get('ColdGas')
            log_cg = np.log10(cg[gi]) if cg is not None and cg[gi] > 0 else np.nan
            h2 = data.get('H2gas')
            log_h2 = np.log10(h2[gi]) if h2 is not None and h2[gi] > 0 else np.nan
            ml = data.get('MassLoading')
            eta = ml[gi] if ml is not None else np.nan
            mcg = data.get('MetalsColdGas')
            if mcg is not None and cg is not None and cg[gi] > 0:
                z_met = mcg[gi] / cg[gi]
                # 12 + log10(O/H) assuming O is ~0.5 of metals by mass, H is 0.75 of gas
                # Simplified: 12 + log10(Z/Z_sun) + 8.69 (solar 12+log(O/H))
                oh12 = 12.0 + np.log10(z_met / Z_SUN) + np.log10(10**(8.69 - 12.0))
                # Or more directly: 12+log(O/H) = log10(Z/Zsun) + 8.69
                oh12 = np.log10(z_met / Z_SUN) + 8.69
            else:
                oh12 = np.nan
            sfrd = data.get('SfrDisk')
            sfrb = data.get('SfrBulge')
            sfr_val = 0.0
            if sfrd is not None:
                sfr_val += sfrd[gi]
            if sfrb is not None:
                sfr_val += sfrb[gi]
            bh = data.get('BlackHoleMass')
            log_bh = np.log10(bh[gi]) if bh is not None and bh[gi] > 0 else np.nan

            vv = data.get('Vvir')
            vvir_val = vv[gi] if vv is not None else np.nan

            ej = data.get('EjectedMass')
            log_ej = np.log10(ej[gi]) if ej is not None and ej[gi] > 0 else np.nan

            reg = data.get('Regime')
            regime_str = 'Hot' if (reg is not None and reg[gi] == 1) else 'CGM'

            print(f"  {i+1:3d}  {log_ms:8.3f}  {log_mh:9.3f}  {vvir_val:7.1f}  {log_cg:9.3f}  "
                  f"{log_h2:8.3f}  {log_ej:10.3f}  {sfr_val:8.2f}  {eta:7.2f}  {oh12:11.3f}  {log_bh:8.3f}  {regime_str:>6s}")

        print()


# ========================== PLOT 32: HI MASS FUNCTION ==========================

def plot_32_hi_mass_function():
    """
    HI mass function at z=0 with bootstrap error shading.

    Compares multiple H2 prescription models with observational data from
    Jones+18 (ALFALFA) and Zwaan+05 (HIPASS).
    """
    print('Plot 32: HI Mass Function')

    binwidth = 0.2
    N_BOOT = 100
    MASS_CUT = 1e8  # Minimum HI mass

    # --- Plot ---
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Loop over all H2 models
    for i, model in enumerate(_GAS_MODELS):
        dirpath = model['dir']
        if not model_files_exist(dirpath):
            print(f"  Skipping {model['label']}: directory not found")
            continue

        # Load model data
        data = load_model(dirpath, properties=['H1gas'])
        h1gas = data['H1gas']

        # Select galaxies with HI mass > 10^8 Msun
        valid = h1gas > MASS_CUT
        log_mhi = np.log10(h1gas[valid])

        print(f"  {model['label']}: {np.sum(valid):,} galaxies with H1gas > {MASS_CUT:.0e}")

        # Compute mass function with bootstrap errors
        centers, phi, phi_lo, phi_hi, _ = mass_function_bootstrap(
            log_mhi, VOLUME, binwidth=binwidth, n_boot=N_BOOT
        )

        # Model line with bootstrap shading
        good = np.isfinite(phi) & np.isfinite(phi_lo) & np.isfinite(phi_hi)
        lw = 3.5 if i == 0 else 2.0
        ax.plot(centers[good], phi[good], color=model['color'], lw=lw,
                label=model['label'], zorder=10 - i)
        ax.fill_between(centers[good], phi_lo[good], phi_hi[good],
                        color=model['color'], alpha=0.2, edgecolor='none', zorder=9 - i)

    # Load and plot observations
    obs_list = load_himf_observations()
    for obs in obs_list:
        mass = obs['mass']
        phi_obs = obs['phi']

        # Filter observations to x >= 8
        obs_mask = mass >= 8.0

        # Handle different error formats
        if 'phi_err_lo' in obs:
            # Errors are relative magnitudes (Zwaan+05 style)
            yerr_lo = obs['phi_err_lo'][obs_mask]
            yerr_hi = obs['phi_err_hi'][obs_mask]
            ax.errorbar(mass[obs_mask], phi_obs[obs_mask], yerr=[yerr_lo, yerr_hi],
                        fmt=obs['marker'], color=obs['color'],
                        markerfacecolor='gray' if obs['color'] == 'gray' else 'white',
                        markeredgecolor=obs['color'] if obs['color'] != 'gray' else 'k',
                        markeredgewidth=1.0,
                        ms=7, lw=1.0, capsize=2, alpha=0.8,
                        label=obs['label'], zorder=8)
        else:
            # Errors are absolute bounds (Jones+18 style)
            phi_lo_obs = obs['phi_lo'][obs_mask]
            phi_hi_obs = obs['phi_hi'][obs_mask]
            yerr_lo = phi_obs[obs_mask] - phi_lo_obs
            yerr_hi = phi_hi_obs - phi_obs[obs_mask]
            ax.errorbar(mass[obs_mask], phi_obs[obs_mask], yerr=[yerr_lo, yerr_hi],
                        fmt=obs['marker'], color=obs['color'],
                        markerfacecolor='gray',
                        markeredgecolor='k',
                        markeredgewidth=1.0,
                        ms=7, lw=1.0, capsize=2, alpha=0.8,
                        label=obs['label'], zorder=8)

    # Axis settings
    ax.set_xlim(8.0, 11.0)
    ax.set_ylim(-5.5, -0.5)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))

    ax.set_xlabel(r'$\log_{10}\ M_{\mathrm{HI}}\ [M_{\odot}]$')
    ax.set_ylabel(r'$\log_{10}\ \phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$')

    # Separate legends: models and observations both inside the plot
    handles, labels = ax.get_legend_handles_labels()
    model_labels = [m['label'] for m in _GAS_MODELS]
    model_h = [h for h, l in zip(handles, labels) if l in model_labels]
    model_l = [l for l in labels if l in model_labels]
    obs_h = [h for h, l in zip(handles, labels) if l not in model_labels]
    obs_l = [l for l in labels if l not in model_labels]

    if model_h:
        model_leg = ax.legend(model_h, model_l, loc='lower left', frameon=False)
        ax.add_artist(model_leg)

    if obs_h:
        ax.legend(obs_h, obs_l, loc='upper right', frameon=False)

    save_figure(fig, os.path.join(OUTPUT_DIR, 'HI_Mass_Function' + OUTPUT_FORMAT))


# ========================== PLOT 33: H2 MASS FUNCTION ==========================

def load_h2mf_observations():
    """
    Load H2 mass function observations from Fletcher+21 and Boselli+14.

    Returns a list of dicts with 'label', 'mass', 'phi', 'phi_lo', 'phi_hi',
    'marker', 'color'.
    """
    observations = []

    # Boselli et al. (2014) - H2 mass function
    # The file contains two sections: constant X_CO and luminosity-dependent X_CO.
    path_b14 = os.path.join(OBS_DIR, 'Gas', 'B14_MH2MF.dat')
    if os.path.exists(path_b14):
        try:
            with open(path_b14, 'r') as fh:
                lines = fh.readlines()

            sections = []
            current = []
            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    if current:
                        sections.append(np.array(current, dtype=float))
                        current = []
                    continue
                current.append([float(x) for x in stripped.split()[:4]])
            if current:
                sections.append(np.array(current, dtype=float))

            if len(sections) >= 1:
                data = sections[0]
                observations.append({
                    'label': 'Boselli+14 (const. X_CO)',
                    'mass': data[:, 0],
                    'phi': data[:, 1],
                    'phi_lo': data[:, 2],
                    'phi_hi': data[:, 3],
                    'marker': 'D',
                    'color': 'gray',
                    'edgecolor': 'k',
                })
            if len(sections) >= 2:
                data = sections[1]
                observations.append({
                    'label': 'Boselli+14 (lum.-dep. X_CO)',
                    'mass': data[:, 0],
                    'phi': data[:, 1],
                    'phi_lo': data[:, 2],
                    'phi_hi': data[:, 3],
                    'marker': '^',
                    'color': 'gray',
                    'edgecolor': 'k',
                })
        except Exception as e:
            print(f"  Warning: could not load {path_b14}: {e}")

    # Fletcher et al. (2021) - Detected + Non-detected
    path_det = os.path.join(OBS_DIR, 'Gas/H2MF_Fletcher21_DetNonDet.dat')
    if os.path.exists(path_det):
        try:
            data = np.loadtxt(path_det, comments='#')
            observations.append({
                'label': 'Fletcher+20 (Det+NonDet)',
                'mass': data[:, 0],
                'phi': data[:, 1],
                'phi_lo': data[:, 2],
                'phi_hi': data[:, 3],
                'marker': 's',
                'color': 'k',
                'edgecolor': 'k',
            })
        except Exception as e:
            print(f"  Warning: could not load {path_det}: {e}")

    # Fletcher et al. (2021) - Estimated
    path_est = os.path.join(OBS_DIR, 'Gas/H2MF_Fletcher21_Estimated.dat')
    if os.path.exists(path_est):
        try:
            data = np.loadtxt(path_est, comments='#')
            observations.append({
                'label': 'Fletcher+20 (Estimated)',
                'mass': data[:, 0],
                'phi': data[:, 1],
                'phi_lo': data[:, 2],
                'phi_hi': data[:, 3],
                'marker': 'o',
                'color': 'gray',
                'edgecolor': 'k',
            })
        except Exception as e:
            print(f"  Warning: could not load {path_est}: {e}")

    return observations


def plot_33_h2_mass_function():
    """
    H2 mass function at z=0 with bootstrap error shading.

    Compares multiple H2 prescription models with observational data from
    Fletcher+21 (xCOLD GASS) and Boselli+14.
    """
    print('Plot 33: H2 Mass Function')

    binwidth = 0.2
    N_BOOT = 100
    MASS_CUT = 1e8  # Minimum H2 mass

    # --- Plot ---
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Loop over all H2 models
    for i, model in enumerate(_GAS_MODELS):
        dirpath = model['dir']
        if not model_files_exist(dirpath):
            print(f"  Skipping {model['label']}: directory not found")
            continue

        # Load model data
        data = load_model(dirpath, properties=['H2gas'])
        h2gas = data['H2gas']

        # Select galaxies with H2 mass > cut
        valid = h2gas > MASS_CUT
        log_mh2 = np.log10(h2gas[valid])

        print(f"  {model['label']}: {np.sum(valid):,} galaxies with H2gas > {MASS_CUT:.0e}")

        # Compute mass function with bootstrap errors
        centers, phi, phi_lo, phi_hi, _ = mass_function_bootstrap(
            log_mh2, VOLUME, binwidth=binwidth, n_boot=N_BOOT
        )

        # Model line with bootstrap shading
        good = np.isfinite(phi) & np.isfinite(phi_lo) & np.isfinite(phi_hi)
        lw = 3.5 if i == 0 else 2.0
        ax.plot(centers[good], phi[good], color=model['color'], lw=lw,
                label=model['label'], zorder=10 - i)
        ax.fill_between(centers[good], phi_lo[good], phi_hi[good],
                        color=model['color'], alpha=0.2, edgecolor='none', zorder=9 - i)

    # Load and plot observations
    obs_list = load_h2mf_observations()
    for obs in obs_list:
        mass = obs['mass']
        phi_obs = obs['phi']

        obs_mask = mass >= 8.0

        # Fletcher+21 uses absolute phi bounds
        phi_lo_obs = obs['phi_lo'][obs_mask]
        phi_hi_obs = obs['phi_hi'][obs_mask]
        yerr_lo = phi_obs[obs_mask] - phi_lo_obs
        yerr_hi = phi_hi_obs - phi_obs[obs_mask]
        ax.errorbar(mass[obs_mask], phi_obs[obs_mask], yerr=[yerr_lo, yerr_hi],
                    fmt=obs['marker'], color=obs['color'],
                    markerfacecolor='gray',
                    markeredgecolor=obs.get('edgecolor', 'k'),
                    markeredgewidth=1.0,
                    ms=7, lw=1.0, capsize=2, alpha=0.8,
                    label=obs['label'], zorder=8)

    # Axis settings
    ax.set_xlim(8.0, 11.0)
    ax.set_ylim(-5.5, -0.5)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))

    ax.set_xlabel(r'$\log_{10}\ M_{\mathrm{H_2}}\ [M_{\odot}]$')
    ax.set_ylabel(r'$\log_{10}\ \phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$')

    # Separate legends: models and observations both inside the plot
    handles, labels = ax.get_legend_handles_labels()
    model_labels = [m['label'] for m in _GAS_MODELS]
    model_h = [h for h, l in zip(handles, labels) if l in model_labels]
    model_l = [l for l in labels if l in model_labels]
    obs_h = [h for h, l in zip(handles, labels) if l not in model_labels]
    obs_l = [l for l in labels if l not in model_labels]

    if model_h:
        model_leg = ax.legend(model_h, model_l, loc='lower left', frameon=False)
        ax.add_artist(model_leg)

    if obs_h:
        ax.legend(obs_h, obs_l, loc='upper right', frameon=False)

    save_figure(fig, os.path.join(OUTPUT_DIR, 'H2_Mass_Function' + OUTPUT_FORMAT))


# ========================== PLOT 34: HI MASS FUNCTION (PRIMARY vs UCHUU) ==========================

def plot_34_hi_mass_function_primary_uchuu():
    """
    HI mass function at z=0 for the Primary (Millennium) and Uchuu models with
    Poisson 1-sigma shading, and all available observational data.
    """
    print('Plot 34: HI Mass Function (Primary vs Uchuu)')

    binwidth = 0.2
    MASS_CUT = 1e8

    models = [
        {
            'dir':          PRIMARY_DIR,
            'label':        'SAGE26 (Millennium)',
            'color':        'black',
            'volume':       VOLUME,
            'mass_correct': 1.0,
            'snapshot':     SNAPSHOT,
            'lw':           3.5,
        },
    ]

    if model_files_exist(MINIUCHUU_DIR):
        models.append({
            'dir':          MINIUCHUU_DIR,
            'label':        'SAGE26 (Uchuu)',
            'color':        'steelblue',
            'volume':       MINIUCHUU_VOLUME,
            'mass_correct': MINIUCHUU_MASS_CONVERT / MASS_CONVERT,
            'snapshot':     f'Snap_{MINIUCHUU_LAST_SNAP}',
            'lw':           2.5,
        })

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i, model in enumerate(models):
        dirpath = model['dir']
        if not model_files_exist(dirpath):
            print(f"  Skipping {model['label']}: directory not found")
            continue

        data = load_model(dirpath, properties=['H1gas'], snapshot=model['snapshot'])
        if not data:
            print(f"  Skipping {model['label']}: no data at {model['snapshot']}")
            continue

        h1gas = data['H1gas'] * model['mass_correct']
        valid = h1gas > MASS_CUT
        log_mhi = np.log10(h1gas[valid])

        print(f"  {model['label']}: {np.sum(valid):,} galaxies with H1gas > {MASS_CUT:.0e}")

        centers, phi, mrange = mass_function(log_mhi, model['volume'], binwidth=binwidth)
        mi, ma = mrange
        nbins = int(round((ma - mi) / binwidth))
        counts, _ = np.histogram(log_mhi, range=(mi, ma), bins=nbins)

        with np.errstate(divide='ignore', invalid='ignore'):
            n_lo = counts - np.sqrt(counts)
            phi_lo = np.where(n_lo > 0, np.log10(n_lo / model['volume'] / binwidth), np.nan)
            phi_hi = np.log10((counts + np.sqrt(counts)) / model['volume'] / binwidth)
        phi_hi = np.where(np.isfinite(phi_hi), phi_hi, np.nan)

        good     = np.isfinite(phi)
        shade_lo = np.isfinite(phi_lo)
        shade_hi = np.isfinite(phi_hi)
        ax.plot(centers[good], phi[good], color=model['color'], lw=model['lw'],
                label=model['label'], zorder=10 - i)
        shade_mask = shade_lo & shade_hi
        ax.fill_between(centers[shade_mask], phi_lo[shade_mask], phi_hi[shade_mask],
                        color=model['color'], alpha=0.2, edgecolor='none', zorder=9 - i)

    obs_list = load_himf_observations()
    for obs in obs_list:
        mass    = obs['mass']
        phi_obs = obs['phi']
        obs_mask = mass >= 8.0

        if 'phi_err_lo' in obs:
            yerr_lo = obs['phi_err_lo'][obs_mask]
            yerr_hi = obs['phi_err_hi'][obs_mask]
            ax.errorbar(mass[obs_mask], phi_obs[obs_mask], yerr=[yerr_lo, yerr_hi],
                        fmt=obs['marker'], color=obs['color'],
                        markerfacecolor='gray' if obs['color'] == 'gray' else 'white',
                        markeredgecolor=obs['color'] if obs['color'] != 'gray' else 'k',
                        markeredgewidth=1.0, ms=7, lw=1.0, capsize=2, alpha=0.8,
                        label=obs['label'], zorder=8)
        else:
            phi_lo_obs = obs['phi_lo'][obs_mask]
            phi_hi_obs = obs['phi_hi'][obs_mask]
            yerr_lo = phi_obs[obs_mask] - phi_lo_obs
            yerr_hi = phi_hi_obs - phi_obs[obs_mask]
            ax.errorbar(mass[obs_mask], phi_obs[obs_mask], yerr=[yerr_lo, yerr_hi],
                        fmt=obs['marker'], color=obs['color'],
                        markerfacecolor='gray', markeredgecolor='k',
                        markeredgewidth=1.0, ms=7, lw=1.0, capsize=2, alpha=0.8,
                        label=obs['label'], zorder=8)

    ax.set_xlim(8.0, 11.0)
    ax.set_ylim(-5.5, -0.5)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))

    ax.set_xlabel(r'$\log_{10}\ M_{\mathrm{HI}}\ [M_{\odot}]$')
    ax.set_ylabel(r'$\log_{10}\ \phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$')

    handles, labels = ax.get_legend_handles_labels()
    model_labels = [m['label'] for m in models]
    model_h = [h for h, l in zip(handles, labels) if l in model_labels]
    model_l = [l for l in labels if l in model_labels]
    obs_h   = [h for h, l in zip(handles, labels) if l not in model_labels]
    obs_l   = [l for l in labels if l not in model_labels]

    if model_h:
        model_leg = ax.legend(model_h, model_l, loc='lower left', frameon=False)
        ax.add_artist(model_leg)
    if obs_h:
        ax.legend(obs_h, obs_l, loc='upper right', frameon=False)

    save_figure(fig, os.path.join(OUTPUT_DIR, 'HI_Mass_Function_Primary_Uchuu' + OUTPUT_FORMAT))


# ========================== PLOT 35: H2 MASS FUNCTION (PRIMARY vs UCHUU) ==========================

def plot_35_h2_mass_function_primary_uchuu():
    """
    H2 mass function at z=0 for the Primary (Millennium) and Uchuu models with
    Poisson 1-sigma shading, and all available observational data.
    """
    print('Plot 35: H2 Mass Function (Primary vs Uchuu)')

    binwidth = 0.2
    MASS_CUT = 1e8

    models = [
        {
            'dir':          PRIMARY_DIR,
            'label':        'SAGE26 (Millennium)',
            'color':        'black',
            'volume':       VOLUME,
            'mass_correct': 1.0,
            'snapshot':     SNAPSHOT,
            'lw':           3.5,
        },
    ]

    if model_files_exist(MINIUCHUU_DIR):
        models.append({
            'dir':          MINIUCHUU_DIR,
            'label':        'SAGE26 (Uchuu)',
            'color':        'steelblue',
            'volume':       MINIUCHUU_VOLUME,
            'mass_correct': MINIUCHUU_MASS_CONVERT / MASS_CONVERT,
            'snapshot':     f'Snap_{MINIUCHUU_LAST_SNAP}',
            'lw':           2.5,
        })

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i, model in enumerate(models):
        dirpath = model['dir']
        if not model_files_exist(dirpath):
            print(f"  Skipping {model['label']}: directory not found")
            continue

        data = load_model(dirpath, properties=['H2gas'], snapshot=model['snapshot'])
        if not data:
            print(f"  Skipping {model['label']}: no data at {model['snapshot']}")
            continue

        h2gas = data['H2gas'] * model['mass_correct']
        valid = h2gas > MASS_CUT
        log_mh2 = np.log10(h2gas[valid])

        print(f"  {model['label']}: {np.sum(valid):,} galaxies with H2gas > {MASS_CUT:.0e}")

        centers, phi, mrange = mass_function(log_mh2, model['volume'], binwidth=binwidth)
        mi, ma = mrange
        nbins = int(round((ma - mi) / binwidth))
        counts, _ = np.histogram(log_mh2, range=(mi, ma), bins=nbins)

        with np.errstate(divide='ignore', invalid='ignore'):
            n_lo = counts - np.sqrt(counts)
            phi_lo = np.where(n_lo > 0, np.log10(n_lo / model['volume'] / binwidth), np.nan)
            phi_hi = np.log10((counts + np.sqrt(counts)) / model['volume'] / binwidth)
        phi_hi = np.where(np.isfinite(phi_hi), phi_hi, np.nan)

        good     = np.isfinite(phi)
        shade_lo = np.isfinite(phi_lo)
        shade_hi = np.isfinite(phi_hi)
        ax.plot(centers[good], phi[good], color=model['color'], lw=model['lw'],
                label=model['label'], zorder=10 - i)
        shade_mask = shade_lo & shade_hi
        ax.fill_between(centers[shade_mask], phi_lo[shade_mask], phi_hi[shade_mask],
                        color=model['color'], alpha=0.2, edgecolor='none', zorder=9 - i)

    obs_list = load_h2mf_observations()
    for obs in obs_list:
        mass    = obs['mass']
        phi_obs = obs['phi']
        obs_mask = mass >= 8.0

        phi_lo_obs = obs['phi_lo'][obs_mask]
        phi_hi_obs = obs['phi_hi'][obs_mask]
        yerr_lo = phi_obs[obs_mask] - phi_lo_obs
        yerr_hi = phi_hi_obs - phi_obs[obs_mask]
        ax.errorbar(mass[obs_mask], phi_obs[obs_mask], yerr=[yerr_lo, yerr_hi],
                    fmt=obs['marker'], color=obs['color'],
                    markerfacecolor='gray',
                    markeredgecolor=obs.get('edgecolor', 'k'),
                    markeredgewidth=1.0, ms=7, lw=1.0, capsize=2, alpha=0.8,
                    label=obs['label'], zorder=8)

    ax.set_xlim(8.0, 11.0)
    ax.set_ylim(-5.5, -0.5)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))

    ax.set_xlabel(r'$\log_{10}\ M_{\mathrm{H_2}}\ [M_{\odot}]$')
    ax.set_ylabel(r'$\log_{10}\ \phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$')

    handles, labels = ax.get_legend_handles_labels()
    model_labels = [m['label'] for m in models]
    model_h = [h for h, l in zip(handles, labels) if l in model_labels]
    model_l = [l for l in labels if l in model_labels]
    obs_h   = [h for h, l in zip(handles, labels) if l not in model_labels]
    obs_l   = [l for l in labels if l not in model_labels]

    if model_h:
        model_leg = ax.legend(model_h, model_l, loc='lower left', frameon=False)
        ax.add_artist(model_leg)
    if obs_h:
        ax.legend(obs_h, obs_l, loc='upper right', frameon=False)

    save_figure(fig, os.path.join(OUTPUT_DIR, 'H2_Mass_Function_Primary_Uchuu' + OUTPUT_FORMAT))

# ========================== PLOT 38: HI MASS FUNCTION (ALL RECIPES) ==========================

def plot_38_hi_mass_function_recipes():
    """
    HI mass function at z=0 with bootstrap error shading.

    Direct analogue of plot_33 (H2 mass function): compares the same set of
    gas-partition prescriptions, here in the atomic phase, against Jones+18
    (ALFALFA) and Zwaan+05 (HIPASS).
    """
    print('Plot 38: HI Mass Function (all recipes)')

    binwidth = 0.2
    N_BOOT = 100
    MASS_CUT = 1e8  # Minimum HI mass

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i, model in enumerate(_GAS_MODELS):
        dirpath = model['dir']
        if not model_files_exist(dirpath):
            print(f"  Skipping {model['label']}: directory not found")
            continue

        data = load_model(dirpath, properties=['H1gas'])
        h1gas = data['H1gas']

        valid = h1gas > MASS_CUT
        log_mh1 = np.log10(h1gas[valid])

        print(f"  {model['label']}: {np.sum(valid):,} galaxies with H1gas > {MASS_CUT:.0e}")

        centers, phi, phi_lo, phi_hi, _ = mass_function_bootstrap(
            log_mh1, VOLUME, binwidth=binwidth, n_boot=N_BOOT
        )

        good = np.isfinite(phi) & np.isfinite(phi_lo) & np.isfinite(phi_hi)
        lw = 3.5 if i == 0 else 2.0
        ax.plot(centers[good], phi[good], color=model['color'], lw=lw,
                label=model['label'], zorder=10 - i)
        ax.fill_between(centers[good], phi_lo[good], phi_hi[good],
                        color=model['color'], alpha=0.2, edgecolor='none', zorder=9 - i)

    obs_list = load_himf_observations()
    for obs in obs_list:
        mass = obs['mass']
        phi_obs = obs['phi']
        obs_mask = mass >= 8.0
        ax.errorbar(mass[obs_mask], phi_obs[obs_mask],
                    yerr=_gasmf_obs_yerr(obs, obs_mask),
                    fmt=obs['marker'], color=obs['color'],
                    markerfacecolor='gray',
                    markeredgecolor=obs.get('edgecolor', 'k'),
                    markeredgewidth=1.0,
                    ms=7, lw=1.0, capsize=2, alpha=0.8,
                    label=obs['label'], zorder=8)

    ax.set_xlim(8.0, 11.0)
    ax.set_ylim(-5.5, -0.5)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))

    ax.set_xlabel(r'$\log_{10}\ M_{\mathrm{HI}}\ [M_{\odot}]$')
    ax.set_ylabel(r'$\log_{10}\ \phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$')

    handles, labels = ax.get_legend_handles_labels()
    model_labels = [m['label'] for m in _GAS_MODELS]
    model_h = [h for h, l in zip(handles, labels) if l in model_labels]
    model_l = [l for l in labels if l in model_labels]
    obs_h = [h for h, l in zip(handles, labels) if l not in model_labels]
    obs_l = [l for l in labels if l not in model_labels]

    if model_h:
        model_leg = ax.legend(model_h, model_l, loc='lower left', frameon=False)
        ax.add_artist(model_leg)
    if obs_h:
        ax.legend(obs_h, obs_l, loc='upper right', frameon=False)

    save_figure(fig, os.path.join(OUTPUT_DIR, 'HI_Mass_Function' + OUTPUT_FORMAT))


# ================ PLOT 39: COLD / HI / H2 MASS FUNCTIONS (STACKED) ================

def plot_39_gas_mass_functions_stacked():
    """
    Cold gas, HI and H2 mass functions at z=0 stacked in one column.

    The cold gas and HI panels show SAGE26 and SAGE16 only, both with
    bootstrap 1-sigma shading.  The H2 panel reproduces plot_33: every gas
    partition recipe, with SAGE26 (BR06) in black.  SAGE16 does not split the
    cold gas and so has no H2 line.
    """
    print('Plot 39: Cold / HI / H2 mass functions (stacked)')

    binwidth = 0.2
    N_BOOT = 100
    MASS_CUT = 1e8

    panels = [
        {'field': 'ColdGas', 'tag': r'$M_{\mathrm{cold}}$',
         'obs': load_himf_observations() + load_h2mf_observations(),
         'recipes': False, 'sage16': True},
        {'field': 'H1gas', 'tag': r'$M_{\mathrm{HI}}$',
         'obs': load_himf_observations(),
         'recipes': False, 'sage16': True},
        {'field': 'H2gas', 'tag': r'$M_{\mathrm{H_2}}$',
         'obs': load_h2mf_observations(),
         'recipes': True, 'sage16': False},
    ]

    fig, axes = plt.subplots(3, 1, figsize=(7, 13), sharex=True)

    def _mf(dirpath, field):
        if not model_files_exist(dirpath):
            return None
        g = load_model(dirpath, properties=[field])[field]
        valid = g > MASS_CUT
        if not np.any(valid):
            return None
        return mass_function_bootstrap(np.log10(g[valid]), VOLUME,
                                       binwidth=binwidth, n_boot=N_BOOT)

    for ax, cfg in zip(axes, panels):
        if cfg['recipes']:
            # H2 panel: reproduce plot_33 exactly
            for i, model in enumerate(_GAS_MODELS):
                res = _mf(model['dir'], cfg['field'])
                if res is None:
                    continue
                centers, phi, plo, phi_hi, _ = res
                good = np.isfinite(phi) & np.isfinite(plo) & np.isfinite(phi_hi)
                lw = 3.5 if i == 0 else 2.0
                ax.plot(centers[good], phi[good], color=model['color'], lw=lw,
                        label=model['label'], zorder=10 - i)
                ax.fill_between(centers[good], plo[good], phi_hi[good],
                                color=model['color'], alpha=0.2,
                                edgecolor='none', zorder=9 - i)
        else:
            # cold gas and HI panels: SAGE16 and SAGE26 only
            if cfg['sage16']:
                res = _mf(VANILLA_DIR, cfg['field'])
                if res is not None:
                    centers, phi, plo, phi_hi, _ = res
                    good = np.isfinite(phi) & np.isfinite(plo) & np.isfinite(phi_hi)
                    ax.fill_between(centers[good], plo[good], phi_hi[good],
                                    color='purple', alpha=0.20, lw=0.0,
                                    zorder=Z_MODEL_BAND_ALT)
                    ax.plot(centers[good], phi[good], color='purple', ls='--',
                            lw=3.0, label='SAGE16', zorder=Z_MODEL_LINE_ALT)

            res = _mf(PRIMARY_DIR, cfg['field'])
            if res is not None:
                centers, phi, plo, phi_hi, _ = res
                good = np.isfinite(phi) & np.isfinite(plo) & np.isfinite(phi_hi)
                ax.fill_between(centers[good], plo[good], phi_hi[good],
                                color='steelblue', alpha=0.25, lw=0.0,
                                zorder=Z_MODEL_BAND)
                ax.plot(centers[good], phi[good], color='steelblue', lw=3.5,
                        label='SAGE26', zorder=Z_MODEL_LINE)

        for obs in cfg['obs']:
            mass = obs['mass']; phi_obs = obs['phi']
            m = mass >= 8.0
            if not np.any(m):
                continue
            ax.errorbar(mass[m], phi_obs[m],
                        yerr=_gasmf_obs_yerr(obs, m),
                        fmt=obs['marker'], color=obs['color'],
                        markerfacecolor='gray',
                        markeredgecolor=obs.get('edgecolor', 'k'),
                        markeredgewidth=1.0, ms=6, lw=1.0, capsize=2,
                        alpha=0.8, label=obs['label'], zorder=Z_OBS)

        ax.set_ylim(-5.5, -0.5)
        ax.set_ylabel(r'$\log_{10}\ \phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$')
        ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
        handles, labels = ax.get_legend_handles_labels()
        obs_labels = [o['label'] for o in cfg['obs']]
        oh = [a for a, b in zip(handles, labels) if b in obs_labels]
        ol = [b for b in labels if b in obs_labels]
        mh = [a for a, b in zip(handles, labels) if b not in obs_labels]
        ml = [b for b in labels if b not in obs_labels]
        if mh:
            leg = ax.legend(mh, ml, loc='lower left', frameon=False, fontsize=10,
                            title=cfg['tag'])
            leg.get_title().set_fontsize(17)
            leg._legend_box.align = 'left'
            ax.add_artist(leg)
        if oh:
            ax.legend(oh, ol, loc='upper right', frameon=False, fontsize=10)

    axes[-1].set_xlim(8.0, 11.0)
    axes[-1].set_xlabel(r'$\log_{10}\ M_{\mathrm{gas}}\ [M_{\odot}]$')
    axes[-1].xaxis.set_major_locator(plt.MultipleLocator(1.0))
    axes[-1].xaxis.set_minor_locator(plt.MultipleLocator(0.2))

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.0)
    save_figure(fig, os.path.join(OUTPUT_DIR,
                'Gas_Mass_Functions_Stacked' + OUTPUT_FORMAT))

    print_gas_recipe_diagnostics()
    print_hi_offsets()


def print_hi_offsets(masses=(9.5, 10.0, 10.5), binwidth=0.2, mass_cut=1e8):
    """Print SAGE26's HI/H2 mass function offsets at specific masses, plus a
    partition test at fixed stellar mass.

    The mass function block gives model - observation in dex at each log M_gas,
    in the style of the metallicity and bulge-size comparisons.  The partition
    block compares M_HI/m* against xGASS and M_H2/m* against xCOLD GASS at fixed
    stellar mass, which separates a genuine cold gas excess (both ratios high)
    from an atomic/molecular partition problem (HI high, H2 low).
    """
    if not model_files_exist(PRIMARY_DIR):
        return
    d = load_model(PRIMARY_DIR, properties=['StellarMass', 'H1gas', 'H2gas'])

    def _mf(g):
        g = g[g > mass_cut]
        if not g.size:
            return None, None
        edges = np.arange(6.0, 12.2 + binwidth, binwidth)
        n, _ = np.histogram(np.log10(g), bins=edges)
        with np.errstate(divide='ignore'):
            return 0.5 * (edges[1:] + edges[:-1]), np.log10(n / VOLUME / binwidth)

    print('  SAGE26 mass function offsets (dex, model - observation)')
    for name, obs_list, field in (('HI', load_himf_observations(), 'H1gas'),
                                  ('H2', load_h2mf_observations(), 'H2gas')):
        cen, phi = _mf(d[field])
        if cen is None or not obs_list:
            continue
        hdr = '    %-8s' % ('log M_%s' % name)
        for o in obs_list:
            hdr += '%22s' % o['label'][:20]
        print(hdr)
        for m in masses:
            line = '    %-8.2f' % m
            for o in obs_list:
                inside = (m >= o['mass'].min()) and (m <= o['mass'].max())
                if inside:
                    line += '%22s' % ('%+.2f' % (np.interp(m, cen, phi)
                                                 - np.interp(m, o['mass'], o['phi'])))
                else:
                    line += '%22s' % '-'
            print(line)

    # partition test at fixed stellar mass
    obs = {}
    for lbl, fn in (('HI/m* (xGASS)', 'Gas/NeutralGasRatio_NonDetEQZero.dat'),
                    ('H2/m* (xCOLDGASS)', 'Gas/MolecularGasRatio_NonDetEQZero.dat')):
        path = os.path.join(OBS_DIR, fn)
        if os.path.exists(path):
            arr = np.loadtxt(path, comments='#')
            obs[lbl] = (arr[:, 0], arr[:, 1])
    if not obs:
        print()
        return

    ms = d['StellarMass']
    w = ms > 0
    lm = np.log10(ms[w])
    print()
    print('  Partition test at fixed stellar mass (dex, model - observation)')
    print('    %-8s%22s%22s' % ('log m*', 'HI/m* (xGASS)', 'H2/m* (xCOLDGASS)'))
    for m in masses:
        sel = (lm > m - 0.15) & (lm < m + 0.15)
        line = '    %-8.2f' % m
        for lbl, field in (('HI/m* (xGASS)', 'H1gas'),
                           ('H2/m* (xCOLDGASS)', 'H2gas')):
            if lbl not in obs or sel.sum() < 20:
                line += '%22s' % '-'
                continue
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.log10(d[field][w][sel] / ms[w][sel])
            mod = np.nanmedian(ratio[np.isfinite(ratio)])
            line += '%22s' % ('%+.2f' % (mod - np.interp(m, obs[lbl][0], obs[lbl][1])))
        print(line)
    print()


def print_gas_recipe_diagnostics(binwidth=0.2, mass_cut=1e8):
    """Print the per-recipe numbers quoted in the gas mass function section.

    Reports, for every gas-partition prescription plus SAGE16:
      f_Q(1e8.5)   dwarf quiescent fraction at m* = 10^8.5 Msun
      crossover    lowest m* where the quiescent SMF exceeds the star-forming SMF
      dSF, dQ      mean offset from Bell+03 blue/red SMFs over 10^10.3-10^11
      dH2(8-9)     mean offset from Fletcher+20 H2MF over 10^8-10^9
      dHI(9-10)    mean offset from Jones+18 HIMF over 10^9-10^10

    Called at the end of plot_40 so the figure and its numbers appear together.
    """
    models = [(m['label'], m['dir']) for m in _GAS_MODELS] + [('SAGE16', VANILLA_DIR)]

    bell_sf = load_bell_smf_sf_data()
    bell_q = load_bell_smf_q_data()
    himf_obs = load_himf_observations()
    h2mf_obs = load_h2mf_observations()

    def _mf(log_masses, lo=8.0, hi=12.2):
        edges = np.arange(lo, hi + binwidth, binwidth)
        n, _ = np.histogram(log_masses, bins=edges)
        with np.errstate(divide='ignore'):
            return 0.5 * (edges[1:] + edges[:-1]), np.log10(n / VOLUME / binwidth)

    def _resid(obs_x, obs_y, cen, phi, lo, hi):
        sel = (obs_x > lo) & (obs_x < hi)
        if not np.any(sel):
            return np.nan
        return np.nanmean(np.interp(obs_x[sel], cen, phi) - obs_y[sel])

    print()
    print('  Gas prescription diagnostics (z=0)')
    print('  %-15s %9s %10s %8s %8s %10s %10s'
          % ('recipe', 'f_Q(8.5)', 'crossover', 'dSF', 'dQ', 'dH2(8-9)', 'dHI(9-10)'))

    for label, dirpath in models:
        if not model_files_exist(dirpath):
            print('  %-15s   (not found)' % label)
            continue
        d = load_model(dirpath, properties=['StellarMass', 'SfrDisk', 'SfrBulge',
                                            'H1gas', 'H2gas'])
        ms = d['StellarMass']
        w = ms > 0
        ms = ms[w]
        sfr = (d['SfrDisk'] + d['SfrBulge'])[w]
        with np.errstate(divide='ignore', invalid='ignore'):
            ssfr = np.log10(sfr / ms)
        lm = np.log10(ms)
        quiescent = ssfr < SSFR_CUT

        sel = (lm > 8.35) & (lm < 8.65)
        f_q = 100.0 * np.mean(quiescent[sel]) if sel.sum() > 20 else np.nan

        cen_sf, phi_sf = _mf(lm[~quiescent])
        cen_q, phi_q = _mf(lm[quiescent])
        cross = np.nan
        for i in range(len(cen_sf)):
            if np.isfinite(phi_sf[i]) and np.isfinite(phi_q[i]) and phi_q[i] > phi_sf[i]:
                cross = cen_sf[i]
                break

        d_sf = d_q = np.nan
        if bell_sf[0] is not None:
            d_sf = _resid(bell_sf[0], bell_sf[1], cen_sf, phi_sf, 10.3, 11.0)
        if bell_q[0] is not None:
            d_q = _resid(bell_q[0], bell_q[1], cen_q, phi_q, 10.3, 11.0)

        d_h2 = np.nan
        h2 = d['H2gas'][d['H2gas'] > mass_cut]
        if h2.size and h2mf_obs:
            c, p = _mf(np.log10(h2))
            d_h2 = np.nanmean([_resid(o['mass'], o['phi'], c, p, 8.0, 9.0)
                               for o in h2mf_obs])

        d_hi = np.nan
        h1 = d['H1gas'][d['H1gas'] > mass_cut]
        if h1.size and himf_obs:
            c, p = _mf(np.log10(h1))
            d_hi = np.nanmean([_resid(o['mass'], o['phi'], c, p, 9.0, 10.0)
                               for o in himf_obs])

        def _f(v):
            return ('%+.2f' % v) if np.isfinite(v) else '-'
        print('  %-15s %8.1f%% %10s %8s %8s %10s %10s'
              % (label, f_q,
                 ('%.2f' % cross) if np.isfinite(cross) else '-',
                 _f(d_sf), _f(d_q), _f(d_h2), _f(d_hi)))
    print()


# ======== PLOT 40: COLD / HI / H2 MASS FUNCTIONS (STACKED, RECIPES ON ALL) ========

def plot_40_gas_mass_functions_stacked_recipes():
    """
    As plot_39, but with the gas-partition recipes also drawn faintly behind
    SAGE26 and SAGE16 on the cold gas and HI panels.

    The H2 panel is unchanged from plot_39 (all recipes, SAGE26 (BR06) black).
    """
    print('Plot 40: Cold / HI / H2 mass functions (stacked, recipes on all)')

    binwidth = 0.2
    N_BOOT = 100
    MASS_CUT = 1e8

    panels = [
        {'field': 'ColdGas', 'tag': r'$M_{\mathrm{cold}}$',
         'obs': load_himf_observations() + load_h2mf_observations(),
         'h2_style': False, 'sage16': True},
        {'field': 'H1gas', 'tag': r'$M_{\mathrm{HI}}$',
         'obs': load_himf_observations(),
         'h2_style': False, 'sage16': True},
        {'field': 'H2gas', 'tag': r'$M_{\mathrm{H_2}}$',
         'obs': load_h2mf_observations(),
         'h2_style': True, 'sage16': False},
    ]

    fig, axes = plt.subplots(3, 1, figsize=(7, 13), sharex=True)

    def _mf(dirpath, field):
        if not model_files_exist(dirpath):
            return None
        g = load_model(dirpath, properties=[field])[field]
        valid = g > MASS_CUT
        if not np.any(valid):
            return None
        return mass_function_bootstrap(np.log10(g[valid]), VOLUME,
                                       binwidth=binwidth, n_boot=N_BOOT)

    for ax, cfg in zip(axes, panels):
        if cfg['h2_style']:
            for i, model in enumerate(_GAS_MODELS):
                res = _mf(model['dir'], cfg['field'])
                if res is None:
                    continue
                centers, phi, plo, phi_hi, _ = res
                good = np.isfinite(phi) & np.isfinite(plo) & np.isfinite(phi_hi)
                lw = 3.5 if i == 0 else 2.0
                ax.plot(centers[good], phi[good], color=model['color'], lw=lw,
                        label=model['label'], zorder=10 - i)
                ax.fill_between(centers[good], plo[good], phi_hi[good],
                                color=model['color'], alpha=0.2,
                                edgecolor='none', zorder=9 - i)
        else:
            # alternative recipes, faint, behind everything
            for model in _GAS_MODELS[1:]:
                res = _mf(model['dir'], cfg['field'])
                if res is None:
                    continue
                centers, phi, _, _, _ = res
                good = np.isfinite(phi)
                # unlabelled: the recipes are identified in the H2 panel legend
                ax.plot(centers[good], phi[good], color=model['color'],
                        lw=2.0, alpha=0.7, zorder=1)

            if cfg['sage16']:
                res = _mf(VANILLA_DIR, cfg['field'])
                if res is not None:
                    centers, phi, plo, phi_hi, _ = res
                    good = np.isfinite(phi) & np.isfinite(plo) & np.isfinite(phi_hi)
                    ax.fill_between(centers[good], plo[good], phi_hi[good],
                                    color='purple', alpha=0.20, lw=0.0,
                                    zorder=Z_MODEL_BAND_ALT)
                    ax.plot(centers[good], phi[good], color='purple', ls='--',
                            lw=3.0, label='SAGE16', zorder=Z_MODEL_LINE_ALT)

            # res = _mf(DISK_SMOOTH_DIR, cfg['field'])
            # if res is not None:
            #     centers, phi, plo, phi_hi, _ = res
            #     good = np.isfinite(phi) & np.isfinite(plo) & np.isfinite(phi_hi)
            #     ax.fill_between(centers[good], plo[good], phi_hi[good],
            #                     color='darkgreen', alpha=0.20, lw=0.0,
            #                     zorder=Z_MODEL_BAND_ALT)
            #     ax.plot(centers[good], phi[good], color='darkgreen', ls=':',
            #             lw=3.5, label='SAGE26 (Disk Smooth)', zorder=Z_MODEL_LINE_ALT)

            res = _mf(PRIMARY_DIR, cfg['field'])
            if res is not None:
                centers, phi, plo, phi_hi, _ = res
                good = np.isfinite(phi) & np.isfinite(plo) & np.isfinite(phi_hi)
                ax.fill_between(centers[good], plo[good], phi_hi[good],
                                color='steelblue', alpha=0.25, lw=0.0,
                                zorder=Z_MODEL_BAND)
                ax.plot(centers[good], phi[good], color='steelblue', lw=3.5,
                        label='SAGE26', zorder=Z_MODEL_LINE)

        for obs in cfg['obs']:
            mass = obs['mass']; phi_obs = obs['phi']
            m = mass >= 8.0
            if not np.any(m):
                continue
            ax.errorbar(mass[m], phi_obs[m],
                        yerr=_gasmf_obs_yerr(obs, m),
                        fmt=obs['marker'], color=obs['color'],
                        markerfacecolor='gray',
                        markeredgecolor=obs.get('edgecolor', 'k'),
                        markeredgewidth=1.0, ms=6, lw=1.0, capsize=2,
                        alpha=0.8, label=obs['label'], zorder=Z_OBS)

        ax.set_ylim(-5.5, -0.5)
        ax.set_ylabel(r'$\log_{10}\ \phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$')
        ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))

        handles, labels = ax.get_legend_handles_labels()
        obs_labels = [o['label'] for o in cfg['obs']]
        oh = [a for a, b in zip(handles, labels) if b in obs_labels]
        ol = [b for b in labels if b in obs_labels]
        mh = [a for a, b in zip(handles, labels) if b not in obs_labels]
        ml = [b for b in labels if b not in obs_labels]
        if mh:
            leg = ax.legend(mh, ml, loc='lower left', frameon=False, fontsize=10,
                            title=cfg['tag'])
            leg.get_title().set_fontsize(17)
            leg._legend_box.align = 'left'
            ax.add_artist(leg)
        if oh:
            ax.legend(oh, ol, loc='upper right', frameon=False, fontsize=10)

    axes[-1].set_xlim(8.0, 11.0)
    axes[-1].set_xlabel(r'$\log_{10}\ M_{\mathrm{gas}}\ [M_{\odot}]$')
    axes[-1].xaxis.set_major_locator(plt.MultipleLocator(1.0))
    axes[-1].xaxis.set_minor_locator(plt.MultipleLocator(0.2))

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.0)
    save_figure(fig, os.path.join(OUTPUT_DIR,
                'Gas_Mass_Functions_Stacked_Recipes' + OUTPUT_FORMAT))

    print_gas_recipe_diagnostics()
    print_hi_offsets()



# ========================== PLOT 36: SELECTION THRESHOLDS IN THE M-z PLANE ==========================

def plot_36_selection_thresholds_mz():
    """
    Two-panel comparison of the FFB selection thresholds in the (z, M_vir) plane.

    Left  (MBK25):  threshold mass locus g_max(M, z, c) = g_crit for several fixed
                    concentrations c, with the selected FFB centrals from the
                    mode-4 (c-scatter) run scattered on top and colour-coded by
                    their Ishiyama+21 mean concentration.
    Right (Li+24 / Dekel): the Eq.-1 threshold line M_ffb(z) plus a sigmoid-scatter
                    envelope (f_ffb = 0.1 -> 0.9), with the Li+24 sigmoid-selected
                    FFB centrals scattered on top.

    The point is to show the two selections pick out the same region of the
    (z, M_vir) plane.
    """
    import matplotlib as mpl

    print('Plot 36: selection thresholds in the M-z plane')

    MBK_DIR = FFB_BK25_SMOOTH_DIR          # mode-4, log-normal c scatter
    DEKEL_DIR = PRIMARY_DIR                # Li+24 threshold sigmoid

    # Redshift range to draw threshold curves over (FFB-relevant high-z window).
    z_grid = np.linspace(4.0, 16.0, 200)

    # ---- Gather selected FFB centrals across snapshots for both runs ----
    def _collect_ffb(directory):
        snaps = [s for s in range(0, len(REDSHIFTS))
                 if 4.0 <= REDSHIFTS[s] <= 16.0]
        snapdata = load_snapshots(directory, snaps,
                                  properties=['Mvir', 'Type', 'FFBRegime'])
        zs, mv = [], []
        for s in snaps:
            d = snapdata.get(s)
            if not d:
                continue
            w = (d['FFBRegime'] == 1) & (d['Mvir'] > 0)
            n = int(np.sum(w))
            if n == 0:
                continue
            zs.append(np.full(n, REDSHIFTS[s]))
            mv.append(d['Mvir'][w])
        if not mv:
            return np.array([]), np.array([])
        return np.concatenate(zs), np.concatenate(mv)

    z_mbk, m_mbk = _collect_ffb(MBK_DIR)
    z_dek, m_dek = _collect_ffb(DEKEL_DIR)
    print(f'  MBK25 galaxies:  {len(m_mbk)}')
    print(f'  Li+24 FFB galaxies:  {len(m_dek)}')

    # Dilute dense scatters for legibility.
    def _dilute(z, m, n=DILUTE):
        if len(m) > n:
            idx = np.random.choice(len(m), n, replace=False)
            return z[idx], m[idx]
        return z, m
    z_mbk, m_mbk = _dilute(z_mbk, m_mbk)
    z_dek, m_dek = _dilute(z_dek, m_dek)

    # Three columns: the two selection panels, then the overplotted threshold
    # curves with a small residual strip beneath them.  The first two panels
    # span both rows so all three read at a comparable size.
    # constrained_layout rather than tight_layout: the first two panels span
    # both rows, which tight_layout lays out badly (it clips the x labels).
    fig = plt.figure(figsize=(15.5, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[4.0, 1.0])
    # constrained_layout ignores the gridspec hspace/wspace; set them on the
    # layout engine instead.
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.02, hspace=0.0)
    axL = fig.add_subplot(gs[:, 0])
    axR = fig.add_subplot(gs[:, 1], sharey=axL)
    axM = fig.add_subplot(gs[0, 2], sharey=axL)
    axD = fig.add_subplot(gs[1, 2], sharex=axM)

    # ---------------- Left panel: MBK25 ----------------
    c_lines = [3.0, 4.0, 5.0, 6.0, 7.0]
    cmap = mpl.cm.viridis
    cnorm = mpl.colors.Normalize(vmin=min(c_lines), vmax=max(c_lines))

    # Scatter selected galaxies, colour-coded by the threshold concentration at
    # which they enter the FFB regime -- i.e. the fixed-c line they sit on.
    if len(m_mbk):
        c_mbk = np.atleast_1d(mbk25_threshold_concentration(m_mbk, z_mbk))
        axL.scatter(z_mbk, np.log10(m_mbk), c=c_mbk, cmap=cmap, norm=cnorm,
                    s=6, alpha=0.55, edgecolors='none', rasterized=True,
                    zorder=1)

    for c in c_lines:
        M_thr = mbk25_threshold_mass_msun(z_grid, c)
        axL.plot(z_grid, np.log10(M_thr), lw=2.0, color=cmap(cnorm(c)),
                 zorder=3)

    axL.set_xlabel(r'Redshift $z$')
    axL.set_ylabel(r'$\log_{10}\ M_{\rm vir}\ [M_\odot]$')

    # Legend: scatter proxy plus one entry per concentration line.
    from matplotlib.lines import Line2D
    mbk_handles = [Line2D([], [], marker='o', linestyle='none', color='0.35',
                          markersize=4, label='MBK25 galaxies')]
    mbk_handles += [Line2D([], [], color=cmap(cnorm(c)), lw=2.0,
                           label=fr'$c={c:g}$') for c in c_lines]
    axL.legend(handles=mbk_handles, loc='upper right', frameon=False, fontsize=8)

    # ---------------- Right panel: Li+24 / Dekel ----------------
    if len(m_dek):
        axR.scatter(z_dek, np.log10(m_dek), s=6, alpha=0.45, color='0.45',
                    edgecolors='none', rasterized=True, zorder=1)

    # Eq.-1 threshold line and sigmoid-scatter envelope (f = 0.1 .. 0.9).
    delta_log_M = 0.15                                   # matches ffb_fraction()
    logit = np.log(9.0)                                  # f=0.9 <-> x=+ln9
    M_line = np.array([ffb_threshold_mass_msun(z) for z in z_grid])
    log_line = np.log10(M_line)
    axR.fill_between(z_grid, log_line - logit * delta_log_M,
                     log_line + logit * delta_log_M, color='firebrick',
                     alpha=0.18, zorder=2)
    axR.plot(z_grid, log_line, lw=2.2, color='firebrick', zorder=3)

    axR.set_xlabel(r'Redshift $z$')
    dek_handles = [
        Line2D([], [], marker='o', linestyle='none', color='0.35',
               markersize=4, label='Li+24 FFB galaxies'),
        Line2D([], [], color='firebrick', lw=2.2, label=r'$M_{\rm FFB}(z)$'),
    ]
    axR.legend(handles=dek_handles, loc='upper right', frameon=False, fontsize=8)

    # Common x and y limits and ticks for both panels.
    for ax in (axL, axR):
        ax.set_xlim(z_grid.min(), z_grid.max())
        ax.set_ylim(9.5, 13.5)

    # One redshift tick per unit on every panel; axD inherits from axM via sharex.
    for ax in (axL, axR, axM, axD):
        ax.set_xlim(z_grid.min(), z_grid.max())
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(plt.NullLocator())

    # ------------- Middle panel: the two thresholds overplotted -------------
    # MBK25 threshold at the *median* concentration.  c depends on M, and M on
    # c, so solve the pair self-consistently at each redshift.
    def _mbk_threshold_median_c(z):
        c = 3.3
        M = float(np.atleast_1d(mbk25_threshold_mass_msun(z, c)))
        for _ in range(20):
            c_new = float(np.atleast_1d(_c_ishiyama21(np.array([M]), z)))
            if not np.isfinite(c_new) or c_new <= 1.0:
                break
            M_new = float(np.atleast_1d(mbk25_threshold_mass_msun(z, c_new)))
            if abs(np.log10(M_new) - np.log10(M)) < 1e-4:
                M, c = M_new, c_new
                break
            M, c = M_new, c_new
        return M, c

    M_mbk_med, c_med = np.array([_mbk_threshold_median_c(z) for z in z_grid]).T
    log_mbk = np.log10(M_mbk_med)

    axM.plot(z_grid, log_line, lw=2.4, color='firebrick',
             label=r'$M_{\rm vir,FFB}(z)$  (Li+24)')
    axM.plot(z_grid, log_mbk, lw=2.4, color='mediumpurple', ls='--',
             label=r'$M_{\rm vir,MBK25}(z)$  (median $c$)')
    axM.fill_between(z_grid, log_line - logit * delta_log_M,
                     log_line + logit * delta_log_M, color='firebrick',
                     alpha=0.15, lw=0.0)
    axM.axvspan(6.0, 12.0, color='0.85', alpha=0.45, zorder=0)
    axM.legend(loc='lower left', frameon=False, fontsize=8)
    axM.tick_params(labelbottom=False)

    # ------------------------- Residual sub-panel -------------------------
    resid = log_mbk - log_line
    axD.axhline(0.0, color='0.4', lw=1.0)
    axD.axvspan(6.0, 12.0, color='0.85', alpha=0.45, zorder=0)
    axD.plot(z_grid, resid, lw=2.2, color='k')
    axD.set_xlabel(r'Redshift $z$')
    axD.set_ylabel(r'$\Delta \log_{10} M$', fontsize=9)
    axD.set_xlim(z_grid.min(), z_grid.max())
    lim = max(0.12, 1.2 * np.nanmax(np.abs(resid)))
    axD.set_ylim(-lim, lim)
    axD.tick_params(labelsize=8)
    axD.yaxis.set_major_locator(plt.MaxNLocator(3))

    # ------------------------------ diagnostics ------------------------------
    inb = (z_grid >= 6.0) & (z_grid <= 12.0)
    print('  threshold separation  delta = log10(M_MBK25/M_FFB):')
    print('    max |delta| over 6<z<12 : %.3f dex' % np.nanmax(np.abs(resid[inb])))
    print('    RMS over 6<z<12         : %.3f dex'
          % np.sqrt(np.nanmean(resid[inb] ** 2)))
    ipk = np.nanargmax(np.abs(np.where(inb, resid, np.nan)))
    print('    peaks at z = %.2f (%+0.3f dex)' % (z_grid[ipk], resid[ipk]))
    sgn = np.sign(resid)
    xz = np.where(np.diff(sgn) != 0)[0]
    if len(xz):
        print('    crosses zero at z = %s'
              % ', '.join('%.2f' % np.interp(0.0, resid[i:i + 2][::int(sgn[i + 1] - sgn[i]) // 2 or 1],
                                             z_grid[i:i + 2][::int(sgn[i + 1] - sgn[i]) // 2 or 1])
                          for i in xz))
    for zq in (6.0, 8.0, 9.0, 10.1, 12.0):
        print('    z=%5.1f : MBK25 %.2f  FFB %.2f  delta %+0.3f dex  (median c=%.2f)'
              % (zq, np.interp(zq, z_grid, log_mbk), np.interp(zq, z_grid, log_line),
                 np.interp(zq, z_grid, resid), np.interp(zq, z_grid, c_med)))

    # Linear masses at the epoch used in the SFE figure, for its caption.
    print('  threshold masses in linear units (for figure captions):')
    for zq in (9.0, 10.1):
        print('    z=%5.2f : M_FFB = %.3e   M_MBK25 = %.3e  M_sun'
              % (zq, 10 ** np.interp(zq, z_grid, log_line),
                 10 ** np.interp(zq, z_grid, log_mbk)))

    # Particle counts at the threshold mass -- the resolution check.
    m_part = {'Millennium': 8.60e8 / 0.73, 'miniUchuu': 3.27e8 / 0.6774}
    print('  particles per threshold-mass halo  (m_part: %s):'
          % ', '.join('%s %.2e' % (k, v) for k, v in m_part.items()))
    print('    %5s %12s %14s %14s' % ('z', 'M_FFB', 'N(Millennium)', 'N(miniUchuu)'))
    for zq in (6.0, 8.0, 10.0, 12.0, 14.0):
        M = 10 ** np.interp(zq, z_grid, log_line)
        print('    %5.1f %12.3e %14.0f %14.0f'
              % (zq, M, M / m_part['Millennium'], M / m_part['miniUchuu']))

    # Section 5.2: sigmoid width <-> concentration scatter equivalence.
    d_logM = delta_log_M
    sig_logistic = np.pi * d_logM / np.sqrt(3.0)
    sig_lnc_implied = sig_logistic / 1.5
    sig_lnc_adopted = 0.2
    print('  transition-width equivalence (Section 5.2):')
    print('    logistic sigma for dlogM=%.2f : %.3f dex' % (d_logM, sig_logistic))
    print('    implied sigma_ln c            : %.3f   (adopted %.2f -> differ by %.1f%%)'
          % (sig_lnc_implied, sig_lnc_adopted,
             100.0 * abs(sig_lnc_implied - sig_lnc_adopted) / sig_lnc_adopted))

    # constrained_layout ignores the gridspec hspace, so it leaves a gap between
    # the third-column panel and its residual strip.  Let it settle the overall
    # spacing first (which is what stops the x labels being clipped), then freeze
    # the layout and slide the residual up flush against the panel above.
    fig.canvas.draw()
    try:
        fig.set_layout_engine('none')
    except AttributeError:          # matplotlib < 3.6
        fig.set_constrained_layout(False)
    pL = axL.get_position()
    pM, pD = axM.get_position(), axD.get_position()
    h_tot = pL.y1 - pL.y0                 # full height of the first two panels
    h_res = h_tot / 5.0                   # residual share, matching height_ratios
    axM.set_position([pM.x0, pL.y0 + h_res, pM.width, h_tot - h_res])
    axD.set_position([pD.x0, pL.y0, pD.width, h_res])

    save_figure(fig, os.path.join(OUTPUT_DIR,
                'Selection_Thresholds_Mz' + OUTPUT_FORMAT))

# Standalone plots (load their own data)
# =====================================================================
# PLOT 99: REFEREE DIAGNOSTICS -- prints numbers, draws nothing
# =====================================================================

# ================= PLOT 37: CGM MASS FRACTION AND BARYON CENSUS =================

def plot_37_cgm_census():
    """
    CGM mass fraction and the baryon census versus halo mass.

    Answers referee Major Comment 9: "no CGM property is ever compared to data...
    Even a plot of CGM mass fraction versus halo mass and redshift, with a sanity
    check against the baryon census, would help."

    Three panels:
      1. f_CGM  = m_CGM / M_vir            vs M_vir, several redshifts
      2. f_hot  = m_hot / M_vir            vs M_vir, with observations
      3. total baryon budget / (f_b M_vir) vs M_vir, the census check

    Millennium is drawn solid, microUchuu dashed where its output is available.

    OBSERVATIONS.  Panel 2 is the panel with a direct observational counterpart:
    hot-gas fractions of groups and clusters from X-ray measurements.  No such
    file ships with the repository, so the overlay is switched on by dropping a
    whitespace table at OBS_HOTFRAC below with columns

        log10(M_500/Msun)   f_gas   err_lo   err_hi

    Sensible sources are Gonzalez et al. (2013), Lovisari et al. (2015) and
    Eckert et al. (2016).  Nothing is plotted if the file is absent -- the
    numbers are deliberately not hard-coded here, since transcribing them by
    hand is how errors get in.
    """
    print('Plot 37: CGM mass fraction and baryon census')

    OBS_HOTFRAC = './data/Gas/hot_gas_fraction_groups.dat'

    z_targets = [0.0, 1.0, 2.0, 3.0, 4.0]
    reservoirs = ('CGMgas', 'HotGas', 'ColdGas', 'StellarMass', 'EjectedMass')
    props = list(reservoirs) + ['Mvir', 'Type']
    edges = np.arange(10.4, 14.2, 0.2)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    colours = plt.get_cmap('plasma')(np.linspace(0.0, 0.82, len(z_targets)))

    sims = [('Millennium', PRIMARY_DIR, REDSHIFTS, MASS_CONVERT, BARYON_FRAC, '-')]
    if model_files_exist(MINIUCHUU_DIR):
        # Each simulation must be normalised by its OWN baryon fraction --
        # Millennium uses 0.17 and microUchuu 0.15 (Table 2), so sharing one
        # value understates the microUchuu census by ~12 per cent and would
        # fabricate a difference between the two simulations.
        _mu_hdr = _read_sim_header(MINIUCHUU_DIR)
        _mu_fb = _mu_hdr.get('baryon_frac', BARYON_FRAC) if _mu_hdr else BARYON_FRAC
        sims.append(('microUchuu', MINIUCHUU_DIR, MINIUCHUU_REDSHIFTS,
                     MINIUCHUU_MASS_CONVERT, _mu_fb, '--'))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    fcgm_max = 0.0
    for sim_name, sim_dir, zlist, mconv, fb, ls in sims:
        for z_t, col in zip(z_targets, colours):
            snap = _snap_nearest_z(zlist, z_t)
            try:
                d = load_model(sim_dir, snapshot=f'Snap_{snap}', properties=props)
            except Exception:                                   # noqa: BLE001
                continue
            mvir = np.asarray(d['Mvir'], float)
            cen = (np.asarray(d['Type'], int) == 0) & (mvir > 0)
            if cen.sum() < 100:
                continue
            lm = np.log10(np.where(mvir > 0, mvir, 1.0))
            res = [np.asarray(d[k], float) for k in reservoirs]

            f_cgm, f_hot, f_tot = [], [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                k = cen & (lm >= lo) & (lm < hi)
                if k.sum() < 20:
                    f_cgm.append(np.nan); f_hot.append(np.nan); f_tot.append(np.nan)
                    continue
                tot = mvir[k].sum()
                f_cgm.append(res[0][k].sum() / tot)
                f_hot.append(res[1][k].sum() / tot)
                f_tot.append(sum(r[k].sum() for r in res) / tot / fb)

            lab = f'$z={zlist[snap]:.1f}$' if sim_name == 'Millennium' else None
            axes[0].plot(ctr, f_cgm, ls, color=col, lw=1.9, marker='o', ms=3.2,
                         label=lab)
            axes[1].plot(ctr, f_hot, ls, color=col, lw=1.9, marker='o', ms=3.2)
            axes[2].plot(ctr, f_tot, ls, color=col, lw=1.9, marker='o', ms=3.2)
            fcgm_max = max(fcgm_max, np.nanmax(f_cgm) if np.any(np.isfinite(f_cgm)) else 0.0)

    # Observational hot-gas fraction: Popesso et al. (2024), eROSITA/eFEDS,
    # their eq. 4,  f_gas,500 = (2.23 +/- 0.18)e-7 (M500/Msun)^(0.39 +/- 0.02),
    # calibrated over M500 ~ 5e12 - 5e14 Msun.
    #
    # CAVEAT, stated in the caption: the observation is gas within R500 relative
    # to M500, while the model plots the whole hot reservoir within R_vir
    # relative to M_vir.  Converting M500 -> M_vir raises the mass by ~1/0.72,
    # and integrating the gas out to R_vir rather than R500 raises the gas by a
    # comparable factor, so the two corrections largely cancel and the band is
    # an approximate rather than an exact comparison.
    _m500 = np.logspace(np.log10(5e12), np.log10(5e14), 40)
    _mvir = _m500 / 0.72
    for _a, _s, _al in ((2.23e-7, 0.39, 0.30),):
        _lo = (_a - 0.18e-7) * _m500 ** (_s - 0.02)
        _hi = (_a + 0.18e-7) * _m500 ** (_s + 0.02)
        axes[1].fill_between(np.log10(_mvir), _lo, _hi, color='0.45',
                             alpha=_al, lw=0, zorder=Z_OBS,
                             label='Popesso+24 (eROSITA)')
    axes[1].legend(fontsize=9, frameon=False, loc='upper left')

    # optional tabulated points, if the user supplies them
    if os.path.exists(OBS_HOTFRAC):
        try:
            o = np.loadtxt(OBS_HOTFRAC)
            axes[1].errorbar(o[:, 0], o[:, 1], yerr=[o[:, 2], o[:, 3]],
                             fmt='s', color='k', ms=5, lw=1, alpha=0.7,
                             zorder=Z_OBS, label='X-ray groups/clusters')
            axes[1].legend(fontsize=9, frameon=False, loc='upper left')
        except Exception as exc:                                # noqa: BLE001
            print(f'  hot-gas fraction observations not plotted: {exc}')
    else:
        print(f'  no observational overlay: {OBS_HOTFRAC} absent (see docstring)')

    mshock = np.log10(6.0e11)
    for a in axes:
        a.axvline(mshock, color='0.45', ls='--', lw=1.3, zorder=1)
        a.set_xlabel(r'$\log_{10}(M_{\rm vir}/{\rm M_\odot})$', fontsize=13)
        a.set_xlim(edges[0], edges[-1])
        a.tick_params(labelsize=10)

    axes[0].set_ylabel(r'$f_{\rm CGM}=m_{\rm CGM}/M_{\rm vir}$', fontsize=13)
    axes[0].set_ylim(0.0, max(0.15, 1.08 * fcgm_max))
    axes[0].text(mshock + 0.06, 0.94 * axes[0].get_ylim()[1],
                 r'$M_{\rm shock}$', fontsize=10, color='0.35')
    axes[0].legend(fontsize=10, frameon=False, loc='upper right')

    axes[1].set_ylabel(r'$f_{\rm hot}=m_{\rm hot}/M_{\rm vir}$', fontsize=13)
    axes[1].set_ylim(0.0, 0.20)

    axes[2].axhline(1.0, color='0.45', ls=':', lw=1.2)
    axes[2].set_ylabel(r'$(\sum_i m_i)/(f_{\rm b}M_{\rm vir})$', fontsize=13)
    axes[2].set_ylim(0.3, 1.15)

    if len(sims) > 1:
        axes[2].plot([], [], '-',  color='0.3', label='Millennium')
        axes[2].plot([], [], '--', color='0.3', label='microUchuu')
        axes[2].legend(fontsize=9, frameon=False, loc='lower right')

    fig.tight_layout()
    outputFile = os.path.join(OUTPUT_DIR, 'CGMCensus' + OUTPUT_FORMAT)
    save_figure(fig, outputFile)


def plot_99_referee_diagnostics():
    """Print every number the referee response needs, labelled by comment.

    Masses arrive from load_model() already in Msun (MASS_CONVERT is
    applied at load time to _MASS_PROPS), so no unit conversion is done
    here. Draws nothing. Run as `python paper_plots.py 99` and keep the stdout;
    each block names the referee comment and the manuscript location that
    needs the value, so the output pastes into docs/referee/ directly.
    """
    print()
    print('#' * 78)
    print('# REFEREE DIAGNOSTICS')
    print(f'#   primary  {PRIMARY_DIR}')
    print(f'#   volume   {VOLUME:.4e} Mpc^3   h = {HUBBLE_H}')
    # Record the configuration so the output is self-documenting: without this
    # a saved log cannot be matched to the run that produced it.
    try:
        import h5py as _h5
        with _h5.File(find_model_files(PRIMARY_DIR)[0], 'r') as _f:
            _rt = dict(_f['Header/Runtime'].attrs)
        _keys = ('FIREmodeOn', 'SFprescription', 'CGMrecipeOn', 'FeedbackFreeModeOn',
                 'CGMDensityProfile', 'PrecipCriterionOn', 'RegimeRandomMode',
                 'FFBRandomMode', 'SNEnergyConservationOn', 'MaxSNEnergyCoupling',
                 'FeedbackReheatingEpsilon', 'FeedbackEjectionEfficiency',
                 'EtaSN', 'EnergySN', 'RamPressureStrippingOn')
        print('#   runtime  ' + ', '.join(f'{k}={_rt[k]}' for k in _keys if k in _rt))
    except Exception as _exc:                          # noqa: BLE001
        print(f'#   runtime  (unavailable: {_exc})')
    print('#' * 78)

    def head(title, comment):
        print()
        print('=' * 74)
        print(title)
        print(f'  [{comment}]')
        print('=' * 74)

    # Precipitation coefficients. The paper defines f_inflow as the bare
    # sigmoid (Eq. 6); the rate of Eq. 5 carries the extra condensation
    # factor. The two give very different transition fractions, so report both.
    def f_sig(r):
        return precipitation_fraction(r, include_condensation=False)

    def f_rate(r):
        return precipitation_fraction(r, include_condensation=True)

    z_targets = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
    snaps = []
    for zt in z_targets:
        s = _snap_for_z(REDSHIFTS, zt) if '_snap_for_z' in globals() \
            else _snap_nearest_z(REDSHIFTS, zt)
        if s not in snaps:
            snaps.append(s)

    # Also load the snapshots Fig. 8 actually plots, matched to the secondary
    # simulation's grid the same way plot_24 matches them.  Without these the
    # block 4 table lands on z = 0.51/1.08/3.06/6.20 for a figure drawn at
    # z = 0.989/2.070/3.866/5.724, and only z = 0 and z = 2.07 can be both
    # quoted in the text and read off the figure.
    for zt in (0.0, 1.0, 2.0, 4.0, 6.0):
        zref = (MINIUCHUU_REDSHIFTS[_snap_nearest_z(MINIUCHUU_REDSHIFTS, zt)]
                if model_files_exist(MINIUCHUU_DIR) else zt)
        s = _snap_nearest_z(REDSHIFTS, zref)
        if s not in snaps:
            snaps.append(s)
    snaps.sort(reverse=True)          # descending snapshot = ascending redshift

    props = ['StellarMass', 'ColdGas', 'MetalsColdGas', 'SfrDisk', 'SfrBulge',
             'Mvir', 'Vvir', 'VvirPeak', 'Regime', 'FFBRegime', 'CGMgas',
             'tcool_over_tff', 'MassLoading', 'mdot_cool', 'mdot_stream', 'Type']
    data = {}
    for s in snaps:
        try:
            data[s] = load_model(PRIMARY_DIR, snapshot=f'Snap_{s}', properties=props)
        except Exception as exc:                      # noqa: BLE001
            print(f'  (Snap_{s} unavailable: {exc})')

    def zof(s):
        try:
            return REDSHIFTS[s]
        except Exception:                             # noqa: BLE001
            return float('nan')

    g0 = data.get(SNAP_Z0, {})

    # -----------------------------------------------------------------
    head('1. CGM virial temperature span',
         'Major 3 / A2 -- fills [GET NUMBERS] on p4')
    if 'Vvir' in g0:
        m = (g0['Regime'] == 0) & (g0['Type'] == 0) & (g0['CGMgas'] > 0) & (g0['Vvir'] > 0)
        T = 35.9 * g0['Vvir'][m] ** 2            # VIRIAL_TEMP_COEFF, mu = 0.59
        if T.size:
            print(f'  CGM-regime centrals at z=0: N = {T.size}')
            print(f'  T_vir   1st pct {np.percentile(T, 1):.2e}   median {np.median(T):.2e}'
                  f'   99th pct {np.percentile(T, 99):.2e} K')
            print(f'  min {T.min():.2e}   max {T.max():.2e} K')
            print(f'  fraction below 1.5e4 K: {100 * np.mean(T < 1.5e4):.2f}%')
            print('  --> quote the 1st-99th percentile range, rounded.')

    # -----------------------------------------------------------------
    head('2. Precipitation transition fractions vs redshift',
         'Major 4(d) -- appendix. BOTH definitions; quoting one invites '
         'the charge of picking the convenient one')
    print(f'{"z":>6}{"N":>9}  |{"sigmoid (Eq.6)":^27}|{"rate (Eq.5)":^27}')
    print(f'{"":>6}{"":>9}  |{"sat>=0.9":>10}{"trans":>9}{"med":>8}'
          f'|{"sat>=0.9":>10}{"trans":>9}{"med":>8}')
    for s in snaps:
        g = data.get(s)
        if not g or 'tcool_over_tff' not in g:
            continue
        # The CGMgas > 0 cut is load-bearing: cooling_recipe_cgm returns early
        # when CGMgas <= 0, leaving tcool_over_tff stale from a previous step.
        m = (g['Regime'] == 0) & (g['Type'] == 0) & (g['CGMgas'] > 0) & \
            (g['tcool_over_tff'] > 0)
        r = g['tcool_over_tff'][m]
        if r.size < 10:
            continue
        cells = f'{zof(s):6.2f}{r.size:9d}  |'
        for fn in (f_sig, f_rate):
            f_ = np.atleast_1d(fn(r))
            cells += (f'{100 * np.mean(f_ >= 0.9):10.1f}'
                      f'{100 * np.mean((f_ > 0.1) & (f_ < 0.9)):9.1f}'
                      f'{np.median(f_):8.3f}|')
        print(cells)
    print('  percentages; "trans" = 0.1 < f < 0.9')
    print('  CGM-mass-weighted transition fraction (rate defn):')
    for s in snaps[:5]:
        g = data.get(s)
        if not g or 'tcool_over_tff' not in g:
            continue
        m = (g['Regime'] == 0) & (g['Type'] == 0) & (g['CGMgas'] > 0) & \
            (g['tcool_over_tff'] > 0)
        r, w = g['tcool_over_tff'][m], g['CGMgas'][m]
        if r.size < 10:
            continue
        f_ = np.atleast_1d(f_rate(r))
        tr = (f_ > 0.1) & (f_ < 0.9)
        print(f'    z={zof(s):5.2f}  {100 * w[tr].sum() / w.sum():5.2f}%'
              f'   median t_cool/t_ff = {np.median(r):.3f}')

    # -----------------------------------------------------------------
    head('3. Mass-metallicity offsets',
         'Major 2(h) -- p7 text and the Fig. 4 caption')
    if 'MetalsColdGas' in g0:
        ms = g0['StellarMass']
        cg = g0['ColdGas']
        mz = g0['MetalsColdGas']
        with np.errstate(invalid='ignore', divide='ignore'):
            sel = (ms > 1e8) & (cg > 0) & (mz > 0) & (cg / (cg + ms) > 0.1)
        lm = np.log10(ms[sel])
        oh = 9.0 + np.log10(mz[sel] / cg[sel] / 0.02)
        print(f'  Figure-4 selection: N = {int(sel.sum())}')
        for label, fn in (('Andrews & Martini 2013', 'MMAdrews13.dat'),
                          ('Curti+20', 'Curti2020.dat')):
            path = os.path.join(OBS_DIR, 'metallicity', fn)
            if not os.path.exists(path):
                print(f'  {label}: not found at {path}')
                continue
            d = np.loadtxt(path)
            for lo, hi in ((8.0, 9.0), (10.0, 11.0)):
                diffs = []
                for mo, zo in zip(d[:, 0], d[:, 1]):
                    if not (lo <= mo <= hi):
                        continue
                    b = (lm > mo - 0.15) & (lm < mo + 0.15)
                    if b.sum() > 20:
                        diffs.append(np.median(oh[b]) - zo)
                if diffs:
                    print(f'  {label:24s} {lo:4.1f}-{hi:4.1f}: '
                          f'mean {np.mean(diffs):+.3f} dex, '
                          f'median {np.median(diffs):+.3f}, N_pts {len(diffs)}')
        print('  model median in 0.25-dex bins (gives the crossing point):')
        for c in np.arange(8.125, 11.6, 0.25):
            b = (lm > c - 0.125) & (lm < c + 0.125)
            if b.sum() > 20:
                print(f'    log m* {c:5.2f}   12+log(O/H) = {np.median(oh[b]):.3f}'
                      f'   N={int(b.sum())}')

    # -----------------------------------------------------------------
    head('4. Mass loading and the energy bound',
         'Major 2(c),(g) and sub-point B1 -- the p17 rewrite')
    esn = 5.0e-3 * 1.0e51                # eta_SN * E_SN, erg per Msun
    cap = 2.0                            # MaxSNEnergyCoupling
    msun_g = 1.989e33
    for s in snaps:
        g = data.get(s)
        if not g or 'MassLoading' not in g:
            continue
        sfr = g['SfrDisk'] + g['SfrBulge']
        m = (sfr > 0) & (g['MassLoading'] > 0)
        if m.sum() < 10:
            continue
        eta, w = g['MassLoading'][m], sfr[m]
        v = (g['VvirPeak'][m] if 'VvirPeak' in g and g['VvirPeak'].size
             else g['Vvir'][m]).astype(float)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            eta_max = cap * esn / ((v * 1e5) ** 2 * msun_g)
        at_cap = np.isfinite(eta_max) & (eta >= 0.999 * eta_max)
        print(f'  z={zof(s):5.2f} N={int(m.sum()):8d}'
              f'  <eta>_SFR={np.average(eta, weights=w):7.2f}'
              f'  median={np.median(eta):6.2f}  max={eta.max():8.1f}'
              f'  SFR at eta>100: {100 * w[eta > 100].sum() / w.sum():5.2f}%'
              f'  | bound acts on {100 * w[at_cap].sum() / w.sum():5.2f}% of SFR')
        if at_cap.sum():
            print(f'         capped population V_vir: min {v[at_cap].min():.0f}'
                  f'  median {np.median(v[at_cap]):.0f} km/s')

    # Values at fixed V_vir, for the Figure 8 text. eta is the stored
    # MassLoading (post-bound), and the ejected fraction is reconstructed
    # exactly as compute_sn_feedback() evaluates it, so both match the figure.
    print()
    print('  medians at fixed V_vir -- quote these in the Fig. 8 paragraph')
    print('  eta = mdot_reheat/mdot_*   eject = mdot_eject/mdot_*  (Msun per Msun of stars)')
    _p = _feedback_params()
    _esn = _sn_energy_per_mass_kms2(_p)
    _vt = (30.0, 50.0, 80.0, 120.0, 160.0)
    print(f'  {"z":>6}' + ''.join(f'{f"V={vv:.0f}":>18}' for vv in _vt))
    print(f'  {"":>6}' + ''.join(f'{"eta":>8}{"eject":>10}' for _ in _vt))
    # Use the SAME snapshots Fig. 8 plots, matched to the secondary simulation's
    # grid exactly as plot_24 does.  Iterating this block's own snapshot list
    # instead produced a table at z = 0.51, 1.08, 3.06, 6.20 for a figure drawn
    # at z = 0.989, 2.070, 3.866, 5.724, so only z = 0 and z = 2.07 could be
    # quoted in the text and read off the figure.
    _fig_snaps = []
    for _zt in (0.0, 1.0, 2.0, 4.0, 6.0):
        if model_files_exist(MINIUCHUU_DIR):
            _zref = MINIUCHUU_REDSHIFTS[_snap_nearest_z(MINIUCHUU_REDSHIFTS, _zt)]
        else:
            _zref = _zt
        _s = _snap_nearest_z(REDSHIFTS, _zref)
        if _s not in _fig_snaps:
            _fig_snaps.append(_s)
    for s in _fig_snaps:
        g = data.get(s)
        if not g or 'MassLoading' not in g:
            print(f'  Snap_{s} (z={zof(s):.3f}) not loaded -- add it to the'
                  ' snapshot list so the table covers every plotted redshift')
            continue
        z_ = zof(s)
        vv_all = (g['VvirPeak'] if 'VvirPeak' in g and g['VvirPeak'].size
                  else g['Vvir']).astype(float)
        et_all = g['MassLoading'].astype(float)
        row = f'  {z_:6.2f}'
        for vt in _vt:
            k = (et_all > 0) & (vv_all > 0) & (np.abs(np.log10(vv_all / vt)) < 0.05)
            if k.sum() < 30:
                row += f'{"--":>8}{"--":>10}'
                continue
            eta_m = float(np.median(et_all[k]))
            coup = _p['eps_halo'] * _fire_scaling(vt, z_, _p)
            if _p['sn_bound']:
                coup = min(coup, _p['eps_max'])
            e_fb, e_lift = coup * 0.5 * _esn, 0.5 * eta_m * vt * vt
            ej = (e_fb - e_lift) / (0.5 * vt * vt) if e_fb > e_lift else 0.0
            row += f'{eta_m:8.1f}{ej:10.0f}'
        print(row)
    print('  --> "eject" is not clamped to the available reservoir gas, so it is')
    print('      an upper bound at low V_vir, as in the right panel of Fig. 8.')

    # Where the energy cap takes over the mass loading.  capped_eta_reheat()
    # limits eta to eps_max*E_SN/V^2, which carries NO redshift dependence,
    # while the uncapped FIRE value goes as (1+z)^alpha (V/60)^-1 above the
    # break.  The cap falls as V^-2 and the FIRE value as V^-1, so they cross,
    # and above the crossing every redshift lies on the same eta_max(V) curve.
    # That is the high-z downturn and the merging of adjacent redshifts at
    # large V_vir -- it needs haloes massive enough to reach these velocities
    # while z is still high, so it only appears in a large box.
    # The OTHER clamp: sn_energy_coupling() caps eps_halo*f at eps_max, so E_FB
    # never exceeds the whole supernova budget.  Distinct from the mass-loading
    # cap below and it binds far more often -- Eq. 14 as printed is unbounded and
    # asks for up to 16x the budget in high-z dwarfs.
    if _p['sn_bound']:
        _fcrit = _p['eps_max'] / _p['eps_halo']
        print()
        print(f'  energy cap on the EJECTION coupling: eps_halo*f <= eps_max'
              f'  <=>  f <= {_fcrit:.2f}')
        print('  V_vir below which the coupling saturates, and the SFR it affects:')
        for s in _fig_snaps + [s for s in snaps if s not in _fig_snaps]:
            g = data.get(s)
            if not g or 'Vvir' not in g:
                continue
            z_ = zof(s)
            fz = (1.0 + z_) ** _p['alpha_z']
            v_sat = (60.0 * (fz / _fcrit) if fz >= _fcrit
                     else 60.0 * (_fcrit / fz) ** (-1.0 / 3.2))
            sfr = g['SfrDisk'] + g['SfrBulge']
            vv = np.asarray(g['Vvir'], float)
            k = (sfr > 0) & (vv > 0)
            if k.sum() < 100:
                continue
            hit = _p['eps_halo'] * _fire_scaling(vv[k], z_, _p) > _p['eps_max']
            print(f'    z={z_:5.2f}  saturates for V_vir < {v_sat:6.1f} km/s'
                  f'   {100 * sfr[k][hit].sum() / sfr[k].sum():6.2f}% of SFR'
                  f'   {100 * hit.mean():6.2f}% of galaxies')
        print()
        print('  energy cap on the mass loading: eta_max = eps_max*E_SN/V^2'
              f'  = {_p["eps_max"] * _esn:.3g} / V^2   (no z dependence)')
        print('  above V_cap the FIRE scaling is capped and all redshifts merge:')
        for zt in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
            v_cap = (_p['eps_max'] * _esn
                     / (_p['eps_disk'] * FIRE_V_CRIT * (1.0 + zt) ** _p['alpha_z']))
            print(f'    z={zt:4.1f}  V_cap = {v_cap:7.0f} km/s'
                  f'   eta there = {_p["eps_max"] * _esn / v_cap ** 2:6.2f}')
        for s in snaps:
            g = data.get(s)
            if not g or 'MassLoading' not in g:
                continue
            z_ = zof(s)
            vv_all = (g['VvirPeak'] if 'VvirPeak' in g and g['VvirPeak'].size
                      else g['Vvir']).astype(float)
            et_all = g['MassLoading'].astype(float)
            k = (et_all > 0) & (vv_all > 0)
            if k.sum() < 50:
                continue
            emax = _p['eps_max'] * _esn / vv_all[k] ** 2
            hit = et_all[k] >= 0.999 * emax
            print(f'    z={z_:5.2f}  {100 * hit.mean():5.2f}% of galaxies on the cap'
                  + (f'   V_vir there: median {np.median(vv_all[k][hit]):.0f},'
                     f' max {vv_all[k][hit].max():.0f} km/s' if hit.any() else ''))

    # -----------------------------------------------------------------
    head('5. Dwarf regulation, not shutdown', 'Major 2(g) -- the p17 rewrite')
    if 'ColdGas' in g0:
        ms = g0['StellarMass']
        cg = g0['ColdGas']
        sfr = g0['SfrDisk'] + g0['SfrBulge']
        for lo, hi in ((1e7, 1e8), (1e8, 1e9), (1e9, 1e10)):
            m = (ms > lo) & (ms < hi)
            if m.sum() < 10:
                continue
            with np.errstate(invalid='ignore', divide='ignore'):
                fg = cg[m] / (cg[m] + ms[m])
                ss = np.where(sfr[m] > 0,
                              np.log10(np.maximum(sfr[m], 1e-30) / ms[m]), -99.0)
            print(f'  m* {lo:.0e}-{hi:.0e}: N={int(m.sum()):8d}'
                  f'  median f_gas={np.median(fg):.3f}'
                  f'  median log sSFR={np.median(ss[ss > -90]):.2f}'
                  f'  quiescent={100 * np.mean(ss < SSFR_CUT):.1f}%')

    # -----------------------------------------------------------------
    head('6. Quiescent fraction vs stellar mass',
         'Major 8(a) and p8 RHS -- quantify "qualitative improvement"')
    if 'StellarMass' in g0:
        print(f'  quiescent: log10(sSFR/yr^-1) < {SSFR_CUT}')
        ms = g0['StellarMass']
        sfr = g0['SfrDisk'] + g0['SfrBulge']
        lms = np.log10(np.maximum(ms, 1.0))
        # Split centrals from satellites: the observational comparisons differ.
        # All-galaxy fractions compare to the type-split mass functions (Bell+03,
        # Baldry+12, Moffett+16); centrals-only compares to Geha+12 and Wetzel+12,
        # which is the relevant test for the dwarf-quiescent problem SAGE16 had.
        typ = g0['Type'] if 'Type' in g0 else np.zeros(len(ms), dtype=int)
        cen = (np.asarray(typ, int) == 0)
        print(f'    {"log m*":>8}{"all":>10}{"centrals":>10}{"satellites":>12}'
              f'{"N_all":>10}{"f_sat":>8}')
        for c in np.arange(8.25, 12.1, 0.5):
            b = (lms > c - 0.25) & (lms < c + 0.25)
            if b.sum() < 20:
                continue
            with np.errstate(invalid='ignore', divide='ignore'):
                ss = np.where(sfr > 0,
                              np.log10(np.maximum(sfr, 1e-30) / np.maximum(ms, 1.0)),
                              -99.0)
            q = ss < SSFR_CUT
            bc, bs = b & cen, b & ~cen
            fa = 100 * np.mean(q[b])
            fc = 100 * np.mean(q[bc]) if bc.sum() >= 20 else np.nan
            fs = 100 * np.mean(q[bs]) if bs.sum() >= 20 else np.nan
            print(f'    {c:8.2f}{fa:9.1f}%{fc:9.1f}%{fs:11.1f}%'
                  f'{int(b.sum()):10d}{bs.sum()/b.sum():8.2f}')
        print('    --> all-galaxy fractions compare to the type-split mass')
        print('        functions; centrals-only is the test for the dwarf')
        print('        quiescent problem (Geha+12, Wetzel+12).')

    # -----------------------------------------------------------------
    head('7. Cold stream fraction vs halo mass',
         'p19 LHS -- the 93% / 56% numbers, which cannot be read off Fig. 13')
    for s in snaps:
        g = data.get(s)
        if not g or 'mdot_stream' not in g or not g['mdot_stream'].size:
            continue
        mv = g['Mvir']
        tot = g['mdot_cool'] + g['mdot_stream']
        m = (g['Type'] == 0) & (tot > 0)
        if m.sum() < 10:
            continue
        parts = [f'  z={zof(s):5.2f}  (N with accretion = {int(m.sum())})']
        for lo, hi in ((1e10, 1e11), (1e11, 1e12), (1e12, 1e13), (1e13, 1e15)):
            b = m & (mv > lo) & (mv < hi)
            if b.sum() >= 5:
                parts.append(f'{lo:.0e}-{hi:.0e}: '
                             f'{100 * g["mdot_stream"][b].sum() / tot[b].sum():5.1f}%'
                             f' (N={int(b.sum())})')
        print('   '.join(parts) if len(parts) > 1 else
              parts[0] + '   no mass bin with >=5 accreting centrals')
    print('  fraction of total accretion delivered by cold streams')

    # -----------------------------------------------------------------
    head('8. FFB population fraction',
         'p15 -- "state what fraction of the z~5 population is in FFB mode"')
    for s in snaps:
        g = data.get(s)
        if not g or 'FFBRegime' not in g:
            continue
        m = g['Type'] == 0
        if m.sum() < 10:
            continue
        ms_ = g['StellarMass'][m]
        ffb = g['FFBRegime'][m] == 1
        big = ms_ > 1e8
        frac_big = 100 * np.mean(ffb[big]) if big.sum() else float('nan')
        print(f'  z={zof(s):5.2f}  N_cen={int(m.sum()):8d}'
              f'  FFB {100 * np.mean(ffb):5.2f}%'
              f'  | among m*>1e8: {frac_big:5.2f}% (N={int(big.sum())})')

    # -----------------------------------------------------------------
    head('9. Is there an M_shock feature in the SMF?',
         'Major 9 / J2 -- answering the Fig. 13 dips in text rather than by runs')
    if 'Mvir' in g0:
        mshock = 6.0e11
        ms = g0['StellarMass']
        mv = g0['Mvir']
        near = (mv > 0.5 * mshock) & (mv < 2.0 * mshock) & (g0['Type'] == 0) & (ms > 0)
        if near.sum() > 10:
            lmn = np.log10(ms[near])
            print(f'  centrals within a factor 2 of M_shock = {mshock:.1e} M_sun:'
                  f'  N={int(near.sum())}')
            print(f'  their log m*: 16th {np.percentile(lmn, 16):.2f}'
                  f'  median {np.median(lmn):.2f}  84th {np.percentile(lmn, 84):.2f}')
        bw = 0.1
        edges = np.arange(9.0, 12.0 + bw, bw)
        n, _ = np.histogram(np.log10(np.maximum(ms[ms > 0], 1.0)), bins=edges)
        ok = n > 10
        if ok.sum() > 5:
            lphi = np.log10(np.maximum(n[ok] / VOLUME / bw, 1e-30))
            d2 = np.gradient(np.gradient(lphi))
            cen = (0.5 * (edges[1:] + edges[:-1]))[ok]
            i = int(np.argmax(np.abs(d2)))
            print(f'  largest |d2 log phi| over 9 < log m* < 12: {np.abs(d2).max():.3f}'
                  f'  at log m* = {cen[i]:.2f}')
            print('  --> a step at M_shock would appear as a localised spike here.')

    # -----------------------------------------------------------------
    head('10. Stellar mass density',
         'Major 5(c) -- claimed in the Conclusion, no figure exists')
    for s in snaps:
        g = data.get(s)
        if not g or 'StellarMass' not in g:
            continue
        rho = g['StellarMass'].sum() / VOLUME
        print(f'  z={zof(s):5.2f}  rho_* = {rho:.4e} M_sun/Mpc^3'
              f'   log10 = {np.log10(max(rho, 1e-30)):.4f}')

    # -----------------------------------------------------------------
    head('11. Stellar mass function vs observations, per redshift bin',
         'Major 6(1) -- the Conclusion claims "within ~0.2 dex from z = 0 to 12"')
    try:
        obs_smf = _load_smf_grid_observations()
    except Exception as exc:                          # noqa: BLE001
        obs_smf = None
        print(f'  observational compilation unavailable: {exc}')
    if obs_smf:
        zbins = [(0, 0.5), (0.5, 0.8), (0.8, 1.1), (1.1, 1.5), (1.5, 2.0),
                 (2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.5), (4.5, 5.5),
                 (5.5, 6.5), (6.5, 7.5), (7.5, 8.5), (8.5, 9.5), (9.5, 12.0)]
        print(f'  {"z bin":>11}{"N pts":>7}{"med|off|":>10}{"RMS":>7}{"max":>7}'
              f'{"SAGE16(N,floor)":>18}{"  obs-obs":>11}')
        every, rows, every_v = [], [], []
        for lo, hi in zbins:
            s = _snap_for_z(REDSHIFTS, 0.5 * (lo + hi)) if '_snap_for_z' in globals() \
                else _snap_nearest_z(REDSHIFTS, 0.5 * (lo + hi))
            try:
                d = load_model(PRIMARY_DIR, snapshot=f'Snap_{s}',
                               properties=['StellarMass'])
            except Exception:                         # noqa: BLE001
                continue
            ms = d['StellarMass'][d['StellarMass'] > 0]
            if ms.size < 50:
                continue
            x, phi, _ = mass_function(np.log10(ms), VOLUME, binwidth=0.2,
                                      mass_range=(7.0, 13.0))
            offs, sets = [], [o for o in obs_smf if lo <= o['z'] < hi]
            for o in sets:
                for lm, lp in zip(o['log_mass'], o['log_phi']):
                    if not np.isfinite(lp):
                        continue
                    j = int(np.argmin(np.abs(x - lm)))
                    # skip bins below the density floor and unmatched masses
                    if abs(x[j] - lm) > 0.15 or not np.isfinite(phi[j]) or phi[j] < -6.3:
                        continue
                    offs.append(phi[j] - lp)

            # The same measurement against SAGE16.  Without it the table states
            # how close SAGE26 sits to the data but not that it is closer than
            # the model it replaces, which is what the SMF grid is showing.
            # Paired against the SAME observational points SAGE26 was scored
            # on.  Scoring each model only where it has galaxies flatters
            # whichever one fails by producing nothing: SAGE16's SMF drops below
            # the density floor at high z, so those points would be skipped
            # rather than counted as large offsets, and SAGE16 would appear to
            # beat SAGE26 in exactly the bins where it is worst.  Points SAGE16
            # cannot reach are counted separately (n_floor) instead of dropped.
            offs_v, n_floor = [], 0
            try:
                dv = load_model(VANILLA_DIR, snapshot=f'Snap_{s}',
                                properties=['StellarMass'])
                msv = dv['StellarMass'][dv['StellarMass'] > 0]
                if msv.size >= 50:
                    xv, phiv, _ = mass_function(np.log10(msv), VOLUME,
                                                binwidth=0.2, mass_range=(7.0, 13.0))
                    for o in sets:
                        for lm, lp in zip(o['log_mass'], o['log_phi']):
                            if not np.isfinite(lp):
                                continue
                            # only points SAGE26 was scored on
                            j = int(np.argmin(np.abs(x - lm)))
                            if (abs(x[j] - lm) > 0.15 or not np.isfinite(phi[j])
                                    or phi[j] < -6.3):
                                continue
                            jv = int(np.argmin(np.abs(xv - lm)))
                            if abs(xv[jv] - lm) > 0.15:
                                continue              # no matching mass bin
                            # An empty bin comes back NaN, not as a small phi, so
                            # the isfinite test has to be counted rather than
                            # skipped -- it IS the "SAGE16 makes no such
                            # galaxies" case and is the whole high-z story.
                            if not np.isfinite(phiv[jv]) or phiv[jv] < -6.3:
                                n_floor += 1
                                continue
                            offs_v.append(phiv[jv] - lp)
            except Exception:                         # noqa: BLE001
                pass
            # scatter between the observational determinations themselves, for
            # context: at high z this exceeds the model-data offset, so the
            # latter is not a clean measure of model error
            oo = []
            for i in range(len(sets)):
                for k in range(i + 1, len(sets)):
                    a, b = sets[i], sets[k]
                    if abs(a['z'] - b['z']) > 0.5:
                        continue
                    bm = np.asarray(b['log_mass'])
                    for lm, lp in zip(a['log_mass'], a['log_phi']):
                        if not np.isfinite(lp) or bm.size == 0:
                            continue
                        q = int(np.argmin(np.abs(bm - lm)))
                        if abs(bm[q] - lm) > 0.15:
                            continue
                        v = b['log_phi'][q]
                        if np.isfinite(v):
                            oo.append(abs(lp - v))
            if len(offs) < 5:
                continue
            a_ = np.asarray(offs)
            every += list(a_)
            oo_med = np.median(oo) if len(oo) >= 5 else None
            oo_s = f'{oo_med:18.2f}' if oo_med is not None else f'{"--":>18}'
            v_ = np.asarray(offs_v)
            v_med = np.median(np.abs(v_)) if v_.size >= 5 else None
            v_s = (f'{v_med:9.2f}({v_.size:3d},{n_floor:2d})' if v_med is not None
                   else f'{"--":>9}        ')
            print(f'  {f"{lo}-{hi}":>11}{a_.size:7d}{np.median(np.abs(a_)):10.2f}'
                  f'{np.sqrt(np.mean(a_ ** 2)):7.2f}{np.max(np.abs(a_)):7.2f}'
                  f'{v_s}{oo_s}')
            rows.append((lo, hi, a_.size, np.median(np.abs(a_)),
                         np.sqrt(np.mean(a_ ** 2)), oo_med, v_med))
            every_v.extend(offs_v)
        if every:
            e = np.asarray(every)
            ev = np.asarray(every_v)
            print(f'\n  all bins: N={e.size}  median|offset|={np.median(np.abs(e)):.2f} dex'
                  f'  RMS={np.sqrt(np.mean(e ** 2)):.2f} dex')
            if ev.size:
                print(f'  SAGE16 over the same comparisons: '
                      f'median|offset|={np.median(np.abs(ev)):.2f} dex  '
                      f'RMS={np.sqrt(np.mean(ev ** 2)):.2f} dex  (N={ev.size})')
            print('  --> quote the median as a median, not as a bound; the RMS is the')
            print('      larger number and is what "matches within X dex" implies.')

            # LaTeX, ready to paste beneath the stellar mass function grid.
            # Emitted rather than transcribed so the table cannot drift from
            # the measurement it reports.
            print()
            print('  ---- LaTeX table (paste beneath the SMF grid) ' + '-' * 26)
            print(r'\begin{table}')
            print(r'\centering')
            print(r'\caption{Agreement between the SAGE26 stellar mass function and the '
                  r'observational compilation of \Fig{fig:smf_grid}, per redshift bin. '
                  r'$N_{\rm obs}$ is the number of model--observation comparisons '
                  r'in the bin, not a galaxy count. '
                  r'$|\Delta|$ is the median absolute difference in $\log_{10}\phi$ '
                  r'between the model and every observational point in that bin, and '
                  r'$|\Delta|_{\rm C16}$ is the same quantity for SAGE16 over the '
                  r'identical set of comparisons, and '
                  r'$\sigma_{\rm obs}$ is the median absolute difference between '
                  r'independent observational determinations at matched stellar mass. '
                  r'Above $z\simeq4.5$ the observations differ from one another by as '
                  r'much as the model differs from them.}')
            print(r'\label{tab:smf_offsets}')
            print(r'\begin{tabular}{lrcccc}')
            print(r'\hline')
            print(r'Redshift & $N_{\rm obs}$ & median $|\Delta|$ & RMS $\Delta$ & '
                  r'$|\Delta|_{\rm C16}$ & '
                  r'$\sigma_{\rm obs}$ \\')
            print(r' & & (dex) & (dex) & (dex) & (dex) \\')
            print(r'\hline')
            for lo, hi, n, med, rms, oo, vmed in rows:
                oos = f'{oo:.2f}' if oo is not None else r'\nodata'
                vs = f'{vmed:.2f}' if vmed is not None else r'\nodata'
                print(rf'${lo}<z<{hi}$ & {n} & {med:.2f} & {rms:.2f} & {vs} & {oos} \\')
            print(r'\hline')
            _allv = (rf'{np.median(np.abs(ev)):.2f}' if ev.size else r'\nodata')
            print(rf'All & {e.size} & {np.median(np.abs(e)):.2f} & '
                  rf'{np.sqrt(np.mean(e ** 2)):.2f} & {_allv} & \nodata \\')
            print(r'\hline')
            print(r'\end{tabular}')
            print(r'\end{table}')
            print('  ' + '-' * 70)

    # ---------------------------------------------------------------
    head('12. CGM mass fraction and the baryon census',
         'Major 9 -- "even a plot of CGM mass fraction versus halo mass and '
         'redshift, with a sanity check against the baryon census, would help"')
    reservoirs = ('CGMgas', 'HotGas', 'ColdGas', 'StellarMass', 'EjectedMass')
    print(f'  {"z":>5}{"logMvir":>9}{"N":>8}{"f_CGM":>8}{"f_hot":>8}'
          f'{"sum/f_b":>9}')
    for z_t in (0.0, 1.0, 2.0, 3.0, 4.0):
        s = _snap_nearest_z(REDSHIFTS, z_t)
        try:
            d = load_model(PRIMARY_DIR, snapshot=f'Snap_{s}',
                           properties=list(reservoirs) + ['Mvir', 'Type'])
        except Exception:                                   # noqa: BLE001
            continue
        mvir = np.asarray(d['Mvir'], float)
        cen = (np.asarray(d['Type'], int) == 0) & (mvir > 0)
        lm = np.log10(np.where(mvir > 0, mvir, 1.0))
        res = [np.asarray(d[k], float) for k in reservoirs]
        for lo, hi in ((10.5, 11.0), (11.0, 11.5), (11.5, 12.0),
                       (12.0, 12.5), (12.5, 13.5), (13.5, 15.0)):
            k = cen & (lm >= lo) & (lm < hi)
            if k.sum() < 20:
                continue
            tot = mvir[k].sum()
            print(f'  {REDSHIFTS[s]:5.2f}{0.5*(lo+hi):9.2f}{k.sum():8d}'
                  f'{res[0][k].sum()/tot:8.3f}{res[1][k].sum()/tot:8.3f}'
                  f'{sum(r[k].sum() for r in res)/tot/BARYON_FRAC:9.3f}')
        print()
    print('  --> quote the f_CGM peak, where it falls, and the census closure.')
    print('      The census must use each simulation\'s OWN baryon fraction.')

    # ---------------------------------------------------------------
    head('13. Millennium vs the second simulation: cosmology or resolution?',
         'Major 9 -- "how much of the residual difference between the two '
         'SAGE26 curves is cosmology and how much is resolution"')
    if not model_files_exist(MINIUCHUU_DIR):
        print(f'  second simulation not present at {MINIUCHUU_DIR}; skipped')
    else:
        _h2 = _read_sim_header(MINIUCHUU_DIR)
        print(f'  primary  : {PRIMARY_DIR}  box={BOX_SIZE:g}')
        print(f'  secondary: {MINIUCHUU_DIR}  box={_h2["box_size"]:g}')
        print('  CHECK BOTH ARE THE PRODUCTION VOLUMES BEFORE QUOTING.\n')
        x = np.arange(10.0, 12.4, 0.25)
        curves = []
        for D, zl, V in ((PRIMARY_DIR, REDSHIFTS, VOLUME),
                         (MINIUCHUU_DIR, MINIUCHUU_REDSHIFTS, MINIUCHUU_VOLUME)):
            s = _snap_nearest_z(zl, 0.0)
            d = load_model(D, snapshot=f'Snap_{s}', properties=['StellarMass'])
            m = np.asarray(d['StellarMass'], float)
            m = m[m > 0]
            n, _ = np.histogram(np.log10(m), bins=np.append(x, x[-1] + 0.25))
            curves.append(np.log10(np.maximum(n, 1e-10) / V / 0.25))
        print(f'  {"logM*":>7}{"primary":>10}{"secondary":>11}{"offset":>9}')
        offs = []
        for i, lm in enumerate(x):
            if curves[0][i] > -5.5 and curves[1][i] > -5.5:
                dd = curves[1][i] - curves[0][i]
                offs.append((lm, dd))
                print(f'  {lm:7.2f}{curves[0][i]:10.3f}{curves[1][i]:11.3f}{dd:+9.3f}')
        if offs:
            a = np.array(offs)
            lo_, hi_ = a[a[:, 0] < 10], a[a[:, 0] >= 10]
            print(f'\n  mean offset below logM*=10 : {lo_[:, 1].mean():+.3f} dex')
            print(f'  mean offset above logM*=10 : {hi_[:, 1].mean():+.3f} dex')
            print(f'  slope with logM*           : '
                  f'{np.polyfit(a[:, 0], a[:, 1], 1)[0]:+.3f} dex/dex')
            print('  --> a mass-INDEPENDENT offset is cosmology; one that grows')
            print('      towards low mass is resolution.')


    print()
    print('=' * 74)
    print('END REFEREE DIAGNOSTICS')
    print('=' * 74)


def plot_58_coolingrate_vs_mvir(primary, vanilla):
    """Compare Primary and Vanilla cooled-gas rates at z=0."""
    print('Plot 58: cooling rate vs halo mass')

    def valid_points(data):
        mass = np.asarray(data.get('Mvir', []), dtype=float)
        vvir = np.asarray(data.get('Vvir', []), dtype=float)
        rate = np.asarray(data.get('CoolingRate', []), dtype=float)
        
        # Only mask out physically invalid halos. Do NOT mask out quenched cooling rates.
        mask = (mass > 0.0) & (vvir > 0.0)
        
        valid_mass = mass[mask]
        valid_rate = rate[mask]
        valid_vvir = vvir[mask]
        
        # Apply an artificial floor to exactly zero (or negative) cooling rates.
        # 1e3 sits just below the plot's y-axis minimum of 10^4.
        valid_rate = np.maximum(valid_rate, 1e3)
        
        temperature = 35.9 * valid_vvir ** 2
        
        return np.log10(valid_mass), valid_rate, temperature

    def binned_statistics(log_mass, rate, temperature):
        bins = np.arange(10.0, 15.51, 0.05)
        centers = 0.5 * (bins[:-1] + bins[1:])
        median = np.full(centers.size, np.nan)
        sigma = np.full(centers.size, np.nan)
        median_temp = np.full(centers.size, np.nan)
        for i in range(centers.size):
            in_bin = (log_mass >= bins[i]) & (log_mass < bins[i + 1])
            if np.count_nonzero(in_bin) < 0:
                continue
            log_rate = np.log10(rate[in_bin])
            median[i] = np.median(log_rate)
            sigma[i] = np.std(log_rate)
            median_temp[i] = np.median(temperature[in_bin])
        valid = np.isfinite(median) & np.isfinite(median_temp)
        return centers[valid], median[valid], sigma[valid], median_temp[valid]

    def draw_curve(ax, x, log_rate, sigma, temperature, norm, cmap, linestyle):
        if x.size == 0:
            return
        ax.fill_between(x, 10 ** (log_rate - sigma), 10 ** (log_rate + sigma),
                        color=cmap(norm(np.median(temperature))), alpha=0.18)
        if x.size == 1:
            ax.plot(x, 10 ** log_rate, linestyle=linestyle, color=cmap(norm(temperature[0])),
                    marker='o', ms=3)
            return
        points = np.column_stack([x, 10 ** log_rate]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        collection = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2.2,
                                    linestyle=linestyle)
        collection.set_array(temperature[:-1])
        ax.add_collection(collection)
        ax.plot(x, 10 ** log_rate, color='none', linestyle=linestyle,
                label='_nolegend_')

    log_mass, primary_rate, primary_temp = valid_points(primary)
    log_mass_v, vanilla_rate, vanilla_temp = valid_points(vanilla)

    if primary_rate.size == 0 and vanilla_rate.size == 0:
        print('  no positive cooling-rate data found; skipped')
        return

    primary_curve = binned_statistics(log_mass, primary_rate, primary_temp)
    vanilla_curve = binned_statistics(log_mass_v, vanilla_rate, vanilla_temp)
    all_temp = np.concatenate([t for t in (primary_curve[3], vanilla_curve[3]) if t.size])
    norm = LogNorm(vmin=all_temp.min(), vmax=all_temp.max())
    cmap = plt.get_cmap('viridis')

    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    draw_curve(ax, *primary_curve, norm, cmap, '-')
    draw_curve(ax, *vanilla_curve, norm, cmap, '--')

    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
    colorbar.set_label(r'$T_{\rm vir}$ (K)')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\log_{10} M_{\rm vir}\ (M_{\odot})$') 
    ax.set_ylabel(r'$\dot{m}_{\rm cool}\ (M_{\odot}\ {\rm Gyr}^{-1})$')
    # ax.set_xlim(10.0, 15.0)
    # ax.set_ylim(10**4, 10**14)
    ax.legend(handles=[
        Line2D([0], [0], color='black', linestyle='-', label='Primary (SAGE26)'),
        Line2D([0], [0], color='black', linestyle='--', label='Vanilla (SAGE16)'),
    ], frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'CoolingRate_vs_Mvir' + OUTPUT_FORMAT),
                dpi=200)
    plt.close(fig)

# ========================== PLOT: COMBINED BARYON FRACTION & COOLING RATE ==========================

def plot_59_baryon_cooling_combined(primary, vanilla):
    """
    1 column, 2 rows figure.
    Top row: Baryonic fraction vs Mvir at z=0 (Primary and Vanilla models).
    Bottom row: Cooling rate vs Mvir.
    """
    print('Plot: Combined Baryon Fraction and Cooling Rate')

    # Bulletproof wrapper: reconstructs grouping and pads missing arrays if they failed to load
    def _prepare_data_for_baryon_fractions(data):
        safe_data = dict(data)
        n_gals = len(safe_data.get('Type', []))
        if n_gals == 0:
            return safe_data
        
        if 'CentralGalaxyIndex' not in safe_data:
            safe_data['CentralGalaxyIndex'] = np.cumsum(safe_data['Type'] == 0) - 1
            
        for key in ['StellarMass', 'ColdGas', 'HotGas', 'CGMgas', 
                    'IntraClusterStars', 'BlackHoleMass', 'EjectedMass']:
            if key not in safe_data:
                safe_data[key] = np.zeros(n_gals)
                
        return safe_data

    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    
    # ==========================
    # Top Row: Baryon Fractions
    # ==========================
    ax1 = axes[0]
    
    # Calculate fractions safely using the wrapper
    mass_centers_p, bf_p = baryon_fractions_by_halo_mass(_prepare_data_for_baryon_fractions(primary))
    mass_centers_v, bf_v = baryon_fractions_by_halo_mass(_prepare_data_for_baryon_fractions(vanilla))

    components = [
        ('Total',             'Total',          'black'),
        ('StellarMass',       'Stars',          'magenta'),
        ('ColdGas',           'Cold gas',       'blue'),
        ('HotGas',            'Hot gas',        'red'),
        ('CGMgas',            'CGM',            'green'),
        ('IntraClusterStars', 'ICS',            'orange'),
        ('BlackHoleMass',     'Black holes',    'purple'),
        ('EjectedMass',       'Ejected gas',    'goldenrod'),
    ]

    ax1.axhline(y=BARYON_FRAC, color='grey', ls=':', lw=1.5,
               label=rf'$f_{{b}}$ = {BARYON_FRAC:.2f}')

    for key, label, color in components:
        # Primary model (Solid lines + Shading)
        if len(mass_centers_p) > 0 and np.max(bf_p[key]['mean']) > 1e-6:
            ax1.fill_between(mass_centers_p, bf_p[key]['lower'], bf_p[key]['upper'],
                            color=color, alpha=0.15)
            ax1.plot(mass_centers_p, bf_p[key]['mean'],
                     color=color, ls='-', lw=2.5)
        
        # Vanilla model (Dashed lines)
        # The > 1e-6 check ensures padded zeros (like missing CGMgas) are NOT plotted!
        if len(mass_centers_v) > 0 and np.max(bf_v[key]['mean']) > 1e-6:
            ax1.plot(mass_centers_v, bf_v[key]['mean'],
                     color=color, ls='--', lw=2.0)
            
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='k', lw=2.5, ls='-'),
                    Line2D([0], [0], color='k', lw=2.0, ls='--')]
    leg_models = ax1.legend(custom_lines, ['Primary (SAGE26)', 'Vanilla (SAGE16)'], loc='upper left')
    ax1.add_artist(leg_models)
    
    comp_lines = [Line2D([0], [0], color=c, lw=2.5) for _, _, c in components]
    ax1.legend(comp_lines, [l for _, l, _ in components], loc='center right', fontsize='small', ncol=2)

    ax1.set_xlim(11.1, 15.0)
    ax1.set_ylim(0.0, 0.20)
    ax1.set_ylabel(r'Baryon Fraction')
    ax1.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax1.yaxis.set_minor_locator(plt.MultipleLocator(0.01))

    # ==========================
    # Bottom Row: Cooling Rate
    # ==========================
    ax2 = axes[1]
    mvir_bins = np.arange(11.0, 15.0 + 0.1, 0.1)

    if 'CoolingRate' in primary and 'Mvir' in primary:
        w_p = (primary['Mvir'] > 0) & (primary['CoolingRate'] > 0) & (primary['Type'] == 0)
        if np.any(w_p):
            log_mvir_p = np.log10(primary['Mvir'][w_p])
            log_cool_p = np.log10(primary['CoolingRate'][w_p])
            plot_binned_median_1sigma(
                ax2, log_mvir_p, log_cool_p, mvir_bins,
                color='steelblue', label='Primary (SAGE26)',
                alpha=0.25, lw=3.0, ls='-', min_count=3,
                zorder_fill=Z_MODEL_BAND, zorder_line=Z_MODEL_LINE
            )

    if 'CoolingRate' in vanilla and 'Mvir' in vanilla:
        w_v = (vanilla['Mvir'] > 0) & (vanilla['CoolingRate'] > 0) & (vanilla['Type'] == 0)
        if np.any(w_v):
            log_mvir_v = np.log10(vanilla['Mvir'][w_v])
            log_cool_v = np.log10(vanilla['CoolingRate'][w_v])
            plot_binned_median_1sigma(
                ax2, log_mvir_v, log_cool_v, mvir_bins,
                color='purple', label='Vanilla (SAGE16)',
                alpha=0.20, lw=2.5, ls='--', min_count=3,
                zorder_fill=Z_MODEL_BAND_ALT, zorder_line=Z_MODEL_LINE_ALT
            )

    ax2.set_xlabel(r'$\log_{10}\ M_{\mathrm{vir}}\ [M_{\odot}]$')
    ax2.set_ylabel(r'$\log_{10}\ \mathrm{Cooling\ Rate}\ [M_{\odot}/\mathrm{yr}]$')
    ax2.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    
    _standard_legend(ax2, loc='upper left')

    fig.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR, 'BaryonFraction_CoolingRate_Combined' + OUTPUT_FORMAT))


# ========================== MAIN ==========================

# Registry of plot functions
# z=0 plots take (primary, vanilla); evolution plots take (snapdata)
Z0_PLOTS = {
    31: plot_1_stellar_mass_function_ssfr_s,
    30: plot_1_stellar_mass_function_ssfr_q,
    32: plot_1_stellar_mass_function_ssfr_combined,
    2: plot_2_baryon_fraction,
    3: plot_3_gas_metallicity_vs_stellar_mass,
    4: plot_4_bh_bulge_mass,
    5: plot_5_stellar_halo_mass,
    51: plot_5b_stellar_halo_mass_ratio,
    6: plot_6_bulge_mass_size,
    61: plot_6b_bulge_mass_size_median,
    15: plot_15_sfr_vs_stellar_mass,
    24: plot_24_mass_loading_vs_velocity,
    58: plot_58_coolingrate_vs_mvir,
    59: plot_59_baryon_cooling_combined,
}

EVOLUTION_PLOTS = {
    # 7: plot_7_tcool_tff_distribution,
    # 71: plot_7b_inflow_transition_fraction,
    # 8: plot_8_precipitation_fraction,
    # 9: plot_9_cgm_fractions_depletion,
    # 91: plot_9b_cgm_fractions_grid,
    # 92: plot_9c_depletion_grid,
    10: plot_10_sfe_ffb,
    11: plot_11_ffb_properties,
    111: plot_11b_ffb_histograms,
    112: plot_11c_ffb_histograms_mbk25,
    113: plot_11d_ffb_histograms_combined,
    114: plot_11e_ffb_histograms_combined_bulge,
    115: plot_11f_ffb_histograms_combined_diskbulge,
    12: plot_12_sfh_ffb,
    121: plot_12b_ffb_regime_history,
    122: plot_12c_ffb_regime_heatmap,
    123: plot_12d_sfh_ffb_transitions,
    124: plot_12e_sfh_ffb_transitions_mbk25,
    125: plot_12f_sfh_ffb_transitions_stacked,
    13: plot_13_ffb_vs_redshift,
}

STANDALONE_PLOTS = {
    14: plot_14_density_evolution,
    142: plot_14c_density_evolution_mbk25,
    141: plot_14b_density_evolution_methods,
    16: plot_16_sfrd_history,
    17: plot_17_smd_history,
    18: plot_18_smf_redshift_grid,
    181: plot_18b_smf_redshift_grid_wide,
    19: plot_19_smf_ffb_grid,
    192: plot_19c_smf_ffb_grid_mbk25,
    191: plot_19b_smf_ffb_methods_grid,
    20: plot_20_smf_lowz_grid,
    21: plot_21_smf_lowz_lowmass_grid,
    22: plot_22_regime_histogram,
    23: plot_23_ffb_histogram,
    231: plot_23b_ffb_histogram_bk25,
    232: plot_23c_ffb_fraction_bk25,
    25: plot_25_hi_mass_ratio,
    26: plot_26_h2_mass_ratio,
    27: plot_27_cold_gas_mass_ratio,
    28: plot_28_mdot_vs_mvir,
    29: plot_29_mdot_vs_vvir,
    32: plot_32_hi_mass_function,
    33: plot_33_h2_mass_function,
    34: plot_34_hi_mass_function_primary_uchuu,
    35: plot_35_h2_mass_function_primary_uchuu,
    36: plot_36_selection_thresholds_mz,
    37: plot_37_cgm_census,
    38: plot_38_hi_mass_function_recipes,
    39: plot_39_gas_mass_functions_stacked,
    40: plot_40_gas_mass_functions_stacked_recipes,
    99: plot_99_referee_diagnostics,
}

ALL_PLOTS = {**Z0_PLOTS, **EVOLUTION_PLOTS, **STANDALONE_PLOTS}


def main():
    seed(SEED)
    np.random.seed(SEED)
    setup_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Determine which plots to generate
    if len(sys.argv) > 1:
        plot_nums = [int(x) for x in sys.argv[1:]]
    else:
        plot_nums = sorted(ALL_PLOTS.keys())

    need_z0 = any(n in Z0_PLOTS for n in plot_nums)
    need_evo = any(n in EVOLUTION_PLOTS for n in plot_nums)

    primary = vanilla = snapdata = None

    # Load z=0 data only if needed
    if need_z0:
        print('Loading primary model from', PRIMARY_DIR)
        primary = load_model(PRIMARY_DIR)
        print(f'  {len(primary["StellarMass"]):,} galaxies loaded')

        print('Loading vanilla model from', VANILLA_DIR)
        vanilla = load_model(VANILLA_DIR,
                             properties=['StellarMass', 'SfrDisk', 'SfrBulge',
                                         'ColdGas', 'MetalsColdGas',
                                         'BlackHoleMass', 'BulgeMass',
                                         'Mvir', 'Vvir', 'CoolingRate', 'Regime', 'Type'])
        print(f'  {len(vanilla["StellarMass"]):,} galaxies loaded')
        print()

    # Load multi-snapshot data only if needed
    if need_evo:
        key_snaps = [SNAP_Z0, SNAP_Z1, SNAP_Z2, SNAP_Z3, SNAP_Z4, SNAP_Z5, SNAP_Z10]
        sfh_snaps = list(range(8, 64))
        all_snaps = sorted(set(key_snaps + sfh_snaps))

        print(f'Loading {len(all_snaps)} snapshots from', PRIMARY_DIR)
        snapdata = load_snapshots(PRIMARY_DIR, all_snaps)
        print(f'  {len(snapdata)} snapshots loaded')
        print()

    # Generate requested plots
    for num in plot_nums:
        if num in Z0_PLOTS:
            Z0_PLOTS[num](primary, vanilla)
        elif num in EVOLUTION_PLOTS:
            EVOLUTION_PLOTS[num](snapdata)
        elif num in STANDALONE_PLOTS:
            STANDALONE_PLOTS[num]()
        else:
            print(f'Warning: Plot {num} not defined, skipping.')
        print()

    print('Done.')


if __name__ == '__main__':
    main()
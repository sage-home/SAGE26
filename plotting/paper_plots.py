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
FFB_BK25_DIR = './output/millennium_ffb_mbk25/'
FFB_BK25_SMOOTH_DIR = './output/millennium_ffb_mbk25_smooth/'
FFB_NOSIGMOID_DIR = './output/millennium_nosigmoid/'
MINIUCHUU_DIR = './output/microuchuu/'
MODEL_FILE = 'model_0.hdf5'
OBS_DIR = './data/'

# Plotting (analysis choices — not simulation parameters)
OUTPUT_FORMAT = '.pdf'
DILUTE = 7500
SEED = 2222

# Analysis thresholds (not simulation parameters)
SSFR_CUT = -11.0       # log10(sSFR/yr^-1) dividing quiescent from star-forming

# Solar metallicity (Asplund et al. 2009)
Z_SUN = 0.0134

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
    'OutflowRate', 'MassLoading', 'Cooling', 'Regime',
]

# Properties to load for evolution (multi-snapshot) plots
_EVOLUTION_PROPERTIES = [
    'StellarMass', 'SfrDisk', 'SfrBulge', 'Mvir', 'Rvir',
    'CGMgas', 'HotGas', 'MetalsStellarMass', 'DiskRadius',
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
    chunks = {prop: [] for prop in properties}
    found_snap = False

    for fp in filepaths:
        try:
            with h5.File(fp, 'r') as f:
                if snap_key not in f:
                    continue
                found_snap = True
                grp = f[snap_key]
                for prop in properties:
                    if prop in grp:
                        chunks[prop].append(np.array(grp[prop]))
        except Exception as e:
            print(f"  Warning: could not read {fp}: {e}")
            continue

    if not found_snap:
        return {}

    data = {}
    for prop in properties:
        if chunks[prop]:
            arr = np.concatenate(chunks[prop])
            if prop in _MASS_PROPS:
                arr = arr * mass_convert
            data[prop] = arr

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
    boot_phi = np.zeros((n_boot, len(centers)))
    for b in range(n_boot):
        idx = np.random.randint(0, n_gal, n_gal)
        boot_masses = log_masses[idx]
        counts, _ = np.histogram(boot_masses, range=(mi, ma), bins=nbins)
        with np.errstate(divide='ignore'):
            boot_phi[b, :] = np.log10(counts / volume / binwidth)

    # Compute percentiles
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


def precipitation_fraction(tcool_over_tff):
    """Calculate precipitation fraction from the SAGE26 model."""
    threshold = 10.0
    width = 2.0

    x = np.atleast_1d(np.array(tcool_over_tff, dtype=float))
    f = np.zeros_like(x)

    # Unstable regime
    mask_unstable = x < threshold
    inst = np.minimum(threshold / x[mask_unstable], 3.0)
    f[mask_unstable] = np.tanh(inst / 2.0)

    # Transition regime
    mask_trans = (x >= threshold) & (x < threshold + width)
    xi = (x[mask_trans] - threshold) / width
    f[mask_trans] = 0.5 * (1.0 - np.tanh(xi))

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


# ========================== FIGURE UTILITIES ==========================

def save_figure(fig, filepath):
    """Save figure to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath)
    print(f'  Saved: {filepath}')
    plt.close(fig)


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
    path = os.path.join(OBS_DIR, 'gama_smf_morph.ecsv')
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
    path = os.path.join(OBS_DIR, 'baldry_blue_red.csv')
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
    path = os.path.join(OBS_DIR, 'Tremonti04.dat')
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
    path = os.path.join(OBS_DIR, 'Curti2020.dat')
    if os.path.exists(path):
        d = np.loadtxt(path)
        obs.append({
            'mass': d[:, 0], 'Z': d[:, 1],
            'yerr': [d[:, 1] - d[:, 2], d[:, 3] - d[:, 1]],
            'fmt': 's', 'color': 'k', 'label': 'Curti+20',
        })

    # Andrews & Martini 2013
    path = os.path.join(OBS_DIR, 'MMAdrews13.dat')
    if os.path.exists(path):
        d = np.loadtxt(path)
        obs.append({
            'mass': d[:, 0], 'Z': d[:, 1],
            'yerr': [d[:, 1] - d[:, 2], d[:, 3] - d[:, 1]],
            'fmt': '^', 'color': 'k',
            'label': _tex_safe(r'Andrews \& Martini 2013'),
        })

    # Kewley & Ellison 2008 - T04 calibration
    path = os.path.join(OBS_DIR, 'MMR-Kewley08.dat')
    if os.path.exists(path):
        d = np.loadtxt(path)
        obs.append({
            'mass': d[59:74, 0], 'Z': d[59:74, 1], 'yerr': None,
            'fmt': 'd', 'color': 'k',
            'label': _tex_safe(r'Kewley \& Ellison 2008'),
        })

    # Gallazzi et al. 2005 (stellar -> gas-phase conversion)
    path = os.path.join(OBS_DIR, 'MSZR-Gallazzi05.dat')
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
    path = os.path.join(OBS_DIR, 'Moster_2013.csv')
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
    for fname in ['Romeo20_SMHM.dat', 'Romeo20_SMHM_ETGs.dat']:
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
    for fname in ['ETGs_Kravtsov18.dat', 'LTGs_Kravtsov18.dat']:
        path = os.path.join(OBS_DIR, fname)
        if os.path.exists(path):
            d = np.loadtxt(path)
            k_mvir.append(d[:, 0])
            k_mstar.append(d[:, 1])
            k_xerr_lo.append(d[:, 0] - d[:, 2])
            k_xerr_hi.append(d[:, 3] - d[:, 0])
            k_has_xerr.append(np.ones(len(d), dtype=bool))
    path = os.path.join(OBS_DIR, 'SatKinsAndClusters_Kravtsov18.dat')
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
    # Format: (log_Mhalo, log_Mhalo_lo, log_Mhalo_hi,
    #          M*/Mhalo, M*/Mhalo_lo, M*/Mhalo_hi)
    path = os.path.join(OBS_DIR, 'Taylor20.dat')
    if os.path.exists(path):
        d = np.loadtxt(path)
        log_mvir = d[:, 0]
        log_mvir_lo = d[:, 1]
        log_mvir_hi = d[:, 2]
        ratio = d[:, 3]
        ratio_lo = d[:, 4]
        ratio_hi = d[:, 5]
        log_mstar = log_mvir + np.log10(ratio)
        log_mstar_lo = log_mvir_lo + np.log10(ratio_lo)
        log_mstar_hi = log_mvir_hi + np.log10(ratio_hi)
        obs['taylor'] = {
            'mvir': log_mvir,
            'mstar': log_mstar,
            'xerr': [log_mvir - log_mvir_lo, log_mvir_hi - log_mvir],
            'yerr': [log_mstar - log_mstar_lo, log_mstar_hi - log_mstar],
        }

    return obs

def load_madau_dickinson_2014_data():
    """Load Madau and Dickinson 2014 SFRD data."""
    if not HAS_ASTROPY:
        return None, None, None, None
    filename = './data/MandD_sfrd_2014.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None

    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z_min']
        re = table['log_psi']
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
    filename = './data/MandD_smd_2014.ecsv'
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return None, None, None, None

    try:
        table = Table.read(filename, format='ascii.ecsv')
        z = table['z_min']
        re = table['log_rho']
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
    filename = './data/kikuchihara_smd_2020.ecsv'
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
    filename = './data/papovich_smd_2023.ecsv'
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
    filename = './data/oesch_sfrd_2018.ecsv'
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
    filename = './data/mcleod_rhouv_2024.ecsv'
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
    filename = './data/harikane_density_2023.ecsv'
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
    filename = './data/Brinchmann04.dat'
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
    filename = './data/MBH_host_gals_Terrazas17.dat'
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
    filename = os.path.join(obsdir, 'GAMA/ProSpect_Claudia.txt') if obsdir else './data/ProSpect_Claudia.txt'
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
    filename = './data/Bell_z0pt0_blue.dat'
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
    filename = './data/Bell_z0pt0_red.dat'
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
    jones_path = os.path.join(OBS_DIR, 'HIMF_Jones18.dat')
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
    zwaan_path = os.path.join(OBS_DIR, 'HIMF_Zwaan2005.dat')
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
        zorder_fill=2, zorder_line=3,
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
            zorder_fill=4, zorder_line=5,
        )

    # --- Observational data ---
    for obs in load_mzr_observations():
        if obs['yerr'] is not None:
            ax.errorbar(obs['mass'], obs['Z'], yerr=obs['yerr'],
                        fmt=obs['fmt'], color=obs['color'],
                        markeredgecolor='k', markeredgewidth=1.0, linewidth=1.0,
                        markerfacecolor = 'gray', ms=8,
                        label=obs['label'], alpha=0.6, zorder=9)
        else:
            ax.plot(obs['mass'], obs['Z'], obs['fmt'],
                    markeredgecolor='k', markeredgewidth=1.0, linewidth=1.0,
                    markerfacecolor = 'gray', ms=8,
                    color=obs['color'], label=obs['label'], alpha=0.6, zorder=9)

    ax.set_xlim(8.0, 12.0)
    ax.set_ylim(8.0, 10.0)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.set_xlabel(r'$\log_{10}\ m_{\mathrm{*}}\ [M_{\odot}]$')
    ax.set_ylabel(r'$12\ +\ \log_{10}\ (\mathrm{O/H})$')

    handles, labels = ax.get_legend_handles_labels()
    sim_set = {'SAGE26', 'SAGE16'}
    sim_h = [h for h, l in zip(handles, labels) if l in sim_set]
    sim_l = [l for l in labels if l in sim_set]
    obs_h = [h for h, l in zip(handles, labels) if l not in sim_set]
    obs_l = [l for l in labels if l not in sim_set]
    leg1 = _standard_legend(ax, loc='upper right', handles=sim_h, labels=sim_l)
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
        zorder_fill=2, zorder_line=3,
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
            zorder_fill=4, zorder_line=5,
        )

    # --- Observational data ---
    obs = load_bh_bulge_observations()
    sersic = ~obs['core']

    ax.errorbar(obs['log_M_sph'][sersic], obs['log_M_BH'][sersic],
                yerr=[obs['yerr'][0][sersic], obs['yerr'][1][sersic]],
                xerr=[obs['xerr'][0][sersic], obs['xerr'][1][sersic]],
                color='k', ls='none', lw=1, marker='d', ms=8, alpha=0.6, zorder=3,
                markeredgecolor='k', markeredgewidth=0.8,
                        markerfacecolor = 'gray',
                label='S13 core')
    ax.errorbar(obs['log_M_sph'][obs['core']], obs['log_M_BH'][obs['core']],
                yerr=[obs['yerr'][0][obs['core']], obs['yerr'][1][obs['core']]],
                xerr=[obs['xerr'][0][obs['core']], obs['xerr'][1][obs['core']]],
                color='k', ls='none', lw=1, marker='o', ms=8,
                markeredgecolor='k', markeredgewidth=0.8, alpha=0.6, zorder=3,
                        markerfacecolor = 'gray',
                label=_tex_safe(r'S13 S\'{e}rsic'))

    ax.plot(obs['haring_rix_x'], obs['haring_rix_y'], 'k--',
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
        zorder_fill=2, zorder_line=3,
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
            zorder_fill=4, zorder_line=5,
        )

    # --- Observational data ---
    obs = load_shmr_observations()

    if 'moster' in obs:
        ax.plot(obs['moster']['mvir'], obs['moster']['mstar'],
                'k-', lw=2, label='Moster+13')

    if 'romeo' in obs:
        ax.scatter(obs['romeo']['mvir'], obs['romeo']['mstar'],
                   marker='o', s=50, c='gray', label='Romeo+20',
                   edgecolor='k', linewidth=0.8, alpha=0.6, zorder=8)

    if 'kravtsov' in obs:
        k = obs['kravtsov']
        xerr = [k['xerr_lo'], k['xerr_hi']]
        ax.errorbar(k['mvir'], k['mstar'], xerr=xerr,
                    fmt='s', color='k', ms=8, lw=1,
                    markeredgecolor='k', markeredgewidth=0.8,
                    markerfacecolor = 'gray', alpha=0.6, zorder=8,
                    label='Kravtsov+18')

    if 'taylor' in obs:
        t = obs['taylor']
        ax.errorbar(t['mvir'], t['mstar'],
                    xerr=t['xerr'], yerr=t['yerr'],
                    fmt='d', color='k', ms=8, lw=1,
                    markeredgecolor='k', markeredgewidth=0.8,
                    markerfacecolor = 'gray', alpha=0.6, zorder=9,
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


# ========================== PLOT 7: t_cool/t_ff DISTRIBUTION ==========================

def plot_7_tcool_tff_distribution(snapdata):
    """
    Violin plot of log10(t_cool/t_ff) for CGM-regime haloes
    at z=4.2, 2.1, 1.2, 0.
    """
    print('Plot 7: t_cool/t_ff distribution')

    snap_info = [
        (SNAP_Z4, f'z = {REDSHIFTS[SNAP_Z4]:.1f}'),
        (SNAP_Z3, f'z = {REDSHIFTS[SNAP_Z3]:.1f}'),
        (SNAP_Z2, f'z = {REDSHIFTS[SNAP_Z2]:.1f}'),
        (SNAP_Z1, f'z = {REDSHIFTS[SNAP_Z1]:.1f}'),
        (SNAP_Z0, f'z = {REDSHIFTS[SNAP_Z0]:.1f}'),
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


# ========================== PLOT 8: PRECIPITATION FRACTION MODEL ==========================

def plot_8_precipitation_fraction(snapdata):
    """
    Precipitation fraction vs t_cool/t_ff: theoretical model curve
    with galaxy scatter at z=0 and z=2.
    """
    print('Plot 8: Precipitation fraction model')

    fig, ax = plt.subplots()

    # Theoretical curve
    ratio_arr = np.logspace(np.log10(0.5), 2.5, 2000)
    f_precip = precipitation_fraction(ratio_arr)

    ax.plot(ratio_arr, f_precip, 'teal', lw=3,
            label='SAGE26 inflow model', zorder=5)

    ax.axvline(x=10, color='goldenrod', ls='--', lw=1.5, alpha=0.6,
               label=r'$t_{\rm cool}/t_{\rm ff} = 10$')
    ax.axvline(x=12, color='goldenrod', ls=':', lw=1.0, alpha=1.0)

    ax.axvspan(0.5, 10, alpha=0.06, color='red')
    ax.axvspan(10, 12, alpha=0.06, color='goldenrod')
    ax.axvspan(12, 300, alpha=0.06, color='steelblue')

    ax.text(3, 0.55, 'Thermally\nUnstable', fontsize=14, ha='center',
            va='center', color='firebrick', fontweight='bold')
    ax.text(11, 0.70, 'Transition', fontsize=11, ha='center',
            va='center', color='goldenrod', fontweight='bold', rotation=90)
    ax.text(50, 0.12, 'Thermally\nStable', fontsize=14, ha='center',
            va='center', color='steelblue', fontweight='bold')

    # Galaxy scatter at z=0 and z=2
    sc = None
    markers = ['x', 'D']
    for (snap, label), mark in zip(
            [(SNAP_Z0, 'z=0'), (SNAP_Z2, 'z=2')], markers):
        if snap not in snapdata:
            continue
        d = snapdata[snap]
        ratio = d['tcool_over_tff']
        w = np.where(
            (d['Regime'] == 0) &
            (ratio > 0) & np.isfinite(ratio) &
            (d['Type'] == 0) &
            (d['Mvir'] > 1e10) &
            (ratio < 200)
        )[0]
        if len(w) > 0:
            if len(w) > 500:
                w = np.random.choice(w, 500, replace=False)
            r_vals = ratio[w]
            f_vals = precipitation_fraction(r_vals)
            sc = ax.scatter(r_vals, f_vals, s=40, alpha=0.6,
                            c=np.log10(d['Mvir'][w]), cmap='plasma',
                            marker=mark, edgecolors='none', zorder=10,
                            label=f'Galaxies ({label})')

    if sc is not None:
        cbar = plt.colorbar(sc, ax=ax, pad=0.02, aspect=30)
        cbar.set_label(r'$\log_{10}(M_{\rm vir}/M_{\odot})$')

    ax.set_xscale('log')
    ax.set_xlim(0.5, 300)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r'$t_{\rm cool}/t_{\rm ff}$')
    ax.set_ylabel(r'$f_{\rm inflow}$')

    _standard_legend(ax, loc='upper right')
    fig.tight_layout()

    save_figure(fig, os.path.join(OUTPUT_DIR,
                'PrecipitationFraction' + OUTPUT_FORMAT))


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
        ax.plot(x, med, color='steelblue', lw=2, label='Non-FFB galaxies', zorder=3)
        ax.fill_between(x, lo, hi, color='steelblue', alpha=0.2, zorder=2)

    if log_mvir_ffb is not None:
        x, med, lo, hi = _median_percentile(log_mvir_ffb, eps_ffb)
        ax.plot(x, med, color='firebrick', lw=2, label='FFB galaxies', zorder=5)
        ax.fill_between(x, lo, hi, color='firebrick', alpha=0.2, zorder=4)

    M_ffb = ffb_threshold_mass_msun(z_snap)
    ax.axvline(x=np.log10(M_ffb), color='goldenrod', ls=':', lw=2,
               alpha=0.6,
               label=fr'$M_{{\rm vir, FFB}}$ = {M_ffb:.1e} $M_\odot$')

    ax.set_yscale('log')
    ax.set_xlabel(r'$\log_{10}(M_{\rm vir}/M_{\odot})$')
    ax.set_ylabel(r'$\varepsilon_{\mathrm{SFE}} \equiv m_*/(\,f_b \, M_{\rm vir})$')
    ax.set_ylim(1e-4, 2.0)

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
    print('Plot 11b: FFB property histograms at z~7')

    snap = _snap_for_z(REDSHIFTS, 7.0)
    if snap not in snapdata:
        print('  Snapshot not available. Skipping.')
        return

    d = snapdata[snap]

    # Determine minimum stellar mass resolution from SMHM relation
    # Use median stellar mass for halos around 10^11 M_sun as the resolution limit
    HALO_MASS_RESOLUTION = 1e11  # M_sun
    halo_bin_width = 0.2  # dex
    log_mvir = np.log10(d['Mvir'])
    log_halo_res = np.log10(HALO_MASS_RESOLUTION)
    in_halo_bin = (np.abs(log_mvir - log_halo_res) < halo_bin_width) & (d['StellarMass'] > 0)
    if np.sum(in_halo_bin) > 0:
        mstar_resolution = np.median(d['StellarMass'][in_halo_bin])
        print(f'  Median stellar mass at M_halo ~ 10^11 M_sun: {mstar_resolution:.2e} M_sun')
        print(f'  (log10 = {np.log10(mstar_resolution):.2f})')
    else:
        mstar_resolution = 0
        print('  Warning: No galaxies found near halo mass resolution limit')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Select FFB and non-FFB central galaxies above resolution limit
    w_ffb = np.where(
        (d['StellarMass'] > mstar_resolution) & (d['FFBRegime'] == 1) & (d['Type'] == 0)
    )[0]
    w_normal = np.where(
        (d['StellarMass'] > mstar_resolution) & (d['FFBRegime'] == 0) & (d['Type'] == 0)
    )[0]
    print(f'  FFB galaxies above resolution: {len(w_ffb)}')
    print(f'  Non-FFB galaxies above resolution: {len(w_normal)}')

    hist_kwargs_ffb = dict(bins=30, alpha=0.7, color='firebrick',
                           edgecolor='darkred', linewidth=1.2,
                           label='FFB galaxies', density=True)
    hist_kwargs_norm = dict(bins=30, alpha=0.5, color='steelblue',
                            edgecolor='navy', linewidth=1.2,
                            label='non-FFB galaxies', density=True)

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
            if not (x_min <= t_trans <= x_max):
                continue
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

    # Top N FFB galaxies by stellar mass
    N_track = min(10, len(w_ffb))
    mass_order = np.argsort(d['StellarMass'][w_ffb])[::-1]
    ffb_idx = w_ffb[mass_order[:N_track]]
    ffb_gal_ids = d['GalaxyIndex'][ffb_idx]

    # Build never-FFB pool for mass-matched comparison
    fig_g_snaps = [s for s in range(8, 64) if s in snapdata]
    ever_ffb_gids = set()
    for s in fig_g_snaps:
        sd = snapdata[s]
        w_ffb_snap = np.where(sd['FFBRegime'] == 1)[0]
        ever_ffb_gids.update(sd['GalaxyIndex'][w_ffb_snap].astype(int))

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

            color = 'firebrick' if r0 == 1 else 'steelblue'
            ls = '-' if r0 == 1 else '--'
            lbl = None
            if r0 == 1 and not ffb_regime_label:
                lbl = 'FFB regime'
                ffb_regime_label = True
            elif r0 == 0 and not nonffb_regime_label:
                lbl = 'Non-FFB regime'
                nonffb_regime_label = True

            ax.plot([t0, t1], [sfr0, sfr1], ls, color=color,
                    alpha=1.0, lw=2.2, label=lbl, zorder=2)

    ax.set_xlabel('Cosmic time [Gyr]')
    ax.set_ylabel(r'SFR [$M_{\odot}\,\mathrm{yr}^{-1}$]')

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


# ========================== PLOT 13: FFB FRACTION vs HALO MASS ==========================

def plot_13_ffb_vs_redshift(snapdata):
    """
    FFB fraction as a function of halo mass at different redshifts.

    Shows theoretical sigmoid curves (from the SAGE26 model) overlaid
    with binned simulation data with bootstrap error bars.
    """
    print('Plot 13: FFB fraction vs redshift')

    fig, ax = plt.subplots()

    # Halo mass range (log10 M_sun)
    log_Mvir = np.linspace(8, 14, 500)
    Mvir = 10.0**log_Mvir

    # Target redshifts
    redshift_targets = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
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

        # Simulation data
        if snap_idx not in snapdata:
            continue
        d = snapdata[snap_idx]

        central = d['Type'] == 0
        Mvir_data = d['Mvir'][central]
        ffb_data = d['FFBRegime'][central].astype(float)

        pos = Mvir_data > 0
        log_Mvir_data = np.log10(Mvir_data[pos])
        ffb_data = ffb_data[pos]

        # Bin by log Mvir and compute FFB fraction
        bin_edges = np.linspace(8, 14, 17)

        ffb_fracs = []
        ffb_errs = []
        mass_errs = []
        valid_bins = []

        for i in range(len(bin_edges) - 1):
            in_bin = ((log_Mvir_data >= bin_edges[i])
                      & (log_Mvir_data < bin_edges[i + 1]))
            n_in_bin = np.sum(in_bin)

            if n_in_bin < 10:
                continue

            bin_vals = ffb_data[in_bin]
            frac = np.mean(bin_vals)

            if frac == 0.0 or frac == 1.0:
                # Wilson score interval for edge cases
                zs = 1.0  # 1-sigma
                n = n_in_bin
                denom = 1 + zs**2 / n
                centre = (frac + zs**2 / (2 * n)) / denom
                margin = (zs * np.sqrt(
                    (frac * (1 - frac) + zs**2 / (4 * n)) / n) / denom)
                err_low = max(0, frac - (centre - margin))
                err_high = max(0, (centre + margin) - frac)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = stats.bootstrap(
                        (bin_vals,), np.mean,
                        n_resamples=1000,
                        confidence_level=0.6827,
                        method='percentile')
                err_low = max(0, frac - res.confidence_interval.low)
                err_high = max(0, res.confidence_interval.high - frac)

            ffb_fracs.append(frac)
            ffb_errs.append([err_low, err_high])

            bin_masses = log_Mvir_data[in_bin]
            mean_mass = np.mean(bin_masses)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mass_res = stats.bootstrap(
                    (bin_masses,), np.mean,
                    n_resamples=1000,
                    confidence_level=0.6827,
                    method='percentile')
            m_err_lo = max(0, mean_mass - mass_res.confidence_interval.low)
            m_err_hi = max(0, mass_res.confidence_interval.high - mean_mass)
            mass_errs.append([m_err_lo, m_err_hi])
            valid_bins.append(mean_mass)

        if len(valid_bins) > 0:
            ffb_errs = np.array(ffb_errs).T
            mass_errs = np.array(mass_errs).T
            ax.errorbar(valid_bins, ffb_fracs,
                        xerr=mass_errs, yerr=ffb_errs,
                        fmt='o', color=color, markersize=8, capsize=3,
                        alpha=0.6, markeredgecolor='k',
                        markeredgewidth=0.3)

    ax.axhline(0.5, color='gray', ls='--', alpha=1.0, lw=1)

    ax.set_xlabel(r'$\log_{10}\ M_{\mathrm{vir}}\ [M_{\odot}]$')
    ax.set_ylabel(r'$f_{\mathrm{FFB}}$')
    ax.set_xlim(8.25, 13)
    ax.set_ylim(0, 1)

    _standard_legend(ax, loc='upper left')
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
        data_FFB = load_model(PRIMARY_DIR, filename=MODEL_FILE,
                              snapshot=Snapshot, properties=props)

        # Load No FFB
        data_noFFB = load_model(NOFFB_DIR, filename=MODEL_FILE,
                                snapshot=Snapshot, properties=props)

        # Load FFB 100% (sfe=1.0)
        FFB100_DIR = './output/millennium_ffb100/'
        data_FFB100 = load_model(FFB100_DIR, filename=MODEL_FILE,
                                 snapshot=Snapshot, properties=props)

        # Load FFB BK25
        data_FFB_BK25 = load_model(FFB_BK25_SMOOTH_DIR, filename=MODEL_FILE,
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

    # --- miniUchuu (if available) ---
    mu_filepath = os.path.join(MINIUCHUU_DIR, MODEL_FILE)
    mu_z_list, mu_sfrd_list, mu_smd_list = [], [], []
    if os.path.exists(mu_filepath):
        mu_redshifts = np.array(MINIUCHUU_REDSHIFTS)
        mu_volume = MINIUCHUU_VOLUME
        for snap_idx in range(MINIUCHUU_FIRST_SNAP, MINIUCHUU_LAST_SNAP + 1):
            z = mu_redshifts[snap_idx]
            if not (4.8 <= z <= 20.0):
                continue
            snap_key = f'Snap_{snap_idx}'
            try:
                d = load_model(MINIUCHUU_DIR, filename=MODEL_FILE,
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
                    color='firebrick', linewidth=3.0, label='No FFB')
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
                    color='firebrick', linewidth=3.0, label='No FFB')
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

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
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

        data_FFB = load_model(PRIMARY_DIR, filename=MODEL_FILE,
                              snapshot=Snapshot, properties=props)
        data_noFFB = load_model(NOFFB_DIR, filename=MODEL_FILE,
                                snapshot=Snapshot, properties=props)
        FFB100_DIR = './output/millennium_ffb100/'
        data_FFB100 = load_model(FFB100_DIR, filename=MODEL_FILE,
                                 snapshot=Snapshot, properties=props)
        data_FFB_BK25 = load_model(FFB_BK25_SMOOTH_DIR, filename=MODEL_FILE,
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

    # --- miniUchuu (if available) ---
    mu_filepath = os.path.join(MINIUCHUU_DIR, MODEL_FILE)
    mu_z_list, mu_sfrd_list, mu_smd_list = [], [], []
    if os.path.exists(mu_filepath):
        mu_redshifts = np.array(MINIUCHUU_REDSHIFTS)
        mu_volume = MINIUCHUU_VOLUME
        for snap_idx in range(MINIUCHUU_FIRST_SNAP, MINIUCHUU_LAST_SNAP + 1):
            z = mu_redshifts[snap_idx]
            if not (4.8 <= z <= 20.0):
                continue
            snap_key = f'Snap_{snap_idx}'
            try:
                d = load_model(MINIUCHUU_DIR, filename=MODEL_FILE,
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
    valid_ffb_bk25 = ~np.isnan(sfrd_ffb_bk25_sorted)
    if np.sum(valid_noffb) > 1:
        axes[0].plot(z_sorted[valid_noffb], sfrd_noffb_sorted[valid_noffb], '-',
                    color='firebrick', linewidth=3.0, label='No FFB')
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
                    color='green', linewidth=3.0, label=r'$\alpha_{\rm FFB}=0.2$ (MBK25)')
        boot_valid = valid_ffb_bk25 & ~np.isnan(sfrd_ffb_bk25_lo_sorted) & ~np.isnan(sfrd_ffb_bk25_hi_sorted)
        if np.sum(boot_valid) > 1:
            axes[0].fill_between(z_sorted[boot_valid], sfrd_ffb_bk25_lo_sorted[boot_valid],
                                sfrd_ffb_bk25_hi_sorted[boot_valid], color='green', alpha=0.2)
    if np.sum(valid_ffb100) > 1:
        axes[0].plot(z_sorted[valid_ffb100], sfrd_ffb100_sorted[valid_ffb100], '-',
                    color='steelblue', linewidth=3.0, label=r'$\alpha_{\rm FFB}=1.0$ (Li+24)')
        boot_valid = valid_ffb100 & ~np.isnan(sfrd_ffb100_lo_sorted) & ~np.isnan(sfrd_ffb100_hi_sorted)
        if np.sum(boot_valid) > 1:
            axes[0].fill_between(z_sorted[boot_valid], sfrd_ffb100_lo_sorted[boot_valid],
                                sfrd_ffb100_hi_sorted[boot_valid], color='steelblue', alpha=0.2)
    if len(mu_z_list) > 1:
        valid_mu = ~np.isnan(mu_sfrd_sorted)
        if np.sum(valid_mu) > 1:
            axes[0].plot(mu_z_sorted[valid_mu], mu_sfrd_sorted[valid_mu], '--',
                        color='steelblue', linewidth=2.5, label='miniUchuu')

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
                    color='firebrick', linewidth=3.0, label='No FFB')
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
                    color='green', linewidth=3.0, label=r'$\alpha_{\rm FFB}=0.2$ (MBK25)')
        boot_valid = valid_smd_bk25 & ~np.isnan(smd_ffb_bk25_lo_sorted) & ~np.isnan(smd_ffb_bk25_hi_sorted)
        if np.sum(boot_valid) > 1:
            axes[1].fill_between(z_sorted[boot_valid], smd_ffb_bk25_lo_sorted[boot_valid],
                                smd_ffb_bk25_hi_sorted[boot_valid], color='green', alpha=0.2)
    if np.sum(valid_smd_ffb100) > 1:
        axes[1].plot(z_sorted[valid_smd_ffb100], smd_ffb100_sorted[valid_smd_ffb100], '-',
                    color='steelblue', linewidth=3.0, label=r'$\alpha_{\rm FFB}=1.0$ (Li+24)')
        boot_valid = valid_smd_ffb100 & ~np.isnan(smd_ffb100_lo_sorted) & ~np.isnan(smd_ffb100_hi_sorted)
        if np.sum(boot_valid) > 1:
            axes[1].fill_between(z_sorted[boot_valid], smd_ffb100_lo_sorted[boot_valid],
                                smd_ffb100_hi_sorted[boot_valid], color='steelblue', alpha=0.2)
    if len(mu_z_list) > 1:
        valid_mu_smd = ~np.isnan(mu_smd_sorted)
        if np.sum(valid_mu_smd) > 1:
            axes[1].plot(mu_z_sorted[valid_mu_smd], mu_smd_sorted[valid_mu_smd], '--',
                        color='steelblue', linewidth=2.5, label='SAGE26 (miniUchuu)')

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
        return 'SAGE26' in l or 'FFB' in l or 'miniUchuu' in l or 'epsilon' in l or 'MBK25' in l
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

    axes[1].set_xlabel(r'Redshift')
    axes[1].set_ylabel(r'$\log_{10} \rho_\star\ [M_\odot\,\mathrm{Mpc}^{-3}]$')
    axes[1].set_xlim(5, 16)
    axes[1].set_ylim(3, 8)
    axes[1].xaxis.set_major_locator(plt.MultipleLocator(2.0))
    axes[1].yaxis.set_major_locator(plt.MultipleLocator(1.0))
    axes[1].xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    axes[1].yaxis.set_minor_locator(plt.MultipleLocator(0.2))

    fig.tight_layout()

    output_file = os.path.join(output_dir, 'FFB_Density_Evolution_MBK25' + OUTPUT_FORMAT)
    save_figure(fig, output_file)

# ========================== PLOT 14b: FFB METHOD COMPARISON ==========================

def plot_14b_density_evolution_methods():
    """
    Create 2x1 figure with SFRD and SMD vs redshift comparing 4 FFB methods:
      - Li+24 with sigmoid (mode 1, PRIMARY_DIR)
      - BK25 with log-normal smoothing (mode 4, FFB_BK25_SMOOTH_DIR)
      - Li+24 no sigmoid (mode 5, FFB_NOSIGMOID_DIR)
      - BK25 no smoothing (mode 2, FFB_BK25_DIR)
    """
    print('Plot 14b: FFB method comparison (SFRD & SMD)')
    seed(SEED)

    output_dir = OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    volume = VOLUME

    target_snaps = []
    for snap_idx, z in enumerate(REDSHIFTS):
        if 4.8 <= z <= 20.0:
            target_snaps.append(f'Snap_{snap_idx}')

    print("Loading data for FFB method comparison plots...")

    # Define the 4 models
    model_keys = ['li_sigmoid', 'MBK25_smooth', 'li_nosig', 'MBK25_sharp']
    model_dirs = {
        'li_sigmoid':  PRIMARY_DIR,
        'MBK25_smooth': FFB_BK25_SMOOTH_DIR,
        'li_nosig':    FFB_NOSIGMOID_DIR,
        'MBK25_sharp':  FFB_BK25_DIR,
    }
    model_labels = {
        'li_sigmoid':  r'Li+24 (sigmoid)',
        'MBK25_smooth': r'MBK25 (log-normal $c$ scatter)',
        'li_nosig':    r'Li+24 (sharp cutoff)',
        'MBK25_sharp':  r'MBK25 (sharp cutoff)',
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
    model_lw = {
        'li_sigmoid':  3.5,
        'MBK25_smooth': 3.0,
        'li_nosig':    3.0,
        'MBK25_sharp':  3.0,
    }

    # Storage for each model
    sfrd = {k: [] for k in model_keys}
    sfrd_lo = {k: [] for k in model_keys}
    sfrd_hi = {k: [] for k in model_keys}
    smd = {k: [] for k in model_keys}
    smd_lo = {k: [] for k in model_keys}
    smd_hi = {k: [] for k in model_keys}
    redshifts_density = []

    N_BOOT = 100
    rng = np.random.default_rng(SEED)

    def bootstrap_density(values, n_boot=N_BOOT):
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
        any_loaded = False

        for key in model_keys:
            data = load_model(model_dirs[key], filename=MODEL_FILE,
                              snapshot=Snapshot, properties=props)
            if data:
                any_loaded = True
                sfr_vals = data['SfrDisk'] + data['SfrBulge']
                sm_vals = data['StellarMass']
                sfrd_med, sfrd_l, sfrd_h = bootstrap_density(sfr_vals)
                smd_med, smd_l, smd_h = bootstrap_density(sm_vals)
            else:
                sfrd_med, sfrd_l, sfrd_h = np.nan, np.nan, np.nan
                smd_med, smd_l, smd_h = np.nan, np.nan, np.nan
            sfrd[key].append(sfrd_med)
            sfrd_lo[key].append(sfrd_l)
            sfrd_hi[key].append(sfrd_h)
            smd[key].append(smd_med)
            smd_lo[key].append(smd_l)
            smd_hi[key].append(smd_h)

        if not any_loaded:
            continue
        redshifts_density.append(z)

    # Convert to arrays and sort by redshift
    redshifts_density = np.array(redshifts_density)
    sort_idx = np.argsort(redshifts_density)
    z_sorted = redshifts_density[sort_idx]

    sfrd_sorted = {k: np.array(sfrd[k])[sort_idx] for k in model_keys}
    sfrd_lo_sorted = {k: np.array(sfrd_lo[k])[sort_idx] for k in model_keys}
    sfrd_hi_sorted = {k: np.array(sfrd_hi[k])[sort_idx] for k in model_keys}
    smd_sorted = {k: np.array(smd[k])[sort_idx] for k in model_keys}
    smd_lo_sorted = {k: np.array(smd_lo[k])[sort_idx] for k in model_keys}
    smd_hi_sorted = {k: np.array(smd_hi[k])[sort_idx] for k in model_keys}

    print("Generating FFB method comparison plots...")

    # ----- Top Panel: SFRD vs Redshift -----
    for key in model_keys:
        valid = ~np.isnan(sfrd_sorted[key])
        if np.sum(valid) > 1:
            axes[0].plot(z_sorted[valid], sfrd_sorted[key][valid],
                        model_ls[key], color=model_colors[key],
                        linewidth=model_lw[key], label=model_labels[key])

    # SFRD observational data
    z_madau, re_madau, re_err_plus_madau, re_err_minus_madau = load_madau_dickinson_2014_data()
    if z_madau is not None:
        mask = (z_madau >= 5) & (z_madau <= 16)
        if np.sum(mask) > 0:
            axes[0].errorbar(z_madau[mask], re_madau[mask],
                            yerr=[re_err_minus_madau[mask], re_err_plus_madau[mask]],
                            fmt='o', color='black', markersize=8, alpha=0.6,
                            markeredgecolor='k', markeredgewidth=0.8,
                            markerfacecolor='gray',
                            label=_tex_safe(r'Madau \& Dickinson 14'), linewidth=1.0, zorder=5)

    z_oesch, re_oesch, re_err_plus_oesch, re_err_minus_oesch = load_oesch_sfrd_2018_data()
    if z_oesch is not None:
        mask = (z_oesch >= 5) & (z_oesch <= 16)
        if np.sum(mask) > 0:
            axes[0].errorbar(z_oesch[mask], re_oesch[mask],
                            yerr=[re_err_minus_oesch[mask], re_err_plus_oesch[mask]],
                            fmt='*', color='black', markersize=8, alpha=0.6,
                            markeredgecolor='k', markeredgewidth=0.8,
                            markerfacecolor='gray',
                            label='Oesch+18', linewidth=1.0, zorder=5)

    z_mcleod, re_mcleod, re_err_plus_mcleod, re_err_minus_mcleod = load_mcleod_rho_sfr_2024_data()
    if z_mcleod is not None:
        mask = (z_mcleod >= 5) & (z_mcleod <= 16)
        if np.sum(mask) > 0:
            axes[0].errorbar(z_mcleod[mask], re_mcleod[mask],
                            yerr=[re_err_minus_mcleod[mask], re_err_plus_mcleod[mask]],
                            fmt='v', color='black', markersize=8, alpha=0.6,
                            markeredgecolor='k', markeredgewidth=0.8,
                            markerfacecolor='gray',
                            label='McLeod+24', linewidth=1.0, zorder=5)

    z_harikane, re_harikane, re_err_plus_harikane, re_err_minus_harikane = load_harikane_sfr_density_2023_data()
    if z_harikane is not None:
        mask = (z_harikane >= 5) & (z_harikane <= 16)
        if np.sum(mask) > 0:
            axes[0].errorbar(z_harikane[mask], re_harikane[mask],
                            yerr=[re_err_minus_harikane[mask], re_err_plus_harikane[mask]],
                            fmt='D', color='black', markersize=8, alpha=0.6,
                            markeredgecolor='k', markeredgewidth=0.8,
                            markerfacecolor='gray',
                            label='Harikane+23', linewidth=1.0, zorder=5)

    # ----- Bottom Panel: SMD vs Redshift -----
    for key in model_keys:
        valid = ~np.isnan(smd_sorted[key])
        if np.sum(valid) > 1:
            axes[1].plot(z_sorted[valid], smd_sorted[key][valid],
                        model_ls[key], color=model_colors[key],
                        linewidth=model_lw[key], label=model_labels[key])

    # SMD observational data
    z_madau_smd, re_madau_smd, re_err_plus_madau_smd, re_err_minus_madau_smd = load_madau_dickinson_smd_2014_data()
    if z_madau_smd is not None:
        mask = (z_madau_smd >= 5) & (z_madau_smd <= 16)
        if np.sum(mask) > 0:
            axes[1].errorbar(z_madau_smd[mask], re_madau_smd[mask],
                            yerr=[re_err_minus_madau_smd[mask], re_err_plus_madau_smd[mask]],
                            fmt='o', color='black', markersize=8, alpha=0.6,
                            markeredgecolor='k', markeredgewidth=0.8,
                            markerfacecolor='gray',
                            label=_tex_safe(r'Madau \& Dickinson 14'), linewidth=1.0, zorder=5)

    z_kiku, re_kiku, re_err_plus_kiku, re_err_minus_kiku = load_kikuchihara_smd_2020_data()
    if z_kiku is not None:
        mask = (z_kiku >= 5) & (z_kiku <= 16)
        if np.sum(mask) > 0:
            axes[1].errorbar(z_kiku[mask], re_kiku[mask],
                            yerr=[re_err_minus_kiku[mask], re_err_plus_kiku[mask]],
                            fmt='d', color='black', markersize=8, alpha=0.6,
                            markeredgecolor='k', markeredgewidth=0.8,
                            markerfacecolor='gray',
                            label='Kikuchihara+20', linewidth=1.0, zorder=5)

    z_papovich, re_papovich, re_err_plus_papovich, re_err_minus_papovich = load_papovich_smd_2023_data()
    if z_papovich is not None:
        mask = (z_papovich >= 5) & (z_papovich <= 16)
        if np.sum(mask) > 0:
            axes[1].errorbar(z_papovich[mask], re_papovich[mask],
                            yerr=[re_err_minus_papovich[mask], re_err_plus_papovich[mask]],
                            fmt='s', color='black', markersize=8, alpha=0.6,
                            markeredgecolor='k', markeredgewidth=0.8,
                            markerfacecolor='gray',
                            label='Papovich+23', linewidth=1.0, zorder=5)

    # Configure axes and split legends
    def is_sim_label(l):
        return any(x in l for x in ['Li+24', 'MBK25', 'epsilon'])
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

    output_file = os.path.join(output_dir, 'FFB_Density_Evolution_Methods' + OUTPUT_FORMAT)
    save_figure(fig, output_file)

# ========================== PLOT 15: sSFR vs STELLAR MASS (DENSITY) ==========================

def plot_15_sfr_vs_stellar_mass(primary, vanilla):
    """
    Star formation rate vs. stellar mass distribution.

    Shows the distribution of galaxies in the SFR-mass plane
    as a KDE contour plot, with C16 as a scatter overlay.
    """
    print('Plot 15: SFR vs stellar mass')

    # --- Primary model ---
    sfr = primary['SfrDisk'] + primary['SfrBulge']
    w = (primary['StellarMass'] > 1e8) & (sfr > 0)
    log_mass = np.log10(primary['StellarMass'][w])
    log_sfr = np.log10(sfr[w])

    # --- Plot ---
    fig = plt.figure()
    ax = fig.add_subplot(111)

    mass_bins = np.arange(8.0, 12.0 + 0.1, 0.1)
    plot_binned_median_1sigma(
        ax, log_mass, log_sfr, mass_bins,
        color='steelblue', label='SAGE26',
        alpha=0.25, lw=3.5, min_count=50,
        zorder_fill=2, zorder_line=3,
    )

    # --- C16 (Vanilla) model ---
    sfr_v = vanilla['SfrDisk'] + vanilla['SfrBulge']
    w_v = (vanilla['StellarMass'] > 1e8) & (sfr_v > 0)
    if np.any(w_v):
        log_mass_v = np.log10(vanilla['StellarMass'][w_v])
        log_sfr_v = np.log10(sfr_v[w_v])
        plot_binned_median_1sigma(
            ax, log_mass_v, log_sfr_v, mass_bins,
            color='purple', label='SAGE16', ls='--',
            alpha=0.20, lw=3.0, min_count=50,
            zorder_fill=4, zorder_line=5,
        )
        
    # --- Load Brinchmann et al. (2004) data ---
    bz04_mass, bz04_sfr = load_brinchmann_sfr_mass_2004_data()
    if bz04_mass is not None and bz04_sfr is not None:
        ax.scatter(bz04_mass, bz04_sfr, marker='d', alpha=0.6, zorder=10,
                facecolors='gray', edgecolors='black', s=50,
                label='Brinchmann+04')
        
    # --- Load Terrazas et al. (2017) data ---
    ter_mass, ter_sfr = load_terrazas17_mbh_host_sfr_data()
    if ter_mass is not None and ter_sfr is not None:
        # Plot with error bars
        ax.errorbar(ter_mass, ter_sfr, xerr=0.2, yerr=0.3, fmt='o', ecolor='black', alpha=0.6, zorder=10,
                   mfc='gray', mec='black', ms=8, mew=1.0, elinewidth=1.0, label='Terrazas+17')
        
     # --- Load and plot GAMA ProSpect Claudia data ---
        log_ms, log_sfr = load_gama_prospect_claudia()
        if log_ms is not None and log_sfr is not None:
            # Plot density contour
            # X, Y, Z = density_contour(log_ms, log_sfr, bins=[25, 25])
            # if Z.max() > 0:
            #     levels = sigma_contour_levels(Z)
            #     if levels is not None:
            #         ax.contourf(X, Y, Z, levels=levels, cmap='Greys', alpha=0.3)
            #         ax.contour(X, Y, Z, levels=levels, colors='black', linestyles='-', alpha=0.5, linewidths=1.0)
            # Plot binned medians/errors
            bins = np.linspace(8, 12, 13)
            centers, med, p25, p75 = binned_median(log_ms, log_sfr, bins)
            valid = ~np.isnan(med)
            ax.errorbar(centers[valid], med[valid], yerr=[med[valid] - p25[valid], p75[valid] - med[valid]],
                        fmt='s', color='black', label='Bellstedt+20', markersize=8, alpha=0.6, zorder=10,
                        markeredgewidth=0.8, markerfacecolor='gray',
                        markeredgecolor='black')
        

    ax.set_xlim(8.0, 12.0)
    ax.set_ylim(-4.0, 2.0)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.set_xlabel(r'$\log_{10}\ m_{\mathrm{*}}\ [M_{\odot}]$')
    ax.set_ylabel(r'$\log_{10}\ \mathrm{SFR}\ [M_{\odot}\,\mathrm{yr}^{-1}]$')

    handles, labels = ax.get_legend_handles_labels()
    sim_set = {'SAGE26', 'SAGE16'}
    sim_h = [h for h, l in zip(handles, labels) if l in sim_set]
    sim_l = [l for l in labels if l in sim_set]
    obs_h = [h for h, l in zip(handles, labels) if l not in sim_set]
    obs_l = [l for l in labels if l not in sim_set]
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
            'color': 'black', 'ls': '-', 'lw': 3.5,
            'redshifts': redshifts, 'first_snap': FirstSnap, 'last_snap': LastSnap,
            'volume': VOLUME,
        })

    # 2. Vanilla Model (C16)
    if os.path.exists(VANILLA_DIR):
        sim_dirs.append({
            'path': VANILLA_DIR, 'label': 'SAGE16',
            'color': 'firebrick', 'ls': '--', 'lw': 1.5,
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
                color='purple', lw=1.0, alpha=0.4, marker='o', ls='none', 
                label='Observations')

    # Madau & Dickinson 2014 Fit
    def MD14_sfrd(z):
        psi = 0.015 * (1+z)**2.7 / (1 + ((1+z)/2.9)**5.6)
        return psi

    f_chab_to_salp = 1/0.63
    z_values = np.linspace(0, 8, 200)
    md14 = np.log10(MD14_sfrd(z_values) * f_chab_to_salp)
    ax.plot(z_values, md14, color='gray', lw=1.5, alpha=0.6, label=_tex_safe(r'Madau \& Dickinson 2014'))

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
        do_bootstrap = (sim_label == 'SAGE26 (Millennium)')
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
                    lw=sim['lw'], color=sim['color'], linestyle=sim['ls'], label=sim_label)

            if do_bootstrap:
                valid = nonzero[sfr_density_lo[nonzero] > 0]
                if len(valid) > 0:
                    ax.fill_between(z_vals[valid],
                                    np.log10(sfr_density_lo[valid]),
                                    np.log10(sfr_density_hi[valid]),
                                    color=sim['color'], alpha=0.2)

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
        csfrd_file = './data/CSFRD_inferred_from_SMD.ecsv'
        if os.path.exists(csfrd_file):
            try:
                csfrd_table = Table.read(csfrd_file, format='ascii.ecsv')
                z_csfrd = np.array(csfrd_table['Redshift'])
                sfrd_50 = np.log10(np.array(csfrd_table['sfrd_50']))
                sfrd_16 = np.log10(np.array(csfrd_table['sfrd_16']))
                sfrd_84 = np.log10(np.array(csfrd_table['sfrd_84']))
                ax.plot(z_csfrd, sfrd_50, color='darkorange', lw=2,
                        label='COSMOS-Web')
                ax.fill_between(z_csfrd, sfrd_16, sfrd_84,
                                color='orange', alpha=0.3)
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

    sim_names = {'SAGE26 (Millennium)', 'SAGE26 (miniUchuu)', 'SAGE16'}
    handles, labels = ax.get_legend_handles_labels()
    sim_h = [h for h, l in zip(handles, labels) if l in sim_names]
    sim_l = [l for l in labels if l in sim_names]
    obs_h = [h for h, l in zip(handles, labels) if l not in sim_names]
    obs_l = [l for l in labels if l not in sim_names]
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

    # --- COSMOS-Web ---
    if HAS_ASTROPY:
        smd_file = './data/SMD.ecsv'
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
    # 2. Thorne+21  (smfvals CSV: logM, phi, phi_16, phi_84 — linear)
    # ------------------------------------------------------------------
    # _thorne = [
    #     ('./data/Thorne21/SMFvals_z2.csv', 2.0),
    #     ('./data/Thorne21/SMFvals_z2.4.csv', 2.4),
    #     ('./data/Thorne21/SMFvals_z3.csv', 3.0),
    #     ('./data/Thorne21/SMFvals_z3.5.csv', 3.5),
    #     ('./data/Thorne21/SMFvals_z4.csv', 4.0),
    # ]
    # for fpath, z_val in _thorne:
    #     try:
    #         if not os.path.exists(fpath):
    #             continue
    #         for delim in [',', '\t', ' ']:
    #             for skip in [0, 1, 2]:
    #                 try:
    #                     d = np.genfromtxt(fpath, delimiter=delim, skip_header=skip)
    #                     if d.ndim == 2 and d.shape[1] >= 4:
    #                         break
    #                 except Exception:
    #                     d = None
    #             if d is not None and d.ndim == 2:
    #                 break
    #         if d is None or d.ndim != 2 or d.shape[1] < 4:
    #             continue
    #         m, phi, p16, p84 = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    #         ok = np.isfinite(m) & (phi > 0) & (p16 > 0) & (p84 > 0)
    #         if np.any(ok):
    #             lp = np.log10(phi[ok])
    #             obs.append({'z': z_val, 'log_mass': m[ok], 'log_phi': lp,
    #                          'err_lo': lp - np.log10(p16[ok]),
    #                          'err_hi': np.log10(p84[ok]) - lp,
    #                          'label': 'Thorne+21', 'marker': 's', 'ms': 6})
    #     except Exception as e:
    #         print(f"  Thorne+21 z={z_val} load error: {e}")

    # ------------------------------------------------------------------
    # 3. Weaver+23  (farmer TXT: logM, bw, phi, phi_lo, phi_hi — linear)
    # ------------------------------------------------------------------
    _weaver = [
        ('./data/COSMOS2020/SMF_Farmer_v2.1_1.5z2.0_total.txt', 1.75),
        ('./data/COSMOS2020/SMF_Farmer_v2.1_2.0z2.5_total.txt', 2.25),
        ('./data/COSMOS2020/SMF_Farmer_v2.1_2.5z3.0_total.txt', 2.75),
        ('./data/COSMOS2020/SMF_Farmer_v2.1_3.0z3.5_total.txt', 3.25),
        ('./data/COSMOS2020/SMF_Farmer_v2.1_3.5z4.5_total.txt', 4.0),
        ('./data/COSMOS2020/SMF_Farmer_v2.1_4.5z5.5_total.txt', 5.0),
        ('./data/COSMOS2020/SMF_Farmer_v2.1_5.5z6.5_total.txt', 6.0),
        ('./data/COSMOS2020/SMF_Farmer_v2.1_6.5z7.5_total.txt', 7.0),
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
        _muz_file = './data/SMF_Muzzin2013.dat'
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
        _san_file = './data/SMF_Santini2012.dat'
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
                    bins[key]['m'].append(lg_m - 0.24)  # Salpeter→Chabrier
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
        _wr_file = './data/Wright18_CombinedSMF.dat'
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
            _obs_file = './data/SMF_data_points.ecsv'
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
    # 8. Harvey+24  (ECSV: z, log10Mstar, phi, phi_error_low, phi_error_upp)
    # ------------------------------------------------------------------
    if HAS_ASTROPY:
        try:
            _har_file = './data/FiducialBagpipesGSMF.ecsv'
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
                                     'label': 'Harvey+24', 'marker': 'H', 'ms': 8})
        except Exception as e:
            print(f"  Harvey+24 load error: {e}")

    # ------------------------------------------------------------------
    # 9–12.  High-z ECSV datasets (Stefanon+21, Navarro-Carrera+23,
    #         Weibel+24, Kikuchihara+20)
    # ------------------------------------------------------------------
    _highz_ecsv = [
        {'file': './data/stefanon_smf_2021.ecsv',
         'label': 'Stefanon+21', 'marker': '*', 'ms': 8,
         'zcol': 'redshift_bin', 'mcol': 'log_M',
         'phi_col': 'phi', 'phi_eu': 'phi_err_up', 'phi_el': 'phi_err_low',
         'phi_scale': 1e-4, 'phi_log': False, 'zbins': [6, 7, 8, 9, 10]},
        {'file': './data/navarro_carrera_smf_2023.ecsv',
         'label': 'Navarro-Carrera+23', 'marker': 'X', 'ms': 8,
         'zcol': 'redshift_bin', 'mcol': 'log_M',
         'phi_col': 'phi', 'phi_eu': 'phi_err_up', 'phi_el': 'phi_err_low',
         'phi_scale': 1e-4, 'phi_log': False, 'zbins': [6, 7, 8]},
        {'file': './data/weibel_smf_2024.ecsv',
         'label': 'Weibel+24', 'marker': 'P', 'ms': 8,
         'zcol': 'redshift_bin', 'mcol': 'log_M',
         'phi_col': 'log_phi', 'phi_eu': 'log_phi_err_up', 'phi_el': 'log_phi_err_low',
         'phi_scale': 1.0, 'phi_log': True, 'zbins': [6, 7, 8, 9]},
        {'file': './data/kikuchihara_smf_2020.ecsv',
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

    for i, (z_lo, z_hi) in enumerate(z_bins):
        ax = axes_flat[i]
        z_mid = 0.5 * (z_lo + z_hi)

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

    # Load observational data
    all_obs = _load_smf_grid_observations()
    labels_used = set()  # track legend entries to avoid duplicates

    nrows, ncols = 3, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(25, 15), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    binwidth = 0.1

    for i, (z_lo, z_hi) in enumerate(z_bins):
        ax = axes_flat[i]
        z_mid = 0.5 * (z_lo + z_hi)

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
    FFB100_DIR = './output/millennium_ffb100/'
    models = []
    if os.path.exists(NOFFB_DIR):
        models.append({
            'path': NOFFB_DIR, 'label': r'No FFB',
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
                # For FFB 100% model at z >= 7, apply resolution cut based on SMHM relation
                if model['label'] == r'$\alpha_{\rm FFB}=1.0$' and z_mid > 50.0:
                    data = load_model(model['path'], snapshot=snap_name,
                                      properties=['StellarMass', 'Mvir'])
                    m_stars = data['StellarMass']
                    mvir = data['Mvir']
                    # Calculate median stellar mass for halos around 10^11 M_sun
                    HALO_MASS_RESOLUTION = 1e11
                    halo_bin_width = 0.3  # dex
                    log_mvir = np.log10(mvir, where=mvir > 0, out=np.full_like(mvir, -np.inf))
                    log_halo_res = np.log10(HALO_MASS_RESOLUTION)
                    in_halo_bin = (np.abs(log_mvir - log_halo_res) < halo_bin_width) & (m_stars > 0)
                    if np.sum(in_halo_bin) > 0:
                        mstar_resolution = np.median(m_stars[in_halo_bin])
                        print(f'    {snap_name} (z={z_mid:.1f}): M* resolution = {mstar_resolution:.2e}')
                    else:
                        mstar_resolution = 0
                    w = m_stars > mstar_resolution
                else:
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
    Plot: 1x4 grid of Stellar Mass Functions at high-z bins.
    Same as plot_19 but with an additional MBK25 (smooth) line in green.
    """
    print('Plot 19c: SMF FFB Grid with MBK25')

    z_bins = [
        (5.0, 6.0), (6.0, 7.0),
        (7.0, 9.0), (9.0, 11.0),
    ]

    mill_redshifts = np.array(REDSHIFTS)

    FFB100_DIR = './output/millennium_ffb100/'
    models = []
    if os.path.exists(NOFFB_DIR):
        models.append({
            'path': NOFFB_DIR, 'label': r'No FFB',
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
            'color': 'green', 'ls': '-', 'lw': 3.0,
            'redshifts': mill_redshifts, 'first_snap': 0, 'last_snap': 63,
            'volume': VOLUME, 'mass_convert': MASS_CONVERT,
        })
    if os.path.exists(FFB100_DIR):
        models.append({
            'path': FFB100_DIR, 'label': r'$\alpha_{\rm FFB}=1.0$ (Li+24)',
            'color': 'steelblue', 'ls': '-', 'lw': 3.0,
            'redshifts': mill_redshifts, 'first_snap': 0, 'last_snap': 63,
            'volume': VOLUME, 'mass_convert': MASS_CONVERT,
        })

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
    Plot: 1x4 grid of Stellar Mass Functions at high-z bins comparing 4 FFB methods:
      - Li+24 with sigmoid (mode 1, PRIMARY_DIR)
      - BK25 with log-normal smoothing (mode 4, FFB_BK25_SMOOTH_DIR)
      - Li+24 no sigmoid (mode 5, FFB_NOSIGMOID_DIR)
      - BK25 no smoothing (mode 2, FFB_BK25_DIR)
    """
    print('Plot 19b: SMF FFB Methods Grid')

    z_bins = [
        (5.0, 6.0), (6.0, 7.0),
        (7.0, 9.0), (9.0, 11.0),
    ]

    mill_redshifts = np.array(REDSHIFTS)

    models = []
    model_defs = [
        (PRIMARY_DIR,         r'Li+24 (sigmoid)',                    'black',      '-',  3.5),
        (FFB_BK25_SMOOTH_DIR, r'MBK25 (log-normal $c$ scatter)',     'steelblue', '-',  3.0),
        (FFB_NOSIGMOID_DIR,   r'Li+24 (sharp cutoff)',              'firebrick',  '--', 3.0),
        (FFB_BK25_DIR,        r'MBK25 (sharp cutoff)',               'darkgreen',  '--', 3.0),
    ]
    for path, label, color, ls, lw in model_defs:
        if os.path.exists(path):
            models.append({
                'path': path, 'label': label,
                'color': color, 'ls': ls, 'lw': lw,
                'redshifts': mill_redshifts, 'first_snap': 0, 'last_snap': 63,
                'volume': VOLUME, 'mass_convert': MASS_CONVERT,
            })

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

        ax.text(0.95, 0.95, rf'${z_lo:.0f} < z < {z_hi:.0f}$',
                transform=ax.transAxes, ha='right', va='top')

    axes_flat[0].set_xlim(7, 12.3)
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
    Plot: Stacked bar chart of FFB vs Non-FFB Galaxies as a function of redshift,
    with Li+24 and BK25 FFB bars overplotted on the same axis.
    Non-FFB: Greys, FFB Li+24: RdPu, FFB BK25: Blues (semi-transparent, overplotted).
    """
    print('Plot 23b: FFB Histogram - Li+24 vs BK25')

    models = [
        {'dir': PRIMARY_DIR,  'label': 'FFB (Li+24)',  'cmap': 'RdPu',  'alpha': 0.6},
        {'dir': FFB_BK25_SMOOTH_DIR, 'label': 'FFB (MBK25)',   'cmap': 'Blues',  'alpha': 0.6},
    ]

    fig = plt.figure()
    ax = plt.subplot(111)

    # We'll use the first model's non-FFB counts as background (they should be similar)
    non_ffb_drawn = False

    for model in models:
        model_files = find_model_files(model['dir'])
        if not model_files:
            print(f"  No model files found in {model['dir']}")
            continue

        num_non_ffb_per_snap = []
        num_ffb_per_snap = []
        redshifts_list = []

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

        log1pz_edges = np.log10(1 + z_edges)
        widths = log1pz_edges[:-1] - log1pz_edges[1:]

        log1pz_values = np.log10(1 + z_edges[:-1])
        norm = plt.Normalize(vmin=np.min(log1pz_values), vmax=np.max(log1pz_values))

        # Draw non-FFB background only once
        if not non_ffb_drawn:
            cmap_non_ffb = plt.get_cmap('Greys')
            colors_non_ffb = cmap_non_ffb(norm(log1pz_values))
            ax.bar(log1pz_edges[:-1], num_non_ffb_filtered, width=widths, align='edge',
                   label='Non-FFB', edgecolor='black', color=colors_non_ffb)
            non_ffb_drawn = True

        # Draw FFB bars
        cmap_ffb = plt.get_cmap(model['cmap'])
        colors_ffb = cmap_ffb(norm(log1pz_values))
        # Set alpha on colors
        colors_ffb[:, 3] = model['alpha']
        ax.bar(log1pz_edges[:-1], num_ffb_filtered, width=widths, align='edge',
               label=model['label'], edgecolor='black', color=colors_ffb)

    ax.set_yscale('log')
    ax.set_ylabel('Number of Galaxies')
    ax.set_xlabel(r'$\log_{10}(1+z)$')
    ax.set_xlim(np.log10(1+15), 0)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=3)

    # Add top x-axis for redshift
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    z_ticks = [0, 1, 2, 3, 5, 7, 10, 15]
    log1pz_ticks = [np.log10(1 + zt) for zt in z_ticks]
    ax2.set_xticks(log1pz_ticks)
    ax2.set_xticklabels([str(zt) for zt in z_ticks])
    ax2.set_xlabel(r'$z$')

    fig.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    outputFile = os.path.join(OUTPUT_DIR, 'FFB_Histogram_Li24_vs_BK25' + OUTPUT_FORMAT)
    plt.savefig(outputFile)
    print(f'Saved file to {outputFile}\n')
    plt.close()


# ========================== PLOT 24: MASS LOADING VS VELOCITY  ==========================

def plot_24_mass_loading_vs_velocity(primary, vanilla):
    """
    Plot: Mass Loading Factor vs Wind Velocity for different feedback models.
    """
    print('Plot 24: Mass Loading Factor vs Wind Velocity')

    # --- Primary model ---
    w = (primary['MassLoading'] > 0) & (primary['Vvir'] > 0)
    vvir = primary['Vvir'][w]
    mass_loading = primary['MassLoading'][w]

    # --- Plot ---
    fig = plt.figure()
    ax = fig.add_subplot(111)

    vvir_bins = np.linspace(0.0, 500.0, 51)
    plot_binned_median_1sigma(
        ax, vvir, mass_loading, vvir_bins,
        color='steelblue', label='SAGE26',
        alpha=0.25, lw=3.5, min_count=50,
        zorder_fill=2, zorder_line=3,
    )

    vvir_theory = np.logspace(1, 3, 100)  # 10 to 1000 km/s
    mass_loading_theory = calculate_muratov_mass_loading(vvir_theory, z=0.0)
    ax.plot(vvir_theory, mass_loading_theory, color='k', lw=2.5, ls='--',
            label='Muratov+16 Theory')

    chisholm_ml = pd.read_csv('./data/Chisholm_17_ml.csv', header=None, delimiter='\t')
    chisholm_x = chisholm_ml[0]  # First column
    chisholm_y = chisholm_ml[1]  # Second column

    heckman_ml = pd.read_csv('./data/Heckman_15_ml.csv', header=None, delimiter='\t')
    heckman_x = heckman_ml[0]  # First column
    heckman_y = heckman_ml[1]  # Second column

    rupke_ml = pd.read_csv('./data/Rupke_05_ml.csv', header=None, delimiter='\t')
    rupke_x = rupke_ml[0]  # First column
    rupke_y = rupke_ml[1]  # Second column

    sugahara_ml = pd.read_csv('./data/Sugahara_17_ml.csv', header=None, delimiter='\t')
    sugahara_x = sugahara_ml[0]  # First column
    sugahara_y = sugahara_ml[1]  # Second column 

    ax.scatter(chisholm_x, chisholm_y, color='k', marker='o', s=50, label='Chisholm+17', edgecolors='k', linewidths=1.0, facecolors='gray', alpha=0.6)
    ax.scatter(heckman_x, heckman_y, color='k', marker='x', s=50, label='Heckman+15', edgecolors='k', linewidths=1.0, facecolors='gray', alpha=0.6)
    ax.scatter(rupke_x, rupke_y, color='k', marker='s', s=50, label='Rupke+05', edgecolors='k', linewidths=1.0, facecolors='gray', alpha=0.6)
    ax.scatter(sugahara_x, sugahara_y, color='k', marker='d', s=50, label='Sugahara+17', edgecolors='k', linewidths=1.0, facecolors='gray', alpha=0.6)
            
    ax.set_xlim(0, 500)
    # ax.set_xscale('log')
    ax.set_ylim(0, 15.0)
    # ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    # ax.yaxis.set_major_locator(plt.MultipleLocator(5.0))
    ax.set_xlabel(r'$V_{\mathrm{vir}}\ [\mathrm{km/s}]$')
    ax.set_ylabel(r'$\eta_{\mathrm{reheat}}$')

    handles, labels = ax.get_legend_handles_labels()
    sim_set = {'SAGE26', 'Muratov+16 Theory'}
    sim_h = [h for h, l in zip(handles, labels) if l in sim_set]
    sim_l = [l for l in labels if l in sim_set]
    obs_h = [h for h, l in zip(handles, labels) if l not in sim_set]
    obs_l = [l for l in labels if l not in sim_set]
    leg1 = _standard_legend(ax, loc='upper right', handles=sim_h, labels=sim_l)
    ax.add_artist(leg1)
    _standard_legend(ax, loc='center right', handles=obs_h, labels=obs_l)

    ax.xaxis.set_major_locator(plt.MultipleLocator(100.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(2.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(20))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1.0))

    fig.tight_layout()
    outputFile = os.path.join(OUTPUT_DIR, 'MassLoading_vs_Velocity' + OUTPUT_FORMAT)
    save_figure(fig, outputFile)
    print(f'Saved file to {outputFile}\n')

    plt.close()


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

_MDOT_SNAP_PANELS = [
    (SNAP_Z0, f'z = {REDSHIFTS[SNAP_Z0]:.1f}'),
    (SNAP_Z1, f'z = {REDSHIFTS[SNAP_Z1]:.1f}'),
    (SNAP_Z2, f'z = {REDSHIFTS[SNAP_Z2]:.1f}'),
    (SNAP_Z3, f'z = {REDSHIFTS[SNAP_Z3]:.1f}'),
    (SNAP_Z4, f'z = {REDSHIFTS[SNAP_Z4]:.1f}'),
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

    nrows = len(_MDOT_SNAP_PANELS)
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
            ax.legend(loc='lower right', frameon=False)

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
        good = np.isfinite(phi)
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
    path_b14 = os.path.join(OBS_DIR, 'GasMF', 'B14_MH2MF.dat')
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
    path_det = os.path.join(OBS_DIR, 'H2MF_Fletcher21_DetNonDet.dat')
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
    path_est = os.path.join(OBS_DIR, 'H2MF_Fletcher21_Estimated.dat')
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
        good = np.isfinite(phi)
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


# ========================== MAIN ==========================

# Registry of plot functions
# z=0 plots take (primary, vanilla); evolution plots take (snapdata)
Z0_PLOTS = {
    31: plot_1_stellar_mass_function_ssfr_s,
    30: plot_1_stellar_mass_function_ssfr_q,
    2: plot_2_baryon_fraction,
    3: plot_3_gas_metallicity_vs_stellar_mass,
    4: plot_4_bh_bulge_mass,
    5: plot_5_stellar_halo_mass,
    6: plot_6_bulge_mass_size,
    15: plot_15_sfr_vs_stellar_mass,
    24: plot_24_mass_loading_vs_velocity,
}

EVOLUTION_PLOTS = {
    7: plot_7_tcool_tff_distribution,
    8: plot_8_precipitation_fraction,
    9: plot_9_cgm_fractions_depletion,
    91: plot_9b_cgm_fractions_grid,
    92: plot_9c_depletion_grid,
    10: plot_10_sfe_ffb,
    11: plot_11_ffb_properties,
    111: plot_11b_ffb_histograms,
    12: plot_12_sfh_ffb,
    121: plot_12b_ffb_regime_history,
    122: plot_12c_ffb_regime_heatmap,
    123: plot_12d_sfh_ffb_transitions,
    13: plot_13_ffb_vs_redshift,
}

# Standalone plots (load their own data)
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
    25: plot_25_hi_mass_ratio,
    26: plot_26_h2_mass_ratio,
    27: plot_27_cold_gas_mass_ratio,
    28: plot_28_mdot_vs_mvir,
    29: plot_29_mdot_vs_vvir,
    32: plot_32_hi_mass_function,
    33: plot_33_h2_mass_function,
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
                                         'Mvir', 'Type'])
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

# print_mdot_stream_cool_stats()
# print_massive_galaxy_stats()

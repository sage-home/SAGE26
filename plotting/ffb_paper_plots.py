#!/usr/bin/env python
"""
ffb_paper_plots.py
==================
Publication-quality figures for the FFB paper.

Usage:
    python ffb_paper_plots.py         # Generate all plots
    python ffb_paper_plots.py A       # Plot A: FFB fraction vs redshift
    python ffb_paper_plots.py B       # Plot B: f_FFB(M_halo, z) heatmap
    python ffb_paper_plots.py A B     # Both
"""

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import os
import sys
import glob
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import norm as _snorm

# ========================== CONFIGURATION ==========================

LI24_DIR   = './output/millennium/'
MBK25_DIR  = './output/millennium_mbk/'
OUTPUT_DIR = './output/ffb_paper/plots/'
OUTPUT_FORMAT = '.pdf'

_MSUN_CGS = 1.989e33

_MASS_PROPS = frozenset({
    'CentralMvir', 'Mvir', 'StellarMass', 'BulgeMass', 'BlackHoleMass',
    'MetalsStellarMass', 'MetalsColdGas', 'MetalsEjectedMass',
    'MetalsHotGas', 'MetalsCGMgas', 'ColdGas', 'HotGas', 'CGMgas',
    'EjectedMass', 'H2gas', 'H1gas', 'IntraClusterStars',
    'MergerBulgeMass', 'InstabilityBulgeMass',
})

# ========================== SIMULATION HEADER ==========================

def _find_model_files(directory):
    files = sorted(glob.glob(os.path.join(directory, 'model_*.hdf5')))
    if not files:
        single = os.path.join(directory, 'model_0.hdf5')
        if os.path.exists(single):
            files = [single]
    return files


def _read_sim_header(directory):
    files = _find_model_files(directory)
    if not files:
        return None
    try:
        with h5.File(files[0], 'r') as f:
            header = {
                'hubble_h':       float(f['Header/Simulation'].attrs['hubble_h']),
                'omega_matter':   float(f['Header/Simulation'].attrs['omega_matter']),
                'omega_lambda':   float(f['Header/Simulation'].attrs['omega_lambda']),
                'unit_mass_in_g': float(f['Header/Runtime'].attrs['UnitMass_in_g']),
                'redshifts':      list(f['Header/snapshot_redshifts'][:]),
            }
        return header
    except Exception as e:
        print(f"Warning: could not read header from {directory}: {e}")
        return None


_hdr = _read_sim_header(LI24_DIR) or _read_sim_header(MBK25_DIR)
if _hdr:
    HUBBLE_H     = _hdr['hubble_h']
    OMEGA_M      = _hdr['omega_matter']
    OMEGA_L      = _hdr['omega_lambda']
    MASS_CONVERT = _hdr['unit_mass_in_g'] / _MSUN_CGS / HUBBLE_H
    REDSHIFTS    = _hdr['redshifts']
else:
    print("Warning: no model header found — using Millennium defaults")
    HUBBLE_H     = 0.73
    OMEGA_M      = 0.25
    OMEGA_L      = 0.75
    MASS_CONVERT = 1.0e10 / 0.73
    REDSHIFTS    = [
        127.000, 79.998, 50.000, 30.000, 19.916, 18.244, 16.725, 15.343,
         14.086, 12.941, 11.897, 10.944, 10.073,  9.278,  8.550,  7.883,
          7.272,  6.712,  6.197,  5.724,  5.289,  4.888,  4.520,  4.179,
          3.866,  3.576,  3.308,  3.060,  2.831,  2.619,  2.422,  2.239,
          2.070,  1.913,  1.766,  1.630,  1.504,  1.386,  1.276,  1.173,
          1.078,  0.989,  0.905,  0.828,  0.755,  0.687,  0.624,  0.564,
          0.509,  0.457,  0.408,  0.362,  0.320,  0.280,  0.242,  0.208,
          0.175,  0.144,  0.116,  0.089,  0.064,  0.041,  0.020,  0.000,
    ]

# ========================== DATA I/O ==========================

def read_snap(directory, snap, properties):
    """Read properties for a single snapshot, concatenated across MPI files."""
    files = _find_model_files(directory)
    if not files:
        return {}
    snap_key = f'Snap_{snap}'
    chunks = {p: [] for p in properties}
    found = False
    for fp in files:
        try:
            with h5.File(fp, 'r') as f:
                if snap_key not in f:
                    continue
                found = True
                grp = f[snap_key]
                for p in properties:
                    if p in grp:
                        chunks[p].append(np.array(grp[p]))
        except Exception as e:
            print(f"  Warning: {fp}: {e}")
    if not found:
        return {}
    data = {}
    for p in properties:
        if chunks[p]:
            arr = np.concatenate(chunks[p])
            data[p] = arr * MASS_CONVERT if p in _MASS_PROPS else arr
    return data

# ========================== PHYSICS ==========================

def ffb_threshold_mass_msun(z):
    """Li+24 FFB threshold mass [M_sun] from Eq. 2."""
    return 3e11 * (1.0 + z)**(-1.5) / HUBBLE_H


def ffb_fraction_li24(Mvir_msun, z, delta_log_M=0.15):
    """Li+24 logistic-sigmoid FFB fraction (Eq. 3)."""
    M_thresh = ffb_threshold_mass_msun(z)
    x = np.log10(np.asarray(Mvir_msun) / M_thresh) / delta_log_M
    return 1.0 / (1.0 + np.exp(-x))


# --- MBK25 helpers ---

try:
    from colossus.cosmology import cosmology as _col_cosmo
    from colossus.halo import concentration as _col_conc
    _col_cosmo.setCosmology('custom_millennium', flat=True,
                            H0=73.0, Om0=OMEGA_M, Ob0=0.045,
                            sigma8=0.90, ns=1.0, relspecies=False)
    _HAS_COLOSSUS = True
except Exception:
    _HAS_COLOSSUS = False


def _delta_vir_bn98(z):
    Ez2 = OMEGA_M * (1.0 + z)**3 + OMEGA_L
    x   = OMEGA_M * (1.0 + z)**3 / Ez2 - 1.0
    return 18.0 * np.pi**2 + 82.0 * x - 39.0 * x**2


def _rvir_m(Mvir_msun, z):
    """Virial radius [m] from M_vir [M_sun] using Bryan & Norman overdensity."""
    H0_si = HUBBLE_H * 1.0e5 / 3.085678e22
    Ez    = np.sqrt(OMEGA_M * (1.0 + z)**3 + OMEGA_L)
    rho_c = 3.0 * (H0_si * Ez)**2 / (8.0 * np.pi * 6.674e-11)
    delta = _delta_vir_bn98(z)
    return (3.0 * np.asarray(Mvir_msun) * 1.989e30 / (4.0 * np.pi * delta * rho_c))**(1.0 / 3.0)


def _c_ishiyama21(Mvir_msun, z):
    """Ishiyama+21 mean concentration (falls back to Bullock+01 power law)."""
    M_h = np.asarray(Mvir_msun) * HUBBLE_H
    if _HAS_COLOSSUS:
        try:
            c = _col_conc.concentration(M_h, '200c', z, model='ishiyama21')
            return np.maximum(np.atleast_1d(np.asarray(c, dtype=float)), 1.0)
        except Exception:
            pass
    c = 9.0 / (1.0 + z) * (M_h / 1.0e12)**(-0.13)
    return np.maximum(c, 1.0)


# g_crit = G * 3100 M_sun / pc^2  (BK25 Table 1)
_G_CRIT_SI = 6.674e-11 * 3100.0 * 1.989e30 / (3.085678e16)**2


def ffb_fraction_mbk25(Mvir_msun, z, sigma_c=0.2):
    """
    MBK25 FFB fraction via log-normal concentration scatter (BK25 Eq. 4).

    f_FFB(M, z) = P(c > c_thresh) = norm.sf((ln c_thresh - ln c_mean) / sigma_c)

    c_thresh is defined implicitly by g_max(c_thresh) = g_crit, where
    g_max = G M_vir c^2 / (2 R_vir^2 mu(c)),  mu(c) = ln(1+c) - c/(1+c).
    """
    from scipy.optimize import brentq
    Mvir_msun = np.atleast_1d(np.asarray(Mvir_msun, dtype=float))
    c_mean = _c_ishiyama21(Mvir_msun, z)
    Rvir   = _rvir_m(Mvir_msun, z)
    g_vir  = 6.674e-11 * Mvir_msun * 1.989e30 / Rvir**2

    if sigma_c == 0.0:
        mu    = np.log(1.0 + c_mean) - c_mean / (1.0 + c_mean)
        g_max = g_vir * c_mean**2 / (2.0 * mu)
        return (g_max > _G_CRIT_SI).astype(float)

    f = np.zeros(len(Mvir_msun))
    for i in range(len(Mvir_msun)):
        gv = float(g_vir[i])

        def _obj(cv):
            mu = np.log(1.0 + cv) - cv / (1.0 + cv)
            return gv * cv**2 / (2.0 * mu) - _G_CRIT_SI

        if _obj(1.0) > 0.0:
            f[i] = 1.0
            continue
        if _obj(200.0) < 0.0:
            f[i] = 0.0
            continue
        try:
            c_thresh = brentq(_obj, 1.0, 200.0, xtol=1e-3, rtol=1e-4)
            f[i] = _snorm.sf((np.log(c_thresh) - np.log(float(c_mean[i]))) / sigma_c)
        except ValueError:
            f[i] = 0.0
    return f

# ========================== STYLE ==========================

def setup_style():
    try:
        plt.style.use('./plotting/kieren_cohare_palatino_sty.mplstyle')
    except Exception:
        pass


def save_figure(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    print(f'  Saved: {path}')
    plt.close(fig)

# ========================== SNAPSHOT SELECTION ==========================

_Z_RANGE = (4.0, 15.0)


def _ffb_snaps():
    """Snapshot indices covering _Z_RANGE, ordered high-z to low-z."""
    return [i for i, z in enumerate(REDSHIFTS) if _Z_RANGE[0] <= z <= _Z_RANGE[1]]

# ========================== PLOT A ==========================

# Three halo-mass bins used in Plot A.
_MASS_BINS_A = [
    (9.0,  10.0, r'$9 < \log M_{\rm vir} < 10$',  '#92c5de'),
    (10.0, 11.0, r'$10 < \log M_{\rm vir} < 11$', '#2166ac'),
    (11.0, 13.5, r'$11 < \log M_{\rm vir} < 13.5$', '#053061'),
]


def _wilson68(n, frac):
    """68% Wilson score interval half-widths (lo, hi)."""
    z_s   = 1.0
    denom = 1 + z_s**2 / n
    cw    = (frac + z_s**2 / (2 * n)) / denom
    margin = z_s * np.sqrt((frac * (1 - frac) + z_s**2 / (4 * n)) / n) / denom
    return max(0.0, frac - (cw - margin)), max(0.0, (cw + margin) - frac)


def plot_A_ffb_fraction_vs_redshift():
    """
    FFB fraction f_FFB = N_FFB / N_central vs redshift for three halo-mass
    bins.  Li+24 shown as solid lines, MBK25 as dashed.  Shading is the
    68% Wilson confidence interval.
    """
    print('Plot A: FFB fraction vs redshift')

    models = [
        {'label': 'Li+24',  'dir': LI24_DIR,  'ls': '-'},
        {'label': 'MBK25',  'dir': MBK25_DIR, 'ls': '--'},
    ]
    props = ['FFBRegime', 'Type', 'Mvir']
    snaps = _ffb_snaps()

    fig, ax = plt.subplots()

    for mlo, mhi, mlabel, color in _MASS_BINS_A:
        for model in models:
            if not _find_model_files(model['dir']):
                print(f"  Skipping {model['label']}: no files in {model['dir']}")
                continue

            z_vals, f_vals, f_lo, f_hi = [], [], [], []
            for snap in snaps:
                d = read_snap(model['dir'], snap, props)
                if not d or 'FFBRegime' not in d:
                    continue
                central  = d['Type'] == 0
                log_mvir = np.log10(np.maximum(d['Mvir'][central], 1e-30))
                in_bin   = (log_mvir >= mlo) & (log_mvir < mhi)
                n = int(np.sum(in_bin))
                if n < 10:
                    continue
                ffb  = d['FFBRegime'][central][in_bin].astype(float)
                frac = np.mean(ffb)
                lo, hi = _wilson68(n, frac)
                z_vals.append(REDSHIFTS[snap])
                f_vals.append(frac)
                f_lo.append(lo)
                f_hi.append(hi)

            if not z_vals:
                continue

            z_arr = np.array(z_vals)
            f_arr = np.array(f_vals)
            ax.plot(z_arr, f_arr, color=color, ls=model['ls'], lw=2)
            ax.fill_between(z_arr,
                            f_arr - np.array(f_lo),
                            f_arr + np.array(f_hi),
                            color=color, alpha=0.12)

    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$f_{\rm FFB} = N_{\rm FFB}\,/\,N_{\rm central}$')
    ax.set_xlim(_Z_RANGE[1], _Z_RANGE[0])
    ax.set_ylim(0, 1)

    # Legend: mass bins (colour patches) + model line styles
    bin_handles = [mpatches.Patch(color=c, label=lbl)
                   for _, _, lbl, c in _MASS_BINS_A]
    style_handles = [
        mlines.Line2D([], [], color='k', ls='-',  lw=2, label='Li+24'),
        mlines.Line2D([], [], color='k', ls='--', lw=2, label='MBK25'),
    ]
    leg1 = ax.legend(handles=bin_handles,   loc='upper left',  frameon=False,
                     fontsize='small', title=r'$\log_{10}\,M_{\rm vir}\ [M_\odot]$',
                     title_fontsize='small')
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, loc='upper right', frameon=False, fontsize='small')

    fig.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR, 'A_ffb_fraction_vs_z' + OUTPUT_FORMAT))

# ========================== PLOT B ==========================

def _build_ffb_grid(directory, snaps, mass_bins):
    """
    Build a 2D array (n_snaps × n_mass_bins) of mean FFB fraction.
    Returns the grid; cells with fewer than MIN_N galaxies are NaN.
    """
    MIN_N = 5
    props = ['FFBRegime', 'Type', 'Mvir']
    grid  = np.full((len(snaps), len(mass_bins) - 1), np.nan)

    if not _find_model_files(directory):
        return grid

    for row, snap in enumerate(snaps):
        d = read_snap(directory, snap, props)
        if not d or 'FFBRegime' not in d:
            continue
        central  = d['Type'] == 0
        log_mvir = np.log10(np.maximum(d['Mvir'][central], 1e-30))
        ffb      = d['FFBRegime'][central].astype(float)
        for col in range(len(mass_bins) - 1):
            mask = (log_mvir >= mass_bins[col]) & (log_mvir < mass_bins[col + 1])
            if np.sum(mask) >= MIN_N:
                grid[row, col] = np.mean(ffb[mask])
    return grid


def _z_edges(snaps):
    """Build N+1 redshift bin edges for N snapshots (decreasing order)."""
    z_arr = np.array([REDSHIFTS[s] for s in snaps])
    dz    = np.abs(np.diff(z_arr))
    top   = z_arr[0]  + 0.5 * dz[0]
    bot   = z_arr[-1] - 0.5 * dz[-1]
    mid   = 0.5 * (z_arr[:-1] + z_arr[1:])
    return np.concatenate([[top], mid, [bot]])


def plot_B_ffb_heatmap():
    """
    2-D heatmap of f_FFB(M_halo, z) for Li+24 (left) and MBK25 (right).
    Colour shows the simulated FFB fraction in each (mass, redshift) bin.
    Dashed white contour marks f_FFB = 0.5 from the respective theoretical
    prediction (Li+24 sigmoid or MBK25 log-normal concentration scatter).
    """
    print('Plot B: f_FFB(M_halo, z) heatmap')

    mass_bins = np.linspace(8.5, 13.0, 32)   # log10(M_vir / M_sun)
    snaps     = _ffb_snaps()
    z_e       = _z_edges(snaps)

    models = [
        {'label': 'Li+24',  'dir': LI24_DIR,  'theory': ffb_fraction_li24},
        {'label': 'MBK25',  'dir': MBK25_DIR, 'theory': ffb_fraction_mbk25},
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    pcm_last = None

    for ax, m in zip(axes, models):
        grid = _build_ffb_grid(m['dir'], snaps, mass_bins)

        pcm = ax.pcolormesh(mass_bins, z_e, grid,
                            cmap='RdPu', vmin=0.0, vmax=1.0,
                            shading='flat')
        pcm_last = pcm

        # Theoretical f_FFB = 0.5 contour
        log_M_th  = np.linspace(8.5, 13.0, 200)
        M_th      = 10.0**log_M_th
        z_th      = np.linspace(_Z_RANGE[0], _Z_RANGE[1], 60)
        F_th = np.zeros((len(z_th), len(log_M_th)))
        for j, zz in enumerate(z_th):
            F_th[j, :] = m['theory'](M_th, zz)
        ax.contour(log_M_th, z_th, F_th, levels=[0.5],
                   colors='white', linewidths=1.8, linestyles='--')

        ax.set_xlabel(r'$\log_{10}\,M_{\rm vir}\ [M_\odot]$')
        ax.set_xlim(8.5, 13.0)
        ax.set_ylim(_Z_RANGE[1], _Z_RANGE[0])
        ax.set_title(m['label'])

    axes[0].set_ylabel(r'$z$')

    cbar = fig.colorbar(pcm_last, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label(r'$f_{\rm FFB}$')

    # Annotate the 50% contour
    for ax in axes:
        ax.annotate(r'$f_{\rm FFB}=0.5$', xy=(0.97, 0.05),
                    xycoords='axes fraction', ha='right', va='bottom',
                    color='white', fontsize='small')

    fig.tight_layout()
    save_figure(fig, os.path.join(OUTPUT_DIR, 'B_ffb_heatmap' + OUTPUT_FORMAT))

# ========================== MAIN ==========================

ALL_PLOTS = {
    'A': plot_A_ffb_fraction_vs_redshift,
    'B': plot_B_ffb_heatmap,
}


def main():
    setup_style()
    keys = [k.upper() for k in sys.argv[1:]] if len(sys.argv) > 1 else list(ALL_PLOTS)
    for key in keys:
        if key in ALL_PLOTS:
            ALL_PLOTS[key]()
        else:
            print(f"Unknown plot '{key}'. Available: {list(ALL_PLOTS)}")


if __name__ == '__main__':
    main()

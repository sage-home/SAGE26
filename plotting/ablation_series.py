#!/usr/bin/env python
"""
SAGE26 ablation series
======================
Isolates the effect of each new SAGE26 ingredient by re-running the fiducial
model with that ingredient -- and only that ingredient -- switched off, then
plotting the stellar mass function and the cosmic star formation rate density
of every variant against the fiducial model.

Each ablation parameter file differs from ``input/millennium_all.par`` by a
single line (see the header of each file).  Nothing is recalibrated: the point
of the series is to isolate one ingredient at fixed calibration, so every other
parameter is held at its fiducial value.  A consequence worth stating in the
text is that the ablated runs are therefore *not* re-tuned models -- they show
what the fiducial calibration does without that piece of physics, which is the
quantity relevant to "what does this module contribute?".

Usage
-----
    python plotting/ablation_series.py            # plot from existing output
    python plotting/ablation_series.py --run      # run any missing variants first
    python plotting/ablation_series.py --run --force   # re-run every variant
    python plotting/ablation_series.py --with rps      # add an optional ablation
    python plotting/ablation_series.py --with rps --with snecons
    python plotting/ablation_series.py --no-sage16     # drop the SAGE16 reference

The figure covers the four ingredients the paper's headline claim rests on: FIRE
stellar feedback, H2-based star formation, the two-regime CGM and the FFB mode, plus
a joint run with all four disabled together.  Ram-pressure stripping and the SN energy
bound are available through --with but are off by default: neither is part of those
claims.

Must be run from the repository root (paths are relative, as in the .par files).
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The paper module owns the simulation constants (volume, mass conversion,
# redshift table), the mass-function estimator and the observational
# compilations.  Reusing it keeps this figure on exactly the same footing as
# every other stellar mass function in the paper.
import paper_plots as pp


# ========================== CONFIGURATION ==========================

SAGE_BINARY = './sage'

# One entry per curve.  'par' is the parameter file, 'out' its OutputDir, and
# 'switch' the single parameter that differs from the fiducial run.
VARIANTS = [
    {'key': 'full',   'par': 'input/millennium_all.par',
     'out': './output/millennium_all/',
     'label': r'SAGE26 (fiducial)',            'switch': None,
     'color': 'black',   'ls': '-',  'lw': 3.6, 'zorder': 12},
    {'key': 'nofire', 'par': 'input/millennium_nofire.par',
     'out': './output/millennium_nofire/',
     'label': r'no FIRE feedback',             'switch': ('FIREmodeOn', 0),
     'color': '#0C5DA5', 'ls': (0, (6, 2)),       'lw': 2.4, 'zorder': 10},
    {'key': 'noh2',   'par': 'input/millennium_noh2.par',
     'out': './output/millennium_noh2/',
     'label': r'no H$_2$ star formation',      'switch': ('SFprescription', 0),
     'color': '#00B945', 'ls': (0, (1, 1.4)),     'lw': 2.6, 'zorder': 10},
    {'key': 'nocgm',  'par': 'input/millennium_nocgm.par',
     'out': './output/millennium_nocgm/',
     'label': r'no two-regime CGM',            'switch': ('CGMrecipeOn', 0),
     'color': '#FF9500', 'ls': (0, (7, 2, 1.5, 2)), 'lw': 2.4, 'zorder': 10},
    {'key': 'noffb',  'par': 'input/millennium_noffb.par',
     'out': './output/millennium_noffb/',
     'label': r'no FFB mode',                  'switch': ('FeedbackFreeModeOn', 0),
     'color': '#FF2C00', 'ls': (0, (3, 1.6)),     'lw': 2.4, 'zorder': 11},
    {'key': 'noallfour', 'par': 'input/millennium_noallfour.par',
     'out': './output/millennium_noallfour/',
     'label': r'all four removed',
     'switch': [('FIREmodeOn', 0), ('SFprescription', 0),
                ('CGMrecipeOn', 0), ('FeedbackFreeModeOn', 0)],
     'color': '#845B97', 'ls': '-',  'lw': 2.8, 'zorder': 8},
    {'key': 'sage16', 'par': 'input/millennium_vanilla.par',
     'out': './output/millennium_vanilla/',
     'label': r'SAGE16 (separately calibrated)', 'switch': None,
     'color': '#474747', 'ls': (0, (10, 3)),      'lw': 2.4, 'zorder': 9},
]

# VARIANTS = [
#     {'key': 'full',   'par': 'input/millennium.par',
#      'out': './output/millennium/',
#      'label': r'Free-Fall Time Inflow',            'switch': ('PrecipCriterionOn', 0),
#      'color': 'black',   'ls': '-',  'lw': 3.6, 'zorder': 12},
#     {'key': 'nofire', 'par': 'input/millennium_cgmdyn.par',
#      'out': './output/millennium_cgmdyn/',
#      'label': r'Dynamical Time Inflow',             'switch': ('PrecipCriterionOn', 5),
#      'color': '#0C5DA5', 'ls': (0, (6, 2)),       'lw': 2.4, 'zorder': 10},
#     {'key': 'noh2',   'par': 'input/millennium_voit.par',
#      'out': './output/millennium_voit/',
#      'label': r'Apparent Voit 17',      'switch': ('PrecipCriterionOn', 1),
#      'color': '#00B945', 'ls': (0, (1, 1.4)),     'lw': 2.6, 'zorder': 10},
#     {'key': 'nocgm',  'par': 'input/millennium_simpleinflow.par',
#      'out': './output/millennium_simpleinflow/',
#      'label': r'Simple Inflow Carr 2022',            'switch': ('CGMsimpleInflowOn', 1),
#      'color': '#FF9500', 'ls': (0, (7, 2, 1.5, 2)), 'lw': 4.5, 'zorder': 10},
#     {'key': 'noffb',  'par': 'input/millennium_disk2.par',
#      'out': './output/millennium_disk2/',
#      'label': r'Simple + Disk smoothing',                  'switch': ('DiskRadiusOn', 1),
#      'color': '#FF2C00', 'ls': (0, (3, 1.6)),     'lw': 2.4, 'zorder': 11},
    # {'key': 'noallfour', 'par': 'input/millennium_noallfour.par',
    #  'out': './output/millennium_noallfour/',
    #  'label': r'all four removed',
    #  'switch': [('FIREmodeOn', 0), ('SFprescription', 0),
    #             ('CGMrecipeOn', 0), ('FeedbackFreeModeOn', 0)],
    #  'color': '#845B97', 'ls': '-',  'lw': 2.8, 'zorder': 8},
#     {'key': 'sage16', 'par': 'input/millennium_vanilla.par',
#      'out': './output/millennium_vanilla/',
#      'label': r'SAGE16 (separately calibrated)', 'switch': None,
#      'color': '#474747', 'ls': (0, (10, 3)),      'lw': 2.4, 'zorder': 9},
# ]

# The four ingredients whose individual contributions sum to the joint ablation.
# Comparing that sum against JOINT_KEY measures how far they are from acting
# independently, without the calibration differences that make SAGE16 unusable
# for the purpose.
FOUR_KEYS = ('nofire', 'noh2', 'nocgm', 'noffb')
JOINT_KEY = 'noallfour'

REFERENCE_KEY = 'full'      # residuals are measured against this variant

# Ingredients that are not part of the claims this figure supports are left out by
# default and switched on individually.  Ram-pressure stripping moves the HI content
# far more than it moves the stellar mass function; the SN energy bound postdates the
# submitted version, so it is not one of the four the referee asked about.
OPTIONAL_VARIANTS = {
    'rps': {
        'key': 'norps',  'par': 'input/millennium_norps.par',
        'out': './output/millennium_norps/',
        'label': r'no ram-pressure stripping', 'switch': ('RamPressureStrippingOn', 0),
        'color': '#8C564B', 'ls': (0, (5, 1.5, 1.5, 1.5, 1.5, 1.5)),
        'lw': 2.4, 'zorder': 10,
    },
    'snecons': {
        'key': 'nosnecons', 'par': 'input/millennium_nosnecons.par',
        'out': './output/millennium_nosnecons/',
        'label': r'no SN energy bound', 'switch': ('SNEnergyConservationOn', 0),
        'color': '#17A2B8', 'ls': (0, (2, 1, 5, 1)), 'lw': 2.4, 'zorder': 10,
    },
    # Ablations of the precipitation criterion itself, inside the two-regime CGM
    # rather than against it: 'nocgm' above removes the CGM machinery entirely,
    # so it cannot say how much of the improvement comes from the Voit criterion
    # as opposed to the regime split plus free-fall accretion.  These three do.
    # 'precip' is the joint control (mdot = M_CGM/t_ff), 'sigmoid' and 'meq' drop
    # one factor each, so their offsets are separately attributable.
    'precip': {
        'key': 'noprecip', 'par': 'input/millennium_noprecip.par',
        'out': './output/millennium_noprecip/',
        'label': r'no precipitation criterion ($f_{\rm inflow}\equiv1$)',
        'switch': ('PrecipCriterionOn', 0),
        'color': '#E377C2', 'ls': (0, (4, 1.5, 1, 1.5)), 'lw': 2.4, 'zorder': 10,
    },
    'sigmoid': {
        'key': 'nosigmoid', 'par': 'input/millennium_nosigmoid.par',
        'out': './output/millennium_nosigmoid/',
        'label': r'no $f_{\rm inflow}$ sigmoid', 'switch': ('PrecipCriterionOn', 2),
        'color': '#BCBD22', 'ls': (0, (1, 1)), 'lw': 2.4, 'zorder': 10,
    },
    'meq': {
        'key': 'nomeq', 'par': 'input/millennium_nomeq.par',
        'out': './output/millennium_nomeq/',
        'label': r'no condensation term ($M_{\rm eq}=0$)',
        'switch': ('PrecipCriterionOn', 3),
        'color': '#7F7F7F', 'ls': (0, (6, 1.5)), 'lw': 2.4, 'zorder': 10,
    },
    # Mode 4 drops both factors while keeping the rest of the precipitation
    # path, so it is the reference the two single-factor rows above should be
    # read against: mode 4 -> 2 isolates M_eq and mode 4 -> 3 isolates the
    # sigmoid, with nothing else changing.  Mode 0 ('precip') additionally
    # skips the hand-over to standard cooling, so it moves two things at once
    # and is the weaker control, even though the two agree in the mean.
    'sage16cold': {
        'key': 'sage16cold', 'par': 'input/millennium_sage16cold.par',
        'out': './output/millennium_sage16cold/',
        'label': r'SAGE16 cold accretion ($t_{\rm dyn}$)',
        'switch': ('PrecipCriterionOn', 5),
        'color': '#9467BD', 'ls': (0, (8, 2, 2, 2)), 'lw': 2.4, 'zorder': 10,
    },
    'precipfactors': {
        'key': 'noprecipfactors', 'par': 'input/millennium_noprecipfactors.par',
        'out': './output/millennium_noprecipfactors/',
        'label': r'no $f_{\rm inflow}$, no $M_{\rm eq}$',
        'switch': ('PrecipCriterionOn', 4),
        'color': '#17BECF', 'ls': (0, (3, 1, 1, 1, 1, 1)), 'lw': 2.4, 'zorder': 10,
    },
}

# Stellar mass function panels.  'select' is 'all', 'sf' or 'q': the sSFR-split
# panels use pp.SSFR_CUT, the same division as every other figure in the paper.
# The star-forming panel at cosmic noon is the one that tests the claim about the
# number density of massive *star-forming* galaxies at z ~ 2.
SMF_PANELS = [
    {'z': 0.0, 'select': 'all', 'tag': 'z=0',
     'xlim': (7.6, 12.4), 'ylim': (-5.5, -0.7)},
    {'z': 2.0, 'select': 'all', 'tag': 'z=2',
     'xlim': (7.6, 12.4), 'ylim': (-5.5, -0.7)},
    {'z': 2.0, 'select': 'sf',  'tag': 'z=2 star-forming',
     'xlim': (7.6, 12.4), 'ylim': (-5.5, -0.7)},
    {'z': 6.0, 'select': 'all', 'tag': 'z=6',
     'xlim': (7.6, 12.4), 'ylim': (-5.5, -0.7)},
]
SMF_BINWIDTH = 0.2
SMF_MASS_RANGE = (6.0, 13.0)    
SMF_OBS_DZ = 0.5                
SMF_ROBUST_MIN_COUNT = 10       

SELECT_LABEL = {'all': None, 'sf': 'star-forming', 'q': 'quiescent'}

RESIDUAL_YLIM = (-1.3, 1.3)
RESIDUAL_NEGLIGIBLE = 0.1   

CSFRD_ZLIM = (0.0, 10.0)
CSFRD_YLIM = (-3.4, -0.4)

TABLE_MASSES = (8.5, 9.5, 10.5, 11.5)
TABLE_REDSHIFTS = (0.0, 1.0, 2.0, 4.0, 6.0, 8.0)

OUTPUT_NAME = 'Ablation_Series'
EXTRA_OUTPUT_NAME = 'Ablation_Series_Extras'

# ========================== RUNNING THE MODEL ==========================

def run_variants(variants, force=False):
    for v in variants:
        if not os.path.exists(v['par']):
            print(f"  {v['key']:>7s}: parameter file {v['par']} missing -- skipped")
            continue
        have_output = pp.model_files_exist(v['out'])
        if have_output and not force:
            print(f"  {v['key']:>7s}: output already present in {v['out']} -- skipped")
            continue
        os.makedirs(v['out'], exist_ok=True)
        print(f"  {v['key']:>7s}: {SAGE_BINARY} {v['par']}")
        res = subprocess.run([SAGE_BINARY, v['par']],
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if res.returncode != 0:
            print(f"    FAILED (exit {res.returncode})")
            sys.stderr.write(res.stderr.decode(errors='replace')[-2000:])
            sys.exit(res.returncode)
    print()


def read_par(path):
    params = {}
    if not os.path.exists(path):
        return params
    with open(path) as f:
        for line in f:
            line = line.split('%', 1)[0].strip()
            if not line or line.startswith('->'):
                continue
            fields = line.split()
            if len(fields) >= 2:
                params[fields[0]] = fields[1]
    return params


def check_switches(variants):
    import h5py as h5

    available, runtime = [], {}
    for v in variants:
        files = pp.find_model_files(v['out'])
        if not files:
            print(f"  {v['key']:>7s}: no output in {v['out']} -- dropped from the figure")
            continue
        with h5.File(files[0], 'r') as f:
            runtime[v['key']] = dict(f['Header/Runtime'].attrs)
        available.append(v)

    ref_hdr = runtime.get(REFERENCE_KEY)
    if ref_hdr is None:
        print('  Warning: fiducial run unavailable, cannot verify switches')
        return available

    ref_variant = next(v for v in VARIANTS if v['key'] == REFERENCE_KEY)
    ref_par = read_par(ref_variant['par'])

    ignored = {'OutputDir', 'FileNameGalaxies'}
    unrecorded = set()

    for v in available:
        if v['key'] == REFERENCE_KEY:
            continue

        hdr = runtime[v['key']]
        hdr_diff = sorted(k for k in set(ref_hdr) | set(hdr)
                          if k not in ignored and ref_hdr.get(k) != hdr.get(k))

        par = read_par(v['par'])
        par_diff = sorted(k for k in set(ref_par) | set(par)
                          if k not in ignored and ref_par.get(k) != par.get(k))

        if v['switch'] is None:
            print(f"  {v['key']:>9s}: reference model, not an ablation "
                  f"-- {len(hdr_diff)} recorded parameters differ: {hdr_diff}")
            continue

        switches = v['switch'] if isinstance(v['switch'], list) else [v['switch']]
        expected = sorted(name for name, _ in switches)
        missing = [name for name, _ in switches
                   if not (name in ref_hdr and name in hdr)]
        unrecorded.update(missing)

        values_ok = all(par.get(name) == str(value) for name, value in switches)
        hdr_expected = sorted(n for n in expected if n not in missing)
        ok = (par_diff == expected and values_ok and hdr_diff == hdr_expected)
        note = '' if not missing \
            else f'  ({", ".join(missing)} not written to the HDF5 header)'
        setting = ', '.join(f'{name}={par.get(name)}' for name, _ in switches)
        print(f"  {v['key']:>9s}: {setting};  .par diff: {par_diff};  "
              f"header diff: {hdr_diff}  [{'OK' if ok else 'CHECK'}]{note}")

    if unrecorded:
        print(f"\n  Note: {', '.join(sorted(unrecorded))} "
              f"{'is' if len(unrecorded) == 1 else 'are'} absent from "
              f"Header/Runtime in the output files...")
    print()
    return available


# ========================== MEASUREMENTS ==========================

def read_sim(directory):
    hdr = pp._read_sim_header(directory)
    if hdr is None:
        return None
    return {
        'box_size': hdr['box_size'],
        'hubble_h': hdr['hubble_h'],
        'volume': (hdr['box_size'] / hdr['hubble_h'])**3 * hdr['volume_fraction'],
        'volume_fraction': hdr['volume_fraction'],
        'mass_convert': hdr['unit_mass_in_g'] / pp._MSUN_CGS / hdr['hubble_h'],
        'redshifts': np.asarray(hdr['redshifts'], dtype=float),
    }


def check_same_simulation(sim, variants):
    mismatched = []
    for v in variants:
        other = read_sim(v['out'])
        if other is None:
            continue
        same = (np.isclose(other['box_size'], sim['box_size'])
                and np.isclose(other['hubble_h'], sim['hubble_h'])
                and np.isclose(other['volume_fraction'], sim['volume_fraction'])
                and other['redshifts'].size == sim['redshifts'].size
                and np.allclose(other['redshifts'], sim['redshifts']))
        if not same:
            mismatched.append(v['key'])
            print(f"  {v['key']:>9s}: WARNING -- different simulation ")
    if not mismatched:
        print(f"  all runs on the same simulation: box {sim['box_size']:g} Mpc/h, "
              f"h = {sim['hubble_h']:g}, {sim['redshifts'].size} snapshots, "
              f"volume {sim['volume']:.3g} Mpc^3")
    print()
    return mismatched


def density_floor(sim, min_count=None, binwidth=None):
    min_count = SMF_ROBUST_MIN_COUNT if min_count is None else min_count
    binwidth = SMF_BINWIDTH if binwidth is None else binwidth
    return float(np.log10(min_count / sim['volume'] / binwidth))


def smf(path, z_target, sim, select='all'):
    redshifts = sim['redshifts']
    snap = pp._snap_nearest_z(redshifts, z_target)
    props = ['StellarMass']
    if select != 'all':
        props += ['SfrDisk', 'SfrBulge']
    data = pp.read_snap_from_files(pp.find_model_files(path), f'Snap_{snap}',
                                   props, mass_convert=sim['mass_convert'])
    if not data:
        return snap, np.nan, None, None

    m = data['StellarMass']
    keep = m > 0
    if select != 'all':
        with np.errstate(divide='ignore', invalid='ignore'):
            ssfr = pp.log_ssfr(data['SfrDisk'], data['SfrBulge'], m)
        keep &= (ssfr > pp.SSFR_CUT) if select == 'sf' else (ssfr <= pp.SSFR_CUT)

    m = m[keep]
    if m.size == 0:
        return snap, redshifts[snap], None, None
    x, phi, _ = pp.mass_function(np.log10(m), sim['volume'],
                                binwidth=SMF_BINWIDTH, mass_range=SMF_MASS_RANGE)
    return snap, redshifts[snap], x, phi


def csfrd(path, sim):
    files = pp.find_model_files(path)
    z = sim['redshifts']
    rho = np.full(z.size, np.nan)
    for snap in range(z.size):
        d = pp.read_snap_from_files(files, f'Snap_{snap}', ['SfrDisk', 'SfrBulge'],
                                    mass_convert=sim['mass_convert'])
        if not d:
            continue
        total = np.sum(d['SfrDisk'] + d['SfrBulge'])
        if total > 0:
            rho[snap] = total / sim['volume']
    with np.errstate(divide='ignore', invalid='ignore'):
        return z, np.log10(rho)


def stellar_mass_density(path, sim):
    """Cosmic Stellar Mass Density over every snapshot."""
    files = pp.find_model_files(path)
    z = sim['redshifts']
    rho_star = np.full(z.size, np.nan)
    for snap in range(z.size):
        d = pp.read_snap_from_files(files, f'Snap_{snap}', ['StellarMass'],
                                    mass_convert=sim['mass_convert'])
        if not d:
            continue
        total = np.sum(d['StellarMass'])
        if total > 0:
            rho_star[snap] = total / sim['volume']
    with np.errstate(divide='ignore', invalid='ignore'):
        return z, np.log10(rho_star)


def extra_z0_metrics(path, sim):
    """
    Computes HI/HII Mass functions, Metallicity Relation, and Quiescent Fraction at z=0.
    """
    snap = pp._snap_nearest_z(sim['redshifts'], 0.0)
    # Safely try fetching extra properties
    props = ['StellarMass', 'SfrDisk', 'SfrBulge', 'ColdGas', 'MetalsColdGas', 'H1gas', 'H2Mass', 'H2gas']
    d = pp.read_snap_from_files(pp.find_model_files(path), f'Snap_{snap}', props, mass_convert=sim['mass_convert'])
    
    if not d:
        return {}
        
    out = {}
    m = d.get('StellarMass', np.array([]))
    if len(m) == 0: 
        return out
        
    keep = m > 0
    m_good = m[keep]
    log_m = np.log10(m_good)

    # 1. HI Mass Function
    hi = d.get('H1gas')
    if hi is not None:
        hi_good = hi[keep]
        valid = hi_good > 0
        if np.any(valid):
            x_hi, phi_hi, _ = pp.mass_function(np.log10(hi_good[valid]), sim['volume'], binwidth=0.2, mass_range=(7.0, 11.5))
            out['himf'] = (x_hi, phi_hi)

    # 2. HII / H2 Mass Function
    # Defaults to checking for 'H2Gas' first, then falls back to 'H2Mass'
    hii = d.get('H2gas') if 'H2gas' in d else d.get('H2Gas')
    if hii is not None:
        hii_good = hii[keep]
        valid = hii_good > 0
        if np.any(valid):
            x_hii, phi_hii, _ = pp.mass_function(np.log10(hii_good[valid]), sim['volume'], binwidth=0.2, mass_range=(7.0, 11.5))
            out['hiimf'] = (x_hii, phi_hii)

    # 3. Mass-Metallicity Relation (Stellar Mass vs 12 + O/H)
    gas = d.get('ColdGas')
    metals = d.get('MetalsColdGas')
    if gas is not None and metals is not None:
        gas_good = gas[keep]
        metals_good = metals[keep]
        valid_gas = (gas_good > 0) & (metals_good > 0)
        
        # Approximate Oxygen Abundance: 12 + log(O/H) ~ 9.0 + log10(Z/0.02)
        Z = np.zeros_like(gas_good)
        Z[valid_gas] = metals_good[valid_gas] / gas_good[valid_gas]
        
        bins = np.arange(8.0, 12.0, 0.2)
        bin_centers = bins[:-1] + 0.1
        mzr = np.full_like(bin_centers, np.nan)
        
        for i in range(len(bins)-1):
            in_bin = (log_m >= bins[i]) & (log_m < bins[i+1]) & valid_gas
            if np.sum(in_bin) >= 1:
                oh = 9.0 + np.log10(Z[in_bin] / 0.02)
                mzr[i] = np.median(oh)
        out['mzr'] = (bin_centers, mzr)

    # 4. Quiescent Fraction vs Stellar Mass
    sfr_d = d.get('SfrDisk')
    sfr_b = d.get('SfrBulge')
    if sfr_d is not None and sfr_b is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            ssfr = pp.log_ssfr(sfr_d[keep], sfr_b[keep], m_good)
            
        bins = np.arange(8.0, 12.0, 0.2)
        bin_centers = bins[:-1] + 0.1
        qfrac = np.full_like(bin_centers, np.nan)
        
        for i in range(len(bins)-1):
            in_bin = (log_m >= bins[i]) & (log_m < bins[i+1])
            if np.sum(in_bin) >= 10:
                qfrac[i] = np.mean(ssfr[in_bin] <= pp.SSFR_CUT)
        out['qfrac'] = (bin_centers, qfrac)
        
    return out


def integrated_z0(path, sim):
    snap = pp._snap_nearest_z(sim['redshifts'], 0.0)
    d = pp.read_snap_from_files(pp.find_model_files(path), f'Snap_{snap}',
                                ['StellarMass', 'ColdGas', 'SfrDisk', 'SfrBulge'],
                                mass_convert=sim['mass_convert'])
    if not d:
        return None
    ms = d['StellarMass']
    good = ms > 0
    out = {'rho_star': ms[good].sum() / sim['volume'],
           'rho_cold': d['ColdGas'][good].sum() / sim['volume'],
           'qfrac': {}}
    with np.errstate(divide='ignore', invalid='ignore'):
        ssfr = pp.log_ssfr(d['SfrDisk'], d['SfrBulge'], ms)
    lm = np.log10(np.maximum(ms, 1.0))
    for centre in (8.5, 9.5, 10.5, 11.5):
        b = good & (lm > centre - 0.5) & (lm < centre + 0.5)
        out['qfrac'][centre] = (float(np.mean(ssfr[b] <= pp.SSFR_CUT)),
                                int(b.sum())) if b.sum() >= 20 else (np.nan, 0)
    return out


def panel_id(panel):
    return (panel['z'], panel['select'])


def measure(variants, sim):
    out = {}
    for v in variants:
        print(f"  {v['key']:>9s}: {v['out']}")
        entry = {'smf': {}}
        for panel in SMF_PANELS:
            snap, z_snap, x, phi = smf(v['out'], panel['z'], sim, panel['select'])
            entry['smf'][panel_id(panel)] = {'snap': snap, 'z': z_snap,
                                             'x': x, 'phi': phi}
        entry['z'], entry['csfrd'] = csfrd(v['out'], sim)
        entry['z_smd'], entry['smd'] = stellar_mass_density(v['out'], sim)
        entry['z0'] = integrated_z0(v['out'], sim)
        entry['extras'] = extra_z0_metrics(v['out'], sim)
        out[v['key']] = entry
    print()
    return out


# ========================== PLOTTING ==========================

def _apply_plasma_colours(variants):
    fixed = {REFERENCE_KEY, 'sage16'}
    ablations = [v for v in variants if v['key'] not in fixed]
    if not ablations:
        return variants
    cmap = plt.get_cmap('plasma')
    stops = np.linspace(0.0, 0.82, len(ablations))
    for v, s in zip(ablations, stops):
        v['color'] = cmap(s)
    return variants


def make_figure(variants, results, sim, outdir):
    variants = _apply_plasma_colours(variants)
    ncols = len(SMF_PANELS) + 1
    fig = plt.figure(figsize=(5.6 * ncols, 9.6))
    fig.set_tight_layout(False)
    gs = fig.add_gridspec(2, ncols, height_ratios=[2.05, 1.0],
                          hspace=0.06, wspace=0.28)

    ref = results[REFERENCE_KEY]

    for col, panel in enumerate(SMF_PANELS):
        z_panel, select = panel['z'], panel['select']
        pid = panel_id(panel)
        ax = fig.add_subplot(gs[0, col])
        axr = fig.add_subplot(gs[1, col], sharex=ax)

        ref_phi = ref['smf'][pid]['phi']
        for v in variants:
            m = results[v['key']]['smf'][pid]
            if m['phi'] is None:
                continue
            good = np.isfinite(m['phi'])
            ax.plot(m['x'][good], m['phi'][good], color=v['color'], ls=v['ls'],
                    lw=v['lw'], zorder=v['zorder'], label=v['label'])
            if v['key'] == REFERENCE_KEY or ref_phi is None:
                continue
            delta = m['phi'] - ref_phi
            good = np.isfinite(delta)
            axr.plot(m['x'][good], delta[good], color=v['color'], ls=v['ls'],
                     lw=v['lw'], zorder=v['zorder'])

        z_snap = ref['smf'][pid]['z']
        sel_label = SELECT_LABEL[select]
        ax.text(0.95, 0.94, rf'$z = {z_snap:.2f}$', transform=ax.transAxes,
                ha='right', va='top')
        if sel_label is not None:
            ax.text(0.05, 0.06, sel_label, transform=ax.transAxes,
                    ha='left', va='bottom')
        ax.set_xlim(*panel['xlim'])
        ax.set_ylim(*panel['ylim'])
        _residual_guides(axr)
        axr.set_ylim(*RESIDUAL_YLIM)
        axr.set_xlabel(r'$\log_{10}\ m_{*}\ [M_{\odot}]$')
        if col == 0:
            ax.set_ylabel(r'$\log_{10}\ \phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$')
            axr.set_ylabel(r'$\Delta \log_{10}\ \phi$')
        _format(ax, xmaj=1.0, xmin=0.2, ymaj=1.0, ymin=0.2, hide_xticklabels=True)
        _format(axr, xmaj=1.0, xmin=0.2, ymaj=0.5, ymin=0.1)

    ax = fig.add_subplot(gs[0, ncols - 1])
    axr = fig.add_subplot(gs[1, ncols - 1], sharex=ax)

    ref_rho = ref['csfrd']
    for v in variants:
        r = results[v['key']]
        good = np.isfinite(r['csfrd'])
        ax.plot(r['z'][good], r['csfrd'][good], color=v['color'], ls=v['ls'],
                lw=v['lw'], zorder=v['zorder'], label=v['label'])
        if v['key'] == REFERENCE_KEY:
            continue
        delta = r['csfrd'] - ref_rho
        good = np.isfinite(delta)
        axr.plot(r['z'][good], delta[good], color=v['color'], ls=v['ls'],
                 lw=v['lw'], zorder=v['zorder'])

    ax.set_xlim(*CSFRD_ZLIM)
    ax.set_ylim(*CSFRD_YLIM)
    ax.set_ylabel(r'$\log_{10}\ \rho_{\rm SFR}\ '
                  r'[M_{\odot}\ \mathrm{yr}^{-1}\ \mathrm{Mpc}^{-3}]$')
    _residual_guides(axr)
    axr.set_ylim(*RESIDUAL_YLIM)
    axr.set_xlabel(r'$\mathrm{Redshift}$')
    axr.set_ylabel(r'$\Delta \log_{10}\ \rho_{\rm SFR}$')
    _format(ax, xmaj=2.0, xmin=0.5, ymaj=1.0, ymin=0.2, hide_xticklabels=True)
    _format(axr, xmaj=2.0, xmin=0.5, ymaj=0.5, ymin=0.1)

    model_labels = [v['label'] for v in variants]
    first_ax = fig.axes[0]
    handles, labels = first_ax.get_legend_handles_labels()
    keep = [(h, l) for h, l in zip(handles, labels) if l in model_labels]
    order = {v['label']: i for i, v in enumerate(variants)}
    keep.sort(key=lambda hl: order[hl[1]])
    first_ax.legend([h for h, _ in keep], [l for _, l in keep],
                    loc='lower left', frameon=False, fontsize=12,
                    handlelength=2.6, labelspacing=0.3, borderaxespad=0.8)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, OUTPUT_NAME + pp.OUTPUT_FORMAT)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')

    return path


def make_extra_figure(variants, results, sim, outdir):
    """
    Second figure containing:
      1. HI Mass Function
      2. HII / H2 Mass Function
      3. Gas Metallicity (12+O/H) vs Stellar Mass
      4. Stellar Mass Density vs Redshift
      5. Quiescent Fraction vs Stellar Mass
    """
    variants = _apply_plasma_colours(variants)
    ncols = 5
    fig = plt.figure(figsize=(5.6 * ncols, 9.6))
    fig.set_tight_layout(False)
    gs = fig.add_gridspec(2, ncols, height_ratios=[2.05, 1.0],
                          hspace=0.06, wspace=0.32)

    ref = results[REFERENCE_KEY]
    
    # 1-5 definitions
    panels = [
        {'id': 'himf', 'xlabel': r'$\log_{10}\ M_{HI}\ [M_{\odot}]$', 'ylabel': r'$\log_{10}\ \phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$', 'title': 'z = 0.00'},
        {'id': 'hiimf', 'xlabel': r'$\log_{10}\ M_{H2}\ [M_{\odot}]$', 'ylabel': r'$\log_{10}\ \phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$', 'title': 'z = 0.00'},
        {'id': 'mzr', 'xlabel': r'$\log_{10}\ m_{*}\ [M_{\odot}]$', 'ylabel': r'$12 + \log(\mathrm{O/H})$', 'title': 'z = 0.00'},
        {'id': 'smd', 'xlabel': r'$\mathrm{Redshift}$', 'ylabel': r'$\log_{10}\ \rho_{*}\ [M_{\odot}\ \mathrm{Mpc}^{-3}]$', 'title': ''},
        {'id': 'qfrac', 'xlabel': r'$\log_{10}\ m_{*}\ [M_{\odot}]$', 'ylabel': r'$\mathrm{Quiescent\ Fraction}$', 'title': 'z = 0.00'},
    ]

    for col, p_info in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        axr = fig.add_subplot(gs[1, col], sharex=ax)
        
        pid = p_info['id']
        
        # Grab reference base data
        if pid == 'smd':
            ref_x, ref_y = ref['z_smd'], ref['smd']
        else:
            if pid not in ref.get('extras', {}):
                ax.text(0.5, 0.5, 'Data Missing', transform=ax.transAxes, ha='center')
                continue
            ref_x, ref_y = ref['extras'][pid]
            
        for v in variants:
            r = results[v['key']]
            
            if pid == 'smd':
                x, y = r['z_smd'], r['smd']
            else:
                if pid not in r.get('extras', {}): continue
                x, y = r['extras'][pid]

            good = np.isfinite(y) & np.isfinite(x)
            if not np.any(good): continue
            
            ax.plot(x[good], y[good], color=v['color'], ls=v['ls'],
                    lw=v['lw'], zorder=v['zorder'], label=v['label'])
            
            if v['key'] == REFERENCE_KEY:
                continue
                
            # Interp reference to current x for valid diff if sizes mismatch
            if len(ref_x) != len(x) or not np.allclose(ref_x, x):
                y_ref_interp = np.interp(x, ref_x, ref_y, left=np.nan, right=np.nan)
            else:
                y_ref_interp = ref_y
                
            delta = y - y_ref_interp
            d_good = np.isfinite(delta) & good
            if np.any(d_good):
                axr.plot(x[d_good], delta[d_good], color=v['color'], ls=v['ls'],
                         lw=v['lw'], zorder=v['zorder'])

        ax.text(0.95, 0.94, p_info['title'], transform=ax.transAxes,
                ha='right', va='top')
        
        # Guide bands and labels
        _residual_guides(axr)
        
        # Styling y-limits dynamically
        if pid in ['himf', 'hiimf']:
            ax.set_ylim(-5.5, -0.7)
            axr.set_ylim(*RESIDUAL_YLIM)
            _format(ax, 1.0, 0.2, 1.0, 0.2, hide_xticklabels=True)
            _format(axr, 1.0, 0.2, 0.5, 0.1)
        elif pid == 'mzr':
            ax.set_ylim(8.0, 9.5)
            axr.set_ylim(-0.5, 0.5)
            _format(ax, 1.0, 0.2, 0.5, 0.1, hide_xticklabels=True)
            _format(axr, 1.0, 0.2, 0.2, 0.1)
        elif pid == 'smd':
            ax.set_xlim(*CSFRD_ZLIM)
            ax.set_ylim(6.0, 9.5)
            axr.set_ylim(*RESIDUAL_YLIM)
            _format(ax, 2.0, 0.5, 1.0, 0.2, hide_xticklabels=True)
            _format(axr, 2.0, 0.5, 0.5, 0.1)
        elif pid == 'qfrac':
            ax.set_ylim(0, 1.05)
            axr.set_ylim(-0.5, 0.5)
            _format(ax, 1.0, 0.2, 0.2, 0.1, hide_xticklabels=True)
            _format(axr, 1.0, 0.2, 0.2, 0.1)
            
        if col == 0:
                    ax.set_ylabel(r'$\log_{10}\ \phi\ [\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1}]$')
                    axr.set_ylabel(r'$\Delta \log_{10}\ \phi$')
        elif col == 3:
            ax.set_ylabel(r'$\log_{10}\ \rho_{*}\ [M_{\odot}\ \mathrm{Mpc}^{-3}]$')
            axr.set_ylabel(r'$\Delta \log_{10}\ \rho_{*}$')
        elif col == 4:
            ax.set_ylabel(r'$\mathrm{Quiescent\ Fraction}$')
            axr.set_ylabel(r'$\Delta \mathrm{Quiescent\ Fraction}$')
        axr.set_xlabel(p_info['xlabel'])

    # Legend on first panel
    first_ax = fig.axes[0]
    handles, labels = first_ax.get_legend_handles_labels()
    model_labels = [v['label'] for v in variants]
    keep = [(h, l) for h, l in zip(handles, labels) if l in model_labels]
    order = {v['label']: i for i, v in enumerate(variants)}
    keep.sort(key=lambda hl: order[hl[1]])
    if keep:
        first_ax.legend([h for h, _ in keep], [l for _, l in keep],
                        loc='lower left', frameon=False, fontsize=12,
                        handlelength=2.6, labelspacing=0.3, borderaxespad=0.8)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, EXTRA_OUTPUT_NAME + pp.OUTPUT_FORMAT)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved Extras: {path}')

    return path

def _residual_guides(ax):
    ax.axhspan(-RESIDUAL_NEGLIGIBLE, RESIDUAL_NEGLIGIBLE,
               color='0.85', alpha=0.6, lw=0, zorder=0)
    ax.axhline(0.0, color='black', lw=1.0, ls='-', alpha=0.6, zorder=1)


def _format(ax, xmaj, xmin, ymaj, ymin, hide_xticklabels=False):
    ax.xaxis.set_major_locator(plt.MultipleLocator(xmaj))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(xmin))
    ax.yaxis.set_major_locator(plt.MultipleLocator(ymaj))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(ymin))
    ax.tick_params(axis='both', which='both', direction='in',
                   top=True, bottom=True, left=True, right=True)
    if hide_xticklabels:
        ax.tick_params(labelbottom=False)

# ========================== TABLES ==========================
# (Tables code has been kept unchanged to provide existing printouts)

def _interp(x, y, x0):
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 2:
        return np.nan
    xs, ys = np.asarray(x)[good], np.asarray(y)[good]
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    if x0 < xs[0] or x0 > xs[-1]:
        return np.nan
    return float(np.interp(x0, xs, ys))

def _cell(value, delta=None, width=15):
    if not np.isfinite(value):
        return f'{"--":>{width}s}'
    if delta is None:
        return f'{value:>{width}.2f}'
    if not np.isfinite(delta):
        return f'{value:>{width}.2f}'
    return f'{f"{value:.2f} ({delta:+.2f})":>{width}s}'

def _largest_deviation(x, delta, reference, xlim=None, floor=None):
    x = np.asarray(x, dtype=float)
    good = np.isfinite(delta) & np.isfinite(x) & np.isfinite(reference)
    if xlim is not None:
        good &= (x >= xlim[0]) & (x <= xlim[1])
    if floor is not None:
        good &= reference >= floor
    if not np.any(good):
        return np.nan, np.nan
    idx = np.nanargmax(np.abs(np.where(good, delta, np.nan)))
    return x[idx], delta[idx]

def write_tables(variants, results, sim, outdir):
    lines = []
    def emit(s=''):
        print(s)
        lines.append(s)

    ref = results[REFERENCE_KEY]
    others = [v for v in variants if v['key'] != REFERENCE_KEY]
    letters = 'abcdefgh'
    summary = {v['key']: [] for v in others}
    floor = density_floor(sim)

    for col, panel in enumerate(SMF_PANELS):
        pid = panel_id(panel)
        ref_m = ref['smf'][pid]
        sel_label = SELECT_LABEL[panel['select']]
        which = 'stellar mass function' if sel_label is None \
            else f'{sel_label} stellar mass function'
        emit()
        emit('=' * 96)
        emit(f'PANEL ({letters[col]})   {which} at z = {ref_m["z"]:.2f}'
             f'  (snapshot {ref_m["snap"]})')
        emit('   log10 phi [Mpc^-3 dex^-1], with (variant - fiducial) in dex')
        emit('=' * 96)
        emit('  ' + f'{"variant":<26s}' +
             ''.join(f'{f"logM*={m:.1f}":>15s}' for m in TABLE_MASSES) +
             f'{"largest offset":>26s}')

        ref_x, ref_phi = ref_m['x'], ref_m['phi']
        if ref_phi is None:
            continue

        emit('  ' + f'{"full (fiducial)":<26s}' +
             ''.join(_cell(_interp(ref_x, ref_phi, m)) for m in TABLE_MASSES))

        for v in others:
            m = results[v['key']]['smf'][pid]
            if m['phi'] is None:
                continue
            delta = m['phi'] - ref_phi
            cells = ''.join(
                _cell(_interp(m['x'], m['phi'], mass),
                      _interp(ref_x, delta, mass)) for mass in TABLE_MASSES)
            at, worst = _largest_deviation(ref_x, delta, ref_phi,
                                           xlim=panel['xlim'],
                                           floor=floor)
            note = ('--' if not np.isfinite(worst)
                    else f'{worst:+.2f} dex at logM*={at:.1f}')
            summary[v['key']].append(note)
            emit('  ' + f'{v["key"]:<26s}' + cells + f'{note:>26s}')

    col = len(SMF_PANELS)
    emit()
    emit('=' * 96)
    emit(f'PANEL ({letters[col]})   cosmic star formation rate density')
    emit('=' * 96)
    emit('  ' + f'{"variant":<26s}' +
         ''.join(f'{f"z={z:.0f}":>15s}' for z in TABLE_REDSHIFTS) +
         f'{"largest offset":>26s}')

    ref_z, ref_rho = ref['z'], ref['csfrd']
    emit('  ' + f'{"full (fiducial)":<26s}' +
         ''.join(_cell(_interp(ref_z, ref_rho, z)) for z in TABLE_REDSHIFTS))

    for v in others:
        r = results[v['key']]
        delta = r['csfrd'] - ref_rho
        cells = ''.join(
            _cell(_interp(r['z'], r['csfrd'], z), _interp(r['z'], delta, z))
            for z in TABLE_REDSHIFTS)
        at, worst = _largest_deviation(r['z'], delta, ref_rho, xlim=CSFRD_ZLIM)
        note = '--' if not np.isfinite(worst) else f'{worst:+.2f} dex at z={at:.1f}'
        summary[v['key']].append(note)
        emit('  ' + f'{v["key"]:<26s}' + cells + f'{note:>26s}')

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, OUTPUT_NAME + '_stats.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n  Saved Stats: {path}')
    return path


# ========================== MAIN ==========================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', action='store_true',
                    help='run SAGE for any variant whose output is missing')
    ap.add_argument('--force', action='store_true',
                    help='with --run, re-run every variant even if output exists')
    ap.add_argument('--with', dest='extra', action='append', default=[],
                    choices=sorted(OPTIONAL_VARIANTS), metavar='NAME',
                    help='also show an optional ablation, repeatable: '
                         + ', '.join(sorted(OPTIONAL_VARIANTS)))
    ap.add_argument('--with-rps', action='store_true',
                    help='shorthand for --with rps')
    ap.add_argument('--no-sage16', action='store_true',
                    help='omit the SAGE16 reference curve')
    ap.add_argument('--outdir', default=None,
                    help='where to write the figure (default: <fiducial>/plots/)')
    args = ap.parse_args()

    variants = [v for v in VARIANTS
                if not (args.no_sage16 and v['key'] == 'sage16')]
    extra = list(args.extra) + (['rps'] if args.with_rps else [])
    for name in dict.fromkeys(extra):
        at = next((i for i, v in enumerate(variants)
                   if v['key'] in (JOINT_KEY, 'sage16')), len(variants))
        variants.insert(at, OPTIONAL_VARIANTS[name])

    if args.run:
        print('Running SAGE:')
        run_variants(variants, force=args.force)

    print('Verifying that each ablation differs by one switch:')
    variants = check_switches(variants)
    ref_variant = next((v for v in variants if v['key'] == REFERENCE_KEY), None)
    if ref_variant is None:
        sys.exit('Fiducial run not found; nothing to compare against.')

    print('Simulation:')
    sim = read_sim(ref_variant['out'])
    if sim is None:
        sys.exit(f'Could not read a simulation header from {ref_variant["out"]}.')
    check_same_simulation(sim, variants)

    print('Measuring:')
    results = measure(variants, sim)

    outdir = args.outdir or os.path.join(ref_variant['out'], 'plots/')
    np.random.seed(pp.SEED)
    pp.setup_style()

    print('Plotting Main Figure:')
    make_figure(variants, results, sim, outdir)
    
    print('Plotting Extra Properties Figure:')
    make_extra_figure(variants, results, sim, outdir)
    
    write_tables(variants, results, sim, outdir)


if __name__ == '__main__':
    main()
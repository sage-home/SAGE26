#!/usr/bin/env python

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from collections import defaultdict
from scipy.stats import gaussian_kde, stats
from random import sample, seed
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import glob

import warnings
warnings.filterwarnings("ignore")

# ========================== USER OPTIONS ==========================
# These are fallback defaults - parameters will be read from HDF5 file

# Plotting options (these are not in HDF5, so keep as user options)
whichimf = 1        # 0=Salpeter; 1=Chabrier
dilute = 7500     # Number of galaxies to plot in scatter plots
sSFRcut = -11.0     # Divide quiescent from star forming galaxies

OutputFormat = '.pdf'
plt.rcParams["figure.figsize"] = (8.34,6.25)
plt.rcParams["figure.dpi"] = 96
plt.rcParams["font.size"] = 14

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['axes.titlecolor'] = 'black'
plt.rcParams['text.color'] = 'black'
plt.rcParams['legend.facecolor'] = 'white'
plt.rcParams['legend.edgecolor'] = 'black'


# ==================================================================

def get_script_dir():
    """Get the directory where this script is located"""
    return os.path.dirname(os.path.abspath(__file__))


def read_simulation_params(filepath):
    """
    Read simulation parameters from HDF5 file header.
    Returns a dictionary with all relevant parameters.
    """
    params = {}

    with h5.File(filepath, 'r') as f:
        # Read from Header/Simulation
        sim = f['Header/Simulation']
        params['Hubble_h'] = float(sim.attrs['hubble_h'])
        params['BoxSize'] = float(sim.attrs['box_size'])
        params['Omega'] = float(sim.attrs['omega_matter'])
        params['OmegaLambda'] = float(sim.attrs['omega_lambda'])
        params['PartMass'] = float(sim.attrs['particle_mass'])

        # Read from Header/Runtime
        runtime = f['Header/Runtime']
        params['VolumeFraction'] = float(runtime.attrs['frac_volume_processed'])
        params['SFprescription'] = int(runtime.attrs['SFprescription'])

        # Read snapshot info
        params['snapshot_redshifts'] = np.array(f['Header/snapshot_redshifts'])
        params['output_snapshots'] = np.array(f['Header/output_snapshots'])

        # Find available snapshot groups in the file
        snap_groups = [key for key in f.keys() if key.startswith('Snap_')]
        snap_numbers = sorted([int(s.replace('Snap_', '')) for s in snap_groups])
        params['available_snapshots'] = snap_numbers
        params['latest_snapshot'] = max(snap_numbers) if snap_numbers else None

    return params


def get_snapshot_redshift(params, snap_num):
    """Get the redshift for a given snapshot number"""
    if snap_num < len(params['snapshot_redshifts']):
        return params['snapshot_redshifts'][snap_num]
    return None


def read_hdf(filepaths, snap_num, param):
    """Read and concatenate a parameter from multiple HDF5 files for a given snapshot"""
    data_list = []
    for filepath in filepaths:
        with h5.File(filepath, 'r') as f:
            if snap_num in f and param in f[snap_num]:
                data = np.array(f[snap_num][param])
                # Only append if the array has data
                if data.size > 0:
                    data_list.append(data)
    
    if not data_list:
        return np.array([])
    
    return np.concatenate(data_list)

def read_obs_data(obs_dir, filename):
    """Read observational data files"""
    filepath = os.path.join(obs_dir, filename)
    if not os.path.exists(filepath):
        print(f"  Warning: Observational data file {filename} not found at {filepath}")
        return None

    data = np.loadtxt(filepath)
    return data


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Plot SAGE26 galaxy model results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s output/millennium/model_0.hdf5
  %(prog)s output/millennium/model_0.hdf5 --snapshot 63
  %(prog)s output/millennium/model_0.hdf5 -s 58
        """
    )

    parser.add_argument('input_pattern', nargs='?',
                        default='./output/millennium/model_*.hdf5',
                        help='Path pattern to model HDF5 files (default: ./output/millennium/model_*.hdf5)')

    parser.add_argument('-s', '--snapshot', type=int, default=None,
                        help='Snapshot number to plot (default: latest available)')

    parser.add_argument('-o', '--output-dir', type=str, default=None,
                        help='Output directory for plots (default: <input_dir>/plots/)')

    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing observational data (default: <script_dir>/../data/)')

    return parser.parse_args()


# ==================================================================

if __name__ == '__main__':

    print('Running allresults (local)\n')

    # Parse command-line arguments
    args = parse_arguments()

    # Determine paths and find files
    script_dir = get_script_dir()
    
    # Use glob to find all files matching the pattern
    file_list = glob.glob(args.input_pattern)
    file_list.sort() # Ensure consistent ordering

    if not file_list:
        print(f"Error: No files found matching: {args.input_pattern}")
        sys.exit(1)

    print(f"Found {len(file_list)} model files.")
    
    # Use the first file to set directories and read global parameters
    first_file = os.path.abspath(file_list[0])
    input_dir = os.path.dirname(first_file)

    # Read simulation parameters from the first HDF5 header
    print(f'Reading simulation parameters from {first_file}')
    sim_params = read_simulation_params(first_file)

    # Calculate the total volume fraction across ALL files
    total_volume_fraction = 0.0
    for f in file_list:
        p = read_simulation_params(f)
        total_volume_fraction += p['VolumeFraction']
    
    sim_params['VolumeFraction'] = total_volume_fraction

    Hubble_h = sim_params['Hubble_h']
    BoxSize = sim_params['BoxSize']
    VolumeFraction = sim_params['VolumeFraction']

    print(f'  Hubble_h = {Hubble_h}')
    print(f'  BoxSize = {BoxSize} h^-1 Mpc')
    print(f'  Total VolumeFraction = {VolumeFraction}')

    # Determine snapshot to use (assuming all files have the same snapshots)
    if args.snapshot is not None:
        snap_num = args.snapshot
        if snap_num not in sim_params['available_snapshots']:
            print(f"Error: Snapshot {snap_num} not available in file.")
            sys.exit(1)
    else:
        snap_num = sim_params['latest_snapshot']
        print(f'  Using latest snapshot: {snap_num}')

    Snapshot = f'Snap_{snap_num}'
    redshift = get_snapshot_redshift(sim_params, snap_num)
    if redshift is not None:
        print(f'  Redshift = {redshift:.4f}')

    # Set up directories
    if args.output_dir:
        OutputDir = args.output_dir
    else:
        OutputDir = os.path.join(input_dir, 'plots')
    if not OutputDir.endswith(os.sep):
        OutputDir += os.sep

    if args.data_dir:
        DataDir = args.data_dir
    else:
        DataDir = os.path.join(script_dir, '..', 'data')

    if not os.path.exists(OutputDir):
        os.makedirs(OutputDir)

    seed(2222)
    volume = (BoxSize / Hubble_h)**3.0 * VolumeFraction

    # Read galaxy properties across ALL files
    print(f'Reading galaxy properties from {len(file_list)} files for {Snapshot}...')

    CentralMvir = read_hdf(file_list, Snapshot, 'CentralMvir') * 1.0e10 / Hubble_h
    Mvir = read_hdf(file_list, Snapshot, 'Mvir') * 1.0e10 / Hubble_h
    StellarMass = read_hdf(file_list, Snapshot, 'StellarMass') * 1.0e10 / Hubble_h
    MetalsStellarMass = read_hdf(file_list, Snapshot, 'MetalsStellarMass') * 1.0e10 / Hubble_h
    BulgeMass = read_hdf(file_list, Snapshot, 'BulgeMass') * 1.0e10 / Hubble_h
    BlackHoleMass = read_hdf(file_list, Snapshot, 'BlackHoleMass') * 1.0e10 / Hubble_h
    ColdGas = read_hdf(file_list, Snapshot, 'ColdGas') * 1.0e10 / Hubble_h
    MetalsColdGas = read_hdf(file_list, Snapshot, 'MetalsColdGas') * 1.0e10 / Hubble_h
    MetalsEjectedMass = read_hdf(file_list, Snapshot, 'MetalsEjectedMass') * 1.0e10 / Hubble_h
    HotGas = read_hdf(file_list, Snapshot, 'HotGas') * 1.0e10 / Hubble_h
    MetalsHotGas = read_hdf(file_list, Snapshot, 'MetalsHotGas') * 1.0e10 / Hubble_h
    EjectedMass = read_hdf(file_list, Snapshot, 'EjectedMass') * 1.0e10 / Hubble_h
    CGMgas = read_hdf(file_list, Snapshot, 'CGMgas') * 1.0e10 / Hubble_h
    MetalsCGMgas = read_hdf(file_list, Snapshot, 'MetalsCGMgas') * 1.0e10 / Hubble_h

    IntraClusterStars = read_hdf(file_list, Snapshot, 'IntraClusterStars') * 1.0e10 / Hubble_h
    DiskRadius = read_hdf(file_list, Snapshot, 'DiskRadius')
    BulgeRadius = read_hdf(file_list, Snapshot, 'BulgeRadius')
    MergerBulgeRadius = read_hdf(file_list, Snapshot, 'MergerBulgeRadius')
    InstabilityBulgeRadius = read_hdf(file_list, Snapshot, 'InstabilityBulgeRadius')
    MergerBulgeMass = read_hdf(file_list, Snapshot, 'MergerBulgeMass') * 1.0e10 / Hubble_h
    InstabilityBulgeMass = read_hdf(file_list, Snapshot, 'InstabilityBulgeMass') * 1.0e10 / Hubble_h

    H2gas = read_hdf(file_list, Snapshot, 'H2gas') * 1.0e10 / Hubble_h
    H1gas = read_hdf(file_list, Snapshot, 'H1gas') * 1.0e10 / Hubble_h
    Vvir = read_hdf(file_list, Snapshot, 'Vvir')
    Vmax = read_hdf(file_list, Snapshot, 'Vmax')
    Rvir = read_hdf(file_list, Snapshot, 'Rvir')
    SfrDisk = read_hdf(file_list, Snapshot, 'SfrDisk')
    SfrBulge = read_hdf(file_list, Snapshot, 'SfrBulge')

    CentralGalaxyIndex = read_hdf(file_list, Snapshot, 'CentralGalaxyIndex')
    Type = read_hdf(file_list, Snapshot, 'Type')
    Posx = read_hdf(file_list, Snapshot, 'Posx')
    Posy = read_hdf(file_list, Snapshot, 'Posy')
    Posz = read_hdf(file_list, Snapshot, 'Posz')

    OutflowRate = read_hdf(file_list, Snapshot, 'OutflowRate')
    MassLoading = read_hdf(file_list, Snapshot, 'MassLoading')
    Cooling = read_hdf(file_list, Snapshot, 'Cooling')
    Regime = read_hdf(file_list, Snapshot, 'Regime')

    w = np.where(StellarMass > 1.0e10)[0]
    print('Number of galaxies read:', len(StellarMass))
    print('Galaxies more massive than 10^10 h-1 Msun:', len(w), '\n')

    Tvir = 35.9 * (Vvir)**2  # in Kelvin
    Tmax = 2.5e5  # K, corresponds to Vvir ~52.7 km/s

    QuasarAccretionMass = read_hdf(file_list, Snapshot, 'QuasarModeBHaccretionMass') * 1.0e10 / Hubble_h

# --------------------------------------------------------

    print('Plotting the stellar mass function, divided by sSFR')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure

    binwidth = 0.1  # mass function histogram bin width

    # Load GAMA morphological SMF data
    # Columns: log_M, E_HE, E_HE_err, cBD, cBD_err, dBD, dBD_err, D, D_err
    gama = np.genfromtxt(os.path.join(DataDir, 'gama_smf_morph.ecsv'), comments='#', skip_header=1)
    gama_mass = gama[:, 0]
    gama_E_HE = gama[:, 1]
    gama_E_HE_err = gama[:, 2]
    gama_D = gama[:, 7]
    gama_D_err = gama[:, 8]

    # Load Baldry et al. blue/red SMF data
    # Columns: SF_mass, SF_phi, Q_mass, Q_phi (all in log)
    baldry = np.genfromtxt(os.path.join(DataDir, 'baldry_blue_red.csv'), delimiter=',', skip_header=2)
    baldry_sf_mass = baldry[:, 0]
    baldry_sf_phi = baldry[:, 1]
    baldry_q_mass = baldry[:, 2]
    baldry_q_phi = baldry[:, 3]

    # calculate all
    w = np.where(StellarMass > 0.0)[0]
    mass = np.log10(StellarMass[w])
    sSFR = np.log10( (SfrDisk[w] + SfrBulge[w]) / StellarMass[w] )

    # Bin parameters for original model
    mi = np.floor(min(mass)) - 2
    ma = np.floor(max(mass)) + 2
    NB = int((ma - mi) / binwidth)
    (counts, binedges) = np.histogram(mass, range=(mi, ma), bins=NB)
    xaxeshisto = binedges[:-1] + 0.5 * binwidth  # Set the x-axis values to be the centre of the bins

    # additionally calculate red for original model
    w = np.where(sSFR < sSFRcut)[0]
    massRED = mass[w]
    (countsRED, binedges) = np.histogram(massRED, range=(mi, ma), bins=NB)

    # additionally calculate blue for original model
    w = np.where(sSFR > sSFRcut)[0]
    massBLU = mass[w]
    (countsBLU, binedges) = np.histogram(massBLU, range=(mi, ma), bins=NB)


    # Overplot the model histograms (in log10 space)
    # plt.plot(xaxeshisto, np.log10(counts / volume / binwidth), 'k-', label='SAGE26')
    plt.plot(xaxeshisto, np.log10(counts / volume / binwidth), color='k', lw=4, label='SAGE26 Total')
    plt.plot(xaxeshisto, np.log10(countsRED / volume / binwidth), color='firebrick', lw=4, label='SAGE26 Quiescent')
    plt.plot(xaxeshisto, np.log10(countsBLU / volume / binwidth), color='dodgerblue', lw=4, label='SAGE26 Star Forming')


    # Create shaded regions from observations (GAMA + Baldry combined)
    from scipy import interpolate

    # Common mass grid for interpolation
    mass_grid = np.linspace(8, 12, 100)

    # Star-forming: combine GAMA D and Baldry SF
    valid_D = ~np.isnan(gama_D)
    gama_sf_interp = interpolate.interp1d(gama_mass[valid_D], gama_D[valid_D],
                                           bounds_error=False, fill_value=np.nan)
    baldry_sf_interp = interpolate.interp1d(baldry_sf_mass, baldry_sf_phi,
                                             bounds_error=False, fill_value=np.nan)
    sf_gama = gama_sf_interp(mass_grid)
    sf_baldry = baldry_sf_interp(mass_grid)
    sf_lower = np.nanmin([sf_gama, sf_baldry], axis=0)
    sf_upper = np.nanmax([sf_gama, sf_baldry], axis=0)

    # Quiescent: combine GAMA E+HE and Baldry Q
    valid_E = ~np.isnan(gama_E_HE)
    gama_q_interp = interpolate.interp1d(gama_mass[valid_E], gama_E_HE[valid_E],
                                          bounds_error=False, fill_value=np.nan)
    baldry_q_interp = interpolate.interp1d(baldry_q_mass, baldry_q_phi,
                                            bounds_error=False, fill_value=np.nan)
    q_gama = gama_q_interp(mass_grid)
    q_baldry = baldry_q_interp(mass_grid)
    q_lower = np.nanmin([q_gama, q_baldry], axis=0)
    q_upper = np.nanmax([q_gama, q_baldry], axis=0)

    # Plot shaded regions
    plt.fill_between(mass_grid, sf_lower, sf_upper, color='dodgerblue', alpha=0.3, edgecolor='none', label='Observations SF')
    plt.fill_between(mass_grid, q_lower, q_upper, color='firebrick', alpha=0.3, edgecolor='none', label='Observations Q')

    plt.axis([8, 12, -6, -1])
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))

    plt.ylabel(r'$\log_{10}\ \phi\ (\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1})$')
    plt.xlabel(r'$\log_{10} M_{\mathrm{*}}\ (M_{\odot})$')

    leg = plt.legend(loc='lower left', numpoints=1, labelspacing=0.1)
    leg.draw_frame(False)  # Don't want a box frame
    for t in leg.get_texts():  # Reduce the size of the text
        t.set_fontsize('medium')

    plt.tight_layout()

    outputFile = OutputDir + 'StellarMassFunction' + OutputFormat
    plt.savefig(outputFile)  # Save the figure
    print('Saved to', outputFile, '\n')
    plt.close()

# ---------------------------------------------------------

    print('Plotting the baryonic mass function')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure

    binwidth = 0.1  # mass function histogram bin width
  
    # calculate BMF
    w = np.where(StellarMass + ColdGas > 0.0)[0]
    mass = np.log10( (StellarMass[w] + ColdGas[w]) )

    mi = np.floor(min(mass)) - 2
    ma = np.floor(max(mass)) + 2
    NB = int((ma - mi) / binwidth)
    (counts, binedges) = np.histogram(mass, range=(mi, ma), bins=NB)
    xaxeshisto = binedges[:-1] + 0.5 * binwidth  # Set the x-axis values to be the centre of the bins

    centrals = np.where(Type[w] == 0)[0]
    satellites = np.where(Type[w] == 1)[0]

    centrals_mass = mass[centrals]
    satellites_mass = mass[satellites]

    mi = np.floor(min(centrals_mass)) - 2
    ma = np.floor(max(centrals_mass)) + 2
    NB = int((ma - mi) / binwidth)
    (counts_centrals, binedges_centrals) = np.histogram(centrals_mass, range=(mi, ma), bins=NB)
    xaxeshisto_centrals = binedges_centrals[:-1] + 0.5 * binwidth  # Set the x-axis values to be the centre of the bins

    mi = np.floor(min(satellites_mass)) - 2
    ma = np.floor(max(satellites_mass)) + 2
    NB = int((ma - mi) / binwidth)
    (counts_satellites, binedges_satellites) = np.histogram(satellites_mass, range=(mi, ma), bins=NB)
    xaxeshisto_satellites = binedges_satellites[:-1] + 0.5 * binwidth  # Set the x-axis values to be the centre of the bins

    # Bell et al. 2003 BMF (h=1.0 converted to h=0.73)
    M = np.arange(7.0, 13.0, 0.01)
    Mstar = np.log10(5.3*1.0e10 /Hubble_h/Hubble_h)
    alpha = -1.21
    phistar = 0.0108 *Hubble_h*Hubble_h*Hubble_h
    xval = 10.0 ** (M-Mstar)
    yval = np.log(10.) * phistar * xval ** (alpha+1) * np.exp(-xval)
    
    if(whichimf == 0):
        # converted diet Salpeter IMF to Salpeter IMF
        plt.plot(np.log10(10.0**M /0.7), yval, 'b-', lw=2.0, label='Bell et al. 2003')  # Plot the SMF
    elif(whichimf == 1):
        # converted diet Salpeter IMF to Salpeter IMF, then to Chabrier IMF
        plt.plot(np.log10(10.0**M /0.7 /1.8), yval, 'g--', lw=1.5, label='Bell et al. 2003')  # Plot the SMF

    # Overplot the model histograms
    plt.plot(xaxeshisto, counts / volume / binwidth, 'k-', label='Model')
    plt.plot(xaxeshisto_centrals, counts_centrals / volume / binwidth, 'b:', lw=2, label='Model - Centrals')
    plt.plot(xaxeshisto_satellites, counts_satellites / volume / binwidth, 'g--', lw=1.5, label='Model - Satellites')

    plt.yscale('log')
    plt.axis([8.0, 12.2, 1.0e-6, 1.0e-1])
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))

    plt.ylabel(r'$\phi\ (\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1})$')  # Set the y...
    plt.xlabel(r'$\log_{10}\ M_{\mathrm{bar}}\ (M_{\odot})$')  # and the x-axis labels

    leg = plt.legend(loc='lower left', numpoints=1, labelspacing=0.1)
    leg.draw_frame(False)  # Don't want a box frame
    for t in leg.get_texts():  # Reduce the size of the text
        t.set_fontsize('medium')
    
    plt.tight_layout()

    outputFile = OutputDir + 'BaryonicMassFunction' + OutputFormat
    plt.savefig(outputFile)  # Save the figure
    print('Saved file to', outputFile, '\n')
    plt.close()

# ---------------------------------------------------------

    print('Plotting the cold gas mass function')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure

    binwidth = 0.1  # mass function histogram bin width

    # calculate all
    w = np.where((ColdGas > 0.0) & (Type==0))[0]
    mass = np.log10(ColdGas[w])
    H2mass = np.log10(H2gas[w])
    H1mass = np.log10(H1gas[w])  # Now read directly from model output
    sSFR = (SfrDisk[w] + SfrBulge[w]) / StellarMass[w]

    mi = np.floor(min(mass)) - 2
    ma = np.floor(max(mass)) + 2
    NB = int((ma - mi) / binwidth)

    (counts, binedges) = np.histogram(mass, range=(mi, ma), bins=NB)
    xaxeshisto = binedges[:-1] + 0.5 * binwidth  # Set the x-axis values to be the centre of the bins

    (counts_h2, binedges_h2) = np.histogram(H2mass, range=(mi, ma), bins=NB)
    xaxeshisto_h2 = binedges_h2[:-1] + 0.5 * binwidth  # Set the x-axis values to be the centre of the bins

    (counts_h1, binedges_h1) = np.histogram(H1mass, range=(mi, ma), bins=NB)
    xaxeshisto_h1 = binedges_h1[:-1] + 0.5 * binwidth  # Set the x-axis values to be the centre of the bins

    # additionally calculate red
    w = np.where(sSFR < sSFRcut)[0]
    massRED = mass[w]
    (countsRED, binedges) = np.histogram(massRED, range=(mi, ma), bins=NB)

    # additionally calculate blue
    w = np.where(sSFR > sSFRcut)[0]
    massBLU = mass[w]
    (countsBLU, binedges) = np.histogram(massBLU, range=(mi, ma), bins=NB)

    # Baldry+ 2008 modified data used for the MCMC fitting
    Zwaan = np.array([[6.933,   -0.333],
        [7.057,   -0.490],
        [7.209,   -0.698],
        [7.365,   -0.667],
        [7.528,   -0.823],
        [7.647,   -0.958],
        [7.809,   -0.917],
        [7.971,   -0.948],
        [8.112,   -0.927],
        [8.263,   -0.917],
        [8.404,   -1.062],
        [8.566,   -1.177],
        [8.707,   -1.177],
        [8.853,   -1.312],
        [9.010,   -1.344],
        [9.161,   -1.448],
        [9.302,   -1.604],
        [9.448,   -1.792],
        [9.599,   -2.021],
        [9.740,   -2.406],
        [9.897,   -2.615],
        [10.053,  -3.031],
        [10.178,  -3.677],
        [10.335,  -4.448],
        [10.492,  -5.083]        ], dtype=np.float32)
    
    ObrRaw = np.array([
        [7.300,   -1.104],
        [7.576,   -1.302],
        [7.847,   -1.250],
        [8.133,   -1.240],
        [8.409,   -1.344],
        [8.691,   -1.479],
        [8.956,   -1.792],
        [9.231,   -2.271],
        [9.507,   -3.198],
        [9.788,   -5.062 ]        ], dtype=np.float32)
    ObrCold = np.array([
        [8.009,   -1.042],
        [8.215,   -1.156],
        [8.409,   -0.990],
        [8.604,   -1.156],
        [8.799,   -1.208],
        [9.020,   -1.333],
        [9.194,   -1.385],
        [9.404,   -1.552],
        [9.599,   -1.677],
        [9.788,   -1.812],
        [9.999,   -2.312],
        [10.172,  -2.656],
        [10.362,  -3.500],
        [10.551,  -3.635],
        [10.740,  -5.010]        ], dtype=np.float32)
    
    ObrCold_xval = np.log10(10**(ObrCold[:, 0])  /Hubble_h/Hubble_h)
    ObrCold_yval = (10**(ObrCold[:, 1]) * Hubble_h*Hubble_h*Hubble_h)
    Zwaan_xval = np.log10(10**(Zwaan[:, 0]) /Hubble_h/Hubble_h)
    Zwaan_yval = (10**(Zwaan[:, 1]) * Hubble_h*Hubble_h*Hubble_h)
    ObrRaw_xval = np.log10(10**(ObrRaw[:, 0])  /Hubble_h/Hubble_h)
    ObrRaw_yval = (10**(ObrRaw[:, 1]) * Hubble_h*Hubble_h*Hubble_h)

    plt.plot(ObrCold_xval, ObrCold_yval, color='k', lw = 7, alpha=0.25, label='Obr. & Raw. 2009 (Cold Gas)')
    plt.plot(Zwaan_xval, Zwaan_yval, color='cyan', lw = 7, alpha=0.25, label='Zwaan et al. 2005 (HI)')
    plt.plot(ObrRaw_xval, ObrRaw_yval, color='magenta', lw = 7, alpha=0.25, label='Obr. & Raw. 2009 (H2)')

    plt.plot(xaxeshisto_h2, counts_h2 / volume / binwidth, 'magenta', linestyle='-', label='Model - H2 Gas')
    plt.plot(xaxeshisto_h1, counts_h1 / volume / binwidth, 'cyan', linestyle='-', label='Model - HI Gas')
    
    # Overplot the model histograms
    plt.plot(xaxeshisto, counts / volume / binwidth, 'k-', label='Model - Cold Gas')

    plt.yscale('log')
    plt.axis([8.0, 11.5, 1.0e-6, 1.0e-1])
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))

    plt.ylabel(r'$\phi\ (\mathrm{Mpc}^{-3}\ \mathrm{dex}^{-1})$')  # Set the y...
    plt.xlabel(r'$\log_{10} M_{\mathrm{X}}\ (M_{\odot})$')  # and the x-axis labels

    leg = plt.legend(loc='lower left', numpoints=1, labelspacing=0.1)
    leg.draw_frame(False)  # Don't want a box frame
    for t in leg.get_texts():  # Reduce the size of the text
        t.set_fontsize('medium')

    plt.tight_layout()

    outputFile = OutputDir + 'GasMassFunction' + OutputFormat
    plt.savefig(outputFile)  # Save the figure
    print('Saved file to', outputFile, '\n')
    plt.close()

# ---------------------------------------------------------

    print('Plotting the baryonic TF relationship')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure

    w = np.where((Type == 0) & (StellarMass + ColdGas > 0.0) & 
      (BulgeMass / StellarMass > 0.1) & (BulgeMass / StellarMass < 0.5))[0]
    if(len(w) > dilute): w = sample(list(range(len(w))), dilute)

    mass = np.log10( (StellarMass[w] + ColdGas[w]) )
    vel = np.log10(Vmax[w])
                
    plt.scatter(vel, mass, marker='x', s=1, c='k', alpha=0.9, label='Model Sb/c galaxies')
            
    # overplot Stark, McGaugh & Swatters 2009 (assumes h=0.75? ... what IMF?)
    w = np.arange(0.5, 10.0, 0.5)
    TF = 3.94*w + 1.79
    TF_upper = TF + 0.26
    TF_lower = TF - 0.26

    # plt.plot(w, TF, 'b-', alpha=0.5, label='Stark, McGaugh & Swatters 2009')
    plt.fill_between(w, TF_lower, TF_upper, color='blue', alpha=0.2)

        
    plt.ylabel(r'$\log_{10}\ M_{\mathrm{bar}}\ (M_{\odot})$')  # Set the y...
    plt.xlabel(r'$\log_{10}V_{max}\ (km/s)$')  # and the x-axis labels
        
    # Set the x and y axis minor ticks
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))
        
    plt.axis([1.4, 2.9, 7.5, 12.0])
        
    leg = plt.legend(loc='lower right')
    leg.draw_frame(False)  # Don't want a box frame
    for t in leg.get_texts():  # Reduce the size of the text
        t.set_fontsize('medium')

    plt.tight_layout()
        
    outputFile = OutputDir + 'BaryonicTullyFisher' + OutputFormat
    plt.savefig(outputFile)  # Save the figure
    print('Saved file to', outputFile, '\n')
    plt.close()

# ---------------------------------------------------------

    print('Plotting the specific sSFR')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure

    w = np.where(StellarMass > 0.01)[0]
    if(len(w) > dilute): w = sample(list(w), dilute)
    mass = np.log10(StellarMass[w])
    sSFR = np.log10( (SfrDisk[w] + SfrBulge[w]) / StellarMass[w] )
    plt.scatter(mass, sSFR, marker='x', s=1, c='k', alpha=0.9, label='Model galaxies')

    # overplot dividing line between SF and passive
    w = np.arange(7.0, 13.0, 1.0)
    plt.plot(w, w/w*sSFRcut, 'b:', lw=2.0)

    plt.ylabel(r'$\log_{10}\ s\mathrm{SFR}\ (\mathrm{yr^{-1}})$')  # Set the y...
    plt.xlabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')  # and the x-axis labels

    # Set the x and y axis minor ticks
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))

    plt.axis([8.0, 12.0, -16.0, -8.0])

    leg = plt.legend(loc='lower right')
    leg.draw_frame(False)  # Don't want a box frame
    for t in leg.get_texts():  # Reduce the size of the text
        t.set_fontsize('medium')

    plt.tight_layout()

    outputFile = OutputDir + 'SpecificStarFormationRate' + OutputFormat
    plt.savefig(outputFile)  # Save the figure
    print('Saved to', outputFile, '\n')
    plt.close()

# ---------------------------------------------------------

    print('Plotting the gas fractions')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure

    w = np.where((Type == 0) & (StellarMass + ColdGas > 0.0) & 
      (BulgeMass / StellarMass > 0.1) & (BulgeMass / StellarMass < 0.5))[0]
    if(len(w) > dilute): w = sample(list(w), dilute)
    
    mass = np.log10(StellarMass[w])
    fraction = ColdGas[w] / (StellarMass[w] + ColdGas[w])

    plt.scatter(mass, fraction, marker='x', s=1, c='k', alpha=0.9, label='Model Sb/c galaxies')
        
    plt.ylabel(r'$\mathrm{Cold\ Mass\ /\ (Cold+Stellar\ Mass)}$')  # Set the y...
    plt.xlabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')  # and the x-axis labels
        
    # Set the x and y axis minor ticks
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))
        
    plt.axis([8.0, 12.0, 0.0, 1.0])
        
    leg = plt.legend(loc='upper right')
    leg.draw_frame(False)  # Don't want a box frame
    for t in leg.get_texts():  # Reduce the size of the text
        t.set_fontsize('medium')

    plt.tight_layout()
        
    outputFile = OutputDir + 'GasFraction' + OutputFormat
    plt.savefig(outputFile)  # Save the figure
    print('Saved file to', outputFile, '\n')
    plt.close()

# -------------------------------------------------------

    print('Plotting the metallicities')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure

    w = np.where((Type == 0) & (ColdGas / (StellarMass + ColdGas) > 0.1) & (StellarMass > 1.0e8))[0]
    if(len(w) > dilute): w = sample(list(w), dilute)
    
    mass = np.log10(StellarMass[w])
    Z = np.log10((MetalsColdGas[w]) / ColdGas[w] / 0.02) + 9.0  # Convert to 12 + log(O/H) scale, assuming solar metallicity of 0.02 and that all metals are oxygen (for simplicity)    
    plt.scatter(mass, Z, marker='x', s=1, c='gray', alpha=0.9, label='Model galaxies')

    # Tremonti et al. 2004 - the primary observational reference
    try:
        tremonti_data = np.loadtxt(os.path.join(DataDir, 'Tremonti04.dat'))
        tremonti_mass = tremonti_data[:, 0]
        tremonti_Z = tremonti_data[:, 1]
        tremonti_Z_err_low = tremonti_data[:, 2]
        tremonti_Z_err_high = tremonti_data[:, 3]
        # Convert IMF if needed
        if whichimf == 0:
            tremonti_mass_corrected = np.log10(10**tremonti_mass * 1.5)
        elif whichimf == 1:
            tremonti_mass_corrected = np.log10(10**tremonti_mass * 1.5 / 1.8)
        else:
            tremonti_mass_corrected = tremonti_mass
        # Plot main line
        ax.plot(tremonti_mass_corrected, tremonti_Z, '-', color='red', linewidth=1.5, alpha=0.7, label='Tremonti+04')
        # Plot error shading
        ax.fill_between(tremonti_mass_corrected, tremonti_Z_err_low, tremonti_Z_err_high, color='firebrick', alpha=0.2, zorder=5)
    except Exception as e:
        print(f"Warning: Could not load Tremonti04.dat: {e}. Using fallback polynomial fit.")
        w_obs = np.arange(7.0, 13.0, 0.1)
        Zobs = -1.492 + 1.847*w_obs - 0.08026*w_obs*w_obs
        if whichimf == 0:
            ax.plot(np.log10((10**w_obs * 1.5)), Zobs, 'o-', linewidth=3, label='Tremonti et al. 2004 (poly fit)')
        elif whichimf == 1:
            ax.plot(np.log10((10**w_obs * 1.5 / 1.8)), Zobs, 'o-', linewidth=3, label='Tremonti et al. 2004 (poly fit)')
        else:
            ax.plot(w_obs, Zobs, 'o-', linewidth=3, label='Tremonti et al. 2004 (poly fit)')
    
    # Curti et al. 2020
    try:
        curti_data = np.loadtxt(os.path.join(DataDir, 'Curti2020.dat'))
        curti_mass = curti_data[:, 0]
        curti_Z = curti_data[:, 1]
        curti_Z_low = curti_data[:, 2]
        curti_Z_high = curti_data[:, 3]
        # Plot main line
        ax.plot(curti_mass, curti_Z, linestyle='-', color='blue', linewidth=2, label='Curti+20')
        # Plot error shading
        ax.fill_between(curti_mass, curti_Z_low, curti_Z_high, color='darkblue', alpha=0.2, zorder=5)
    except Exception as e:
        print(f"Warning: Could not load Curti2020.dat: {e}")
    
    # Andrews & Martini 2013
    try:
        andrews_data = np.loadtxt(os.path.join(DataDir, 'MMAdrews13.dat'))
        andrews_mass = andrews_data[:, 0]
        andrews_Z = andrews_data[:, 1]
        if whichimf == 0:
            andrews_mass_corrected = np.log10(10**andrews_mass * 1.5)
        elif whichimf == 1:
            andrews_mass_corrected = np.log10(10**andrews_mass * 1.5 / 1.8)
        else:
            andrews_mass_corrected = andrews_mass
        ax.scatter(andrews_mass_corrected, andrews_Z, marker='s', s=30, color='green', edgecolors='darkgreen', linewidth=0.5, alpha=0.8, label='Andrews & Martini 2013')
    except Exception as e:
        print(f"Warning: Could not load MMAdrews13.dat: {e}")
    
    # Kewley & Ellison 2008 - T04 calibration (most commonly used)
    try:
        kewley_data = np.loadtxt(os.path.join(DataDir, 'MMR-Kewley08.dat'))
        t04_start = 59
        t04_end = 74
        kewley_mass_t04 = kewley_data[t04_start:t04_end, 0]
        kewley_Z_t04 = kewley_data[t04_start:t04_end, 1]
        if whichimf == 0:
            kewley_mass_corrected = np.log10(10**kewley_mass_t04 * 1.5)
        elif whichimf == 1:
            kewley_mass_corrected = np.log10(10**kewley_mass_t04 * 1.5 / 1.8)
        else:
            kewley_mass_corrected = kewley_mass_t04
        ax.scatter(kewley_mass_corrected, kewley_Z_t04, marker='D', s=40, color='yellow', edgecolors='goldenrod', linewidth=0.8, alpha=0.8, label='Kewley & Ellison 2008')
    except Exception as e:
        print(f"Warning: Could not load MMR-Kewley08.dat: {e}")
    
    # Gallazzi et al. 2005 - Stellar metallicity (note: this is different from gas metallicity)
    try:
        gallazzi_data = np.loadtxt(os.path.join(DataDir, 'MSZR-Gallazzi05.dat'))
        gallazzi_mass = gallazzi_data[7:, 0]
        gallazzi_Z_stellar = gallazzi_data[7:, 1]
        gallazzi_Z_gas_approx = gallazzi_Z_stellar + 8.69
        gallazzi_mass_corrected = gallazzi_mass
        ax.scatter(gallazzi_mass_corrected, gallazzi_Z_gas_approx, marker='P', s=100, color='k', edgecolors='gray', linewidth=0.5, alpha=0.8, label='Gallazzi+05')
    except Exception as e:
        print(f"Warning: Could not load MSZR-Gallazzi05.dat: {e}")
        
    plt.ylabel(r'$12\ +\ \log_{10}(\mathrm{O/H})$')  # Set the y...
    plt.xlabel(r'$\log_{10} M_{\mathrm{*}}\ (M_{\odot})$')  # and the x-axis labels
        
    # Set the x and y axis minor ticks
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
        
    plt.axis([8.0, 12.0, 8.0, 9.5])
        
    leg = plt.legend(loc='lower right')
    leg.draw_frame(False)  # Don't want a box frame
    for t in leg.get_texts():  # Reduce the size of the text
        t.set_fontsize('medium')

    plt.tight_layout()
        
    outputFile = OutputDir + 'Metallicity' + OutputFormat
    plt.savefig(outputFile)  # Save the figure
    print('Saved file to', outputFile, '\n')
    plt.close()

# -------------------------------------------------------

    print('Plotting the black hole-bulge relationship')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure

    w = np.where((BulgeMass > 1.0e8) & (BlackHoleMass > 01.0e6))[0]
    if(len(w) > dilute): w = sample(list(w), dilute)

    bh = np.log10(BlackHoleMass[w])
    bulge = np.log10(BulgeMass[w])
                
    plt.scatter(bulge, bh, marker='x', s=1, c='k', alpha=0.9, label='Model galaxies', zorder=10)
            
    # overplot Haring & Rix 2004
    w = 10. ** np.arange(20)
    BHdata = 10. ** (8.2 + 1.12 * np.log10(w / 1.0e11))
    plt.plot(np.log10(w), np.log10(BHdata), 'b-', label=r"Haring \& Rix 2004")

    # Observational points
    M_BH_obs = (0.7/Hubble_h)**2*1e8*np.array([39, 11, 0.45, 25, 24, 0.044, 1.4, 0.73, 9.0, 58, 0.10, 8.3, 0.39, 0.42, 0.084, 0.66, 0.73, 15, 4.7, 0.083, 0.14, 0.15, 0.4, 0.12, 1.7, 0.024, 8.8, 0.14, 2.0, 0.073, 0.77, 4.0, 0.17, 0.34, 2.4, 0.058, 3.1, 1.3, 2.0, 97, 8.1, 1.8, 0.65, 0.39, 5.0, 3.3, 4.5, 0.075, 0.68, 1.2, 0.13, 4.7, 0.59, 6.4, 0.79, 3.9, 47, 1.8, 0.06, 0.016, 210, 0.014, 7.4, 1.6, 6.8, 2.6, 11, 37, 5.9, 0.31, 0.10, 3.7, 0.55, 13, 0.11])
    M_BH_hi = (0.7/Hubble_h)**2*1e8*np.array([4, 2, 0.17, 7, 10, 0.044, 0.9, 0.0, 0.9, 3.5, 0.10, 2.7, 0.26, 0.04, 0.003, 0.03, 0.69, 2, 0.6, 0.004, 0.02, 0.09, 0.04, 0.005, 0.2, 0.024, 10, 0.1, 0.5, 0.015, 0.04, 1.0, 0.01, 0.02, 0.3, 0.008, 1.4, 0.5, 1.1, 30, 2.0, 0.6, 0.07, 0.01, 1.0, 0.9, 2.3, 0.002, 0.13, 0.4, 0.08, 0.5, 0.03, 0.4, 0.38, 0.4, 10, 0.2, 0.014, 0.004, 160, 0.014, 4.7, 0.3, 0.7, 0.4, 1, 18, 2.0, 0.004, 0.001, 2.6, 0.26, 5, 0.005])
    M_BH_lo = (0.7/Hubble_h)**2*1e8*np.array([5, 2, 0.10, 7, 10, 0.022, 0.3, 0.0, 0.8, 3.5, 0.05, 1.3, 0.09, 0.04, 0.003, 0.03, 0.35, 2, 0.6, 0.004, 0.13, 0.1, 0.05, 0.005, 0.2, 0.012, 2.7, 0.06, 0.5, 0.015, 0.06, 1.0, 0.02, 0.02, 0.3, 0.008, 0.6, 0.5, 0.6, 26, 1.9, 0.3, 0.07, 0.01, 1.0, 2.5, 1.5, 0.002, 0.13, 0.9, 0.08, 0.5, 0.09, 0.4, 0.33, 0.4, 10, 0.1, 0.014, 0.004, 160, 0.007, 3.0, 0.4, 0.7, 1.5, 1, 11, 2.0, 0.004, 0.001, 1.5, 0.19, 4, 0.005])
    M_sph_obs = (0.7/Hubble_h)**2*1e10*np.array([69, 37, 1.4, 55, 27, 2.4, 0.46, 1.0, 19, 23, 0.61, 4.6, 11, 1.9, 4.5, 1.4, 0.66, 4.7, 26, 2.0, 0.39, 0.35, 0.30, 3.5, 6.7, 0.88, 1.9, 0.93, 1.24, 0.86, 2.0, 5.4, 1.2, 4.9, 2.0, 0.66, 5.1, 2.6, 3.2, 100, 1.4, 0.88, 1.3, 0.56, 29, 6.1, 0.65, 3.3, 2.0, 6.9, 1.4, 7.7, 0.9, 3.9, 1.8, 8.4, 27, 6.0, 0.43, 1.0, 122, 0.30, 29, 11, 20, 2.8, 24, 78, 96, 3.6, 2.6, 55, 1.4, 64, 1.2])
    M_sph_hi = (0.7/Hubble_h)**2*1e10*np.array([59, 32, 2.0, 80, 23, 3.5, 0.68, 1.5, 16, 19, 0.89, 6.6, 9, 2.7, 6.6, 2.1, 0.91, 6.9, 22, 2.9, 0.57, 0.52, 0.45, 5.1, 5.7, 1.28, 2.7, 1.37, 1.8, 1.26, 1.7, 4.7, 1.7, 7.1, 2.9, 0.97, 7.4, 3.8, 2.7, 86, 2.1, 1.30, 1.9, 0.82, 25, 5.2, 0.96, 4.9, 3.0, 5.9, 1.2, 6.6, 1.3, 5.7, 2.7, 7.2, 23, 5.2, 0.64, 1.5, 105, 0.45, 25, 10, 17, 2.4, 20, 67, 83, 5.2, 3.8, 48, 2.0, 55, 1.8])
    M_sph_lo = (0.7/Hubble_h)**2*1e10*np.array([32, 17, 0.8, 33, 12, 1.4, 0.28, 0.6, 9, 10, 0.39, 2.7, 5, 1.1, 2.7, 0.8, 0.40, 2.8, 12, 1.2, 0.23, 0.21, 0.18, 2.1, 3.1, 0.52, 1.1, 0.56, 0.7, 0.51, 0.9, 2.5, 0.7, 2.9, 1.2, 0.40, 3.0, 1.5, 1.5, 46, 0.9, 0.53, 0.8, 0.34, 13, 2.8, 0.39, 2.0, 1.2, 3.2, 0.6, 3.6, 0.5, 2.3, 1.1, 3.9, 12, 2.8, 0.26, 0.6, 57, 0.18, 13, 5, 9, 1.3, 11, 36, 44, 2.1, 1.5, 26, 0.8, 30, 0.7])
    core = np.array([1,1,0,1,1,0,0,0,1,1,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,0,0,0,1,1,0,0,0,0,0,1,0,1,0,0,1,0,0,0,1,0,1,0,1,0,1,1,1,0,0,1,0,1,0])
    yerr2, yerr1 = np.log10((M_BH_obs+M_BH_hi)/M_BH_obs), -np.log10((M_BH_obs-M_BH_lo)/M_BH_obs)
    xerr2, xerr1 = np.log10((M_sph_obs+M_sph_hi)/M_sph_obs), -np.log10((M_sph_obs-M_sph_lo)/M_sph_obs)
    plt.errorbar(np.log10(M_sph_obs[core==0]), np.log10(M_BH_obs[core==0]), yerr=[yerr1[core==0],yerr2[core==0]], xerr=[xerr1[core==0],xerr2[core==0]], color='orange', alpha=0.6, label=r'S13 core', ls='none', lw=2, ms=0)
    plt.errorbar(np.log10(M_sph_obs[core==1]), np.log10(M_BH_obs[core==1]), yerr=[yerr1[core==1],yerr2[core==1]], xerr=[xerr1[core==1],xerr2[core==1]], color='c', alpha=0.6, label=r'S13 Sersic', ls='none', lw=2, ms=0)
    
    plt.ylabel(r'$\log\ M_{\mathrm{BH}}\ (M_{\odot})$')  # Set the y...
    plt.xlabel(r'$\log\ M_{\mathrm{bulge}}\ (M_{\odot})$')  # and the x-axis labels
        
    # Set the x and y axis minor ticks
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))
        
    plt.axis([8.0, 12.0, 6.0, 10.0])
        
    leg = plt.legend(loc='upper left')
    leg.draw_frame(False)  # Don't want a box frame
    for t in leg.get_texts():  # Reduce the size of the text
        t.set_fontsize('medium')

    plt.tight_layout()
        
    outputFile = OutputDir + 'BlackHoleBulgeRelationship' + OutputFormat
    plt.savefig(outputFile)  # Save the figure
    print('Saved file to', outputFile, '\n')
    plt.close()

# -------------------------------------------------------

    print('Plotting the quiescent fraction vs stellar mass')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure
    
    groupscale = 12.5
    
    w = np.where(StellarMass > 0.0)[0]
    stars = np.log10(StellarMass[w])
    halo = np.log10(CentralMvir[w])
    galtype = Type[w]
    sSFR = (SfrDisk[w] + SfrBulge[w]) / StellarMass[w]

    MinRange = 9.5
    MaxRange = 12.5
    Interval = 0.1
    Nbins = int((MaxRange-MinRange)/Interval)
    Range = np.arange(MinRange, MaxRange, Interval)
    
    Mass = []
    Fraction = []
    CentralFraction = []
    SatelliteFraction = []
    SatelliteFractionLo = []
    SatelliteFractionHi = []

    for i in range(Nbins-1):
        
        w = np.where((stars >= Range[i]) & (stars < Range[i+1]))[0]
        if len(w) > 0:
            wQ = np.where((stars >= Range[i]) & (stars < Range[i+1]) & (sSFR < 10.0**sSFRcut))[0]
            Fraction.append(1.0*len(wQ) / len(w))
        else:
            Fraction.append(0.0)
        
        w = np.where((galtype == 0) & (stars >= Range[i]) & (stars < Range[i+1]))[0]
        if len(w) > 0:
            wQ = np.where((galtype == 0) & (stars >= Range[i]) & (stars < Range[i+1]) & (sSFR < 10.0**sSFRcut))[0]
            CentralFraction.append(1.0*len(wQ) / len(w))
        else:
            CentralFraction.append(0.0)
        
        w = np.where((galtype == 1) & (stars >= Range[i]) & (stars < Range[i+1]))[0]
        if len(w) > 0:
            wQ = np.where((galtype == 1) & (stars >= Range[i]) & (stars < Range[i+1]) & (sSFR < 10.0**sSFRcut))[0]
            SatelliteFraction.append(1.0*len(wQ) / len(w))
            wQ = np.where((galtype == 1) & (stars >= Range[i]) & (stars < Range[i+1]) & (sSFR < 10.0**sSFRcut) & (halo < groupscale))[0]
            SatelliteFractionLo.append(1.0*len(wQ) / len(w))
            wQ = np.where((galtype == 1) & (stars >= Range[i]) & (stars < Range[i+1]) & (sSFR < 10.0**sSFRcut) & (halo > groupscale))[0]
            SatelliteFractionHi.append(1.0*len(wQ) / len(w))                
        else:
            SatelliteFraction.append(0.0)
            SatelliteFractionLo.append(0.0)
            SatelliteFractionHi.append(0.0)
            
        Mass.append((Range[i] + Range[i+1]) / 2.0)                
    
    Mass = np.array(Mass)
    Fraction = np.array(Fraction)
    CentralFraction = np.array(CentralFraction)
    SatelliteFraction = np.array(SatelliteFraction)
    SatelliteFractionLo = np.array(SatelliteFractionLo)
    SatelliteFractionHi = np.array(SatelliteFractionHi)
    
    w = np.where(Fraction > 0)[0]
    plt.plot(Mass[w], Fraction[w], label='All')

    w = np.where(CentralFraction > 0)[0]
    plt.plot(Mass[w], CentralFraction[w], color='Blue', label='Centrals')

    w = np.where(SatelliteFraction > 0)[0]
    plt.plot(Mass[w], SatelliteFraction[w], color='Red', label='Satellites')

    w = np.where(SatelliteFractionLo > 0)[0]
    plt.plot(Mass[w], SatelliteFractionLo[w], 'r--', label='Satellites-Lo')

    w = np.where(SatelliteFractionHi > 0)[0]
    plt.plot(Mass[w], SatelliteFractionHi[w], 'r-.', label='Satellites-Hi')
    
    plt.xlabel(r'$\log_{10} M_{\mathrm{stellar}}\ (M_{\odot})$')  # Set the x-axis label
    plt.ylabel(r'$\mathrm{Quescient\ Fraction}$')  # Set the y-axis label
    
    # Set the x and y axis minor ticks
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))
    
    plt.axis([9.5, 12.0, 0.0, 1.05])
    
    leg = plt.legend(loc='lower right')
    leg.draw_frame(False)  # Don't want a box frame
    for t in leg.get_texts():  # Reduce the size of the text
        t.set_fontsize('medium')

    plt.tight_layout()
    
    outputFile = OutputDir + 'QuiescentFraction' + OutputFormat
    plt.savefig(outputFile)  # Save the figure
    print('Saved file to', outputFile, '\n')
    plt.close()

# -------------------------------------------------------

    print('Plotting the mass fraction of galaxies')

    w = np.where(StellarMass > 0.0)[0]
    fBulge = BulgeMass[w] / StellarMass[w]
    fDisk = 1.0 - fBulge
    mass = np.log10(StellarMass[w])
    sSFR = np.log10((SfrDisk[w] + SfrBulge[w]) / StellarMass[w])
    
    binwidth = 0.2
    shift = binwidth/2.0
    mass_range = np.arange(8.5-shift, 12.0+shift, binwidth)
    bins = len(mass_range)
    
    fBulge_ave = np.zeros(bins)
    fBulge_var = np.zeros(bins)
    fDisk_ave = np.zeros(bins)
    fDisk_var = np.zeros(bins)
    
    for i in range(bins-1):
        w = np.where( (mass >= mass_range[i]) & (mass < mass_range[i+1]))[0]
        if(len(w) > 0):
            fBulge_ave[i] = np.mean(fBulge[w])
            fBulge_var[i] = np.var(fBulge[w])
            fDisk_ave[i] = np.mean(fDisk[w])
            fDisk_var[i] = np.var(fDisk[w])
    
    w = np.where(fBulge_ave > 0.0)[0]
    plt.plot(mass_range[w]+shift, fBulge_ave[w], 'r-', label='bulge')
    plt.fill_between(mass_range[w]+shift, 
        fBulge_ave[w]+fBulge_var[w], 
        fBulge_ave[w]-fBulge_var[w], 
        facecolor='red', alpha=0.25)
    
    w = np.where(fDisk_ave > 0.0)[0]
    plt.plot(mass_range[w]+shift, fDisk_ave[w], 'k-', label='disk stars')
    plt.fill_between(mass_range[w]+shift, 
        fDisk_ave[w]+fDisk_var[w], 
        fDisk_ave[w]-fDisk_var[w], 
        facecolor='k', alpha=0.25)
    
    plt.axis([mass_range[0], mass_range[bins-1], 0.0, 1.05])

    plt.ylabel(r'$\mathrm{Stellar\ Mass\ Fraction}$')  # Set the y...
    plt.xlabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')  # and the x-axis labels

    leg = plt.legend(loc='upper right', numpoints=1, labelspacing=0.1)
    leg.draw_frame(False)  # Don't want a box frame
    for t in leg.get_texts():  # Reduce the size of the text
            t.set_fontsize('medium')

    plt.tight_layout()
    
    outputFile = OutputDir + 'BulgeMassFraction' + OutputFormat
    plt.savefig(outputFile)  # Save the figure
    print('Saved file to', outputFile, '\n')
    plt.close()

# -------------------------------------------------------

    print('Plotting the average baryon fraction vs halo mass (can take time)')

    # Find halos at log Mvir = 13.5-14.0
    mask = (np.log10(Mvir) > 13.5) & (np.log10(Mvir) < 14.0)

    total_baryons = (StellarMass[mask] + ColdGas[mask] + HotGas[mask] + CGMgas[mask] + IntraClusterStars[mask] + BlackHoleMass[mask] + EjectedMass[mask]) / (0.17 * Mvir[mask])

    plt.figure()
    ax = plt.subplot(111)

    HaloMass = np.log10(Mvir)
    Baryons = StellarMass + ColdGas + HotGas + CGMgas + IntraClusterStars + BlackHoleMass + EjectedMass

    MinHalo, MaxHalo, Interval = 11.0, 16.0, 0.1
    HaloBins = np.arange(MinHalo, MaxHalo + Interval, Interval)
    Nbins = len(HaloBins) - 1

    MeanCentralHaloMass = []
    MeanBaryonFraction = []
    MeanBaryonFractionU = []
    MeanBaryonFractionL = []
    MeanStars = []
    MeanStarsU = []
    MeanStarsL = []
    MeanCold = []
    MeanColdU = []
    MeanColdL = []
    MeanHot = []
    MeanHotU = []
    MeanHotL = []
    MeanCGM = []
    MeanCGMU = []
    MeanCGML = []
    MeanICS = []
    MeanICSU = []
    MeanICSL = []
    MeanBH = []
    MeanBHU = []
    MeanBHL = []
    MeanEjected = []
    MeanEjectedU = []
    MeanEjectedL = []

    bin_indices = np.digitize(HaloMass, HaloBins) - 1

    # Pre-compute unique CentralGalaxyIndex for faster lookup
    halo_to_galaxies = defaultdict(list)
    for i, central_idx in enumerate(CentralGalaxyIndex):
        halo_to_galaxies[central_idx].append(i)

    for i in range(Nbins - 1):
        w1 = np.where((Type == 0) & (bin_indices == i))[0]
        HalosFound = len(w1)
        
        if HalosFound > 2:
            # Pre-allocate arrays for better performance
            BaryonFractions = np.zeros(HalosFound)
            StarsFractions = np.zeros(HalosFound)
            ColdFractions = np.zeros(HalosFound)
            HotFractions = np.zeros(HalosFound)
            CGMFractions = np.zeros(HalosFound)
            ICSFractions = np.zeros(HalosFound)
            BHFractions = np.zeros(HalosFound)
            EjectedFractions = np.zeros(HalosFound)
            
            # Vectorized calculation for each halo
            for idx, halo_idx in enumerate(w1):
                halo_galaxies = np.array(halo_to_galaxies[CentralGalaxyIndex[halo_idx]])
                halo_mvir = Mvir[halo_idx]
                
                # Use advanced indexing for faster summing
                BaryonFractions[idx] = np.sum(Baryons[halo_galaxies]) / halo_mvir
                StarsFractions[idx] = np.sum(StellarMass[halo_galaxies]) / halo_mvir
                ColdFractions[idx] = np.sum(ColdGas[halo_galaxies]) / halo_mvir
                HotFractions[idx] = np.sum(HotGas[halo_galaxies]) / halo_mvir
                CGMFractions[idx] = np.sum(CGMgas[halo_galaxies]) / halo_mvir
                ICSFractions[idx] = np.sum(IntraClusterStars[halo_galaxies]) / halo_mvir
                BHFractions[idx] = np.sum(BlackHoleMass[halo_galaxies]) / halo_mvir
                EjectedFractions[idx] = np.sum(EjectedMass[halo_galaxies]) / halo_mvir
            
            # Calculate statistics once for all arrays
            CentralHaloMass = np.log10(Mvir[w1])
            MeanCentralHaloMass.append(np.mean(CentralHaloMass))
            
            n_halos = len(BaryonFractions)
            sqrt_n = np.sqrt(n_halos)
            
            # Vectorized mean and std calculations
            means = [np.mean(arr) for arr in [BaryonFractions, StarsFractions, ColdFractions, 
                                             HotFractions, CGMFractions, ICSFractions, BHFractions, EjectedFractions]]
            stds = [np.std(arr) / sqrt_n for arr in [BaryonFractions, StarsFractions, ColdFractions, 
                                                    HotFractions, CGMFractions, ICSFractions, BHFractions, EjectedFractions]]
            
            # Append all means and bounds
            MeanBaryonFraction.append(means[0])
            MeanBaryonFractionU.append(means[0] + stds[0])
            MeanBaryonFractionL.append(means[0] - stds[0])
            
            MeanStars.append(means[1])
            MeanStarsU.append(means[1] + stds[1])
            MeanStarsL.append(means[1] - stds[1])
            
            MeanCold.append(means[2])
            MeanColdU.append(means[2] + stds[2])
            MeanColdL.append(means[2] - stds[2])
            
            MeanHot.append(means[3])
            MeanHotU.append(means[3] + stds[3])
            MeanHotL.append(means[3] - stds[3])
            
            MeanCGM.append(means[4])
            MeanCGMU.append(means[4] + stds[4])
            MeanCGML.append(means[4] - stds[4])
            
            MeanICS.append(means[5])
            MeanICSU.append(means[5] + stds[5])
            MeanICSL.append(means[5] - stds[5])
            
            MeanBH.append(means[6])
            MeanBHU.append(means[6] + stds[6])
            MeanBHL.append(means[6] - stds[6])

            MeanEjected.append(means[7])
            MeanEjectedU.append(means[7] + stds[7])
            MeanEjectedL.append(means[7] - stds[7])

    # Convert lists to arrays and ensure positive values for log scale
    MeanCentralHaloMass = np.array(MeanCentralHaloMass)
    MeanBaryonFraction = np.array(MeanBaryonFraction)
    MeanBaryonFractionU = np.array(MeanBaryonFractionU)
    MeanBaryonFractionL = np.maximum(np.array(MeanBaryonFractionL), 1e-6)  # Prevent negative values on log scale
    
    MeanStars = np.array(MeanStars)
    MeanStarsU = np.array(MeanStarsU)
    MeanStarsL = np.maximum(np.array(MeanStarsL), 1e-6)
    
    MeanCold = np.array(MeanCold)
    MeanColdU = np.array(MeanColdU)
    MeanColdL = np.maximum(np.array(MeanColdL), 1e-6)
    
    MeanHot = np.array(MeanHot)
    MeanHotU = np.array(MeanHotU)
    MeanHotL = np.maximum(np.array(MeanHotL), 1e-6)
    
    MeanCGM = np.array(MeanCGM)
    MeanCGMU = np.array(MeanCGMU)
    MeanCGML = np.maximum(np.array(MeanCGML), 1e-6)
    
    MeanICS = np.array(MeanICS)
    MeanICSU = np.array(MeanICSU)
    MeanICSL = np.maximum(np.array(MeanICSL), 1e-6)

    MeanBH = np.array(MeanBH)
    MeanBHU = np.array(MeanBHU)
    MeanBHL = np.maximum(np.array(MeanBHL), 1e-6)

    MeanEjected = np.array(MeanEjected)
    MeanEjectedU = np.array(MeanEjectedU)
    MeanEjectedL = np.maximum(np.array(MeanEjectedL), 1e-6)

    baryon_frac = 0.17
    plt.axhline(y=baryon_frac, color='grey', linestyle='--', linewidth=1.0, 
            label='Baryon Fraction = {:.2f}'.format(baryon_frac))

    # Add 1-sigma shading for each mass reservoir
    plt.fill_between(MeanCentralHaloMass, MeanBaryonFractionL, MeanBaryonFractionU, 
                     color='k', alpha=0.2)
    plt.fill_between(MeanCentralHaloMass, MeanStarsL, MeanStarsU, 
                     color='magenta', alpha=0.2)
    plt.fill_between(MeanCentralHaloMass, MeanColdL, MeanColdU, 
                     color='blue', alpha=0.2)
    plt.fill_between(MeanCentralHaloMass, MeanHotL, MeanHotU, 
                     color='red', alpha=0.2)
    plt.fill_between(MeanCentralHaloMass, MeanCGML, MeanCGMU, 
                     color='green', alpha=0.2)
    plt.fill_between(MeanCentralHaloMass, MeanICSL, MeanICSU, 
                     color='orange', alpha=0.2)
    plt.fill_between(MeanCentralHaloMass, MeanEjectedL, MeanEjectedU, 
                     color='yellow', alpha=0.2)

    plt.plot(MeanCentralHaloMass, MeanBaryonFraction, 'k-', label='Total')
    plt.plot(MeanCentralHaloMass, MeanStars, label='Stars', color='magenta', linestyle='--')
    plt.plot(MeanCentralHaloMass, MeanCold, label='Cold gas', color='blue', linestyle=':')
    plt.plot(MeanCentralHaloMass, MeanHot, label='Hot gas', color='red')
    plt.plot(MeanCentralHaloMass, MeanCGM, label='Circumgalactic Medium', color='green', linestyle='-.')
    plt.plot(MeanCentralHaloMass, MeanICS, label='Intracluster Stars', color='orange', linestyle='-.')
    plt.plot(MeanCentralHaloMass, MeanEjected, label='Ejected gas', color='yellow', linestyle='--')

    #plt.yscale('log')

    plt.xlabel(r'$\log_{10} M_{\mathrm{vir}}\ (M_{\odot})$')
    plt.ylabel(r'$\log_{10} \mathrm{Baryon\ Fraction}$')
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))
    plt.axis([11.1, 15.0, 0.0, 0.2])

    leg = plt.legend(loc='upper right', numpoints=1, labelspacing=0.1, bbox_to_anchor=(1.0, 0.5))
    leg.draw_frame(False)
    for t in leg.get_texts():
        t.set_fontsize('medium')

    plt.tight_layout()

    outputFile = OutputDir + 'BaryonFraction' + OutputFormat
    plt.savefig(outputFile)
    print('Saved file to', outputFile, '\n')
    plt.close()


# -------------------------------------------------------

    print('Plotting the mass in stellar, cold, hot, ejected, ICS reservoirs')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure

    w = np.where((Type == 0) & (Mvir > 1.0e10) & (StellarMass > 0.0))[0]
    dilute_mass_reservoir = 10000
    if(len(w) > dilute_mass_reservoir): w = sample(list(w), dilute_mass_reservoir)

    HaloMass = np.log10(Mvir[w])
    plt.scatter(HaloMass, np.log10(StellarMass[w]), marker='x', s=0.3, c='w', alpha=0.5, label='Stars')
    plt.scatter(HaloMass, np.log10(ColdGas[w]), marker='x', s=0.3, color='blue', alpha=0.5, label='Cold gas')
    plt.scatter(HaloMass, np.log10(HotGas[w]), marker='x', s=0.3, color='red', alpha=0.5, label='Hot gas')
    plt.scatter(HaloMass, np.log10(EjectedMass[w]), marker='x', s=0.3, color='green', alpha=0.5, label='Ejected gas')
    plt.scatter(HaloMass, np.log10(IntraClusterStars[w]), marker='x', s=0.3, color='yellow', alpha=0.5, label='Intracluster stars')
    plt.scatter(HaloMass, np.log10(CGMgas[w]), marker='x', s=0.3, color='orange', alpha=0.5, label='CGM gas')

    plt.ylabel(r'$\mathrm{stellar,\ cold,\ hot,\ ejected,\ CGM,\ ICS\ mass}$')  # Set the y...
    plt.xlabel(r'$\log\ M_{\mathrm{vir}}\ (h^{-1}\ M_{\odot})$')  # and the x-axis labels
    
    plt.axis([10.0, 15.0, 7.5, 14.0])

    leg = plt.legend(loc='upper left')
    leg.draw_frame(False)  # Don't want a box frame
    for t in leg.get_texts():  # Reduce the size of the text
        t.set_fontsize('medium')

    plt.tight_layout()
        
    outputFile = OutputDir + 'MassReservoirScatter' + OutputFormat
    plt.savefig(outputFile)  # Save the figure
    print('Saved file to', outputFile, '\n')
    plt.close()

# -------------------------------------------------------

    print('Plotting the spatial distribution of all galaxies')

    plt.figure(figsize=(18, 5))  # New figure

    w = np.where((Mvir > 0.0) & (StellarMass > 1.0e9))[0]
    if(len(w) > dilute): w = sample(list(w), dilute)

    xx = Posx[w]
    yy = Posy[w]
    zz = Posz[w]

    buff = BoxSize*0.1

    ax = plt.subplot(131)  # 1 plot on the figure
    xy = np.vstack([xx,yy])
    z_xy = gaussian_kde(xy)(xy)
    plt.scatter(xx, yy, marker='o', s=0.3, c=z_xy, cmap='plasma', alpha=0.5)
    plt.axis([0.0-buff, BoxSize+buff, 0.0-buff, BoxSize+buff])
    plt.ylabel(r'$\mathrm{x}$')  # Set the y...
    plt.xlabel(r'$\mathrm{y}$')  # and the x-axis labels
    
    ax = plt.subplot(132)  # 1 plot on the figure
    xz = np.vstack([xx,zz])
    z_xz = gaussian_kde(xz)(xz)
    plt.scatter(xx, zz, marker='o', s=0.3, c=z_xz, cmap='plasma', alpha=0.5)
    plt.axis([0.0-buff, BoxSize+buff, 0.0-buff, BoxSize+buff])
    plt.ylabel(r'$\mathrm{x}$')  # Set the y...
    plt.xlabel(r'$\mathrm{z}$')  # and the x-axis labels
    
    ax = plt.subplot(133)  # 1 plot on the figure
    yz = np.vstack([yy,zz])
    z_yz = gaussian_kde(yz)(yz)
    plt.scatter(yy, zz, marker='o', s=0.3, c=z_yz, cmap='plasma', alpha=0.5)
    plt.axis([0.0-buff, BoxSize+buff, 0.0-buff, BoxSize+buff])
    plt.ylabel(r'$\mathrm{y}$')  # Set the y...
    plt.xlabel(r'$\mathrm{z}$')  # and the x-axis labels

    # Set face color to black for all 2D plots
    for ax in plt.gcf().axes:
        ax.set_facecolor('black')
        ax.grid(False) # No grid
        # Set axis ticks to white
        ax.tick_params(axis='x', colors='w')
        ax.tick_params(axis='y', colors='w') 


    plt.tight_layout()
        
    outputFile = OutputDir + 'SpatialDistribution' + OutputFormat
    plt.savefig(outputFile)  # Save the figure
    print('Saved file to', outputFile, '\n')
    plt.close()

# -------------------------------------------------------

    print('Plotting the spatial distribution of all galaxies in 3D Box')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    w = np.where((Mvir > 0.0) & (StellarMass > 1.0e9))[0]
    if(len(w) > dilute): w = sample(list(w), dilute)

    xx = Posx[w]
    yy = Posy[w]
    zz = Posz[w]

    # Scatter plot
    xyz = np.vstack([xx,yy,zz])
    density = gaussian_kde(xyz)(xyz)
    ax.scatter(xx, yy, zz, s=0.3, c=density, cmap='plasma', alpha=0.8)

    # Draw the box edges
    points = np.array([[0,0,0], [BoxSize,0,0], [BoxSize,BoxSize,0], [0,BoxSize,0],
                       [0,0,BoxSize], [BoxSize,0,BoxSize], [BoxSize,BoxSize,BoxSize], [0,BoxSize,BoxSize]])

    edges = [[points[0], points[1]], [points[1], points[2]], [points[2], points[3]], [points[3], points[0]],
             [points[4], points[5]], [points[5], points[6]], [points[6], points[7]], [points[7], points[4]],
             [points[0], points[4]], [points[1], points[5]], [points[2], points[6]], [points[3], points[7]]]
    
    line_collection = Line3DCollection(edges, colors='k')
    ax.add_collection3d(line_collection)

    ax.set_xlabel('X (Mpc/h)')
    ax.set_ylabel('Y (Mpc/h)')
    ax.set_zlabel('Z (Mpc/h)')

    ax.set_xlim([0, BoxSize])
    ax.set_ylim([0, BoxSize])
    ax.set_zlim([0, BoxSize])

    # Set background color to black for the 3D plot
    ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    
    ax.grid(False) # No grid
    
    # Set axis ticks to white
    ax.tick_params(axis='x', colors='k')
    ax.tick_params(axis='y', colors='k')
    ax.tick_params(axis='z', colors='k')

    # Make the aspect ratio equal
    ax.set_box_aspect([1,1,1])

    plt.tight_layout()

    outputFile = OutputDir + 'SpatialDistribution3D_Box' + OutputFormat
    plt.savefig(outputFile)  # Save the figure
    print('Saved file to', outputFile, '\n')
    plt.close()

# -------------------------------------------------------

    print('Plotting the SFR')

    plt.figure()  # New figure
    ax = plt.subplot(111)  # 1 plot on the figure

    w2 = np.where(StellarMass > 0.01)[0]
    dilute_sfr = 10000
    if(len(w2) > dilute_sfr): w2 = sample(list(w2), dilute_sfr)
    mass = np.log10(StellarMass[w2])
    starformationrate =  (SfrDisk[w2] + SfrBulge[w2])

    # Create scatter plot with metallicity coloring
    plt.scatter(mass, np.log10(starformationrate), c='k', marker='x', s=1, alpha=0.9)

    plt.ylabel(r'$\log_{10} \mathrm{SFR}\ (M_{\odot}\ \mathrm{yr^{-1}})$')  # Set the y...
    plt.xlabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')  # and the x-axis labels

    # Set the x and y axis minor ticks
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))

    plt.xlim(6.0, 12.2)
    plt.ylim(-5, 3)  # Set y-axis limits for SFR

    plt.tight_layout()

    outputFile = OutputDir + 'StarFormationRate' + OutputFormat
    plt.savefig(outputFile)  # Save the figure
    print('Saved to', outputFile, '\n')
    plt.close()

    # -------------------------------------------------------

    print('Plotting outflow vs Vvir')

    plt.figure()
    w = np.where((StellarMass > 0.0) & (OutflowRate > 0.0))[0]
    if(len(w) > dilute): w = sample(list(w), dilute)

    log10_stellar_mass = np.log10(StellarMass[w])
    mass_loading = OutflowRate[w]

    plt.scatter(Vvir[w], mass_loading, c='k', marker='x', s=1, alpha=0.9)

    plt.xlabel(r'$V_{\mathrm{vir}}\ (\mathrm{km/s})$')
    plt.ylabel(r'$\dot{M}_{\mathrm{outflow}}\ (M_{\odot}\ \mathrm{yr}^{-1})$')

    plt.xlim(min(Vvir[w]), 300)
    plt.ylim(0.01, max(mass_loading)*1.1)

    plt.tight_layout()

    outputFile = OutputDir + 'outflow_rate_vs_stellar_mass' + OutputFormat
    plt.savefig(outputFile)
    print('Saved file to', outputFile, '\n')
    plt.close()

    # -------------------------------------------------------
    
    print('Plotting stellar mass vs halo mass colored by regime')

    plt.figure()

    w = np.where((Mvir > 0.0) & (StellarMass > 0.0))[0]
    if(len(w) > dilute): w = sample(list(w), dilute)

    log10_halo_mass = np.log10(Mvir[w])
    log10_stellar_mass = np.log10(StellarMass[w])
    regime_values = Regime[w]
    
    # Separate the data by regime for different colors
    cgm_regime = (regime_values == 0)
    hot_regime = (regime_values == 1)
    
    # Plot each regime separately with different colors
    if np.any(cgm_regime):
        plt.scatter(log10_halo_mass[cgm_regime], log10_stellar_mass[cgm_regime], 
                   c='blue', s=1, alpha=0.6, label='CGM Regime')
    
    if np.any(hot_regime):
        plt.scatter(log10_halo_mass[hot_regime], log10_stellar_mass[hot_regime], 
                   c='red', s=1, alpha=0.6, label='Hot Regime')

    plt.xlabel(r'$\log_{10} M_{\mathrm{vir}}\ (M_{\odot})$')
    plt.ylabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$')
    plt.xlim(10, 15)
    plt.ylim(8, 12)
    
    # Add legend
    plt.legend(loc='upper left', frameon=False)
    plt.tight_layout()


    outputFile = OutputDir + 'stellar_vs_halo_mass_by_regime' + OutputFormat
    plt.savefig(outputFile)
    print('Saved file to', outputFile, '\n')
    plt.close()

    # -------------------------------------------------------

    print('Plotting specific SFR vs stellar mass')

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)

    dilute = 100000

    w2 = np.where(StellarMass > 0.0)[0]
    if(len(w2) > dilute): w2 = sample(list(range(len(w2))), dilute)
    mass = np.log10(StellarMass[w2])
    starformationrate = (SfrDisk[w2] + SfrBulge[w2])
    sSFR = np.full_like(starformationrate, -99.0)
    mask = (StellarMass[w2] > 0)
    sSFR[mask] = np.log10(starformationrate[mask] / StellarMass[w2][mask])

    sSFRcut = -11.0
    # print(f'sSFR cut at {sSFRcut} yr^-1')

    # Separate populations
    sf_mask = (sSFR > sSFRcut) & (sSFR > -99.0)  # Star-forming
    q_mask = (sSFR <= sSFRcut) & (sSFR > -99.0)  # Quiescent

    mass_sf = mass[sf_mask]
    sSFR_sf = sSFR[sf_mask]
    mass_q = mass[q_mask]
    sSFR_q = sSFR[q_mask]

    # Define grid for density calculation
    x_bins = np.linspace(8.0, 12.2, 100)
    y_bins = np.linspace(-13, -8, 100)

    def plot_density_contours(x, y, color, label, clip_above=None, clip_below=None):
        """Plot filled contours with 1, 2, 3 sigma levels"""
        if len(x) < 10:
            return
        
        # Create 2D histogram
        H, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins])
        H = H.T  # Transpose for correct orientation
        
        # Smooth the histogram
        from scipy.ndimage import gaussian_filter
        H_smooth = gaussian_filter(H, sigma=1.5)
        
        # Apply clipping if specified
        y_centers = (yedges[:-1] + yedges[1:]) / 2
        if clip_above is not None:
            # Mask out regions above the clip line
            mask_2d = y_centers[:, np.newaxis] <= clip_above
            H_smooth = H_smooth * mask_2d
        if clip_below is not None:
            # Mask out regions below the clip line
            mask_2d = y_centers[:, np.newaxis] >= clip_below
            H_smooth = H_smooth * mask_2d
        
        # Calculate contour levels
        sorted_H = np.sort(H_smooth.flatten())[::-1]
        sorted_H = sorted_H[sorted_H > 0]  # Remove zeros
        if len(sorted_H) == 0:
            return
            
        cumsum = np.cumsum(sorted_H)
        cumsum = cumsum / cumsum[-1]
        
        level_3sigma = sorted_H[np.where(cumsum >= 0.997)[0][0]] if np.any(cumsum >= 0.997) else sorted_H[-1]
        level_2sigma = sorted_H[np.where(cumsum >= 0.95)[0][0]] if np.any(cumsum >= 0.95) else sorted_H[-1]
        level_1sigma = sorted_H[np.where(cumsum >= 0.68)[0][0]] if np.any(cumsum >= 0.68) else sorted_H[-1]
        
        levels = [level_3sigma, level_2sigma, level_1sigma]
        alphas = [0.3, 0.5, 0.7]
        
        x_centers = (xedges[:-1] + xedges[1:]) / 2
        
        # Plot filled contours
        for i, (level, alpha) in enumerate(zip(levels, alphas)):
            if i == len(levels) - 1:
                ax.contourf(x_centers, y_centers, H_smooth, 
                        levels=[level, H_smooth.max()],
                        colors=[color], alpha=alpha, label=label)
            else:
                ax.contourf(x_centers, y_centers, H_smooth, 
                        levels=[level, levels[i+1] if i+1 < len(levels) else H_smooth.max()],
                        colors=[color], alpha=alpha)
        
        # Add contour lines
        ax.contour(x_centers, y_centers, H_smooth, 
                levels=levels, colors=color, linewidths=1.0, alpha=0.8)

    # Plot quiescent population (red) - clip above -11
    if len(mass_q) > 0:
        plot_density_contours(mass_q, sSFR_q, 'firebrick', 'Quiescent', clip_above=sSFRcut)

    # Plot star-forming population (blue) - clip below -11
    if len(mass_sf) > 0:
        plot_density_contours(mass_sf, sSFR_sf, 'dodgerblue', 'Star-forming', clip_below=sSFRcut)
    # Add the sSFR cut line
    plt.axhline(y=sSFRcut, color='k', linestyle='--', linewidth=2, 
            label=f'sSFR cut = {sSFRcut}', zorder=10)

    plt.ylabel(r'$\log_{10} \mathrm{sSFR}\ (\mathrm{yr^{-1}})$', fontsize=14)
    plt.xlabel(r'$\log_{10} M_{\mathrm{stars}}\ (M_{\odot})$', fontsize=14)

    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.25))

    plt.xlim(8.0, 12.2)
    plt.ylim(-13, -8)

    plt.legend(loc='upper right', fontsize=12, frameon=False)
    # plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    plt.tight_layout()

    plt.savefig(OutputDir + 'specific_star_formation_rate' + OutputFormat, dpi=150)
    print('Saved file to', outputFile, '\n')
    plt.close()
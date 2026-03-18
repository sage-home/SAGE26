#!/usr/bin/env python3
"""
Black hole growth tracking: validates that the three accretion channels
(quasar mode, radio mode, BH-BH mergers) sum to the total BlackHoleMass,
and plots the relative contributions.
"""
import numpy as np
import h5py
import glob
import os
import sys
import argparse
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (8.34, 6.25)
plt.rcParams["figure.dpi"] = 96
plt.rcParams["font.size"] = 14
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'

OutputFormat = '.pdf'


def read_hdf(file_list, snap, field):
    """Read a field from multiple HDF5 files."""
    data = []
    for f in file_list:
        with h5py.File(f, 'r') as hf:
            snap_key = f'Snap_{snap}' if isinstance(snap, int) else snap
            if snap_key in hf and field in hf[snap_key]:
                data.append(hf[snap_key][field][:])
    return np.concatenate(data) if data else np.array([])


def read_simulation_params(filepath):
    """Read simulation parameters from HDF5 header."""
    params = {}
    with h5py.File(filepath, 'r') as f:
        params['Hubble_h'] = float(f['Header/Simulation'].attrs['hubble_h'])
        snap_groups = [key for key in f.keys() if key.startswith('Snap_')]
        snap_numbers = sorted([int(s.replace('Snap_', '')) for s in snap_groups])
        params['available_snapshots'] = snap_numbers
        params['latest_snapshot'] = max(snap_numbers) if snap_numbers else None
        if 'snapshot_redshifts' in f['Header']:
            params['snapshot_redshifts'] = np.array(f['Header/snapshot_redshifts'])
    return params


def main():
    parser = argparse.ArgumentParser(description='Black hole growth tracking validation and plots')
    parser.add_argument('-i', '--input-pattern', type=str,
                        default='./output/millennium/model_*.hdf5',
                        help='Path pattern to model HDF5 files')
    parser.add_argument('-s', '--snapshot', type=int, default=None,
                        help='Snapshot number (default: latest)')
    parser.add_argument('-o', '--output-dir', type=str, default=None,
                        help='Output directory for plots')
    args = parser.parse_args()

    file_list = sorted(glob.glob(args.input_pattern))
    if not file_list:
        print(f"Error: No files found matching: {args.input_pattern}")
        sys.exit(1)

    print(f"Found {len(file_list)} model files.")

    sim_params = read_simulation_params(file_list[0])
    Hubble_h = sim_params['Hubble_h']

    if args.snapshot is not None:
        snap_num = args.snapshot
    else:
        snap_num = sim_params['latest_snapshot']
    print(f"Using snapshot: {snap_num}")

    if 'snapshot_redshifts' in sim_params and snap_num < len(sim_params['snapshot_redshifts']):
        print(f"Redshift: {sim_params['snapshot_redshifts'][snap_num]:.4f}")

    if args.output_dir:
        OutputDir = args.output_dir
    else:
        input_dir = os.path.dirname(os.path.abspath(file_list[0]))
        OutputDir = os.path.join(input_dir, 'plots')
    os.makedirs(OutputDir, exist_ok=True)

    # Read data
    print("Reading black hole data...")
    BlackHoleMass = read_hdf(file_list, snap_num, 'BlackHoleMass') * 1.0e10 / Hubble_h
    QuasarMode = read_hdf(file_list, snap_num, 'QuasarModeBHaccretionMass') * 1.0e10 / Hubble_h
    RadioMode = read_hdf(file_list, snap_num, 'RadioModeBHaccretionMass') * 1.0e10 / Hubble_h
    BHMerger = read_hdf(file_list, snap_num, 'BHMergerMass') * 1.0e10 / Hubble_h
    StellarMass = read_hdf(file_list, snap_num, 'StellarMass') * 1.0e10 / Hubble_h
    BulgeMass = read_hdf(file_list, snap_num, 'BulgeMass') * 1.0e10 / Hubble_h
    Mvir = read_hdf(file_list, snap_num, 'Mvir') * 1.0e10 / Hubble_h
    Type = read_hdf(file_list, snap_num, 'Type')

    if len(BlackHoleMass) == 0:
        print("No galaxies found!")
        sys.exit(1)

    print(f"Total galaxies: {len(BlackHoleMass)}")

    bh_mask = BlackHoleMass > 0
    n_bh = np.sum(bh_mask)
    print(f"Galaxies with BH: {n_bh} ({100*n_bh/len(BlackHoleMass):.1f}%)")

    # ===================== VALIDATION =====================
    print("\n" + "="*60)
    print("VALIDATION: Channel sum vs BlackHoleMass")
    print("="*60)

    channel_sum = QuasarMode + RadioMode + BHMerger
    residual = BlackHoleMass - channel_sum

    # Only check galaxies with BHs
    if n_bh > 0:
        bh = BlackHoleMass[bh_mask]
        cs = channel_sum[bh_mask]
        res = residual[bh_mask]
        frac_res = res / bh

        print(f"\nBlackHoleMass total:  {bh.sum():.6e} M_sun")
        print(f"Channel sum total:    {cs.sum():.6e} M_sun")
        print(f"  Quasar mode:        {QuasarMode[bh_mask].sum():.6e} M_sun  ({100*QuasarMode[bh_mask].sum()/bh.sum():.1f}%)")
        print(f"  Radio mode:         {RadioMode[bh_mask].sum():.6e} M_sun  ({100*RadioMode[bh_mask].sum()/bh.sum():.1f}%)")
        print(f"  BH-BH mergers:      {BHMerger[bh_mask].sum():.6e} M_sun  ({100*BHMerger[bh_mask].sum()/bh.sum():.1f}%)")
        print(f"\nResidual (BH - sum):  {res.sum():.6e} M_sun")
        print(f"  This is the seed mass contribution (should be small).")
        print(f"\nPer-galaxy fractional residual (BH - sum) / BH:")
        print(f"  Median: {np.median(frac_res):.6f}")
        print(f"  Max:    {np.max(np.abs(frac_res)):.6f}")
        print(f"  99th percentile: {np.percentile(np.abs(frac_res), 99):.6f}")

        # Flag any large discrepancies
        bad = np.abs(frac_res) > 0.01
        if np.sum(bad) > 0:
            print(f"\n  WARNING: {np.sum(bad)} galaxies have >1% residual")
        else:
            print(f"\n  PASS: All galaxies have <1% residual")

    # ===================== STATISTICS =====================
    print("\n" + "="*60)
    print("CHANNEL STATISTICS (galaxies with BH > 0)")
    print("="*60)

    if n_bh > 0:
        qm = QuasarMode[bh_mask]
        rm = RadioMode[bh_mask]
        bm = BHMerger[bh_mask]

        has_qm = qm > 0
        has_rm = rm > 0
        has_bm = bm > 0

        print(f"\nGalaxies with quasar mode accretion: {np.sum(has_qm)} ({100*np.sum(has_qm)/n_bh:.1f}%)")
        print(f"Galaxies with radio mode accretion:  {np.sum(has_rm)} ({100*np.sum(has_rm)/n_bh:.1f}%)")
        print(f"Galaxies with BH-BH mergers:         {np.sum(has_bm)} ({100*np.sum(has_bm)/n_bh:.1f}%)")

        # Dominant channel per galaxy
        dominant = np.argmax(np.column_stack([qm, rm, bm]), axis=1)
        labels = ['Quasar mode', 'Radio mode', 'BH mergers']
        for i, lab in enumerate(labels):
            n = np.sum(dominant == i)
            print(f"  Dominant channel = {lab}: {n} ({100*n/n_bh:.1f}%)")

    # ===================== PLOTS =====================
    print(f"\nGenerating plots in {OutputDir}...")

    # -- Plot 1: Stacked bar chart of channel fractions in BH mass bins --
    fig, ax = plt.subplots()

    mass_bins = np.arange(5, 11.5, 0.5)
    bin_centres = 0.5 * (mass_bins[:-1] + mass_bins[1:])
    log_bh = np.log10(BlackHoleMass[bh_mask])

    qm_frac_bins = np.zeros(len(bin_centres))
    rm_frac_bins = np.zeros(len(bin_centres))
    bm_frac_bins = np.zeros(len(bin_centres))

    for i in range(len(bin_centres)):
        in_bin = (log_bh >= mass_bins[i]) & (log_bh < mass_bins[i+1])
        if np.sum(in_bin) > 5:
            total = BlackHoleMass[bh_mask][in_bin].sum()
            qm_frac_bins[i] = QuasarMode[bh_mask][in_bin].sum() / total
            rm_frac_bins[i] = RadioMode[bh_mask][in_bin].sum() / total
            bm_frac_bins[i] = BHMerger[bh_mask][in_bin].sum() / total

    # Use grouped bars (not stacked) with log y-axis so small fractions are visible
    bar_width = 0.14
    offsets = [-bar_width, 0, bar_width]
    for frac, offset, label, colour in zip([qm_frac_bins, rm_frac_bins, bm_frac_bins],
                                            offsets,
                                            ['Quasar mode', 'Radio mode', 'BH-BH mergers'],
                                            ['#2196F3', '#FF5722', '#4CAF50']):
        mask = frac > 0
        if np.any(mask):
            ax.bar(bin_centres[mask] + offset, frac[mask], width=bar_width,
                   label=label, color=colour, alpha=0.85)

    ax.set_xlabel(r'$\log_{10}(M_{\rm BH}\, /\, M_\odot)$')
    ax.set_ylabel('Fraction of BH mass')
    ax.set_title('Black Hole Growth Channels by BH Mass')
    ax.legend(loc='best')
    ax.set_yscale('log')
    ax.set_xlim(5, 11)
    ax.set_ylim(1e-6, 2.0)
    plt.tight_layout()
    plt.savefig(os.path.join(OutputDir, f'bh_growth_channels_by_mass{OutputFormat}'))
    plt.close()
    print(f"  Saved: bh_growth_channels_by_mass{OutputFormat}")

    # -- Plot 2: BH mass vs channel contributions (scatter) --
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    dilute = min(5000, n_bh)
    idx = np.random.choice(n_bh, size=dilute, replace=False) if n_bh > dilute else np.arange(n_bh)

    bh_plot = BlackHoleMass[bh_mask][idx]
    qm_plot = QuasarMode[bh_mask][idx]
    rm_plot = RadioMode[bh_mask][idx]
    bm_plot = BHMerger[bh_mask][idx]

    for ax, channel, label, colour in zip(axes,
                                           [qm_plot, rm_plot, bm_plot],
                                           ['Quasar Mode', 'Radio Mode', 'BH-BH Mergers'],
                                           ['#2196F3', '#FF5722', '#4CAF50']):
        pos = channel > 0
        if np.sum(pos) > 0:
            log_bh_pos = np.log10(bh_plot[pos])
            log_ch_pos = np.log10(channel[pos])
            ax.scatter(log_bh_pos, log_ch_pos, s=10, alpha=0.5, color=colour)
            # set axis range from data with padding
            all_vals = np.concatenate([log_bh_pos, log_ch_pos])
            lo = np.floor(all_vals.min() * 2) / 2 - 0.25
            hi = np.ceil(all_vals.max() * 2) / 2 + 0.25
            ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.5)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
        ax.set_xlabel(r'$\log_{10}(M_{\rm BH}\, /\, M_\odot)$')
        ax.set_ylabel(r'$\log_{10}(M_{\rm channel}\, /\, M_\odot)$')
        ax.set_title(label)

    plt.tight_layout()
    plt.savefig(os.path.join(OutputDir, f'bh_growth_channel_scatter{OutputFormat}'))
    plt.close()
    print(f"  Saved: bh_growth_channel_scatter{OutputFormat}")

    # -- Plot 3: Dominant channel as a function of halo mass --
    centrals = Type == 0
    central_bh = (centrals) & (BlackHoleMass > 0)

    if np.sum(central_bh) > 10:
        fig, ax = plt.subplots()

        halo_bins = np.arange(10, 15.5, 0.25)
        halo_centres = 0.5 * (halo_bins[:-1] + halo_bins[1:])
        log_mvir = np.log10(Mvir[central_bh])

        qm_halo = np.zeros(len(halo_centres))
        rm_halo = np.zeros(len(halo_centres))
        bm_halo = np.zeros(len(halo_centres))

        for i in range(len(halo_centres)):
            in_bin = (log_mvir >= halo_bins[i]) & (log_mvir < halo_bins[i+1])
            if np.sum(in_bin) > 5:
                total = BlackHoleMass[central_bh][in_bin].sum()
                qm_halo[i] = QuasarMode[central_bh][in_bin].sum() / total
                rm_halo[i] = RadioMode[central_bh][in_bin].sum() / total
                bm_halo[i] = BHMerger[central_bh][in_bin].sum() / total

        for frac, marker, label, colour in zip([qm_halo, rm_halo, bm_halo],
                                                  ['o', 's', '^'],
                                                  ['Quasar mode', 'Radio mode', 'BH-BH mergers'],
                                                  ['#2196F3', '#FF5722', '#4CAF50']):
            mask = frac > 0
            if np.any(mask):
                ax.plot(halo_centres[mask], frac[mask], marker=marker, linestyle='-',
                        color=colour, label=label, markersize=6, linewidth=1.5)

        ax.set_xlabel(r'$\log_{10}(M_{\rm vir}\, /\, M_\odot)$')
        ax.set_ylabel('Fraction of BH mass')
        ax.set_title('BH Growth Channels vs Halo Mass (Centrals)')
        ax.legend(loc='best')
        ax.set_yscale('log')
        ax.set_xlim(10, 15)
        ax.set_ylim(1e-7, 1.0)
        plt.tight_layout()
        plt.savefig(os.path.join(OutputDir, f'bh_growth_channels_vs_halo{OutputFormat}'))
        plt.close()
        print(f"  Saved: bh_growth_channels_vs_halo{OutputFormat}")

    # -- Plot 4: Validation - residual histogram --
    if n_bh > 0:
        fig, ax = plt.subplots()
        frac_res = residual[bh_mask] / BlackHoleMass[bh_mask]

        ax.hist(frac_res, bins=100, color='grey', edgecolor='black', linewidth=0.5)
        ax.axvline(0, color='red', ls='--', lw=1.5)
        ax.set_xlabel('(BlackHoleMass - ChannelSum) / BlackHoleMass')
        ax.set_ylabel('N galaxies')
        ax.set_title('Validation: Fractional Residual')
        plt.tight_layout()
        plt.savefig(os.path.join(OutputDir, f'bh_growth_validation{OutputFormat}'))
        plt.close()
        print(f"  Saved: bh_growth_validation{OutputFormat}")

    # -- Plot 5: Channel fractions as a function of redshift --
    if 'snapshot_redshifts' in sim_params:
        print("  Computing BH growth channels across all snapshots...")
        all_snaps = sim_params['available_snapshots']
        all_redshifts = sim_params['snapshot_redshifts']

        snap_z = []
        snap_qm_frac = []
        snap_rm_frac = []
        snap_bm_frac = []
        snap_total_bh = []

        for sn in all_snaps:
            bh = read_hdf(file_list, sn, 'BlackHoleMass') * 1.0e10 / Hubble_h
            qm = read_hdf(file_list, sn, 'QuasarModeBHaccretionMass') * 1.0e10 / Hubble_h
            rm = read_hdf(file_list, sn, 'RadioModeBHaccretionMass') * 1.0e10 / Hubble_h
            bm = read_hdf(file_list, sn, 'BHMergerMass') * 1.0e10 / Hubble_h

            total = bh.sum()
            if total <= 0 or len(bh) == 0:
                continue

            z = all_redshifts[sn] if sn < len(all_redshifts) else None
            if z is None:
                continue

            snap_z.append(z)
            snap_qm_frac.append(qm.sum() / total)
            snap_rm_frac.append(rm.sum() / total)
            snap_bm_frac.append(bm.sum() / total)
            snap_total_bh.append(total)

        snap_z = np.array(snap_z)
        snap_qm_frac = np.array(snap_qm_frac)
        snap_rm_frac = np.array(snap_rm_frac)
        snap_bm_frac = np.array(snap_bm_frac)
        snap_total_bh = np.array(snap_total_bh)

        if len(snap_z) > 1:
            # Sort by redshift (high to low for time progression left to right)
            order = np.argsort(snap_z)[::-1]
            snap_z = snap_z[order]
            snap_qm_frac = snap_qm_frac[order]
            snap_rm_frac = snap_rm_frac[order]
            snap_bm_frac = snap_bm_frac[order]
            snap_total_bh = snap_total_bh[order]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.34, 10), sharex=True,
                                            gridspec_kw={'height_ratios': [2, 1]})

            # Top panel: channel fractions vs redshift
            for frac, marker, label, colour in zip(
                    [snap_qm_frac, snap_rm_frac, snap_bm_frac],
                    ['o', 's', '^'],
                    ['Quasar mode', 'Radio mode', 'BH-BH mergers'],
                    ['#2196F3', '#FF5722', '#4CAF50']):
                mask = frac > 0
                if np.any(mask):
                    ax1.plot(snap_z[mask], frac[mask], marker=marker, linestyle='-',
                             color=colour, label=label, markersize=5, linewidth=1.5)

            ax1.set_ylabel('Fraction of total BH mass')
            ax1.set_title('BH Growth Channels vs Redshift')
            ax1.legend(loc='best')
            ax1.set_yscale('log')
            ax1.set_ylim(1e-7, 2.0)
            ax1.invert_xaxis()

            # Bottom panel: total BH mass density vs redshift
            ax2.plot(snap_z, np.log10(snap_total_bh), 'k-o', markersize=4)
            ax2.set_xlabel('Redshift')
            ax2.set_ylabel(r'$\log_{10}(\Sigma\, M_{\rm BH}\, /\, M_\odot)$')
            ax2.invert_xaxis()

            plt.tight_layout()
            plt.savefig(os.path.join(OutputDir, f'bh_growth_channels_vs_redshift{OutputFormat}'))
            plt.close()
            print(f"  Saved: bh_growth_channels_vs_redshift{OutputFormat}")

    print("\nDone.")


if __name__ == '__main__':
    main()

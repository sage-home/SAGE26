/*
 * save_gals_hdf5_fields.h -- single source of truth for the galaxy output fields.
 *
 * GALAXY_OUTPUT_FIELDS(X) applies X(DatasetName, StructField, ctype, h5_dtype,
 * Description, Unit) to every per-galaxy HDF5 output dataset, in output order.
 * The struct definition, the field metadata tables, the buffer alloc/free
 * stacks, and the dataset write stack are all generated from this list, so
 * adding an output field means adding exactly one entry here plus the one
 * line in save_hdf5_galaxies() that fills the buffer value.
 *
 * Not listed here (special cases handled explicitly): TaskForestNr (internal
 * per-tree bookkeeping, never written) and SFHMassDisk/SFHMassBulge (2-D
 * star-formation-history datasets with their own shape and write path).
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#define GALAXY_OUTPUT_FIELDS(X) \
    X(SnapNum, SnapNum, int32_t, H5T_NATIVE_INT, \
      "Snapshot the galaxy is located at.", \
      "Unitless") \
    X(Type, Type, int32_t, H5T_NATIVE_INT, \
      "0: Central galaxy of the main FoF halo. 1: Central of a sub-halo. 2: Orphan galaxy that will merge within the current timestep.", \
      "Unitless") \
    X(GalaxyIndex, GalaxyIndex, long long, H5T_NATIVE_LLONG, \
      "Galaxy ID, unique across all trees and files. Calculated as local galaxy number + tree number * factor + file number * factor ", \
      "Unitless") \
    X(CentralGalaxyIndex, CentralGalaxyIndex, long long, H5T_NATIVE_LLONG, \
      "GalaxyIndex of the central galaxy within this galaxy's FoF group.  Calculated the same as 'GalaxyIndex'.", \
      "Unitless") \
    X(SAGEHaloIndex, SAGEHaloIndex, int32_t, H5T_NATIVE_INT, \
      "Halo number from the restructured trees. This is different to the tree file because we order the trees. Note: This is the host halo, not necessarily the main FoF halo.", \
      "Unitless") \
    X(SAGETreeIndex, SAGETreeIndex, int32_t, H5T_NATIVE_INT, \
      "Tree number this galaxy belongs to.", \
      "Unitless") \
    X(SimulationHaloIndex, SimulationHaloIndex, long long, H5T_NATIVE_LLONG, \
      "Most bound particle ID from the tree files.", \
      "Unitless") \
    X(mergeType, mergeType, int32_t, H5T_NATIVE_INT, \
      "Denotes how this galaxy underwent a merger. 0: None. 1: Minor merger. 2: Major merger. 3: Disk instability. 4: Disrupt to intra-cluster stars.", \
      "Unitless") \
    X(mergeIntoID, mergeIntoID, int32_t, H5T_NATIVE_INT, \
      "Galaxy ID this galaxy is merging into.", \
      "Unitless") \
    X(mergeIntoSnapNum, mergeIntoSnapNum, int32_t, H5T_NATIVE_INT, \
      "Snapshot number of the galaxy this galaxy is merging into.", \
      "Unitless") \
    X(dT, dT, float, H5T_NATIVE_FLOAT, \
      "Time between this snapshot and when the galaxy was last evolved.", \
      "Myr") \
    X(Posx, Posx, float, H5T_NATIVE_FLOAT, \
      "Galaxy spatial x position.", \
      "Mpc/h") \
    X(Posy, Posy, float, H5T_NATIVE_FLOAT, \
      "Galaxy spatial y position.", \
      "Mpc/h") \
    X(Posz, Posz, float, H5T_NATIVE_FLOAT, \
      "Galaxy spatial z position.", \
      "Mpc/h") \
    X(Velx, Velx, float, H5T_NATIVE_FLOAT, \
      "Galaxy velocity in x direction.", \
      "km/s") \
    X(Vely, Vely, float, H5T_NATIVE_FLOAT, \
      "Galaxy velocity in y direction.", \
      "km/s") \
    X(Velz, Velz, float, H5T_NATIVE_FLOAT, \
      "Galaxy velocity in z direction.", \
      "km/s") \
    X(Spinx, Spinx, float, H5T_NATIVE_FLOAT, \
      "Halo spin in the x direction.", \
      "Mpc * km/s") \
    X(Spiny, Spiny, float, H5T_NATIVE_FLOAT, \
      "Halo spin in the y direction.", \
      "Mpc * km/s") \
    X(Spinz, Spinz, float, H5T_NATIVE_FLOAT, \
      "Halo spin in the z direction.", \
      "Mpc * km/s") \
    X(Len, Len, int32_t, H5T_NATIVE_INT, \
      "Number of particles in this galaxy's halo.", \
      "Unitless") \
    X(Mvir, Mvir, float, H5T_NATIVE_FLOAT, \
      "Virial mass of this galaxy's halo.", \
      "1.0e10 Msun/h") \
    X(CentralMvir, CentralMvir, float, H5T_NATIVE_FLOAT, \
      "Virial mass of the main FoF halo.", \
      "1.0e10 Msun/h") \
    X(Rvir, Rvir, float, H5T_NATIVE_FLOAT, \
      "Virial radius of this galaxy's halo.", \
      "Mpc/h") \
    X(Vvir, Vvir, float, H5T_NATIVE_FLOAT, \
      "Virial velocity of this galaxy's halo.", \
      "km/s") \
    X(VvirPeak, VvirPeak, float, H5T_NATIVE_FLOAT, \
      "Peak virial velocity of this galaxy's halo, updated only when Mvir grows. " \
      "This is the value the supernova feedback is evaluated with; Vvir above is " \
      "the instantaneous value recomputed at output time and is smaller for haloes " \
      "past their peak.", \
      "km/s") \
    X(Vmax, Vmax, float, H5T_NATIVE_FLOAT, \
      "Maximum circular speed for this galaxy's halo.", \
      "km/s") \
    X(VelDisp, VelDisp, float, H5T_NATIVE_FLOAT, \
      "Velocity dispersion for this galaxy's halo.", \
      "km/s") \
    X(ColdGas, ColdGas, float, H5T_NATIVE_FLOAT, \
      "Mass of gas in the cold reseroivr.", \
      "1.0e10 Msun/h") \
    X(StellarMass, StellarMass, float, H5T_NATIVE_FLOAT, \
      "Mass of stars.", \
      "1.0e10 Msun/h") \
    X(BulgeMass, BulgeMass, float, H5T_NATIVE_FLOAT, \
      "Mass of stars in the bulge. Bulge stars are added either through disk instabilities or mergers.", \
      "1.0e10 Msun/h") \
    X(HotGas, HotGas, float, H5T_NATIVE_FLOAT, \
      "Mass of gas in the hot reservoir.", \
      "1.0e10 Msun/h") \
    X(EjectedMass, EjectedMass, float, H5T_NATIVE_FLOAT, \
      "Mass of gass in the ejected reseroivr.", \
      "1.0e10 Msun/h") \
    X(BlackHoleMass, BlackHoleMass, float, H5T_NATIVE_FLOAT, \
      "Mass of this galaxy's black hole.", \
      "1.0e10 Msun/h") \
    X(IntraClusterStars, ICS, float, H5T_NATIVE_FLOAT, \
      "Mass of intra-cluster stars.", \
      "1.0e10 Msun/h") \
    X(MetalsColdGas, MetalsColdGas, float, H5T_NATIVE_FLOAT, \
      "Mass of metals in the cold reseroivr.", \
      "1.0e10 Msun/h") \
    X(MetalsStellarMass, MetalsStellarMass, float, H5T_NATIVE_FLOAT, \
      "Mass of metals in stars.", \
      "1.0e10 Msun/h") \
    X(MetalsBulgeMass, MetalsBulgeMass, float, H5T_NATIVE_FLOAT, \
      "Mass of metals in the bulge.", \
      "1.0e10 Msun/h") \
    X(MetalsHotGas, MetalsHotGas, float, H5T_NATIVE_FLOAT, \
      "Mass of metals in the hot reservoir.", \
      "1.0e10 Msun/h") \
    X(MetalsEjectedMass, MetalsEjectedMass, float, H5T_NATIVE_FLOAT, \
      "Mass of metals in the ejected reseroivr.", \
      "1.0e10 Msun/h") \
    X(MetalsIntraClusterStars, MetalsICS, float, H5T_NATIVE_FLOAT, \
      "Mass of metals in intra-cluster stars.", \
      "1.0e10 Msun/h") \
    X(SfrDisk, SfrDisk, float, H5T_NATIVE_FLOAT, \
      "Star formation rate within the disk.", \
      "Msun/yr") \
    X(SfrBulge, SfrBulge, float, H5T_NATIVE_FLOAT, \
      "Star formation rate within the bulge.", \
      "Msun/yr") \
    X(SfrDiskZ, SfrDiskZ, float, H5T_NATIVE_FLOAT, \
      "Average metallicity of star-forming disk gas.", \
      "Msun/yr") \
    X(SfrBulgeZ, SfrBulgeZ, float, H5T_NATIVE_FLOAT, \
      "Average metallicity of star-forming bulge gas.", \
      "Msun/yr") \
    X(DiskRadius, DiskScaleRadius, float, H5T_NATIVE_FLOAT, \
      "Disk scale radius based on Mo, Shude & White (1998)", \
      "Mpc/h") \
    X(BulgeRadius, BulgeRadius, float, H5T_NATIVE_FLOAT, \
      "Bulge scale radius based on Lange et al. (2015), Shen et al. (2003)", \
      "Mpc/h") \
    X(MergerBulgeRadius, MergerBulgeRadius, float, H5T_NATIVE_FLOAT, \
      "Bulge radius formed from mergers (classical bulge).", \
      "Mpc/h") \
    X(InstabilityBulgeRadius, InstabilityBulgeRadius, float, H5T_NATIVE_FLOAT, \
      "Bulge radius formed from disk instabilities (pseudo-bulge).", \
      "Mpc/h") \
    X(MergerBulgeMass, MergerBulgeMass, float, H5T_NATIVE_FLOAT, \
      "Mass of stars in the bulge formed from mergers.", \
      "1.0e10 Msun/h") \
    X(InstabilityBulgeMass, InstabilityBulgeMass, float, H5T_NATIVE_FLOAT, \
      "Mass of stars in the bulge formed from disk instabilities.", \
      "1.0e10 Msun/h") \
    X(Cooling, Cooling, float, H5T_NATIVE_FLOAT, \
      "Energy rate for gas cooling in the galaxy.", \
      "erg/s") \
    X(Heating, Heating, float, H5T_NATIVE_FLOAT, \
      "Energy rate for gas heating in the galaxy.", \
      "erg/s") \
    X(QuasarModeBHaccretionMass, QuasarModeBHaccretionMass, float, H5T_NATIVE_FLOAT, \
      "Mass that this galaxy's black hole accreted during the last time step.", \
      "1.0e10 Msun/h") \
    X(TimeOfLastMajorMerger, TimeOfLastMajorMerger, float, H5T_NATIVE_FLOAT, \
      "Time since this galaxy had a major merger.", \
      "Myr") \
    X(TimeOfLastMinorMerger, TimeOfLastMinorMerger, float, H5T_NATIVE_FLOAT, \
      "Time since this galaxy had a minor merger.", \
      "Myr") \
    X(OutflowRate, OutflowRate, float, H5T_NATIVE_FLOAT, \
      "Rate at which cold gas is reheated to hot gas.", \
      "Msun/yr") \
    X(infallMvir, infallMvir, float, H5T_NATIVE_FLOAT, \
      "Virial mass of this galaxy's halo at the previous timestep.", \
      "1.0e10 Msun/yr") \
    X(infallVvir, infallVvir, float, H5T_NATIVE_FLOAT, \
      "Virial velocity of this galaxy's halo at the previous timestep.", \
      "km/s") \
    X(infallVmax, infallVmax, float, H5T_NATIVE_FLOAT, \
      "Maximum circular speed of this galaxy's halo at the previous timestep.", \
      "km/s") \
    X(infallStellarMass, infallStellarMass, float, H5T_NATIVE_FLOAT, \
      "Stellar mass of this galaxy at the time it became a satellite.", \
      "1.0e10 Msun/h") \
    X(Regime, Regime, int32_t, H5T_NATIVE_INT, \
      "Regime of gas accretion onto this galaxy's halo: 0 = CGM-regime 1 = ICM-regime.", \
      "Unitless") \
    X(CGMgas, CGMgas, float, H5T_NATIVE_FLOAT, \
      "Mass of gas in the circum-galactic medium (CGM).", \
      "1.0e10 Msun/h") \
    X(MetalsCGMgas, MetalsCGMgas, float, H5T_NATIVE_FLOAT, \
      "Mass of metals in the circum-galactic medium (CGM).", \
      "1.0e10 Msun/h") \
    X(MassLoading, MassLoading, float, H5T_NATIVE_FLOAT, \
      "Mass loading factor defined as the ratio of outflow rate to star formation rate.", \
      "Unitless") \
    X(H2gas, H2gas, float, H5T_NATIVE_FLOAT, \
      "Mass of molecular hydrogen (H2) in the cold gas reservoir.", \
      "1.0e10 Msun/h") \
    X(H1gas, H1gas, float, H5T_NATIVE_FLOAT, \
      "Mass of atomic hydrogen (HI) in the cold gas reservoir.", \
      "1.0e10 Msun/h") \
    X(tcool, tcool, float, H5T_NATIVE_FLOAT, \
      "Cooling time of the CGM gas in the halo.", \
      "Myr") \
    X(tff, tff, float, H5T_NATIVE_FLOAT, \
      "Free-fall time of the CGM gas in the halo.", \
      "Myr") \
    X(tdeplete, tdeplete, float, H5T_NATIVE_FLOAT, \
      "Depletion time of the CGM gas reservoir.", \
      "Myr") \
    X(H2DepletionTime_Gyr, H2DepletionTime_Gyr, float, H5T_NATIVE_FLOAT, \
      "H2 depletion time from the K13 prescription. -1 if not applicable.", \
      "Gyr") \
    X(RcoolToRvir, RcoolToRvir, float, H5T_NATIVE_FLOAT, \
      "Ratio of the cooling radius to the virial radius of the halo.", \
      "Unitless") \
    X(TimeOfInfall, TimeOfInfall, float, H5T_NATIVE_FLOAT, \
      "Time when the galaxy last became a satellite galaxy.", \
      "Myr") \
    X(FFBRegime, FFBRegime, int32_t, H5T_NATIVE_INT, \
      "FFB Regime of this galaxy's halo: 0 = Normal halo 1 = FFB halo.", \
      "Unitless") \
    X(Concentration, Concentration, float, H5T_NATIVE_FLOAT, \
      "NFW halo concentration parameter from Ishiyama+21 c-M relation.", \
      "Unitless") \
    X(mdot_cool, mdot_cool, float, H5T_NATIVE_FLOAT, \
      "Cooling rate of hot halo gas.", \
      "1.0e10 Msun/yr") \
    X(mdot_stream, mdot_stream, float, H5T_NATIVE_FLOAT, \
      "Cooling rate of cold streams.", \
      "1.0e10 Msun/yr") \
    X(ICS_disrupt, ICS_disrupt, float, H5T_NATIVE_FLOAT, \
      "Cumulative stellar mass disrupted to ICS (tracks assembly).", \
      "1.0e10 Msun/h") \
    X(ICS_accrete, ICS_accrete, float, H5T_NATIVE_FLOAT, \
      "Cumulative ICS accreted from satellites (tracks assembly).", \
      "1.0e10 Msun/h") \
    X(ICS_sum_mt, ICS_sum_mt, float, H5T_NATIVE_FLOAT, \
      "Mass-weighted sum m*t (code time) at ICS deposition; divide by (ICS_disrupt+ICS_accrete) for mean assembly lookback.", \
      "1.0e10 Msun/h * code_time") \
    X(g_max, g_max, double, H5T_NATIVE_DOUBLE, \
      "Maximum g value for this galaxy's halo across all snapshots.", \
      "1.0e10 Msun/h") \
    X(r_heat, r_heat, float, H5T_NATIVE_FLOAT, \
      "AGN radio-mode heating radius (ratchet, capped at Rvir in the CGM regime). Cooling is suppressed at r < r_heat.", \
      "Mpc/h")\
    X(CoolingRate, CoolingRate, float, H5T_NATIVE_FLOAT, \
      "Cooling rate of the CGM gas in the halo.", \
      "Msun/Gyr")

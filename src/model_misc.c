#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "core_allvars.h"

#include "model_misc.h"

void init_galaxy(const int p, const int halonr, int *galaxycounter, const struct halo_data *halos,
                 struct GALAXY *galaxies, const struct params *run_params)
{

	XASSERT(halonr == halos[halonr].FirstHaloInFOFgroup, -1,
            "Error: halonr = %d should be equal to the FirsthaloInFOFgroup = %d\n",
            halonr, halos[halonr].FirstHaloInFOFgroup);

    galaxies[p].Type = 0;
    galaxies[p].Regime = -1;
    galaxies[p].FFBRegime = 0;
    galaxies[p].Concentration = 0.0;

    galaxies[p].GalaxyNr = *galaxycounter;
    (*galaxycounter)++;

    galaxies[p].HaloNr = halonr;
    galaxies[p].MostBoundID = halos[halonr].MostBoundID;
    galaxies[p].SnapNum = halos[halonr].SnapNum - 1;

    galaxies[p].mergeType = 0;
    galaxies[p].mergeIntoID = -1;
    galaxies[p].mergeIntoSnapNum = -1;
    galaxies[p].dT = -1.0;

    for(int j = 0; j < 3; j++) {
        galaxies[p].Pos[j] = halos[halonr].Pos[j];
        galaxies[p].Vel[j] = halos[halonr].Vel[j];
    }

    galaxies[p].Len = halos[halonr].Len;
    galaxies[p].Vmax = halos[halonr].Vmax;
    galaxies[p].Vvir = get_virial_velocity(halonr, halos, run_params);
    galaxies[p].Mvir = get_virial_mass(halonr, halos, run_params);
    galaxies[p].Rvir = get_virial_radius(halonr, halos, run_params);

    galaxies[p].deltaMvir = 0.0;

    galaxies[p].ColdGas = 0.0;
    galaxies[p].StellarMass = 0.0;
    galaxies[p].BulgeMass = 0.0;
    galaxies[p].MergerBulgeMass = 0.0;   
    galaxies[p].InstabilityBulgeMass = 0.0; 
    galaxies[p].HotGas = 0.0;
    galaxies[p].EjectedMass = 0.0;
    galaxies[p].BlackHoleMass = 0.0;
    
    galaxies[p].ICS = 0.0;
    galaxies[p].CGMgas = 0.0;
    galaxies[p].H2gas = 0.0;
    galaxies[p].H1gas = 0.0;

    galaxies[p].MetalsColdGas = 0.0;
    galaxies[p].MetalsStellarMass = 0.0;
    galaxies[p].MetalsBulgeMass = 0.0;
    galaxies[p].MetalsHotGas = 0.0;
    galaxies[p].MetalsEjectedMass = 0.0;
    galaxies[p].MetalsICS = 0.0;
    galaxies[p].MetalsCGMgas = 0.0;

    for(int step = 0; step < STEPS; step++) {
        galaxies[p].SfrDisk[step] = 0.0;
        galaxies[p].SfrBulge[step] = 0.0;
        galaxies[p].SfrDiskColdGas[step] = 0.0;
        galaxies[p].SfrDiskColdGasMetals[step] = 0.0;
        galaxies[p].SfrBulgeColdGas[step] = 0.0;
        galaxies[p].SfrBulgeColdGasMetals[step] = 0.0;
    }

    // Initialize star formation history arrays (tracks mass formed at each snapshot)
    // Only need to initialize if SaveFullSFH is enabled, otherwise these arrays are unused
    if(run_params->SaveFullSFH) {
        for(int snap = 0; snap < ABSOLUTEMAXSNAPS; snap++) {
            galaxies[p].SFHMassDisk[snap] = 0.0;
            galaxies[p].SFHMassBulge[snap] = 0.0;
        }
    }
    // Initialize ICS assembly tracking (cumulative mass through each channel)
    galaxies[p].ICS_disrupt = 0.0;
    galaxies[p].ICS_accrete = 0.0;

    galaxies[p].DiskScaleRadius = get_disk_radius(halonr, p, halos, galaxies);
    galaxies[p].BulgeRadius = get_bulge_radius(p, galaxies, run_params);
    galaxies[p].MergerBulgeRadius = get_bulge_radius(p, galaxies, run_params);
    galaxies[p].InstabilityBulgeRadius = get_bulge_radius(p, galaxies, run_params);
    galaxies[p].MergTime = 999.9f;
    galaxies[p].Cooling = 0.0;
    galaxies[p].Heating = 0.0;
    galaxies[p].r_heat = 0.0;
    galaxies[p].QuasarModeBHaccretionMass = 0.0;
    galaxies[p].TimeOfLastMajorMerger = -1.0;
    galaxies[p].TimeOfLastMinorMerger = -1.0;
    galaxies[p].OutflowRate = 0.0;
	galaxies[p].TotalSatelliteBaryons = 0.0;
    galaxies[p].RcoolToRvir = -1.0;
    galaxies[p].MassLoading = 0.0;
    galaxies[p].tcool = -1.0;
    galaxies[p].tff = -1.0;
    galaxies[p].tcool_over_tff = -1.0;
    galaxies[p].tdeplete = -1.0;

	// infall properties
    galaxies[p].infallMvir = -1.0;
    galaxies[p].infallVvir = -1.0;
    galaxies[p].infallVmax = -1.0;
    galaxies[p].infallStellarMass = -1.0;
    galaxies[p].TimeOfInfall = -1.0;

    galaxies[p].mdot_cool = 0.0;
    galaxies[p].mdot_stream = 0.0;

    galaxies[p].g_max = 0.0;


}



double get_disk_radius(const int halonr, const int p, const struct halo_data *halos, const struct GALAXY *galaxies)
{
	if(galaxies[p].Vvir > 0.0 && galaxies[p].Rvir > 0.0) {
		// See Mo, Shude & White (1998) eq12, and using a Bullock style lambda.
		double SpinMagnitude = sqrt(halos[halonr].Spin[0] * halos[halonr].Spin[0] +
                                    halos[halonr].Spin[1] * halos[halonr].Spin[1] + halos[halonr].Spin[2] * halos[halonr].Spin[2]);

		double SpinParameter = SpinMagnitude / ( 1.414 * galaxies[p].Vvir * galaxies[p].Rvir);
		return (SpinParameter / 1.414 ) * galaxies[p].Rvir;
        /* return SpinMagnitude * 0.5 / galaxies[p].Vvir; /\* should be equivalent to previous call *\/ */
	} else {
		return 0.1 * galaxies[p].Rvir;
    }
}


double get_bulge_radius(const int p, struct GALAXY *galaxies, const struct params *run_params)
{
    // BulgeSizeOn == 0: No bulge size calculation
    if(run_params->BulgeSizeOn == 0) {
        galaxies[p].BulgeRadius = 0.0;
        galaxies[p].MergerBulgeRadius = 0.0;
        galaxies[p].InstabilityBulgeRadius = 0.0;
        return 0.0;
    }
    
    const double h = run_params->Hubble_h;
    
    // BulgeSizeOn == 1: Shen equation 33 (simple power-law)
    if(run_params->BulgeSizeOn == 1) {
        if(galaxies[p].BulgeMass <= 0.0) {
            galaxies[p].BulgeRadius = 0.0;
            galaxies[p].MergerBulgeRadius = 0.0;
            galaxies[p].InstabilityBulgeRadius = 0.0;
            return 0.0;
        }
        
        // Convert bulge mass from 10^10 M_sun/h to M_sun
        const double M_bulge_sun = galaxies[p].BulgeMass * 1.0e10 / h;
        
        // Shen+2003 equation (33): log(R/kpc) = 0.56 log(M/Msun) - 5.54
        const double log_R_kpc = 0.56 * log10(M_bulge_sun) - 5.54;
        double R_bulge_kpc = pow(10.0, log_R_kpc);
        
        // Convert to code units (Mpc/h)
        const double R_bulge = R_bulge_kpc * 1.0e-3 * h;
        
        galaxies[p].BulgeRadius = R_bulge;
        galaxies[p].MergerBulgeRadius = 0.0;
        galaxies[p].InstabilityBulgeRadius = 0.0;
        
        return R_bulge;
    }
    
    // BulgeSizeOn == 2: Shen equation 32 (two-regime power-law)
    if(run_params->BulgeSizeOn == 2) {
        if(galaxies[p].BulgeMass <= 0.0) {
            galaxies[p].BulgeRadius = 0.0;
            galaxies[p].MergerBulgeRadius = 0.0;
            galaxies[p].InstabilityBulgeRadius = 0.0;
            return 0.0;
        }
        
        // Convert bulge mass from 10^10 M_sun/h to M_sun
        const double M_bulge_sun = galaxies[p].BulgeMass * 1.0e10 / h;
        
        // Transition mass from Shen et al. (2003) equation (32)
        const double M_transition = 2.0e10;  // M_sun
        
        double R_bulge_kpc;
        
        if(M_bulge_sun > M_transition) {
            // High-mass regime: like giant ellipticals
            // log(R/kpc) = 0.56 log(M) - 5.54
            const double log_R = 0.56 * log10(M_bulge_sun) - 5.54;
            R_bulge_kpc = pow(10.0, log_R);
        } else {
            // Low-mass regime: like dwarf ellipticals  
            // log(R/kpc) = 0.14 log(M) - 1.21
            const double log_R = 0.14 * log10(M_bulge_sun) - 1.21;
            R_bulge_kpc = pow(10.0, log_R);
        }
        
        // Convert to code units (Mpc/h)
        const double R_bulge = R_bulge_kpc * 1.0e-3 * h;
        
        galaxies[p].BulgeRadius = R_bulge;
        galaxies[p].MergerBulgeRadius = 0.0;
        galaxies[p].InstabilityBulgeRadius = 0.0;
        
        return R_bulge;
    }
    
    // BulgeSizeOn == 3: Tonini setup (separate merger and instability bulges)
    if(run_params->BulgeSizeOn == 3) {
        const double M_merger = galaxies[p].MergerBulgeMass;
        const double M_instability = galaxies[p].InstabilityBulgeMass;
        const double M_total = M_merger + M_instability;
        
        if(M_total <= 0.0) {
            return 0.0;
        }
        
        // 1. Retrieve the Merger Radius
        // This is now calculated in model_mergers.c via Energy Conservation
        // and stored persistently.
        double R_merger = galaxies[p].MergerBulgeRadius;
        
        // Failsafe: If mass exists but radius is 0 (e.g. initialization), use Shen as fallback
        if(M_merger > 0.0 && R_merger <= 0.0) {
             const double M_merger_sun = M_merger * 1.0e10 / h;
             const double log_R_kpc = 0.56 * log10(M_merger_sun) - 5.54;
             R_merger = pow(10.0, log_R_kpc) * 1.0e-3 * h;
             // Store it so we don't recalculate
             galaxies[p].MergerBulgeRadius = R_merger;
        }

        // 2. Retrieve Instability Radius (Already correct in your code)
        double R_instability = galaxies[p].InstabilityBulgeRadius;
        
        // Failsafe for instability radius
        if(M_instability > 0.0 && R_instability <= 0.0) {
            const double R_disc = galaxies[p].DiskScaleRadius;
            R_instability = 0.2 * R_disc;
            galaxies[p].InstabilityBulgeRadius = R_instability;
        }
        
        // 3. Weighted Average (Equation 25)
        double R_bulge = (M_merger * R_merger + M_instability * R_instability) / M_total;
        
        galaxies[p].BulgeRadius = R_bulge;
        return R_bulge;
    }
    
    // Default fallback (should not reach here)
    galaxies[p].BulgeRadius = 0.0;
    galaxies[p].MergerBulgeRadius = 0.0;
    galaxies[p].InstabilityBulgeRadius = 0.0;
    return 0.0;
}


void update_instability_bulge_radius(const int p, const double delta_mass, 
                                     const double old_disk_radius,
                                     struct GALAXY *galaxies, const struct params *run_params)
{
    // Tonini+2016 equation (15): incremental radius evolution
    // R_i = (R_i,OLD * M_i,OLD + δM * 0.2 * R_D) / (M_i,OLD + δM)
    //
    // IMPORTANT: old_disk_radius should be the disc radius BEFORE the instability event
    // This ensures we use the correct R_D value as prescribed in the paper
    
    if(run_params->BulgeSizeOn != 3) return;  // Only for Tonini mode
    if(delta_mass <= 0.0) return;
    
    const double h = run_params->Hubble_h;
    const double M_old = galaxies[p].InstabilityBulgeMass - delta_mass;  // Mass before addition
    const double R_old = galaxies[p].InstabilityBulgeRadius;
    
    // Use the OLD disc radius (pre-instability) passed as parameter
    // Convert to kpc for calculation
    const double R_disc_kpc = old_disk_radius * 1.0e3 / h;
    
    // New mass contribution scales with 0.2 * R_disc (Tonini+2016 eq. 15)
    const double R_new_contribution_kpc = 0.2 * R_disc_kpc;
    const double R_new_contribution = R_new_contribution_kpc * 1.0e-3 * h;  // to Mpc/h
    
    double R_new;
    if(M_old > 0.0 && R_old > 0.0) {
        // Incremental update (equation 15)
        const double R_old_kpc = R_old * 1.0e3 / h;
        const double M_new = galaxies[p].InstabilityBulgeMass;
        const double R_new_kpc = (R_old_kpc * M_old + R_new_contribution_kpc * delta_mass) / M_new;
        R_new = R_new_kpc * 1.0e-3 * h;
    } else {
        // First mass addition: initialize with 0.2 * R_disc
        R_new = R_new_contribution;
    }
    
    galaxies[p].InstabilityBulgeRadius = R_new;
}


double get_metallicity(const double gas, const double metals)
{
  double metallicity = 0.0;

  if(gas > 0.0 && metals > 0.0) {
      metallicity = metals / gas;
      metallicity = metallicity >= 1.0 ? 1.0:metallicity;
  }

  return metallicity;
}



double dmax(const double x, const double y)
{
    return (x > y) ? x:y;
}



double get_virial_mass(const int halonr, const struct halo_data *halos, const struct params *run_params)
{
  if(halonr == halos[halonr].FirstHaloInFOFgroup && halos[halonr].Mvir >= 0.0)
    return halos[halonr].Mvir;   /* take spherical overdensity mass estimate */
  else
    return halos[halonr].Len * run_params->PartMass;
}



double get_virial_velocity(const int halonr, const struct halo_data *halos, const struct params *run_params)
{
	double Rvir;

	Rvir = get_virial_radius(halonr, halos, run_params);

    if(Rvir > 0.0)
		return sqrt(run_params->G * get_virial_mass(halonr, halos, run_params) / Rvir);
	else
		return 0.0;
}


double get_virial_radius(const int halonr, const struct halo_data *halos, const struct params *run_params)
{
  // return halos[halonr].Rvir;  // Used for Bolshoi
  const int snapnum = halos[halonr].SnapNum;
  const double zplus1 = 1.0 + run_params->ZZ[snapnum];
  const double hubble_of_z_sq =
      run_params->Hubble * run_params->Hubble *(run_params->Omega * zplus1 * zplus1 * zplus1 + (1.0 - run_params->Omega - run_params->OmegaLambda) * zplus1 * zplus1 +
                                              run_params->OmegaLambda);

  const double rhocrit = 3.0 * hubble_of_z_sq / (8.0 * M_PI * run_params->G);
  const double fac = 1.0 / (200.0 * 4.0 * M_PI / 3.0 * rhocrit);

  return cbrt(get_virial_mass(halonr, halos, run_params) * fac);
}

void determine_and_store_regime(const int ngal, struct GALAXY *galaxies,
                                const struct params *run_params)
{
    for(int p = 0; p < ngal; p++) {
        if(galaxies[p].mergeType > 0) continue;

        // Convert Mvir to physical units (Msun)
        // Mvir is stored in units of 10^10 Msun/h
        const double Mvir_physical = galaxies[p].Mvir * 1.0e10 / run_params->Hubble_h;

        // Shock mass threshold (Dekel & Birnboim 2006)
        const double Mshock = 6.0e11;  // Msun

        // Calculate mass ratio for sigmoid
        const double mass_ratio = Mvir_physical / Mshock;

        // BUG FIX: Protect against log10(0) or log10(negative)
        if(mass_ratio <= 0.0) {
            galaxies[p].Regime = 0;  // Default to CGM regime for invalid mass
            continue;
        }

        // Smooth sigmoid transition (consistent with FFB approach)
        // Width of transition in dex
        const double delta_log_M = 0.1;

        // Sigmoid argument: x = log10(M/Mshock) / width
        const double x = log10(mass_ratio) / delta_log_M;

        // Sigmoid function: probability of being in Hot regime
        // Smoothly varies from 0 (well below Mshock) to 1 (well above Mshock)
        const double hot_fraction = 1.0 / (1.0 + exp(-x));

        // Probabilistic assignment based on sigmoid
        const double random_uniform = (double)rand() / (double)RAND_MAX;

        galaxies[p].Regime = (random_uniform < hot_fraction) ? 1 : 0;

    }
}

void determine_and_store_ffb_regime(const int ngal, const double Zcurr, struct GALAXY *galaxies,
                                     const struct params *run_params)
{
    // Only apply FFB if the mode is enabled
    if(run_params->FeedbackFreeModeOn == 0) {
        // FFB mode disabled - mark all galaxies as normal
        for(int p = 0; p < ngal; p++) {
            galaxies[p].FFBRegime = 0;
        }
        return;
    }

    // Classify each galaxy as FFB or normal
    for(int p = 0; p < ngal; p++) {
        if(galaxies[p].mergeType > 0) continue;

        // Only CGM regime halos can be FFB, so we check that first
        if(galaxies[p].Regime == 1) {
            galaxies[p].FFBRegime = 0;  // Normal halo - in hot CGM regime, not eligible for FFB
            continue;
        }

        if(run_params->FeedbackFreeModeOn == 1) {
            // Li et al. 2024 mass-based method (original)
            const double Mvir = galaxies[p].Mvir;

            // Calculate smooth FFB fraction using sigmoid transition (Li et al. 2024, eq. 3)
            const double f_ffb = calculate_ffb_fraction(Mvir, Zcurr, run_params);

            // Probabilistic assignment based on smooth sigmoid function
            const double random_uniform = (double)rand() / (double)RAND_MAX;

            if(random_uniform < f_ffb) {
                galaxies[p].FFBRegime = 1;  // FFB halo
            } else {
                galaxies[p].FFBRegime = 0;  // Normal halo
            }
        } else if(run_params->FeedbackFreeModeOn == 2) {
            // Boylan-Kolchin 2025 acceleration-based method
            // FFB regime when g_max > g_crit (sharp cutoff)
            // Uses Ishiyama+21 c-M relation for concentration (lookup table)
            const double g_max = calculate_gmax_BK25(p, galaxies, run_params);

            // Store g_max for analysis (optional)
            galaxies[p].g_max = g_max;

            // g_crit/G = 3100 M_sun/pc^2 (Boylan-Kolchin 2025, Table 1)
            // g_crit = G * 3100 * M_sun / pc^2  (~4.3e-10 m/s^2)
            // In code units: g_crit = run_params->G * 3100 * (M_sun/UnitMass) / (pc/UnitLength)^2
            const double Msun_code = SOLAR_MASS / run_params->UnitMass_in_g;
            const double pc_code = 3.08568e18 / run_params->UnitLength_in_cm;  // 1 pc in cm
            const double g_crit = run_params->G * 3100.0 * Msun_code / (pc_code * pc_code);

            if(g_max > g_crit) {
                galaxies[p].FFBRegime = 1;  // FFB halo - above critical acceleration
            } else {
                galaxies[p].FFBRegime = 0;  // Normal halo
            }
        }
    }
}

double interpolate_concentration_ishiyama21(const double logM, const double z)
{
    // Ishiyama+21 concentration-mass lookup table (mdef=vir, halo_sample=all, c_type=fit)
    // Generated by Colossus (colossus_cm_comparison.py) with planck18 cosmology
    // Rows: log10(Mvir / [Msun/h]), Columns: redshift
    #define CM_TABLE_N_MASS 41
    #define CM_TABLE_N_Z 31
    const double cm_table_logmass[CM_TABLE_N_MASS] = {8.0, 8.2, 8.4, 8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8, 10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 11.2, 11.4, 11.6, 11.8, 12.0, 12.2, 12.4, 12.6, 12.8, 13.0, 13.2, 13.4, 13.6, 13.8, 14.0, 14.2, 14.4, 14.6, 14.8, 15.0, 15.2, 15.4, 15.6, 15.8, 16.0};
    const double cm_table_z[CM_TABLE_N_Z] = {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0};
    const double cm_table[CM_TABLE_N_MASS][CM_TABLE_N_Z] = {
        {22.780, 15.998, 11.800, 9.177, 7.451, 6.259, 5.401, 4.767, 4.287, 3.918, 3.631, 3.407, 3.230, 3.091, 2.983, 2.900, 2.838, 2.792, 2.762, 2.744, 2.736, 2.738, 2.748, 2.765, 2.788, 2.818, 2.852, 2.891, 2.933, 2.980, 3.030},
        {22.110, 15.520, 11.446, 8.903, 7.232, 6.080, 5.254, 4.644, 4.184, 3.832, 3.560, 3.348, 3.183, 3.055, 2.956, 2.882, 2.827, 2.789, 2.766, 2.755, 2.754, 2.762, 2.778, 2.801, 2.831, 2.866, 2.906, 2.950, 2.999, 3.051, 3.107},
        {21.443, 15.045, 11.093, 8.630, 7.015, 5.904, 5.109, 4.524, 4.084, 3.750, 3.492, 3.293, 3.139, 3.022, 2.933, 2.867, 2.821, 2.791, 2.775, 2.771, 2.777, 2.792, 2.815, 2.844, 2.880, 2.921, 2.968, 3.018, 3.072, 3.131, 3.192},
        {20.777, 14.571, 10.742, 8.360, 6.800, 5.730, 4.966, 4.406, 3.987, 3.670, 3.428, 3.242, 3.100, 2.993, 2.913, 2.857, 2.820, 2.798, 2.790, 2.793, 2.807, 2.829, 2.858, 2.895, 2.937, 2.985, 3.037, 3.094, 3.155, 3.219, 3.287},
        {20.114, 14.100, 10.394, 8.091, 6.588, 5.558, 4.826, 4.291, 3.893, 3.594, 3.366, 3.194, 3.064, 2.968, 2.899, 2.852, 2.824, 2.810, 2.811, 2.822, 2.843, 2.872, 2.909, 2.953, 3.002, 3.057, 3.116, 3.180, 3.247, 3.319, 3.393},
        {19.452, 13.630, 10.047, 7.825, 6.378, 5.389, 4.689, 4.180, 3.803, 3.521, 3.309, 3.151, 3.033, 2.948, 2.890, 2.853, 2.833, 2.829, 2.838, 2.857, 2.886, 2.924, 2.968, 3.019, 3.076, 3.138, 3.205, 3.276, 3.351, 3.429, 3.511},
        {18.799, 13.167, 9.706, 7.564, 6.172, 5.225, 4.556, 4.073, 3.717, 3.453, 3.257, 3.112, 3.008, 2.934, 2.887, 2.860, 2.850, 2.855, 2.873, 2.901, 2.939, 2.984, 3.037, 3.096, 3.161, 3.231, 3.305, 3.384, 3.466, 3.552, 3.642},
        {18.154, 12.710, 9.371, 7.308, 5.972, 5.066, 4.429, 3.971, 3.636, 3.390, 3.210, 3.080, 2.988, 2.927, 2.891, 2.874, 2.875, 2.890, 2.917, 2.954, 3.000, 3.055, 3.116, 3.184, 3.257, 3.336, 3.418, 3.505, 3.596, 3.690, 3.787},
        {17.517, 12.260, 9.041, 7.057, 5.776, 4.911, 4.306, 3.873, 3.561, 3.333, 3.170, 3.054, 2.975, 2.926, 2.902, 2.897, 2.908, 2.933, 2.970, 3.017, 3.073, 3.136, 3.207, 3.284, 3.366, 3.454, 3.545, 3.641, 3.741, 3.844, 3.950},
        {16.888, 11.816, 8.716, 6.811, 5.585, 4.761, 4.188, 3.781, 3.490, 3.282, 3.135, 3.034, 2.970, 2.934, 2.921, 2.927, 2.950, 2.986, 3.033, 3.091, 3.157, 3.230, 3.311, 3.398, 3.490, 3.587, 3.688, 3.794, 3.903, 4.015, 4.131},
        {16.263, 11.376, 8.395, 6.570, 5.399, 4.616, 4.075, 3.695, 3.426, 3.237, 3.107, 3.022, 2.972, 2.949, 2.949, 2.968, 3.002, 3.050, 3.108, 3.177, 3.254, 3.338, 3.429, 3.527, 3.629, 3.737, 3.849, 3.964, 4.084, 4.207, 4.333},
        {15.644, 10.941, 8.080, 6.333, 5.217, 4.476, 3.968, 3.614, 3.368, 3.199, 3.087, 3.018, 2.982, 2.974, 2.988, 3.019, 3.066, 3.126, 3.197, 3.277, 3.366, 3.462, 3.564, 3.673, 3.787, 3.906, 4.029, 4.156, 4.287, 4.421, 4.558},
        {15.030, 10.511, 7.768, 6.100, 5.041, 4.341, 3.866, 3.540, 3.317, 3.168, 3.074, 3.022, 3.003, 3.009, 3.037, 3.083, 3.143, 3.216, 3.300, 3.393, 3.494, 3.603, 3.718, 3.839, 3.965, 4.096, 4.232, 4.371, 4.514, 4.661, 4.811},
        {14.419, 10.084, 7.461, 5.873, 4.869, 4.212, 3.770, 3.472, 3.273, 3.146, 3.071, 3.037, 3.034, 3.056, 3.100, 3.160, 3.235, 3.322, 3.419, 3.526, 3.641, 3.763, 3.892, 4.027, 4.167, 4.311, 4.460, 4.613, 4.770, 4.930, 5.093},
        {13.812, 9.662, 7.158, 5.650, 4.703, 4.089, 3.682, 3.413, 3.238, 3.133, 3.078, 3.063, 3.077, 3.117, 3.176, 3.253, 3.343, 3.445, 3.558, 3.680, 3.810, 3.947, 4.091, 4.240, 4.395, 4.554, 4.718, 4.885, 5.057, 5.232, 5.410},
        {13.210, 9.244, 6.861, 5.433, 4.544, 3.973, 3.601, 3.361, 3.213, 3.130, 3.097, 3.101, 3.135, 3.192, 3.270, 3.363, 3.470, 3.589, 3.718, 3.857, 4.003, 4.157, 4.316, 4.482, 4.653, 4.829, 5.009, 5.192, 5.380, 5.571, 5.766},
        {12.620, 8.837, 6.573, 5.225, 4.393, 3.867, 3.531, 3.321, 3.199, 3.141, 3.130, 3.155, 3.209, 3.286, 3.382, 3.494, 3.619, 3.757, 3.904, 4.061, 4.225, 4.396, 4.574, 4.758, 4.946, 5.140, 5.338, 5.540, 5.745, 5.954, 6.167},
        {12.042, 8.440, 6.295, 5.026, 4.253, 3.771, 3.471, 3.293, 3.199, 3.166, 3.178, 3.226, 3.302, 3.400, 3.516, 3.649, 3.794, 3.952, 4.119, 4.295, 4.479, 4.670, 4.868, 5.071, 5.280, 5.493, 5.711, 5.933, 6.158, 6.388, 6.620},
        {11.476, 8.053, 6.026, 4.838, 4.122, 3.686, 3.424, 3.278, 3.213, 3.207, 3.245, 3.317, 3.416, 3.537, 3.676, 3.831, 3.999, 4.178, 4.367, 4.565, 4.771, 4.984, 5.204, 5.429, 5.660, 5.895, 6.135, 6.379, 6.627, 6.879, 7.134},
        {10.921, 7.677, 5.767, 4.660, 4.003, 3.614, 3.391, 3.279, 3.245, 3.267, 3.332, 3.431, 3.555, 3.701, 3.865, 4.044, 4.237, 4.440, 4.654, 4.876, 5.107, 5.344, 5.588, 5.838, 6.093, 6.353, 6.618, 6.887, 7.160, 7.436, 7.716},
        {10.377, 7.310, 5.519, 4.493, 3.897, 3.555, 3.373, 3.297, 3.295, 3.349, 3.444, 3.570, 3.723, 3.897, 4.088, 4.295, 4.514, 4.745, 4.985, 5.235, 5.493, 5.757, 6.029, 6.306, 6.589, 6.877, 7.169, 7.466, 7.767, 8.071, 8.379},
        {9.844, 6.954, 5.282, 4.338, 3.804, 3.512, 3.372, 3.334, 3.368, 3.455, 3.582, 3.741, 3.924, 4.129, 4.351, 4.588, 4.837, 5.098, 5.370, 5.650, 5.938, 6.233, 6.536, 6.844, 7.158, 7.477, 7.801, 8.129, 8.461, 8.797, 9.137},
        {9.321, 6.610, 5.057, 4.196, 3.726, 3.486, 3.392, 3.395, 3.467, 3.591, 3.754, 3.947, 4.165, 4.404, 4.660, 4.931, 5.215, 5.510, 5.816, 6.130, 6.453, 6.783, 7.120, 7.464, 7.813, 8.167, 8.526, 8.890, 9.258, 9.630, 10.006},
        {8.810, 6.277, 4.845, 4.070, 3.666, 3.481, 3.435, 3.483, 3.597, 3.761, 3.963, 4.196, 4.452, 4.730, 5.024, 5.334, 5.656, 5.990, 6.335, 6.689, 7.051, 7.420, 7.797, 8.180, 8.569, 8.964, 9.363, 9.767, 10.176, 10.589, 11.006},
        {8.317, 5.962, 4.651, 3.963, 3.627, 3.501, 3.507, 3.603, 3.764, 3.973, 4.219, 4.495, 4.795, 5.116, 5.455, 5.808, 6.174, 6.553, 6.941, 7.340, 7.747, 8.161, 8.583, 9.012, 9.447, 9.887, 10.333, 10.783, 11.238, 11.698, 12.162},
        {7.848, 5.668, 4.479, 3.881, 3.616, 3.551, 3.613, 3.762, 3.975, 4.234, 4.530, 4.856, 5.206, 5.576, 5.965, 6.368, 6.784, 7.213, 7.653, 8.103, 8.561, 9.027, 9.501, 9.982, 10.470, 10.963, 11.462, 11.966, 12.474, 12.988, 13.506},
        {7.403, 5.397, 4.331, 3.824, 3.635, 3.637, 3.761, 3.969, 4.239, 4.555, 4.908, 5.290, 5.697, 6.124, 6.570, 7.030, 7.505, 7.992, 8.491, 8.999, 9.517, 10.043, 10.577, 11.119, 11.667, 12.222, 12.782, 13.348, 13.919, 14.495, 15.076},
        {6.983, 5.151, 4.209, 3.799, 3.690, 3.765, 3.957, 4.232, 4.567, 4.948, 5.366, 5.813, 6.285, 6.778, 7.289, 7.817, 8.359, 8.913, 9.480, 10.057, 10.644, 11.240, 11.844, 12.456, 13.075, 13.701, 14.333, 14.971, 15.614, 16.263, 16.916},
        {6.588, 4.931, 4.118, 3.809, 3.788, 3.943, 4.212, 4.562, 4.972, 5.428, 5.920, 6.443, 6.991, 7.561, 8.149, 8.755, 9.375, 10.009, 10.655, 11.313, 11.980, 12.657, 13.343, 14.038, 14.740, 15.449, 16.165, 16.887, 17.615, 18.349, 19.088},
        {6.221, 4.742, 4.062, 3.861, 3.936, 4.182, 4.539, 4.976, 5.472, 6.015, 6.595, 7.206, 7.843, 8.502, 9.182, 9.879, 10.591, 11.318, 12.058, 12.811, 13.573, 14.346, 15.129, 15.921, 16.721, 17.528, 18.343, 19.164, 19.992, 20.826, 21.666},
        {5.885, 4.586, 4.047, 3.965, 4.148, 4.496, 4.955, 5.492, 6.090, 6.735, 7.418, 8.134, 8.875, 9.642, 10.428, 11.234, 12.055, 12.893, 13.744, 14.609, 15.484, 16.371, 17.268, 18.176, 19.092, 20.016, 20.948, 21.887, 22.834, 23.787, 24.747},
        {5.583, 4.472, 4.083, 4.132, 4.438, 4.905, 5.482, 6.138, 6.855, 7.621, 8.427, 9.266, 10.134, 11.027, 11.942, 12.876, 13.828, 14.798, 15.782, 16.781, 17.792, 18.814, 19.849, 20.894, 21.949, 23.013, 24.085, 25.165, 26.253, 27.349, 28.453},
        {5.321, 4.406, 4.182, 4.378, 4.825, 5.432, 6.147, 6.945, 7.805, 8.716, 9.668, 10.656, 11.674, 12.719, 13.788, 14.878, 15.987, 17.115, 18.259, 19.419, 20.592, 21.778, 22.978, 24.189, 25.410, 26.642, 27.883, 29.133, 30.392, 31.659, 32.934},
        {5.109, 4.402, 4.359, 4.724, 5.335, 6.107, 6.989, 7.956, 8.988, 10.073, 11.203, 12.370, 13.570, 14.799, 16.054, 17.333, 18.632, 19.951, 21.289, 22.644, 24.014, 25.398, 26.797, 28.208, 29.632, 31.068, 32.513, 33.968, 35.433, 36.908, 38.391},
        {4.959, 4.475, 4.637, 5.198, 6.004, 6.973, 8.056, 9.227, 10.468, 11.766, 13.111, 14.497, 15.919, 17.372, 18.855, 20.362, 21.893, 23.446, 25.021, 26.613, 28.223, 29.848, 31.490, 33.147, 34.817, 36.500, 38.195, 39.900, 41.617, 43.344, 45.082},
        {4.888, 4.649, 5.043, 5.835, 6.875, 8.083, 9.411, 10.833, 12.329, 13.887, 15.497, 17.152, 18.846, 20.575, 22.336, 24.126, 25.941, 27.782, 29.646, 31.531, 33.435, 35.357, 37.298, 39.255, 41.228, 43.215, 45.216, 47.228, 49.254, 51.292, 53.342},
        {4.917, 4.950, 5.616, 6.684, 8.008, 9.509, 11.138, 12.868, 14.680, 16.560, 18.498, 20.485, 22.515, 24.586, 26.692, 28.830, 30.998, 33.194, 35.417, 37.663, 39.932, 42.220, 44.530, 46.860, 49.207, 51.570, 53.949, 56.342, 58.749, 61.172, 63.607},
        {5.077, 5.418, 6.404, 7.807, 9.481, 11.344, 13.348, 15.462, 17.667, 19.948, 22.294, 24.695, 27.144, 29.640, 32.176, 34.749, 37.354, 39.994, 42.663, 45.360, 48.082, 50.828, 53.598, 56.390, 59.204, 62.036, 64.885, 67.751, 70.634, 73.535, 76.450},
        {5.403, 6.100, 7.472, 9.287, 11.395, 13.711, 16.182, 18.777, 21.474, 24.257, 27.113, 30.033, 33.007, 36.035, 39.110, 42.225, 45.380, 48.573, 51.801, 55.062, 58.351, 61.668, 65.013, 68.385, 71.780, 75.198, 78.637, 82.094, 85.572, 89.069, 92.585},
        {5.946, 7.061, 8.906, 11.235, 13.886, 16.772, 19.831, 23.032, 26.349, 29.764, 33.263, 36.835, 40.472, 44.169, 47.921, 51.721, 55.565, 59.456, 63.387, 67.356, 71.359, 75.394, 79.463, 83.562, 87.690, 91.845, 96.023, 100.223, 104.448, 108.696, 112.966},
        {6.769, 8.392, 10.824, 13.800, 17.141, 20.748, 24.555, 28.524, 32.627, 36.844, 41.159, 45.559, 50.034, 54.581, 59.191, 63.858, 68.578, 73.351, 78.174, 83.040, 87.947, 92.891, 97.876, 102.897, 107.953, 113.039, 118.154, 123.295, 128.465, 133.663, 138.887}
    };

    // Bilinear interpolation of the Ishiyama+21 c-M table
    // Clamp to table boundaries
    double lm = logM;
    double zz = z;
    if(lm < cm_table_logmass[0]) lm = cm_table_logmass[0];
    if(lm > cm_table_logmass[CM_TABLE_N_MASS - 1]) lm = cm_table_logmass[CM_TABLE_N_MASS - 1];
    if(zz < cm_table_z[0]) zz = cm_table_z[0];
    if(zz > cm_table_z[CM_TABLE_N_Z - 1]) zz = cm_table_z[CM_TABLE_N_Z - 1];

    // Find mass index (step = 0.2)
    int im = (int)((lm - 8.0) / 0.2);
    if(im < 0) im = 0;
    if(im >= CM_TABLE_N_MASS - 1) im = CM_TABLE_N_MASS - 2;

    // Find redshift index (step = 0.5)
    int iz = (int)(zz / 0.5);
    if(iz < 0) iz = 0;
    if(iz >= CM_TABLE_N_Z - 1) iz = CM_TABLE_N_Z - 2;

    // Fractional positions
    const double fm = (lm - cm_table_logmass[im]) / (cm_table_logmass[im + 1] - cm_table_logmass[im]);
    const double fz = (zz - cm_table_z[iz]) / (cm_table_z[iz + 1] - cm_table_z[iz]);

    // Bilinear interpolation
    const double c00 = cm_table[im][iz];
    const double c10 = cm_table[im + 1][iz];
    const double c01 = cm_table[im][iz + 1];
    const double c11 = cm_table[im + 1][iz + 1];

    return c00 * (1.0 - fm) * (1.0 - fz)
         + c10 * fm * (1.0 - fz)
         + c01 * (1.0 - fm) * fz
         + c11 * fm * fz;
}

double get_halo_concentration(const int p, const double z, const struct GALAXY *galaxies,
                               const struct params *run_params)
{
    (void)run_params;  // reserved for future c-M options
    const double Mvir = galaxies[p].Mvir;  // 10^10 M_sun / h
    if(Mvir <= 0.0) return 1.0;

    const double Mvir_Msun_h = Mvir * 1.0e10;  // Msun/h (table units)
    const double logM = log10(Mvir_Msun_h);

    // Ishiyama+21 c-M relation (lookup table from Colossus)
    double c = interpolate_concentration_ishiyama21(logM, z);
    if(c < 1.0) c = 1.0;

    return c;
}

double calculate_gmax_BK25(const int p, const struct GALAXY *galaxies,
                            const struct params *run_params)
{
    // Boylan-Kolchin 2025: maximum NFW gravitational acceleration
    //
    // g_vir = G * M_vir / R_vir^2                                (Eq. 2)
    // g_max = (g_vir / mu(c)) * (c^2 / 2)                         (Eq. 4)
    // where mu(x) = ln(1+x) - x/(1+x)
    //
    // Returns g_max in code units (UnitLength / UnitTime^2)

    const double Mvir = galaxies[p].Mvir;  // code mass units (10^10 M_sun / h)
    const double Rvir = galaxies[p].Rvir;  // code length units (Mpc / h)

    if(Mvir <= 0.0 || Rvir <= 0.0) {
        return 0.0;
    }

    // g_vir = G * M_vir / R_vir^2  (code units)
    const double g_vir = run_params->G * Mvir / (Rvir * Rvir);

    // Use pre-computed concentration from the galaxy struct
    double c = galaxies[p].Concentration;
    if(c < 1.0) c = 1.0;

    // mu(c) = ln(1+c) - c/(1+c)
    const double mu_c = log(1.0 + c) - c / (1.0 + c);

    // g_max = (g_vir / mu(c)) * (c^2 / 2)   [BK25 Eq. 4]
    return (g_vir / mu_c) * (c * c / 2.0);
}


float calculate_stellar_scale_height_BR06(float disk_scale_length_pc)
{
    // BR06 equation (9): log h* = -0.23 - 0.8 log R*
    // where h* and R* are measured in parsecs
    if (disk_scale_length_pc <= 0.0) {
        return 0.0; // Default fallback value in pc
    }
    
    float log_h_star = -0.23 + 0.8 * log10(disk_scale_length_pc);
    float h_star_pc = pow(10.0, log_h_star);
    
    // Apply reasonable physical bounds (from 10 pc to 10 kpc)
    // if (h_star_pc < 10.0) h_star_pc = 10.0;
    // if (h_star_pc > 10000.0) h_star_pc = 10000.0;
    
    return h_star_pc;
}


float calculate_midplane_pressure_BR06(float sigma_gas, float sigma_stars, float disk_scale_length_pc)
{
    // Early termination for edge cases
    if (sigma_gas <= 0.5 || disk_scale_length_pc <= 0.0) {
        return 0.0;
    }
    
    // For very low stellar surface density, use a minimal value to avoid numerical issues
    // but don't artificially boost it like before
    float effective_sigma_stars = sigma_stars;
    if (sigma_stars < 0.1) {
        effective_sigma_stars = 0.1;  // Minimal floor just to avoid sqrt(0)
    }
    
    // Calculate stellar scale height using exact BR06 equation (9)
    float h_star_pc = calculate_stellar_scale_height_BR06(disk_scale_length_pc);
    
    // BR06 hardcoded parameters EXACTLY as in paper
    const float v_g = 8.0;          // km/s, gas velocity dispersion (BR06 Table text)
    
    // BR06 Equation (5) EXACTLY as written in paper:
    // P_ext/k = 272 cm⁻³ K × (Σ_gas/M_⊙ pc⁻²) × (Σ_*/M_⊙ pc⁻²)^0.5 × (v_g/km s⁻¹) × (h_*/pc)^-0.5
    float pressure = 272.0 * sigma_gas * sqrt(effective_sigma_stars) * v_g / sqrt(h_star_pc);


    return pressure; // K cm⁻³
}


float calculate_molecular_fraction_BR06(float gas_surface_density, float stellar_surface_density, 
                                         float disk_scale_length_pc)
{

    // Calculate midplane pressure using exact BR06 formula
    float pressure = calculate_midplane_pressure_BR06(gas_surface_density, stellar_surface_density, 
                                                     disk_scale_length_pc);
    
    if (pressure <= 0.0) {
        return 0.0;
    }
    
    // BR06 parameters from equation (13) for non-interacting galaxies
    // These are the exact values from the paper
    const float P0 = 4.54e4;    // Reference pressure, K cm⁻³ (equation 13)
    const float alpha = 0.92;  // Power law index (equation 13)
    
    // BR06 Equation (11): R_mol = (P_ext/P₀)^α
    float pressure_ratio = pressure / P0;
    float R_mol = pow(pressure_ratio, alpha);
    
    // Convert to molecular fraction: f_mol = R_mol / (1 + R_mol)
    // This is the standard conversion from molecular-to-atomic ratio to molecular fraction
    double f_mol = R_mol / (1.0 + R_mol);
    
    return f_mol;
}

float calculate_molecular_fraction_radial_integration(const int gal, struct GALAXY *galaxies, 
                                                      const struct params *run_params)
{
    const float h = run_params->Hubble_h;
    const float rs_pc = galaxies[gal].DiskScaleRadius * 1.0e6 / h;  // Scale radius in pc
    
    if (rs_pc <= 0.0 || galaxies[gal].ColdGas <= 0.0) {
        return 0.0;
    }
    
    // Total masses in physical units (M☉)
    const float M_gas_total = galaxies[gal].ColdGas * 1.0e10 / h;
    const float M_star_total = galaxies[gal].StellarMass * 1.0e10 / h;
    
    // Central surface densities for exponential profiles: Σ₀ = M_total / (2π r_s²)
    const float sigma_gas_0 = M_gas_total / (2.0 * M_PI * rs_pc * rs_pc);
    const float sigma_star_0 = M_star_total / (2.0 * M_PI * rs_pc * rs_pc);
    
    // Radial integration parameters
    const int N_BINS = 10;  // Number of radial bins
    const float R_MAX = 3.0 * rs_pc;  // Integrate out to 3 scale radii (~95% of mass)
    const float dr = R_MAX / N_BINS;
    
    // Integrate molecular gas mass
    float M_H2_total = 0.0;
    
    for (int i = 0; i < N_BINS; i++) {
        // Bin center radius
        const float r = (i + 0.5) * dr;
        
        // Exponential surface density profiles: Σ(r) = Σ₀ exp(-r/r_s)
        const float exp_factor = exp(-r / rs_pc);
        const float sigma_gas_r = sigma_gas_0 * exp_factor;
        const float sigma_star_r = sigma_star_0 * exp_factor;
        
        // Skip bins with negligible gas
        if (sigma_gas_r < 1e-3) continue;
        
        // Calculate molecular fraction at this radius using BR06
        const float f_mol_r = calculate_molecular_fraction_BR06(sigma_gas_r, sigma_star_r, 
                                                                rs_pc);
        
        // Mass of molecular gas in this annulus: dM = 2π r Σ_gas f_mol dr
        const float dM_H2 = 2.0 * M_PI * r * sigma_gas_r * f_mol_r * dr;
        
        M_H2_total += dM_H2;
    }
    
    // Convert back to code units (10^10 M☉/h)
    const float H2_code_units = M_H2_total * h / 1.0e10;
    
    // Store and return
    galaxies[gal].H2gas = H2_code_units;
    return H2_code_units;
}

double calculate_ffb_threshold_mass(const double z, const struct params *run_params)
{
    // Equation (2) from Li et al. 2024
    // M_v,ffb / 10^10.8 M_sun ~ ((1+z)/10)^-6.2
    //
    // In code units (10^10 M_sun/h):
    // log(M_code) = log(M_sun) - 10 + log(h)
    //             = 10.8 - 6.2*log((1+z)/10) - 10 + log(h)
    //             = 0.8 + log(h) - 6.2*log((1+z)/10)

    const double h = run_params->Hubble_h;
    const double z_norm = (1.0 + z) / 10.0;
    const double log_Mvir_ffb_code = 0.8 + log10(h) - 6.2 * log10(z_norm);

    return pow(10.0, log_Mvir_ffb_code);
}


double calculate_ffb_fraction(const double Mvir, const double z, const struct params *run_params)
{
    // Calculate the fraction of galaxies in FFB regime
    // Uses smooth sigmoid transition from Li et al. 2024, equation (3)
    
    if (run_params->FeedbackFreeModeOn == 0) {
        return 0.0;  // FFB mode disabled
    }

    // if (z < 5.0) {
    //     return 0.0;  // FFB only active at z >= 6.2
    // }
    
    // Calculate FFB threshold mass
    const double Mvir_ffb = calculate_ffb_threshold_mass(z, run_params);

    // BUG FIX: Protect against log10(0) or log10(negative)
    if(Mvir <= 0.0 || Mvir_ffb <= 0.0) {
        return 0.0;  // Return no FFB for invalid masses
    }

    // Width of transition in dex (Li et al. use 0.15 dex)
    const double delta_log_M = 0.15;

    // Calculate argument for sigmoid function
    const double x = log10(Mvir / Mvir_ffb) / delta_log_M;
    
    // Sigmoid function: S(x) = 1 / (1 + exp(-x))
    // const double k = 5.0;  // Steepness parameter (can be adjusted)
    // Sigmoid function with adjustable steepness
    // Smoothly varies from 0 (well below threshold) to 1 (well above threshold)
    const double f_ffb = 1.0 / (1.0 + exp(-x));
    // Steeper transition
    // const double f_ffb = 1.0 / (1.0 + exp(-k * x));
    
    return f_ffb;
}

// Calculate molecular fraction using Krumholz & Dekel (2012) model
// Based on equations 18-21 from Krumholz & Dekel 2012, ApJ 753:16

float calculate_H2_fraction_KD12(const float surface_density, const float metallicity, const float clumping_factor) 
{
    if (surface_density <= 0.0) {
        return 0.0;
    }
    
    // Metallicity normalized to solar (Z_sun = 0.02)
    // Z0 = (M_Z/M_g)/Z_sun as defined in KD12 equation after (17)
    // Apply floor to prevent numerical issues and unphysical zero H2 at very low Z
    float metallicity_floored = metallicity;
    if (metallicity_floored < 0.0002) {  // Z = 0.01 Z_sun minimum
        metallicity_floored = 0.0002;
    }
    float Z0 = metallicity_floored / 0.02;
    
    // Convert surface density from M_sun/pc^2 to g/cm^2
    // Conversion: 1 M_sun/pc^2 = 2.088 × 10^-4 g/cm^2
    float Sigma_gcm2 = surface_density * 2.088e-4;
    
    // Surface density normalized to 1 g/cm^2 (as defined after KD12 Eq. 16)
    // Sigma_0 = Sigma / (1 g cm^-2)
    float Sigma_0 = Sigma_gcm2;  // dimensionless, in units of 1 g/cm^2
    
    // Calculate dust optical depth parameter (KD12 Eq. 21)
    // tau_c = 320 * c * Z0 * Sigma_0
    // where c is the clumping factor:
    //   c ≈ 1 for Sigma measured on 100 pc scales
    //   c ≈ 5 for Sigma measured on ~1 kpc scales (from text after Eq. 21)
    float tau_c = 320.0 * clumping_factor * Z0 * Sigma_0;
    
    // Self-shielding parameter chi (KD12 Eq. 20)
    // chi = 3.1 * (1 + Z0^0.365) / 4.1
    float chi = 3.1 * (1.0 + pow(Z0, 0.365)) / 4.1;
    
    // Compute s parameter (KD12 Eq. 19)
    // s = ln(1 + 0.6*chi + 0.01*chi^2) / (0.6 * tau_c)
    float chi_sq = chi * chi;
    float s;
    // BUG FIX: Protect against division by zero when tau_c is very small
    if(tau_c > 1e-10) {
        s = log(1.0 + 0.6 * chi + 0.01 * chi_sq) / (0.6 * tau_c);
    } else {
        s = 100.0;  // Large s implies f_H2 -> 0 (atomic dominated)
    }
    
    // Molecular fraction (KD12 Eq. 18)
    // f_H2 = 1 - (3/4) * s/(1 + 0.25*s)  for s < 2
    // f_H2 = 0                            for s >= 2
    float f_H2;
    if (s < 2.0) {
        f_H2 = 1.0 - 0.75 * s / (1.0 + 0.25 * s);
    } else {
        f_H2 = 0.0;
    }
    
    // Ensure fraction stays within bounds
    if (f_H2 < 0.0) f_H2 = 0.0;
    if (f_H2 > 1.0) f_H2 = 1.0;
    
    return f_H2;
}
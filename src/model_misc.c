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
        {19.444, 15.542, 12.268, 9.853, 8.119, 6.857, 5.922, 5.213, 4.667, 4.238, 3.898, 3.626, 3.407, 3.230, 3.087, 2.972, 2.879, 2.806, 2.748, 2.705, 2.673, 2.652, 2.639, 2.634, 2.636, 2.645, 2.659, 2.678, 2.701, 2.728, 2.760},
        {18.825, 15.041, 11.869, 9.533, 7.857, 6.640, 5.740, 5.059, 4.535, 4.126, 3.803, 3.545, 3.339, 3.173, 3.040, 2.934, 2.850, 2.785, 2.735, 2.698, 2.674, 2.659, 2.652, 2.654, 2.662, 2.676, 2.695, 2.720, 2.748, 2.781, 2.818},
        {18.213, 14.545, 11.475, 9.216, 7.599, 6.427, 5.562, 4.909, 4.408, 4.018, 3.711, 3.468, 3.274, 3.120, 2.997, 2.901, 2.826, 2.769, 2.727, 2.697, 2.679, 2.671, 2.671, 2.679, 2.693, 2.713, 2.739, 2.769, 2.803, 2.842, 2.884},
        {17.606, 14.053, 11.084, 8.904, 7.345, 6.217, 5.387, 4.762, 4.284, 3.913, 3.623, 3.394, 3.214, 3.071, 2.959, 2.872, 2.806, 2.757, 2.723, 2.702, 2.691, 2.690, 2.697, 2.711, 2.732, 2.758, 2.790, 2.826, 2.867, 2.911, 2.959},
        {17.003, 13.566, 10.698, 8.595, 7.094, 6.011, 5.216, 4.619, 4.164, 3.813, 3.539, 3.325, 3.158, 3.027, 2.926, 2.849, 2.792, 2.752, 2.726, 2.712, 2.709, 2.716, 2.730, 2.751, 2.778, 2.811, 2.849, 2.892, 2.939, 2.989, 3.043},
        {16.405, 13.083, 10.315, 8.290, 6.848, 5.809, 5.048, 4.479, 4.048, 3.716, 3.460, 3.261, 3.106, 2.988, 2.897, 2.831, 2.783, 2.752, 2.735, 2.730, 2.735, 2.748, 2.770, 2.798, 2.833, 2.873, 2.918, 2.967, 3.021, 3.078, 3.139},
        {15.812, 12.605, 9.937, 7.989, 6.605, 5.610, 4.885, 4.344, 3.936, 3.624, 3.385, 3.201, 3.060, 2.954, 2.875, 2.818, 2.781, 2.759, 2.751, 2.754, 2.767, 2.789, 2.819, 2.855, 2.897, 2.944, 2.997, 3.054, 3.114, 3.179, 3.247},
        {15.225, 12.132, 9.564, 7.693, 6.367, 5.417, 4.726, 4.214, 3.829, 3.537, 3.315, 3.147, 3.020, 2.926, 2.858, 2.813, 2.786, 2.774, 2.775, 2.787, 2.809, 2.839, 2.877, 2.921, 2.971, 3.027, 3.087, 3.152, 3.220, 3.292, 3.367},
        {14.649, 11.669, 9.200, 7.404, 6.135, 5.229, 4.573, 4.089, 3.728, 3.457, 3.252, 3.099, 2.986, 2.905, 2.850, 2.815, 2.799, 2.797, 2.808, 2.829, 2.860, 2.900, 2.946, 2.999, 3.058, 3.122, 3.190, 3.263, 3.340, 3.420, 3.504},
        {14.083, 11.214, 8.842, 7.122, 5.910, 5.048, 4.427, 3.971, 3.634, 3.382, 3.196, 3.058, 2.960, 2.892, 2.849, 2.826, 2.821, 2.830, 2.851, 2.882, 2.923, 2.972, 3.028, 3.090, 3.157, 3.230, 3.308, 3.389, 3.475, 3.564, 3.656},
        {13.525, 10.767, 8.492, 6.847, 5.691, 4.873, 4.286, 3.859, 3.545, 3.315, 3.146, 3.025, 2.941, 2.887, 2.857, 2.846, 2.852, 2.872, 2.904, 2.946, 2.997, 3.056, 3.122, 3.194, 3.272, 3.354, 3.441, 3.533, 3.628, 3.726, 3.828},
        {12.976, 10.327, 8.149, 6.577, 5.479, 4.704, 4.152, 3.754, 3.464, 3.254, 3.104, 3.000, 2.931, 2.891, 2.874, 2.876, 2.895, 2.927, 2.970, 3.023, 3.085, 3.155, 3.232, 3.314, 3.402, 3.495, 3.593, 3.695, 3.800, 3.909, 4.021},
        {12.434, 9.894, 7.812, 6.315, 5.273, 4.542, 4.025, 3.655, 3.390, 3.202, 3.070, 2.983, 2.930, 2.905, 2.902, 2.918, 2.949, 2.994, 3.049, 3.115, 3.189, 3.270, 3.358, 3.452, 3.552, 3.656, 3.765, 3.878, 3.995, 4.115, 4.238},
        {11.898, 9.468, 7.481, 6.058, 5.073, 4.386, 3.905, 3.564, 3.324, 3.157, 3.046, 2.976, 2.940, 2.930, 2.942, 2.972, 3.017, 3.075, 3.144, 3.222, 3.309, 3.403, 3.504, 3.610, 3.722, 3.839, 3.960, 4.085, 4.215, 4.347, 4.483},
        {11.370, 9.049, 7.157, 5.809, 4.880, 4.238, 3.792, 3.481, 3.267, 3.122, 3.031, 2.980, 2.961, 2.968, 2.996, 3.041, 3.101, 3.173, 3.256, 3.348, 3.449, 3.557, 3.671, 3.791, 3.916, 4.047, 4.181, 4.320, 4.463, 4.609, 4.758},
        {10.849, 8.636, 6.840, 5.566, 4.694, 4.097, 3.687, 3.407, 3.219, 3.098, 3.027, 2.996, 2.996, 3.020, 3.064, 3.126, 3.201, 3.289, 3.387, 3.495, 3.610, 3.733, 3.863, 3.998, 4.138, 4.283, 4.432, 4.586, 4.743, 4.904, 5.068},
        {10.335, 8.230, 6.530, 5.331, 4.517, 3.964, 3.592, 3.342, 3.181, 3.084, 3.036, 3.025, 3.044, 3.087, 3.150, 3.229, 3.322, 3.426, 3.541, 3.665, 3.797, 3.937, 4.082, 4.234, 4.390, 4.551, 4.717, 4.887, 5.061, 5.238, 5.418},
        {9.830, 7.834, 6.229, 5.105, 4.348, 3.842, 3.507, 3.289, 3.156, 3.085, 3.060, 3.071, 3.110, 3.173, 3.255, 3.353, 3.465, 3.588, 3.721, 3.863, 4.013, 4.171, 4.334, 4.504, 4.678, 4.857, 5.041, 5.229, 5.420, 5.615, 5.814},
        {9.339, 7.451, 5.940, 4.890, 4.192, 3.731, 3.435, 3.250, 3.146, 3.100, 3.100, 3.134, 3.196, 3.281, 3.384, 3.502, 3.634, 3.778, 3.931, 4.093, 4.263, 4.440, 4.624, 4.813, 5.007, 5.206, 5.410, 5.618, 5.829, 6.045, 6.263},
        {8.862, 7.080, 5.663, 4.688, 4.047, 3.634, 3.376, 3.225, 3.151, 3.134, 3.160, 3.219, 3.305, 3.413, 3.539, 3.680, 3.834, 4.000, 4.175, 4.359, 4.552, 4.751, 4.956, 5.167, 5.384, 5.605, 5.831, 6.061, 6.295, 6.533, 6.774},
        {8.397, 6.721, 5.399, 4.498, 3.916, 3.550, 3.333, 3.217, 3.175, 3.187, 3.241, 3.327, 3.439, 3.573, 3.724, 3.890, 4.069, 4.260, 4.460, 4.668, 4.885, 5.109, 5.339, 5.575, 5.816, 6.062, 6.313, 6.568, 6.827, 7.090, 7.356},
        {7.944, 6.374, 5.146, 4.320, 3.798, 3.482, 3.306, 3.228, 3.220, 3.264, 3.348, 3.463, 3.604, 3.766, 3.945, 4.139, 4.345, 4.563, 4.791, 5.027, 5.271, 5.523, 5.780, 6.044, 6.313, 6.587, 6.866, 7.149, 7.436, 7.727, 8.022},
        {7.503, 6.040, 4.907, 4.157, 3.696, 3.430, 3.299, 3.260, 3.288, 3.367, 3.484, 3.632, 3.804, 3.997, 4.207, 4.432, 4.670, 4.918, 5.176, 5.444, 5.719, 6.001, 6.290, 6.585, 6.886, 7.191, 7.502, 7.817, 8.136, 8.459, 8.785},
        {7.076, 5.719, 4.682, 4.010, 3.611, 3.398, 3.313, 3.317, 3.385, 3.501, 3.655, 3.838, 4.046, 4.274, 4.518, 4.778, 5.050, 5.333, 5.627, 5.929, 6.239, 6.557, 6.881, 7.211, 7.547, 7.889, 8.235, 8.586, 8.941, 9.301, 9.664},
        {6.662, 5.413, 4.472, 3.880, 3.546, 3.389, 3.353, 3.402, 3.513, 3.671, 3.865, 4.089, 4.336, 4.603, 4.887, 5.186, 5.497, 5.820, 6.153, 6.494, 6.845, 7.202, 7.567, 7.938, 8.314, 8.696, 9.083, 9.475, 9.872, 10.273, 10.677},
        {6.263, 5.123, 4.281, 3.770, 3.504, 3.406, 3.424, 3.522, 3.681, 3.885, 4.124, 4.392, 4.684, 4.996, 5.324, 5.667, 6.024, 6.391, 6.769, 7.156, 7.552, 7.955, 8.366, 8.782, 9.205, 9.634, 10.068, 10.507, 10.951, 11.399, 11.851},
        {5.886, 4.854, 4.112, 3.685, 3.490, 3.455, 3.530, 3.684, 3.895, 4.150, 4.441, 4.759, 5.102, 5.464, 5.843, 6.238, 6.645, 7.064, 7.494, 7.933, 8.381, 8.837, 9.300, 9.770, 10.246, 10.728, 11.216, 11.710, 12.208, 12.711, 13.218},
        {5.531, 4.610, 3.970, 3.629, 3.510, 3.542, 3.681, 3.894, 4.165, 4.479, 4.827, 5.204, 5.604, 6.024, 6.462, 6.915, 7.381, 7.860, 8.349, 8.848, 9.357, 9.874, 10.398, 10.930, 11.468, 12.012, 12.562, 13.118, 13.680, 14.246, 14.817},
        {5.201, 4.391, 3.855, 3.605, 3.567, 3.674, 3.883, 4.165, 4.502, 4.882, 5.297, 5.740, 6.208, 6.695, 7.201, 7.722, 8.257, 8.804, 9.363, 9.932, 10.511, 11.099, 11.694, 12.298, 12.908, 13.524, 14.147, 14.777, 15.412, 16.052, 16.697},
        {4.896, 4.200, 3.773, 3.621, 3.670, 3.859, 4.147, 4.507, 4.921, 5.378, 5.870, 6.391, 6.936, 7.502, 8.086, 8.687, 9.302, 9.930, 10.570, 11.221, 11.882, 12.553, 13.233, 13.920, 14.614, 15.316, 16.025, 16.740, 17.461, 18.188, 18.920},
        {4.618, 4.041, 3.729, 3.681, 3.828, 4.109, 4.488, 4.937, 5.440, 5.987, 6.569, 7.180, 7.816, 8.474, 9.151, 9.846, 10.556, 11.279, 12.015, 12.762, 13.521, 14.290, 15.068, 15.854, 16.648, 17.450, 18.259, 19.076, 19.900, 20.729, 21.564},
        {4.370, 3.918, 3.729, 3.797, 4.053, 4.440, 4.923, 5.475, 6.083, 6.735, 7.423, 8.141, 8.885, 9.652, 10.439, 11.245, 12.067, 12.903, 13.753, 14.615, 15.489, 16.374, 17.269, 18.173, 19.085, 20.006, 20.935, 21.873, 22.817, 23.768, 24.725},
        {4.158, 3.838, 3.784, 3.980, 4.360, 4.870, 5.474, 6.149, 6.880, 7.657, 8.470, 9.315, 10.189, 11.086, 12.004, 12.942, 13.898, 14.869, 15.854, 16.853, 17.865, 18.889, 19.924, 20.968, 22.023, 23.086, 24.158, 25.239, 26.328, 27.425, 28.528},
        {3.988, 3.811, 3.905, 4.247, 4.771, 5.424, 6.172, 6.993, 7.871, 8.796, 9.761, 10.759, 11.787, 12.840, 13.916, 15.013, 16.130, 17.263, 18.412, 19.576, 20.753, 21.945, 23.148, 24.362, 25.586, 26.821, 28.065, 29.320, 30.583, 31.854, 33.133},
        {3.869, 3.848, 4.110, 4.620, 5.313, 6.135, 7.056, 8.051, 9.107, 10.212, 11.359, 12.542, 13.757, 14.999, 16.266, 17.557, 18.869, 20.199, 21.546, 22.909, 24.289, 25.683, 27.090, 28.510, 29.941, 31.383, 32.837, 34.302, 35.777, 37.260, 38.752},
        {3.812, 3.966, 4.419, 5.127, 6.021, 7.048, 8.177, 9.384, 10.655, 11.980, 13.349, 14.757, 16.201, 17.675, 19.176, 20.702, 22.252, 23.823, 25.413, 27.021, 28.646, 30.289, 31.946, 33.616, 35.300, 36.997, 38.706, 40.428, 42.161, 43.905, 45.658},
        {3.830, 4.184, 4.863, 5.806, 6.942, 8.218, 9.602, 11.069, 12.605, 14.199, 15.843, 17.529, 19.254, 21.013, 22.801, 24.619, 26.463, 28.330, 30.218, 32.127, 34.056, 36.005, 37.970, 39.950, 41.945, 43.954, 45.979, 48.018, 50.070, 52.133, 54.207},
        {3.943, 4.531, 5.478, 6.707, 8.141, 9.723, 11.421, 13.211, 15.076, 17.005, 18.988, 21.020, 23.095, 25.207, 27.353, 29.532, 31.741, 33.977, 36.236, 38.519, 40.826, 43.154, 45.501, 47.866, 50.248, 52.646, 55.062, 57.495, 59.942, 62.403, 64.876},
        {4.176, 5.044, 6.317, 7.898, 9.701, 11.665, 13.758, 15.950, 18.227, 20.576, 22.985, 25.449, 27.962, 30.518, 33.112, 35.744, 38.411, 41.108, 43.832, 46.584, 49.363, 52.167, 54.993, 57.840, 60.706, 63.592, 66.498, 69.424, 72.367, 75.325, 78.299},
        {4.564, 5.773, 7.450, 9.471, 11.737, 14.183, 16.772, 19.473, 22.269, 25.147, 28.094, 31.103, 34.169, 37.284, 40.445, 43.648, 46.892, 50.171, 53.482, 56.825, 60.200, 63.604, 67.034, 70.488, 73.966, 77.466, 80.990, 84.538, 88.106, 91.691, 95.295},
        {5.150, 6.782, 8.965, 11.539, 14.391, 17.445, 20.662, 24.006, 27.459, 31.006, 34.633, 38.331, 42.096, 45.918, 49.792, 53.717, 57.690, 61.703, 65.755, 69.843, 73.969, 78.131, 82.323, 86.543, 90.791, 95.066, 99.369, 103.700, 108.055, 112.431, 116.829}
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

    // printf("Galaxy %d: Mvir = %.3e (Msun/h), logM = %.3f, z = %.2f, c = %.3f\n", p, Mvir_Msun_h, logM, z, c);

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
    const double g_vir = run_params->G * Mvir / (Rvir * Rvir);  // Convert Mvir to physical units (Msun) for g_vir

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
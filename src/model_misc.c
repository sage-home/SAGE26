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
    
    // AGNrecipeOn==4: Seed black holes in halos with Mvir > 10^10 Msun/h
    if(run_params->AGNrecipeOn == 4 && galaxies[p].Mvir > 1.0) {
        galaxies[p].BlackHoleMass = 1.0e-6;  // 10^4 Msun/h in units of 10^10 Msun/h
    }
    
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

    // Counter for diagnostics
    // static int total_galaxies_checked = 0;
    // static int ffb_galaxies_assigned = 0;

    // Classify each galaxy as FFB or normal based on equation (2)
    for(int p = 0; p < ngal; p++) {
        if(galaxies[p].mergeType > 0) continue;

        const double Mvir = galaxies[p].Mvir;
        const double stellar_mass = galaxies[p].StellarMass;

        // Massive galaxies (>= 10^12 M☉) are never FFB
        // const double stellar_mass_threshold = 100.0;  // 10^12 M☉ in code units
        // if(stellar_mass >= stellar_mass_threshold) {
        //     galaxies[p].FFBRegime = 0;  // Normal halo - too massive for FFB
        //     continue;
        // }

        // Only CGM regime halos can be FFB, so we check that first
        if(galaxies[p].Regime == 1) {
            galaxies[p].FFBRegime = 0;  // Normal halo - in hot CGM regime, not eligible for FFB
            continue;
        }

        // Calculate smooth FFB fraction using sigmoid transition (Li et al. 2024, eq. 3)
        const double f_ffb = calculate_ffb_fraction(Mvir, Zcurr, run_params);

        // Probabilistic assignment based on smooth sigmoid function
        // Galaxies near threshold have intermediate probability of being FFB
        const double random_uniform = (double)rand() / (double)RAND_MAX;

        if(random_uniform < f_ffb) {
            galaxies[p].FFBRegime = 1;  // FFB halo
        } else {
            galaxies[p].FFBRegime = 0;  // Normal halo
        }
    }
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
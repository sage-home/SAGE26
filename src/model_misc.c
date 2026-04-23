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
    galaxies[p].FFBRandom = (float)rand() / (float)RAND_MAX;
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
    galaxies[p].ICS_sum_mt = 0.0;

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

// Inverse normal CDF (probit function) — Peter Acklam's rational approximation.
// Converts a uniform variate p ∈ (0,1) to a standard normal variate.
// Accurate to ~1e-9 across the full range.
static double inverse_normal_cdf(double p)
{
    const double a[] = {-3.969683028665376e+01,  2.209460984245205e+02,
                        -2.759285104469687e+02,  1.383577518672690e+02,
                        -3.066479806614716e+01,  2.506628277459239e+00};
    const double b[] = {-5.447609879822406e+01,  1.615858368580409e+02,
                        -1.556989798598866e+02,  6.680131188771972e+01,
                        -1.328068155288572e+01};
    const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01,
                        -2.400758277161838e+00, -2.549732539343734e+00,
                         4.374664141464968e+00,  2.938163982698783e+00};
    const double d[] = { 7.784695709041462e-03,  3.224671290700398e-01,
                         2.445134137142996e+00,  3.754408661907416e+00};

    const double p_low  = 0.02425;
    const double p_high = 1.0 - p_low;

    double q, r;

    if(p < p_low) {
        q = sqrt(-2.0 * log(p));
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else if(p <= p_high) {
        q = p - 0.5;
        r = q * q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    } else {
        q = sqrt(-2.0 * log(1.0 - p));
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                 ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
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

    // Pre-compute g_crit in code units for BK25 modes (constant, doesn't depend on galaxy)
    // g_crit/G = 3100 M_sun/pc^2 (Boylan-Kolchin 2025, Table 1)
    double g_crit = 0.0;
    if(run_params->FeedbackFreeModeOn == 2 || run_params->FeedbackFreeModeOn == 3 ||
       run_params->FeedbackFreeModeOn == 4 || run_params->FeedbackFreeModeOn == 7) {
        const double Msun_code = SOLAR_MASS / run_params->UnitMass_in_g;
        const double pc_code = 3.08568e18 / run_params->UnitLength_in_cm;
        g_crit = run_params->G * 3100.0 * Msun_code / (pc_code * pc_code) / run_params->Hubble_h;
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

            // Use persistent random number for consistent sigmoid-based determination
            // (avoids re-rolling each timestep, while still allowing transitions as mass evolves)
            const double random_uniform = (double)galaxies[p].FFBRandom;

            if(random_uniform < f_ffb) {
                galaxies[p].FFBRegime = 1;  // FFB halo
            } else {
                galaxies[p].FFBRegime = 0;  // Normal halo
            }
        } else if(run_params->FeedbackFreeModeOn == 2) {
            // Boylan-Kolchin 2025 acceleration-based method (Ishiyama+21 lookup table concentration)
            // FFB regime when g_max > g_crit (sharp cutoff)
            const double g_max = calculate_gmax_BK25(p, Zcurr, galaxies, run_params);

            galaxies[p].g_max = g_max;

            if(g_max > g_crit) {
                galaxies[p].FFBRegime = 1;  // FFB halo - above critical acceleration
            } else {
                galaxies[p].FFBRegime = 0;  // Normal halo
            }
        } else if(run_params->FeedbackFreeModeOn == 3) {
            // BK25 acceleration-based method using galaxy's stored concentration
            // (Vmax/Vvir with infall freeze when ConcentrationOn=3)
            const double Mvir = galaxies[p].Mvir;
            const double Rvir = galaxies[p].Rvir;

            if(Mvir <= 0.0 || Rvir <= 0.0) {
                galaxies[p].FFBRegime = 0;
                galaxies[p].g_max = 0.0;
                continue;
            }

            double c = (double)galaxies[p].Concentration;
            if(c < 1.0) c = 1.0;

            const double g_vir = run_params->G * Mvir / (Rvir * Rvir);
            const double mu_c = log(1.0 + c) - c / (1.0 + c);
            const double g_max = (g_vir / mu_c) * (c * c / 2.0);

            galaxies[p].g_max = g_max;

            if(g_max > g_crit) {
                galaxies[p].FFBRegime = 1;  // FFB halo - above critical acceleration
            } else {
                galaxies[p].FFBRegime = 0;  // Normal halo
            }
        } else if(run_params->FeedbackFreeModeOn == 4) {
            // BK25 acceleration-based with log-normal concentration scatter.
            // The Ishiyama+21 table gives the mean concentration; individual halos
            // scatter around it following p(c)dc ∝ exp(-(ln c - ln c0)^2 / 2σ_c^2) d(ln c)
            // with σ_c ≈ 0.2 (Jing 2000; Bullock+01; Dolag+04).
            // The persistent FFBRandom draws a fixed quantile for each halo,
            // giving a deterministic scattered concentration and thus a smooth
            // FFb transition across the halo population.
            const double Mvir = galaxies[p].Mvir;
            const double Rvir = galaxies[p].Rvir;

            if(Mvir <= 0.0 || Rvir <= 0.0) {
                galaxies[p].FFBRegime = 0;
                galaxies[p].g_max = 0.0;
                continue;
            }

            // Mean concentration from Ishiyama+21 lookup table
            const double Mvir_Msun_h = Mvir * 1.0e10;
            const double logM = log10(Mvir_Msun_h);
            double c = interpolate_concentration_ishiyama21(logM, Zcurr, run_params);
            if(c < 1.0) c = 1.0;

            // Apply log-normal scatter: ln(c) ~ Normal(ln(c_mean), σ_c)
            if(run_params->FFBConcSigma > 0.0) {
                double u = (double)galaxies[p].FFBRandom;
                if(u < 1.0e-6) u = 1.0e-6;
                if(u > 1.0 - 1.0e-6) u = 1.0 - 1.0e-6;
                const double z_normal = inverse_normal_cdf(u);
                c = c * exp(run_params->FFBConcSigma * z_normal);
                if(c < 1.0) c = 1.0;
            }

            // g_max with scattered concentration (BK25 Eq. 4)
            const double g_vir = run_params->G * Mvir / (Rvir * Rvir);
            const double mu_c = log(1.0 + c) - c / (1.0 + c);
            const double g_max = (g_vir / mu_c) * (c * c / 2.0);

            galaxies[p].g_max = g_max;

            if(g_max > g_crit) {
                galaxies[p].FFBRegime = 1;  // FFB halo
            } else {
                galaxies[p].FFBRegime = 0;  // Normal halo
            }
        } else if(run_params->FeedbackFreeModeOn == 5) {
            // Li et al. 2024 mass-based method with hard cutoff (no sigmoid)
            // FFB regime when Mvir > Mvir_ffb (sharp threshold)
            const double Mvir = galaxies[p].Mvir;
            const double Mvir_ffb = calculate_ffb_threshold_mass(Zcurr, run_params);

            if(Mvir > Mvir_ffb) {
                galaxies[p].FFBRegime = 1;  // FFB halo - above threshold mass
            } else {
                galaxies[p].FFBRegime = 0;  // Normal halo
            }
        } else if(run_params->FeedbackFreeModeOn == 6) {
            // Li+24 sigmoid + H2-based SF (same regime detection as mode 1)
            const double Mvir = galaxies[p].Mvir;
            const double f_ffb = calculate_ffb_fraction(Mvir, Zcurr, run_params);
            const double random_uniform = (double)galaxies[p].FFBRandom;

            if(random_uniform < f_ffb) {
                galaxies[p].FFBRegime = 1;  // FFB halo
            } else {
                galaxies[p].FFBRegime = 0;  // Normal halo
            }
        } else if(run_params->FeedbackFreeModeOn == 7) {
            // BK25 log-normal c scatter + H2-based SF (same regime detection as mode 4)
            const double Mvir = galaxies[p].Mvir;
            const double Rvir = galaxies[p].Rvir;

            if(Mvir <= 0.0 || Rvir <= 0.0) {
                galaxies[p].FFBRegime = 0;
                galaxies[p].g_max = 0.0;
                continue;
            }

            const double Mvir_Msun_h = Mvir * 1.0e10;
            const double logM = log10(Mvir_Msun_h);
            double c = interpolate_concentration_ishiyama21(logM, Zcurr, run_params);
            if(c < 1.0) c = 1.0;

            if(run_params->FFBConcSigma > 0.0) {
                double u = (double)galaxies[p].FFBRandom;
                if(u < 1.0e-6) u = 1.0e-6;
                if(u > 1.0 - 1.0e-6) u = 1.0 - 1.0e-6;
                const double z_normal = inverse_normal_cdf(u);
                c = c * exp(run_params->FFBConcSigma * z_normal);
                if(c < 1.0) c = 1.0;
            }

            const double g_vir = run_params->G * Mvir / (Rvir * Rvir);
            const double mu_c = log(1.0 + c) - c / (1.0 + c);
            const double g_max = (g_vir / mu_c) * (c * c / 2.0);

            galaxies[p].g_max = g_max;

            if(g_max > g_crit) {
                galaxies[p].FFBRegime = 1;  // FFB halo
            } else {
                galaxies[p].FFBRegime = 0;  // Normal halo
            }
        }
    }
}

double interpolate_concentration_ishiyama21(const double logM, const double z, const struct params *run_params)
{
    // Ishiyama+21 concentration-mass lookup table (mdef=200c, halo_sample=all, c_type=fit)
    // Two tables available, auto-selected by Omega:
    //   Omega < 0.28 -> Millennium (WMAP1): H0=73, Om=0.25, sigma8=0.9, ns=1.0
    //   Omega >= 0.28 -> Uchuu (Planck15):  H0=67.74, Om=0.3089, sigma8=0.8159, ns=0.9667
    // Both tables share the same mass/redshift grid.
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

    // Uchuu / Planck15 cosmology: H0=67.74, Om=0.3089, Ob=0.0486, sigma8=0.8159, ns=0.9667
    const double cm_uchuu_table[CM_TABLE_N_MASS][CM_TABLE_N_Z] = {
        {17.803, 13.795, 10.682, 8.500, 6.980, 5.898, 5.108, 4.517, 4.068, 3.721, 3.450, 3.238, 3.072, 2.941, 2.839, 2.761, 2.702, 2.660, 2.632, 2.615, 2.608, 2.611, 2.621, 2.638, 2.661, 2.689, 2.722, 2.759, 2.801, 2.846, 2.894},
        {17.271, 13.378, 10.356, 8.242, 6.773, 5.728, 4.967, 4.400, 3.970, 3.640, 3.383, 3.183, 3.028, 2.907, 2.814, 2.745, 2.694, 2.659, 2.638, 2.628, 2.628, 2.636, 2.653, 2.675, 2.704, 2.738, 2.777, 2.820, 2.867, 2.917, 2.971},
        {16.743, 12.963, 10.034, 7.988, 6.568, 5.561, 4.830, 4.286, 3.876, 3.562, 3.319, 3.132, 2.988, 2.877, 2.794, 2.733, 2.690, 2.663, 2.649, 2.646, 2.653, 2.668, 2.690, 2.719, 2.754, 2.794, 2.839, 2.888, 2.941, 2.997, 3.056},
        {16.218, 12.551, 9.714, 7.736, 6.366, 5.397, 4.695, 4.175, 3.784, 3.487, 3.259, 3.085, 2.952, 2.851, 2.777, 2.725, 2.691, 2.672, 2.665, 2.670, 2.684, 2.706, 2.735, 2.771, 2.812, 2.858, 2.909, 2.965, 3.024, 3.086, 3.151},
        {15.695, 12.141, 9.396, 7.486, 6.166, 5.235, 4.563, 4.067, 3.696, 3.416, 3.202, 3.041, 2.920, 2.830, 2.766, 2.723, 2.698, 2.687, 2.688, 2.700, 2.722, 2.751, 2.787, 2.830, 2.878, 2.931, 2.989, 3.051, 3.116, 3.185, 3.257},
        {15.173, 11.733, 9.081, 7.239, 5.969, 5.076, 4.434, 3.962, 3.611, 3.348, 3.150, 3.002, 2.892, 2.814, 2.760, 2.726, 2.710, 2.708, 2.718, 2.738, 2.767, 2.804, 2.848, 2.898, 2.953, 3.014, 3.078, 3.147, 3.220, 3.296, 3.375},
        {14.654, 11.327, 8.768, 6.994, 5.775, 4.920, 4.308, 3.861, 3.530, 3.284, 3.102, 2.967, 2.870, 2.803, 2.759, 2.736, 2.729, 2.736, 2.754, 2.783, 2.820, 2.866, 2.918, 2.975, 3.039, 3.107, 3.179, 3.256, 3.336, 3.419, 3.506},
        {14.137, 10.924, 8.457, 6.752, 5.584, 4.767, 4.186, 3.763, 3.454, 3.225, 3.058, 2.937, 2.853, 2.798, 2.765, 2.752, 2.755, 2.772, 2.799, 2.837, 2.883, 2.937, 2.998, 3.064, 3.135, 3.212, 3.292, 3.377, 3.465, 3.557, 3.651},
        {13.628, 10.527, 8.153, 6.516, 5.398, 4.620, 4.069, 3.671, 3.382, 3.172, 3.021, 2.914, 2.843, 2.800, 2.779, 2.777, 2.790, 2.816, 2.854, 2.901, 2.957, 3.020, 3.089, 3.165, 3.245, 3.330, 3.420, 3.513, 3.610, 3.710, 3.813},
        {13.125, 10.137, 7.854, 6.285, 5.217, 4.478, 3.957, 3.584, 3.317, 3.125, 2.990, 2.898, 2.840, 2.809, 2.801, 2.810, 2.834, 2.871, 2.919, 2.976, 3.042, 3.115, 3.194, 3.279, 3.369, 3.464, 3.563, 3.665, 3.772, 3.881, 3.994},
        {12.629, 9.753, 7.560, 6.059, 5.041, 4.341, 3.851, 3.504, 3.257, 3.084, 2.966, 2.889, 2.846, 2.828, 2.832, 2.853, 2.889, 2.937, 2.996, 3.064, 3.140, 3.224, 3.314, 3.409, 3.509, 3.614, 3.723, 3.836, 3.953, 4.073, 4.195},
        {12.139, 9.373, 7.272, 5.838, 4.871, 4.209, 3.750, 3.429, 3.205, 3.051, 2.949, 2.889, 2.860, 2.856, 2.873, 2.907, 2.955, 3.015, 3.086, 3.166, 3.254, 3.349, 3.450, 3.556, 3.668, 3.784, 3.904, 4.028, 4.156, 4.287, 4.420},
        {11.655, 9.000, 6.989, 5.623, 4.707, 4.084, 3.656, 3.361, 3.159, 3.025, 2.942, 2.897, 2.884, 2.895, 2.925, 2.973, 3.035, 3.108, 3.191, 3.284, 3.384, 3.491, 3.604, 3.723, 3.847, 3.975, 4.107, 4.244, 4.383, 4.526, 4.672},
        {11.175, 8.630, 6.711, 5.413, 4.547, 3.964, 3.569, 3.300, 3.121, 3.008, 2.943, 2.916, 2.919, 2.945, 2.991, 3.053, 3.129, 3.216, 3.313, 3.419, 3.533, 3.654, 3.780, 3.912, 4.049, 4.191, 4.336, 4.486, 4.639, 4.795, 4.954},
        {10.699, 8.266, 6.438, 5.208, 4.394, 3.851, 3.488, 3.247, 3.092, 3.000, 2.955, 2.946, 2.966, 3.009, 3.071, 3.149, 3.240, 3.342, 3.454, 3.575, 3.703, 3.839, 3.980, 4.127, 4.278, 4.434, 4.594, 4.758, 4.926, 5.096, 5.270},
        {10.228, 7.907, 6.171, 5.009, 4.248, 3.746, 3.416, 3.203, 3.073, 3.003, 2.979, 2.989, 3.028, 3.088, 3.168, 3.262, 3.370, 3.489, 3.617, 3.754, 3.899, 4.050, 4.207, 4.370, 4.537, 4.709, 4.885, 5.065, 5.249, 5.436, 5.626},
        {9.762, 7.553, 5.909, 4.817, 4.108, 3.648, 3.352, 3.169, 3.065, 3.019, 3.016, 3.047, 3.105, 3.185, 3.283, 3.396, 3.522, 3.659, 3.805, 3.959, 4.122, 4.291, 4.466, 4.646, 4.831, 5.020, 5.214, 5.412, 5.613, 5.818, 6.025},
        {9.301, 7.205, 5.654, 4.632, 3.977, 3.559, 3.299, 3.146, 3.069, 3.048, 3.068, 3.122, 3.201, 3.302, 3.420, 3.553, 3.699, 3.855, 4.021, 4.195, 4.377, 4.566, 4.760, 4.960, 5.164, 5.373, 5.587, 5.804, 6.025, 6.249, 6.477},
        {8.852, 6.867, 5.409, 4.458, 3.857, 3.482, 3.258, 3.137, 3.089, 3.094, 3.139, 3.216, 3.319, 3.442, 3.583, 3.737, 3.905, 4.083, 4.271, 4.467, 4.670, 4.880, 5.096, 5.317, 5.544, 5.775, 6.010, 6.249, 6.492, 6.738, 6.987},
        {8.414, 6.541, 5.175, 4.295, 3.748, 3.418, 3.232, 3.144, 3.125, 3.158, 3.231, 3.334, 3.462, 3.610, 3.774, 3.954, 4.145, 4.348, 4.559, 4.779, 5.006, 5.241, 5.481, 5.726, 5.977, 6.232, 6.491, 6.755, 7.022, 7.293, 7.567},
        {7.987, 6.226, 4.953, 4.143, 3.653, 3.368, 3.221, 3.168, 3.182, 3.245, 3.347, 3.478, 3.633, 3.809, 4.001, 4.207, 4.425, 4.654, 4.893, 5.139, 5.393, 5.654, 5.922, 6.194, 6.472, 6.754, 7.041, 7.332, 7.627, 7.925, 8.227},
        {7.571, 5.921, 4.742, 4.005, 3.571, 3.334, 3.229, 3.213, 3.261, 3.357, 3.491, 3.653, 3.839, 4.045, 4.266, 4.503, 4.751, 5.010, 5.278, 5.555, 5.840, 6.131, 6.429, 6.732, 7.040, 7.353, 7.670, 7.992, 8.318, 8.647, 8.980},
        {7.166, 5.628, 4.543, 3.880, 3.506, 3.319, 3.257, 3.281, 3.367, 3.499, 3.667, 3.864, 4.084, 4.323, 4.579, 4.849, 5.131, 5.424, 5.726, 6.037, 6.355, 6.681, 7.013, 7.350, 7.693, 8.041, 8.393, 8.750, 9.111, 9.476, 9.844},
        {6.772, 5.348, 4.359, 3.771, 3.458, 3.324, 3.308, 3.376, 3.503, 3.675, 3.882, 4.117, 4.376, 4.653, 4.947, 5.255, 5.575, 5.906, 6.247, 6.596, 6.953, 7.318, 7.688, 8.065, 8.447, 8.834, 9.226, 9.623, 10.024, 10.429, 10.837},
        {6.390, 5.081, 4.189, 3.680, 3.432, 3.353, 3.388, 3.503, 3.676, 3.892, 4.143, 4.421, 4.723, 5.043, 5.380, 5.731, 6.095, 6.469, 6.853, 7.247, 7.648, 8.057, 8.472, 8.893, 9.320, 9.753, 10.190, 10.632, 11.079, 11.529, 11.983},
        {6.021, 4.829, 4.038, 3.610, 3.430, 3.411, 3.501, 3.668, 3.892, 4.158, 4.458, 4.785, 5.135, 5.505, 5.890, 6.291, 6.704, 7.128, 7.563, 8.006, 8.458, 8.917, 9.384, 9.856, 10.335, 10.819, 11.308, 11.802, 12.301, 12.804, 13.311},
        {5.670, 4.596, 3.908, 3.565, 3.457, 3.503, 3.654, 3.880, 4.160, 4.482, 4.838, 5.221, 5.627, 6.052, 6.494, 6.951, 7.422, 7.903, 8.395, 8.896, 9.406, 9.924, 10.450, 10.981, 11.519, 12.063, 12.612, 13.166, 13.726, 14.290, 14.858},
        {5.343, 4.389, 3.805, 3.551, 3.520, 3.637, 3.856, 4.147, 4.492, 4.878, 5.298, 5.745, 6.215, 6.705, 7.212, 7.734, 8.271, 8.819, 9.377, 9.945, 10.523, 11.109, 11.703, 12.303, 12.910, 13.523, 14.141, 14.766, 15.396, 16.030, 16.669},
        {5.041, 4.207, 3.733, 3.573, 3.627, 3.823, 4.117, 4.482, 4.901, 5.361, 5.854, 6.375, 6.920, 7.485, 8.068, 8.667, 9.280, 9.905, 10.541, 11.188, 11.845, 12.510, 13.183, 13.864, 14.551, 15.245, 15.946, 16.652, 17.364, 18.081, 18.803},
        {4.764, 4.055, 3.696, 3.638, 3.785, 4.069, 4.450, 4.900, 5.404, 5.950, 6.529, 7.137, 7.769, 8.422, 9.094, 9.782, 10.485, 11.201, 11.929, 12.668, 13.418, 14.177, 14.944, 15.719, 16.501, 17.290, 18.087, 18.889, 19.698, 20.512, 21.332},
        {4.514, 3.936, 3.700, 3.753, 4.005, 4.390, 4.871, 5.421, 6.024, 6.670, 7.350, 8.060, 8.796, 9.553, 10.329, 11.123, 11.933, 12.757, 13.594, 14.442, 15.301, 16.171, 17.049, 17.936, 18.831, 19.734, 20.643, 21.561, 22.484, 23.414, 24.349},
        {4.295, 3.856, 3.753, 3.931, 4.302, 4.804, 5.401, 6.068, 6.789, 7.553, 8.354, 9.185, 10.043, 10.924, 11.826, 12.746, 13.684, 14.636, 15.602, 16.580, 17.571, 18.572, 19.584, 20.604, 21.634, 22.671, 23.716, 24.770, 25.831, 26.898, 27.971},
        {4.113, 3.823, 3.867, 4.186, 4.695, 5.334, 6.069, 6.874, 7.735, 8.642, 9.586, 10.563, 11.568, 12.598, 13.649, 14.721, 15.812, 16.918, 18.040, 19.175, 20.323, 21.483, 22.654, 23.836, 25.026, 26.226, 27.434, 28.652, 29.877, 31.110, 32.350},
        {3.974, 3.847, 4.057, 4.539, 5.210, 6.011, 6.910, 7.882, 8.912, 9.990, 11.107, 12.260, 13.443, 14.652, 15.885, 17.140, 18.416, 19.709, 21.019, 22.343, 23.682, 25.035, 26.399, 27.775, 29.161, 30.557, 31.963, 33.379, 34.804, 36.237, 37.677},
        {3.886, 3.941, 4.341, 5.014, 5.877, 6.874, 7.971, 9.144, 10.379, 11.665, 12.994, 14.360, 15.759, 17.187, 18.641, 20.120, 21.621, 23.142, 24.680, 26.235, 27.806, 29.393, 30.993, 32.605, 34.229, 35.865, 37.511, 39.169, 40.837, 42.514, 44.199},
        {3.863, 4.124, 4.747, 5.647, 6.742, 7.975, 9.313, 10.733, 12.219, 13.759, 15.346, 16.974, 18.639, 20.335, 22.060, 23.812, 25.590, 27.390, 29.209, 31.047, 32.903, 34.777, 36.666, 38.568, 40.484, 42.412, 44.354, 46.308, 48.274, 50.250, 52.235},
        {3.922, 4.423, 5.308, 6.482, 7.859, 9.382, 11.017, 12.741, 14.536, 16.391, 18.296, 20.247, 22.240, 24.267, 26.327, 28.417, 30.537, 32.681, 34.847, 37.034, 39.242, 41.471, 43.716, 45.977, 48.253, 50.543, 52.849, 55.169, 57.502, 59.847, 62.203},
        {4.087, 4.869, 6.071, 7.580, 9.305, 11.186, 13.191, 15.291, 17.470, 19.716, 22.018, 24.372, 26.771, 29.211, 31.687, 34.198, 36.742, 39.314, 41.912, 44.534, 47.180, 49.849, 52.537, 55.244, 57.967, 60.708, 63.466, 66.241, 69.031, 71.835, 74.651},
        {4.386, 5.507, 7.097, 9.021, 11.180, 13.510, 15.975, 18.547, 21.208, 23.944, 26.744, 29.601, 32.512, 35.468, 38.465, 41.504, 44.580, 47.688, 50.826, 53.993, 57.187, 60.408, 63.652, 66.916, 70.201, 73.505, 76.830, 80.174, 83.537, 86.914, 90.308},
        {4.862, 6.398, 8.470, 10.914, 13.619, 16.514, 19.562, 22.731, 26.000, 29.355, 32.782, 36.276, 39.832, 43.440, 47.097, 50.801, 54.549, 58.335, 62.155, 66.008, 69.895, 73.813, 77.757, 81.726, 85.718, 89.734, 93.773, 97.837, 101.920, 106.023, 110.144},
        {5.566, 7.620, 10.299, 13.401, 16.798, 20.412, 24.200, 28.127, 32.169, 36.309, 40.535, 44.837, 49.212, 53.648, 58.141, 62.689, 67.290, 71.935, 76.621, 81.346, 86.110, 90.911, 95.744, 100.605, 105.494, 110.412, 115.357, 120.331, 125.330, 130.350, 135.393}
    };

    // Auto-select table based on Omega (Uchuu=0.3089; default to Millennium/WMAP1)
    const double (*table)[CM_TABLE_N_Z] = cm_table;
    if(fabs(run_params->Omega - 0.3089) < 0.01) {
        table = cm_uchuu_table;
    }

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
    const double c00 = table[im][iz];
    const double c10 = table[im + 1][iz];
    const double c01 = table[im][iz + 1];
    const double c11 = table[im + 1][iz + 1];

    return c00 * (1.0 - fm) * (1.0 - fz)
         + c10 * fm * (1.0 - fz)
         + c01 * (1.0 - fm) * fz
         + c11 * fm * fz;
}

double concentration_from_vmax_vvir(const double Vmax, const double Vvir)
{
    // Invert the NFW relation: (Vmax/Vvir)^2 = 0.2162 * c / mu(c)
    // where mu(c) = ln(1+c) - c/(1+c)
    // Uses bisection since the RHS is monotonically increasing in c.

    if(Vvir <= 0.0 || Vmax <= 0.0) return 0.0;

    const double ratio_sq = (Vmax / Vvir) * (Vmax / Vvir);

    // Target function: f(c) = 0.2162 * c / mu(c) - ratio_sq
    // The constant 0.2162 = f(s_max)/s_max where s_max ~ 2.1626 is the
    // location of max V_circ for an NFW profile.
    const double A = 0.21621;  // f(2.1626)/2.1626

    double c_lo = 1.0, c_hi = 200.0;

    // Quick sanity: if ratio < 1 it's unphysical for NFW, return 0 (unresolved)
    if(ratio_sq < 1.0) return 0.0;

    for(int iter = 0; iter < 60; iter++) {
        const double c_mid = 0.5 * (c_lo + c_hi);
        const double mu = log(1.0 + c_mid) - c_mid / (1.0 + c_mid);
        const double f_mid = A * c_mid / mu;
        if(f_mid < ratio_sq) {
            c_lo = c_mid;
        } else {
            c_hi = c_mid;
        }
    }
    return 0.5 * (c_lo + c_hi);
}

double get_halo_concentration(const int p, const double z, const struct GALAXY *galaxies,
                               const struct params *run_params)
{
    if(run_params->ConcentrationOn == 0) {
        return 0.0;  // Concentration disabled
    }

    if(run_params->ConcentrationOn == 2) {
        // Vmax/Vvir from simulation for all galaxy types
        return concentration_from_vmax_vvir(galaxies[p].Vmax, galaxies[p].Vvir);
    }

    if(run_params->ConcentrationOn == 3) {
        // Hybrid: Vmax/Vvir for centrals, infall-frozen Vmax/Vvir for satellites
        double vmax, vvir;
        if(galaxies[p].Type == 0) {
            vmax = galaxies[p].Vmax;
            vvir = galaxies[p].Vvir;
        } else {
            vmax = galaxies[p].infallVmax;
            vvir = galaxies[p].infallVvir;
        }
        return concentration_from_vmax_vvir(vmax, vvir);
    }

    // ConcentrationOn == 1 or 3 (satellite): Ishiyama+21 lookup table
    const double Mvir = galaxies[p].Mvir;  // 10^10 M_sun / h
    if(Mvir <= 0.0) return 0.0;

    const double Mvir_Msun_h = Mvir * 1.0e10;  // Msun/h (table units)
    const double logM = log10(Mvir_Msun_h);

    double c = interpolate_concentration_ishiyama21(logM, z, run_params);
    if(c < 1.0) c = 0.0;  // table extrapolation failure
    return c;
}

double calculate_gmax_BK25(const int p, const double z, const struct GALAXY *galaxies,
                            const struct params *run_params)
{
    // Boylan-Kolchin 2025: maximum NFW gravitational acceleration
    //
    // g_vir = G * M_vir / R_vir^2                                (Eq. 2)
    // g_max = (g_vir / mu(c)) * (c^2 / 2)                         (Eq. 4)
    // where mu(x) = ln(1+x) - x/(1+x)
    //
    // Always uses the Ishiyama+21 lookup table concentration for the FFB
    // threshold, even when ConcentrationOn=2 (Vmax/Vvir).  The BK25 threshold
    // is derived from average halo properties; using individual scatter would
    // produce spurious FFB activation at low redshift.
    //
    // Returns g_max in code units (UnitLength / UnitTime^2)

    const double Mvir = galaxies[p].Mvir;  // code mass units (10^10 M_sun / h)
    const double Rvir = galaxies[p].Rvir;  // code length units (Mpc / h)

    if(Mvir <= 0.0 || Rvir <= 0.0) {
        return 0.0;
    }

    // g_vir = G * M_vir / R_vir^2  (code units)
    const double g_vir = run_params->G * Mvir / (Rvir * Rvir);

    // Always use the lookup table concentration for the FFB determination
    const double Mvir_Msun_h = Mvir * 1.0e10;
    const double logM = log10(Mvir_Msun_h);
    double c = interpolate_concentration_ishiyama21(logM, z, run_params);
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
    if (sigma_gas <= 0.0 || disk_scale_length_pc <= 0.0) {
        return 0.0;
    }

    // For very low stellar surface density, use a minimal value to avoid numerical issues
    float effective_sigma_stars = sigma_stars;
    if (sigma_stars < 0.1) {
        effective_sigma_stars = 0.1;
    }

    // Calculate stellar scale height using exact BR06 equation (9)
    float h_star_pc = calculate_stellar_scale_height_BR06(disk_scale_length_pc);
    if (h_star_pc <= 0.0) return 0.0;

    const float v_g = 8.0;  // km/s, gas velocity dispersion (BR06)

    // BR06 Equation (5) - stellar-dominated approximation:
    // P_ext/k = 272 × Σ_gas × √Σ_* × v_g × h_*^(-0.5)
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
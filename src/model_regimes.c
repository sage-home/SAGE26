/*
 * model_regimes.c -- CGM/hot-halo and feedback-free-burst regime classification.
 *
 * determine_and_store_regime() implements the Dekel & Birnboim (2006)
 * shock-mass criterion; determine_and_store_ffb_regime() implements the
 * Li+24 and BK25 feedback-free burst thresholds with optional lognormal
 * concentration scatter.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "core_allvars.h"

#include "model_misc.h"

#include "core_cool_func.h"

#include "model_cooling_heating.h"

/* -------------------------------------------------------------------------
 * File-scope empirical constants (lifted per STYLE_C.md SS8).
 * -------------------------------------------------------------------------*/

/* Dekel & Birnboim (2006) critical virial-shock stability mass: halos below it
 * lack a stable virial shock and are classified as CGM-regime.  Default
 * 6e11 Msun from DB06 Fig. 1 / eq. 4, now settable as MShockMsun in the
 * parameter file so it can be varied without a rebuild. */

/* Parsec in cm (IAU 2012).  Used when converting radii between Mpc/h
 * (code units) and pc for surface-density calculations. */
static const double PC_IN_CM              =  3.08568e18;

/* Boylan-Kolchin (2025) Table 1: critical gravitational acceleration for FFB.
 * Units: M_sun / pc^2 (pre-multiplication by G to get acceleration). */
static const double BK25_G_CRIT_MSUN_PC2 = 3100.0;


/*
 * Classify each central galaxy as CGM-regime or hot-halo regime (Voit 2015).
 *
 * Uses the Dekel & Birnboim (2006) criterion: halos below ~6e11 M_sun are in
 * the CGM regime (Regime == 0); more massive halos are in the hot regime
 * (Regime == 1). Regime is stored on the galaxy struct and used by
 * cooling_recipe_regime_aware() and model_infall.c.
 */
void determine_and_store_regime(const int ngal, struct GALAXY *galaxies,
                                const struct params *run_params)
{
    for(int p = 0; p < ngal; p++) {
        if(galaxies[p].mergeType > 0) continue;

        // Convert Mvir to physical units (Msun)
        // Mvir is stored in units of 10^10 Msun/h
        const double Mvir_physical = CODE_MASS_TO_MSUN(galaxies[p].Mvir, run_params->Hubble_h);

        // Shock mass threshold (Dekel & Birnboim 2006)
        const double Mshock = run_params->MShockMsun;  // Msun

        // Calculate mass ratio for sigmoid
        const double mass_ratio = Mvir_physical / Mshock;

        int32_t new_regime;
        if(mass_ratio <= 0.0) {
            new_regime = 0;  // Default to CGM regime for invalid mass
        } else {
            // Smooth sigmoid transition (consistent with FFB approach)
            // Width of transition in dex
            const double delta_log_M = 0.1;

            // Sigmoid argument: x = log10(M/Mshock) / width
            const double x = log10(mass_ratio) / delta_log_M;

            // Sigmoid function: probability of being in Hot regime
            // Smoothly varies from 0 (well below Mshock) to 1 (well above Mshock)
            const double hot_fraction = 1.0 / (1.0 + exp(-x));

            // RegimeRandomMode=1: reuse the persistent RegimeRandom draw from
            // galaxy creation, so the regime evolves deterministically with
            // Mvir relative to a fixed per-galaxy quantile and never thrashes
            // for borderline-mass centrals.
            // RegimeRandomMode=0: fresh draw each snapshot (original behaviour).
            const double random_uniform = (run_params->RegimeRandomMode == 1)
                ? (double)galaxies[p].RegimeRandom
                : (double)rand() / (double)RAND_MAX;
            new_regime = (random_uniform < hot_fraction) ? 1 : 0;
        }

        galaxies[p].Regime = new_regime;
    }
}

// void determine_and_store_regime(const int ngal, struct GALAXY *galaxies,
//                                 const struct params *run_params)
// {
//     for(int p = 0; p < ngal; p++) {
//         // Skip satellites
//         if(galaxies[p].mergeType > 0) continue;

//         int32_t new_regime = 1; // Default to CGM/Cold accretion regime

//         // Recombine the total diffuse gas to replicate SAGE16's unified hot reservoir
//         double total_halo_gas = galaxies[p].HotGas + galaxies[p].CGMgas;
//         double total_metals = galaxies[p].MetalsHotGas + galaxies[p].MetalsCGMgas;

//         if(total_halo_gas > 0.0 && galaxies[p].Vvir > 0.0) {
//             const double tcool = galaxies[p].Rvir / galaxies[p].Vvir;
//             const double temp = 35.9 * galaxies[p].Vvir * galaxies[p].Vvir; // in Kelvin

//             double logZ = -10.0;
//             if(total_metals > 0.0) {
//                 logZ = log10(total_metals / total_halo_gas);
//             }

//             double lambda = get_metaldependent_cooling_rate(log10(temp), logZ);

//             if(lambda > 0.0) {
//                 double x = PROTONMASS * BOLTZMANN * temp / lambda;
//                 x /= (run_params->UnitDensity_in_cgs * run_params->UnitTime_in_s);
//                 const double rho_rcool = x / tcool * 0.885;

//                 if(rho_rcool > 0.0) {
//                     // Density profile is based on the entire diffuse gas envelope
//                     const double rho0 = total_halo_gas / (4.0 * M_PI * galaxies[p].Rvir);
//                     const double rcool = sqrt(rho0 / rho_rcool);

//                     // SAGE16 Classification
//                     if(rcool > galaxies[p].Rvir) {
//                         new_regime = 0; // Cold accretion / CGM regime
//                     } else {
//                         new_regime = 1; // Hot halo cooling regime
//                     }
//                 } else {
//                     new_regime = 0; // rho_rcool -> 0 implies rcool -> infinity (Cold regime)
//                 }
//             } else {
//                 new_regime = 1; // No cooling (lambda <= 0) implies rcool = 0 (Hot regime)
//             }
//         }

//         galaxies[p].Regime = new_regime;
//     }
// }

/*
 * Inverse normal CDF (probit function) via Peter Acklam's rational approximation.
 *
 * Converts a uniform variate p in (0, 1) to a standard normal variate.
 * Accurate to ~1e-9 across the full range.
 */
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

/*
 * Classify galaxies as feedback-free burst (FFB) or normal mode.
 *
 * When FeedbackFreeModeOn > 0, evaluates each central galaxy against the FFB
 * mass and redshift criteria (Li+2024) and sets galaxies[p].FFBmode.
 * Uses a lognormal scatter (via inverse_normal_cdf) around the threshold when
 * the scatter mode is enabled. Skips all galaxies when FFBmodeOn == 0.
 */
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
        const double pc_code = PC_IN_CM / run_params->UnitLength_in_cm;
        g_crit = run_params->G * BK25_G_CRIT_MSUN_PC2 * Msun_code / (pc_code * pc_code) / run_params->Hubble_h;
    }

    // Classify each galaxy as FFB or normal
    for(int p = 0; p < ngal; p++) {
        if(galaxies[p].mergeType > 0) continue;

        // By default, only CGM-regime halos are eligible for FFB.
        // FFBIgnoreRegime=1 removes this restriction, letting the Li+24/BK25
        // criteria apply regardless of halo regime.
        if(galaxies[p].Regime == 1 && !run_params->FFBIgnoreRegime) {
            galaxies[p].FFBRegime = 0;
            continue;
        }

        // FFBRandomMode=1: reuse the persistent draw assigned at galaxy creation.
        // FFBRandomMode=0: fresh draw each snapshot (no memory across timesteps).
        const double draw = (run_params->FFBRandomMode == 1)
            ? (double)galaxies[p].FFBRandom
            : (double)rand() / (double)RAND_MAX;

        if(run_params->FeedbackFreeModeOn == 1) {
            // Li et al. 2024 mass-based method (original)
            const double Mvir = galaxies[p].Mvir;

            // Calculate smooth FFB fraction using sigmoid transition (Li et al. 2024, eq. 3)
            const double f_ffb = calculate_ffb_fraction(Mvir, Zcurr, run_params);

            const double random_uniform = draw;

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
            // scatter around it following p(c)dc ~ exp(-(ln c - ln c0)^2 / 2sigma_c^2) d(ln c)
            // with sigma_c ~ 0.2 (Jing 2000; Bullock+01; Dolag+04).
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

            // Apply log-normal scatter: ln(c) ~ Normal(ln(c_mean), sigma_c)
            if(run_params->FFBConcSigma > 0.0) {
                double u = draw;
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
            const double random_uniform = draw;

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
                double u = draw;
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

/*
 * FFB virial mass threshold at redshift z (Li+2024 eq. 2).
 *
 * M_v,ffb / 10^10.8 Msun ~ ((1+z)/10)^-6.2.  Returns the threshold in
 * code units (10^10 Msun/h).
 */
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
    /* FFBThresholdSlope defaults to -6.2 (Li+24). The 10^10.8 normalisation is
     * pinned at z = 9, where z_norm = 1, so changing the slope pivots the
     * threshold about that redshift rather than shifting it wholesale. */
    const double log_Mvir_ffb_code = 0.8 + log10(h) + run_params->FFBThresholdSlope * log10(z_norm);

    return pow(10.0, log_Mvir_ffb_code);
}

/*
 * Fraction of galaxies in the FFB regime at (Mvir, z) via Li+2024 eq. (3).
 *
 * Returns a sigmoid value in [0, 1] that rises sharply as Mvir approaches
 * the FFB threshold; returns 0 when FeedbackFreeModeOn == 0.
 */
double calculate_ffb_fraction(const double Mvir, const double z, const struct params *run_params)
{
    // Calculate the fraction of galaxies in FFB regime
    // Uses smooth sigmoid transition from Li et al. 2024, equation (3)
    
    if (run_params->FeedbackFreeModeOn == 0) {
        return 0.0;
    }

    const double Mvir_ffb = calculate_ffb_threshold_mass(z, run_params);

    if(Mvir <= 0.0 || Mvir_ffb <= 0.0) {
        return 0.0;
    }

    /* Li+2024 eq. (3): sigmoid over 0.15 dex around M_v,ffb. */
    const double delta_log_M = 0.15;
    const double x = log10(Mvir / Mvir_ffb) / delta_log_M;
    const double f_ffb = 1.0 / (1.0 + exp(-x));

    return f_ffb;
}

/*
 * Maximum NFW gravitational acceleration g_max (Boylan-Kolchin 2025).
 *
 * Computes g_vir = G*M_vir/R_vir^2 and then the NFW peak factor from the
 * halo concentration, returning g_max in CGS units (cm/s^2).  Used as the
 * FFB feedback threshold in the FeedbackFreeModeOn == 4 prescription.
 */
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

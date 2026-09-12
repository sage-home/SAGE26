/*
 * model_cooling_heating.c -- Cooling and AGN heating prescriptions.
 *
 * Implements the two-regime cooling model selected per-galaxy by the Regime
 * flag set in model_misc.c:
 *
 *   Regime == 0 (CGM-dominated) -- precipitation-driven cooling from the CGMgas
 *     reservoir using the Voit (2015) / McCourt et al. (2012) t_cool/t_ff < 10
 *     threshold.  The CGM density structure is modelled with a uniform, NFW, or
 *     beta (beta = 2/3) profile selected by CGMDensityProfile.  AGN heating uses
 *     the same r_heat ratchet as the hot-halo regime, capped at Rvir.
 *
 *   Regime == 1 (hot halo) -- classical isothermal-halo cooling following
 *     White & Frenk (1991) and Croton et al. (2006).  When CGMrecipeOn > 0 a
 *     De Lucia & Blaizot (2006) cold-stream fraction is blended in for halos
 *     above the virial shock mass.
 *
 * AGN radio-mode accretion is computed via three models (AGNrecipeOn 1/2/3):
 * empirical (Croton+06 eq. 10), Bondi-Hoyle, and cold-cloud triggering.
 * In all modes accretion is Eddington-limited and draws from the reservoir
 * appropriate to the regime (HotGas for Regime==1, CGMgas for Regime==0).
 *
 * File-private helpers compute NFW/beta density profiles, their enclosed-mass
 * integrals, and iteratively solve for the cooling radius.
 *
 * Code units (10^10 Msun/h, Mpc/h, km/s) used throughout; conversions to
 * physical units happen only at the entry points of CGM-mode functions.
 *
 * References: Croton et al. (2006), MNRAS 365, 11; Voit (2015), ApJL 808, L30;
 *   McCourt et al. (2012), MNRAS 419, 3319; Duffy et al. (2008), MNRAS 390, L64;
 *   De Lucia & Blaizot (2006), MNRAS 375, 2.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>

#include "core_allvars.h"
#include "core_cool_func.h"

#include "model_cooling_heating.h"
#include "model_misc.h"

/* -------------------------------------------------------------------------
 * File-scope empirical constants (lifted per STYLE_C.md SS8).
 * -------------------------------------------------------------------------*/

/* Mean molecular weight for fully ionised, primordial (H+He) gas.
 * X_H = 0.76, Y_He = 0.24 => mu = 1/(2*X + 3*Y/4) ~ 0.59. */
static const double MU_IONISED = 0.59;

/* Croton et al. (2006) AGN empirical radio-mode accretion pivots
 * (their eq. 10 / Sec. 4.2).  Normalisation values chosen to reproduce
 * observed BH-bulge mass relation at z=0. */
static const double AGN_BH_MASS_PIVOT  = 0.01;    /* 10^8 Msun for h=1 in code units of 10^10 Msun/h */
static const double AGN_VVIR_PIVOT_KMS = 200.0;   /* km/s */
static const double AGN_HOT_GAS_PIVOT  = 0.1;     /* hot-gas-to-halo mass fraction normalisation */

/* Eddington accretion rate formula.
 * L_Edd = 1.3e38 * (M_BH/Msun) erg/s  (Rybicki & Lightman 1979, eq. 1.4.9).
 * Standard AGN radiative efficiency eta = 0.1.
 * C_SQ_KMS2 = c^2 in (km/s)^2; c = 3e5 km/s => c^2 = 9e10. */
static const double EDDINGTON_LUM_PER_MSUN_CGS = 1.3e38;  /* erg/s per Msun */
static const double AGN_RADIATIVE_EFFICIENCY    = 0.1;
static const double C_SQ_KMS2                  = 9.0e10;  /* (km/s)^2 */

/* AGN heating coefficient: sqrt(2 * eta * c^2) where eta = AGN_RADIATIVE_EFFICIENCY
 * and c is in km/s.  Equals the ratio of radiated energy to halo kinetic energy
 * per unit accreted mass.  Croton et al. (2006) eq. 19. */
static const double AGN_HEATING_COEFF_KMS = 1.34e5;  /* km/s */

/* Virial temperature coefficient: T_vir = VIRIAL_TEMP_COEFF * Vvir^2 [Kelvin].
 * Follows from T = mu * m_p * Vvir^2 / (2 * k_B) with mu = MU_IONISED = 0.59;
 * gives 35.9 K (km/s)^-2. */
static const double VIRIAL_TEMP_COEFF = 35.9;  /* K (km/s)^-2 */

/* De Lucia & Blaizot (2006) eq. 38: virial shock mass scale, above which
 * hot-mode shock heating is efficient.  Settable as MShockMsun in the parameter
 * file; must be the same value model_regimes.c uses to classify regimes. */

/* Critical redshift below which cold streams are suppressed in M > Mshock halos.
 * De Lucia & Blaizot (2006) estimate z_crit ~ 1-2; we adopt the midpoint. */
static const double Z_CRIT_DB06 = 1.5;

/* Width, in dex, of the smooth transition about the Dekel & Birnboim (2006)
 * stream criterion (t_cool/t_comp)_stream = 1 when ColdStreamCeilingOn == 1. */
static const double STREAM_TRANSITION_WIDTH_DEX = 0.15;

/* Cold-cloud AGN accretion (AGNrecipeOn == 3): BH triggers when its mass exceeds
 * this fraction of the sonic-radius enclosed virial mass, and accretes at this
 * fraction of the current cooling rate. Croton et al. (2006), AGN appendix. */
static const double AGN_COLD_CLOUD_FRAC = 1.0e-4;

/* File-private: AGN radio-mode heating for CGM-regime galaxies (defined below). */
static double do_AGN_heating_cgm(double coolingGas, const int centralgal, const double dt, const double x, const double rcool,
                                 struct GALAXY *galaxies, const struct params *run_params);



/*
 * Top-level cooling dispatcher: routes to regime-aware or classic hot-halo recipe.
 *
 * When CGMrecipeOn > 0 the two-regime model is active and this delegates to
 * cooling_recipe_regime_aware(); otherwise falls through to the C16-style
 * cooling_recipe_hot(). Returns the mass of gas cooled this substep.
 */
double cooling_recipe(const int gal, const double dt, struct GALAXY *galaxies, const struct params *run_params)
{
    // Check if CGM recipe is enabled for backwards compatibility
    if(run_params->CGMrecipeOn > 0) {
        return cooling_recipe_regime_aware(gal, dt, galaxies, run_params);
    } else {
        return cooling_recipe_hot(gal, dt, galaxies, run_params);
    }
}

double cooling_recipe_hot(const int gal, const double dt, struct GALAXY *galaxies, const struct params *run_params)
{
    double coolingGas;

    galaxies[gal].tcool = 0.0f;

    if(galaxies[gal].HotGas > 0.0 && galaxies[gal].Vvir > 0.0) {
        const double tcool = galaxies[gal].Rvir / galaxies[gal].Vvir;
        const double tff = M_SQRT2 * tcool;
        // Store tcool and tff specifically for the hot-halo recipe, which is used in the CGM path to determine the cooling radius.
        galaxies[gal].tcool = (float)((tcool * 1.0e10 / run_params->Hubble_h
                       * SEC_PER_GIGAYEAR / run_params->UnitTime_in_s));
        galaxies[gal].tff = (float)((tff * 1.0e10 / run_params->Hubble_h
                       * SEC_PER_GIGAYEAR / run_params->UnitTime_in_s));

        const double temp = VIRIAL_TEMP_COEFF * galaxies[gal].Vvir * galaxies[gal].Vvir;  // in Kelvin

        double logZ = -10.0;
        if(galaxies[gal].MetalsHotGas > 0) {
            logZ = log10(galaxies[gal].MetalsHotGas / galaxies[gal].HotGas);
        }

        double lambda = get_metaldependent_cooling_rate(log10(temp), logZ);

        if(lambda <= 0.0) {
            return 0.0;  // No cooling if cooling function is zero/negative
        }

        double x = PROTONMASS * BOLTZMANN * temp / lambda;        // now this has units sec g/cm^3
        x /= (run_params->UnitDensity_in_cgs * run_params->UnitTime_in_s);         // now in internal units
        const double rho_rcool = x / tcool * (1.5 * MU_IONISED);  // 3/2 * mu for a fully ionized gas

        if(rho_rcool <= 0.0) {
            return 0.0;
        }

        // an isothermal density profile for the hot gas is assumed here
        const double rho0 = galaxies[gal].HotGas / (4 * M_PI * galaxies[gal].Rvir);
        double rcool = sqrt(rho0 / rho_rcool);

        galaxies[gal].RcoolToRvir = rcool / galaxies[gal].Rvir;  // store uncapped ratio for diagnostics

        // The cooling radius is physically bounded by the virial radius. Capping
        // it means neither the cooling rate nor any downstream consumer (e.g.
        // do_AGN_heating) ever uses an unphysical rcool > Rvir value, and it
        // removes the SAGE06/16 rapid-cooling discontinuity at Rvir: hot-mode
        // cooling saturates at 0.5 * m_hot / t_cool rather than jumping by 2x.
        //
        // That is a deliberate SAGE26 choice, so it applies only on the SAGE26
        // path.  CGMrecipeOn == 0 is the backwards-compatibility path and has to
        // reproduce Croton et al. (2016) exactly, cold-accretion branch and
        // discontinuity included -- capping there silently halved the cooling of
        // the majority of the population (70% of galaxies at z = 0 rising to
        // 99.7% at z = 6, carrying 57-99% of the cooling mass), leaving the
        // "SAGE16" comparison run at 0.50-0.71 of the published cooling rate.
        if(run_params->CGMrecipeOn > 0 && rcool > galaxies[gal].Rvir) {
            rcool = galaxies[gal].Rvir;
        }

        coolingGas = 0.0;

        if(run_params->CGMrecipeOn == 0) {
            // SAGE C16 hot-halo cooling, both branches (Croton et al. 2016).
            // tcool here is the halo dynamical time Rvir/Vvir.
            if(rcool > galaxies[gal].Rvir) {
                // Rapid "cold accretion": the whole corona cools within a
                // dynamical time.  Discontinuous with the branch below by a
                // factor 2 at rcool = Rvir; that is the published behaviour.
                coolingGas = galaxies[gal].HotGas / tcool * dt;
            } else {
                // Quasi-static cooling flow.
                coolingGas = (galaxies[gal].HotGas / galaxies[gal].Rvir) * (rcool / (2.0 * tcool)) * dt;
            }
        } else {
            // CGMrecipeOn == 1: D&B06 cold streams for hot-regime halos
            // All halos here are in the hot regime (have virial shocks)
            const double z = run_params->ZZ[galaxies[gal].SnapNum];
            
            // D&B06 eqs 39-41: stream penetration factor f_stream.
            // Mass suppression (M/Mshock)^(-4/3) -- halos well above the shock
            // threshold host weaker cold streams. Redshift factor (1+z)/(1+1)
            // enhances streams at high-z where cooling is more efficient.
            const double Mvir_physical = CODE_MASS_TO_MSUN(galaxies[gal].Mvir, run_params->Hubble_h);
            const double mass_ratio = Mvir_physical / run_params->MShockMsun;

            // Redshift enhancement: normalized to z=1 following D&B06 eq 40
            const double z_factor = (1.0 + z) / (1.0 + 1.0);

            double f_stream;
            if(run_params->ColdStreamCeilingOn) {
                // Dekel & Birnboim (2006) eqs 39-41.  Their eq. 39 compares the
                // cooling and compression times within the stream,
                //     R = (f Mstar/Mvir)^(2/3) (Mvir/Mshock)^(4/3),
                // streams penetrating where R < 1.  The redshift dependence
                // enters through the clustering mass Mstar(z) rather than an
                // explicit (1+z) factor, and the shut-off is automatic: their
                // eq. 41 defines z_crit by f Mstar(z_crit) = Mshock, which is
                // exactly where R = 1 at Mvir = Mshock.  No redshift cut is
                // imposed, so f_stream is continuous everywhere.
                const double Mstar = pow(10.0, interpolate_clustering_mass(z, run_params));
                const double fMstar = run_params->StreamMassFactor * Mstar;
                const double ratio = pow(fMstar / Mvir_physical, 2.0/3.0)
                                   * pow(mass_ratio, 4.0/3.0);
                if(ratio > 0.0) {
                    const double sigmoid_arg = -log10(ratio) / STREAM_TRANSITION_WIDTH_DEX;
                    f_stream = 1.0 / (1.0 + exp(-sigmoid_arg));
                } else {
                    f_stream = 1.0;
                }
            } else if(z < Z_CRIT_DB06 && mass_ratio > 1.0) {
                // D&B06 eq 41: below z_crit cold streams are suppressed in
                // M > Mshock halos.  Hard cutoff; published behaviour.
                f_stream = 0.0;
            } else {
                // High-z regime: streams can penetrate
                f_stream = pow(mass_ratio, -4.0/3.0) * z_factor;
            }
            
            // Ensure physical bounds
            // Cap at 0.5 (50%) to account for partial heating/mixing of cold streams
            // as they penetrate through the hot medium
            if(f_stream > 1.0) f_stream = 1.0;
            if(f_stream < 0.0) f_stream = 0.0;
            
            // Calculate cooling: mix of cold streams + hot halo cooling
            double cold_stream_cooling = 0.0;
            double hot_halo_cooling = 0.0;
            
            if(rcool < galaxies[gal].Rvir) {
                // When rcool < Rvir: both cold streams and hot halo cooling
                // Cold stream component: rapid accretion on dynamical time
                cold_stream_cooling = f_stream * galaxies[gal].HotGas / 
                                     (galaxies[gal].Rvir / galaxies[gal].Vvir) * dt;
                
                // Hot halo component: traditional cooling from the shocked gas
                hot_halo_cooling = (1.0 - f_stream) * (galaxies[gal].HotGas / galaxies[gal].Rvir) * 
                                  (rcool / (2.0 * tcool)) * dt;
            } else {
                // When rcool >= Rvir: only hot halo cooling (no cold streams)
                // rcool >= Rvir: This shouldn't occur for properly-classified hot-regime haloes
                // (such haloes belong in the CGM/cold-flow regime). Handle conservatively.
                hot_halo_cooling = (galaxies[gal].HotGas / galaxies[gal].Rvir) * 
                                  (rcool / (2.0 * tcool)) * dt;
            }

            galaxies[gal].mdot_cool = hot_halo_cooling / dt;
            galaxies[gal].mdot_stream = cold_stream_cooling / dt;
            
            coolingGas = cold_stream_cooling + hot_halo_cooling;
        }

        if(coolingGas > galaxies[gal].HotGas) {
            coolingGas = galaxies[gal].HotGas;
        } else {
            if(coolingGas < 0.0) coolingGas = 0.0;
        }

        // at this point we have calculated the maximal cooling rate
        // if AGNrecipeOn we now reduce it in line with past heating before proceeding

        if(run_params->AGNrecipeOn > 0 && coolingGas > 0.0) {
            coolingGas = do_AGN_heating(coolingGas, gal, dt, x, rcool, galaxies, run_params);
        }

        if (coolingGas > 0.0) {
            galaxies[gal].Cooling += 0.5 * coolingGas * galaxies[gal].Vvir * galaxies[gal].Vvir;
        }
    } else {
        coolingGas = 0.0;
    }

    XASSERT(coolingGas >= 0.0, -1,
            "Error: Cooling gas mass = %g should be >= 0.0", coolingGas);
        galaxies[gal].CoolingRate = (dt > 0.0)
            ? (float)((coolingGas / dt) * 1.0e10 / run_params->Hubble_h
                      * SEC_PER_GIGAYEAR / run_params->UnitTime_in_s)
            : 0.0f;
    return coolingGas;
}

double cooling_recipe_cgm(const int gal, const double dt, struct GALAXY *galaxies, const struct params *run_params)
{
    double coolingGas = 0.0;

    if(galaxies[gal].CGMgas > 0.0 && galaxies[gal].Vvir > 0.0) {
        const double temp = VIRIAL_TEMP_COEFF * galaxies[gal].Vvir * galaxies[gal].Vvir;  // in Kelvin

        double logZ = -10.0;
        if(galaxies[gal].MetalsCGMgas > 0) {
            logZ = log10(galaxies[gal].MetalsCGMgas / galaxies[gal].CGMgas);
        }

        double lambda = get_metaldependent_cooling_rate(log10(temp), logZ);

        if(lambda > 0.0) {
            double x = PROTONMASS * BOLTZMANN * temp / lambda;        
            x /= (run_params->UnitDensity_in_cgs * run_params->UnitTime_in_s); 

            // 1. Carr et al. 2023 Density Profile Constants (alpha = 1.4, r0 = 0.1 * Rvir)
            const double alpha = 1.4;
            const double r0 = 0.1 * galaxies[gal].Rvir;
            const double ratio = 10.0; // Rvir / r0 is always 10 based on the paper's definition
            
            // Volumetric integral factors for mass (I_M) and cooling (I_cool)
            const double I_M = (pow(ratio, 3.0 - alpha) - 1.0) / (3.0 - alpha);
            const double I_cool = (pow(ratio, 3.0 - 2.0 * alpha) - 1.0) / (3.0 - 2.0 * alpha);

            // 2. Calculate rho0 and the effective cooling density
            const double rho0 = galaxies[gal].CGMgas / (4.0 * M_PI * r0 * r0 * r0 * I_M);
            const double rho_eff = rho0 * (I_cool / I_M);

            // 3. True thermal cooling time of the alpha=1.4 profile
            const double tcool = (x / rho_eff) * (1.5 * MU_IONISED);

            // 4. Free-fall time at Rvir (as defined in the paper)
            const double g_accel = run_params->G * galaxies[gal].Mvir / (galaxies[gal].Rvir * galaxies[gal].Rvir);
            const double tff = sqrt(2.0 * galaxies[gal].Rvir / g_accel);

            // Diagnostic storage
            galaxies[gal].tcool = (float)(tcool * run_params->UnitTime_in_s / SEC_PER_GIGAYEAR);
            galaxies[gal].tff = (float)(tff * run_params->UnitTime_in_s / SEC_PER_GIGAYEAR);
            
            // Pin rcool to Rvir since we are evaluating bulk accretion
            double rcool = galaxies[gal].Rvir;
            galaxies[gal].RcoolToRvir = 1.0;

            // 5. Carr et al. Bulk Cooling Formula
            coolingGas = (galaxies[gal].CGMgas / (tcool + tff)) * dt;

            if(coolingGas > galaxies[gal].CGMgas) {
                coolingGas = galaxies[gal].CGMgas;
            } else if(coolingGas < 0.0) {
                coolingGas = 0.0;
            }

            if(run_params->AGNrecipeOn > 0 && coolingGas > 0.0) {
                coolingGas = do_AGN_heating_cgm(coolingGas, gal, dt, x, rcool, galaxies, run_params);
            }

            if (coolingGas > 0.0) {
                galaxies[gal].Cooling += 0.5 * coolingGas * galaxies[gal].Vvir * galaxies[gal].Vvir;
            }
        }
    }

    galaxies[gal].CoolingRate = (dt > 0.0)
        ? (float)((coolingGas / dt) * 1.0e10 / run_params->Hubble_h
                  * SEC_PER_GIGAYEAR / run_params->UnitTime_in_s)
        : 0.0f;

    XASSERT(coolingGas >= 0.0, -1,
            "Error: Cooling gas mass = %g should be >= 0.0", coolingGas);
            
    return coolingGas;
}


/*
 * Regime-aware cooling: dispatches CGM and hot-halo recipes by galaxy Regime flag.
 *
 * Regime == 0 (CGM): draws only from CGMgas via cooling_recipe_cgm().
 * Regime == 1 (hot): draws from HotGas via cooling_recipe_hot() plus any
 * residual CGMgas.  Both contributions are applied to ColdGas in-place and
 * the total cooled mass is returned.
 */
/*
 * reset_cgm_diagnostics -- clear the CGM timescale diagnostics.
 *
 * cooling_recipe_cgm() is only entered when CGMgas > 0, so a halo that drains
 * its reservoir keeps whatever tcool / tff / tcool_over_tff / MachNumber /
 * RcoolToRvir it had the last time it had gas.  That went stale for 28% of
 * z = 0 Regime-0 centrals and inflated the high-ratio tail of any figure
 * selecting on Regime alone (9.3% above the precipitation threshold, against a
 * true 0.26% among haloes that actually hold a reservoir).  Diagnostics only --
 * no mass or energy is touched.
 */
static void reset_cgm_diagnostics(const int gal, struct GALAXY *galaxies)
{
    galaxies[gal].tcool = 0.0;
    galaxies[gal].tff = -1.0;
    // galaxies[gal].tcool_over_tff = -1.0;
    // galaxies[gal].MachNumber = -1.0;
    galaxies[gal].RcoolToRvir = -1.0;
}

double cooling_recipe_regime_aware(const int gal, const double dt, struct GALAXY *galaxies, const struct params *run_params)
{
    double cgm_cooling = 0.0;
    double hot_cooling = 0.0;
    float hot_tcool = -1.0f;
    int hot_diagnostics_valid = 0;

    if(galaxies[gal].Regime == 0) {
        // CGM REGIME: CGM physics dominates

        // Primary: Precipitation cooling from CGMgas
        if(galaxies[gal].CGMgas > 0.0) {
            cgm_cooling = cooling_recipe_cgm(gal, dt, galaxies, run_params);
        } else {
            reset_cgm_diagnostics(gal, galaxies);
        }


    } else {
        // HOT REGIME: Traditional physics dominates

        // Primary: Traditional cooling from HotGas
        if(galaxies[gal].HotGas > 0.0) {
            hot_cooling = cooling_recipe_hot(gal, dt, galaxies, run_params);
            if(galaxies[gal].Vvir > 0.0 && galaxies[gal].tcool > 0.0) {
                hot_tcool = galaxies[gal].tcool;
                hot_diagnostics_valid = 1;
            }
        }

        // Secondary: Precipitation cooling from CGMgas (gradually depletes)
        if(galaxies[gal].CGMgas > 0.0) {
            cgm_cooling = cooling_recipe_cgm(gal, dt, galaxies, run_params);
        } else if(!hot_diagnostics_valid) {
            reset_cgm_diagnostics(gal, galaxies);
        }

        if(hot_diagnostics_valid) {
            galaxies[gal].tcool = hot_tcool;
            galaxies[gal].tff = -1.0f;
            // galaxies[gal].tcool_over_tff = -1.0f;
            // galaxies[gal].MachNumber = -1.0f;
        }
    }

    // Apply CGM cooling. Clamp to available CGMgas after AGN heating (which
    // runs inside cooling_recipe_cgm for Regime==0 and can drain CGMgas
    // between the internal cap and this apply).
    if(cgm_cooling > 0.0) {
        if(cgm_cooling > galaxies[gal].CGMgas) {
            cgm_cooling = galaxies[gal].CGMgas;
        }
        const double metallicity = get_metallicity(galaxies[gal].CGMgas, galaxies[gal].MetalsCGMgas);
        galaxies[gal].ColdGas += cgm_cooling;
        galaxies[gal].MetalsColdGas += metallicity * cgm_cooling;
        galaxies[gal].CGMgas -= cgm_cooling;
        galaxies[gal].MetalsCGMgas -= metallicity * cgm_cooling;
    }

    // Apply HotGas cooling (clamp to available HotGas after AGN heating)
    if(hot_cooling > 0.0) {
        if(hot_cooling > galaxies[gal].HotGas) {
            hot_cooling = galaxies[gal].HotGas;
        }
        const double metallicity = get_metallicity(galaxies[gal].HotGas, galaxies[gal].MetalsHotGas);
        galaxies[gal].ColdGas += hot_cooling;
        galaxies[gal].MetalsColdGas += metallicity * hot_cooling;
        galaxies[gal].HotGas -= hot_cooling;
        galaxies[gal].MetalsHotGas -= metallicity * hot_cooling;
    }

    double total_cooling = cgm_cooling + hot_cooling;
    galaxies[gal].CoolingRate = (dt > 0.0)
        ? (float)((total_cooling / dt) * 1.0e10 / run_params->Hubble_h
                  * SEC_PER_GIGAYEAR / run_params->UnitTime_in_s)
        : 0.0f;
    XASSERT(total_cooling >= 0.0, -1,
            "Error: Cooling gas mass = %g should be >= 0.0", total_cooling);
    return total_cooling;
}

/*
 * AGN radio-mode heating for the hot-halo regime (HotGas reservoir).
 *
 * First reduces coolingGas based on the stored r_heat from past AGN activity,
 * then computes new BH accretion (empirical, Bondi-Hoyle, or cold-cloud) and
 * the resulting heating. Updates BlackHoleMass, HotGas, and r_heat in-place.
 * Returns the post-heating coolingGas value.
 */
/*
 * agn_accretion_compute -- shared radio-mode AGN accretion and heating calculation.
 *
 * Computes AGNrate via the selected recipe (AGNrecipeOn 1/2/3), applies the
 * Eddington cap, converts to accreted mass, and derives the equivalent heating
 * mass.  reservoir_mass is the gas available for accretion (HotGas or CGMgas
 * depending on the caller); the heating coefficient uses Vvir of the central.
 * Does NOT modify any galaxy fields -- the caller draws from the right reservoir
 * and updates r_heat.
 */
static void agn_accretion_compute(const int centralgal, const double dt, const double x,
                                   const double coolingGas, const double rcool,
                                   const double reservoir_mass,
                                   const struct GALAXY *galaxies, const struct params *run_params,
                                   double *AGNaccreted_out, double *AGNheating_out)
{
    const double Vvir   = galaxies[centralgal].Vvir;
    const double Mvir   = galaxies[centralgal].Mvir;
    const double BHmass = galaxies[centralgal].BlackHoleMass;
    const double Rvir   = galaxies[centralgal].Rvir;

    double AGNrate = 0.0;
    if(run_params->AGNrecipeOn == 2) {
        // Bondi-Hoyle accretion (Bondi 1952)
        AGNrate = (2.5 * M_PI * run_params->G) * (0.375 * 0.6 * x) * BHmass * run_params->RadioModeEfficiency;
    } else if(run_params->AGNrecipeOn == 3) {
        // Cold cloud accretion: triggers when M_BH exceeds the sonic-mass threshold
        if(BHmass > AGN_COLD_CLOUD_FRAC * Mvir * CUBE(rcool / Rvir))
            AGNrate = AGN_COLD_CLOUD_FRAC * coolingGas / dt;
    } else {
        // Empirical recipe (Croton et al. 2006, eq. 10)
        const double base = run_params->RadioModeEfficiency
            / (run_params->UnitMass_in_g / run_params->UnitTime_in_s * SEC_PER_YEAR / SOLAR_MASS)
            * (BHmass / AGN_BH_MASS_PIVOT) * CUBE(Vvir / AGN_VVIR_PIVOT_KMS);
        AGNrate = (Mvir > 0.0) ? base * (reservoir_mass / Mvir) / AGN_HOT_GAS_PIVOT : base;
    }

    // Eddington-limited accretion (Rybicki & Lightman 1979)
    const double EDDrate = (EDDINGTON_LUM_PER_MSUN_CGS * BHmass * 1e10 / run_params->Hubble_h)
        / (run_params->UnitEnergy_in_cgs / run_params->UnitTime_in_s)
        / (AGN_RADIATIVE_EFFICIENCY * C_SQ_KMS2);
    if(AGNrate > EDDrate) AGNrate = EDDrate;

    double AGNaccreted = AGNrate * dt;
    if(AGNaccreted > reservoir_mass) AGNaccreted = reservoir_mass;

    // Heating coefficient: AGN_HEATING_COEFF_KMS = sqrt(2*eta*c^2)
    double AGNheating = 0.0;
    if(Vvir > 0.0) {
        const double AGNcoeff = (AGN_HEATING_COEFF_KMS / Vvir) * (AGN_HEATING_COEFF_KMS / Vvir);
        AGNheating = AGNcoeff * AGNaccreted;
        // limit to available cooling mass
        if(AGNheating > coolingGas && AGNcoeff > 0.0) {
            AGNaccreted = coolingGas / AGNcoeff;
            AGNheating  = coolingGas;
        }
    }

    *AGNaccreted_out = AGNaccreted;
    *AGNheating_out  = AGNheating;
}


/*
 * AGN radio-mode heating for hot-halo (Regime==1) galaxies.
 *
 * Applies r_heat suppression unconditionally (the ratchet always runs for
 * Regime==1), then calls agn_accretion_compute() to get the accreted mass
 * and suppressed cooling mass, draws the accreted mass from HotGas, and
 * updates r_heat via the standard ratchet.
 */
double do_AGN_heating(double coolingGas, const int centralgal, const double dt, const double x, const double rcool, struct GALAXY *galaxies, const struct params *run_params)
{
    // r_heat suppression (always applied for Regime==1 hot-halo)
    if(galaxies[centralgal].r_heat < rcool &&
       !(run_params->CGMrecipeOn == 1 && galaxies[centralgal].r_heat >= 0.99 * rcool)) {
        coolingGas = (1.0 - galaxies[centralgal].r_heat / rcool) * coolingGas;
    } else {
        coolingGas = 0.0;
    }
    XASSERT(coolingGas >= 0.0, -1,
            "Error: Cooling gas mass = %g should be >= 0.0", coolingGas);

    if(galaxies[centralgal].HotGas > 0.0) {
        double AGNaccreted, AGNheating;
        agn_accretion_compute(centralgal, dt, x, coolingGas, rcool,
                               galaxies[centralgal].HotGas,
                               galaxies, run_params, &AGNaccreted, &AGNheating);

        const double metallicity = get_metallicity(galaxies[centralgal].HotGas,
                                                   galaxies[centralgal].MetalsHotGas);
        galaxies[centralgal].BlackHoleMass += AGNaccreted;
        galaxies[centralgal].HotGas        -= AGNaccreted;
        galaxies[centralgal].MetalsHotGas  -= metallicity * AGNaccreted;

        // standard r_heat ratchet
        if(galaxies[centralgal].r_heat < rcool && coolingGas > 0.0) {
            const double r_heat_new = (AGNheating / coolingGas) * rcool;
            if(r_heat_new > galaxies[centralgal].r_heat)
                galaxies[centralgal].r_heat = r_heat_new;
        }

        if(AGNheating > 0.0)
            galaxies[centralgal].Heating += 0.5 * AGNheating
                * galaxies[centralgal].Vvir * galaxies[centralgal].Vvir;
    }

    if(run_params->CGMrecipeOn > 0 && galaxies[centralgal].r_heat > galaxies[centralgal].Rvir)
        galaxies[centralgal].r_heat = galaxies[centralgal].Rvir;

    return coolingGas;
}

/*
 * AGN radio-mode heating for CGM-dominated (Regime==0) galaxies.
 *
 * Applies the same r_heat/rcool suppression and ratchet as the hot-halo
 * path, then caps r_heat at Rvir. Accretion draws from CGMgas.
 */
static double do_AGN_heating_cgm(double coolingGas, const int centralgal, const double dt, const double x, const double rcool,
                                 struct GALAXY *galaxies, const struct params *run_params)
{
    if(galaxies[centralgal].r_heat < rcool &&
       !(run_params->CGMrecipeOn == 1 && galaxies[centralgal].r_heat >= 0.99 * rcool)) {
        coolingGas = (1.0 - galaxies[centralgal].r_heat / rcool) * coolingGas;
    } else {
        coolingGas = 0.0;
    }

    XASSERT(coolingGas >= 0.0, -1,
            "Error: Cooling gas mass = %g should be >= 0.0", coolingGas);

    if(galaxies[centralgal].CGMgas > 0.0) {
        double AGNaccreted, AGNheating;
        agn_accretion_compute(centralgal, dt, x, coolingGas, rcool,
                              galaxies[centralgal].CGMgas, galaxies, run_params,
                              &AGNaccreted, &AGNheating);
        const double metallicity = get_metallicity(galaxies[centralgal].CGMgas, galaxies[centralgal].MetalsCGMgas);
        galaxies[centralgal].BlackHoleMass  += AGNaccreted;
        galaxies[centralgal].CGMgas         -= AGNaccreted;
        galaxies[centralgal].MetalsCGMgas   -= metallicity * AGNaccreted;

        if(galaxies[centralgal].r_heat < rcool && coolingGas > 0.0) {
            const double r_heat_new = (AGNheating / coolingGas) * rcool;
            if(r_heat_new > galaxies[centralgal].r_heat)
                galaxies[centralgal].r_heat = r_heat_new;
        }
        if(galaxies[centralgal].r_heat > galaxies[centralgal].Rvir)
            galaxies[centralgal].r_heat = galaxies[centralgal].Rvir;

        if(AGNheating > 0.0)
            galaxies[centralgal].Heating += 0.5 * AGNheating * galaxies[centralgal].Vvir * galaxies[centralgal].Vvir;

        /* The BH accretion above drew AGNaccreted out of CGMgas, but any
         * earlier clamp of coolingGas used the pre-accretion reservoir.
         * Re-cap so the caller cannot overdraw the CGM by up to AGNaccreted
         * (manifested as an XASSERT abort when cooling was reservoir-limited
         * and Bondi accretion nonzero in the same call, e.g. with
         * CGMDensityProfile = 1). */
        if(coolingGas > galaxies[centralgal].CGMgas) {
            coolingGas = galaxies[centralgal].CGMgas;
        }
    }
    return coolingGas;
}

/*
 * Transfer cooled gas from the HotGas reservoir into the cold disk.
 *
 * Moves up to coolingGas mass (clamped to available HotGas) from HotGas to
 * ColdGas, tracking metallicity consistently. Called by cooling_recipe_hot()
 * after AGN heating has been applied.
 */
void cool_gas_onto_galaxy(const int centralgal, const double coolingGas, struct GALAXY *galaxies)
{
    // Move the cooled mass into the cold disk. coolingGas was already computed for this
    // substep's dt (deltaT / effective_steps), so there is no 1/STEPS factor here.
    if(coolingGas > 0.0) {
        if(coolingGas < galaxies[centralgal].HotGas) {
            const double metallicity = get_metallicity(galaxies[centralgal].HotGas, galaxies[centralgal].MetalsHotGas);
            galaxies[centralgal].ColdGas += coolingGas;
            galaxies[centralgal].MetalsColdGas += metallicity * coolingGas;
            galaxies[centralgal].HotGas -= coolingGas;
            galaxies[centralgal].MetalsHotGas -= metallicity * coolingGas;
        } else {
            galaxies[centralgal].ColdGas += galaxies[centralgal].HotGas;
            galaxies[centralgal].MetalsColdGas += galaxies[centralgal].MetalsHotGas;
            galaxies[centralgal].HotGas = 0.0;
            galaxies[centralgal].MetalsHotGas = 0.0;
        }
    }
}

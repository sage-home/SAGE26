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

/* Duffy et al. (2008) NFW concentration-mass-redshift relation,
 * Table 1 "Full sample" (relaxed halos, NFW profile, 200c overdensity).
 * c = A * (M/M_pivot)^B * (1+z)^C */
static const double DUFFY08_A       =  7.85;
static const double DUFFY08_M_PIVOT =  2.0e12;  /* pivot mass in Msun */
static const double DUFFY08_B       = -0.081;
static const double DUFFY08_C       = -0.71;

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

/* Gravitational constant in CGS (NIST CODATA 2018). */
static const double G_CGS = 6.674e-8;  /* cm^3 g^-1 s^-2 */

/* Virial temperature coefficient: T_vir = VIRIAL_TEMP_COEFF * Vvir^2 [Kelvin].
 * Follows from T = mu * m_p * Vvir^2 / (2 * k_B) with mu = MU_IONISED = 0.59;
 * gives 35.9 K (km/s)^-2. */
static const double VIRIAL_TEMP_COEFF = 35.9;  /* K (km/s)^-2 */

/* McCourt et al. (2012) thermal instability threshold: precipitation occurs
 * when t_cool / t_ff < PRECIP_THRESHOLD (= 10). */
// static const double PRECIP_THRESHOLD = 10.0;

/* Width of the tanh/sigmoid transition zone around PRECIP_THRESHOLD.
 * Smooths the discontinuity at exactly t_cool/t_ff = 10. */
// static const double PRECIP_TRANSITION_WIDTH = 2.0;

/* Stern et al. (2021) analytic cooling-flow structure (CGMDensityProfile = 3).
 *
 * The volume-filling gas follows a power law rho ~ r^-a normalised to CGMgas
 * inside Rvir,
 *     rho(r) = (3 - a) M_CGM / (4 pi Rvir^3) * (r/Rvir)^-a,
 * which is their Equation 13 with f_gas taken from the model's own reservoir
 * rather than the cosmic baryon budget, so the profile integrates to CGMgas by
 * construction.  The slope is the cooling-flow value n_H ~ r^-1.6 derived in
 * Stern et al. (2019).
 *
 * Both timescales are evaluated at the gas circularisation radius
 *     R_circ = sqrt(2) lambda Rvir ~ 0.05 Rvir     (their Section 2.1, lambda ~ 0.035)
 * instead of at Rvir.  For a > 1 the ratio t_cool/t_ff rises outward, so R_circ
 * is where it is MINIMISED -- which is the radius the precipitation criterion is
 * defined on (Voit et al. 2017 evaluate min(t_cool/t_ff)), and the radius Stern
 * et al. (2021) use for their virialisation condition. */
static const double STERN_PROFILE_SLOPE = 1.6;
static const double STERN_RCIRC_FRAC    = 0.05;
static const double STERN_TEMP_BOOST    = 1.2;  /* their Eq 11: T^(s) = (6/5A) T_vir, A ~ 1 */

/* Stern et al. (2019) Eq 28: in a cooling flow the ratio of the cooling time to
 * the free-fall time is the inverse Mach number up to a factor of order unity,
 *     t_cool / t_ff = sqrt(A) / (sqrt(2) B) * Mach^-1 = 0.845 / Mach,
 * using their A = 1.08, B = 0.87 for d ln v_c / d ln r = -0.1.  Mach > 1 (i.e.
 * t_cool/t_ff < 0.845) marks the supersonic regime, in which no steady-state
 * cooling-flow solution exists and "all the halo gas collapses on a dynamical
 * timescale" (their Section 2.5) -- the free-falling CGM of Stern et al. (2021). */
static const double STERN_MACH_COEFF = 0.845;

/* Beta-profile core radius as a fraction of the virial radius: r_c = frac * Rvir.
 * A value of 0.1 is the standard choice for the hot CGM (e.g., Makino+98). */
static const double CGM_BETA_CORE_RADIUS_FRAC = 0.1;

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

// ============================================================================
// CGM Density Profile Helper Functions (file-private)
// ============================================================================

/*
 * NFW profile normalisation rho_s for total mass M_CGM within R_vir.
 *
 * Solves M_CGM = 4*pi * rho_s * r_s^3 * [ln(1+c) - c/(1+c)] for rho_s,
 * where r_s = Rvir / c_NFW is the scale radius.
 */
static double nfw_rho_s(const double M_CGM, const double Rvir, const double c_NFW)
{
    const double r_s = Rvir / c_NFW;
    // M_CGM = 4pi rho_s r_s^3 * [ln(1+c) - c/(1+c)]
    const double f_c = log(1.0 + c_NFW) - c_NFW / (1.0 + c_NFW);
    return M_CGM / (4.0 * M_PI * r_s * r_s * r_s * f_c);
}

/* NFW density rho(r) = rho_s / [x*(1+x)^2] where x = r/r_s. */
static double nfw_density(const double r, const double rho_s, const double r_s)
{
    const double x = r / r_s;
    if(x < 1e-10) return rho_s / (1e-10 * 1.0 * 1.0);  // Avoid singularity at r=0
    return rho_s / (x * (1.0 + x) * (1.0 + x));
}

/* NFW concentration c(M, z) from Duffy et al. (2008): c = 7.85*(M/2e12)^-0.081*(1+z)^-0.71. */
static double nfw_concentration(const double Mvir_Msun, const double z)
{
    /* The mass exponent is negative, so pow(0, B) returns +inf and poisons rho_s
     * (and every downstream density) with a NaN.  Mvir == 0 does occur: Type-2
     * satellites that have lost their subhalo keep a CGM reservoir but carry
     * Mvir == 0.  Return 0 so callers fall back to the uniform profile. */
    if(!(Mvir_Msun > 0.0)) {
        return 0.0;
    }
    // c = A * (M/M_pivot)^B * (1+z)^C  (Duffy et al. 2008)
    return DUFFY08_A * pow(Mvir_Msun / DUFFY08_M_PIVOT, DUFFY08_B) * pow(1.0 + z, DUFFY08_C);
}

/*
 * Beta-profile normalisation rho_0 for total mass M_CGM within R_vir.
 *
 * Profile: rho(r) = rho_0 / [1 + (r/r_c)^2]^(3*beta/2).
 * Uses an analytic form for beta ~ 2/3 and Simpson quadrature otherwise.
 */
static double beta_rho_0(const double M_CGM, const double Rvir, const double r_c, const double beta)
{
    // For general beta, the enclosed mass integral is:
    // M(<R) = 4pi rho_0 integral_0^R r^2 / [1 + (r/r_c)^2]^(3beta/2) dr
    //
    // For beta = 2/3 (common value), this simplifies to:
    // M(<R) = 4pi rho_0 r_c^3 * [arctan(R/r_c) - (R/r_c)/(1 + (R/r_c)^2)]
    // But we need the more general form...
    //
    // Use numerical approximation for general beta:
    // For large R/r_c, M ~ 4pi rho_0 r_c^3 * (some function of beta)

    const double x = Rvir / r_c;
    double mass_integral;

    if(fabs(beta - 2.0/3.0) < 0.01) {
        // beta ~ 2/3: use analytic form
        // M = 4pi rho_0 r_c^3 * [arctan(x) - x/(1+x^2)]
        mass_integral = atan(x) - x / (1.0 + x * x);
    } else {
        // General beta: numerical integration using Simpson's rule
        const int n_steps = 100;
        const double dr = Rvir / n_steps;
        double integral = 0.0;
        for(int i = 0; i <= n_steps; i++) {
            const double r = i * dr;
            const double y = r / r_c;
            const double rho_factor = 1.0 / pow(1.0 + y * y, 1.5 * beta);
            double weight = (i == 0 || i == n_steps) ? 1.0 : ((i % 2 == 0) ? 2.0 : 4.0);
            integral += weight * r * r * rho_factor;
        }
        integral *= dr / 3.0;
        // M = 4pi rho_0 * integral, so mass_integral = integral / r_c^3
        mass_integral = integral / (r_c * r_c * r_c);
    }

    if(mass_integral <= 0.0) return 0.0;
    return M_CGM / (4.0 * M_PI * r_c * r_c * r_c * mass_integral);
}

/* Beta-profile density rho(r) = rho_0 / [1 + (r/r_c)^2]^(3*beta/2). */
static double beta_density(const double r, const double rho_0, const double r_c, const double beta)
{
    const double y = r / r_c;
    return rho_0 / pow(1.0 + y * y, 1.5 * beta);
}

// ============================================================================
// Enclosed Mass Functions for each profile
// ============================================================================

/* NFW enclosed mass M(<r) = M_total * f(x)/f(c), where f(u) = ln(1+u) - u/(1+u). */
static double nfw_enclosed_mass(const double r, const double M_total, const double Rvir, const double c_NFW)
{
    const double r_s = Rvir / c_NFW;
    const double x = r / r_s;

    // M(<r) / M_total = f(x) / f(c)
    const double f_x = log(1.0 + x) - x / (1.0 + x);
    const double f_c = log(1.0 + c_NFW) - c_NFW / (1.0 + c_NFW);

    if(f_c <= 0.0) return M_total;
    return M_total * f_x / f_c;
}

/* Beta-profile (beta=2/3) enclosed mass using the analytic arctan form. */
static double beta_enclosed_mass(const double r, const double M_total, const double Rvir, const double r_c)
{
    const double x = r / r_c;
    const double X = Rvir / r_c;

    const double f_x = atan(x) - x / (1.0 + x * x);
    const double f_X = atan(X) - X / (1.0 + X * X);

    if(f_X <= 0.0) return M_total;
    return M_total * f_x / f_X;
}

/* Dispatch enclosed-mass calculation to the appropriate profile model (0=uniform, 1=NFW, 2=beta). */
static double cgm_enclosed_mass(const double r, const double M_total, const double Rvir,
                                 const double Mvir_Msun, const double z, const int profile_type)
{
    if(r >= Rvir) return M_total;
    if(r <= 0.0) return 0.0;

    if(profile_type == 0) {
        // Uniform density: M(<r) = M_total * (r/Rvir)^3
        const double ratio = r / Rvir;
        return M_total * ratio * ratio * ratio;
    } else if(profile_type == 1) {
        // NFW profile (falls back to uniform when the concentration is unusable)
        const double c_NFW = nfw_concentration(Mvir_Msun, z);
        if(!(c_NFW > 0.0)) {
            const double ratio = r / Rvir;
            return M_total * ratio * ratio * ratio;
        }
        return nfw_enclosed_mass(r, M_total, Rvir, c_NFW);
    } else if(profile_type == 2) {
        // Beta profile (beta = 2/3, r_c = CGM_BETA_CORE_RADIUS_FRAC * Rvir)
        const double r_c = CGM_BETA_CORE_RADIUS_FRAC * Rvir;
        return beta_enclosed_mass(r, M_total, Rvir, r_c);
    } else if(profile_type == 3) {
        /* Stern profile: the GAS follows r^-a, but this function returns the
         * GRAVITATING mass, which is the halo's.  A uniform r^3 law
         * underestimates M(<0.05 Rvir) by a factor ~400 for c = 10 and would
         * make t_ff at R_circ ~20x too long, so use NFW here as profile 1 does. */
        const double c_NFW = nfw_concentration(Mvir_Msun, z);
        if(!(c_NFW > 0.0)) {
            const double ratio = r / Rvir;
            return M_total * ratio * ratio * ratio;
        }
        return nfw_enclosed_mass(r, M_total, Rvir, c_NFW);

    } else {
        // Default to uniform
        const double ratio = r / Rvir;
        return M_total * ratio * ratio * ratio;
    }
}

/*
 * CGM gas density at radius r in CGS units (g/cm^3).
 *
 * Dispatches to the selected profile model (profile_type: 0=uniform, 1=NFW, 2=beta).
 * Falls back to uniform for unrecognised profile_type values.
 * External so the ram-pressure stripping module can evaluate the same ambient
 * profile the cooling recipe assumes (see model_ram_pressure.c).
 */
double cgm_density_at_radius(const double r_cgs, const double CGMgas_cgs, const double Rvir_cgs,
                             const double Mvir_Msun, const double z, const int profile_type)
{
    if(profile_type == 0) {
        // Uniform density
        const double volume_cgs = (4.0 * M_PI / 3.0) * Rvir_cgs * Rvir_cgs * Rvir_cgs;
        return CGMgas_cgs / volume_cgs;

    } else if(profile_type == 1) {
        // NFW profile (falls back to uniform when the concentration is unusable)
        const double c_NFW = nfw_concentration(Mvir_Msun, z);
        if(!(c_NFW > 0.0)) {
            const double volume_cgs = (4.0 * M_PI / 3.0) * Rvir_cgs * Rvir_cgs * Rvir_cgs;
            return CGMgas_cgs / volume_cgs;
        }
        const double r_s_cgs = Rvir_cgs / c_NFW;
        const double rho_s = nfw_rho_s(CGMgas_cgs, Rvir_cgs, c_NFW);
        return nfw_density(r_cgs, rho_s, r_s_cgs);

    } else if(profile_type == 2) {
        // Beta profile with beta = 2/3 and r_c = CGM_BETA_CORE_RADIUS_FRAC * Rvir
        const double beta = 2.0 / 3.0;
        const double r_c_cgs = CGM_BETA_CORE_RADIUS_FRAC * Rvir_cgs;
        const double rho_0 = beta_rho_0(CGMgas_cgs, Rvir_cgs, r_c_cgs, beta);
        return beta_density(r_cgs, rho_0, r_c_cgs, beta);

    } else if(profile_type == 3) {
        // Stern et al. (2021) power law, normalised to CGMgas inside Rvir
        const double a = STERN_PROFILE_SLOPE;
        const double rho_vir = (3.0 - a) * CGMgas_cgs
                             / (4.0 * M_PI * Rvir_cgs * Rvir_cgs * Rvir_cgs);
        const double x = (r_cgs > 0.0) ? (r_cgs / Rvir_cgs) : STERN_RCIRC_FRAC;
        return rho_vir * pow(x, -a);

    } else {
        // Default to uniform if unknown profile type
        const double volume_cgs = (4.0 * M_PI / 3.0) * Rvir_cgs * Rvir_cgs * Rvir_cgs;
        return CGMgas_cgs / volume_cgs;
    }
}

/*
 * Solve iteratively for the cooling radius r_cool where t_cool(r) = t_ff(r).
 *
 * Returns r_cool in CGS units. For uniform and beta profiles the isothermal
 * analytic approximation is used (the profile is too flat for iteration to
 * converge). For NFW, a Newton-like iteration over t_cool/t_ff converges in
 * typically < 10 steps.  Result is bounded to [0.001, 1.0] * Rvir.
 */
static double solve_for_rcool(const double CGMgas_cgs, const double Rvir_cgs, const double Mvir_cgs,
                              const double Mvir_Msun, const double temp, const double lambda,
                              const double z, const int profile_type,
                              __attribute__((unused)) const struct params *run_params)
{
    const double mu = MU_IONISED;

    // ========================================================================
    // UNIFORM: r_cool = R_vir
    // ========================================================================
    // With a uniform gas profile both t_cool and t_ff are radius-independent:
    // rho is constant, and M(<r) = Mvir (r/Rvir)^3 gives g ~ r, so
    // t_ff = sqrt(2r/g) = sqrt(2 Rvir^3 / G Mvir) at every radius.  t_cool(r) =
    // t_ff(r) therefore has no interior solution -- the reservoir either cools
    // everywhere inside R_vir or nowhere -- and r_cool = R_vir states that
    // honestly.  This is what Equations 1 and 2 of the paper assume.
    //
    // The isothermal (rho ~ r^-2) formula retained below for the beta profile
    // was previously applied here as well.  It is numerically a no-op for the
    // uniform case, since neither t_cool nor t_ff depends on the radius it
    // returns, but it is inconsistent: r_cool is derived from a rho ~ r^-2
    // profile and then used with a uniform density.  Only RcoolToRvir, a
    // diagnostic, changes.
    if(profile_type == 0) {
        return Rvir_cgs;
    }

    // ========================================================================
    // STERN: r_cool is not solved for -- both timescales are evaluated at the
    // circularisation radius, which is where t_cool/t_ff is minimised.
    // ========================================================================
    if(profile_type == 3) {
        return STERN_RCIRC_FRAC * Rvir_cgs;
    }

    // ========================================================================
    // BETA: Use isothermal r_cool formula (like hot-regime)
    // ========================================================================
    // The beta profile (beta = 2/3) is too flat for the iterative solver to
    // converge, so use the isothermal approach: assume rho(r) ~ 1/r^2, giving
    // r_cool = sqrt(rho0 / rho_cool) where rho_cool is the critical density.
    if(profile_type == 2) {
        // t_ff at R_vir: t_ff = sqrt(2 R^3 / (G M))
        const double t_ff_Rvir = sqrt(2.0 * Rvir_cgs * Rvir_cgs * Rvir_cgs / (G_CGS * Mvir_cgs));

        // Critical density where t_cool = t_ff
        // rho_cool = (3/2) mu m_p k T / (Lambda t_ff)
        const double rho_cool = (1.5 * mu * PROTONMASS * BOLTZMANN * temp) / (lambda * t_ff_Rvir);

        // Isothermal profile normalization: rho0 = M / (4pi R)
        const double rho0 = CGMgas_cgs / (4.0 * M_PI * Rvir_cgs);

        // r_cool from isothermal: rho(r_cool) = rho0/r_cool^2 = rho_cool
        double r_cool = sqrt(rho0 / rho_cool);

        // Apply bounds
        if(r_cool > Rvir_cgs) r_cool = Rvir_cgs;
        if(r_cool < 0.001 * Rvir_cgs) r_cool = 0.001 * Rvir_cgs;

        return r_cool;
    }

    // ========================================================================
    // NFW PROFILE: Iterative solver (cuspy profile converges well)
    // ========================================================================
    const double prefactor = 1.5 * mu * PROTONMASS * BOLTZMANN * temp / lambda;

    double r_cool = 0.5 * Rvir_cgs;
    const int max_iter = 30;
    const double tolerance = 0.01;

    for(int iter = 0; iter < max_iter; iter++) {
        const double rho = cgm_density_at_radius(r_cool, CGMgas_cgs, Rvir_cgs, Mvir_Msun, z, profile_type);

        if(rho <= 0.0) {
            r_cool = Rvir_cgs;
            break;
        }

        const double t_cool = prefactor / rho;

        const double M_enclosed = cgm_enclosed_mass(r_cool, Mvir_cgs, Rvir_cgs, Mvir_Msun, z, profile_type);
        const double g_accel = (M_enclosed > 0.0) ? G_CGS * M_enclosed / (r_cool * r_cool) : 0.0;

        if(g_accel <= 0.0) {
            r_cool = Rvir_cgs;
            break;
        }

        const double t_ff = sqrt(2.0 * r_cool / g_accel);
        const double ratio = t_cool / t_ff;

        const double r_cool_new = r_cool * pow(ratio, -0.3);

        double r_bounded = r_cool_new;
        if(r_bounded > Rvir_cgs) r_bounded = Rvir_cgs;
        if(r_bounded < 0.001 * Rvir_cgs) r_bounded = 0.001 * Rvir_cgs;

        if(fabs(r_bounded - r_cool) / r_cool < tolerance) {
            r_cool = r_bounded;
            break;
        }

        r_cool = r_bounded;
    }

    return r_cool;
}

/*
 * Preventive (non-AGN) suppression of the cooling flow.
 *
 * Every cooling suppression in SAGE26 otherwise flows through the AGN-driven r_heat
 * ratchet, which is cumulative: weak early, strong late. Measured on microUchuu at fixed
 * halo mass 10^12-10^12.5, AGN heating is 1/150 of the cooling rate at z = 2.2 against
 * parity at z = 0, because M_BH tracks a bulge that is 10x smaller at fixed Mvir (B/T
 * 0.05 vs 0.23), while 97% of those haloes are already flagged Regime == 1 at every
 * epoch. The model therefore has no quenching channel at cosmic noon and can only be
 * calibrated at z = 0.
 *
 * This term supplies a suppression whose energy source is gravitational -- virial-shock
 * heating of the corona, dynamical friction from infalling substructure -- rather than
 * the black hole, so it does not wait for a bulge to grow and is epoch-independent by
 * construction. Halo-mass gated after Bower et al. (2006), who make the quasi-hydrostatic
 * halo the condition for radio-mode suppression rather than the black-hole mass:
 *
 *     f = 1 / (1 + (Mvir / M_prev)^slope)
 *
 * Applied multiplicatively to the cooling rate *after* the AGN block, so the r_heat
 * bookkeeping is untouched and the two suppressions stay separable.
 *
 * Returns a factor in [0, 1]; exactly 1.0 when disabled.
 */
// static double preventive_suppression(const int gal, const struct GALAXY *galaxies, const struct params *run_params)
// {
//     if(run_params->PreventiveHeatingOn == 0 || run_params->PreventiveHeatingMass <= 0.0) {
//         return 1.0;
//     }
//     const double Mvir_msun = CODE_MASS_TO_MSUN(galaxies[gal].Mvir, run_params->Hubble_h);
//     if(Mvir_msun <= 0.0) {
//         return 1.0;
//     }
//     const double ratio = Mvir_msun / run_params->PreventiveHeatingMass;
//     return 1.0 / (1.0 + pow(ratio, run_params->PreventiveHeatingSlope));
// }

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

/*
 * Classical hot-halo cooling recipe following Croton et al. (2006).
 *
 * Computes the cooling radius from the isothermal beta-model and returns
 * the mass cooled over timestep dt. When CGMrecipeOn == 1 an additional
 * cold-stream component (De Lucia & Blaizot 2006) is blended in for
 * hot-regime halos. AGN heating is applied before the return.
 */
// double cooling_recipe_hot(const int gal, const double dt, struct GALAXY *galaxies, const struct params *run_params)
// {
//     double coolingGas;

//     galaxies[gal].tcool = 0.0f;
//     // galaxies[gal].tcool = -1.0f;
//     // galaxies[gal].tff = -1.0f;
//     // galaxies[gal].tcool_over_tff = -1.0f;
//     // galaxies[gal].MachNumber = -1.0f;

//     if(galaxies[gal].HotGas > 0.0 && galaxies[gal].Vvir > 0.0) {
//         const double tcool_dyn = galaxies[gal].Rvir / galaxies[gal].Vvir;
//         const double temp = VIRIAL_TEMP_COEFF * galaxies[gal].Vvir * galaxies[gal].Vvir;  // in Kelvin

//         double logZ = -10.0;
//         if(galaxies[gal].MetalsHotGas > 0) {
//             logZ = log10(galaxies[gal].MetalsHotGas / galaxies[gal].HotGas);
//         }

//         double lambda = get_metaldependent_cooling_rate(log10(temp), logZ);

//         if(lambda <= 0.0) {
//             return 0.0;  // No cooling if cooling function is zero/negative
//         }

//         double x = PROTONMASS * BOLTZMANN * temp / lambda;        // now this has units sec g/cm^3
//         x /= (run_params->UnitDensity_in_cgs * run_params->UnitTime_in_s);         // now in internal units
//         const double rho_rcool = x / tcool_dyn * (1.5 * MU_IONISED);  // 3/2 * mu for a fully ionized gas

//         if(rho_rcool <= 0.0) {
//             return 0.0;
//         }

//         // an isothermal density profile for the hot gas is assumed here
//         const double rho0 = galaxies[gal].HotGas / (4 * M_PI * galaxies[gal].Rvir);
//         double rcool = sqrt(rho0 / rho_rcool);

//         galaxies[gal].RcoolToRvir = rcool / galaxies[gal].Rvir;  // store uncapped ratio for diagnostics

//         coolingGas = 0.0;

//         if(run_params->CGMrecipeOn == 0) {
//             // SAGE C16 hot-halo cooling, both branches (Croton et al. 2016).
//             // tcool here is the halo dynamical time Rvir/Vvir.
//             if(rcool > galaxies[gal].Rvir) {
//                 // Rapid "cold accretion": the whole corona cools within a
//                 // dynamical time.  Discontinuous with the branch below by a
//                 // factor 2 at rcool = Rvir; that is the published behaviour.
//                 coolingGas = galaxies[gal].HotGas / tcool_dyn * dt;
//             } else {
//                 // Quasi-static cooling flow.
//                 coolingGas = (galaxies[gal].HotGas / galaxies[gal].Rvir) * (rcool / (2.0 * tcool_dyn)) * dt;
//             }
//         } else {
//             // CGMrecipeOn == 1: D&B06 cold streams for hot-regime halos
//             // All halos here are in the hot regime (have virial shocks)
//             const double z = run_params->ZZ[galaxies[gal].SnapNum];
            
//             // D&B06 eqs 39-41: stream penetration factor f_stream.
//             // Mass suppression (M/Mshock)^(-4/3) -- halos well above the shock
//             // threshold host weaker cold streams. Redshift factor (1+z)/(1+1)
//             // enhances streams at high-z where cooling is more efficient.
//             const double Mvir_physical = CODE_MASS_TO_MSUN(galaxies[gal].Mvir, run_params->Hubble_h);
//             const double mass_ratio = Mvir_physical / run_params->MShockMsun;

//             // Redshift enhancement: normalized to z=1 following D&B06 eq 40
//             const double z_factor = (1.0 + z) / (1.0 + 1.0);

//             double f_stream;
//             if(run_params->ColdStreamCeilingOn) {
//                 // Dekel & Birnboim (2006) eqs 39-41.  Their eq. 39 compares the
//                 // cooling and compression times within the stream,
//                 //     R = (f Mstar/Mvir)^(2/3) (Mvir/Mshock)^(4/3),
//                 // streams penetrating where R < 1.  The redshift dependence
//                 // enters through the clustering mass Mstar(z) rather than an
//                 // explicit (1+z) factor, and the shut-off is automatic: their
//                 // eq. 41 defines z_crit by f Mstar(z_crit) = Mshock, which is
//                 // exactly where R = 1 at Mvir = Mshock.  No redshift cut is
//                 // imposed, so f_stream is continuous everywhere.
//                 const double Mstar = pow(10.0, interpolate_clustering_mass(z, run_params));
//                 const double fMstar = run_params->StreamMassFactor * Mstar;
//                 const double ratio = pow(fMstar / Mvir_physical, 2.0/3.0)
//                                    * pow(mass_ratio, 4.0/3.0);
//                 if(ratio > 0.0) {
//                     const double sigmoid_arg = -log10(ratio) / STREAM_TRANSITION_WIDTH_DEX;
//                     f_stream = 1.0 / (1.0 + exp(-sigmoid_arg));
//                 } else {
//                     f_stream = 1.0;
//                 }
//             } else if(z < Z_CRIT_DB06 && mass_ratio > 1.0) {
//                 // D&B06 eq 41: below z_crit cold streams are suppressed in
//                 // M > Mshock halos.  Hard cutoff; published behaviour.
//                 f_stream = 0.0;
//             } else {
//                 // High-z regime: streams can penetrate
//                 f_stream = pow(mass_ratio, -4.0/3.0) * z_factor;
//             }
            
//             // Ensure physical bounds
//             // Cap at 0.5 (50%) to account for partial heating/mixing of cold streams
//             // as they penetrate through the hot medium
//             if(f_stream > 1.0) f_stream = 1.0;
//             if(f_stream < 0.0) f_stream = 0.0;
            
//             // Calculate cooling: mix of cold streams + hot halo cooling
//             double cold_stream_cooling = 0.0;
//             double hot_halo_cooling = 0.0;
            
//             if(rcool < galaxies[gal].Rvir) {
//                 // When rcool < Rvir: both cold streams and hot halo cooling
//                 // Cold stream component: rapid accretion on dynamical time
//                 cold_stream_cooling = f_stream * galaxies[gal].HotGas / 
//                                      (galaxies[gal].Rvir / galaxies[gal].Vvir) * dt;
                
//                 // Hot halo component: traditional cooling from the shocked gas
//                 hot_halo_cooling = (1.0 - f_stream) * (galaxies[gal].HotGas / galaxies[gal].Rvir) * 
//                                   (rcool / (2.0 * tcool_dyn)) * dt;
//             } else {
//                 // When rcool >= Rvir: only hot halo cooling (no cold streams)
//                 // rcool >= Rvir: This shouldn't occur for properly-classified hot-regime haloes
//                 // (such haloes belong in the CGM/cold-flow regime). Handle conservatively.
//                 hot_halo_cooling = (galaxies[gal].HotGas / galaxies[gal].Rvir) * 
//                                   (rcool / (2.0 * tcool_dyn)) * dt;
//             }

//             galaxies[gal].mdot_cool = hot_halo_cooling / dt;
//             galaxies[gal].mdot_stream = cold_stream_cooling / dt;
            
//             coolingGas = cold_stream_cooling + hot_halo_cooling;
//         }

//         if(coolingGas > galaxies[gal].HotGas) {
//             coolingGas = galaxies[gal].HotGas;
//         } else {
//             if(coolingGas < 0.0) coolingGas = 0.0;
//         }

//         // at this point we have calculated the maximal cooling rate
//         // if AGNrecipeOn we now reduce it in line with past heating before proceeding

//         /* Kept so the preventive term can be combined with the AGN suppression rather
//          * than stacked on top of it (PreventiveHeatingOn 3/4). */
//         const double coolingGas_preAGN = coolingGas;

//         if(run_params->AGNrecipeOn > 0 && coolingGas > 0.0) {
//             coolingGas = do_AGN_heating(coolingGas, gal, dt, x, rcool, galaxies, run_params);
//         }

//         /* ---- Mode 5: Voit t_cool/t_ff ceiling on the hot-regime cooling rate ----
//          *
//          * The CGM path already limits condensation to the mass above the thermally stable
//          * equilibrium, m_eq = M * (t_cool/t_ff) / 10 (Voit 2015; McCourt et al. 2012).
//          * The same construction here, for the isothermal hot halo: rcool is defined as the
//          * radius where t_cool = Rvir/Vvir and rho ~ r^-2, so
//          *     t_cool(Rvir) / t_dyn = (Rvir / rcool_uncapped)^2,
//          * and t_ff(Rvir) = sqrt(2) * Rvir/Vvir for an isothermal sphere.
//          *
//          * Applied as a CEILING, so it can only ever reduce the cooling rate. Note this is
//          * expected to bind rarely: measured on microUchuu at log Mvir 12.0-12.5, hot-regime
//          * haloes sit at t_cool/t_ff ~ 1.1 (z=0) to 1.4 (z=3), far below the threshold of 10,
//          * so the Voit rate (~0.6 M_hot/t_dyn) exceeds SAGE's standard hot-mode rate
//          * (~0.4 M_hot/t_dyn). The ratio does move the right way with redshift, but the
//          * criterion says these haloes should precipitate freely rather than be braked. */
//         if(run_params->PreventiveHeatingOn == 5 && coolingGas > 0.0 && galaxies[gal].RcoolToRvir > 0.0) {
//             /* PrecipCriterionOn gates the two factors here as it does on the CGM
//              * path, so the toggle means the same thing wherever this rate appears.
//              * Modes 0 and 4 both leave the ceiling at the bare free-fall rate;
//              * they differ only on the CGM path, where mode 4 keeps the hand-over. */
//             const int use_sigmoid = (run_params->PrecipCriterionOn == 1 ||
//                                      run_params->PrecipCriterionOn == 3);
//             const int use_meq     = (run_params->PrecipCriterionOn == 1 ||
//                                      run_params->PrecipCriterionOn == 2);
//             const double t_dyn = galaxies[gal].Rvir / galaxies[gal].Vvir;
//             const double t_ff  = M_SQRT2 * t_dyn;
//             const double tcool_over_tff = 1.0 / (galaxies[gal].RcoolToRvir * galaxies[gal].RcoolToRvir) / M_SQRT2;
//             const double sig = use_sigmoid
//                 ? 1.0 / (1.0 + exp(-(PRECIP_THRESHOLD - tcool_over_tff) / PRECIP_TRANSITION_WIDTH))
//                 : 1.0;
//             double m_eq = use_meq ? galaxies[gal].HotGas * (tcool_over_tff / PRECIP_THRESHOLD) : 0.0;
//             if(m_eq > galaxies[gal].HotGas) m_eq = galaxies[gal].HotGas;
//             double condensable = galaxies[gal].HotGas - m_eq;
//             if(condensable < 0.0) condensable = 0.0;
//             const double voit_max = sig * condensable / t_ff * dt;
//             if(coolingGas > voit_max) coolingGas = voit_max;
//         }

//         /* ---- Mode 6: gravitational (halo-accretion) heating offset ----
//          *
//          * Infalling gas and subhaloes thermalise part of their kinetic energy in the corona
//          * (Dekel & Birnboim 2008; Khochfar & Ostriker 2008). SAGE books a cooling mass m as
//          * carrying 0.5 m Vvir^2, and the infalling material arrives with the same specific
//          * energy, so a coupling efficiency epsilon offsets a cooling MASS of
//          * epsilon * dMvir, shared equally across the substeps of this snapshot.
//          *
//          * Unlike the AGN term this scales with the halo accretion rate, which rises steeply
//          * toward high z, so it brakes hardest where the cooling is fastest -- the epoch
//          * dependence the r_heat ratchet gets backwards. Negative dMvir (a stripped or
//          * mis-linked halo) contributes no heating. */
//         if(run_params->PreventiveHeatingOn == 6 && coolingGas > 0.0 && galaxies[gal].deltaMvir > 0.0) {
//             const int nsub = (galaxies[gal].SubstepsUsed > 0) ? galaxies[gal].SubstepsUsed : STEPS;
//             const double m_offset = run_params->PreventiveHeatingEfficiency * galaxies[gal].deltaMvir / nsub;
//             coolingGas -= m_offset;
//             if(coolingGas < 0.0) coolingGas = 0.0;
//         }

//         /* Preventive, non-AGN suppression. Modes 1/2 multiply it onto whatever the AGN
//          * left, which double-counts at z = 0 where the r_heat ratchet has already
//          * saturated. Modes 3/4 instead apply whichever of the two suppressions is
//          * stronger: the corona is held up either by gravitational heating or by the
//          * black hole, and those are not independent reservoirs to be stacked. */
//         if(run_params->PreventiveHeatingOn > 0 && coolingGas > 0.0) {
//             const double f_prev = preventive_suppression(gal, galaxies, run_params);
//             if(run_params->PreventiveHeatingOn >= 3) {
//                 const double f_agn = (coolingGas_preAGN > 0.0) ? coolingGas / coolingGas_preAGN : 1.0;
//                 coolingGas = coolingGas_preAGN * ((f_prev < f_agn) ? f_prev : f_agn);
//             } else {
//                 coolingGas *= f_prev;
//             }
//         }

//         if (coolingGas > 0.0) {
//             galaxies[gal].Cooling += 0.5 * coolingGas * galaxies[gal].Vvir * galaxies[gal].Vvir;
//         }
//     } else {
//         coolingGas = 0.0;
//     }

//     XASSERT(coolingGas >= 0.0, -1,
//             "Error: Cooling gas mass = %g should be >= 0.0", coolingGas);
//         galaxies[gal].tcool = (dt > 0.0)
//             ? (float)((coolingGas / dt) * 1.0e10 / run_params->Hubble_h
//                       * SEC_PER_GIGAYEAR / run_params->UnitTime_in_s)
//             : 0.0f;
//     return coolingGas;
// }
double cooling_recipe_hot(const int gal, const double dt, struct GALAXY *galaxies, const struct params *run_params)
{
    double coolingGas;

    galaxies[gal].tcool = 0.0f;

    if(galaxies[gal].HotGas > 0.0 && galaxies[gal].Vvir > 0.0) {
        const double tcool_dyn = galaxies[gal].Rvir / galaxies[gal].Vvir;
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
        const double rho_rcool = x / tcool_dyn * (1.5 * MU_IONISED);  // 3/2 * mu for a fully ionized gas

        if(rho_rcool <= 0.0) {
            return 0.0;
        }

        // an isothermal density profile for the hot gas is assumed here
        const double rho0 = galaxies[gal].HotGas / (4 * M_PI * galaxies[gal].Rvir);
        double rcool = sqrt(rho0 / rho_rcool);

        galaxies[gal].RcoolToRvir = rcool / galaxies[gal].Rvir;  // store uncapped ratio for diagnostics

        coolingGas = 0.0;

        if(run_params->CGMrecipeOn == 0) {
            // SAGE C16 hot-halo cooling, both branches (Croton et al. 2016).
            // tcool here is the halo dynamical time Rvir/Vvir.
            if(rcool > galaxies[gal].Rvir) {
                // Rapid "cold accretion": the whole corona cools within a
                // dynamical time.  Discontinuous with the branch below by a
                // factor 2 at rcool = Rvir; that is the published behaviour.
                coolingGas = galaxies[gal].HotGas / tcool_dyn * dt;
            } else {
                // Quasi-static cooling flow.
                coolingGas = (galaxies[gal].HotGas / galaxies[gal].Rvir) * (rcool / (2.0 * tcool_dyn)) * dt;
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
                                  (rcool / (2.0 * tcool_dyn)) * dt;
            } else {
                // When rcool >= Rvir: only hot halo cooling (no cold streams)
                // rcool >= Rvir: This shouldn't occur for properly-classified hot-regime haloes
                // (such haloes belong in the CGM/cold-flow regime). Handle conservatively.
                hot_halo_cooling = (galaxies[gal].HotGas / galaxies[gal].Rvir) * 
                                  (rcool / (2.0 * tcool_dyn)) * dt;
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
        galaxies[gal].tcool = (dt > 0.0)
            ? (float)((coolingGas / dt) * 1.0e10 / run_params->Hubble_h
                      * SEC_PER_GIGAYEAR / run_params->UnitTime_in_s)
            : 0.0f;
    return coolingGas;
}

/*
 * CGM precipitation-driven cooling recipe (SAGE26 two-regime model).
 *
 * Computes cooling from the CGMgas reservoir using the Voit (2015) t_cool/t_ff
 * criterion. Solves for the cooling radius via solve_for_rcool(), evaluates the
 * mean density within that radius, and returns the cooled mass for this substep.
 * AGN heating via do_AGN_heating_cgm() is applied before the return.
 */
// double cooling_recipe_cgm(const int gal, const double dt, struct GALAXY *galaxies,
//                          const struct params *run_params)
// {
//     double coolingGas = 0.0;

//     // ========================================================================
//     // EARLY EXIT CONDITIONS
//     // ========================================================================
//     if(galaxies[gal].CGMgas <= 0.0 || galaxies[gal].Vvir <= 0.0 || galaxies[gal].Rvir <= 0.0) {
//         /* Reset the diagnostics rather than returning straight away.  This exit
//          * previously left tcool / tff / tcool_over_tff / RcoolToRvir holding
//          * whatever they were the last time the halo had a reservoir, which went
//          * stale for every drained halo -- 22% of z = 0 Regime-0 centrals -- and
//          * inflated the high-ratio tail of any figure that selects on Regime
//          * alone (7.3% above the threshold instead of the true 0.29%).
//          * Diagnostics only; no mass or energy is affected. */
//         galaxies[gal].tcool = 0.0;
//         galaxies[gal].tff = -1.0;
//         galaxies[gal].tcool_over_tff = -1.0;
//         galaxies[gal].MachNumber = -1.0;
//         galaxies[gal].RcoolToRvir = -1.0;
//         return 0.0;
//     }

//     // ========================================================================
//     // STEP 1: CALCULATE COOLING TIME (CGS UNITS) WITH DENSITY PROFILE
//     // ========================================================================

//     // Get density profile type (0: uniform, 1: NFW, 2: beta, 3: Stern+21 cooling flow)
//     // IMPORTANT: Density profile physics only applies to CGM-regime haloes (Regime == 0)
//     // Hot-regime haloes always use uniform density for simple CGM depletion
//     const int profile_type = (galaxies[gal].Regime == 0) ? run_params->CGMDensityProfile : 0;

//     // Virial temperature.  Profile 3 uses the cooling-flow temperature
//     // T^(s) = (6/5A) T_vir of Stern et al. (2021) Eq 11 with A ~ 1, which is
//     // 20% above T_vir; it feeds both the cooling-function lookup and t_cool.
//     const double temp = VIRIAL_TEMP_COEFF * galaxies[gal].Vvir * galaxies[gal].Vvir
//                       * ((profile_type == 3) ? STERN_TEMP_BOOST : 1.0); // Kelvin

//     // Metallicity
//     double logZ = -10.0;
//     if(galaxies[gal].MetalsCGMgas > 0) {
//         logZ = log10(galaxies[gal].MetalsCGMgas / galaxies[gal].CGMgas);
//     }

//     // Cooling function (erg cm^3 s^-1)
//     double lambda = get_metaldependent_cooling_rate(log10(temp), logZ);

//     if(lambda <= 0.0) {
//         return 0.0;
//     }

//     // Convert CGM mass and radius to CGS
//     const double CGMgas_cgs = galaxies[gal].CGMgas * 1e10 * SOLAR_MASS / run_params->Hubble_h; // g
//     const double Rvir_cgs = galaxies[gal].Rvir * CM_PER_MPC / run_params->Hubble_h; // cm
//     const double Mvir_cgs = galaxies[gal].Mvir * 1e10 * SOLAR_MASS / run_params->Hubble_h; // g
//     const double Mvir_Msun = CODE_MASS_TO_MSUN(galaxies[gal].Mvir, run_params->Hubble_h); // Msun
//     const double z = run_params->ZZ[galaxies[gal].SnapNum];

//     // ========================================================================
//     // STEP 1b: SOLVE FOR COOLING RADIUS (consistent with hot regime approach)
//     // ========================================================================
//     // Find r_cool where t_cool(r_cool) = t_ff(r_cool)
//     // This is done iteratively for all profile types

//     const double r_cool_cgs = solve_for_rcool(CGMgas_cgs, Rvir_cgs, Mvir_cgs, Mvir_Msun,
//                                                temp, lambda, z, profile_type, run_params);

//     // Get density at the cooling radius
//     const double mass_density_cgs = cgm_density_at_radius(r_cool_cgs, CGMgas_cgs, Rvir_cgs,
//                                                            Mvir_Msun, z, profile_type);

//     if(!(mass_density_cgs > 0.0)) {   /* also rejects NaN */
//         return 0.0;
//     }

//     // Store r_cool / R_vir for diagnostics
//     galaxies[gal].RcoolToRvir = r_cool_cgs / Rvir_cgs;

//     // Convert r_cool to code units
//     const double r_cool = r_cool_cgs / (CM_PER_MPC / run_params->Hubble_h);

//     // Cooling time at r_cool: tcool = (3/2) * mu * m_p * k * T / (rho * Lambda)
//     const double mu = MU_IONISED;
//     const double tcool_cgs = (1.5 * mu * PROTONMASS * BOLTZMANN * temp) / (mass_density_cgs * lambda);
//     const double tcool = tcool_cgs / run_params->UnitTime_in_s; // code units

//     // ========================================================================
//     // STEP 2: CALCULATE FREE-FALL TIME AT r_cool
//     // ========================================================================

//     // Enclosed mass at r_cool (using proper profile)
//     const double M_enclosed_rcool = cgm_enclosed_mass(r_cool_cgs, Mvir_cgs, Rvir_cgs,
//                                                        Mvir_Msun, z, profile_type);
//     // Convert to code units
//     const double M_enclosed_code = M_enclosed_rcool / (1e10 * SOLAR_MASS / run_params->Hubble_h);

//     // Gravitational acceleration at r_cool
//     const double g_accel = (M_enclosed_code > 0.0 && r_cool > 0.0)
//         ? run_params->G * M_enclosed_code / (r_cool * r_cool)
//         : 0.0;

//     // Free-fall time at r_cool: tff = sqrt(2*r_cool/g)
//     if(g_accel <= 0.0) {
//         galaxies[gal].tcool = (float)(tcool * run_params->UnitTime_in_s / SEC_PER_GIGAYEAR);
//         galaxies[gal].tff = -1.0;
//         galaxies[gal].tcool_over_tff = -1.0;
//         galaxies[gal].MachNumber = -1.0;
//         galaxies[gal].tdeplete = -1.0;
//         galaxies[gal].RcoolToRvir = -1.0;
//         return 0.0;
//     }
//     const double tff = sqrt(2.0 * r_cool / g_accel); // code units

//     // ========================================================================
//     // STEP 2b: CHARACTERISTIC RADIUS FOR PRECIPITATION CRITERION
//     // ========================================================================
//     // Evaluate t_cool/t_ff at r_cool (traditional Voit-style choice).
//     const double tcool_char = tcool;
//     const double tff_char = tff;
//     const double tcool_over_tff_char = tcool / tff;

//     galaxies[gal].tcool = (float)(tcool_char * run_params->UnitTime_in_s / SEC_PER_GIGAYEAR);
//     galaxies[gal].tff = (float)(tff_char * run_params->UnitTime_in_s / SEC_PER_GIGAYEAR);
//     galaxies[gal].tcool_over_tff = (float)tcool_over_tff_char;

//     // // Convert to Myr for plotting
//     // const double tcool_char_Myr = tcool_char * run_params->UnitTime_in_s / (3.154e12);
//     // const double tff_char_Myr = tff_char * run_params->UnitTime_in_s / (3.154e12);
//     // const double tcool_over_tff_char_Myr = tcool_char * run_params->UnitTime_in_s / (3.154e12) / (tff_char * run_params->UnitTime_in_s / (3.154e12));

//     // // Conver to Gyr for plotting
//     // const double tcool_char_Gyr = tcool_char * run_params->UnitTime_in_s / (3.154e16);
//     // const double tff_char_Gyr = tff_char * run_params->UnitTime_in_s / (3.154e16);
//     // const double tcool_over_tff_char_Gyr = tcool_over_tff_char * run_params->UnitTime_in_s / (3.154e16);

//     // // Store characteristic-radius values for diagnostics/plotting
//     // galaxies[gal].tcool = tcool_char_Gyr;
//     // galaxies[gal].tff = tff_char_Gyr;
//     // galaxies[gal].tcool_over_tff = tcool_over_tff_char_Gyr;

//     /* Inflow Mach number.  tcool is normalised from CGS by UnitTime_in_s while
//      * tff is in code units whose length carries an implicit 1/h, so the stored
//      * ratio is larger than the physical one by 1/h; undo that here so the
//      * reported Mach number is right even though the criterion above still uses
//      * the uncorrected ratio (changing that would alter the fiducial model). */
//     galaxies[gal].MachNumber = (tcool_over_tff_char > 0.0)
//         ? STERN_MACH_COEFF / (tcool_over_tff_char * run_params->Hubble_h)
//         : -1.0;

//     // ========================================================================
//     // STEP 3: PRECIPITATION CRITERION
//     // ========================================================================

//     double precipitation_fraction = 0.0;

//     /* The precipitation rate carries two independent suppression factors,
//      *
//      *     mdot = S((threshold - r)/width) * (M_CGM - M_eq) / t_ff,
//      *     r = t_cool/t_ff,   M_eq = M_CGM * r / threshold,
//      *
//      * and PrecipCriterionOn selects which of them are applied, so each can be
//      * ablated on its own:
//      *
//      *   0  neither -- every CGM-regime halo accretes its whole reservoir on a
//      *      free-fall time, mdot = M_CGM/t_ff.  The f_inflow == 1 control.
//      *   1  both (default) -- the rate as submitted.
//      *   2  M_eq only -- drops the sigmoid.  Measured on Millennium the sigmoid
//      *      never leaves its ceiling S(threshold/width) = 0.9933: its
//      *      inflow-weighted mean over the CGM population is 0.992 at z = 0
//      *      rising to 0.993 at z = 6, in every mass bin from log Mvir 10 to
//      *      12.5, because CGM-regime haloes sit at r ~ 0.1-0.4 rather than near
//      *      the threshold.  Switching it off is therefore a near-uniform 0.7%
//      *      rescaling of the inflow rate, and removes PRECIP_TRANSITION_WIDTH
//      *      from the model.
//      *   3  sigmoid only -- drops the condensation term.  This is the bare
//      *      sigmoid printed as the rate in the first submission, kept so the
//      *      printed and implemented forms can be run against each other.  It is
//      *      the weaker of the two: the condensation term is what drives the rate
//      *      to exactly zero at the threshold, where the sigmoid is still 0.5.
//      *   5  SAGE16 cold accretion -- mdot = M_CGM / (Rvir/Vvir), the published
//      *      rate on the old rcool > Rvir branch.  Bypasses the criterion like
//      *      mode 0 but drains on the dynamical rather than the free-fall time,
//      *      so it is exactly sqrt(2) faster than mode 0.
//      *   4  neither, but keeping everything else about the precipitation path.
//      *      This is the reference the single-factor ablations should be measured
//      *      against, and it is NOT the same as mode 0: mode 0 leaves the
//      *      criterion before the hand-over to standard cooling below, so it
//      *      accretes at M_CGM/t_ff no matter how stable the halo is, whereas
//      *      mode 4 still hands very stable haloes (r > ~19) to M_CGM/t_cool.
//      *      The two therefore differ by the hand-over alone, which is what makes
//      *      mode 0 an imperfect control: it changes two things at once.
//      *
//      * The sigmoid is evaluated in all four active modes even where it is not
//      * applied to the rate, because it also supplies the hand-over test to
//      * standard cooling below.  Keeping that common makes modes 2, 3 and 4
//      * differ from mode 1 by exactly the factors named and nothing else, and
//      * leaves mode 1 unchanged.  With mode 4 the set is a 2x2 factorial in the
//      * two factors, so each can be read off against a common reference. */
//     const int use_precip  = (run_params->PrecipCriterionOn >= 1 &&
//                              run_params->PrecipCriterionOn <= 4);
//     const int use_sigmoid = (run_params->PrecipCriterionOn == 1 ||
//                              run_params->PrecipCriterionOn == 3);
//     const int use_meq     = (run_params->PrecipCriterionOn == 1 ||
//                              run_params->PrecipCriterionOn == 2);

//     if(!use_precip) {
//         /* Modes 0 and 5 both bypass the criterion entirely and drain the whole
//          * reservoir on a single timescale, differing only in which one:
//          *
//          *   0  t_ff  at r_cool -- the free-fall control.
//          *   5  t_dyn = Rvir/Vvir -- SAGE16 cold accretion, the rate the
//          *      published model used on its rcool > Rvir rapid-cooling branch.
//          *
//          * For the uniform profile t_ff = sqrt(2) Rvir/Vvir exactly, so mode 5 is
//          * a uniform sqrt(2) = 1.41x faster than mode 0 -- not a different shape,
//          * just a different constant in front of the same M_CGM/t rate.  Neither
//          * mode takes the hand-over to standard cooling; the reservoir always
//          * drains on the chosen timescale however thermally stable the halo is. */
//         precipitation_fraction = 1.0;
//     //     if(run_params->CGMsimpleInflowOn == 1 && run_params->PrecipCriterionOn == 0) {
//     //         // fprintf(stderr, "\nCGMsimpleInflowOn=1 and PrecipCriterionOn=0: draining entire CGM on free-fall time\n");
//     //         // fflush(stderr);
//     //         coolingGas = (galaxies[gal].CGMgas / (tff_char + tcool)) * dt;
//     //     }
//     //     const double t_inflow = (run_params->PrecipCriterionOn == 5)
//     //         ? galaxies[gal].Rvir / galaxies[gal].Vvir
//     //         : tff_char;
//     //     if(t_inflow > 0.0) {
//     //         coolingGas = galaxies[gal].CGMgas / t_inflow * dt;
//     //         if(coolingGas > galaxies[gal].CGMgas) coolingGas = galaxies[gal].CGMgas;
//     //         if(coolingGas < 0.0) coolingGas = 0.0;
//     //     }
//     // }

//         if(run_params->CGMsimpleInflowOn == 1 && run_params->PrecipCriterionOn == 0) {
//             coolingGas = (galaxies[gal].CGMgas / (tff + tcool)) * dt;
//             // fprintf(stderr, "\nCGMsimpleInflowOn=1 and PrecipCriterionOn=0: coolingGas = %g, CGMgas = %g, tff_char = %g, tcool = %g, dt = %g\n",
//             //         coolingGas, galaxies[gal].CGMgas, tff_char, tcool, dt);
//             // fflush(stderr);
//         } else {
//             const double t_inflow = (run_params->PrecipCriterionOn == 5)
//                 ? galaxies[gal].Rvir / galaxies[gal].Vvir
//                 : tff_char;
//             // fprintf(stderr, "\nUsing different inflow time: t_inflow = %g\n", t_inflow);
//             // fflush(stderr);

//             if(t_inflow > 0.0) {
//                 coolingGas = (galaxies[gal].CGMgas / t_inflow) * dt;
//             }
//         }
//         if(coolingGas > galaxies[gal].CGMgas) {
//             coolingGas = galaxies[gal].CGMgas;
//         }
//         if(coolingGas < 0.0) {
//             coolingGas = 0.0;
//         }
//     }

//     // Logistic sigmoid centred on PRECIP_THRESHOLD, characteristic width = 2.
//     // f = 1 / (1 + exp(-(threshold - r) / 2))
//     // Smoothly ranges from ~1 (very unstable) through 0.5 at threshold to ~0 (very stable).
//     // Falls back to standard cooling once the sigmoid is negligible (< 0.01),
//     // which occurs at t_cool/t_ff ~ 19.  The hand-over is tested in every active
//     // mode (1-4); only the multiplication into the rate is gated on use_sigmoid.
//     if(use_precip) {
//         const double x = (PRECIP_THRESHOLD - tcool_over_tff_char) / PRECIP_TRANSITION_WIDTH;
//         const double f = 1.0 / (1.0 + exp(-x));
//         if(f >= 0.01) {
//             precipitation_fraction = use_sigmoid ? f : 1.0;
//         } else {
//             if(tcool_char > 0) {
//                 coolingGas = galaxies[gal].CGMgas / tcool_char * dt;
//                 // fprintf(stderr, "\nUsing standard cooling: coolingGas = %g, CGMgas = %g, tcool_char = %g, dt = %g\n",
//                 //         coolingGas, galaxies[gal].CGMgas, tcool_char, dt);
//                 // fflush(stderr);
//                 if(coolingGas > galaxies[gal].CGMgas)
//                     coolingGas = galaxies[gal].CGMgas;
//             }
//         }
//     }

    

//     // ========================================================================
//     // STEP 4: CALCULATE PRECIPITATION RATE
//     // ========================================================================

//     if(use_precip && precipitation_fraction > 0.0) {
//         // Self-regulating precipitation: gas precipitates on the free-fall
//         // timescale, but only the CGM *above* the tcool/tff = PRECIP_THRESHOLD
//         // equilibrium condenses. At fixed profile shape and temperature t_cool
//         // scales as 1/rho, i.e. as 1/M_CGM, while t_ff is set by the
//         // (DM-dominated) potential -- so the reservoir the halo can stably hold
//         // is
//         //     M_eq = M_CGM * (tcool/tff) / PRECIP_THRESHOLD
//         // and dM/dt = f_precip * (M_CGM - M_eq) / t_ff relaxes toward the Voit
//         // equilibrium instead of emptying the reservoir: as the CGM drains,
//         // tcool/tff rises, M_eq -> M_CGM, and the flow shuts off. Late-time
//         // inflow is then limited to the rate at which infall and SN-reheated
//         // gas push the CGM back over the equilibrium mass, rather than the
//         // free-fall dump of the entire stored reservoir.
//         const double m_eq = use_meq
//             ? galaxies[gal].CGMgas * (tcool_over_tff_char / PRECIP_THRESHOLD)
//             : 0.0;
//         double condensing_mass = galaxies[gal].CGMgas - m_eq;
//         if(condensing_mass < 0.0) {
//             condensing_mass = 0.0;   /* sigmoid tail above threshold: stable, no condensation */
//         }

//         const double precip_rate = precipitation_fraction * condensing_mass / tff_char;

//         // Apply the precipitation rate to the cooling gas
//         coolingGas = precip_rate * dt;

//         // fprintf(stderr, "\nCGM precipitation: coolingGas = %g, CGMgas = %g, tff_char = %g, tcool = %g, dt = %g\n",
//         //         coolingGas, galaxies[gal].CGMgas, tff_char, tcool, dt);
//         // fflush(stderr);

//         // coolingGas = precip_rate * dt;

//         // Physical limits
//         if(coolingGas > galaxies[gal].CGMgas) {
//             coolingGas = galaxies[gal].CGMgas;
//         }
//         if(coolingGas < 0.0) {
//             coolingGas = 0.0;
//         }
//     }

//     // fprintf(stderr, "\nCGMsimpleInflowOn=1 and PrecipCriterionOn=0: coolingGas = %g, CGMgas = %g, tff_char = %g, tcool = %g, dt = %g\n",
//     //                 coolingGas, galaxies[gal].CGMgas, tff_char, tcool, dt);
//     // fflush(stderr);

//     // AGN heating only fires for proper CGM-regime (Regime==0) halos.
//     // Regime==1 residual CGMgas drains naturally; do_AGN_heating() on HotGas
//     // in cooling_recipe_hot() handles all AGN for hot-halo galaxies.
//     /* Kept so PreventiveHeatingOn == 4 can combine with the AGN suppression instead of
//      * stacking on top of it; see the preventive block below. */
//     if(galaxies[gal].Regime == 0) {
//         // AGN x parameter: (k_B T / lambda) in code-units density*time -- passed to
//         // both AGN heating paths (Bondi-Hoyle uses it; empirical and cold-cloud do not).
//         const double x_agn = (PROTONMASS * BOLTZMANN * temp / lambda)
//                              / (run_params->UnitDensity_in_cgs * run_params->UnitTime_in_s);

//         // r_heat ratchet, no decay, capped at Rvir (suppression and ratchet
//         // update handled inside do_AGN_heating_cgm when AGN is active).
//         if(run_params->AGNrecipeOn > 0) {
//             coolingGas = do_AGN_heating_cgm(coolingGas, gal, dt, x_agn, r_cool, galaxies, run_params);
//         } else {
//             // No AGN: still apply r_heat suppression so quenching persists
//             if(galaxies[gal].r_heat >= r_cool ||
//                (run_params->CGMrecipeOn == 1 && galaxies[gal].r_heat >= 0.99 * r_cool)) {
//                 coolingGas = 0.0;
//             } else if(galaxies[gal].r_heat > 0.0f) {
//                 coolingGas *= 1.0 - galaxies[gal].r_heat / r_cool;
//             }
//         }
//     }

//     // /* Preventive, non-AGN suppression. Mode 1 is hot-regime only, so the CGM path
//     //  * applies it at mode 2 alone. */
//     // if((run_params->PreventiveHeatingOn == 2 || run_params->PreventiveHeatingOn == 4) && coolingGas > 0.0) {
//     //     const double f_prev = preventive_suppression(gal, galaxies, run_params);
//     //     if(run_params->PreventiveHeatingOn == 4) {
//     //         const double f_agn = (coolingGas_preAGN_cgm > 0.0) ? coolingGas / coolingGas_preAGN_cgm : 1.0;
//     //         coolingGas = coolingGas_preAGN_cgm * ((f_prev < f_agn) ? f_prev : f_agn);
//     //     } else {
//     //         coolingGas *= f_prev;
//     //     }
//     // }

//     // ========================================================================
//     // STEP 5: TRACK COOLING ENERGY
//     // ========================================================================

//     // Energy associated with cooling (for feedback balance tracking)
//     if(coolingGas > 0.0) {
//         // Specific energy ~ 0.5 * Vvir^2 (thermal + kinetic)
//         galaxies[gal].Cooling += 0.5 * coolingGas * galaxies[gal].Vvir * galaxies[gal].Vvir;
//     }

//     // ========================================================================
//     // STEP 6: CALCULATE DEPLETION TIMESCALE (DIAGNOSTIC)
//     // ========================================================================

//     // Depletion timescale (only meaningful for CGM-regime haloes)
//     if(galaxies[gal].Regime == 0) {
//         if(precipitation_fraction > 1e-6 && isfinite(tff_char)) {
//             const double depletion_time = tff_char / precipitation_fraction;
//             galaxies[gal].tdeplete = isfinite(depletion_time) ? (float)depletion_time : -1.0f;
//         } else {
//             galaxies[gal].tdeplete = -1.0f;
//         }
//     } else {
//         // Hot-regime haloes: reset diagnostic fields (density profile physics doesn't apply)
//         galaxies[gal].tcool = -1.0f;
//         galaxies[gal].tff = -1.0f;
//         galaxies[gal].tcool_over_tff = -1.0f;
//         galaxies[gal].MachNumber = -1.0f;
//         galaxies[gal].tdeplete = -1.0f;
//     }

//     // Sanity check
//     XASSERT(coolingGas >= 0.0, -1, "Error: Cooling gas mass = %g should be >= 0.0", coolingGas);
//     XASSERT(coolingGas <= galaxies[gal].CGMgas + 1e-12, -1,
//             "Error: Cooling gas = %g exceeds CGM gas = %g", coolingGas, galaxies[gal].CGMgas);

//     galaxies[gal].tcool = (dt > 0.0)
//         ? (float)((coolingGas / dt) * 1.0e10 / run_params->Hubble_h
//                   * SEC_PER_GIGAYEAR / run_params->UnitTime_in_s)
//         : 0.0f;

//     return coolingGas;
// }
double cooling_recipe_cgm(const int gal, const double dt, struct GALAXY *galaxies,
                         const struct params *run_params)
{
    double coolingGas = 0.0;

    // ========================================================================
    // EARLY EXIT CONDITIONS
    // ========================================================================
    if(galaxies[gal].CGMgas <= 0.0 || galaxies[gal].Vvir <= 0.0 || galaxies[gal].Rvir <= 0.0) {
        /* Reset the diagnostics rather than returning straight away.  This exit
         * previously left tcool / tff / tcool_over_tff / RcoolToRvir holding
         * whatever they were the last time the halo had a reservoir, which went
         * stale for every drained halo -- 22% of z = 0 Regime-0 centrals -- and
         * inflated the high-ratio tail of any figure that selects on Regime
         * alone (7.3% above the threshold instead of the true 0.29%).
         * Diagnostics only; no mass or energy is affected. */
        galaxies[gal].tcool = 0.0;
        galaxies[gal].tff = -1.0;
        galaxies[gal].tcool_over_tff = -1.0;
        galaxies[gal].MachNumber = -1.0;
        galaxies[gal].RcoolToRvir = -1.0;
        return 0.0;
    }

    // Get density profile type (0: uniform, 1: NFW, 2: beta, 3: Stern+21 cooling flow)
    // IMPORTANT: Density profile physics only applies to CGM-regime haloes (Regime == 0)
    // Hot-regime haloes always use uniform density for simple CGM depletion
    const int profile_type = (galaxies[gal].Regime == 0) ? run_params->CGMDensityProfile : 0;

    // Virial temperature.  Profile 3 uses the cooling-flow temperature
    // T^(s) = (6/5A) T_vir of Stern et al. (2021) Eq 11 with A ~ 1, which is
    // 20% above T_vir; it feeds both the cooling-function lookup and t_cool.
    const double temp = VIRIAL_TEMP_COEFF * galaxies[gal].Vvir * galaxies[gal].Vvir
                      * ((profile_type == 3) ? STERN_TEMP_BOOST : 1.0); // Kelvin

    // Metallicity
    double logZ = -10.0;
    if(galaxies[gal].MetalsCGMgas > 0) {
        logZ = log10(galaxies[gal].MetalsCGMgas / galaxies[gal].CGMgas);
    }

    // Cooling function (erg cm^3 s^-1)
    double lambda = get_metaldependent_cooling_rate(log10(temp), logZ);

    if(lambda <= 0.0) {
        return 0.0;
    }

    // Convert CGM mass and radius to CGS
    const double CGMgas_cgs = galaxies[gal].CGMgas * 1e10 * SOLAR_MASS / run_params->Hubble_h; // g
    const double Rvir_cgs = galaxies[gal].Rvir * CM_PER_MPC / run_params->Hubble_h; // cm
    const double Mvir_cgs = galaxies[gal].Mvir * 1e10 * SOLAR_MASS / run_params->Hubble_h; // g
    const double Mvir_Msun = CODE_MASS_TO_MSUN(galaxies[gal].Mvir, run_params->Hubble_h); // Msun
    const double z = run_params->ZZ[galaxies[gal].SnapNum];


    // Find r_cool where t_cool(r_cool) = t_ff(r_cool)
    // This is done iteratively for all profile types
    const double r_cool_cgs = solve_for_rcool(CGMgas_cgs, Rvir_cgs, Mvir_cgs, Mvir_Msun,
                                               temp, lambda, z, profile_type, run_params);

    // Get density at the cooling radius
    const double mass_density_cgs = cgm_density_at_radius(r_cool_cgs, CGMgas_cgs, Rvir_cgs,
                                                           Mvir_Msun, z, profile_type);

    if(!(mass_density_cgs > 0.0)) {   /* also rejects NaN */
        return 0.0;
    }

    // Store r_cool / R_vir for diagnostics
    galaxies[gal].RcoolToRvir = r_cool_cgs / Rvir_cgs;

    // Convert r_cool to code units
    const double r_cool = r_cool_cgs / (CM_PER_MPC / run_params->Hubble_h);

    // Cooling time at r_cool: tcool = (3/2) * mu * m_p * k * T / (rho * Lambda)
    const double mu = MU_IONISED;
    const double tcool_cgs = (1.5 * mu * PROTONMASS * BOLTZMANN * temp) / (mass_density_cgs * lambda);
    const double tcool = tcool_cgs / run_params->UnitTime_in_s; // code units


    // Enclosed mass at r_cool (using proper profile)
    const double M_enclosed_rcool = cgm_enclosed_mass(r_cool_cgs, Mvir_cgs, Rvir_cgs,
                                                       Mvir_Msun, z, profile_type);
    // Convert to code units
    const double M_enclosed_code = M_enclosed_rcool / (1e10 * SOLAR_MASS / run_params->Hubble_h);

    // Gravitational acceleration at r_cool
    const double g_accel = (M_enclosed_code > 0.0 && r_cool > 0.0)
        ? run_params->G * M_enclosed_code / (r_cool * r_cool)
        : 0.0;

    // Free-fall time at r_cool: tff = sqrt(2*r_cool/g)
    if(g_accel <= 0.0) {
        galaxies[gal].tcool = (float)(tcool * run_params->UnitTime_in_s / SEC_PER_GIGAYEAR);
        galaxies[gal].tff = -1.0;
        galaxies[gal].tcool_over_tff = -1.0;
        galaxies[gal].MachNumber = -1.0;
        galaxies[gal].tdeplete = -1.0;
        galaxies[gal].RcoolToRvir = -1.0;
        return 0.0;
    }
    const double tff = sqrt(2.0 * r_cool / g_accel); // code units

    const double tcool_char = tcool;
    const double tff_char = tff;
    const double tcool_over_tff_char = tcool / tff;

    galaxies[gal].tcool = (float)(tcool_char * run_params->UnitTime_in_s / SEC_PER_GIGAYEAR);
    galaxies[gal].tff = (float)(tff_char * run_params->UnitTime_in_s / SEC_PER_GIGAYEAR);
    galaxies[gal].tcool_over_tff = (float)tcool_over_tff_char;



    /* Inflow Mach number.  tcool is normalised from CGS by UnitTime_in_s while
     * tff is in code units whose length carries an implicit 1/h, so the stored
     * ratio is larger than the physical one by 1/h; undo that here so the
     * reported Mach number is right even though the criterion above still uses
     * the uncorrected ratio (changing that would alter the fiducial model). */
    galaxies[gal].MachNumber = (tcool_over_tff_char > 0.0)
        ? STERN_MACH_COEFF / (tcool_over_tff_char * run_params->Hubble_h)
        : -1.0;

        if(run_params->CGMrecipeOn == 1) {
            coolingGas = (galaxies[gal].CGMgas / (tff + tcool)) * dt;
            // fprintf(stderr, "\nCGMsimpleInflowOn=1 and PrecipCriterionOn=0: coolingGas = %g, CGMgas = %g, tff_char = %g, tcool = %g, dt = %g\n",
            //         coolingGas, galaxies[gal].CGMgas, tff_char, tcool, dt);
            // fflush(stderr);
        } else {
            // CGM recipe off, no CGM should be accreted or even there, CGMgas = 0.0, so coolingGas = 0.0
            coolingGas = 0.0;
        }

        if(coolingGas > galaxies[gal].CGMgas) {
            coolingGas = galaxies[gal].CGMgas;
        }
        if(coolingGas < 0.0) {
            coolingGas = 0.0;
        }

    
    // AGN heating only fires for proper CGM-regime (Regime==0) halos.
    // Regime==1 residual CGMgas drains naturally; do_AGN_heating() on HotGas
    // in cooling_recipe_hot() handles all AGN for hot-halo galaxies.
    /* Kept so PreventiveHeatingOn == 4 can combine with the AGN suppression instead of
     * stacking on top of it; see the preventive block below. */
    if(galaxies[gal].Regime == 0) {
        // AGN x parameter: (k_B T / lambda) in code-units density*time -- passed to
        // both AGN heating paths (Bondi-Hoyle uses it; empirical and cold-cloud do not).
        const double x_agn = (PROTONMASS * BOLTZMANN * temp / lambda)
                             / (run_params->UnitDensity_in_cgs * run_params->UnitTime_in_s);

        // r_heat ratchet, no decay, capped at Rvir (suppression and ratchet
        // update handled inside do_AGN_heating_cgm when AGN is active).
        if(run_params->AGNrecipeOn > 0) {
            coolingGas = do_AGN_heating_cgm(coolingGas, gal, dt, x_agn, r_cool, galaxies, run_params);
        } else {
            // No AGN: still apply r_heat suppression so quenching persists
            if(galaxies[gal].r_heat >= r_cool ||
               (run_params->CGMrecipeOn == 1 && galaxies[gal].r_heat >= 0.99 * r_cool)) {
                coolingGas = 0.0;
            } else if(galaxies[gal].r_heat > 0.0f) {
                coolingGas *= 1.0 - galaxies[gal].r_heat / r_cool;
            }
        }
    }

    // Energy associated with cooling (for feedback balance tracking)
    if(coolingGas > 0.0) {
        // Specific energy ~ 0.5 * Vvir^2 (thermal + kinetic)
        galaxies[gal].Cooling += 0.5 * coolingGas * galaxies[gal].Vvir * galaxies[gal].Vvir;
    }


    // Depletion timescale (only meaningful for CGM-regime haloes)
    if(galaxies[gal].Regime == 0) {
        if(galaxies[gal].CGMgas > 0.0 && isfinite(tff_char)) {
            const double depletion_time = tff_char / galaxies[gal].CGMgas * coolingGas;
            galaxies[gal].tdeplete = isfinite(depletion_time) ? (float)depletion_time : -1.0f;
        } else {
            galaxies[gal].tdeplete = -1.0f;
        }
    } else {
        // Hot-regime haloes: reset diagnostic fields (density profile physics doesn't apply)
        galaxies[gal].tcool = -1.0f;
        galaxies[gal].tff = -1.0f;
        galaxies[gal].tcool_over_tff = -1.0f;
        galaxies[gal].MachNumber = -1.0f;
        galaxies[gal].tdeplete = -1.0f;
    }

    // Sanity check
    XASSERT(coolingGas >= 0.0, -1, "Error: Cooling gas mass = %g should be >= 0.0", coolingGas);
    XASSERT(coolingGas <= galaxies[gal].CGMgas + 1e-12, -1,
            "Error: Cooling gas = %g exceeds CGM gas = %g", coolingGas, galaxies[gal].CGMgas);

    galaxies[gal].tcool = (dt > 0.0)
        ? (float)((coolingGas / dt) * 1.0e10 / run_params->Hubble_h
                  * SEC_PER_GIGAYEAR / run_params->UnitTime_in_s)
        : 0.0f;

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
    galaxies[gal].tcool_over_tff = -1.0;
    galaxies[gal].MachNumber = -1.0;
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
            galaxies[gal].tcool_over_tff = -1.0f;
            galaxies[gal].MachNumber = -1.0f;
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
    galaxies[gal].tcool = (dt > 0.0)
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

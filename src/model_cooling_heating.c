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


// ============================================================================
// CGM Density Profile Helper Functions
// ============================================================================

// NFW profile: ρ(r) = ρ_s / [(r/r_s)(1 + r/r_s)²]
// Returns the normalization ρ_s given total mass M_CGM within R_vir
double nfw_rho_s(const double M_CGM, const double Rvir, const double c_NFW)
{
    const double r_s = Rvir / c_NFW;
    // M_CGM = 4π ρ_s r_s³ × [ln(1+c) - c/(1+c)]
    const double f_c = log(1.0 + c_NFW) - c_NFW / (1.0 + c_NFW);
    return M_CGM / (4.0 * M_PI * r_s * r_s * r_s * f_c);
}

// NFW density at radius r
double nfw_density(const double r, const double rho_s, const double r_s)
{
    const double x = r / r_s;
    if(x < 1e-10) return rho_s / (1e-10 * 1.0 * 1.0);  // Avoid singularity at r=0
    return rho_s / (x * (1.0 + x) * (1.0 + x));
}

// NFW concentration parameter (Duffy et al. 2008 relation)
double nfw_concentration(const double Mvir_Msun, const double z)
{
    // c = 7.85 * (M/2e12)^(-0.081) * (1+z)^(-0.71)
    return 7.85 * pow(Mvir_Msun / 2.0e12, -0.081) * pow(1.0 + z, -0.71);
}

// Beta profile: ρ(r) = ρ_0 / [1 + (r/r_c)²]^(3β/2)
// For β = 2/3: ρ(r) = ρ_0 / [1 + (r/r_c)²]
// Returns the normalization ρ_0 given total mass M_CGM within R_vir
double beta_rho_0(const double M_CGM, const double Rvir, const double r_c, const double beta)
{
    // For general β, the enclosed mass integral is:
    // M(<R) = 4π ρ_0 ∫_0^R r² / [1 + (r/r_c)²]^(3β/2) dr
    //
    // For β = 2/3 (common value), this simplifies to:
    // M(<R) = 4π ρ_0 r_c³ × [arctan(R/r_c) - (R/r_c)/(1 + (R/r_c)²)]
    // But we need the more general form...
    //
    // Use numerical approximation for general β:
    // For large R/r_c, M ≈ 4π ρ_0 r_c³ × (some function of β)

    const double x = Rvir / r_c;
    double mass_integral;

    if(fabs(beta - 2.0/3.0) < 0.01) {
        // β ≈ 2/3: use analytic form
        // M = 4π ρ_0 r_c³ × [arctan(x) - x/(1+x²)]
        mass_integral = atan(x) - x / (1.0 + x * x);
    } else {
        // General β: numerical integration using Simpson's rule
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
        // M = 4π ρ_0 × integral, so mass_integral = integral / r_c³
        mass_integral = integral / (r_c * r_c * r_c);
    }

    if(mass_integral <= 0.0) return 0.0;
    return M_CGM / (4.0 * M_PI * r_c * r_c * r_c * mass_integral);
}

// Beta-profile density at radius r
double beta_density(const double r, const double rho_0, const double r_c, const double beta)
{
    const double y = r / r_c;
    return rho_0 / pow(1.0 + y * y, 1.5 * beta);
}

// ============================================================================
// Enclosed Mass Functions for each profile
// ============================================================================

// NFW enclosed mass: M(<r) = M_total × [ln(1+x) - x/(1+x)] / [ln(1+c) - c/(1+c)]
double nfw_enclosed_mass(const double r, const double M_total, const double Rvir, const double c_NFW)
{
    const double r_s = Rvir / c_NFW;
    const double x = r / r_s;

    // M(<r) / M_total = f(x) / f(c)
    const double f_x = log(1.0 + x) - x / (1.0 + x);
    const double f_c = log(1.0 + c_NFW) - c_NFW / (1.0 + c_NFW);

    if(f_c <= 0.0) return M_total;
    return M_total * f_x / f_c;
}

// Beta-profile enclosed mass for β = 2/3
// M(<r) = M_total × [arctan(x) - x/(1+x²)] / [arctan(X) - X/(1+X²)]
double beta_enclosed_mass(const double r, const double M_total, const double Rvir, const double r_c)
{
    const double x = r / r_c;
    const double X = Rvir / r_c;

    const double f_x = atan(x) - x / (1.0 + x * x);
    const double f_X = atan(X) - X / (1.0 + X * X);

    if(f_X <= 0.0) return M_total;
    return M_total * f_x / f_X;
}

// Unified enclosed mass function
double cgm_enclosed_mass(const double r, const double M_total, const double Rvir,
                                 const double Mvir_Msun, const double z, const int profile_type)
{
    if(r >= Rvir) return M_total;
    if(r <= 0.0) return 0.0;

    if(profile_type == 0) {
        // Uniform density: M(<r) = M_total × (r/Rvir)³
        const double ratio = r / Rvir;
        return M_total * ratio * ratio * ratio;
    } else if(profile_type == 1) {
        // NFW profile
        const double c_NFW = nfw_concentration(Mvir_Msun, z);
        return nfw_enclosed_mass(r, M_total, Rvir, c_NFW);
    } else if(profile_type == 2) {
        // Beta profile (β = 2/3, r_c = 0.1 Rvir)
        const double r_c = 0.1 * Rvir;
        return beta_enclosed_mass(r, M_total, Rvir, r_c);
    } else {
        // Default to uniform
        const double ratio = r / Rvir;
        return M_total * ratio * ratio * ratio;
    }
}

// Calculate CGM gas density at radius r given the profile choice
// Returns density in CGS units (g/cm³)
double cgm_density_at_radius(const double r_cgs, const double CGMgas_cgs, const double Rvir_cgs,
                                    const double Mvir_Msun, const double z, const int profile_type)
{
    if(profile_type == 0) {
        // Uniform density
        const double volume_cgs = (4.0 * M_PI / 3.0) * Rvir_cgs * Rvir_cgs * Rvir_cgs;
        return CGMgas_cgs / volume_cgs;

    } else if(profile_type == 1) {
        // NFW profile
        const double c_NFW = nfw_concentration(Mvir_Msun, z);
        const double r_s_cgs = Rvir_cgs / c_NFW;
        const double rho_s = nfw_rho_s(CGMgas_cgs, Rvir_cgs, c_NFW);
        return nfw_density(r_cgs, rho_s, r_s_cgs);

    } else if(profile_type == 2) {
        // Beta profile with β = 2/3 and r_c = 0.1 R_vir
        const double beta = 2.0 / 3.0;
        const double r_c_cgs = 0.1 * Rvir_cgs;
        const double rho_0 = beta_rho_0(CGMgas_cgs, Rvir_cgs, r_c_cgs, beta);
        return beta_density(r_cgs, rho_0, r_c_cgs, beta);

    } else {
        // Default to uniform if unknown profile type
        const double volume_cgs = (4.0 * M_PI / 3.0) * Rvir_cgs * Rvir_cgs * Rvir_cgs;
        return CGMgas_cgs / volume_cgs;
    }
}

// Iteratively solve for cooling radius r_cool where t_cool(r) = t_ff(r)
// Returns r_cool in CGS units
// Uses the correct enclosed mass for each density profile
double solve_for_rcool(const double CGMgas_cgs, const double Rvir_cgs, const double Mvir_cgs,
                              const double Mvir_Msun, const double temp, const double lambda,
                              const double z, const int profile_type,
                              __attribute__((unused)) const struct params *run_params)
{
    const double G_cgs = 6.674e-8;  // cm³ g⁻¹ s⁻²
    const double mu = 0.59;

    // ========================================================================
    // UNIFORM / BETA: Use isothermal r_cool formula (like hot-regime)
    // ========================================================================
    // For uniform density, t_cool and t_ff are both roughly constant with radius,
    // so the iterative solver doesn't converge meaningfully.
    // For beta profile (β=2/3), the profile is too flat and has similar issues.
    // Instead, use the isothermal approach: assume ρ(r) ∝ 1/r² for r_cool,
    // which gives r_cool = sqrt(ρ0 / ρ_cool) where ρ_cool is the critical density.
    if(profile_type == 0 || profile_type == 2) {
        // t_ff at R_vir: t_ff = sqrt(2 R³ / (G M))
        const double t_ff_Rvir = sqrt(2.0 * Rvir_cgs * Rvir_cgs * Rvir_cgs / (G_cgs * Mvir_cgs));

        // Critical density where t_cool = t_ff
        // ρ_cool = (3/2) μ m_p k T / (Λ t_ff)
        const double rho_cool = (1.5 * mu * PROTONMASS * BOLTZMANN * temp) / (lambda * t_ff_Rvir);

        // Isothermal profile normalization: ρ0 = M / (4π R)
        const double rho0 = CGMgas_cgs / (4.0 * M_PI * Rvir_cgs);

        // r_cool from isothermal: ρ(r_cool) = ρ0/r_cool² = ρ_cool
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
        const double g_accel = (M_enclosed > 0.0) ? G_cgs * M_enclosed / (r_cool * r_cool) : 0.0;

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

    if(galaxies[gal].HotGas > 0.0 && galaxies[gal].Vvir > 0.0) {
        const double tcool = galaxies[gal].Rvir / galaxies[gal].Vvir;
        const double temp = 35.9 * galaxies[gal].Vvir * galaxies[gal].Vvir;         // in Kelvin

        double logZ = -10.0;
        if(galaxies[gal].MetalsHotGas > 0) {
            logZ = log10(galaxies[gal].MetalsHotGas / galaxies[gal].HotGas);
        }

        double lambda = get_metaldependent_cooling_rate(log10(temp), logZ);

        // BUG FIX: Check lambda > 0 to avoid division by zero
        if(lambda <= 0.0) {
            return 0.0;  // No cooling if cooling function is zero/negative
        }

        double x = PROTONMASS * BOLTZMANN * temp / lambda;        // now this has units sec g/cm^3
        x /= (run_params->UnitDensity_in_cgs * run_params->UnitTime_in_s);         // now in internal units
        const double rho_rcool = x / tcool * 0.885;  // 0.885 = 3/2 * mu, mu=0.59 for a fully ionized gas

        // BUG FIX: Check rho_rcool > 0 to avoid sqrt of negative or division by zero
        if(rho_rcool <= 0.0) {
            return 0.0;
        }

        // an isothermal density profile for the hot gas is assumed here
        const double rho0 = galaxies[gal].HotGas / (4 * M_PI * galaxies[gal].Rvir);
        const double rcool = sqrt(rho0 / rho_rcool);

        galaxies[gal].RcoolToRvir = rcool / galaxies[gal].Rvir;

        coolingGas = 0.0;
        
        if(run_params->CGMrecipeOn == 0) {
            // Original behavior: SAGE C16 cooling recipe
            if(rcool > galaxies[gal].Rvir) {
                // "cold accretion" regime
                coolingGas = galaxies[gal].HotGas / (galaxies[gal].Rvir / galaxies[gal].Vvir) * dt;
            } else {
                // "hot halo cooling" regime
                coolingGas = (galaxies[gal].HotGas / galaxies[gal].Rvir) * (rcool / (2.0 * tcool)) * dt;
            }
        } else {
            // CGMrecipeOn == 1: D&B06 cold streams for hot-regime halos
            // All halos here are in the hot regime (have virial shocks)
            const double z = run_params->ZZ[galaxies[gal].SnapNum];
            
            // Calculate mass ratio for penetration factor
            const double Mvir_physical = galaxies[gal].Mvir * 1.0e10 / run_params->Hubble_h;
            const double Mshock = 6.0e11;  // Msun (D&B06 shock heating threshold)
            const double mass_ratio = Mvir_physical / Mshock;
            
            // D&B06 equations 39-41: Stream penetration factor
            // The characteristic mass for streams is M_stream ~ M_shock / (fM_*)
            // where fM_* is the universal baryon fraction that collapses into stars
            // At high-z, M_stream > M_shock, allowing cold streams in massive halos
            // At low-z, M_stream < M_shock, cold streams only in halos below shock threshold
            
            // Simplified prescription: cold stream fraction depends on M/Mshock and redshift
            // f_stream decreases with mass: halos much larger than Mshock have fewer cold streams
            // f_stream increases with redshift: high-z universe has more cold streams
            
            // Mass suppression: (M/Mshock)^(-4/3) from D&B06 eq 39
            // double f_stream = pow(mass_ratio, -4.0/3.0);
            
            // Redshift enhancement: cold streams more prominent at high z
            // Use smooth scaling that enhances at high-z, suppresses at low-z
            const double z_factor = (1.0 + z) / (1.0 + 1.0);  // Normalized to z=1
            // f_stream *= z_factor;
            // D&B06 eq 41: critical redshift where fM* = Mshock
            // Below this redshift, no cold streams in M > Mshock haloes
            const double z_crit = 1.5;  // D&B06 estimate ~1-2
            double f_stream;

            if(z < z_crit && mass_ratio > 1.0) {
                f_stream = 0.0;  // Hard cutoff per D&B06
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
    return coolingGas;
}


double cooling_recipe_cgm(const int gal, const double dt, struct GALAXY *galaxies,
                         const struct params *run_params)
{
    long precipitation_debug_counter = 0;
    precipitation_debug_counter++;

    double coolingGas = 0.0;

    // ========================================================================
    // EARLY EXIT CONDITIONS
    // ========================================================================
    if(galaxies[gal].CGMgas <= 0.0 || galaxies[gal].Vvir <= 0.0 || galaxies[gal].Rvir <= 0.0) {
        if(precipitation_debug_counter % 50000 == 0) {
            printf("DEBUG PRECIP [%ld]: Early exit - CGMgas=%.2e, Vvir=%.2f, Rvir=%.2e\n",
                   precipitation_debug_counter, galaxies[gal].CGMgas,
                   galaxies[gal].Vvir, galaxies[gal].Rvir);
        }
        return 0.0;
    }

    // ========================================================================
    // STEP 1: CALCULATE COOLING TIME (CGS UNITS) WITH DENSITY PROFILE
    // ========================================================================

    // Virial temperature
    const double temp = 35.9 * galaxies[gal].Vvir * galaxies[gal].Vvir; // Kelvin

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
    const double Mvir_Msun = galaxies[gal].Mvir * 1e10 / run_params->Hubble_h; // Msun
    const double z = run_params->ZZ[galaxies[gal].SnapNum];

    // Get density profile type (0: uniform, 1: NFW, 2: beta)
    // IMPORTANT: Density profile physics only applies to CGM-regime haloes (Regime == 0)
    // Hot-regime haloes always use uniform density for simple CGM depletion
    const int profile_type = (galaxies[gal].Regime == 0) ? run_params->CGMDensityProfile : 0;

    // ========================================================================
    // STEP 1b: SOLVE FOR COOLING RADIUS (consistent with hot regime approach)
    // ========================================================================
    // Find r_cool where t_cool(r_cool) = t_ff(r_cool)
    // This is done iteratively for all profile types

    const double r_cool_cgs = solve_for_rcool(CGMgas_cgs, Rvir_cgs, Mvir_cgs, Mvir_Msun,
                                               temp, lambda, z, profile_type, run_params);

    // Get density at the cooling radius
    const double mass_density_cgs = cgm_density_at_radius(r_cool_cgs, CGMgas_cgs, Rvir_cgs,
                                                           Mvir_Msun, z, profile_type);

    if(mass_density_cgs <= 0.0) {
        return 0.0;
    }

    // Store r_cool / R_vir for diagnostics
    galaxies[gal].RcoolToRvir = r_cool_cgs / Rvir_cgs;

    // Convert r_cool to code units
    const double r_cool = r_cool_cgs / (CM_PER_MPC / run_params->Hubble_h);

    // Cooling time at r_cool: tcool = (3/2) * μ * m_p * k * T / (ρ * Λ)
    const double mu = 0.59;
    const double tcool_cgs = (1.5 * mu * PROTONMASS * BOLTZMANN * temp) / (mass_density_cgs * lambda);
    const double tcool = tcool_cgs / run_params->UnitTime_in_s; // code units

    // ========================================================================
    // STEP 2: CALCULATE FREE-FALL TIME AT r_cool
    // ========================================================================

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
        galaxies[gal].tcool = tcool;
        galaxies[gal].tff = -1.0;
        galaxies[gal].tcool_over_tff = -1.0;
        galaxies[gal].tdeplete = -1.0;
        galaxies[gal].RcoolToRvir = -1.0;
        return 0.0;
    }
    const double tff = sqrt(2.0 * r_cool / g_accel); // code units

    // ========================================================================
    // STEP 2b: CHARACTERISTIC RADIUS FOR PRECIPITATION CRITERION (0.1 R_vir)
    // ========================================================================
    // Evaluating t_cool/t_ff at r_cool is circular for the NFW solver, which
    // converges to the radius where t_cool = t_ff (ratio ≈ 1 by construction),
    // making all haloes appear thermally unstable regardless of mass.
    // Instead, evaluate at 0.1 R_vir: a fixed CGM scale where the ratio
    // correctly reflects halo mass (massive → lower NFW density → higher
    // t_cool/t_ff → thermally stable). McCourt et al. 2012 framework.
    const double G_cgs = 6.674e-8;
    const double r_char_cgs = 0.1 * Rvir_cgs;
    const double rho_char = cgm_density_at_radius(r_char_cgs, CGMgas_cgs, Rvir_cgs,
                                                   Mvir_Msun, z, profile_type);

    // Default to r_cool values if 0.1 Rvir evaluation fails
    double tcool_char = tcool;
    double tff_char = tff;
    double tcool_over_tff_char = tcool / tff;

    if(rho_char > 0.0) {
        const double tcool_char_cgs = (1.5 * mu * PROTONMASS * BOLTZMANN * temp) / (rho_char * lambda);
        const double M_enc_char = cgm_enclosed_mass(r_char_cgs, Mvir_cgs, Rvir_cgs,
                                                     Mvir_Msun, z, profile_type);
        if(M_enc_char > 0.0) {
            const double g_char_cgs = G_cgs * M_enc_char / (r_char_cgs * r_char_cgs);
            if(g_char_cgs > 0.0) {
                const double tff_char_cgs = sqrt(2.0 * r_char_cgs / g_char_cgs);
                tcool_char = tcool_char_cgs / run_params->UnitTime_in_s;
                tff_char = tff_char_cgs / run_params->UnitTime_in_s;
                tcool_over_tff_char = tcool_char_cgs / tff_char_cgs;
            }
        }
    }

    // Store characteristic-radius values for diagnostics/plotting
    galaxies[gal].tcool = tcool_char;
    galaxies[gal].tff = tff_char;
    galaxies[gal].tcool_over_tff = tcool_over_tff_char;

    // ========================================================================
    // STEP 3: PRECIPITATION CRITERION
    // ========================================================================

    const double precipitation_threshold = 10;  // default=10, McCourt et al. 2012
    const double transition_width = 2.0;  // Smooth transition over factor ~2

    double precipitation_fraction = 0.0;

    if(tcool_over_tff_char < precipitation_threshold) {
        // UNSTABLE: Precipitation cooling
        double instability_factor = precipitation_threshold / tcool_over_tff_char;
        instability_factor = fmin(instability_factor, 3.0);
        precipitation_fraction = tanh(instability_factor / 2.0);

    } else if(tcool_over_tff_char < precipitation_threshold + transition_width) {
        // TRANSITION: Smoothly reduce precipitation_fraction to zero
        const double x = (tcool_over_tff_char - precipitation_threshold) / transition_width;
        precipitation_fraction = 0.5 * (1.0 - tanh(x));

    } else {
        if(tcool_char > 0) {
        // Cooling rate: dM/dt = M_CGM / t_cool
        coolingGas = galaxies[gal].CGMgas / tcool_char * dt;

        // Safety check
        if(coolingGas > galaxies[gal].CGMgas) {
            coolingGas = galaxies[gal].CGMgas;
        }
    } else {
        coolingGas = 0.0;
        }

    }

    // ========================================================================
    // STEP 4: CALCULATE PRECIPITATION RATE
    // ========================================================================

    if(precipitation_fraction > 0.0) {
        // Gas precipitates on the free-fall timescale when thermally unstable
        // This is the key physical insight: dM/dt = f_precip * M_CGM / t_ff
        const double precip_rate = precipitation_fraction * galaxies[gal].CGMgas / tff_char;
        coolingGas = precip_rate * dt;
        
        // Physical limits
        if(coolingGas > galaxies[gal].CGMgas) {
            coolingGas = galaxies[gal].CGMgas;
        }
        if(coolingGas < 0.0) {
            coolingGas = 0.0;
        }
    }

    // double x = PROTONMASS * BOLTZMANN * temp / lambda;        // now this has units sec g/cm^3
    // x /= (run_params->UnitDensity_in_cgs * run_params->UnitTime_in_s);         // now in internal units

    // if(run_params->AGNrecipeOn > 0 && coolingGas > 0.0) {
	// 		coolingGas = do_AGN_heating_cgm(coolingGas, gal, dt, x, r_cool, galaxies, run_params);
    // }

    // ========================================================================
    // STEP 5: TRACK COOLING ENERGY
    // ========================================================================
    
    // Energy associated with cooling (for feedback balance tracking)
    if(coolingGas > 0.0) {
        // Specific energy ~ 0.5 * Vvir^2 (thermal + kinetic)
        galaxies[gal].Cooling += 0.5 * coolingGas * galaxies[gal].Vvir * galaxies[gal].Vvir;
    }

    // ========================================================================
    // STEP 6: CALCULATE DEPLETION TIMESCALE (DIAGNOSTIC)
    // ========================================================================

    // Depletion timescale (only meaningful for CGM-regime haloes)
    if(galaxies[gal].Regime == 0) {
        if(precipitation_fraction > 1e-6 && isfinite(tff_char)) {
            const double depletion_time = tff_char / precipitation_fraction;
            galaxies[gal].tdeplete = isfinite(depletion_time) ? (float)depletion_time : -1.0f;
        } else {
            galaxies[gal].tdeplete = -1.0f;
        }
    } else {
        // Hot-regime haloes: reset diagnostic fields (density profile physics doesn't apply)
        galaxies[gal].tcool = -1.0f;
        galaxies[gal].tff = -1.0f;
        galaxies[gal].tcool_over_tff = -1.0f;
        galaxies[gal].tdeplete = -1.0f;
    }

    // Sanity check
    XASSERT(coolingGas >= 0.0, -1, "Error: Cooling gas mass = %g should be >= 0.0", coolingGas);
    XASSERT(coolingGas <= galaxies[gal].CGMgas + 1e-12, -1,
            "Error: Cooling gas = %g exceeds CGM gas = %g", coolingGas, galaxies[gal].CGMgas);

    return coolingGas;
}


double cooling_recipe_regime_aware(const int gal, const double dt, struct GALAXY *galaxies, const struct params *run_params)
{
    double cgm_cooling = 0.0;
    double hot_cooling = 0.0;

    if(galaxies[gal].Regime == 0) {
        // CGM REGIME: CGM physics dominates

        // Primary: Precipitation cooling from CGMgas
        if(galaxies[gal].CGMgas > 0.0) {
            cgm_cooling = cooling_recipe_cgm(gal, dt, galaxies, run_params);
        }


    } else {
        // HOT REGIME: Traditional physics dominates

        // Primary: Traditional cooling from HotGas
        if(galaxies[gal].HotGas > 0.0) {
            hot_cooling = cooling_recipe_hot(gal, dt, galaxies, run_params);
        }

        // Secondary: Precipitation cooling from CGMgas (gradually depletes)
        if(galaxies[gal].CGMgas > 0.0) {
            cgm_cooling = cooling_recipe_cgm(gal, dt, galaxies, run_params);
        }
    }

    // Now apply the cooling directly to preserve the physics-based split
    // Apply CGM cooling
    if(cgm_cooling > 0.0) {
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
    XASSERT(total_cooling >= 0.0, -1,
            "Error: Cooling gas mass = %g should be >= 0.0", total_cooling);
    return total_cooling;
}



double do_AGN_heating(double coolingGas, const int centralgal, const double dt, const double x, const double rcool, struct GALAXY *galaxies, const struct params *run_params)
{
    double AGNrate, EDDrate, AGNaccreted, AGNcoeff, AGNheating, metallicity;

	// first update the cooling rate based on the past AGN heating
	if(galaxies[centralgal].r_heat < rcool) {
		coolingGas = (1.0 - galaxies[centralgal].r_heat / rcool) * coolingGas;
    } else {
		coolingGas = 0.0;
    }

	XASSERT(coolingGas >= 0.0, -1,
            "Error: Cooling gas mass = %g should be >= 0.0", coolingGas);

	// now calculate the new heating rate
    if(galaxies[centralgal].HotGas > 0.0) {
        if(run_params->AGNrecipeOn == 2) {
            // Bondi-Hoyle accretion recipe
            AGNrate = (2.5 * M_PI * run_params->G) * (0.375 * 0.6 * x) * galaxies[centralgal].BlackHoleMass * run_params->RadioModeEfficiency;
        } else if(run_params->AGNrecipeOn == 3) {
            // Cold cloud accretion: trigger: rBH > 1.0e-4 Rsonic, and accretion rate = 0.01% cooling rate
            if(galaxies[centralgal].BlackHoleMass > 0.0001 * galaxies[centralgal].Mvir * CUBE(rcool/galaxies[centralgal].Rvir)) {
                AGNrate = 0.0001 * coolingGas / dt;
            } else {
                AGNrate = 0.0;
            }
        } else {
            // empirical (standard) accretion recipe
            if(galaxies[centralgal].Mvir > 0.0) {
                AGNrate = run_params->RadioModeEfficiency / (run_params->UnitMass_in_g / run_params->UnitTime_in_s * SEC_PER_YEAR / SOLAR_MASS)
                    * (galaxies[centralgal].BlackHoleMass / 0.01) * CUBE(galaxies[centralgal].Vvir / 200.0)
                    * ((galaxies[centralgal].HotGas / galaxies[centralgal].Mvir) / 0.1);
            } else {
                AGNrate = run_params->RadioModeEfficiency / (run_params->UnitMass_in_g / run_params->UnitTime_in_s * SEC_PER_YEAR / SOLAR_MASS)
                    * (galaxies[centralgal].BlackHoleMass / 0.01) * CUBE(galaxies[centralgal].Vvir / 200.0);
            }
        }

        // Eddington rate
        EDDrate = (1.3e38 * galaxies[centralgal].BlackHoleMass * 1e10 / run_params->Hubble_h) / (run_params->UnitEnergy_in_cgs / run_params->UnitTime_in_s) / (0.1 * 9e10);

        // accretion onto BH is always limited by the Eddington rate
        if(AGNrate > EDDrate) {
            AGNrate = EDDrate;
        }

        // accreted mass onto black hole
        AGNaccreted = AGNrate * dt;

        // cannot accrete more mass than is available!
        if(AGNaccreted > galaxies[centralgal].HotGas) {
            AGNaccreted = galaxies[centralgal].HotGas;
        }

        // coefficient to heat the cooling gas back to the virial temperature of the halo
        // 1.34e5 = sqrt(2*eta*c^2), eta=0.1 (standard efficiency) and c in km/s
        // BUG FIX: Check Vvir > 0 to avoid division by zero
        if(galaxies[centralgal].Vvir <= 0.0) {
            AGNcoeff = 0.0;
            AGNheating = 0.0;
        } else {
            AGNcoeff = (1.34e5 / galaxies[centralgal].Vvir) * (1.34e5 / galaxies[centralgal].Vvir);

            // cooling mass that can be suppresed from AGN heating
            AGNheating = AGNcoeff * AGNaccreted;

            /// the above is the maximal heating rate. we now limit it to the current cooling rate
            if(AGNheating > coolingGas && AGNcoeff > 0.0) {
                AGNaccreted = coolingGas / AGNcoeff;
                AGNheating = coolingGas;
            }
        }

        // accreted mass onto black hole
        metallicity = get_metallicity(galaxies[centralgal].HotGas, galaxies[centralgal].MetalsHotGas);
        galaxies[centralgal].BlackHoleMass += AGNaccreted;
        galaxies[centralgal].HotGas -= AGNaccreted;
        galaxies[centralgal].MetalsHotGas -= metallicity * AGNaccreted;

        // update the heating radius as needed
        if(galaxies[centralgal].r_heat < rcool && coolingGas > 0.0) {
            double r_heat_new = (AGNheating / coolingGas) * rcool;
            if(r_heat_new > galaxies[centralgal].r_heat) {
                galaxies[centralgal].r_heat = r_heat_new;
            }
        }

        if (AGNheating > 0.0) {
            galaxies[centralgal].Heating += 0.5 * AGNheating * galaxies[centralgal].Vvir * galaxies[centralgal].Vvir;
        }
    }

    return coolingGas;
}

double do_AGN_heating_cgm(double coolingGas, const int centralgal, const double dt, const double x, const double rcool, 
                         struct GALAXY *galaxies, const struct params *run_params)
{
    double AGNrate, EDDrate, AGNaccreted, AGNcoeff, AGNheating, metallicity;

	// first update the cooling rate based on the past AGN heating
	if(galaxies[centralgal].r_heat < rcool) {
		coolingGas = (1.0 - galaxies[centralgal].r_heat / rcool) * coolingGas;
    } else {
		coolingGas = 0.0;
    }

	XASSERT(coolingGas >= 0.0, -1,
            "Error: Cooling gas mass = %g should be >= 0.0", coolingGas);

	// now calculate the new heating rate
    if(galaxies[centralgal].CGMgas > 0.0) {
        if(run_params->AGNrecipeOn == 2) {
            // Bondi-Hoyle accretion recipe
            AGNrate = (2.5 * M_PI * run_params->G) * (0.375 * 0.6 * x) * galaxies[centralgal].BlackHoleMass * run_params->RadioModeEfficiency;
        } else if(run_params->AGNrecipeOn == 3) {
            // Cold cloud accretion: trigger: rBH > 1.0e-4 Rsonic, and accretion rate = 0.01% cooling rate
            if(galaxies[centralgal].BlackHoleMass > 0.0001 * galaxies[centralgal].Mvir * CUBE(rcool/galaxies[centralgal].Rvir)) {
                AGNrate = 0.0001 * coolingGas / dt;
            } else {
                AGNrate = 0.0;
            }
        } else {
            // empirical (standard) accretion recipe
            if(galaxies[centralgal].Mvir > 0.0) {
                AGNrate = run_params->RadioModeEfficiency / (run_params->UnitMass_in_g / run_params->UnitTime_in_s * SEC_PER_YEAR / SOLAR_MASS)
                    * (galaxies[centralgal].BlackHoleMass / 0.01) * CUBE(galaxies[centralgal].Vvir / 200.0)
                    * ((galaxies[centralgal].CGMgas / galaxies[centralgal].Mvir) / 0.1);
            } else {
                AGNrate = run_params->RadioModeEfficiency / (run_params->UnitMass_in_g / run_params->UnitTime_in_s * SEC_PER_YEAR / SOLAR_MASS)
                    * (galaxies[centralgal].BlackHoleMass / 0.01) * CUBE(galaxies[centralgal].Vvir / 200.0);
            }
        }

        // Eddington rate
        EDDrate = (1.3e38 * galaxies[centralgal].BlackHoleMass * 1e10 / run_params->Hubble_h) / (run_params->UnitEnergy_in_cgs / run_params->UnitTime_in_s) / (0.1 * 9e10);

        // accretion onto BH is always limited by the Eddington rate
        if(AGNrate > EDDrate) {
            AGNrate = EDDrate;
        }

        // accreted mass onto black hole
        AGNaccreted = AGNrate * dt;

        // cannot accrete more mass than is available!
        if(AGNaccreted > galaxies[centralgal].CGMgas) {
            AGNaccreted = galaxies[centralgal].CGMgas;
        }

        // coefficient to heat the cooling gas back to the virial temperature of the halo
        // 1.34e5 = sqrt(2*eta*c^2), eta=0.1 (standard efficiency) and c in km/s
        // BUG FIX: Check Vvir > 0 to avoid division by zero
        if(galaxies[centralgal].Vvir <= 0.0) {
            AGNcoeff = 0.0;
            AGNheating = 0.0;
        } else {
            AGNcoeff = (1.34e5 / galaxies[centralgal].Vvir) * (1.34e5 / galaxies[centralgal].Vvir);

            // cooling mass that can be suppresed from AGN heating
            AGNheating = AGNcoeff * AGNaccreted;

            /// the above is the maximal heating rate. we now limit it to the current cooling rate
            if(AGNheating > coolingGas && AGNcoeff > 0.0) {
                AGNaccreted = coolingGas / AGNcoeff;
                AGNheating = coolingGas;
            }
        }

        // accreted mass onto black hole
        metallicity = get_metallicity(galaxies[centralgal].CGMgas, galaxies[centralgal].MetalsCGMgas);
        galaxies[centralgal].BlackHoleMass += AGNaccreted;
        galaxies[centralgal].CGMgas -= AGNaccreted;
        galaxies[centralgal].MetalsCGMgas -= metallicity * AGNaccreted;

        // update the heating radius as needed
        if(galaxies[centralgal].r_heat < rcool && coolingGas > 0.0) {
            double r_heat_new = (AGNheating / coolingGas) * rcool;
            if(r_heat_new > galaxies[centralgal].r_heat) {
                galaxies[centralgal].r_heat = r_heat_new;
            }
        }

        if (AGNheating > 0.0) {
            galaxies[centralgal].Heating += 0.5 * AGNheating * galaxies[centralgal].Vvir * galaxies[centralgal].Vvir;
        }
    }
    return coolingGas;
}

void cool_gas_onto_galaxy(const int centralgal, const double coolingGas, struct GALAXY *galaxies)
{
    // add the fraction 1/STEPS of the total cooling gas to the cold disk
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

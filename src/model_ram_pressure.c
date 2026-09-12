/*
 * model_ram_pressure.c -- ram-pressure stripping of satellite ISM.
 *
 * Implements the Gunn & Gott (1972) criterion: cold disk gas at radius r is
 * stripped where the ram pressure of the host's ambient medium exceeds the
 * gravitational restoring force per unit area of the disk,
 *
 *     rho_host(R_orb) * v_sat^2  >  2*pi*G * Sigma_disk(r) * Sigma_gas(r).
 *
 * With the gas and stellar disks both exponential in the same scale radius
 * r_s (the assumption already used by the H2 machinery), the restoring side
 * scales as exp(-2r/r_s), so the stripping radius is analytic,
 *
 *     r_strip = (r_s/2) * ln( 2*pi*G * Sigma_gas(0) * Sigma_disk(0) / P_ram ),
 *
 * and the stripped mass fraction of the exponential gas disk is
 *
 *     f_strip = (1 + r_strip/r_s) * exp(-r_strip/r_s).
 *
 * The driver, ram_pressure_strip_satellite(), evaluates the host ambient
 * density at the satellite's orbital radius (CGM profile for CGM-regime
 * hosts, isothermal M_hot/(4*pi*Rvir*r^2) for hot-regime hosts -- the same
 * profiles the cooling recipes assume), removes ColdGas gradually as
 * M_strip = ColdGas * f_strip * (1 - exp(-dt/t_strip)) on the host stripping
 * timescale, and donates the gas and its metals to the central's hot/CGM
 * reservoir.  It runs once per snapshot from evolve_galaxies() when
 * RamPressureStrippingOn == 1 (the default; set 0 to disable).
 *
 * This channel strips the ISM (ColdGas) and is complementary to and
 * independent of PhysicalStrippingOn, which strips the satellite's hot/CGM
 * gas (starvation).
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#include <math.h>

#include "core_allvars.h"

#include "model_ram_pressure.h"
#include "model_cooling_heating.h"
#include "model_misc.h"

/* Gravitational constant in cgs [cm^3 g^-1 s^-2] (matches model_cooling_heating.c). */
static const double G_CGS = 6.674e-8;

/* GALAXY.Vel / Vvir are stored in km/s; the pressure evaluation is in cgs. */
static const double CM_PER_KM = 1.0e5;

/* Floor on the orbital radius as a fraction of the host Rvir: the isothermal
 * and NFW ambient profiles diverge at r -> 0, and the halo-centre positions
 * are not meaningful below this scale anyway.  Same floor solve_for_rcool()
 * uses for the cooling radius. */
static const double RPS_MIN_RADIUS_FRAC = 0.001;

/*
 * ram_pressure_stripped_fraction -- stripped mass fraction of an exponential
 * gas disk under ram pressure P_ram (Gunn & Gott 1972).
 *
 * Sigma_gas0_cgs and Sigma_disk0_cgs are the CENTRAL (r=0) surface densities
 * of the gas disk and of the total disk (gas + disk stars) in g/cm^2; both
 * profiles are assumed exponential with a shared scale radius.  P_ram_cgs is
 * in g cm^-1 s^-2.  Returns f_strip in [0, 1]; the same (1+x)exp(-x) form as
 * the f_ion outer-disk cut in the HI ionisation model.
 */
double ram_pressure_stripped_fraction(const double P_ram_cgs,
                                      const double Sigma_gas0_cgs,
                                      const double Sigma_disk0_cgs)
{
    if(P_ram_cgs <= 0.0 || Sigma_gas0_cgs <= 0.0 || Sigma_disk0_cgs <= 0.0) {
        return 0.0;
    }

    /* Restoring force per unit area at the disk centre; it falls off as
     * exp(-2r/r_s), so this is also its maximum. */
    const double restoring0 = 2.0 * M_PI * G_CGS * Sigma_disk0_cgs * Sigma_gas0_cgs;

    if(P_ram_cgs >= restoring0) {
        return 1.0;   /* r_strip <= 0: ram pressure wins everywhere */
    }

    const double x = 0.5 * log(restoring0 / P_ram_cgs);   /* r_strip / r_s */
    return (1.0 + x) * exp(-x);
}

/*
 * ram_pressure_strip_satellite -- remove ram-pressure-stripped ColdGas from
 * satellite `gal` and donate it to the central's hot/CGM reservoir.
 *
 * Called once per snapshot from evolve_galaxies(), outside the substep loop
 * (operator-split like the PhysicalStrippingOn == 2 hot-gas scheme).  `dt` is
 * the full snapshot interval and `t_strip` the host stripping timescale
 * (StrippingTimescaleFactor * t_dyn(host), shared with the hot-gas channel),
 * so the removed fraction f_strip * (1 - exp(-dt/t_strip)) is substep-count
 * independent.
 *
 * SAGE stores only (ColdGas, DiskScaleRadius): removal lowers Sigma_gas
 * everywhere at fixed r_s instead of truncating the disk at r_strip -- the
 * standard SAM compromise.  No extra phase bookkeeping is needed: H2/H1 are
 * re-partitioned from the reduced ColdGas at the next substep, and because
 * the outer disk is HI-dominated in that equilibrium partition, stripping
 * preferentially removes HI.
 */
void ram_pressure_strip_satellite(const int centralgal, const int gal,
                                  const double Zcurr, const double dt,
                                  const double t_strip,
                                  struct GALAXY *galaxies,
                                  const struct params *run_params)
{
    if(galaxies[gal].ColdGas <= 0.0 || galaxies[gal].DiskScaleRadius <= 0.0) {
        return;
    }
    if(galaxies[centralgal].Rvir <= 0.0 || galaxies[centralgal].Vvir <= 0.0) {
        return;
    }

    const double h = run_params->Hubble_h;
    const double a = 1.0 / (1.0 + Zcurr);
    const double Rvir = galaxies[centralgal].Rvir;   /* physical Mpc/h */

    /* Orbital radius: comoving position offset (minimum image across the
     * periodic box) converted to physical, clamped to
     * [RPS_MIN_RADIUS_FRAC, 1] * Rvir. */
    double dr2 = 0.0;
    for(int i = 0; i < 3; i++) {
        double dx = (double)galaxies[gal].Pos[i] - (double)galaxies[centralgal].Pos[i];
        if(run_params->BoxSize > 0.0) {
            if(dx > 0.5 * run_params->BoxSize) {
                dx -= run_params->BoxSize;
            } else if(dx < -0.5 * run_params->BoxSize) {
                dx += run_params->BoxSize;
            }
        }
        dr2 += dx * dx;
    }
    double r_orb = a * sqrt(dr2);
    if(r_orb > Rvir) r_orb = Rvir;
    if(r_orb < RPS_MIN_RADIUS_FRAC * Rvir) r_orb = RPS_MIN_RADIUS_FRAC * Rvir;

    /* Orbital velocity: peculiar-velocity difference vs the central; fall
     * back to the host Vvir when the difference is degenerate.  Orphans
     * (Type 2) always use the host Vvir: their stored velocity is frozen at
     * the snapshot the subhalo was lost, and its offset from the CURRENT
     * central velocity is not a meaningful orbital speed.  (Their frozen
     * position is still used above -- it marks where the subhalo disrupted,
     * and since the true orbit only decays further in, it underestimates
     * rho_host and therefore errs on the side of less stripping.) */
    double v_kms;
    if(galaxies[gal].Type == 2) {
        v_kms = galaxies[centralgal].Vvir;
    } else {
        double v2 = 0.0;
        for(int i = 0; i < 3; i++) {
            const double dv = (double)galaxies[gal].Vel[i] - (double)galaxies[centralgal].Vel[i];
            v2 += dv * dv;
        }
        v_kms = sqrt(v2);
        if(v_kms <= 0.0) {
            v_kms = galaxies[centralgal].Vvir;
        }
    }

    /* Host ambient density at the orbital radius [g/cm^3].  CGM-regime hosts
     * use the same profile family the CGM cooling recipe integrates over;
     * hot-regime (and legacy-mode) hosts use the isothermal profile the
     * hot-mode cooling recipe assumes: rho(r) = M_hot / (4*pi*Rvir*r^2). */
    const double r_cgs = r_orb * CM_PER_MPC / h;
    const double Rvir_cgs = Rvir * CM_PER_MPC / h;
    double rho_host;
    if(run_params->CGMrecipeOn == 1 && galaxies[centralgal].Regime == 0
       && galaxies[centralgal].CGMgas > 0.0) {
        
        const double CGMgas_cgs = galaxies[centralgal].CGMgas * 1e10 * SOLAR_MASS / h;
        
        // Carr et al. 2023 CGM density profile: rho(r) = rho0 * (r/r0)^(-1.4)
        const double alpha = 1.4;
        const double r0_cgs = 0.1 * Rvir_cgs;
        const double ratio = 10.0; // Rvir / r0 = 10
        
        // Volumetric integral factor for mass to find rho0
        const double I_M = (pow(ratio, 3.0 - alpha) - 1.0) / (3.0 - alpha);
        const double rho0 = CGMgas_cgs / (4.0 * M_PI * r0_cgs * r0_cgs * r0_cgs * I_M);
        
        // Evaluate local density at the satellite's orbital radius.
        // Bounded by r0 to prevent non-physical singularities if the orbit decays deeply.
        double r_eval = (r_cgs < r0_cgs) ? r0_cgs : r_cgs; 
        
        rho_host = rho0 * pow(r_eval / r0_cgs, -alpha);
        
    } else if(galaxies[centralgal].HotGas > 0.0) {
        const double HotGas_cgs = galaxies[centralgal].HotGas * 1e10 * SOLAR_MASS / h;
        rho_host = HotGas_cgs / (4.0 * M_PI * Rvir_cgs * r_cgs * r_cgs);
    } else {
        return;   /* no ambient medium to strip against */
    }
    
    if(!(rho_host > 0.0)) {   /* also rejects NaN */
        return;
    }

    /* Ram pressure, with the order-unity RamPressureEpsilon prefactor
     * absorbing the geometry uncertainty (face-on vs edge-on infall). */
    const double v_cgs = v_kms * CM_PER_KM;
    const double P_ram = run_params->RamPressureEpsilon * rho_host * v_cgs * v_cgs;

    /* Satellite central surface densities: exponential gas and stellar disks
     * sharing DiskScaleRadius, disk stars only (no bulge) -- the same
     * assumptions as calculate_molecular_fraction_radial_integration(). */
    const double rs_cgs = galaxies[gal].DiskScaleRadius * CM_PER_MPC / h;
    const double disk_area0 = 2.0 * M_PI * rs_cgs * rs_cgs;
    const double Mcold_cgs = galaxies[gal].ColdGas * 1e10 * SOLAR_MASS / h;
    double Mdiskstar = galaxies[gal].StellarMass - galaxies[gal].BulgeMass;
    if(Mdiskstar < 0.0) Mdiskstar = 0.0;
    const double Mdiskstar_cgs = Mdiskstar * 1e10 * SOLAR_MASS / h;

    const double Sigma_gas0 = Mcold_cgs / disk_area0;
    const double Sigma_disk0 = (Mcold_cgs + Mdiskstar_cgs) / disk_area0;

    const double f_strip = ram_pressure_stripped_fraction(P_ram, Sigma_gas0, Sigma_disk0);
    if(f_strip <= 0.0) {
        return;
    }

    /* Gradual removal over the snapshot, same analytic 1-exp(-dt/t_strip)
     * cadence as the hot-gas stripping scheme. */
    const double time_frac = (t_strip > 0.0) ? (1.0 - exp(-dt / t_strip)) : 1.0;
    double strip_mass = galaxies[gal].ColdGas * f_strip * time_frac;
    if(strip_mass <= 0.0) {
        return;
    }
    if(strip_mass > galaxies[gal].ColdGas) strip_mass = galaxies[gal].ColdGas;

    const double metallicity = get_metallicity(galaxies[gal].ColdGas, galaxies[gal].MetalsColdGas);
    double strip_metals = strip_mass * metallicity;
    if(strip_metals > galaxies[gal].MetalsColdGas) strip_metals = galaxies[gal].MetalsColdGas;

    galaxies[gal].ColdGas       -= strip_mass;
    galaxies[gal].MetalsColdGas -= strip_metals;

    add_gas_to_hot_reservoir(&galaxies[centralgal], run_params, strip_mass, strip_metals);
}

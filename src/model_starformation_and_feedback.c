/*
 * model_starformation_and_feedback.c -- Star formation and supernova feedback.
 *
 * Implements multiple star formation prescriptions (SFprescription 0-7),
 * FIRE stellar feedback, and the feedback-free burst (FFB) mode. Updates cold
 * gas, stellar mass, metals, and reheated/ejected gas reservoirs each substep.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "core_allvars.h"

#include "model_starformation_and_feedback.h"
#include "model_misc.h"
#include "model_disk_instability.h"

/* -------------------------------------------------------------------------
 * File-scope empirical constants (lifted per STYLE_C.md SS8).
 * -------------------------------------------------------------------------*/

/* -------------------------------------------------------------------------
 * Silent-clamp diagnostics.
 *
 * The physics deliberately clamps in a few places (H2 capped at the cold-gas
 * hydrogen budget, negative HI zeroed, reheated mass capped at the available
 * cold gas).  The clamps are part of the calibrated behaviour and must stay;
 * these counters only make them observable, so an upstream bug that floods a
 * clamp cannot hide.  Totals are printed by report_sf_clamp_counts() from
 * finalize_sage() in VERBOSE builds.  The post-depletion H2/H1 resync clamps
 * in update_from_star_formation()/update_from_feedback() fire routinely by
 * construction (H2 is computed before SF debits ColdGas) and are not counted.
 * -------------------------------------------------------------------------*/
static int64_t clamp_count_h2_cap = 0;         /* H2 fit exceeded ColdGas * HYDROGEN_MASS_FRAC */
static int64_t clamp_count_h1_negative = 0;    /* HI went negative after ionisation/H2 subtraction */
static int64_t clamp_count_reheat_coldgas = 0; /* reheated mass exceeded available ColdGas */

/* Print the silent-clamp totals accumulated over the run (VERBOSE builds
 * call this from finalize_sage()). Quiet when nothing fired -- the report
 * exists to flag anomalies, and all-zero is the expected healthy state. */
void report_sf_clamp_counts(void)
{
    if(clamp_count_h2_cap == 0 && clamp_count_h1_negative == 0 && clamp_count_reheat_coldgas == 0) {
        return;
    }
    printf("SF/feedback clamp totals: H2 capped to hydrogen budget = %" PRId64
           ", negative HI zeroed = %" PRId64
           ", reheated mass capped to ColdGas = %" PRId64 "\n",
           clamp_count_h2_cap, clamp_count_h1_negative, clamp_count_reheat_coldgas);
}

/* SF disk effective radius: reff = SF_DISK_RADIUS_FRAC * r_s, calibrated to
 * the Milky Way disk.  Used for both the dynamical time and disk area in all
 * SF prescriptions. */
static const double SF_DISK_RADIUS_FRAC = 3.0;


/* Kauffmann (1996) eq. 7 cold-gas surface density threshold coefficient.
 * cold_crit = KAUFFMANN96_SF_THRESHOLD * Vvir * reff in code units
 * (Vvir in km/s, reff in Mpc/h, cold_crit in 10^10 Msun/h). */
static const double KAUFFMANN96_SF_THRESHOLD = 0.19;

/* Somerville et al. (2025) eq. 2: critical gas surface density below which
 * cloud formation is inefficient.  30/(pi*G) where G = 4.302e-3 pc (km/s)^2 Msun^-1.
 * Evaluates to ~2217 Msun/pc^2. */
static const double SOMERVILLE25_SIGMA_CRIT = 30.0 / (M_PI * 4.302e-3);  /* Msun/pc^2 */

/* FIRE (Muratov et al. 2015) critical circular velocity separating the
 * two power-law slopes of the wind loading factor (their eq. 11, Table 1). */
static const double FIRE_V_CRIT_KMS = 60.0;  /* km/s */

/* Krumholz & Dekel (2011) eq. 22 characteristic halo mass for metal enrichment.
 * FracZleaveDisk ~ exp(-Mvir / KD11_METAL_HALO_MASS) in code units (10^10 Msun/h).
 * Same constant used in model_mergers.c. */
static const double KD11_METAL_HALO_MASS = 30.0;  /* 10^10 Msun/h */

/* Somerville et al. (2025) Equation 8: fraction of gas in dense clouds. */
static const double SOMERVILLE25_F_DENSE = 0.5;

/* Solar metallicity (Asplund et al. 2009): used to normalise Z' in K13 and KMT09.
 * Z' = Z_gas / Z_SOLAR_ASPLUND09.
 * Also the canonical definition in model_misc.c (calculate_H2_fraction_K13). */
static const double Z_SOLAR_ASPLUND09 = 0.014;

/* Critical neutral surface density for the HI ionisation cut [Msun/pc^2]
 * (Shark sigma_hi_crit): cold disk gas below this column is kept ionised by
 * the UV background and removed from the atomic (HI) remainder. */
static const double SIGMA_HI_CRIT = 0.5;

/*
 * Ionised-gas fraction of the cold disk.
 *
 * SAGE assigns all non-molecular cold hydrogen to HI, with no allowance for the
 * diffuse low-column outer disk that is kept ionised by the UV background and is
 * invisible in HI. Peer H2-based SAMs correct for this: Shark truncates the
 * neutral disk at a critical surface density (ionised_gas_fraction, sigma_hi_crit
 * ~ 0.5 Msun/pc^2), DarkSage caps the neutral fraction with the cosmic HI
 * photoionisation rate. We follow Shark: for an exponential gas disk with central
 * surface density Sigma0 = M_gas / (2 pi r_s^2), the gas beyond the radius where
 * Sigma(r) = SigmaHIcrit is ionised. With x = ln(Sigma0/Sigma_crit) the ionised
 * mass fraction is f_ion = (1 + x) * Sigma_crit / Sigma0. Returns 0 when the
 * correction is disabled or inapplicable, clamped to [0, 1].
 *
 * coldgas_code : ColdGas in code units (10^10 Msun/h)
 * rs_code      : atomic-disk scale radius in code units (Mpc/h). Callers pass
 *                GasDiskRadiusFactor * DiskScaleRadius: chi = 1 makes the atomic disk
 *                cospatial with the stellar/H2 disk (published behaviour), while the
 *                observed chi ~ 1.5-2 spreads the same HI over a larger area and so
 *                pushes more of it below the neutral threshold. H2 is untouched -- it is
 *                central and shielded, which is exactly why one radius should not set both.
 */
/*
 * Scale radius of the atomic disk: chi * r_s, where chi = GasDiskRadiusFactor.
 *
 * A zeroed struct params (the unit-test harnesses memset theirs) must behave like the
 * published chi = 1, and read_parameter_file() already rejects chi <= 0, so a non-positive
 * value here can only mean "never initialised" rather than a real configuration choice.
 */
static double atomic_disk_radius(const double rs_code, const struct params *run_params)
{
    const double chi = run_params->GasDiskRadiusFactor;
    return (chi > 0.0) ? chi * rs_code : rs_code;
}

static double ionized_gas_fraction(const double coldgas_code, const double rs_code,
                                   const double h, const double sigma_hi_crit)
{
    if(coldgas_code <= 0.0 || rs_code <= 0.0 || sigma_hi_crit <= 0.0 || h <= 0.0) {
        return 0.0;
    }
    const double mgas   = CODE_MASS_TO_MSUN(coldgas_code, h);                 /* Msun */
    const double rs_pc  = CODE_LENGTH_TO_PC(rs_code, h);                       /* pc */
    const double sigma0 = mgas / (2.0 * M_PI * rs_pc * rs_pc);       /* Msun/pc^2 */
    if(sigma0 <= sigma_hi_crit) {
        return 1.0;   /* entire disk below the neutral threshold -> fully ionised */
    }
    const double x     = log(sigma0 / sigma_hi_crit);
    double f_ion       = (1.0 + x) * sigma_hi_crit / sigma0;         /* = 1 - m(<r_thresh)/m */
    if(f_ion < 0.0) f_ion = 0.0;
    if(f_ion > 1.0) f_ion = 1.0;
    return f_ion;
}

/*
 * SFprescription == 0 -- Croton et al. (2006): stars form from cold gas
 * above the Kauffmann (1996) eq. 7 critical mass.  Does not touch H2gas.
 *
 * Returns the instantaneous SFR strdot [10^10 Msun/h / code time].
 */
static double sfr_croton06(const int p, const struct GALAXY *galaxies, const struct params *run_params)
{
    double reff, tdyn, strdot;

    strdot = 0.0;
    tdyn = 0.0;

    // we take the typical star forming region as 3.0*r_s using the Milky Way as a guide
    reff = SF_DISK_RADIUS_FRAC * galaxies[p].DiskScaleRadius;

    if(galaxies[p].Vvir <= 0.0) {
        strdot = 0.0;
    } else {
        tdyn = reff / galaxies[p].Vvir;

        // from Kauffmann (1996) eq7 x piR^2, (Vvir in km/s, reff in Mpc/h) in units of 10^10Msun/h
        const double cold_crit = KAUFFMANN96_SF_THRESHOLD * galaxies[p].Vvir * reff;
        if(galaxies[p].ColdGas > cold_crit && tdyn > 0.0) {
            strdot = run_params->SfrEfficiency * (galaxies[p].ColdGas - cold_crit) / tdyn;
        } else {
            strdot = 0.0;
        }
    }

    return strdot;
}

/*
 * SFprescription == 1 -- Blitz & Rosolowsky (2006) pressure-based H2
 * fraction; SF follows the molecular gas.  Sets galaxies[p].H2gas.
 *
 * Returns the instantaneous SFR strdot [10^10 Msun/h / code time].
 */
static double sfr_br06(const int p, struct GALAXY *galaxies, const struct params *run_params)
{
    double reff, tdyn, strdot;

    strdot = 0.0;
    tdyn = 0.0;


    // ========================================================================
    // Blitz and Rosolowsky (2006) - BR06 Model
    // ========================================================================

    // we take the typical star forming region as 3.0*r_s using the Milky Way as a guide
    reff = SF_DISK_RADIUS_FRAC * galaxies[p].DiskScaleRadius;

    if(galaxies[p].Vvir <= 0.0) {
        galaxies[p].H2gas = 0.0;
        strdot = 0.0;
    } else {
        tdyn = reff / galaxies[p].Vvir;
        // BR06 model
        const float h = run_params->Hubble_h;  /* float on purpose: frozen single-precision behaviour, do not promote (see docs/physics/units.md) */
        const float rs_pc = CODE_LENGTH_TO_PC(galaxies[p].DiskScaleRadius, h);
        if (rs_pc <= 0.0) {
            galaxies[p].H2gas = 0.0;
            strdot = 0.0;
        } else {
            // H2 mass via BR06: radial integration or single-slab
            if(run_params->H2RadialIntegrationOn) {
                calculate_molecular_fraction_radial_integration(p, galaxies, run_params, NULL);
                // result already stored in galaxies[p].H2gas by the function
            } else {
                float disk_area_pc2;
                if (run_params->H2DiskAreaOption == 0) {
                    disk_area_pc2 = M_PI * pow(rs_pc, 2);
                } else if (run_params->H2DiskAreaOption == 1) {
                    disk_area_pc2 = M_PI * pow(3.0 * rs_pc, 2);
                } else {
                    disk_area_pc2 = 2.0 * M_PI * pow(rs_pc, 2);
                }
                const float gas_surface_density  = (CODE_MASS_TO_MSUN(galaxies[p].ColdGas, h)) / disk_area_pc2;
                const float star_surface_density = CODE_MASS_TO_MSUN(galaxies[p].StellarMass - galaxies[p].BulgeMass, h) / disk_area_pc2;
                galaxies[p].H2gas = calculate_molecular_fraction_BR06(gas_surface_density, star_surface_density,
                                                                       rs_pc) * (galaxies[p].ColdGas * HYDROGEN_MASS_FRAC);
            }

            if(galaxies[p].H2gas > galaxies[p].ColdGas * HYDROGEN_MASS_FRAC) {
                galaxies[p].H2gas = galaxies[p].ColdGas * HYDROGEN_MASS_FRAC;
                clamp_count_h2_cap++;
            }

            if(galaxies[p].H2gas > 0.0 && tdyn > 0.0) {
                strdot = run_params->SfrEfficiency * galaxies[p].H2gas / tdyn;
            } else {
                strdot = 0.0;
            }
        }
    }

    return strdot;
}

/*
 * SFprescription == 2 -- Somerville et al. (2025) density-modulated
 * efficiency (their eqs. 2, 3, 8) applied to total cold gas; H2gas is
 * forced to zero (no H2 tracking in this prescription).
 *
 * Returns the instantaneous SFR strdot [10^10 Msun/h / code time].
 */
static double sfr_somerville25_coldgas(const int p, struct GALAXY *galaxies, const struct params *run_params)
{
    double reff, tdyn, strdot;

    strdot = 0.0;
    tdyn = 0.0;


    // =======================================================================
    // Somerville et al. 2025: Density Modulated Star Formation Efficiency
    // Using Equation 3 for efficiency: epsilon = (Sigma/Sigma_crit)/(1 + Sigma/Sigma_crit)
    // =======================================================================

    // No H2 tracking in this prescription
    galaxies[p].H2gas = 0.0;

    // we take the typical star forming region as 3.0*r_s using the Milky Way as a guide
    reff = SF_DISK_RADIUS_FRAC * galaxies[p].DiskScaleRadius;

    if(galaxies[p].Vvir <= 0.0) {
        strdot = 0.0;
    } else {
        tdyn = reff / galaxies[p].Vvir;
        const float h = run_params->Hubble_h;  /* float on purpose: frozen single-precision behaviour, do not promote (see docs/physics/units.md) */
        const float rs_pc = CODE_LENGTH_TO_PC(galaxies[p].DiskScaleRadius, h);
        float disk_area_pc2 = M_PI * pow(3.0 * rs_pc, 2); // pc^2
        float gas_surface_density = (disk_area_pc2 > 0.0) ?
            (CODE_MASS_TO_MSUN(galaxies[p].ColdGas, h)) / disk_area_pc2 : 0.0; // Msun/pc^2

        // Critical surface density from Equation 2
        const double Sigma_crit = SOMERVILLE25_SIGMA_CRIT;

        // Cloud-scale star formation efficiency from Equation 3
        double epsilon_cl = (gas_surface_density / Sigma_crit) / (1.0 + gas_surface_density / Sigma_crit);

        // Fraction of gas in dense clouds (f_dense from Equation 8)
        const double f_dense = SOMERVILLE25_F_DENSE;

        // Star formation rate: SFR ~ epsilon_cl * f_dense * m_gas / tdyn
        if(tdyn > 0.0 && gas_surface_density > 0.0) {
            strdot = epsilon_cl * f_dense * galaxies[p].ColdGas / tdyn;
        } else {
            strdot = 0.0;
        }
    }

    return strdot;
}

/*
 * SFprescription == 3 -- Somerville et al. (2025) efficiency applied to
 * BR06 molecular gas.  Sets galaxies[p].H2gas.
 *
 * Returns the instantaneous SFR strdot [10^10 Msun/h / code time].
 */
static double sfr_somerville25_h2(const int p, struct GALAXY *galaxies, const struct params *run_params)
{
    double reff, tdyn, strdot;

    strdot = 0.0;
    tdyn = 0.0;
    double total_molecular_gas;


    // =======================================================================
    // Somerville et al. 2025: Density Modulated Star Formation Efficiency with H2
    // Using Equation 3 for efficiency: epsilon = (Sigma/Sigma_crit)/(1 + Sigma/Sigma_crit)
    // But replacing cold gas with H2 gas using Blitz & Rosolowsky 2006
    // =======================================================================

    // we take the typical star forming region as 3.0*r_s using the Milky Way as a guide
    reff = SF_DISK_RADIUS_FRAC * galaxies[p].DiskScaleRadius;

    if(galaxies[p].Vvir <= 0.0) {
        galaxies[p].H2gas = 0.0;
        strdot = 0.0;
    } else {
        tdyn = reff / galaxies[p].Vvir;
        const float h = run_params->Hubble_h;  /* float on purpose: frozen single-precision behaviour, do not promote (see docs/physics/units.md) */
        const float rs_pc = CODE_LENGTH_TO_PC(galaxies[p].DiskScaleRadius, h);

        if (rs_pc <= 0.0) {
            galaxies[p].H2gas = 0.0;
            strdot = 0.0;
        } else {
            // H2 mass via BR06: radial integration or single-slab
            float gas_surface_density = 0.0f;
            if(run_params->H2RadialIntegrationOn) {
                calculate_molecular_fraction_radial_integration(p, galaxies, run_params, NULL);
                // result already stored in galaxies[p].H2gas by the function
                // compute gas_surface_density for epsilon_cl below using pi*(3*r_s)^2 as reference
                const float ref_area = (float)(M_PI * pow(3.0 * rs_pc, 2));
                gas_surface_density = (ref_area > 0.0f) ? (galaxies[p].ColdGas * 1.0e10f / h) /* 1.0e10f float literal on purpose: frozen single-precision behaviour */ / ref_area : 0.0f;
            } else {
                float disk_area_pc2;
                if (run_params->H2DiskAreaOption == 0) {
                    disk_area_pc2 = M_PI * pow(rs_pc, 2);
                } else if (run_params->H2DiskAreaOption == 1) {
                    disk_area_pc2 = M_PI * pow(3.0 * rs_pc, 2);
                } else {
                    disk_area_pc2 = 2.0 * M_PI * pow(rs_pc, 2);
                }
                gas_surface_density = (CODE_MASS_TO_MSUN(galaxies[p].ColdGas, h)) / disk_area_pc2;
                const float stellar_surface_density = (CODE_MASS_TO_MSUN(galaxies[p].StellarMass - galaxies[p].BulgeMass, h)) / disk_area_pc2;
                total_molecular_gas = calculate_molecular_fraction_BR06(gas_surface_density, stellar_surface_density,
                                                                        rs_pc) * (galaxies[p].ColdGas * HYDROGEN_MASS_FRAC);
                galaxies[p].H2gas = total_molecular_gas;
            }

            if(galaxies[p].H2gas > galaxies[p].ColdGas * HYDROGEN_MASS_FRAC) {
                galaxies[p].H2gas = galaxies[p].ColdGas * HYDROGEN_MASS_FRAC;
                clamp_count_h2_cap++;
            }

            // Critical surface density from Equation 2
            const double Sigma_crit = SOMERVILLE25_SIGMA_CRIT;

            // Cloud-scale star formation efficiency from Equation 3
            double epsilon_cl = (gas_surface_density / Sigma_crit) / (1.0 + gas_surface_density / Sigma_crit);

            // Fraction of gas in dense clouds (f_dense from Equation 8)
            const double f_dense = SOMERVILLE25_F_DENSE;

            // Star formation rate using H2 gas instead of total cold gas
            if(tdyn > 0.0 && gas_surface_density > 0.0 && galaxies[p].H2gas > 0.0) {
                strdot = epsilon_cl * f_dense * galaxies[p].H2gas / tdyn;
            } else {
                strdot = 0.0;
            }
        }
    }

    return strdot;
}

/*
 * SFprescription == 4 -- Krumholz & Dekel (2012) H2 fraction.
 * Sets galaxies[p].H2gas.
 *
 * Returns the instantaneous SFR strdot [10^10 Msun/h / code time].
 */
static double sfr_kd12(const int p, struct GALAXY *galaxies, const struct params *run_params)
{
    double reff, tdyn, strdot;

    strdot = 0.0;
    tdyn = 0.0;
    double metallicity, total_molecular_gas;
    metallicity = 0.0;


    // ========================================================================
    // Krumholz and Dekel (2012) - KD12 Model
    // ========================================================================

    if(galaxies[p].Vvir <= 0.0) {
        galaxies[p].H2gas = 0.0;
        strdot = 0.0;
    } else {
        reff = SF_DISK_RADIUS_FRAC * galaxies[p].DiskScaleRadius;
        tdyn = reff / galaxies[p].Vvir;
        const float h = run_params->Hubble_h;  /* float on purpose: frozen single-precision behaviour, do not promote (see docs/physics/units.md) */
        const float rs_pc = CODE_LENGTH_TO_PC(galaxies[p].DiskScaleRadius, h);
        if (rs_pc <= 0.0) {
            galaxies[p].H2gas = 0.0;
            strdot = 0.0;
        } else {
            if(run_params->H2RadialIntegrationOn) {
                calculate_molecular_fraction_radial_integration(p, galaxies, run_params, NULL);
            } else {
                // Choose disk area based on H2DiskAreaOption
                float disk_area;
                if (run_params->H2DiskAreaOption == 0) {
                    disk_area = M_PI * pow(rs_pc, 2);
                } else if (run_params->H2DiskAreaOption == 1) {
                    disk_area = M_PI * pow(3.0 * rs_pc, 2);
                } else {
                    disk_area = 2.0 * M_PI * pow(rs_pc, 2);
                }
                if(disk_area <= 0.0) {
                    galaxies[p].H2gas = 0.0;
                } else {
                    float surface_density = (CODE_MASS_TO_MSUN(galaxies[p].ColdGas, h)) / disk_area;
                    if(galaxies[p].ColdGas > 0.0) {
                        metallicity = galaxies[p].MetalsColdGas / galaxies[p].ColdGas;
                    }
                    float clumping_factor = 5.0;
                    total_molecular_gas = calculate_H2_fraction_KD12(surface_density, metallicity, clumping_factor) * (galaxies[p].ColdGas * HYDROGEN_MASS_FRAC);
                    galaxies[p].H2gas = total_molecular_gas;
                }
            }
            // Safety check: H2 fraction cannot exceed 1.0
            if(galaxies[p].H2gas > galaxies[p].ColdGas * HYDROGEN_MASS_FRAC) {
                galaxies[p].H2gas = galaxies[p].ColdGas * HYDROGEN_MASS_FRAC;
                clamp_count_h2_cap++;
            }

            if (galaxies[p].H2gas > 0.0 && tdyn > 0.0) {
                strdot = run_params->SfrEfficiency * galaxies[p].H2gas / tdyn;
            } else {
                strdot = 0.0;
            }
        }
    }

    return strdot;
}

/*
 * SFprescription == 5 -- Krumholz, McKee & Tumlinson (2009) H2 fraction
 * (inline two-phase fit).  Sets galaxies[p].H2gas.
 *
 * Returns the instantaneous SFR strdot [10^10 Msun/h / code time].
 */
static double sfr_kmt09(const int p, struct GALAXY *galaxies, const struct params *run_params)
{
    double reff, tdyn, strdot;

    strdot = 0.0;
    tdyn = 0.0;


    // ========================================================================
    // Krumholz, McKee, & Tumlinson (2009) - KMT09 Model
    // ========================================================================

    
    // 1. Geometry and Units
    reff = SF_DISK_RADIUS_FRAC * galaxies[p].DiskScaleRadius;
    tdyn = reff / galaxies[p].Vvir;
    
    // Check for physical validity
    if(galaxies[p].Vvir <= 0.0 || galaxies[p].DiskScaleRadius <= 0.0) {
        galaxies[p].H2gas = 0.0;
        strdot = 0.0;
    } else {
        const float h = run_params->Hubble_h;  /* float on purpose: frozen single-precision behaviour, do not promote (see docs/physics/units.md) */
        // Scale radius in pc
        const float rs_pc = CODE_LENGTH_TO_PC(galaxies[p].DiskScaleRadius, h);

        if(run_params->H2RadialIntegrationOn) {
            calculate_molecular_fraction_radial_integration(p, galaxies, run_params, NULL);
        } else {
            // Choose disk area based on H2DiskAreaOption
            float disk_area_pc2;
            if (run_params->H2DiskAreaOption == 0) {
                disk_area_pc2 = M_PI * pow(rs_pc, 2);
            } else if (run_params->H2DiskAreaOption == 1) {
                disk_area_pc2 = M_PI * pow(3.0 * rs_pc, 2);
            } else {
                disk_area_pc2 = 2.0 * M_PI * pow(rs_pc, 2);
            }

            // Gas Surface Density (Msun/pc^2) - Sigma_g
            float gas_surface_density = (disk_area_pc2 > 0.0) ?
                (CODE_MASS_TO_MSUN(galaxies[p].ColdGas, h)) / disk_area_pc2 : 0.0;

            float metallicity_abs = 0.0;
            if(galaxies[p].ColdGas > 0.0) {
                metallicity_abs = galaxies[p].MetalsColdGas / galaxies[p].ColdGas;
            }
            float Z_prime = (metallicity_abs > 0.0) ? metallicity_abs / 0.02 : 0.0;

            const float clumping_factor = 3.0;
            float Sigma_comp = clumping_factor * gas_surface_density;
            double tau_c = 0.066 * clumping_factor * Z_prime * gas_surface_density;
            float chi = 0.77 * (1.0 + 3.1 * pow(Z_prime, 0.365));
            float s = 0.0;
            if (Sigma_comp > 0.0 && tau_c > 1e-10) {
                s = log(1.0 + 0.6 * chi + 0.01 * chi * chi) / (0.6 * tau_c);
            } else {
                s = 100.0;
            }

            float f_H2 = 0.0;
            if (s < 2.0) {
                f_H2 = 1.0 - (3.0 * s) / (4.0 + s);
            }
            if (f_H2 < 0.0) f_H2 = 0.0;
            if (f_H2 > 1.0) f_H2 = 1.0;

            galaxies[p].H2gas = f_H2 * (galaxies[p].ColdGas * HYDROGEN_MASS_FRAC);
        }

        // Can't create more H2 than total cold gas
        if(galaxies[p].H2gas > galaxies[p].ColdGas * HYDROGEN_MASS_FRAC) {
            galaxies[p].H2gas = galaxies[p].ColdGas * HYDROGEN_MASS_FRAC;
            clamp_count_h2_cap++;
        }

        if (galaxies[p].H2gas > 0.0 && tdyn > 0.0) {
            strdot = run_params->SfrEfficiency * galaxies[p].H2gas / tdyn;
        } else {
            strdot = 0.0;
        }
    }

    return strdot;
}

/*
 * SFprescription == 6 -- Krumholz (2013) molecule-poor SF law using the
 * eq. 28 depletion time.  Sets galaxies[p].H2gas and H2DepletionTime_Gyr.
 *
 * Returns the instantaneous SFR strdot [10^10 Msun/h / code time].
 */
static double sfr_k13(const int p, struct GALAXY *galaxies, const struct params *run_params)
{
    double reff, tdyn, strdot;

    strdot = 0.0;
    tdyn = 0.0;


    // ========================================================================
    // Krumholz 2013 (KMT+) Model
    // "The star formation law in molecule-poor galaxies"
    // Uses the analytic approximation for depletion time (Equation 28)
    // ========================================================================

    reff = SF_DISK_RADIUS_FRAC * galaxies[p].DiskScaleRadius;
    tdyn = reff / galaxies[p].Vvir;

    // Basic safety checks
    if(galaxies[p].Vvir <= 0.0 || galaxies[p].ColdGas <= 0.0 || galaxies[p].DiskScaleRadius <= 0.0) {
        strdot = 0.0;
        galaxies[p].H2gas = 0.0;
    } else {
        tdyn = reff / galaxies[p].Vvir; // Code units

        const float h = run_params->Hubble_h;  /* float on purpose: frozen single-precision behaviour, do not promote (see docs/physics/units.md) */
        const float rs_pc = CODE_LENGTH_TO_PC(galaxies[p].DiskScaleRadius, h);

        if(run_params->H2RadialIntegrationOn) {
            // Radially integrate both H2 mass and K13 SFR consistently.
            // Sigma(r)/t_dep(r) is summed over the disk using the local f_H2(r) at each annulus,
            // avoiding the single-slab Sigma = M/(pi r_s^2) = 2Sigma0 overestimate.
            double strdot_k13 = 0.0;
            calculate_molecular_fraction_radial_integration(p, galaxies, run_params, &strdot_k13);
            if(galaxies[p].H2gas > galaxies[p].ColdGas * HYDROGEN_MASS_FRAC) {
                galaxies[p].H2gas = galaxies[p].ColdGas * HYDROGEN_MASS_FRAC;
                clamp_count_h2_cap++;
            }
            // H2DepletionTime_Gyr = M_gas / SFR_K13_integrated, set inside function
            strdot = strdot_k13;
        } else {
            // Slab path: single representative surface density from H2DiskAreaOption
            double Sigma_gas_k13 = 0.0, Sigma_star_k13 = 0.0, Z_prime_k13 = 0.01, f_H2_2p_k13 = 0.0;
            float area_pc2;
            if (run_params->H2DiskAreaOption == 0) {
                area_pc2 = M_PI * pow(rs_pc, 2);
            } else if (run_params->H2DiskAreaOption == 1) {
                area_pc2 = M_PI * pow(3.0 * rs_pc, 2);
            } else {
                area_pc2 = 2.0 * M_PI * pow(rs_pc, 2);
            }

            if(area_pc2 > 0.0) {
                Sigma_gas_k13  = (CODE_MASS_TO_MSUN(galaxies[p].ColdGas, h)) / area_pc2;
                Sigma_star_k13 = (CODE_MASS_TO_MSUN(galaxies[p].StellarMass - galaxies[p].BulgeMass, h)) / area_pc2;
                const double Z_gas = (galaxies[p].ColdGas > 0.0) ? (galaxies[p].MetalsColdGas / galaxies[p].ColdGas) : 0.0;
                f_H2_2p_k13 = calculate_H2_fraction_K13(Sigma_gas_k13, Z_gas, 5.0);
                Z_prime_k13 = (Z_gas > 0.0) ? Z_gas / Z_SOLAR_ASPLUND09 : 0.0;
                if(Z_prime_k13 < 0.01) Z_prime_k13 = 0.01;
                galaxies[p].H2gas = f_H2_2p_k13 * (galaxies[p].ColdGas * HYDROGEN_MASS_FRAC);
            } else {
                galaxies[p].H2gas = 0.0;
            }

            if(galaxies[p].H2gas > galaxies[p].ColdGas * HYDROGEN_MASS_FRAC) {
                galaxies[p].H2gas = galaxies[p].ColdGas * HYDROGEN_MASS_FRAC;
                clamp_count_h2_cap++;
            }

            const double t_dep_Gyr = calculate_tdep_K13_Gyr((float)Sigma_gas_k13, (float)Sigma_star_k13,
                                                               rs_pc, (float)Z_prime_k13, (float)f_H2_2p_k13);
            galaxies[p].H2DepletionTime_Gyr = (t_dep_Gyr > 0.0) ? (float)t_dep_Gyr : -1.0f;

            strdot = (galaxies[p].H2gas > 0.0 && tdyn > 0.0)
                     ? run_params->SfrEfficiency * galaxies[p].H2gas / tdyn : 0.0;
        }
    }

    return strdot;
}

/*
 * SFprescription == 7 -- Gnedin & Draine (2014; 2016 erratum fit) H2
 * fraction.  Sets galaxies[p].H2gas.
 *
 * Returns the instantaneous SFR strdot [10^10 Msun/h / code time].
 */
static double sfr_gd14(const int p, struct GALAXY *galaxies, const struct params *run_params)
{
    double reff, tdyn, strdot;

    strdot = 0.0;
    tdyn = 0.0;

    
    // ========================================================================
    // Gnedin & Draine (2014) - GD14 Model
    // Implemented using the "more accurate and simpler fit" from the 
    // 2016 Erratum (ApJ, 830, 54)
    // ========================================================================

    // we take the typical star forming region as 3.0*r_s using the Milky Way as a guide
    reff = SF_DISK_RADIUS_FRAC * galaxies[p].DiskScaleRadius;

    // Basic safety checks
    if(galaxies[p].Vvir <= 0.0 || galaxies[p].ColdGas <= 0.0 || galaxies[p].DiskScaleRadius <= 0.0) {
        strdot = 0.0;
        galaxies[p].H2gas = 0.0;
    } else {
        tdyn = reff / galaxies[p].Vvir; // Code units
        
        const float h = run_params->Hubble_h;  /* float on purpose: frozen single-precision behaviour, do not promote (see docs/physics/units.md) */
        // Scale radius in pc
        const float rs_pc = CODE_LENGTH_TO_PC(galaxies[p].DiskScaleRadius, h);

        if(run_params->H2RadialIntegrationOn) {
            calculate_molecular_fraction_radial_integration(p, galaxies, run_params, NULL);
        } else {
            // Choose disk area based on H2DiskAreaOption
            float disk_area_pc2;
            if (run_params->H2DiskAreaOption == 0) {
                disk_area_pc2 = M_PI * pow(rs_pc, 2);
            } else if (run_params->H2DiskAreaOption == 1) {
                disk_area_pc2 = M_PI * pow(3.0 * rs_pc, 2);
            } else {
                disk_area_pc2 = 2.0 * M_PI * pow(rs_pc, 2);
            }

            double Sigma_gas = 0.0;
            if(disk_area_pc2 > 0.0) {
                Sigma_gas = (CODE_MASS_TO_MSUN(galaxies[p].ColdGas, h)) / disk_area_pc2;
            }

            const double metallicity_abs = (galaxies[p].ColdGas > 0.0) ?
                galaxies[p].MetalsColdGas / galaxies[p].ColdGas : 0.0;
            const double f_H2 = calculate_H2_fraction_GD14(Sigma_gas, metallicity_abs, rs_pc);
            galaxies[p].H2gas = f_H2 * (galaxies[p].ColdGas * HYDROGEN_MASS_FRAC);
        }

        // Can't create more H2 than total cold gas
        if(galaxies[p].H2gas > galaxies[p].ColdGas * HYDROGEN_MASS_FRAC) {
            galaxies[p].H2gas = galaxies[p].ColdGas * HYDROGEN_MASS_FRAC;
            clamp_count_h2_cap++;
        }

        if(galaxies[p].H2gas > 0.0 && tdyn > 0.0) {
            strdot = run_params->SfrEfficiency * galaxies[p].H2gas / tdyn;
        } else {
            strdot = 0.0;
        }
    }

    return strdot;
}

/*
 * SN feedback masses for one SF event (main, non-FFB path).
 *
 * Given the stellar mass formed this substep (*stars_inout), computes the
 * reheated and ejected gas masses: FIRE scalings (Muratov+15 reheating,
 * Hirschmann+16 energy-based ejection) when FIREmodeOn == 1, otherwise the
 * fixed-epsilon Croton+06 forms.  Rescales *stars_inout and the reheated
 * mass together when their sum exceeds the available cold gas.  Sets
 * galaxies[p].MassLoading in FIRE mode.  All masses [10^10 Msun/h].
 */
static void compute_sn_feedback(const int p, double *stars_inout, double *reheated_out,
                                double *ejected_out, struct GALAXY *galaxies,
                                const struct params *run_params)
{
    double stars = *stars_inout;
    double ejected_mass;

// FIRE velocity/redshift scaling (Muratov et al. 2015, eq. 9/11).
// Pre-computed once and reused for both reheating and ejection to avoid
// duplication. scaling = (1+z)^alpha * (V/V_crit)^beta, where beta has two
// slopes: -3.2 below FIRE_V_CRIT_KMS and -1.0 above.  Zero when FIRE is off.
double fire_scaling = 0.0;
if(run_params->FIREmodeOn == 1 && run_params->SupernovaRecipeOn == 1) {
    const double z_fire = run_params->ZZ[galaxies[p].SnapNum];
    const double vc_fire = galaxies[p].Vvir;
    if(vc_fire > 0.0 && z_fire >= 0.0) {
        const double vc_floored = (vc_fire < 1.0) ? 1.0 : vc_fire;
        const double v_term = (vc_floored < FIRE_V_CRIT_KMS)
            ? pow(vc_floored / FIRE_V_CRIT_KMS, -3.2)
            : pow(vc_floored / FIRE_V_CRIT_KMS, -1.0);
        fire_scaling = pow(1.0 + z_fire, run_params->RedshiftPowerLawExponent) * v_term;
    }
}

// Calculate reheated mass - use FIRE model if enabled, otherwise use original feedback
double reheated_mass = 0.0;

if(run_params->SupernovaRecipeOn == 1) {
    if(run_params->FIREmodeOn == 1) {
        // FIRE: eta = FeedbackReheatingEpsilon * fire_scaling (Muratov+2015)
        const double eta_reheat = capped_eta_reheat(
            run_params->FeedbackReheatingEpsilon * fire_scaling, galaxies[p].Vvir, run_params);
        galaxies[p].MassLoading = (float)eta_reheat;
        reheated_mass = eta_reheat * stars;
    } else {
        reheated_mass = run_params->FeedbackReheatingEpsilon * stars;
    }
}

XASSERT(reheated_mass >= 0.0, -1,
        "Error: Expected reheated gas-mass = %g to be >=0.0\n", reheated_mass);

// cant use more cold gas than is available! so balance SF and feedback
if((stars + reheated_mass) > galaxies[p].ColdGas && (stars + reheated_mass) > 0.0) {
    const double fac = galaxies[p].ColdGas / (stars + reheated_mass);
    stars *= fac;
    reheated_mass *= fac;
}

// determine ejection
if(run_params->SupernovaRecipeOn == 1) {
    if(galaxies[p].Vvir > 0.0) {
        if(run_params->FIREmodeOn == 1) {
            // FIRE: energy-based ejection (Hirschmann+2016).
            // E_FB = epsilon_eject * fire_scaling * 0.5 * M_* * (eta_SN * E_SN)
            // Eject whatever energy remains after lifting the reheated gas.
            const double vc = galaxies[p].Vvir;
            const double E_FB = sn_energy_coupling(fire_scaling, run_params) *
                                0.5 * stars * (run_params->EtaSNcode * run_params->EnergySNcode);
            const double E_lift = 0.5 * reheated_mass * vc * vc;
            ejected_mass = (E_FB > E_lift) ? (E_FB - E_lift) / (0.5 * vc * vc) : 0.0;
        } else {
            // Original non-FIRE calculation
            ejected_mass = (run_params->FeedbackEjectionEfficiency * 
                           (run_params->EtaSNcode * run_params->EnergySNcode) / 
                           (galaxies[p].Vvir * galaxies[p].Vvir) -
                           run_params->FeedbackReheatingEpsilon) * stars;
        }
        if (run_params->KarpovModeOn == 1 && galaxies[p].Vvir > 0.0 &&
            galaxies[p].ColdGas > 0.0 && galaxies[p].DiskScaleRadius > 0.0 &&
            run_params->Hubble_h > 0.0 && isfinite(stars) && isfinite(galaxies[p].MetalsColdGas)) {
            // Calculate Metallicity (Z/Z_sun)
            double Z_ratio = 0.0;
            if(galaxies[p].ColdGas > 0.0) {
                // Assuming MetalsColdGas is in the same mass units as ColdGas
                Z_ratio = (galaxies[p].MetalsColdGas / galaxies[p].ColdGas) / Z_SOLAR_ASPLUND09; 
            }

            // Enforce the low-metallicity floor
            if (Z_ratio < 0.01) {
                Z_ratio = 0.01; // H and He cooling dominates below this
            }

            // --- Calculate Gas Number Density (n) in cm^-3 ---
            // 1. Convert DiskRadius to physical cm (assuming input is comoving Mpc/h)
            double r_disk_cm = (galaxies[p].DiskScaleRadius / run_params->Hubble_h) * CM_PER_MPC;
            
            // 2. Estimate disk volume (Cylinder: V = pi * r^2 * h)
            double scale_height_cm = 0.1 * r_disk_cm; // Typical SAM assumption: h is 10% of R
            double volume_cm3 = M_PI * r_disk_cm * r_disk_cm * scale_height_cm;
            
            // 3. Convert ColdGas mass to grams
            double mass_msun = CODE_MASS_TO_MSUN(galaxies[p].ColdGas, run_params->Hubble_h);
            double mass_grams = mass_msun * SOLAR_MASS;
            
            // 4. Calculate number density (n = rho / mean_molecular_weight)
            double rho = mass_grams / volume_cm3;
            double n = rho / (1.4 * PROTONMASS);

            // KARPOV ET AL. (2020) RECIPE
            
            // Assuming 1 SN per 100 M_sun of stars formed[cite: 1]
            double m_stars_msun = CODE_MASS_TO_MSUN(stars, run_params->Hubble_h);
            double num_sn = m_stars_msun * run_params->EtaSN;
            
            // Table 2 Parameters[cite: 1]
            double M_cool_A = 5.914e2;
            double M_cool_alpha = 1.944;
            double M_cool_beta = -0.057;
            double M_cool_gamma = -0.302;
            double P_A = 3.252e43;
            double P_alpha = 1.404;
            double P_beta = -5.730;
            double P_gamma = -1.487;

            /* 
             * IMPORTANT: The OCR of Equation 5 in the PDF is distorted ("X = A10810(2 (-) Z \beta").
             * The mathematical form below is a placeholder power-law. You must open the original 
             * PDF to verify the exact arrangement of the alpha, beta, and gamma exponents. 
             */
            // double M_cool_A = 5.914e2, M_cool_alpha = 1.944, M_cool_beta = -0.057, M_cool_gamma = -0.302;
            // double P_A = 3.252e43,     P_alpha = 1.404,      P_beta = -5.730,      P_gamma = -1.487;

            if (isfinite(n) && n > 0.0 && isfinite(num_sn) && num_sn >= 0.0) {
                double m_cool_single = M_cool_A * pow(M_cool_alpha, log10(100.0 / n)) * pow(Z_ratio - M_cool_beta, M_cool_gamma);
                double p_single = P_A * pow(P_alpha, log10(100.0 / n)) * pow(Z_ratio - P_beta, P_gamma);

                // 1. Reheated Mass: SN cooling mass * number of SN[cite: 1]
                double total_reheated_msun = m_cool_single * num_sn;
                reheated_mass = MSUN_TO_CODE_MASS(total_reheated_msun, run_params->Hubble_h);

                // 2. Ejected Mass: Momentum-driven ejection[cite: 1]
                double total_momentum = p_single * num_sn; 
            
            // Convert Vvir (km/s) to cm/s for escape velocity calculation
                double v_esc_cm_s = galaxies[p].Vvir * 1e5; 
            
                if (v_esc_cm_s > 0.0 && isfinite(m_cool_single) && isfinite(p_single)) {
                    double total_ejected_mass_g = total_momentum / v_esc_cm_s;
                    double total_ejected_mass_msun = total_ejected_mass_g / SOLAR_MASS;
                    ejected_mass = MSUN_TO_CODE_MASS(total_ejected_mass_msun, run_params->Hubble_h);
                }
            }
        }
    } else {
        ejected_mass = 0.0;
    }
    
    if(ejected_mass < 0.0) {
        ejected_mass = 0.0;
    }
} else {
    ejected_mass = 0.0;
}

    *stars_inout  = stars;
    *reheated_out = reheated_mass;
    *ejected_out  = ejected_mass;
}

/*
 * SN feedback masses for one FFB star formation event.
 *
 * Same scalings as compute_sn_feedback(), but kept as a separate function
 * because the FFB path deliberately differs: the FIRE branch guards on a
 * validity flag and leaves the reheated mass at zero when the FIRE scaling
 * is undefined (Vvir <= 0 or z < 0) instead of falling through.
 * All masses [10^10 Msun/h].
 */
static void compute_sn_feedback_ffb(const int p, double *stars_inout, double *reheated_out,
                                    double *ejected_out, struct GALAXY *galaxies,
                                    const struct params *run_params)
{
    double stars = *stars_inout;
    double reheated_mass = 0.0;
    double ejected_mass  = 0.0;

// FIRE velocity/redshift scaling (Muratov et al. 2015), computed once and
// reused for both reheating and ejection: z and Vvir do not change between
// the two blocks, so the value is identical. Mirrors the main SF path.
double fire_scaling = 0.0;
int fire_scaling_valid = 0;
if(run_params->SupernovaRecipeOn == 1 && run_params->FIREmodeOn == 1) {
    const double z  = run_params->ZZ[galaxies[p].SnapNum];
    const double vc = galaxies[p].Vvir;
    if(vc > 0.0 && z >= 0.0) {
        const double vc_floored = (vc < 1.0) ? 1.0 : vc;
        const double z_term     = pow(1.0 + z, run_params->RedshiftPowerLawExponent);
        const double v_term     = (vc_floored < FIRE_V_CRIT_KMS) ?
            pow(vc_floored / FIRE_V_CRIT_KMS, -3.2) : pow(vc_floored / FIRE_V_CRIT_KMS, -1.0);
        fire_scaling = z_term * v_term;
        fire_scaling_valid = 1;
    }
}

if(run_params->SupernovaRecipeOn == 1) {
    if(run_params->FIREmodeOn == 1) {
        if(fire_scaling_valid) {
            const double eta_reheat  = capped_eta_reheat(
                run_params->FeedbackReheatingEpsilon * fire_scaling, galaxies[p].Vvir, run_params);
            galaxies[p].MassLoading  = (float)eta_reheat;
            reheated_mass            = eta_reheat * stars;
        }
    } else {
        reheated_mass = run_params->FeedbackReheatingEpsilon * stars;
    }
}

XASSERT(reheated_mass >= 0.0, -1,
        "Error: Expected reheated gas-mass = %g to be >=0.0\n", reheated_mass);

if((stars + reheated_mass) > galaxies[p].ColdGas && (stars + reheated_mass) > 0.0) {
    const double fac = galaxies[p].ColdGas / (stars + reheated_mass);
    stars         *= fac;
    reheated_mass *= fac;
}

if(run_params->SupernovaRecipeOn == 1) {
    if(galaxies[p].Vvir > 0.0) {
        if(run_params->FIREmodeOn == 1) {
            if(fire_scaling_valid) {
                const double vc         = galaxies[p].Vvir;
                const double E_FB       = sn_energy_coupling(fire_scaling, run_params)
                                          * 0.5 * stars
                                          * (run_params->EtaSNcode * run_params->EnergySNcode);
                const double E_lift     = 0.5 * reheated_mass * vc * vc;
                ejected_mass = (E_FB > E_lift) ? (E_FB - E_lift) / (0.5 * vc * vc) : 0.0;
            }
        } else {
            ejected_mass = (run_params->FeedbackEjectionEfficiency
                            * (run_params->EtaSNcode * run_params->EnergySNcode)
                            / (galaxies[p].Vvir * galaxies[p].Vvir)
                            - run_params->FeedbackReheatingEpsilon) * stars;
        }
    }
    if(ejected_mass < 0.0) ejected_mass = 0.0;
}

    *stars_inout  = stars;
    *reheated_out = reheated_mass;
    *ejected_out  = ejected_mass;
}

/*
 * Main star formation and feedback driver for one galaxy per substep.
 *
 * Selects the active SF prescription (run_params->SFprescription):
 *   0 = Croton+06 cold-gas threshold (Kauffmann 1996 Eq. 7)
 *   1 = Blitz & Rosolowsky 2006 (BR06) H2 fraction
 *   2 = Somerville+2025 density-modulated efficiency, cold gas
 *   3 = Somerville+2025 density-modulated efficiency + BR06 H2
 *   4 = Krumholz & Dekel 2012 (KD12)
 *   5 = Krumholz, McKee & Tumlinson 2009 (KMT09)
 *   6 = Krumholz 2013 (K13)
 *   7 = Gnedin & Draine 2014 (GD14), 2016 Erratum fit
 *
 * SN feedback uses FIRE (Muratov+2015) when FIREmodeOn=1, otherwise fixed
 * reheating epsilon. Ejected mass set by energy budget. Calls
 * update_from_star_formation() and update_from_feedback() to commit reservoir
 * changes, then checks disk instability and applies instantaneous metal
 * recycling (Krumholz & Dekel 2011 Eq. 22). If the galaxy is in FFB regime,
 * delegates to starformation_ffb() and returns immediately.
 */
void starformation_and_feedback(const int p, const int centralgal, const double time, const double dt, const int halonr, const int step,
                                struct GALAXY *galaxies, const struct params *run_params)
{
    XASSERT(step >= 0 && step < STEPS, -1,
            "Error: step = %d is out of bounds [0, %d)\n", step, STEPS);

    // ========================================================================
    // CHECK FOR FFB REGIME - EARLY EXIT IF FFB
    // ========================================================================
    if(run_params->FeedbackFreeModeOn >= 1 && galaxies[p].FFBRegime == 1) {
        // This is a Feedback-Free Burst halo
        // Use specialized FFB star formation (no feedback)
        starformation_ffb(p, centralgal, dt, step, galaxies, run_params);
        return;  // Exit early - FFB path complete
    }

    double stars, ejected_mass, metallicity;

    // star formation recipes: one prescription, one function (see sfr_* above)
    double strdot = 0.0;
    switch(run_params->SFprescription) {
        case 0: strdot = sfr_croton06(p, galaxies, run_params); break;
        case 1: strdot = sfr_br06(p, galaxies, run_params); break;
        case 2: strdot = sfr_somerville25_coldgas(p, galaxies, run_params); break;
        case 3: strdot = sfr_somerville25_h2(p, galaxies, run_params); break;
        case 4: strdot = sfr_kd12(p, galaxies, run_params); break;
        case 5: strdot = sfr_kmt09(p, galaxies, run_params); break;
        case 6: strdot = sfr_k13(p, galaxies, run_params); break;
        case 7: strdot = sfr_gd14(p, galaxies, run_params); break;
        default:
            fprintf(stderr, "No star formation prescription selected!\n");
            ABORT(0);
    }

    // Calculate HI (atomic hydrogen) as the remainder of hydrogen after H2.
    // Total hydrogen = ColdGas * HYDROGEN_MASS_FRAC (0.74). The ionisation cut
    // (see ionized_gas_fraction) applies to the *atomic remainder* only -- H2 is
    // central and shielded -- so the molecular and ionised claims can no longer
    // overdraw the hydrogen budget and HI is non-negative by construction. Only
    // HI is debited; SF/ColdGas are untouched.
    {
        double atomicH = galaxies[p].ColdGas * HYDROGEN_MASS_FRAC - galaxies[p].H2gas;
        if(atomicH < 0.0) {
            atomicH = 0.0;  // H2 is capped at the budget; this guards float rounding only
            clamp_count_h1_negative++;
        }
        const double f_ion = ionized_gas_fraction(galaxies[p].ColdGas,
                                                  atomic_disk_radius(galaxies[p].DiskScaleRadius, run_params),
                                                  run_params->Hubble_h, SIGMA_HI_CRIT);
        atomicH *= (1.0 - f_ion);
        galaxies[p].H1gas = atomicH;
    }

    stars = strdot * dt;
    if(stars < 0.0) {
        stars = 0.0;
    }

    double reheated_mass;
    compute_sn_feedback(p, &stars, &reheated_mass, &ejected_mass, galaxies, run_params);


    // update the star formation rate
    galaxies[p].SfrDisk[step] += stars / dt;
    galaxies[p].SfrDiskColdGas[step] = galaxies[p].ColdGas;
    galaxies[p].SfrDiskColdGasMetals[step] = galaxies[p].MetalsColdGas;

    // update for star formation
    metallicity = get_metallicity(galaxies[p].ColdGas, galaxies[p].MetalsColdGas);
    update_from_star_formation(p, stars, metallicity, galaxies, run_params);

    // Track star formation history - accumulate stellar mass formed at this snapshot
    // Note: RecycleFraction * stars is instantly recycled, so actual stellar mass added is (1 - RecycleFraction) * stars
    if(run_params->SaveFullSFH) {
        const int snapnum = galaxies[p].SnapNum;
        if(snapnum >= 0 && snapnum < ABSOLUTEMAXSNAPS) {
            galaxies[p].SFHMassDisk[snapnum] += (1.0 - run_params->RecycleFraction) * stars;
        }
    }

    // recompute the metallicity of the cold phase
    metallicity = get_metallicity(galaxies[p].ColdGas, galaxies[p].MetalsColdGas);

    // Safety check: ensure reheated_mass doesn't exceed remaining ColdGas (floating-point precision)
    if(reheated_mass > galaxies[p].ColdGas) {
        reheated_mass = galaxies[p].ColdGas;
        clamp_count_reheat_coldgas++;
    }

    // update from SN feedback
    update_from_feedback(p, centralgal, reheated_mass, ejected_mass, metallicity, galaxies, run_params);

    // check for disk instability
    if(run_params->DiskInstabilityOn) {
        check_disk_instability(p, centralgal, halonr, time, dt, step, galaxies, run_params);
    }

    // formation of new metals - instantaneous recycling approximation - only SNII
    if(galaxies[p].ColdGas > 1.0e-8) {
        const double FracZleaveDiskVal = run_params->FracZleaveDisk * exp(-1.0 * galaxies[centralgal].Mvir / KD11_METAL_HALO_MASS);  /* Krumholz & Dekel 2011 Eq. 22 */
        
        // Metals that stay in disk (same for all regimes)
        galaxies[p].MetalsColdGas += run_params->Yield * (1.0 - FracZleaveDiskVal) * stars;
        
        // Metals that leave disk - regime dependent
        const double metals_leaving_disk = run_params->Yield * FracZleaveDiskVal * stars;
        
        add_metals_to_hot_reservoir(&galaxies[centralgal], run_params, metals_leaving_disk);
        
    } else {
        // All metals leave disk when ColdGas is very low - regime dependent
        const double all_metals = run_params->Yield * stars;
        
        add_metals_to_hot_reservoir(&galaxies[centralgal], run_params, all_metals);
    }
}

/*
 * Apply a star formation event: update cold gas, metals, and stellar mass.
 *
 * Removes (1 - RecycleFraction) * stars from ColdGas (the rest is recycled
 * immediately), increments StellarMass, and tracks metals consistently.
 * Called for each SF event within starformation_and_feedback().
 */
void update_from_star_formation(const int p, const double stars, const double metallicity, struct GALAXY *galaxies, const struct params *run_params)
{
    const double RecycleFraction = run_params->RecycleFraction;
    // update gas and metals from star formation
    galaxies[p].ColdGas -= (1 - RecycleFraction) * stars;
    galaxies[p].MetalsColdGas -= metallicity * (1 - RecycleFraction) * stars;
    galaxies[p].StellarMass += (1 - RecycleFraction) * stars;
    galaxies[p].MetalsStellarMass += metallicity * (1 - RecycleFraction) * stars;

    // H2gas and H1gas were computed before SF depleted ColdGas; clamp so they
    // remain consistent with the remaining cold gas. Only applies to H2-tracking
    // prescriptions (0=Croton and 2=Somerville-noH2 never set H2gas/H1gas).
    if(sf_prescription_tracks_h2(run_params->SFprescription)) {
        const float max_h = (galaxies[p].ColdGas > 0.0f) ? galaxies[p].ColdGas * HYDROGEN_MASS_FRAC : 0.0f;
        if(galaxies[p].H2gas > max_h) galaxies[p].H2gas = max_h;
        if(galaxies[p].H1gas > max_h) galaxies[p].H1gas = max_h;
    }
}

// ============================================================================
/*
 * Apply supernova feedback: reheat cold gas and eject hot gas.
 *
 * Transfers reheated_mass from ColdGas to HotGas (tracking metals), and
 * ejects ejected_mass from HotGas to EjectedMass. Handles routing to CGMgas
 * when CGMrecipeOn is active. Both quantities may be zero for quiescent steps.
 */
void update_from_feedback(const int p, const int centralgal, double reheated_mass, double ejected_mass, const double metallicity,
                          struct GALAXY *galaxies, const struct params *run_params)
{
    // Safety: Clamp reheated_mass to available ColdGas to handle floating-point precision errors
    // This can occur when dealing with very small masses (e.g., 1e-44) where rounding errors
    // cause reheated_mass to slightly exceed ColdGas after star formation has consumed gas

    if(reheated_mass > galaxies[p].ColdGas) {
        reheated_mass = galaxies[p].ColdGas;
        clamp_count_reheat_coldgas++;
    }

    XASSERT(reheated_mass >= 0.0, -1,
            "Error: For galaxy = %d (halonr = %d, centralgal = %d) with MostBoundID = %lld, the reheated mass = %g should be >=0.0",
            p, galaxies[p].HaloNr, centralgal, galaxies[p].MostBoundID, reheated_mass);
    XASSERT(reheated_mass <= galaxies[p].ColdGas, -1,
            "Error: Reheated mass = %g should be <= the coldgas mass of the galaxy = %g",
            reheated_mass, galaxies[p].ColdGas);

    if(run_params->SupernovaRecipeOn == 1) {
        // Remove reheated mass from cold gas (same for all regimes)
        galaxies[p].ColdGas -= reheated_mass;
        galaxies[p].MetalsColdGas -= metallicity * reheated_mass;
        if(sf_prescription_tracks_h2(run_params->SFprescription)) {
            const float max_h_fb = (galaxies[p].ColdGas > 0.0f) ? galaxies[p].ColdGas * HYDROGEN_MASS_FRAC : 0.0f;
            if(galaxies[p].H2gas > max_h_fb) galaxies[p].H2gas = max_h_fb;
            if(galaxies[p].H1gas > max_h_fb) galaxies[p].H1gas = max_h_fb;
        }

        if(run_params->CGMrecipeOn == 1) {
            if(galaxies[centralgal].Regime == 0) {
                // CGM-regime: Cold --> CGM --> Ejected

                // Add reheated gas to CGM
                galaxies[centralgal].CGMgas += reheated_mass;
                galaxies[centralgal].MetalsCGMgas += metallicity * reheated_mass;

                // Check if ejection is possible from CGM
                if(ejected_mass > galaxies[centralgal].CGMgas) {
                    ejected_mass = galaxies[centralgal].CGMgas;
                }
                const double metallicityCGM = get_metallicity(galaxies[centralgal].CGMgas, galaxies[centralgal].MetalsCGMgas);

                double metalsCGM_to_eject = metallicityCGM * ejected_mass;
                if(metalsCGM_to_eject > galaxies[centralgal].MetalsCGMgas) {
                    metalsCGM_to_eject = galaxies[centralgal].MetalsCGMgas;
                }

                // Eject from CGM to EjectedMass
                galaxies[centralgal].CGMgas -= ejected_mass;
                galaxies[centralgal].MetalsCGMgas -= metalsCGM_to_eject;
                galaxies[centralgal].EjectedMass += ejected_mass;
                galaxies[centralgal].MetalsEjectedMass += metalsCGM_to_eject;

            } else {
                // Hot-ICM-regime: Cold --> HotGas --> Ejected

                // Add reheated gas to HotGas
                galaxies[centralgal].HotGas += reheated_mass;
                galaxies[centralgal].MetalsHotGas += metallicity * reheated_mass;

                // Check if ejection is possible from HotGas
                if(ejected_mass > galaxies[centralgal].HotGas) {
                    ejected_mass = galaxies[centralgal].HotGas;
                }
                const double metallicityHot = get_metallicity(galaxies[centralgal].HotGas, galaxies[centralgal].MetalsHotGas);

                double metalsHot_to_eject = metallicityHot * ejected_mass;
                if(metalsHot_to_eject > galaxies[centralgal].MetalsHotGas) {
                    metalsHot_to_eject = galaxies[centralgal].MetalsHotGas;
                }

                // Eject from HotGas to EjectedMass
                galaxies[centralgal].HotGas -= ejected_mass;
                galaxies[centralgal].MetalsHotGas -= metalsHot_to_eject;
                galaxies[centralgal].EjectedMass += ejected_mass;
                galaxies[centralgal].MetalsEjectedMass += metalsHot_to_eject;
            }
        } else {
            // Original SAGE behavior: Cold --> HotGas --> Ejected

            // Add reheated gas to HotGas
            galaxies[centralgal].HotGas += reheated_mass;
            galaxies[centralgal].MetalsHotGas += metallicity * reheated_mass;

            // Check if ejection is possible from HotGas
            if(ejected_mass > galaxies[centralgal].HotGas) {
                ejected_mass = galaxies[centralgal].HotGas;
            }
            const double metallicityHot = get_metallicity(galaxies[centralgal].HotGas, galaxies[centralgal].MetalsHotGas);

            double metalsHot_to_eject = metallicityHot * ejected_mass;
            if(metalsHot_to_eject > galaxies[centralgal].MetalsHotGas) {
                metalsHot_to_eject = galaxies[centralgal].MetalsHotGas;
            }

            // Eject from HotGas to EjectedMass
            galaxies[centralgal].HotGas -= ejected_mass;
            galaxies[centralgal].MetalsHotGas -= metalsHot_to_eject;
            galaxies[centralgal].EjectedMass += ejected_mass;
            galaxies[centralgal].MetalsEjectedMass += metalsHot_to_eject;
        }

        galaxies[p].OutflowRate += reheated_mass;
    }
}

// ============================================================================
/*
 * Feedback-free burst (FFB) star formation (Li et al. 2024).
 *
 * Triggered when FeedbackFreeModeOn > 0 and FFBRegime==1. Computes a burst
 * SFR from cold gas (or H2 for modes 6/7) and runs standard SN feedback,
 * updating StellarMass, SFR history, and cold gas per substep. Metal
 * production and CGM/HotGas routing follow the main SF path.
 */
void starformation_ffb(const int p, const int centralgal, const double dt, const int step,
                       struct GALAXY *galaxies, const struct params *run_params)
{
    // ========================================================================
    // FEEDBACK-FREE BURST (FFB) STAR FORMATION
    // Implementation of Li et al. 2024 - Equation (4) (modified to be Kauffmann-like)
    // ========================================================================

    double reff, tdyn, strdot, stars, metallicity;

    // Calculate dynamical time
    reff = SF_DISK_RADIUS_FRAC * galaxies[p].DiskScaleRadius;
    tdyn = (reff > 0.0 && galaxies[p].Vvir > 0.0) ? reff / galaxies[p].Vvir : 0.0;

    // ========================================================================
    // H2 CALCULATION -- only for FeedbackFreeModeOn=6/7 (H2-based FFB SF modes).
    // All other FFB modes use ColdGas for SF and leave H2gas = 0.
    // H1 is derived immediately after.
    // ========================================================================
    const int uses_h2 = (run_params->FeedbackFreeModeOn == 6 || run_params->FeedbackFreeModeOn == 7);
    galaxies[p].H2gas = 0.0;

    if(uses_h2 && galaxies[p].ColdGas > 0.0 && galaxies[p].DiskScaleRadius > 0.0) {
        const float h     = run_params->Hubble_h;  /* float on purpose: frozen single-precision behaviour, do not promote (see docs/physics/units.md) */
        const float rs_pc = CODE_LENGTH_TO_PC(galaxies[p].DiskScaleRadius, h);
        const int sfpres  = run_params->SFprescription;
        const int has_h2  = sf_prescription_tracks_h2(sfpres);

        if(rs_pc > 0.0 && has_h2) {
            if(run_params->H2RadialIntegrationOn) {
                // Unified radial integration path -- handles all H2 prescriptions internally
                calculate_molecular_fraction_radial_integration(p, galaxies, run_params, NULL);
            } else {
                // Single-slab path
                float disk_area_pc2;
                if(run_params->H2DiskAreaOption == 0)
                    disk_area_pc2 = M_PI * pow(rs_pc, 2);
                else if(run_params->H2DiskAreaOption == 1)
                    disk_area_pc2 = M_PI * pow(3.0 * rs_pc, 2);
                else
                    disk_area_pc2 = 2.0 * M_PI * pow(rs_pc, 2);

                if(disk_area_pc2 > 0.0) {
                    const float Sigma_gas = (CODE_MASS_TO_MSUN(galaxies[p].ColdGas, h)) / disk_area_pc2;

                    if(sf_prescription_is_br06(sfpres)) {
                        // BR06
                        const float Sigma_star = CODE_MASS_TO_MSUN(galaxies[p].StellarMass - galaxies[p].BulgeMass, h) / disk_area_pc2;
                        galaxies[p].H2gas = calculate_molecular_fraction_BR06(Sigma_gas, Sigma_star, rs_pc)
                                            * (galaxies[p].ColdGas * HYDROGEN_MASS_FRAC);

                    } else if(sfpres == 4) {
                        // KD12
                        const double met = (galaxies[p].ColdGas > 0.0) ?
                            galaxies[p].MetalsColdGas / galaxies[p].ColdGas : 0.0;
                        galaxies[p].H2gas = calculate_H2_fraction_KD12(Sigma_gas, met, 5.0f)
                                            * (galaxies[p].ColdGas * HYDROGEN_MASS_FRAC);

                    } else if(sfpres == 5) {
                        // KMT09
                        float met_abs = (galaxies[p].ColdGas > 0.0) ?
                            galaxies[p].MetalsColdGas / galaxies[p].ColdGas : 0.0;
                        float Z_prime = (met_abs > 0.0f) ? met_abs / 0.02f : 0.0f;
                        const float tau_c = 0.066f * 3.0f * Z_prime * Sigma_gas;
                        const float chi = 0.77f * (1.0f + 3.1f * powf(Z_prime, 0.365f));
                        const float s = (tau_c > 1e-10f) ?
                            logf(1.0f + 0.6f*chi + 0.01f*chi*chi) / (0.6f*tau_c) : 100.0f;
                        float f_H2 = (s < 2.0f) ? 1.0f - (3.0f*s)/(4.0f+s) : 0.0f;
                        if(f_H2 < 0.0f) f_H2 = 0.0f;
                        if(f_H2 > 1.0f) f_H2 = 1.0f;
                        galaxies[p].H2gas = f_H2 * (galaxies[p].ColdGas * HYDROGEN_MASS_FRAC);

                    } else if(sfpres == 6) {
                        // K13: two-phase molecular fraction
                        const double Z_gas = (galaxies[p].ColdGas > 0.0) ?
                            galaxies[p].MetalsColdGas / galaxies[p].ColdGas : 0.0;
                        const double f_H2_2p = calculate_H2_fraction_K13(Sigma_gas, Z_gas, 5.0);
                        galaxies[p].H2gas = f_H2_2p * (galaxies[p].ColdGas * HYDROGEN_MASS_FRAC);

                    } else if(sfpres == 7) {
                        // GD14
                        const double met_abs = (galaxies[p].ColdGas > 0.0) ?
                            galaxies[p].MetalsColdGas / galaxies[p].ColdGas : 0.0;
                        const double f_H2 = calculate_H2_fraction_GD14(Sigma_gas, met_abs, rs_pc);
                        galaxies[p].H2gas = f_H2 * (galaxies[p].ColdGas * HYDROGEN_MASS_FRAC);
                    }
                }
            }
        }
    }

    if(galaxies[p].H2gas > galaxies[p].ColdGas * HYDROGEN_MASS_FRAC) { galaxies[p].H2gas = galaxies[p].ColdGas * HYDROGEN_MASS_FRAC; clamp_count_h2_cap++; }

    // HI = atomic remainder after H2, with the ionisation cut applied to the
    // remainder only -- matching the non-FFB path.
    {
        double atomicH = galaxies[p].ColdGas * HYDROGEN_MASS_FRAC - galaxies[p].H2gas;
        if(atomicH < 0.0) { atomicH = 0.0; clamp_count_h1_negative++; }  // float-rounding guard only
        const double f_ion = ionized_gas_fraction(galaxies[p].ColdGas,
                                                  atomic_disk_radius(galaxies[p].DiskScaleRadius, run_params),
                                                  run_params->Hubble_h, SIGMA_HI_CRIT);
        atomicH *= (1.0 - f_ion);
        galaxies[p].H1gas = atomicH;
    }

    // ========================================================================
    // SELECT GAS RESERVOIR FOR FFB STAR FORMATION
    // ========================================================================
    const double gas_for_sf = uses_h2 ? galaxies[p].H2gas : galaxies[p].ColdGas;

    // ========================================================================
    // COMPUTE STAR FORMATION RATE
    // SFR = epsilon_FFB * M_gas / t_dyn  (no critical density threshold in FFB)
    // ========================================================================
    if(isnan(galaxies[p].ColdGas) || isinf(galaxies[p].ColdGas) ||
       isnan(galaxies[p].Vvir)    || isinf(galaxies[p].Vvir)    ||
       isnan(reff) || isinf(reff) || isnan(tdyn) || isinf(tdyn)) {
        stars = 0.0;
    } else if(tdyn > 0.0 && gas_for_sf > 0.0) {
        const double epsilon_ffb = run_params->FFBMaxEfficiency;
        strdot = epsilon_ffb * gas_for_sf / tdyn;

        if(isnan(strdot) || isinf(strdot) || strdot < 0.0) {
            stars = 0.0;
        } else {
            stars = strdot * dt;
            if(stars > galaxies[p].ColdGas) stars = galaxies[p].ColdGas;
            if(isnan(stars) || isinf(stars) || stars < 0.0) stars = 0.0;
        }
    } else {
        stars = 0.0;
    }

    // ========================================================================
    // SFR TRACKING AND STAR FORMATION UPDATE
    // ========================================================================
    galaxies[p].SfrDisk[step] += stars / dt;
    galaxies[p].SfrDiskColdGas[step]       = galaxies[p].ColdGas;
    galaxies[p].SfrDiskColdGasMetals[step] = galaxies[p].MetalsColdGas;

    metallicity = get_metallicity(galaxies[p].ColdGas, galaxies[p].MetalsColdGas);
    update_from_star_formation(p, stars, metallicity, galaxies, run_params);

    if(run_params->SaveFullSFH) {
        const int snapnum = galaxies[p].SnapNum;
        if(snapnum >= 0 && snapnum < ABSOLUTEMAXSNAPS) {
            galaxies[p].SFHMassDisk[snapnum] += (1.0 - run_params->RecycleFraction) * stars;
        }
    }

    // ========================================================================
    // SUPERNOVA FEEDBACK
    // ========================================================================
    double reheated_mass = 0.0;
    double ejected_mass  = 0.0;
    compute_sn_feedback_ffb(p, &stars, &reheated_mass, &ejected_mass, galaxies, run_params);

    if(reheated_mass > galaxies[p].ColdGas) { reheated_mass = galaxies[p].ColdGas; clamp_count_reheat_coldgas++; }

    update_from_feedback(p, centralgal, reheated_mass, ejected_mass, metallicity, galaxies, run_params);

    // ========================================================================
    // METAL PRODUCTION (instantaneous recycling approximation - SNII only)
    // ========================================================================
    if(galaxies[p].ColdGas > 1.0e-8) {
        const double FracZleaveDiskVal = run_params->FracZleaveDisk
                                         * exp(-1.0 * galaxies[centralgal].Mvir / KD11_METAL_HALO_MASS);
        galaxies[p].MetalsColdGas += run_params->Yield * (1.0 - FracZleaveDiskVal) * stars;

        const double metals_leaving_disk = run_params->Yield * FracZleaveDiskVal * stars;
        add_metals_to_hot_reservoir(&galaxies[centralgal], run_params, metals_leaving_disk);
    } else {
        const double all_metals = run_params->Yield * stars;
        add_metals_to_hot_reservoir(&galaxies[centralgal], run_params, all_metals);
    }

    // ========================================================================
    // NO DISK INSTABILITY CHECK -- rapid SF stabilizes the disk
    // ========================================================================
}
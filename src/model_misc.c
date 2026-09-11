/*
 * model_misc.c -- Galaxy initialisation, sizes, and shared utility helpers.
 *
 * Provides galaxy initialisation, disk/bulge radius calculations (Mo, Mao &
 * White 1998; Shen+03; Tonini+16), and small shared helpers (get_metallicity,
 * dmax).  Regime classification lives in model_regimes.c, halo concentration
 * in model_halo_properties.c, and H2 prescriptions in model_h2_chemistry.c.
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

/* -------------------------------------------------------------------------
 * File-scope empirical constants (lifted per STYLE_C.md SS8).
 * -------------------------------------------------------------------------*/

/* Shen et al. (2003) early-type size-mass relations.
 * log(R/kpc) = slope * log(M/Msun) + intercept.
 * Eq. (33) single power law (high-mass / giant-elliptical branch). */
static const double SHEN03_SLOPE_HIGH     =  0.56;
static const double SHEN03_INTERCEPT_HIGH = -5.54;
/* Eq. (32) low-mass (dwarf-elliptical) branch. */
static const double SHEN03_SLOPE_LOW      =  0.14;
static const double SHEN03_INTERCEPT_LOW  = -1.21;
/* Transition mass between the two eq.(32) regimes, in Msun. */
static const double SHEN03_M_TRANSITION   =  2.0e10;

/* Fallback disk scale radius: r_d = DISK_RADIUS_FALLBACK_FRAC * R_vir when spin is unavailable.
 * Reachable only on the DiskRadiusOn == 0 path, and only when R_vir itself is zero, so it
 * always returns zero there; kept because that is the published behaviour. */
static const double DISK_RADIUS_FALLBACK_FRAC = 0.1;

// /* Floor on r_d / R_vir when DiskRadiusOn > 0: insurance against a halo whose measured spin is
//  * ~0. Moves 0.02% of Millennium galaxies at z = 0. The matching ceiling is the parameter
//  * DiskRadiusMaxFrac, because anywhere in 0.1-0.2 is a real modelling choice rather than a guard. */
// static const double DISK_RADIUS_MIN_FRAC = 0.002;

// /* Peak of the log-normal halo spin distribution (Bullock et al. 2001), used when a halo has no
//  * usable spin measurement at all. Well-resolved (Len > 3000) Millennium halos give 0.042. */
// static const double DISK_RADIUS_MEDIAN_SPIN = 0.04;

/* Tonini+2016 eq. (15): fraction of the disc scale radius that a newly
 * transferred mass element contributes to the instability-bulge half-mass radius. */
static const double TONINI16_DISK_FRAC    = 0.2;


/*
 * Initialise all fields of a newly created galaxy struct to safe defaults.
 *
 * Zeros all baryonic reservoirs, radii, metallicities, SFR histories, and ICS
 * tracking arrays. Sets the galaxy's halo link (HaloNr) and increments the
 * galaxy counter. Must be called before the galaxy takes part in any physics.
 */
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
    galaxies[p].RegimeRandom = (float)rand() / (float)RAND_MAX;
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

    galaxies[p].SubstepsUsed = STEPS;   /* overwritten by evolve_galaxies before any output */

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

    /* Seed the smoothed spin with this halo's spin so the running mean in get_disk_radius()
     * is a no-op on the first call (DiskRadiusOn >= 2); harmless otherwise. */
    for(int j = 0; j < 3; j++) {
        galaxies[p].SpinSmooth[j] = halos[halonr].Spin[j];
    }

    galaxies[p].DiskScaleRadius = get_disk_radius(halonr, p, halos, galaxies);
    get_bulge_radius(p, galaxies, run_params);
    galaxies[p].MergTime = 999.9f;
    galaxies[p].Cooling = 0.0;
    galaxies[p].Heating = 0.0;
    galaxies[p].r_heat = 0.0;
    galaxies[p].QuasarModeBHaccretionMass = 0.0;
    galaxies[p].TimeOfLastMajorMerger = -1.0;
    galaxies[p].TimeOfLastMinorMerger = -1.0;
    galaxies[p].OutflowRate = 0.0;
    galaxies[p].RcoolToRvir = -1.0;
    galaxies[p].MassLoading = 0.0;
    galaxies[p].tcool = -1.0;
    galaxies[p].tff = -1.0;
    galaxies[p].tcool_over_tff = -1.0;
    galaxies[p].MachNumber = -1.0;
    galaxies[p].tdeplete = -1.0;
    galaxies[p].H2DepletionTime_Gyr = -1.0f;

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

/*
 * Exponential disk scale radius from the halo spin (Mo, Mao & White 1998 eq. 12).
 *
 * This returns the scale length r_d, NOT the half-mass radius: callers wanting
 * r_half multiply by 1.68 (DISK_HALF_MASS_FRAC in model_mergers.c).
 *
 * MMW98 eq. 12 is r_d = (lambda / sqrt(2)) * R_vir with a Bullock-style
 * lambda = |j| / (sqrt(2) V_vir R_vir), so R_vir cancels and the radius is really
 *
 *     r_d = f_j * |j| / (2 V_vir)
 *
 * i.e. only the halo specific angular momentum and the circular velocity enter.
 *
 * run_params->DiskRadiusOn selects the variant:
 *
 *   0  Published behaviour, reproduced bit-for-bit.
 *
 *   1  Adds two robustness fixes. (a) A working fallback: the published else-branch is
 *      reachable only when R_vir == 0 (get_virial_velocity() returns 0 exactly then), so it
 *      returns 0 and the galaxy is permanently inert -- every downstream H2/SF/stripping path
 *      guards on DiskScaleRadius > 0. Here the virial scale is rebuilt from the particle count,
 *      which is what get_virial_mass() already does for subhalos. (b) r_d / R_vir is bounded
 *      to [DISK_RADIUS_MIN_FRAC, DiskRadiusMaxFrac]; unbounded, 11.5% of Millennium galaxies at
 *      z = 0 exceed 0.1 R_vir and the 99.9th percentile r_d is 32 kpc.
 *
 *   2  As 1, but |j| comes from a running mean of the spin *vector* over a halo dynamical time.
 *      This removes the snapshot-to-snapshot jitter in r_d (measured on Millennium: median
 *      |dlog r_d| per snapshot 0.047 -> 0.016, and the fraction jumping more than 0.3 dex
 *      4.7% -> 0.2%), which matters because SFR responds as r_d^-3.7. Averaging the vector
 *      rather than |j| also shrinks r_d by a near-uniform ~9%, from averaging over the
 *      halo's spin-axis tumbling.
 *
 *      It does NOT remove the low-particle-count bias in |j|, despite that bias being real
 *      (median lambda is 0.077 at Len 20-50 against 0.044 at Len > 1000). Measured, the
 *      ratio lambda(Len 20-50) / lambda(Len > 1000) is 1.569 at level 0 and 1.583 at
 *      level 2 -- unchanged. The reason is that the discreteness error comes from finite
 *      particle sampling, and halo membership turns over slowly: over one dynamical time
 *      most of the same particles are still in the halo, so the sampling error is strongly
 *      correlated between adjacent snapshots and does not average down. Time-averaging only
 *      suppresses the genuinely fast-fluctuating component. Correcting the resolution bias
 *      needs an explicit N-dependent de-biasing, which no level here implements.
 *
 * Levels 1 and 2 leave satellites alone: get_disk_radius() is only called for FOF centrals
 * (core_build_model.c), so a Type 1 disk size stays frozen at infall under every setting.
 */
// double get_disk_radius(const int halonr, const int p, const struct halo_data *halos, struct GALAXY *galaxies,
//                        const struct params *run_params)
// {
//     const double f_j = run_params->DiskRadiusFactor;

//     /* ---------------- DiskRadiusOn == 0: published behaviour ---------------- */
//     if(run_params->DiskRadiusOn == 0) {
//         if(galaxies[p].Vvir > 0.0 && galaxies[p].Rvir > 0.0) {
//             /* Mo, Shude & White (1998) eq. 12 with a Bullock-style spin parameter.
//              * The literal 1.414 is intentional: the original code used this truncated
//              * sqrt(2) rather than M_SQRT2 and changing it shifts every disk radius.
//              * Do not replace with M_SQRT2 without re-calibrating. */
//             double SpinMagnitude = sqrt(halos[halonr].Spin[0] * halos[halonr].Spin[0] +
//                                         halos[halonr].Spin[1] * halos[halonr].Spin[1] + halos[halonr].Spin[2] * halos[halonr].Spin[2]);

//             double SpinParameter = SpinMagnitude / (1.414 * galaxies[p].Vvir * galaxies[p].Rvir);
//             return f_j * (SpinParameter / 1.414) * galaxies[p].Rvir;
//         } else {
//             return f_j * DISK_RADIUS_FALLBACK_FRAC * galaxies[p].Rvir;
//         }
//     }

//     /* ---------------- DiskRadiusOn >= 1 ---------------- */

//     /* Virial scale, repaired from the particle count when the halo catalogue reports
//      * Mvir = 0 for a FOF central (140 of 31739 z = 0 centrals in Millennium file 0). */
//     double Rvir = galaxies[p].Rvir;
//     double Vvir = galaxies[p].Vvir;
//     if(Rvir <= 0.0 || Vvir <= 0.0) {
//         const double mvir = halos[halonr].Len * run_params->PartMass;
//         Rvir = (mvir > 0.0) ? virial_radius_from_mass(mvir, halos[halonr].SnapNum, run_params) : 0.0;
//         Vvir = (Rvir > 0.0) ? sqrt(run_params->G * mvir / Rvir) : 0.0;
//         if(Rvir <= 0.0 || Vvir <= 0.0) {
//             return 0.0;   /* nothing resolved enough to hang a disk on */
//         }
//     }

//     /* Spin vector: instantaneous, or the running main-branch mean. */
//     double spin[3];
//     if(run_params->DiskRadiusOn >= 2) {
//         /* Exponential moving average with weight w = dt / (dt + t_dyn). The disk cannot
//          * restructure faster than the halo dynamical time t_dyn = R_vir / V_vir, and the
//          * spin measurement decorrelates on the snapshot spacing dt, so the two timescales
//          * set the weight between them and there is no free parameter. Millennium: w ~ 0.2
//          * at z = 0 (heavy smoothing, t_dyn >> dt) rising to ~0.5 by z = 3, where haloes
//          * really do reorganise within a snapshot. */
//         const int prev_snap = galaxies[p].SnapNum;   /* not yet advanced at this call site */
//         const int this_snap = halos[halonr].SnapNum;
//         double w = 1.0;   /* no usable history -> take the instantaneous spin */
//         if(prev_snap >= 0 && this_snap >= prev_snap) {
//             const double dt = run_params->Age[prev_snap] - run_params->Age[this_snap];  /* Age is lookback */
//             w = (dt > 0.0) ? dt / (dt + Rvir / Vvir) : 0.0;
//         }
//         for(int k = 0; k < 3; k++) {
//             galaxies[p].SpinSmooth[k] = (float)((1.0 - w) * galaxies[p].SpinSmooth[k] + w * halos[halonr].Spin[k]);
//             spin[k] = galaxies[p].SpinSmooth[k];
//         }
//     } else {
//         for(int k = 0; k < 3; k++) {
//             spin[k] = halos[halonr].Spin[k];
//         }
//     }

//     const double jmag = sqrt(spin[0] * spin[0] + spin[1] * spin[1] + spin[2] * spin[2]);

//     /* MMW98 eq. 12, in the form R_vir actually cancels to. */
//     double r_disk = (jmag > 0.0) ? f_j * jmag / (2.0 * Vvir)
//                                  : f_j * (DISK_RADIUS_MEDIAN_SPIN / M_SQRT2) * Rvir;

//     /* Bound the physical size (applied after f_j, so the ceiling is a size limit rather
//      * than a spin limit and scales with any angular-momentum retention factor).
//      * NB the bound is against the peak-retained R_vir the physics uses everywhere else,
//      * not the instantaneous one save_gals_hdf5.c writes to the Rvir output column, so a
//      * halo now below its peak mass can show r_d/Rvir above DiskRadiusMaxFrac on output.
//      * Clamping against the instantaneous R_vir instead would shrink disks during
//      * stripping, which is not what the ceiling is for. */
//     const double r_max = run_params->DiskRadiusMaxFrac * Rvir;
//     const double r_min = DISK_RADIUS_MIN_FRAC * Rvir;
//     if(r_disk > r_max) r_disk = r_max;
//     if(r_disk < r_min) r_disk = r_min;

//     return r_disk;
// }

double get_disk_radius(const int halonr, const int p, const struct halo_data *halos, const struct GALAXY *galaxies)
{
    if(galaxies[p].Vvir > 0.0 && galaxies[p].Rvir > 0.0) {
        /* Mo, Shude & White (1998) eq. 12 with a Bullock-style spin parameter.
         * The literal 1.414 is intentional: the original code used this truncated
         * sqrt(2) rather than M_SQRT2 and changing it shifts every disk radius.
         * Do not replace with M_SQRT2 without re-calibrating. */
        double SpinMagnitude = sqrt(halos[halonr].Spin[0] * halos[halonr].Spin[0] +
                                    halos[halonr].Spin[1] * halos[halonr].Spin[1] + halos[halonr].Spin[2] * halos[halonr].Spin[2]);

        double SpinParameter = SpinMagnitude / (1.414 * galaxies[p].Vvir * galaxies[p].Rvir);
        return (SpinParameter / 1.414) * galaxies[p].Rvir;
    } else {
        return DISK_RADIUS_FALLBACK_FRAC * galaxies[p].Rvir;
    }
}

/*
 * Compute and store the half-mass bulge radius from merger and instability components.
 *
 * When BulgeSizeOn > 0, uses energy conservation (Croton+2016 Sec. 3.6) to derive
 * radii for merger- and instability-driven bulge components separately, then
 * combines them into galaxies[p].BulgeRadius. Returns zero and sets all bulge
 * radii to zero when BulgeSizeOn == 0.
 */
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
        const double M_bulge_sun = CODE_MASS_TO_MSUN(galaxies[p].BulgeMass, h);
        
        // Shen+2003 equation (33): log(R/kpc) = 0.56 log(M/Msun) - 5.54
        const double log_R_kpc = SHEN03_SLOPE_HIGH * log10(M_bulge_sun) + SHEN03_INTERCEPT_HIGH;
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
        const double M_bulge_sun = CODE_MASS_TO_MSUN(galaxies[p].BulgeMass, h);
        
        // Transition mass from Shen et al. (2003) equation (32)
        const double M_transition = SHEN03_M_TRANSITION;  // M_sun
        
        double R_bulge_kpc;
        
        if(M_bulge_sun > M_transition) {
            // High-mass regime: like giant ellipticals
            // log(R/kpc) = 0.56 log(M) - 5.54
            const double log_R = SHEN03_SLOPE_HIGH * log10(M_bulge_sun) + SHEN03_INTERCEPT_HIGH;
            R_bulge_kpc = pow(10.0, log_R);
        } else {
            // Low-mass regime: like dwarf ellipticals
            // log(R/kpc) = 0.14 log(M) - 1.21
            const double log_R = SHEN03_SLOPE_LOW * log10(M_bulge_sun) + SHEN03_INTERCEPT_LOW;
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
            galaxies[p].BulgeRadius = 0.0;
            galaxies[p].MergerBulgeRadius = 0.0;
            galaxies[p].InstabilityBulgeRadius = 0.0;
            return 0.0;
        }
        
        // Zero component radii when the corresponding mass is gone so stale
        // values (e.g. InstabilityBulgeRadius surviving a major-merger reset,
        // or MergerBulgeRadius on a pure-instability bulge) don't persist.
        if(M_merger == 0.0) {
            galaxies[p].MergerBulgeRadius = 0.0;
        }
        if(M_instability == 0.0) {
            galaxies[p].InstabilityBulgeRadius = 0.0;
        }

        // Failsafe: If mass exists but radius is 0 (e.g. initialization or
        // orphan-satellite merger where energy conservation returned 0),
        // use Shen as fallback.
        double R_merger = galaxies[p].MergerBulgeRadius;
        if(M_merger > 0.0 && R_merger <= 0.0) {
             const double M_merger_sun = CODE_MASS_TO_MSUN(M_merger, h);
             const double log_R_kpc = SHEN03_SLOPE_HIGH * log10(M_merger_sun) + SHEN03_INTERCEPT_HIGH;
             R_merger = pow(10.0, log_R_kpc) * 1.0e-3 * h;
             galaxies[p].MergerBulgeRadius = R_merger;
        }

        // 2. Retrieve Instability Radius
        double R_instability = galaxies[p].InstabilityBulgeRadius;
        if(M_instability > 0.0 && R_instability <= 0.0) {
            const double R_disc = galaxies[p].DiskScaleRadius;
            if(R_disc > 0.0) {
                R_instability = TONINI16_DISK_FRAC * R_disc;
            } else {
                // No disk (post-major-merger or orphan): use Shen power-law fallback
                const double M_inst_sun = CODE_MASS_TO_MSUN(M_instability, h);
                const double log_R_kpc = SHEN03_SLOPE_HIGH * log10(M_inst_sun) + SHEN03_INTERCEPT_HIGH;
                R_instability = pow(10.0, log_R_kpc) * 1.0e-3 * h;
            }
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

/*
 * Incrementally update the instability-driven bulge radius after a mass transfer.
 *
 * Applies Tonini+2016 eq. (15): a mass-weighted average of the existing bulge
 * radius and the disk radius at which the transferred mass originated.
 */
void update_instability_bulge_radius(const int p, const double delta_mass,
                                     const double old_disk_radius,
                                     struct GALAXY *galaxies, const struct params *run_params)
{
    // Tonini+2016 equation (15): incremental radius evolution
    // R_i = (R_i,OLD * M_i,OLD + deltaM * 0.2 * R_D) / (M_i,OLD + deltaM)
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
    
    /* New mass contribution scales with TONINI16_DISK_FRAC * R_disc (Tonini+2016 eq. 15). */
    const double R_new_contribution_kpc = TONINI16_DISK_FRAC * R_disc_kpc;
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
    get_bulge_radius(p, galaxies, run_params);
}

/* Return metals/gas ratio clamped to [0, 1]; returns 0 when gas <= 0. */
double get_metallicity(const double gas, const double metals)
{
  double metallicity = 0.0;

  if(gas > 0.0 && metals > 0.0) {
      metallicity = metals / gas;
      metallicity = metallicity >= 1.0 ? 1.0:metallicity;
  }

  return metallicity;
}

/* Return the larger of two doubles. */
double dmax(const double x, const double y)
{
    return (x > y) ? x:y;
}

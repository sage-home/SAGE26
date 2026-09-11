/*
 * core_build_model.c -- merger tree traversal and the per-timestep physics loop.
 *
 * Implements three functions that together constitute the inner loop of SAGE:
 *   construct_galaxies -- recursive depth-first traversal of one merger tree;
 *                         calls itself on all progenitors, then delegates FOF
 *                         group assembly to join_galaxies_of_progenitors and
 *                         per-snapshot evolution to evolve_galaxies.
 *   join_galaxies_of_progenitors -- links progenitor galaxies to their
 *                         descendants, identifies the central, handles mergers
 *                         for halos that no longer exist, and grows the galaxy
 *                         array as needed.
 *   evolve_galaxies     -- drives the physics sub-steps for one snapshot
 *                         interval: infall, cooling, star formation, feedback,
 *                         disk instability, reincorporation, mergers, output.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>
#include <sys/stat.h>

#include "core_allvars.h"
#include "core_build_model.h"
#include "core_mymalloc.h"
#include "core_save.h"
#include "core_utils.h"

#include "model_misc.h"
#include "model_mergers.h"
#include "model_infall.h"
#include "model_ram_pressure.h"
#include "model_reincorporation.h"
#include "model_starformation_and_feedback.h"
#include "model_cooling_heating.h"


static int evolve_galaxies(const int halonr, const int ngal, int *numgals, int *maxgals, struct halo_data *halos,
                           struct halo_aux_data *haloaux, struct GALAXY **ptr_to_galaxies, struct GALAXY **ptr_to_halogal, struct params *run_params);
static int join_galaxies_of_progenitors(const int halonr, const int ngalstart, int *galaxycounter, int *maxgals, struct halo_data *halos,
                                        struct halo_aux_data *haloaux, struct GALAXY **ptr_to_galaxies, struct GALAXY **ptr_to_halogal, struct params *run_params);

/* Conversion from Mpc (code length unit) to km, used for dynamical time calculation. */
static const double KM_PER_MPC = 3.086e19;



/*
 * construct_galaxies -- recursively traverse the merger tree rooted at halonr
 * and build/evolve the galaxy population within it.
 *
 * Visits all progenitors depth-first (setting DoneFlag), then assembles the
 * FOF group at the current snapshot via join_galaxies_of_progenitors() and
 * advances galaxies through the physics loop via evolve_galaxies().
 * Returns EXIT_SUCCESS or a negative SAGE error code.
 */
int construct_galaxies(const int halonr, int *numgals, int *galaxycounter, int *maxgals, struct halo_data *halos,
                       struct halo_aux_data *haloaux, struct GALAXY **ptr_to_galaxies, struct GALAXY **ptr_to_halogal,
                       struct params *run_params)
{
  int prog, fofhalo;

  haloaux[halonr].DoneFlag = 1;

  prog = halos[halonr].FirstProgenitor;
  while(prog >= 0) {
      if(haloaux[prog].DoneFlag == 0) {
          int status = construct_galaxies(prog, numgals, galaxycounter, maxgals, halos, haloaux, ptr_to_galaxies, ptr_to_halogal, run_params);

          if(status != EXIT_SUCCESS) {
              return status;
          }
      }
      prog = halos[prog].NextProgenitor;
  }

  fofhalo = halos[halonr].FirstHaloInFOFgroup;
  if(haloaux[fofhalo].HaloFlag == 0) {
      haloaux[fofhalo].HaloFlag = 1;
      while(fofhalo >= 0) {
          prog = halos[fofhalo].FirstProgenitor;
          while(prog >= 0) {
              if(haloaux[prog].DoneFlag == 0) {
                  int status = construct_galaxies(prog, numgals, galaxycounter, maxgals, halos, haloaux, ptr_to_galaxies, ptr_to_halogal, run_params);

                  if(status != EXIT_SUCCESS) {
                      return status;
                  }
              }
              prog = halos[prog].NextProgenitor;
          }
          fofhalo = halos[fofhalo].NextHaloInFOFgroup;
      }
  }

  // At this point, the galaxies for all progenitors of this halo have been
  // properly constructed. Also, the galaxies of the progenitors of all other
  // halos in the same FOF group have been constructed as well. We can hence go
  // ahead and construct all galaxies for the subhalos in this FOF halo, and
  // evolve them in time.

  fofhalo = halos[halonr].FirstHaloInFOFgroup;
  if(haloaux[fofhalo].HaloFlag == 1 ) {
      int ngal = 0;
      haloaux[fofhalo].HaloFlag = 2;

      while(fofhalo >= 0) {
          ngal = join_galaxies_of_progenitors(fofhalo, ngal, galaxycounter, maxgals, halos, haloaux, ptr_to_galaxies, ptr_to_halogal, run_params);
          if(ngal < 0) {
              return EXIT_FAILURE;
          }
          fofhalo = halos[fofhalo].NextHaloInFOFgroup;
      }

      int status = evolve_galaxies(halos[halonr].FirstHaloInFOFgroup, ngal, numgals, maxgals, halos, haloaux, ptr_to_galaxies, ptr_to_halogal, run_params);

      if(status != EXIT_SUCCESS) {
          return status;
      }
  }


  return EXIT_SUCCESS;
}


/*
 * join_galaxies_of_progenitors -- link progenitor galaxies to the current halo
 * and process any immediate mergers.
 *
 * For each progenitor FOF halo, identifies the most massive central (lenmax
 * criterion), reassigns galaxy hosting, and calls deal_with_galaxy_merger()
 * for galaxies whose host halo no longer exists.  Grows the galaxy array
 * (reallocating via myrealloc) if the active count approaches maxgals.
 * Returns EXIT_SUCCESS or a negative SAGE error code.
 */
static int join_galaxies_of_progenitors(const int halonr, const int ngalstart, int *galaxycounter, int *maxgals, struct halo_data *halos,
                                 struct halo_aux_data *haloaux, struct GALAXY **ptr_to_galaxies, struct GALAXY **ptr_to_halogal, struct params *run_params)
{
    int ngal, prog,  first_occupied, lenmax, lenoccmax;
    struct GALAXY *galaxies = *ptr_to_galaxies;
    struct GALAXY *halogal = *ptr_to_halogal;

    lenmax = 0;
    lenoccmax = 0;
    first_occupied = halos[halonr].FirstProgenitor;
    prog = halos[halonr].FirstProgenitor;

    if(prog >=0) {
        if(haloaux[prog].NGalaxies > 0) {
            /* FirstProgenitor already has a galaxy: keep it as first_occupied unconditionally.
             * lenoccmax = -1 is a sentinel that prevents the loop below from updating first_occupied,
             * since FirstProgenitor is by definition the most bound and should host the central. */
            lenoccmax = -1;
        }
    }

    // Find most massive progenitor that contains an actual galaxy.
    // Maybe FirstProgenitor never was FirstHaloInFOFGroup and thus has no galaxy.

    while(prog >= 0) {
        if(halos[prog].Len > lenmax) {
            lenmax = halos[prog].Len;
        }

        if(lenoccmax != -1 && halos[prog].Len > lenoccmax && haloaux[prog].NGalaxies > 0) {
            lenoccmax = halos[prog].Len;
            first_occupied = prog;
        }
        prog = halos[prog].NextProgenitor;
    }

    ngal = ngalstart;
    prog = halos[halonr].FirstProgenitor;

    while(prog >= 0) {
        for(int i = 0; i < haloaux[prog].NGalaxies; i++) {
            if(ngal == (*maxgals - 1)) {
                *maxgals += 10000;

                *ptr_to_galaxies = myrealloc(*ptr_to_galaxies, *maxgals * sizeof(struct GALAXY));
                *ptr_to_halogal  = myrealloc(*ptr_to_halogal, *maxgals * sizeof(struct GALAXY));
                galaxies = *ptr_to_galaxies;
                halogal = *ptr_to_halogal;
            }

            XRETURN(ngal < *maxgals, -1,
                    "Error: ngal = %d exceeds the number of galaxies allocated = %d\n"
                    "This would result in invalid memory access...exiting\n",
                    ngal, *maxgals);

            // This is the crucial line in which the properties of the progenitor galaxies
            // are copied over (as a whole) to the (temporary) galaxies galaxies[xxx] in the current snapshot
            // After updating their properties and evolving them
            // they are copied to the end of the list of permanent galaxies halogal[xxx]

            galaxies[ngal] = halogal[haloaux[prog].FirstGalaxy + i];
            galaxies[ngal].HaloNr = halonr;

            galaxies[ngal].dT = -1.0;

            // this deals with the central galaxies of (sub)halos
            if(galaxies[ngal].Type == 0 || galaxies[ngal].Type == 1) {
                // this halo shouldn't hold a galaxy that has already merged; remove it from future processing
                if(galaxies[ngal].mergeType != 0) {
                    galaxies[ngal].Type = 3;
                    continue;
                }

                // remember properties from the last snapshot
                const double previousMvir = galaxies[ngal].Mvir;
                const double previousVvir = galaxies[ngal].Vvir;
                const double previousVmax = galaxies[ngal].Vmax;

                if(prog == first_occupied) {
                    // update properties of this galaxy with physical properties of halo
                    galaxies[ngal].MostBoundID = halos[halonr].MostBoundID;

                    for(int j = 0; j < 3; j++) {
                        galaxies[ngal].Pos[j] = halos[halonr].Pos[j];
                        galaxies[ngal].Vel[j] = halos[halonr].Vel[j];
                    }

                    galaxies[ngal].Len = halos[halonr].Len;
                    galaxies[ngal].Vmax = halos[halonr].Vmax;

                    const double new_Mvir = get_virial_mass(halonr, halos, run_params);
                    galaxies[ngal].deltaMvir = new_Mvir - galaxies[ngal].Mvir;

                    if(new_Mvir > galaxies[ngal].Mvir) {
                        galaxies[ngal].Rvir = get_virial_radius(halonr, halos, run_params);
                        galaxies[ngal].Vvir = get_virial_velocity(halonr, halos, run_params);
                    }
                    galaxies[ngal].Mvir = new_Mvir;

                    galaxies[ngal].Cooling = 0.0;
                    galaxies[ngal].Heating = 0.0;
                    galaxies[ngal].QuasarModeBHaccretionMass = 0.0;
                    galaxies[ngal].OutflowRate = 0.0;

                    for(int step = 0; step < STEPS; step++) {
                        galaxies[ngal].SfrDisk[step] = galaxies[ngal].SfrBulge[step] = 0.0;
                        galaxies[ngal].SfrDiskColdGas[step] = galaxies[ngal].SfrDiskColdGasMetals[step] = 0.0;
                        galaxies[ngal].SfrBulgeColdGas[step] = galaxies[ngal].SfrBulgeColdGasMetals[step] = 0.0;
                    }

                    if(halonr == halos[halonr].FirstHaloInFOFgroup) {
                        // a central galaxy
                        galaxies[ngal].mergeType = 0;
                        galaxies[ngal].mergeIntoID = -1;
                        galaxies[ngal].MergTime = 999.9f;

                        galaxies[ngal].DiskScaleRadius = get_disk_radius(halonr, ngal, halos, galaxies);
                        get_bulge_radius(ngal, galaxies, run_params);

                        galaxies[ngal].Type = 0;
                    } else {
                        // a satellite with subhalo
                        galaxies[ngal].mergeType = 0;
                        galaxies[ngal].mergeIntoID = -1;

                        if(galaxies[ngal].Type == 0) {  // remember the infall properties before becoming a subhalo
                            galaxies[ngal].infallMvir = previousMvir;
                            galaxies[ngal].infallVvir = previousVvir;
                            galaxies[ngal].infallVmax = previousVmax;
                            galaxies[ngal].infallStellarMass = galaxies[ngal].StellarMass;
                            galaxies[ngal].TimeOfInfall = halos[halonr].SnapNum;  // Track snapshot of infall

                        }

                        if(galaxies[ngal].Type == 0 || galaxies[ngal].MergTime > 999.0f) {
                            // here the galaxy has gone from type 1 to type 2 or otherwise doesn't have a merging time.
                            galaxies[ngal].MergTime = estimate_merging_time(halonr, halos[halonr].FirstHaloInFOFgroup, ngal, halos, galaxies, run_params);
                        }

                        galaxies[ngal].Type = 1;
                    }
                } else {
                    // an orphan satellite galaxy - these will merge or disrupt within the current timestep
                    galaxies[ngal].deltaMvir = -1.0*galaxies[ngal].Mvir;
                    galaxies[ngal].Mvir = 0.0;

                    if(galaxies[ngal].MergTime > 999.0 || galaxies[ngal].Type == 0) {
                        // here the galaxy has gone from type 0 to type 2 - merge it!
                        galaxies[ngal].MergTime = 0.0;

                        galaxies[ngal].infallMvir = previousMvir;
                        galaxies[ngal].infallVvir = previousVvir;
                        galaxies[ngal].infallVmax = previousVmax;
                        galaxies[ngal].infallStellarMass = galaxies[ngal].StellarMass;
                    }

                    galaxies[ngal].Type = 2;
                }
            }

            ngal++;
        }

        prog = halos[prog].NextProgenitor;
    }

    if(ngal == 0) {
        // We have no progenitors with galaxies. This means we create a new galaxy.
        init_galaxy(ngal, halonr, galaxycounter, halos, galaxies, run_params);
        ngal++;
    }

    // Per Halo there can be only one Type 0 or 1 galaxy, all others are Type 2  (orphan)
    // In fact, this galaxy is very likely to be the first galaxy in the halo if
    // first_occupied==FirstProgenitor and the Type0/1 galaxy in FirstProgenitor was also the first one
    // This cannot be guaranteed though for the pathological first_occupied!=FirstProgenitor case

    int centralgal = -1;
    for(int i = ngalstart; i < ngal; i++) {
        if(galaxies[i].Type == 0 || galaxies[i].Type == 1) {
            XRETURN(centralgal == -1, -1,
                    "Error: Expected to find centralgal=-1. instead centralgal=%d\n", centralgal);

            centralgal = i;
        }
    }

    for(int i = ngalstart; i < ngal; i++) {
        galaxies[i].CentralGal = centralgal;
    }

    return ngal;

}

/*
 * evolve_galaxies -- advance ngal galaxies through one snapshot interval.
 *
 * Drives STEPS (or up to MAX_STEPS adaptively) sub-steps.  Each sub-step
 * calls: infall_recipe, cooling_recipe, starformation_and_feedback,
 * check_disk_instability, and reincorporate_gas.  After the sub-steps,
 * handles any remaining mergers; save_galaxies() is called afterwards by
 * sage_per_forest().
 * Returns EXIT_SUCCESS or a negative SAGE error code.
 */
static int evolve_galaxies(const int halonr, const int ngal, int *numgals, int *maxgals, struct halo_data *halos,
                    struct halo_aux_data *haloaux, struct GALAXY **ptr_to_galaxies, struct GALAXY **ptr_to_halogal,
                    struct params *run_params)
{
    struct GALAXY *galaxies = *ptr_to_galaxies;
    struct GALAXY *halogal = *ptr_to_halogal;

    const int centralgal = galaxies[0].CentralGal;
    XRETURN(galaxies[centralgal].Type == 0 && galaxies[centralgal].HaloNr == halonr,
            EXIT_FAILURE,
            "Error: For centralgal, halonr = %d, %d.\n"
            "Expected to find galaxy.type = 0, and found type = %d.\n"
            "Expected to find galaxies[halonr] = %d and found halonr = %d\n",
            centralgal, halonr, galaxies[centralgal].Type,
            halonr,  galaxies[centralgal].HaloNr);

    /*
      MS: Note save halo_snapnum and galaxy_snapnum to local variables
          and replace all instances of snapnum to those local variables
     */

    const int halo_snapnum = halos[halonr].SnapNum;
    const double Zcurr = run_params->ZZ[halo_snapnum];

    // Compute and store halo concentration if enabled
    if(run_params->ConcentrationOn > 0) {
        for(int p = 0; p < ngal; p++) {
            if(galaxies[p].mergeType > 0) continue;
            galaxies[p].Concentration = (float)get_halo_concentration(p, Zcurr, galaxies, run_params);
        }
    }
    
    if (run_params->CGMrecipeOn == 1) {
        determine_and_store_regime(ngal, galaxies, run_params);
    }
    
    if (run_params->FeedbackFreeModeOn >= 1) {
        determine_and_store_ffb_regime(ngal, Zcurr, galaxies, run_params);
    }

    const double halo_age = run_params->Age[halo_snapnum];
    const double infallingGas = infall_recipe(centralgal, ngal, Zcurr, galaxies, run_params);

    // We integrate things forward by using a number of intervals equal to STEPS
    // Adaptive timesteps: at high-z, snapshot spacing can exceed dynamical time
    // so we use more substeps when needed
    const double deltaT_total = run_params->Age[galaxies[0].SnapNum] - halo_age;

    /* t_dyn = Rvir [Mpc/h] / Vvir [km/s] * KM_PER_MPC [km/Mpc] gives t_dyn in seconds. */
    double t_dyn_seconds = (galaxies[centralgal].Rvir / (galaxies[centralgal].Vvir + 1e-10)) * KM_PER_MPC;
    double t_dyn = t_dyn_seconds / run_params->UnitTime_in_s;

    // Scale steps proportionally: ensure we resolve evolution within each dynamical time
    // If deltaT/t_dyn > 1, the snapshot spans multiple dynamical times and we need finer resolution
    // (minimum STEPS, maximum MAX_STEPS).
    //
    // SubstepResolution (default 1.0) is a runtime multiplier that scales both the floor
    // and the cap, so the integration substep count N can be swept from the parameter file
    // for convergence / N-invariance testing without recompiling. It does NOT resize the
    // compile-time SFR history arrays (still STEPS long); adaptive substeps map back into
    // those STEPS bins as before. The -1e-9 guards against float rounding pushing an exact
    // integer product up to the next ceil.
    const double res = (run_params->SubstepResolution > 0.0) ? run_params->SubstepResolution : 1.0;
    int floor_steps = (int)ceil(STEPS * res - 1e-9);
    if(floor_steps < 1) floor_steps = 1;
    int cap_steps = (int)ceil(MAX_STEPS * res - 1e-9);
    if(cap_steps < floor_steps) cap_steps = floor_steps;

    int effective_steps = floor_steps;
    if(t_dyn > 0.0) {
        double ratio = deltaT_total / t_dyn;
        int needed = (int)ceil(floor_steps * ratio);
        if(needed > floor_steps) {
            effective_steps = needed;
        }
    }
    if(effective_steps > cap_steps) {
        effective_steps = cap_steps;
    }

    // Satellite hot-gas stripping timescale: t_strip = t_dyn(host) = Rvir/Vvir.
    const double t_strip = t_dyn;

    // Analytic satellite hot-gas stripping applied ONCE per snapshot, outside
    // the substep loop, fully decoupled from the substep count. Each satellite
    // loses exactly a fraction 1-exp(-dT/t_dyn) of its baryon excess (computed
    // inside strip_from_satellite from dt=deltaT). This is operator-split before
    // the substeps, mirroring how infallingGas is computed once up front.
    for(int p = 0; p < ngal; p++) {
        if(p == centralgal || galaxies[p].mergeType > 0) {
            continue;
        }
        // Strip satellites holding hot-phase gas in either reservoir: Hot-regime
        // in HotGas, CGM-regime in CGMgas (CGMgas is zeroed for satellites when
        // CGMrecipeOn != 1, so legacy runs are unchanged).
        if(galaxies[p].Type == 1 && (galaxies[p].HotGas > 0.0 || galaxies[p].CGMgas > 0.0)) {
            const double deltaT = run_params->Age[galaxies[p].SnapNum] - halo_age;
            strip_from_satellite(centralgal, p, Zcurr, deltaT, t_strip, galaxies, run_params);
        }
    }

    // RamPressureStrippingOn == 1: Gunn & Gott (1972) ram-pressure stripping of
    // satellite ISM (ColdGas), applied once per snapshot outside the substep
    // loop with the same analytic 1-exp(-dT/t_strip) cadence as scheme 2 above.
    // Complementary to and independent of PhysicalStrippingOn, which strips the
    // hot/CGM phase (starvation). Covers Type 1 satellites and Type 2 orphans;
    // orphans use a frozen-orbit approximation (position frozen at subhalo
    // loss, velocity replaced by the host Vvir -- see
    // ram_pressure_strip_satellite).
    if(run_params->RamPressureStrippingOn == 1) {
        for(int p = 0; p < ngal; p++) {
            if(p == centralgal || galaxies[p].mergeType > 0) {
                continue;
            }
            if((galaxies[p].Type == 1 || galaxies[p].Type == 2) && galaxies[p].ColdGas > 0.0) {
                const double deltaT = run_params->Age[galaxies[p].SnapNum] - halo_age;
                ram_pressure_strip_satellite(centralgal, p, Zcurr, deltaT, t_strip, galaxies, run_params);
            }
        }
    }

    /* Record the substep count on every galaxy in this halo. The Sfr* arrays accumulate one
     * entry per substep into STEPS fixed bins, so the output average has to divide by the
     * number of substeps actually taken rather than by STEPS -- otherwise the reported SFR
     * scales as effective_steps / STEPS. Set for all galaxies, including already-merged ones,
     * because they are still written out. */
    for(int p = 0; p < ngal; p++) {
        galaxies[p].SubstepsUsed = effective_steps;
    }

    for(int step = 0; step < effective_steps; step++) {

        // Loop over all galaxies in the halo
        for(int p = 0; p < ngal; p++) {
            // Don't treat galaxies that have already merged
            if(galaxies[p].mergeType > 0) {
                continue;
            }

            const double deltaT = run_params->Age[galaxies[p].SnapNum] - halo_age;
            const double time = run_params->Age[galaxies[p].SnapNum] - (step + 0.5) * (deltaT / effective_steps);

            if(galaxies[p].dT < 0.0) {
                galaxies[p].dT = deltaT;
            }

            // For the central galaxy only
            if(p == centralgal) {
                add_infall_to_hot(centralgal, infallingGas / effective_steps, galaxies, run_params);

                if(run_params->ReIncorporationFactor > 0.0) {
                    reincorporate_gas(centralgal, deltaT / effective_steps, galaxies, run_params);
                }
            }

            // Determine the cooling gas given the halo properties
            double coolingGas;
            if(run_params->CGMrecipeOn == 1) {

                cooling_recipe_regime_aware(p, deltaT / effective_steps, galaxies, run_params);

            } else {
                coolingGas = cooling_recipe(p, deltaT / effective_steps, galaxies, run_params);
                cool_gas_onto_galaxy(p, coolingGas, galaxies);
            }

            // stars form and then explode!
            // Map adaptive step to fixed STEPS bins for SFR arrays
            int step_bin = (step * STEPS) / effective_steps;
            if(step_bin >= STEPS) step_bin = STEPS - 1;
            starformation_and_feedback(p, centralgal, time, deltaT / effective_steps, halonr, step_bin, galaxies, run_params);
        }

        // check for satellite disruption and merger events
        for(int p = 0; p < ngal; p++) {

            // satellite galaxy!
            if((galaxies[p].Type == 1 || galaxies[p].Type == 2) && galaxies[p].mergeType == 0) {
                XRETURN(galaxies[p].MergTime < 999.0,
                        EXIT_FAILURE,
                        "Error: galaxies[%d].MergTime = %lf is too large! Should have been within the age of the Universe\n",
                        p, galaxies[p].MergTime);

                const double deltaT = run_params->Age[galaxies[p].SnapNum] - halo_age;
                galaxies[p].MergTime -= deltaT / effective_steps;

                // only consider mergers or disruption for halo-to-baryonic mass ratios below the threshold
                // or for satellites with no baryonic mass (they don't grow and will otherwise hang around forever)
                double currentMvir = galaxies[p].Mvir - galaxies[p].deltaMvir * (1.0 - ((double)step + 1.0) / (double)effective_steps);
                double galaxyBaryons = galaxies[p].StellarMass + galaxies[p].ColdGas;
                if((galaxyBaryons == 0.0) || (galaxyBaryons > 0.0 && (currentMvir / galaxyBaryons <= run_params->ThresholdSatDisruption))) {

                    int merger_centralgal = galaxies[p].Type==1 ? centralgal:galaxies[p].CentralGal;

                    if(galaxies[merger_centralgal].mergeType > 0) {
                        merger_centralgal = galaxies[merger_centralgal].CentralGal;
                    }

                    galaxies[p].mergeIntoID = *numgals + merger_centralgal;  // position in output

                    if(isfinite(galaxies[p].MergTime)) {
                        // Time at which this event occurs (same formula used for mergers)
                        const double event_time = run_params->Age[galaxies[p].SnapNum] - (step + 0.5) * (deltaT / effective_steps);
                        // disruption has occurred!
                        if(galaxies[p].MergTime > 0.0) {
                            disrupt_satellite_to_ICS(merger_centralgal, p, event_time, galaxies, run_params);
                        } else {
                            // a merger has occurred!
                            // Map adaptive step to fixed STEPS bins for SFR arrays
                            int step_bin = (step * STEPS) / effective_steps;
                            if(step_bin >= STEPS) step_bin = STEPS - 1;
                            deal_with_galaxy_merger(p, merger_centralgal, centralgal, event_time, deltaT / effective_steps, halonr, step_bin, galaxies, run_params);
                        }
                    }
                }

            }
        }
    } // Go on to the next STEPS substep

    // Extra miscellaneous stuff before finishing this halo
    const double deltaT = run_params->Age[galaxies[0].SnapNum] - halo_age;
    const double inv_deltaT = 1.0/deltaT;

    for(int p = 0; p < ngal; p++) {

        // Don't bother with galaxies that have already merged
        if(galaxies[p].mergeType > 0) {
            continue;
        }

        galaxies[p].Cooling *= inv_deltaT;
        galaxies[p].Heating *= inv_deltaT;
        galaxies[p].OutflowRate *= inv_deltaT;
    }


    // Attach final galaxy list to halo
    for(int p = 0, currenthalo = -1; p < ngal; p++) {
        if(galaxies[p].HaloNr != currenthalo) {
            currenthalo = galaxies[p].HaloNr;
            haloaux[currenthalo].FirstGalaxy = *numgals;
            haloaux[currenthalo].NGalaxies = 0;
        }

        // Merged galaxies won't be output. So go back through its history and find it
        // in the previous timestep. Then copy the current merger info there.
        /* mergeIntoID was stored as a raw output-array index including all galaxies.
         * Merged galaxies are not written to output, so every preceding merged galaxy
         * whose mergeIntoID < galaxies[p].mergeIntoID would occupy a slot that is now
         * absent -- shift the target index down by one for each such gap. */
        int offset = 0;
        int i = p-1;
        while(i >= 0) {
            if(galaxies[i].mergeType > 0) {
                if(galaxies[p].mergeIntoID > galaxies[i].mergeIntoID) {
                    offset++;
                }
            }

            i--;
        }

        i = -1;
        if(galaxies[p].mergeType > 0) {
            i = haloaux[currenthalo].FirstGalaxy - 1;
            while(i >= 0) {
                if(halogal[i].GalaxyNr == galaxies[p].GalaxyNr) {
                    break;
                }

                i--;
            }

            XRETURN(i >= 0, EXIT_FAILURE, "Error: This should not happen - i=%d should be >=0", i);

            halogal[i].mergeType = galaxies[p].mergeType;
            halogal[i].mergeIntoID = galaxies[p].mergeIntoID - offset;
            halogal[i].mergeIntoSnapNum = halos[currenthalo].SnapNum;
        }

        if(galaxies[p].mergeType == 0) {
            /* realloc if needed */
            if(*numgals == (*maxgals - 1)) {
                *maxgals += 10000;

                *ptr_to_galaxies = myrealloc(*ptr_to_galaxies, *maxgals * sizeof(struct GALAXY));
                *ptr_to_halogal  = myrealloc(*ptr_to_halogal, *maxgals * sizeof(struct GALAXY));
                galaxies = *ptr_to_galaxies;
                halogal = *ptr_to_halogal;
            }

            XRETURN(*numgals < *maxgals, INVALID_MEMORY_ACCESS_REQUESTED,
                    "Error: numgals = %d exceeds the number of galaxies allocated = %d\n"
                    "This would result in invalid memory access...exiting\n",
                    *numgals, *maxgals);

            galaxies[p].SnapNum = halos[currenthalo].SnapNum;
            halogal[*numgals] = galaxies[p];
            (*numgals)++;
            haloaux[currenthalo].NGalaxies++;
        }
    }

    return EXIT_SUCCESS;
}

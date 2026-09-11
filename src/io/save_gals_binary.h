/*
 * save_gals_binary.h -- public interface for the binary galaxy catalogue writer.
 *
 * Defines GALAXY_OUTPUT (the on-disk fixed-size struct written per galaxy in
 * binary output mode) and declares the three entry points -- initialize, save,
 * finalize -- called by core_save.c when OutputFormat is sage_binary.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif /* working with c++ compiler */

#include "../core_allvars.h"

    struct GALAXY_OUTPUT
    {
      int   SnapNum;

      int Type;

      long long   GalaxyIndex;
      long long   CentralGalaxyIndex;
      int   SAGEHaloIndex;
      int   SAGETreeIndex;
      long long   SimulationHaloIndex;

      int   mergeType;  /* 0=none; 1=minor merger; 2=major merger; 3=disk instability; 4=disrupt to ICS */
      int   mergeIntoID;
      int   mergeIntoSnapNum;
      float dT;

      /* (sub)halo properties */
      float Pos[3];
      float Vel[3];
      float Spin[3];
      int   Len;
      float Mvir;
      float CentralMvir;
      float Rvir;
      float Vvir;
      float VvirPeak;
      float Vmax;
      float VelDisp;

      /* baryonic reservoirs */
      float ColdGas;
      float StellarMass;
      float BulgeMass;
      float HotGas;
      float EjectedMass;
      float BlackHoleMass;
      float ICS;
      float ICS_disrupt;
      float ICS_accrete;
      float ICS_sum_mt;
      float H2gas;
      float H1gas;

      /* metals */
      float MetalsColdGas;
      float MetalsStellarMass;
      float MetalsBulgeMass;
      float MetalsHotGas;
      float MetalsEjectedMass;
      float MetalsICS;
      float MassLoading;

      /* to calculate magnitudes */
      float SfrDisk;
      float SfrBulge;
      float SfrDiskZ;
      float SfrBulgeZ;

      /* misc */
      float DiskScaleRadius;
      float BulgeRadius;
      float MergerBulgeRadius;
      float InstabilityBulgeRadius;
      float MergerBulgeMass;
      float InstabilityBulgeMass;
      float Cooling;
      float Heating;
      float QuasarModeBHaccretionMass;
      float TimeOfLastMajorMerger;
      float TimeOfLastMinorMerger;
      float OutflowRate;

      /* infall properties */
      float infallMvir;
      float infallVvir;
      float infallVmax;
      float infallStellarMass;
      float TimeOfInfall;

      /* CGM properties */
      int Regime;
      float CGMgas;
      float MetalsCGMgas;
      float tcool;
      float tff;
      float tcool_over_tff;
      float MachNumber;
      float tdeplete;
      float H2DepletionTime_Gyr;
      float RcoolToRvir;

      int FFBRegime;
      float Concentration;
      float mdot_cool;
      float mdot_stream;
      double g_max;
      float r_heat;          /* AGN radio-mode heating radius [Mpc/h], capped at Rvir in the CGM regime */
    };

    /* Proto-Types */
    extern int32_t initialize_binary_galaxy_files(const int filenr, const struct forest_info *forest_info,
                                                  struct save_info *save_info,
                                                  const struct params *run_params);

    extern int32_t save_binary_galaxies(const int32_t task_treenr, const int32_t num_gals,
                                        const int32_t *OutputGalCount, struct forest_info *forest_info,
                                        struct halo_data *halos, struct halo_aux_data *haloaux,
                                        struct GALAXY *halogal, struct save_info *save_info, const struct params *run_params);

    extern int32_t finalize_binary_galaxy_files(const struct forest_info *forest_info,
                                                struct save_info *save_info,
                                                const struct params *run_params);
#ifdef __cplusplus
}
#endif

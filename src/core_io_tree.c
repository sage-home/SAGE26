/*
 * core_io_tree.c -- tree format dispatch layer.
 *
 * Provides the three lifecycle entry points used by sage.c to interact with
 * merger tree data regardless of on-disk format.  setup_forests_io() selects
 * the appropriate format-specific setup routine based on run_params->TreeType
 * and validates the GalaxyIndex multiplier fields.  load_forest() dispatches a
 * single-forest read to the correct reader.  cleanup_forests_io() calls the
 * format cleanup and frees shared arrays (FileNr, original_treenr).
 * Supported TreeTypes: lhalo_binary, mimic_lhalo_binary, lhalo_hdf5,
 * consistent_trees_ascii, consistent_trees_hdf5, genesis_hdf5, gadget4_hdf5.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

#include "core_allvars.h"
#include "core_mymalloc.h"
#include "core_io_tree.h"

#include "io/read_tree_lhalo_binary.h"
#include "io/read_tree_mimic_lhalo_binary.h"
#include "io/read_tree_consistentrees_ascii.h"

#ifdef HDF5
#include "io/read_tree_lhalo_hdf5.h"
#include "io/read_tree_genesis_hdf5.h"
#include "io/read_tree_consistentrees_hdf5.h"
#include "io/read_tree_gadget4_hdf5.h"
#endif

/*
 * setup_forests_io -- initialise the tree I/O layer for the requested format.
 *
 * Dispatches to the format-specific setup routine, then validates that
 * FileNr_Mulfac, ForestNr_Mulfac, and frac_volume_processed were set to
 * valid values by the reader.  Returns EXIT_SUCCESS or a negative SAGE error
 * code on failure.
 */
int setup_forests_io(struct params *run_params, struct forest_info *forests_info,
                     const int ThisTask, const int NTasks)
{
    int status = EXIT_FAILURE;/* initialize to FAIL  */
    forests_info->firstfile = run_params->FirstFile;
    forests_info->lastfile = run_params->LastFile;
    const enum Valid_TreeTypes TreeType = run_params->TreeType;

    /* MS: 21/9/2019 initialise the mulfac's so we can check later
              that these vital factors (required to generate unique galaxy ID's)
              have been setup appropriately  */
    run_params->FileNr_Mulfac = -1;
    run_params->ForestNr_Mulfac = -1;
    forests_info->frac_volume_processed = -1.0;

    switch (TreeType)
        {
#ifdef HDF5
        case lhalo_hdf5:
            //MS: 22/07/2021 - Why is firstfile, lastfile still passed even though those could be constructed
            //from run_params (like done within this __FUNCTION__)
            status = setup_forests_io_lht_hdf5(forests_info, ThisTask, NTasks, run_params);
            break;

        case gadget4_hdf5:
            status = setup_forests_io_gadget4_hdf5(forests_info, ThisTask, NTasks, run_params);
            break;

        case genesis_hdf5:
            status = setup_forests_io_genesis_hdf5(forests_info, ThisTask, NTasks, run_params);
            break;

        case consistent_trees_hdf5:
            status = setup_forests_io_ctrees_hdf5(forests_info, ThisTask, NTasks, run_params);
            break;
#endif

        case lhalo_binary:
            status = setup_forests_io_lht_binary(forests_info, ThisTask, NTasks, run_params);
            break;

        case mimic_lhalo_binary:
            status = setup_forests_io_mimic_lhalo_binary(forests_info, ThisTask, NTasks, run_params);
            break;

        case consistent_trees_ascii:
            status = setup_forests_io_ctrees(forests_info, ThisTask, NTasks, run_params);
            break;

        default:
            fprintf(stderr, "Your tree type has not been included in the switch statement for function ``%s`` in file ``%s``.\n", __FUNCTION__, __FILE__);
            fprintf(stderr, "Please add it there.\n");
            return INVALID_OPTION_IN_PARAMS;
        }

    if(status != EXIT_SUCCESS) {
        return status;
    }

    /*MS: Check that the mechanism to generate unique GalaxyID's was
      initialised correctly in the setup */
    if(run_params->FileNr_Mulfac < 0 || run_params->ForestNr_Mulfac < 0) {
        fprintf(stderr,"Error: Looks like the multiplicative factors to generate unique "
                       "galaxyID's were not setup correctly.\n"
                       "FileNr_Mulfac = %"PRId64" and ForestNr_Mulfac = %"PRId64" should both be >=0\n",
                       run_params->FileNr_Mulfac, run_params->ForestNr_Mulfac);
        return -1;
    }

    if(forests_info->frac_volume_processed < 0.0 || forests_info->frac_volume_processed > 1.0) {
        fprintf(stderr,"Error: The fraction of the entire simulation volume processed should be in [0.0, 1.0]. Instead, found %g\n",
                forests_info->frac_volume_processed);
        return -1;
    }


    return status;
}


/*
 * cleanup_forests_io -- release all tree I/O resources after the main loop.
 *
 * Dispatches to the format-specific cleanup, then frees forests_info->FileNr
 * and forests_info->original_treenr which are shared across all formats.
 * Must be called only after all forests have been processed.
 */
void cleanup_forests_io(enum Valid_TreeTypes TreeType, struct forest_info *forests_info)
{
    /* Don't forget to free the open file handle */
    switch (TreeType) {
#ifdef HDF5
    case lhalo_hdf5:
        cleanup_forests_io_lht_hdf5(forests_info);
        break;

    case gadget4_hdf5:
        cleanup_forests_io_gadget4_hdf5(forests_info);
        break;

    case genesis_hdf5:
        cleanup_forests_io_genesis_hdf5(forests_info);
        break;

    case consistent_trees_hdf5:
        cleanup_forests_io_ctrees_hdf5(forests_info);
        break;

#endif

    case lhalo_binary:
        cleanup_forests_io_lht_binary(forests_info);
        break;

    case mimic_lhalo_binary:
        cleanup_forests_io_mimic_lhalo_binary(forests_info);
        break;

    case consistent_trees_ascii:

        /* because consistent trees can only be cleaned up after *ALL* forests
           have been processed (and not on a `per file` basis)
         */
        cleanup_forests_io_ctrees(forests_info);
        break;

    default:
        fprintf(stderr, "Your tree type has not been included in the switch statement for function ``%s`` in file ``%s``.\n", __FUNCTION__, __FILE__);
        fprintf(stderr, "Please add it there.\n");
        ABORT(EXIT_FAILURE);

    }

    // Finally, things that are common across forest types.
    free(forests_info->FileNr);
    free(forests_info->original_treenr);

    return;
}

/*
 * load_forest -- load one forest from disk into a freshly allocated halos array.
 *
 * Dispatches to the format-specific load routine.  Returns the number of halos
 * in the forest (>= 0) or a negative SAGE error code on failure.
 */
int64_t load_forest(struct params *run_params, const int64_t forestnr, struct halo_data **halos, struct forest_info *forests_info)
{

    int64_t nhalos;
    const enum Valid_TreeTypes TreeType = run_params->TreeType;

    switch (TreeType) {

#ifdef HDF5
    case lhalo_hdf5:
        nhalos = load_forest_lht_hdf5(forestnr, halos, forests_info);
        break;

    case gadget4_hdf5:
        nhalos = load_forest_gadget4_hdf5(forestnr, halos, forests_info);
        break;

    case genesis_hdf5:
        nhalos = load_forest_genesis_hdf5(forestnr, halos, forests_info, run_params);
        break;

    case consistent_trees_hdf5:
        nhalos = load_forest_ctrees_hdf5(forestnr, halos, forests_info, run_params);
        break;

#endif

    case lhalo_binary:
        nhalos = load_forest_lht_binary(forestnr, halos, forests_info);
        break;

    case mimic_lhalo_binary:
        nhalos = load_forest_mimic_lhalo_binary(forestnr, halos, forests_info);
        break;

    case consistent_trees_ascii:
        nhalos = load_forest_ctrees(forestnr, halos, forests_info, run_params);
        break;

    default:
        fprintf(stderr, "Your tree type has not been included in the switch statement for ``%s`` in ``%s``.\n",
                __FUNCTION__, __FILE__);
        fprintf(stderr, "Please add it there.\n");
        return -EXIT_FAILURE;
    }

    return nhalos;
}

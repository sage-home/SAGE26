/*
 * core_io_tree.h -- public interface for the tree I/O dispatch layer.
 *
 * Declares setup_forests_io(), load_forest(), and cleanup_forests_io(), which
 * dispatch to the correct format-specific reader (lhalo binary/HDF5, Mimic
 * lhalo binary, Consistent Trees, Genesis, Gadget-4) based on the TreeType
 * parameter.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

    #include "core_allvars.h"

    /* core_io_tree.c */
    extern int setup_forests_io(struct params *run_params, struct forest_info *forests_info,
                                const int ThisTask, const int NTasks);
    extern int64_t load_forest(struct params *run_params, const int64_t forestnr, struct halo_data **halos, struct forest_info *forests_info);
    extern void cleanup_forests_io(enum Valid_TreeTypes my_TreeType, struct forest_info *forests_info);

#ifdef __cplusplus
}
#endif

/*
 * read_tree_lhalo_binary.h -- public interface for the lhalo-binary tree reader.
 *
 * Declares the three entry points used by core_io_tree.c to load merger trees
 * in the lhalo-binary on-disk format: setup (per process, at startup), load
 * (per forest, during the main loop), cleanup (at shutdown). Implementation
 * lives in read_tree_lhalo_binary.c.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif /* working with c++ compiler */

#include "../core_allvars.h"

    /* Proto-Types */
    extern int setup_forests_io_lht_binary(struct forest_info *forests_info,
                                           const int ThisTask, const int NTasks, struct params *run_params);
    extern int setup_forests_io_lht_binary_recsize(struct forest_info *forests_info,
                                                   const int ThisTask, const int NTasks, struct params *run_params,
                                                   const size_t record_size);
    extern int64_t load_forest_lht_binary(const int64_t forestnr, struct halo_data **halos, struct forest_info *forests_info);
    extern void cleanup_forests_io_lht_binary(struct forest_info *forests_info);

#ifdef __cplusplus
}
#endif

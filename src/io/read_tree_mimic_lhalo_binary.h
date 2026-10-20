/*
 * read_tree_mimic_lhalo_binary.h -- public interface for the Mimic lhalo-binary tree reader.
 *
 * Declares the three entry points used by core_io_tree.c to load merger trees
 * written by Mimic's tree converter: an lhalo-binary header followed by
 * Mimic's 88-byte halo record. Implementation lives in
 * read_tree_mimic_lhalo_binary.c.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif /* working with c++ compiler */

#include "../core_allvars.h"

    /* Bytes per halo record on disk. */
#define MIMIC_LHALO_RECORD_SIZE 88

    /* Proto-Types */
    extern int setup_forests_io_mimic_lhalo_binary(struct forest_info *forests_info,
                                                   const int ThisTask, const int NTasks, struct params *run_params);
    extern int64_t load_forest_mimic_lhalo_binary(const int64_t forestnr, struct halo_data **halos, struct forest_info *forests_info);
    extern void cleanup_forests_io_mimic_lhalo_binary(struct forest_info *forests_info);

#ifdef __cplusplus
}
#endif

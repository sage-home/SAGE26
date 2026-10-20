/*
 * read_tree_mimic_lhalo_binary.c -- reader for the lhalo-binary trees written by Mimic.
 *
 * Mimic's tree converter (write_vertical_trees.py) writes merger trees as
 * lhalo-binary files: the classic header (totnforests, totnhalos, then one
 * int32 halo count per forest) followed by every forest's halo records back
 * to back, with the same five forest-local link indices that SAGE's own
 * lhalo-binary format uses. The halo record is different, though. It is the
 * 88-byte layout of Mimic's halo_properties.yaml rather than SAGE's 104-byte
 * struct halo_data, and it differs in three ways that matter to the model:
 *
 *   - M_Crit200 is stored in Msun/h, not SAGE's 10^10 Msun/h, and it is the
 *     only mass present. It becomes halo_data.Mvir (the M200c union member).
 *   - Spin is already the specific angular momentum J/Mvir, in
 *     (Mpc/h)(km/s), so it is copied without the division by mass that the
 *     Consistent-Trees readers apply to raw J.
 *   - M_Mean200, M_TopHat, SubhaloIndex and SubHalfMass are absent. The model
 *     never reads them. SubhaloIndex and SubHalfMass take the -1 that SAGE26's
 *     HDF5 readers use when they have no value; the two masses take -1 as a
 *     "not available" marker rather than a copy of Mvir, so that an M200c
 *     value is never labelled as M200b or as a top-hat mass. The halo's
 *     FileNr is set to the file being read.
 *
 * The header, the forest distribution across tasks, the byte offset of each
 * forest and the GalaxyIndex multipliers depend only on the header and the
 * record size, so setup and cleanup are shared with read_tree_lhalo_binary.c
 * through setup_forests_io_lht_binary_recsize(). Only the per-forest load is
 * specific to this format: it reads the raw records in bounded chunks and
 * converts each one into a struct halo_data.
 *
 * Every value on disk is little-endian and is read into host-order fields
 * without swapping, so setup refuses to run on a big-endian host.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>

#include "read_tree_mimic_lhalo_binary.h"
#include "read_tree_lhalo_binary.h"
#include "../core_mymalloc.h"
#include "../core_utils.h"

/*
 * The on-disk halo record. Field order, types and offsets follow Mimic's
 * halo_properties.yaml for the shin-uchuu-lhalo package (layout digest
 * ab04dcbae52ffd11b1e773fe3c9aafecc22371f4583eeadf2353930e5bdf5e64). With
 * natural alignment the int64 MostBoundID sits at offset 72, SnapNum at 80,
 * and four bytes of padding bring the record to 88, exactly as written, so a
 * record can be read straight into this struct on a little-endian host.
 */
struct mimic_lhalo_record {
    int32_t Descendant;          /* forest-local index of the descendant; -1 if none */
    int32_t FirstProgenitor;     /* forest-local index of the main progenitor; -1 if none */
    int32_t NextProgenitor;      /* forest-local index of the next progenitor of the same descendant; -1 if none */
    int32_t FirstHaloInFOFgroup; /* forest-local index of the FoF central; itself for a central; never -1 */
    int32_t NextHaloInFOFgroup;  /* forest-local index of the next FoF member; -1 if last */
    int32_t Len;                 /* particles: round(Mvir / particle mass) */
    float M_Crit200;             /* Msun/h; the Rockstar/Consistent-Trees Mvir (M200c) */
    float Pos[3];                /* Mpc/h, comoving */
    float Vel[3];                /* km/s, peculiar */
    float Spin[3];               /* (Mpc/h)(km/s); already the specific angular momentum J/Mvir */
    float VelDisp;               /* km/s */
    float Vmax;                  /* km/s */
    int64_t MostBoundID;         /* Consistent-Trees halo id; strictly positive, unique in the catalogue */
    int32_t SnapNum;             /* snapshot number */
    int32_t padding;             /* alignment padding; carries no meaning */
};

/* The struct above must match the on-disk layout byte for byte. */
BUILD_BUG_OR_ZERO(sizeof(struct mimic_lhalo_record) == MIMIC_LHALO_RECORD_SIZE, mimic_lhalo_record_is_not_88_bytes);
BUILD_BUG_OR_ZERO(offsetof(struct mimic_lhalo_record, M_Crit200) == 24, mimic_lhalo_record_M_Crit200_offset);
BUILD_BUG_OR_ZERO(offsetof(struct mimic_lhalo_record, Spin) == 52, mimic_lhalo_record_Spin_offset);
BUILD_BUG_OR_ZERO(offsetof(struct mimic_lhalo_record, MostBoundID) == 72, mimic_lhalo_record_MostBoundID_offset);
BUILD_BUG_OR_ZERO(offsetof(struct mimic_lhalo_record, SnapNum) == 80, mimic_lhalo_record_SnapNum_offset);

/* Msun/h on disk to SAGE26's 10^10 Msun/h code unit. */
#define MIMIC_LHALO_MASS_TO_CODE_UNITS 1.0e-10

/* Records converted per read. Bounds the temporary buffer at
 * MIMIC_LHALO_RECORDS_PER_READ * 88 bytes (about 92 MB) whatever the forest
 * size, so a forest of 10^8 halos does not need a second copy of itself.
 * Overridable so a test can force several chunks on a small forest. */
#ifndef MIMIC_LHALO_RECORDS_PER_READ
#define MIMIC_LHALO_RECORDS_PER_READ (1024L * 1024L)
#endif

/*
 * host_is_little_endian -- true when the host stores the low byte of an
 * integer first. The on-disk values are little-endian and are not swapped.
 */
static int host_is_little_endian(void)
{
    const uint32_t probe = 1;
    uint8_t first_byte;
    memcpy(&first_byte, &probe, 1);
    return first_byte == 1;
}

/*
 * convert_record -- fill one struct halo_data from one on-disk record.
 *
 * Links, Len, positions, velocities, Spin, VelDisp, Vmax, MostBoundID and
 * SnapNum are copied unchanged. M_Crit200 is scaled from Msun/h to code
 * units in double and stored as float32. The four fields the format lacks
 * are set to -1 (see the file header); FileNr is the file the forest came
 * from, which the setup pass records per forest.
 */
static inline void convert_record(const struct mimic_lhalo_record *rec, struct halo_data *halo, const int filenr)
{
    halo->Descendant = rec->Descendant;
    halo->FirstProgenitor = rec->FirstProgenitor;
    halo->NextProgenitor = rec->NextProgenitor;
    halo->FirstHaloInFOFgroup = rec->FirstHaloInFOFgroup;
    halo->NextHaloInFOFgroup = rec->NextHaloInFOFgroup;
    halo->Len = rec->Len;

    halo->M_Mean200 = -1.0f;
    halo->Mvir = (float) ((double) rec->M_Crit200 * MIMIC_LHALO_MASS_TO_CODE_UNITS);
    halo->M_TopHat = -1.0f;

    for(int k = 0; k < 3; k++) {
        halo->Pos[k] = rec->Pos[k];
        halo->Vel[k] = rec->Vel[k];
        halo->Spin[k] = rec->Spin[k];
    }
    halo->VelDisp = rec->VelDisp;
    halo->Vmax = rec->Vmax;
    halo->MostBoundID = rec->MostBoundID;
    halo->SnapNum = rec->SnapNum;
    halo->FileNr = filenr;
    halo->SubhaloIndex = -1;
    halo->SubHalfMass = -1.0f;
}

/*
 * setup_forests_io_mimic_lhalo_binary -- per-process startup for the reader.
 *
 * Refuses a big-endian host, then runs the shared lhalo-binary setup pass with
 * the 88-byte record size. That pass reads every file's header, distributes
 * the forests over tasks, records each forest's file and byte offset, checks
 * that each opened file is exactly header plus one record per halo, and sets
 * the GalaxyIndex multipliers (FileNr_Mulfac 10^15, ForestNr_Mulfac 10^9).
 *
 * Returns EXIT_SUCCESS or the shared setup's error code.
 */
int setup_forests_io_mimic_lhalo_binary(struct forest_info *forests_info,
                                        const int ThisTask, const int NTasks, struct params *run_params)
{
    XRETURN(host_is_little_endian(), EXIT_FAILURE,
            "Error: TreeType mimic_lhalo_binary stores every value little-endian and this reader does not swap bytes, "
            "but this host is big-endian.\n");

    return setup_forests_io_lht_binary_recsize(forests_info, ThisTask, NTasks, run_params, MIMIC_LHALO_RECORD_SIZE);
}

/*
 * load_forest_mimic_lhalo_binary -- read one forest and convert it to struct halo_data.
 *
 * Allocates the forest's halo_data array, then reads the forest's records
 * from the byte offset the setup pass recorded, at most
 * MIMIC_LHALO_RECORDS_PER_READ at a time into a temporary buffer, converting
 * each chunk as it arrives. Both allocations go through the tracked
 * allocator, so the run's reported peak memory includes the buffer. It is
 * freed before returning, so only the halo_data array is live when the
 * caller allocates its galaxy arrays.
 *
 * Inputs:
 *   forestnr      -- local forest index in [0, nforests_this_task).
 *   halos         -- output: set to the newly allocated halo array, which the
 *                    caller owns and must myfree().
 *   forests_info  -- populated by setup_forests_io_mimic_lhalo_binary().
 *
 * Returns the number of halos in the forest, or a negative error code:
 *   -INVALID_MEMORY_ACCESS_REQUESTED (forestnr out of range),
 *   -INVALID_FILE_POINTER (no open descriptor) or -FILE_READ_ERROR (bad
 *   offset). An allocation failure aborts inside mymalloc().
 */
int64_t load_forest_mimic_lhalo_binary(const int64_t forestnr, struct halo_data **halos, struct forest_info *forests_info)
{
    const struct lhalotree_info *lht = &(forests_info->lht);

    XRETURN(forestnr >= 0 && forestnr < lht->nforests, -INVALID_MEMORY_ACCESS_REQUESTED,
            "Error: Attempting to access forest = %"PRId64" but memory is allocated for only %"PRId64" forests\n",
            forestnr, lht->nforests);

    const int64_t nhalos = lht->nhalos_per_forest[forestnr];
    const int fd = lht->fd[forestnr];
    XRETURN(fd > 0, -INVALID_FILE_POINTER,
            "Error: no open file descriptor for forest = %"PRId64"\n", forestnr);

    off_t offset = lht->bytes_offset_for_forest[forestnr];
    XRETURN(offset >= 0, -FILE_READ_ERROR,
            "Error: offset = %"PRId64" for forest = %"PRId64" must be at least 0\n", (int64_t) offset, forestnr);

    struct halo_data *local_halos = mymalloc(sizeof(*local_halos) * nhalos);

    const int64_t records_per_read = nhalos < MIMIC_LHALO_RECORDS_PER_READ ? nhalos : MIMIC_LHALO_RECORDS_PER_READ;
    struct mimic_lhalo_record *records = mymalloc(records_per_read * sizeof(*records));

    const int filenr = forests_info->FileNr[forestnr];
    for(int64_t start = 0; start < nhalos; start += records_per_read) {
        const int64_t nread = (nhalos - start) < records_per_read ? (nhalos - start) : records_per_read;
        const size_t nbytes = (size_t) nread * sizeof(*records);
        mypread(fd, records, nbytes, offset);
        offset += (off_t) nbytes;
        for(int64_t i = 0; i < nread; i++) {
            convert_record(&records[i], &local_halos[start + i], filenr);
        }
    }
    myfree(records);

    *halos = local_halos;
    return nhalos;
}

/*
 * cleanup_forests_io_mimic_lhalo_binary -- release per-process reader state.
 *
 * The bookkeeping is the shared lhalo-binary state, so this is the shared
 * cleanup: free the per-forest arrays and close the open descriptors.
 */
void cleanup_forests_io_mimic_lhalo_binary(struct forest_info *forests_info)
{
    cleanup_forests_io_lht_binary(forests_info);
}

/*
 * read_tree_lhalo_binary.c -- reader for the lhalo-binary merger-tree format.
 *
 * The lhalo-binary format stores one or more forests per file, written
 * sequentially in a packed on-disk layout. Each file begins with a small
 * header (totnforests, totnhalos, then an array of nhalos per forest),
 * followed by the halo records -- one fixed-size `struct halo_data` per
 * halo, written tree-major. Within each tree, halos are ordered such that
 * Descendant / FirstProgenitor / NextProgenitor / FirstHaloInFOFgroup
 * indices into the record array form a self-consistent merger tree.
 *
 * This file implements the three-call lifecycle that core_io_tree.c uses:
 *   1. setup_forests_io_lht_binary()  -- at startup, read every file's
 *      header, compute the total forest count, and assign forests to
 *      this MPI task using the parameter file's ForestDistributionScheme.
 *      Open each file once and keep it open for the duration of the run.
 *   2. load_forest_lht_binary()       -- during the main loop, called once
 *      per forest assigned to this task. Allocates a halo buffer and
 *      reads the forest's halos at the precomputed byte offset.
 *   3. cleanup_forests_io_lht_binary() -- at shutdown, free per-forest
 *      bookkeeping arrays and close the open file descriptors.
 *
 * The on-disk widths matter: nhalos-per-forest is stored as 32-bit ints
 * in the file but we promote to int64_t in memory (load_tree_table_lht_binary
 * does the per-file read-and-promote dance).
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <limits.h>
#include <sys/stat.h>

#include "read_tree_lhalo_binary.h"
#include "../core_mymalloc.h"
#include "../core_utils.h"
#include "forest_utils.h"


static void get_forests_filename_lht_binary(char *filename, const size_t len, const int filenr, const struct params *run_params);
static int load_tree_table_lht_binary(const int firstfile, const int lastfile, const int64_t *totnforests_per_file,
                                      const struct params *run_params, const int ThisTask,
                                      int64_t *nhalos_per_forest);

/*
 * get_forests_filename_lht_binary -- build a tree-file path from a file number.
 *
 * Composes "<SimulationDir>/<TreeName>.<filenr><TreeExtension>" into the
 * caller-supplied buffer. Used by both the setup pass (to read headers from
 * every file) and the per-forest load (the open file descriptor is cached in
 * lht->fd, so this is mainly a setup-time helper).
 *
 * Inputs:
 *   filename     -- destination buffer; must be large enough for the full path.
 *   len          -- buffer length; snprintf truncates safely at len-1.
 *   filenr       -- file index in [FirstFile, LastFile].
 *   run_params   -- provides SimulationDir, TreeName, TreeExtension.
 */
static void get_forests_filename_lht_binary(char *filename, const size_t len, const int filenr, const struct params *run_params)
{
    snprintf(filename, len - 1, "%s/%s.%d%s", run_params->SimulationDir, run_params->TreeName, filenr, run_params->TreeExtension);
}

/*
 * setup_forests_io_lht_binary -- per-process startup for the lhalo-binary reader.
 *
 * Called once per MPI task at startup. Walks every tree file in
 * [FirstFile, LastFile], reads its header to learn how many forests and halos
 * it contains, then assigns forests to this task using the parameter file's
 * ForestDistributionScheme. Opens each file containing this task's forests
 * once and stores the file descriptor for use by load_forest_lht_binary().
 *
 * Algorithm:
 *   1. Read each file's two-int header (totnforests, totnhalos) to learn the
 *      total forest count across all files.
 *   2. If the distribution scheme is anything other than uniform_in_forests,
 *      read every file's nhalos-per-forest array so the forest-to-task
 *      assignment can weight by forest size (some forests are 10x larger
 *      than others -- purely uniform assignment leaves tasks load-imbalanced).
 *   3. Call distribute_weighted_forests_over_ntasks() to compute this task's
 *      [start_forestnum, end_forestnum) range.
 *   4. Convert the global forest range into per-file information: which files
 *      this task touches, and where in each file the task's forests live.
 *   5. For each touched file: open it (keep it open for the run), read its
 *      nhalos-per-forest array, and compute the byte offset into the halo
 *      record stream for each of this task's forests.
 *   6. Compute frac_volume_processed -- the fraction of total simulation
 *      volume this task is responsible for. Each file is assumed to span
 *      the same volume; task-volume = sum over files of (this task's
 *      forests in file / total forests in file) / (number of files).
 *
 * On-disk header layout:
 *     bytes  0..3   int32  totnforests
 *     bytes  4..7   int32  totnhalos
 *     bytes  8..   int32  nhalos_per_forest[totnforests]
 *     after that:   packed halo records, struct halo_data, tree-major.
 *
 * Side effects:
 *   Mutates forests_info: fills in totnforests, totnhalos, nforests_this_task,
 *   FileNr (per forest), original_treenr (per forest), frac_volume_processed,
 *   and the lht sub-struct (fd-per-forest, byte offset per forest, open file
 *   descriptors, nhalos per forest). Also sets run_params->FileNr_Mulfac and
 *   run_params->ForestNr_Mulfac, used downstream to compose unique GalaxyIndex
 *   values across all files, trees, and tasks.
 *
 * Returns:
 *   EXIT_SUCCESS, or a negative error code (MALLOC_FAILURE, FILE_NOT_FOUND,
 *   or whatever distribute_weighted_forests_over_ntasks / find_start_and_end_filenum
 *   returned).
 */
int setup_forests_io_lht_binary(struct forest_info *forests_info,
                                const int ThisTask, const int NTasks, struct params *run_params)
{
    return setup_forests_io_lht_binary_recsize(forests_info, ThisTask, NTasks, run_params, sizeof(struct halo_data));
}

/*
 * setup_forests_io_lht_binary_recsize -- the setup pass for any lhalo-binary layout.
 *
 * Everything in setup_forests_io_lht_binary() except the size of one halo
 * record, which is a parameter here. The header pass, the forest
 * distribution, the per-forest FileNr / original_treenr bookkeeping and the
 * GalaxyIndex multipliers depend only on the header and the record count,
 * so a reader for a different record layout (read_tree_mimic_lhalo_binary.c)
 * shares this function and supplies its own load routine.
 *
 * Inputs:
 *   record_size -- bytes per halo record on disk. The byte offset to a forest
 *                  is the header size plus record_size times the halos before
 *                  it in the file.
 *
 * Checks:
 *   Each file this task opens must be exactly header + record_size * totnhalos
 *   bytes long. A mismatch means the file was not written with this record
 *   layout (most likely the parameter file names the wrong TreeType), and the
 *   forest offsets would all be wrong, so it is an error.
 */
int setup_forests_io_lht_binary_recsize(struct forest_info *forests_info,
                                        const int ThisTask, const int NTasks, struct params *run_params,
                                        const size_t record_size)
{
    const int firstfile = run_params->FirstFile;
    const int lastfile = run_params->LastFile;

    const int numfiles = lastfile - firstfile + 1;
    if(numfiles <= 0) {
        return -1;
    }

    /* wasteful to allocate for lastfile + 1 indices, rather than numfiles; but makes indexing easier */
    int64_t *totnforests_per_file = calloc(lastfile + 1, sizeof(totnforests_per_file[0]));
    int64_t *totnhalos_per_file = calloc(lastfile + 1, sizeof(totnhalos_per_file[0]));
    if(totnforests_per_file == NULL || totnhalos_per_file == NULL) {
        fprintf(stderr,"Error: Could not allocate memory to store the number of forests and halos in each file\n");
        perror(NULL);
        return MALLOC_FAILURE;
    }

    /* Step 1: read every file's two-int header to learn the total forest and
     * halo counts across all files. We open, read 8 bytes, close -- nothing
     * is kept open here (Step 5 reopens the files this task actually needs). */
    int64_t totnforests = 0, totnhalos = 0;
    for(int filenr=firstfile;filenr<=lastfile;filenr++) {
        char filename[4*MAX_STRING_LEN];
        get_forests_filename_lht_binary(filename, 4*MAX_STRING_LEN, filenr, run_params);
        int fd = open(filename, O_RDONLY);
        if(fd < 0) {
            fprintf(stderr, "Error: can't open file `%s'\n", filename);
            perror(NULL);
            free(totnforests_per_file);
            free(totnhalos_per_file);
            return FILE_NOT_FOUND;
        }
        int tmp;
        mypread(fd, &tmp, sizeof(tmp), 0);
        totnforests_per_file[filenr] = tmp;
        totnforests += totnforests_per_file[filenr];

        mypread(fd, &tmp, sizeof(tmp), 4);
        totnhalos_per_file[filenr] = tmp;
        totnhalos += tmp;
        close(fd);
    }
    forests_info->totnforests = totnforests;
    forests_info->totnhalos = totnhalos;

    /* Step 2: if the distribution scheme weights by forest size, read the
     * per-file nhalos-per-forest array so distribute_weighted_forests can
     * balance load. uniform_in_forests skips this -- every forest costs the
     * same, so we don't need the sizes at all. */
    const int need_nhalos_per_forest = run_params->ForestDistributionScheme == uniform_in_forests ? 0:1;
    int64_t *nhalos_per_forest = NULL;
    if(need_nhalos_per_forest) {
        nhalos_per_forest = mycalloc(totnforests, sizeof(*nhalos_per_forest));
        int status = load_tree_table_lht_binary(firstfile, lastfile, totnforests_per_file, run_params, ThisTask, nhalos_per_forest);
        if(status != EXIT_SUCCESS) {
            return status;
        }
    }

    /* Step 3: assign a [start_forestnum, end_forestnum) global range of
     * forests to this task. The weighted-distribution helper handles all
     * the scheme variants (uniform, linear-in-nhalos, power-in-nhalos, ...). */
    int64_t nforests_this_task, start_forestnum;
    int status = distribute_weighted_forests_over_ntasks(totnforests, nhalos_per_forest,
                                                         run_params->ForestDistributionScheme, run_params->Exponent_Forest_Dist_Scheme,
                                                         NTasks, ThisTask, &nforests_this_task, &start_forestnum);
    if(status != EXIT_SUCCESS) {
        return status;
    }
    if(need_nhalos_per_forest) {
        myfree(nhalos_per_forest);
    }

    const int64_t end_forestnum = start_forestnum + nforests_this_task; /* not inclusive, i.e., do not process forestnr == end_forestnum */

    // Now that we know the number of trees being processed by each task, let's set up and malloc the structs.
    struct lhalotree_info *lht = &(forests_info->lht);
    forests_info->nforests_this_task = nforests_this_task;

    forests_info->FileNr = malloc(nforests_this_task * sizeof(*(forests_info->FileNr)));
    CHECK_POINTER_AND_RETURN_ON_NULL(forests_info->FileNr,
                                     "Failed to allocate %"PRId64" elements of size %zu for forests_info->FileNr", nforests_this_task,
                                     sizeof(*(forests_info->FileNr)));

    forests_info->original_treenr = malloc(nforests_this_task * sizeof(*(forests_info->original_treenr)));
    CHECK_POINTER_AND_RETURN_ON_NULL(forests_info->original_treenr,
                                     "Failed to allocate %"PRId64" elements of size %zu for forests_info->original_treenr", nforests_this_task,
                                     sizeof(*(forests_info->original_treenr)));

    lht->nforests = nforests_this_task;
    lht->nhalos_per_forest = mymalloc(nforests_this_task * sizeof(lht->nhalos_per_forest[0]));
    lht->bytes_offset_for_forest = mymalloc(nforests_this_task * sizeof(lht->bytes_offset_for_forest[0]));
    lht->fd = mymalloc(nforests_this_task * sizeof(lht->fd[0]));

    int64_t *num_forests_to_process_per_file = calloc(lastfile + 1, sizeof(num_forests_to_process_per_file[0]));/* calloc is required */
    int64_t *start_forestnum_to_process_per_file = malloc((lastfile + 1) * sizeof(start_forestnum_to_process_per_file[0]));
    if(num_forests_to_process_per_file == NULL || start_forestnum_to_process_per_file == NULL) {
        fprintf(stderr,"Error: Could not allocate memory to store the number of forests that need to be processed per file (on thistask=%d)\n", ThisTask);
        perror(NULL);
        return MALLOC_FAILURE;
    }

    /* Step 4: convert the global forest range into per-file information.
     * find_start_and_end_filenum() fills num_forests_to_process_per_file[]
     * and start_forestnum_to_process_per_file[] so we know which files
     * this task touches and where in each file its forests live. */
    int start_filenum = -1, end_filenum = -1;
    status = find_start_and_end_filenum(start_forestnum, end_forestnum,
                                        totnforests_per_file, totnforests,
                                        firstfile, lastfile,
                                        ThisTask, NTasks,
                                        num_forests_to_process_per_file, start_forestnum_to_process_per_file,
                                        &start_filenum, &end_filenum);
    if(status != EXIT_SUCCESS) {
        return status;
    }


    lht->numfiles = end_filenum - start_filenum + 1;
    lht->open_fds = mymalloc(lht->numfiles * sizeof(lht->open_fds[0]));

    /* Step 5: for each file this task touches, open it (keep it open for
     * the run), read its full nhalos-per-forest array into a temp buffer,
     * and compute the byte offset to each of this task's forests in the
     * packed halo record stream. The offsets cached here are what
     * load_forest_lht_binary() will use later to pread the halos. */
    int64_t *forestnhalos = lht->nhalos_per_forest;
    int64_t nforests_so_far = 0;
    for(int filenr=start_filenum;filenr<=end_filenum;filenr++) {
        XRETURN(start_forestnum_to_process_per_file[filenr] >= 0 && start_forestnum_to_process_per_file[filenr] < totnforests_per_file[filenr],
                EXIT_FAILURE,
                "Error: Num forests to process = %"PRId64" for filenr = %d should be in range [0, %"PRId64")\n",
                start_forestnum_to_process_per_file[filenr],
                filenr,
                totnforests_per_file[filenr]);

        XRETURN(num_forests_to_process_per_file[filenr] >= 0 && num_forests_to_process_per_file[filenr] <= totnforests_per_file[filenr],
                EXIT_FAILURE,
                "Error: Num forests to process = %"PRId64" for filenr = %d should be in range [0, %"PRId64")\n",
                num_forests_to_process_per_file[filenr],
                filenr,
                totnforests_per_file[filenr]);

        int file_index = filenr - start_filenum;
        char filename[4*MAX_STRING_LEN];
        get_forests_filename_lht_binary(filename, 4*MAX_STRING_LEN, filenr, run_params);
        int fd = open(filename, O_RDONLY);
        XRETURN(fd > 0, FILE_NOT_FOUND,
                "Error: can't open file `%s'\n", filename);
        lht->open_fds[file_index] = fd;/* keep the file open, will be closed at the cleanup stage */

        const int64_t nforests_to_process_this_file = num_forests_to_process_per_file[filenr];
        const size_t nbytes = totnforests_per_file[filenr] * sizeof(int32_t);

        /* The file must hold exactly the header plus one record per halo. Anything
         * else means these files were not written with this record layout, and
         * every forest offset computed below would be wrong. */
        struct stat file_stat;
        XRETURN(fstat(fd, &file_stat) == 0, FILE_READ_ERROR,
                "Error: could not stat file `%s'\n", filename);
        const int64_t expected_bytes = (int64_t) (2 * sizeof(int32_t) + nbytes) + totnhalos_per_file[filenr] * (int64_t) record_size;
        XRETURN((int64_t) file_stat.st_size == expected_bytes, EXIT_FAILURE,
                "Error: file `%s' is %"PRId64" bytes, but its header (%"PRId64" forests, %"PRId64" halos) "
                "with %zu-byte halo records implies %"PRId64" bytes.\n"
                "The file was not written in the record layout that TreeType selects. Check TreeType.\n",
                filename, (int64_t) file_stat.st_size, totnforests_per_file[filenr], totnhalos_per_file[filenr],
                record_size, expected_bytes);
        int32_t *buffer = malloc(nbytes);
        XRETURN(buffer != NULL, MALLOC_FAILURE,
                "Error: Could not allocate memory to read nhalos per forest. Bytes requested = %zu\n", nbytes);

        mypread(fd, buffer, nbytes, 8); /* the last argument says to start after sizeof(totntrees) + sizeof(totnhalos) */

        /* The per-forest counts must add up to the header's halo total, or the
         * size check above proved nothing about where the forests start. */
        int64_t sum_nhalos_in_file = 0;
        for(int64_t i=0;i<totnforests_per_file[filenr];i++) {
            XRETURN(buffer[i] >= 0, EXIT_FAILURE,
                    "Error: file `%s' forest %"PRId64" has a negative halo count (%d)\n", filename, i, buffer[i]);
            sum_nhalos_in_file += buffer[i];
        }
        XRETURN(sum_nhalos_in_file == totnhalos_per_file[filenr], EXIT_FAILURE,
                "Error: file `%s' header says %"PRId64" halos but its per-forest counts sum to %"PRId64"\n",
                filename, totnhalos_per_file[filenr], sum_nhalos_in_file);

        buffer += start_forestnum_to_process_per_file[filenr];
        for(int k=0;k<nforests_to_process_this_file;k++) {
            forestnhalos[k] = buffer[k];
        }
        buffer -= start_forestnum_to_process_per_file[filenr];

        /* first compute the byte offset to the halos in start_forestnum */
        size_t byte_offset_to_halos = sizeof(int32_t) + sizeof(int32_t) + nbytes;/* start at the beginning of halo #0 in tree #0 */
        for(int64_t i=0;i<start_forestnum_to_process_per_file[filenr];i++) {
            byte_offset_to_halos += buffer[i]*record_size;
        }
        free(buffer);

        nforests_so_far = forestnhalos - lht->nhalos_per_forest;
        if(filenr == start_filenum) {
            XRETURN(nforests_so_far == 0, EXIT_FAILURE,
                    "For the first iteration total forests already processed should be identically zero. Instead we got = %"PRId64"\n",
                    nforests_so_far);
        }

        for(int64_t i=0; i<nforests_to_process_this_file; i++) {
            lht->bytes_offset_for_forest[i + nforests_so_far] = byte_offset_to_halos;
            XRETURN(i + nforests_so_far < lht->nforests, EXIT_FAILURE,
                    "ThisTask = %d Assigning to index = %"PRId64" but only space of %"PRId64" forest fds\n", ThisTask, i + nforests_so_far, lht->nforests);
            lht->fd[i + nforests_so_far] = fd;
            byte_offset_to_halos += forestnhalos[i]*record_size;

            // Can't guarantee that the `FileNr` variable in the tree file is correct.
            // Hence let's track it explicitly here.
            forests_info->FileNr[i + nforests_so_far] = filenr;

            // We want to track the original tree number from these files.  Since we split multiple
            // files across tasks, we can't guarantee that tree N processed by a certain task is
            // actually the Nth tree in any arbitrary file.
            forests_info->original_treenr[i + nforests_so_far] = i + start_forestnum_to_process_per_file[filenr];
            // We add `start_forestnum...`` because we could start reading from the middle of a
            // forest file.  Hence we would want "Forest 0" processed by that task to be
            // appropriately shifted.
        }

        forestnhalos += nforests_to_process_this_file;
    }

    /* Step 6: compute the fraction of total simulation volume this task is
     * responsible for. We assume each input tree file spans the same volume,
     * so the per-file contribution is (this task's forests in file / total
     * forests in file). We weight by total-forests-per-file because some
     * files may contain more or fewer forests while spanning the same volume
     * (e.g., a void has few large forests; a dense knot has many small ones). */
    forests_info->frac_volume_processed = 0.0;
    for(int32_t filenr = start_filenum; filenr <= end_filenum; filenr++) {
        forests_info->frac_volume_processed += (double) num_forests_to_process_per_file[filenr] / (double) totnforests_per_file[filenr];
    }
    forests_info->frac_volume_processed /= (double) run_params->NumSimulationTreeFiles;

    free(num_forests_to_process_per_file);
    free(start_forestnum_to_process_per_file);
    free(totnforests_per_file);
    free(totnhalos_per_file);

    /* Finally setup the multiplication factors necessary to generate
       unique galaxy indices (across all files, all trees and all tasks) for this run*/
    run_params->FileNr_Mulfac = 1000000000000000LL;
    run_params->ForestNr_Mulfac = 1000000000LL;

    return EXIT_SUCCESS;
}

/*
 * load_forest_lht_binary -- read one forest's halos from its tree file.
 *
 * Called once per forest assigned to this task (typically inside the main
 * forest loop in core_io_tree.c). Allocates the halo buffer, validates the
 * cached file descriptor and byte offset that setup() recorded, and reads
 * the contiguous block of struct halo_data records for this forest.
 *
 * The on-disk halos are written tree-major, packed; no per-halo headers
 * separate them. The byte offset for this forest was computed during setup,
 * so we can seek directly with pread() -- no file pointer state to manage
 * (pread does not advance the descriptor's current position).
 *
 * Inputs:
 *   forestnr      -- local forest index in [0, nforests_this_task).
 *   halos         -- output: set to point at the newly allocated halo buffer.
 *                   Caller owns the buffer and must myfree() it.
 *   forests_info  -- populated by setup_forests_io_lht_binary(); reads
 *                   lht.nhalos_per_forest[], lht.fd[], lht.bytes_offset_for_forest[].
 *
 * Returns:
 *   Number of halos in the forest (positive int64_t) on success, or a
 *   negative error code: -MALLOC_FAILURE, -INVALID_MEMORY_ACCESS_REQUESTED
 *   (forestnr out of range), -INVALID_FILE_POINTER (fd was never opened),
 *   or -FILE_READ_ERROR (corrupted offset).
 */
int64_t load_forest_lht_binary(const int64_t forestnr, struct halo_data **halos, struct forest_info *forests_info)
{
    const int64_t nhalos = (int64_t) forests_info->lht.nhalos_per_forest[forestnr];/* the array itself contains int32_t, since the LHT format*/
    struct halo_data *local_halos = mymalloc(sizeof(struct halo_data) * nhalos);
    XRETURN(local_halos != NULL, -MALLOC_FAILURE,
            "Error: Could not allocate memory for %"PRId64" halos in forestnr = %"PRId64"\n",
            nhalos, forestnr);

    if(forestnr >= forests_info->lht.nforests) {
        fprintf(stderr,"Error: Attempting to access forest = %"PRId64" but memory is allocated for only %"PRId64"\n"
                "Perhaps, the starting forest offset was not accounted for?\n", forestnr, forests_info->lht.nforests);
        return -INVALID_MEMORY_ACCESS_REQUESTED;
    }

    int fd = forests_info->lht.fd[forestnr];

    /* must have a valid file pointer  */
    if(fd <= 0 ) {
        fprintf(stderr,"Error: File pointer is NULL (i.e., you need to open the file before reading) \n");
        return -INVALID_FILE_POINTER;
    }

    const off_t offset = forests_info->lht.bytes_offset_for_forest[forestnr];
    if(offset < 0) {
        fprintf(stderr,"Error: offset = %"PRId64" must be at least 0. (Can't interpret negative offsets)\n", offset);
        return -FILE_READ_ERROR;/* negative offset would lead to file read error */
    }

    /* file descriptor can be pointing anywhere, does not get modified by this pread */
    mypread(fd, local_halos, sizeof(struct halo_data) * nhalos, offset);

    *halos = local_halos;

    return nhalos;
}


/*
 * cleanup_forests_io_lht_binary -- release per-process state for the reader.
 *
 * Called once at shutdown. Frees the per-forest bookkeeping arrays
 * (nhalos_per_forest, bytes_offset_for_forest, fd) and closes the file
 * descriptors that setup() opened and kept open for the duration of the run.
 *
 * Note: the per-forest fd[] array stores duplicate fd values (multiple
 * forests share the same open file). We close once per unique file via the
 * open_fds[] array, which has one entry per touched file -- not per forest.
 */
void cleanup_forests_io_lht_binary(struct forest_info *forests_info)
{
    struct lhalotree_info *lht = &(forests_info->lht);
    myfree(lht->nhalos_per_forest);
    myfree(lht->bytes_offset_for_forest);
    myfree(lht->fd);

    for(int32_t i=0;i<lht->numfiles;i++) {
        close(lht->open_fds[i]);
    }
    myfree(lht->open_fds);
}

/*
 * load_tree_table_lht_binary -- read the nhalos-per-forest array from every file.
 *
 * Used by Step 2 of setup_forests_io_lht_binary() when the
 * ForestDistributionScheme weights by forest size. Reads each file's
 * nhalos-per-forest array (located at byte offset 8, after the two 4-byte
 * header ints) and writes the values into the caller's output buffer.
 *
 * Width-conversion note:
 *   On disk these are 32-bit integers (the LHT format predates the need
 *   for >2^31 halos per forest). In memory we use int64_t for consistency
 *   with the rest of the pipeline. We read each file's 32-bit values into
 *   a temporary buffer, then widen-and-copy into the int64_t output array.
 *   The temp buffer is sized once to the largest file's forest count to
 *   avoid per-file malloc/free.
 *
 * Inputs:
 *   firstfile, lastfile   -- inclusive file-number range to read.
 *   totnforests_per_file  -- number of forests in each file (already read
 *                           by Step 1 of setup_forests_io_lht_binary()).
 *   run_params            -- used by get_forests_filename_lht_binary().
 *   ThisTask              -- MPI rank, used only for the warning on empty
 *                           first files (printed by rank 0 only).
 *   nhalos_per_forest     -- output buffer; must be sized to totnforests.
 *                           Filled in file-order, contiguously.
 *
 * Returns:
 *   EXIT_SUCCESS on success, FILE_NOT_FOUND if a tree file is missing,
 *   or -1 on inconsistent counts.
 */
static int load_tree_table_lht_binary(const int firstfile, const int lastfile, const int64_t *totnforests_per_file,
                                 const struct params *run_params, const int ThisTask,
                                 int64_t *nhalos_per_forest)
{
    int64_t max_nforests_per_file = 0;
    int32_t *buffer = NULL;
    for(int32_t ifile=firstfile;ifile<=lastfile;ifile++) {
        int64_t nforests_this_file = totnforests_per_file[ifile];
        if (nforests_this_file > max_nforests_per_file) max_nforests_per_file = nforests_this_file;
    }
    buffer = malloc(max_nforests_per_file * sizeof(*buffer));
    CHECK_POINTER_AND_RETURN_ON_NULL(buffer,
                                    "Failed to allocate %"PRId64" buffer to hold (32-bit integers, size = %zu) nhalos_per_forest\n",
                                    max_nforests_per_file,
                                    sizeof(*buffer));

    for(int32_t ifile=firstfile;ifile<=lastfile;ifile++) {
        int64_t nforests_this_file = totnforests_per_file[ifile];
        if(nforests_this_file == 0) {
            if(ThisTask == 0 && ifile==firstfile) {
                fprintf(stderr, "WARNING: The first file = %d does not contain any halos from a *new* tree (i.e., "
                                "the first file *only* contains halos belonging to a tree that starts in a previous file\n", ifile);
            }
            continue;
        }
        if(nforests_this_file > max_nforests_per_file) {
            fprintf(stderr,"Error: The number of forests in this file = %"PRId64" exceeds the max. number of expected forests = %"PRId64"\n",
                           nforests_this_file, max_nforests_per_file);
            return -1;
        }

        char filename[4*MAX_STRING_LEN];
        get_forests_filename_lht_binary(filename, 4*MAX_STRING_LEN, ifile, run_params);
        int fd = open(filename, O_RDONLY);
        XRETURN( fd > 0, FILE_NOT_FOUND,"Error: can't open file `%s'\n", filename);

        //the last argument is the offset for nhalos_per_forest -> the first 4 bytes
        //are for totnforests, and the next 4 bytes are for totnhalos, after that
        //the nhalos_per_forest[totnforests] starts.
        mypread(fd, buffer, sizeof(*buffer)*nforests_this_file, 8);
        for(int64_t i=0;i<nforests_this_file;i++) {
            nhalos_per_forest[i] = (int64_t) buffer[i];
        }
        nhalos_per_forest += nforests_this_file;
        XRETURN ( close(fd) >= 0, -1, "Error: Could not properly close the binary file for filename = '%s'\n", filename);
    }
    free(buffer);
    return EXIT_SUCCESS;
}
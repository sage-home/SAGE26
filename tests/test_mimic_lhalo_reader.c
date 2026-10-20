/*
 * test_mimic_lhalo_reader.c -- unit tests for the Mimic lhalo-binary merger-tree reader.
 *
 * Validates: setup_forests_io_mimic_lhalo_binary (forest and halo counts,
 * per-forest FileNr and original_treenr, GalaxyIndex multipliers, volume
 * fraction), load_forest_mimic_lhalo_binary (every struct halo_data field:
 * links and Len copied, M_Crit200 in Msun/h scaled to 10^10 Msun/h as Mvir,
 * Pos/Vel/Spin copied unchanged with Spin NOT divided by mass, 64-bit
 * MostBoundID above 2^32 preserved, FileNr set, and the -1 sentinels for the
 * fields the format lacks), cleanup, and rejection of files whose size does
 * not match header + 88-byte records (too short, too long, or written with
 * the 104-byte SAGE record) or whose per-forest counts do not sum to the
 * header total. The binary is built with MIMIC_LHALO_RECORDS_PER_READ=2 so
 * the 5-halo forest is read in three chunks, exercising the chunked read path.
 *
 * The test writes its own tree files into a private temporary directory,
 * building each 88-byte record byte by byte at the documented offsets so that
 * it checks the on-disk layout independently of the reader's struct. The
 * files are removed at the end.
 *
 * Run: from tests/, `make test_mimic_lhalo_reader`
 * (builds test_build/test_mimic_lhalo_reader and runs it).
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>

#include "../src/core_allvars.h"
#include "../src/core_mymalloc.h"
#include "../src/io/read_tree_mimic_lhalo_binary.h"
#include "test_framework.h"

/* The chunked read path is only exercised when the reader is compiled with a
 * chunk smaller than the 5-halo forest; the tests/Makefile rule passes -D. */
#if !defined(MIMIC_LHALO_RECORDS_PER_READ) || MIMIC_LHALO_RECORDS_PER_READ > 2
#error "Build this test with -DMIMIC_LHALO_RECORDS_PER_READ=2 (see tests/Makefile) so the chunked read path is exercised."
#endif

#define RECORD_SIZE 88
#define NFORESTS 2
#define NHALOS_TOTAL 8
#define M_CRIT200_TO_CODE_UNITS 1.0e-10

/* Halo counts of the two forests in the synthetic file. */
static const int32_t FOREST_NHALOS[NFORESTS] = {3, 5};

/* Directory holding all temporary tree files; set in main(). */
static char tmp_dir[MAX_STRING_LEN];

/* ========================================================================
   Synthetic data. Every field of every halo gets a value that differs from
   the same field of every other halo, so a swapped or misplaced field shows
   up as a mismatch.
   ======================================================================== */

/* The values the test writes for halo g (global index 0..7). */
struct expected_halo {
    int32_t Descendant, FirstProgenitor, NextProgenitor, FirstHaloInFOFgroup, NextHaloInFOFgroup, Len;
    float M_Crit200;      /* Msun/h, as written to disk */
    float Pos[3], Vel[3], Spin[3];
    float VelDisp, Vmax;
    int64_t MostBoundID;
    int32_t SnapNum;
};

/* M_Crit200 values in Msun/h; the first is a typical low-mass Uchuu halo. */
static const float M_CRIT200_TABLE[NHALOS_TOTAL] = {
    1.3461e7f, 2.5e9f, 7.77e11f, 4.0e13f, 1.0e8f, 3.3333e10f, 9.87654e14f, 5.5e6f
};

static struct expected_halo make_expected_halo(const int g)
{
    struct expected_halo e;
    /* Links: forest-local indices, with -1 where "none" applies. */
    e.Descendant          = (g % 3 == 0) ? -1 : (g % 3) + 100;
    e.FirstProgenitor     = (g % 2 == 0) ? (g + 1) : -1;
    e.NextProgenitor      = (g % 4 == 1) ? -1 : (g + 20);
    e.FirstHaloInFOFgroup = g + 30;
    e.NextHaloInFOFgroup  = (g == NHALOS_TOTAL - 1) ? -1 : (g + 40);
    e.Len                 = 1000 + 17 * g;
    e.M_Crit200           = M_CRIT200_TABLE[g];
    for(int k = 0; k < 3; k++) {
        e.Pos[k]  = 10.25f + 3.5f * g + 0.125f * k;
        e.Vel[k]  = -250.5f + 31.75f * g - 7.0f * k;
        /* Spin is the specific angular momentum already (Mpc/h km/s); non-zero in every component. */
        e.Spin[k] = 0.5f + 0.0625f * g + 0.015625f * k;
    }
    e.VelDisp     = 55.5f + 2.25f * g;
    e.Vmax        = 120.75f + 4.5f * g;
    /* Above 2^32 for every halo, and not representable in 32 bits. */
    e.MostBoundID = ((int64_t) 1 << 33) + 0x0123456789LL * (g + 1);
    e.SnapNum     = 60 + 3 * g;
    return e;
}

/* ========================================================================
   Little-endian record writer, independent of the reader's struct layout.
   ======================================================================== */
static void put_i32(unsigned char *rec, const size_t off, const int32_t v) { memcpy(rec + off, &v, sizeof(v)); }
static void put_f32(unsigned char *rec, const size_t off, const float v)   { memcpy(rec + off, &v, sizeof(v)); }
static void put_i64(unsigned char *rec, const size_t off, const int64_t v) { memcpy(rec + off, &v, sizeof(v)); }

/* Pack one halo into an 88-byte record at the offsets of Mimic's halo_properties.yaml. */
static void pack_record(unsigned char rec[RECORD_SIZE], const struct expected_halo *e)
{
    memset(rec, 0, RECORD_SIZE); /* includes the 4 bytes of trailing padding */
    put_i32(rec, 0,  e->Descendant);
    put_i32(rec, 4,  e->FirstProgenitor);
    put_i32(rec, 8,  e->NextProgenitor);
    put_i32(rec, 12, e->FirstHaloInFOFgroup);
    put_i32(rec, 16, e->NextHaloInFOFgroup);
    put_i32(rec, 20, e->Len);
    put_f32(rec, 24, e->M_Crit200);
    for(int k = 0; k < 3; k++) {
        put_f32(rec, 28 + 4 * k, e->Pos[k]);
        put_f32(rec, 40 + 4 * k, e->Vel[k]);
        put_f32(rec, 52 + 4 * k, e->Spin[k]);
    }
    put_f32(rec, 64, e->VelDisp);
    put_f32(rec, 68, e->Vmax);
    put_i64(rec, 72, e->MostBoundID);
    put_i32(rec, 80, e->SnapNum);
    /* bytes 84..87: padding, left zero */
}

/*
 * write_tree_file -- write `<tmp_dir>/<treename>.0` with the two-forest header
 * and `nrecords` halo records of `record_size` bytes each. Records beyond
 * NHALOS_TOTAL, or any bytes beyond the 88 packed ones when record_size > 88,
 * are zero filled. The header declares NFORESTS forests with the fixed
 * per-forest counts {3, 5} and `header_nhalos` as its halo total, whatever
 * nrecords is, so a mismatch between the header, the per-forest counts and
 * the file size can be constructed. Returns 0 on success.
 */
static int write_tree_file(const char *treename, const size_t record_size, const int nrecords, const int32_t header_nhalos)
{
    char path[2 * MAX_STRING_LEN];
    snprintf(path, sizeof(path), "%s/%s.0", tmp_dir, treename);
    FILE *fp = fopen(path, "wb");
    if(fp == NULL) {
        perror(path);
        return -1;
    }

    const int32_t header[2] = {NFORESTS, header_nhalos};
    int ok = fwrite(header, sizeof(int32_t), 2, fp) == 2;
    ok = ok && fwrite(FOREST_NHALOS, sizeof(int32_t), NFORESTS, fp) == NFORESTS;

    unsigned char *rec = calloc(1, record_size);
    if(rec == NULL) {
        fclose(fp);
        return -1;
    }
    for(int g = 0; ok && g < nrecords; g++) {
        memset(rec, 0, record_size);
        if(g < NHALOS_TOTAL) {
            const struct expected_halo e = make_expected_halo(g);
            pack_record(rec, &e);
        }
        ok = fwrite(rec, 1, record_size, fp) == record_size;
    }
    free(rec);
    return (fclose(fp) == 0 && ok) ? 0 : -1;
}

static void remove_tree_file(const char *treename)
{
    char path[2 * MAX_STRING_LEN];
    snprintf(path, sizeof(path), "%s/%s.0", tmp_dir, treename);
    unlink(path);
}

/* Fill the run parameters the reader and the shared setup pass consume. */
static void init_run_params(struct params *run_params, const char *treename)
{
    memset(run_params, 0, sizeof(*run_params));
    run_params->FirstFile = 0;
    run_params->LastFile = 0;
    run_params->NumSimulationTreeFiles = 1;
    run_params->ForestDistributionScheme = uniform_in_forests;
    run_params->Exponent_Forest_Dist_Scheme = 0.0;
    snprintf(run_params->SimulationDir, sizeof(run_params->SimulationDir), "%s", tmp_dir);
    snprintf(run_params->TreeName, sizeof(run_params->TreeName), "%s", treename);
    run_params->TreeExtension[0] = '\0';
}

/* Release the forest bookkeeping arrays the shared setup allocates with plain malloc. */
static void free_forest_arrays(struct forest_info *forests_info)
{
    free(forests_info->FileNr);
    free(forests_info->original_treenr);
    forests_info->FileNr = NULL;
    forests_info->original_treenr = NULL;
}

/* Bit-exact float comparison for values that must be copied unchanged. */
static int same_float(const float a, const float b) { return memcmp(&a, &b, sizeof(float)) == 0; }

/* ========================================================================
   Test 1: setup and load of a valid two-forest file.
   ======================================================================== */

/*
 * test_valid_file_setup_and_load -- the full lifecycle on a well-formed file.
 *
 * Input: one tree file `<TreeName>.0` with forests of 3 and 5 halos (8 total)
 * in the 88-byte layout, with distinctive values in every field.
 *
 * Expected: setup reports 2 forests and 8 halos, assigns both forests to the
 * single task, records file 0 and tree numbers 0 and 1, and sets the
 * GalaxyIndex multipliers. Each loaded halo reproduces the written values:
 * integers and float32 fields exactly, MostBoundID exactly (it exceeds 2^32),
 * Mvir as M_Crit200 * 1e-10 in float32 (Msun/h to 10^10 Msun/h), Spin
 * unchanged (it is already J/Mvir, so no division by mass), and the four
 * fields the format lacks as -1.
 */
static void test_valid_file_setup_and_load(void)
{
    BEGIN_TEST("Mimic lhalo-binary: setup, load and cleanup on a valid file");

    ASSERT_EQUAL_INT(0, write_tree_file("mimic_valid", RECORD_SIZE, NHALOS_TOTAL, NHALOS_TOTAL), "wrote synthetic tree file");

    struct params run_params;
    init_run_params(&run_params, "mimic_valid");
    struct forest_info forests_info;
    memset(&forests_info, 0, sizeof(forests_info));

    const int status = setup_forests_io_mimic_lhalo_binary(&forests_info, 0, 1, &run_params);
    ASSERT_EQUAL_INT(EXIT_SUCCESS, status, "setup returns EXIT_SUCCESS");
    if(status != EXIT_SUCCESS) {
        return;
    }

    ASSERT_TRUE(forests_info.totnforests == NFORESTS, "totnforests == 2");
    ASSERT_TRUE(forests_info.totnhalos == NHALOS_TOTAL, "totnhalos == 8");
    ASSERT_TRUE(forests_info.nforests_this_task == NFORESTS, "nforests_this_task == 2");
    ASSERT_TRUE(run_params.FileNr_Mulfac == 1000000000000000LL, "FileNr_Mulfac == 10^15");
    ASSERT_TRUE(run_params.ForestNr_Mulfac == 1000000000LL, "ForestNr_Mulfac == 10^9");
    ASSERT_CLOSE(1.0, forests_info.frac_volume_processed, 1e-12, "frac_volume_processed == 1 (all forests, one file)");
    ASSERT_TRUE(forests_info.original_treenr[0] == 0 && forests_info.original_treenr[1] == 1,
                "original_treenr == {0, 1}");
    ASSERT_TRUE(forests_info.FileNr[0] == 0 && forests_info.FileNr[1] == 0, "FileNr == {0, 0}");
    ASSERT_TRUE(forests_info.lht.nhalos_per_forest[0] == 3 && forests_info.lht.nhalos_per_forest[1] == 5,
                "per-forest halo counts == {3, 5}");

    int g = 0; /* global halo index of the first halo of the current forest */
    for(int64_t forestnr = 0; forestnr < NFORESTS; forestnr++) {
        char msg[128];
        struct halo_data *halos = NULL;
        const int64_t nhalos = load_forest_mimic_lhalo_binary(forestnr, &halos, &forests_info);
        snprintf(msg, sizeof(msg), "forest %d: load returns %d halos", (int) forestnr, FOREST_NHALOS[forestnr]);
        ASSERT_TRUE(nhalos == FOREST_NHALOS[forestnr] && halos != NULL, msg);
        if(nhalos != FOREST_NHALOS[forestnr] || halos == NULL) {
            continue;
        }

        for(int64_t i = 0; i < nhalos; i++, g++) {
            const struct expected_halo e = make_expected_halo(g);
            const struct halo_data *h = &halos[i];

            snprintf(msg, sizeof(msg), "forest %d halo %d: five tree links copied (including -1)", (int) forestnr, (int) i);
            ASSERT_TRUE(h->Descendant == e.Descendant && h->FirstProgenitor == e.FirstProgenitor &&
                        h->NextProgenitor == e.NextProgenitor && h->FirstHaloInFOFgroup == e.FirstHaloInFOFgroup &&
                        h->NextHaloInFOFgroup == e.NextHaloInFOFgroup, msg);

            snprintf(msg, sizeof(msg), "forest %d halo %d: Len copied", (int) forestnr, (int) i);
            ASSERT_TRUE(h->Len == e.Len, msg);

            /* Msun/h -> 10^10 Msun/h, scaled in double and stored as float. */
            const float expected_mvir = (float) ((double) e.M_Crit200 * M_CRIT200_TO_CODE_UNITS);
            snprintf(msg, sizeof(msg), "forest %d halo %d: Mvir == (float)(M_Crit200 * 1e-10)", (int) forestnr, (int) i);
            ASSERT_TRUE(same_float(h->Mvir, expected_mvir), msg);
            snprintf(msg, sizeof(msg), "forest %d halo %d: M200c union member aliases Mvir", (int) forestnr, (int) i);
            ASSERT_TRUE(same_float(h->M200c, expected_mvir), msg);

            int vectors_ok = 1;
            for(int k = 0; k < 3; k++) {
                vectors_ok = vectors_ok && same_float(h->Pos[k], e.Pos[k]) && same_float(h->Vel[k], e.Vel[k]) &&
                             same_float(h->Spin[k], e.Spin[k]);
            }
            snprintf(msg, sizeof(msg), "forest %d halo %d: Pos, Vel, Spin copied unchanged (Spin not divided by mass)",
                     (int) forestnr, (int) i);
            ASSERT_TRUE(vectors_ok, msg);

            snprintf(msg, sizeof(msg), "forest %d halo %d: VelDisp and Vmax copied", (int) forestnr, (int) i);
            ASSERT_TRUE(same_float(h->VelDisp, e.VelDisp) && same_float(h->Vmax, e.Vmax), msg);

            snprintf(msg, sizeof(msg), "forest %d halo %d: 64-bit MostBoundID exact (> 2^32)", (int) forestnr, (int) i);
            ASSERT_TRUE(h->MostBoundID == e.MostBoundID && h->MostBoundID > ((int64_t) 1 << 32), msg);

            snprintf(msg, sizeof(msg), "forest %d halo %d: SnapNum copied, FileNr == 0", (int) forestnr, (int) i);
            ASSERT_TRUE(h->SnapNum == e.SnapNum && h->FileNr == 0, msg);

            snprintf(msg, sizeof(msg), "forest %d halo %d: absent fields are -1 sentinels", (int) forestnr, (int) i);
            ASSERT_TRUE(h->M_Mean200 == -1.0f && h->M_TopHat == -1.0f && h->SubhaloIndex == -1 && h->SubHalfMass == -1.0f, msg);
        }
        myfree(halos);
    }

    /* The motivating case from the Uchuu data: M_Crit200 = 1.3461e7 Msun/h is 1.3461e-3 in code units. */
    struct halo_data *halos = NULL;
    ASSERT_TRUE(load_forest_mimic_lhalo_binary(0, &halos, &forests_info) == 3 && halos != NULL, "reload forest 0");
    if(halos != NULL) {
        ASSERT_CLOSE(1.3461e-3, (double) halos[0].Mvir, 1e-6, "Mvir of 1.3461e7 Msun/h halo ~= 1.3461e-3 (10^10 Msun/h)");
        myfree(halos);
    }

    /* Out-of-range forest numbers are rejected rather than read. */
    halos = NULL;
    ASSERT_TRUE(load_forest_mimic_lhalo_binary(NFORESTS, &halos, &forests_info) < 0, "forestnr == nforests_this_task is rejected");
    ASSERT_TRUE(load_forest_mimic_lhalo_binary(-1, &halos, &forests_info) < 0, "negative forestnr is rejected");

    cleanup_forests_io_mimic_lhalo_binary(&forests_info);
    free_forest_arrays(&forests_info);
    remove_tree_file("mimic_valid");
}

/* ========================================================================
   Test 2: files whose size disagrees with header + 88-byte records.
   ======================================================================== */

/*
 * expect_setup_rejects -- write a tree file whose header claims `header_nhalos`
 * halos (per-forest counts {3, 5}) but whose record stream has `nrecords`
 * records of `record_size` bytes, and check that setup returns EXIT_FAILURE,
 * the code of the file-size and per-forest-count checks in
 * setup_forests_io_lht_binary_recsize (an open or allocation failure would
 * return a different code).
 */
static void expect_setup_rejects(const char *treename, const size_t record_size, const int nrecords,
                                 const int32_t header_nhalos, const char *description)
{
    ASSERT_EQUAL_INT(0, write_tree_file(treename, record_size, nrecords, header_nhalos), "wrote malformed tree file");

    struct params run_params;
    init_run_params(&run_params, treename);
    struct forest_info forests_info;
    memset(&forests_info, 0, sizeof(forests_info));

    printf("    (the following error message is expected)\n");
    const int status = setup_forests_io_mimic_lhalo_binary(&forests_info, 0, 1, &run_params);
    ASSERT_EQUAL_INT(EXIT_FAILURE, status, description);

    /* The failed setup has already allocated the lht arrays and opened the file; release them. */
    cleanup_forests_io_mimic_lhalo_binary(&forests_info);
    free_forest_arrays(&forests_info);
    remove_tree_file(treename);
}

/*
 * test_wrong_file_size_rejected -- the guard against a wrong TreeType.
 *
 * Input: header declares 8 halos (so 88-byte records imply header + 704 bytes).
 * Cases: one record too few, one record too many, 104-byte records (the size
 * of SAGE's own struct halo_data, i.e. the file is really TreeType
 * lhalo_binary), and a file whose size matches a header total of 7 while the
 * per-forest counts {3, 5} still sum to 8.
 *
 * Expected: setup returns EXIT_FAILURE each time, because every forest offset
 * computed from a wrong record size or inconsistent counts would be wrong.
 */
static void test_wrong_file_size_rejected(void)
{
    BEGIN_TEST("Mimic lhalo-binary: files of the wrong size are rejected at setup");

    expect_setup_rejects("mimic_short", RECORD_SIZE, NHALOS_TOTAL - 1, NHALOS_TOTAL, "truncated record stream (7 of 8 records) rejected");
    expect_setup_rejects("mimic_long", RECORD_SIZE, NHALOS_TOTAL + 1, NHALOS_TOTAL, "padded record stream (9 of 8 records) rejected");
    expect_setup_rejects("mimic_sage104", 104, NHALOS_TOTAL, NHALOS_TOTAL, "104-byte (SAGE lhalo-binary) records rejected");
    expect_setup_rejects("mimic_badsum", RECORD_SIZE, NHALOS_TOTAL - 1, NHALOS_TOTAL - 1,
                         "per-forest counts that do not sum to the header total rejected");
}

/* ========================================================================
   main
   ======================================================================== */
int main(void)
{
    BEGIN_TEST_SUITE("MIMIC LHALO-BINARY TREE READER");

    /* Private scratch directory, so concurrent runs cannot collide and nothing is left behind. */
    const char *base = getenv("TMPDIR");
    snprintf(tmp_dir, sizeof(tmp_dir), "%s/sage26_mimic_test_XXXXXX", (base != NULL && base[0] != '\0') ? base : "/tmp");
    if(mkdtemp(tmp_dir) == NULL) {
        perror("mkdtemp");
        return EXIT_FAILURE;
    }

    test_valid_file_setup_and_load();
    test_wrong_file_size_rejected();

    rmdir(tmp_dir);

    END_TEST_SUITE();
    PRINT_TEST_SUMMARY();
    return TEST_EXIT_CODE();
}

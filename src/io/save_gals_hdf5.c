/*
 * save_gals_hdf5.c -- HDF5 galaxy catalogue output writer.
 *
 * Implements the four public entry points for writing galaxy catalogues in
 * SAGE26's HDF5 struct-of-arrays format: initialize (creates the file,
 * snapshot groups, and extensible datasets), save (buffers galaxies for one
 * tree and flushes in 8192-galaxy chunks), finalize (flushes the final buffer,
 * writes the TreeInfo group and Header attributes, closes all HDF5 handles),
 * and create_hdf5_master_file (writes a top-level master file that links
 * the per-task output files via HDF5 external links, run by Task 0 only).
 * Output hierarchy: File -> "Snap_<N>" group -> property datasets.
 * Each dataset is extensible (chunked at NUM_GALS_PER_BUFFER = 8192 rows).
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <hdf5.h>
#include <math.h>

#include "save_gals_hdf5.h"
#include "../core_mymalloc.h"
#include "../core_utils.h"
#include "../macros.h"
#include "../model_misc.h"
#include "../sage.h"


/* Field count derived from the single source of truth in save_gals_hdf5_fields.h */
#define SAGE_FIELD_COUNT(dset, field, ctype, h5t, desc, unit) + 1
#define NUM_OUTPUT_FIELDS (0 GALAXY_OUTPUT_FIELDS(SAGE_FIELD_COUNT))

#define NUM_GALS_PER_BUFFER 8192

// Local Proto-Types //
static int32_t generate_field_metadata(char (*field_names)[MAX_STRING_LEN], char (*field_descriptions)[MAX_STRING_LEN],
                                       char (*field_units)[MAX_STRING_LEN], hsize_t *field_dtypes);

static int32_t prepare_galaxy_for_hdf5_output(const struct GALAXY *g, struct save_info *save_info,
                                              const int32_t output_snap_idx,  const struct halo_data *halos,
                                              const int64_t task_forestnr,
                                              const int64_t original_treenr,
                                              const struct params *run_params);

static int32_t trigger_buffer_write(const int32_t snap_idx, const int32_t num_to_write, const int64_t num_already_written,
                                    struct save_info *save_info, const struct params *run_params);

static int32_t write_header(hid_t file_id, const struct forest_info *forest_info, const struct params *run_params);



// HDF5 is a self-describing data format.  Each dataset will contain a number of attributes to
// describe properties such as units or number of elements. These macros create attributes for a
// single number or a string.
/* MS: 17/9/2019 -- the group_id has already been checked and should be valid at this point */
#define CREATE_SINGLE_ATTRIBUTE(group_id, attribute_name, attribute_value, h5_dtype) { \
        hid_t macro_dataspace_id = H5Screate(H5S_SCALAR);               \
        CHECK_STATUS_AND_RETURN_ON_FAIL(macro_dataspace_id, (int32_t) macro_dataspace_id, \
                                        "Could not create an attribute dataspace.\n" \
                                        "The attribute we wanted to create was '" #attribute_name"' and the HDF5 datatype was '" #h5_dtype".\n"); \
        if(sizeof(attribute_value) != H5Tget_size(h5_dtype)) {    \
            fprintf(stderr,"Error: attribute " #attribute_name" the C size = %zu does not match the hdf5 datatype size=%zu\n", \
                    sizeof(attribute_value), H5Tget_size(h5_dtype));    \
            return -1;                                                  \
        }                                                               \
        hid_t macro_attribute_id = H5Acreate(group_id, attribute_name, h5_dtype, macro_dataspace_id, H5P_DEFAULT, H5P_DEFAULT); \
        CHECK_STATUS_AND_RETURN_ON_FAIL(macro_attribute_id, (int32_t) macro_attribute_id, \
                                        "Could not create an attribute ID.\n" \
                                        "The attribute we wanted to create was '" #attribute_name"' and the HDF5 datatype was '" #h5_dtype".\n"); \
        herr_t status = H5Awrite(macro_attribute_id, h5_dtype, &(attribute_value)); \
        CHECK_STATUS_AND_RETURN_ON_FAIL(status, (int32_t) status,       \
                                        "Could not write an attribute.\n" \
                                        "The attribute we wanted to create was '" #attribute_name"' and the HDF5 datatype was '" #h5_dtype".\n"); \
        status = H5Aclose(macro_attribute_id);                          \
        CHECK_STATUS_AND_RETURN_ON_FAIL(status, (int32_t) status,       \
                                        "Could not close an attribute ID.\n" \
                                        "The attribute we wanted to create was '" #attribute_name"' and the HDF5 datatype was '" #h5_dtype".\n"); \
        status = H5Sclose(macro_dataspace_id);                          \
        CHECK_STATUS_AND_RETURN_ON_FAIL(status, (int32_t) status,       \
                                    "Could not close an attribute dataspace.\n" \
                                        "The attribute we wanted to create was '" #attribute_name"' and the HDF5 datatype was '" #h5_dtype".\n"); \
    }

#define CREATE_STRING_ATTRIBUTE(group_id, attribute_name, attribute_value, stringlen) { \
    hid_t macro_dataspace_id = H5Screate(H5S_SCALAR);                   \
    CHECK_STATUS_AND_RETURN_ON_FAIL(macro_dataspace_id, (int32_t) macro_dataspace_id, \
                                    "Could not create an attribute dataspace for a String.\n" \
                                    "The attribute we wanted to create was '" #attribute_name"'.\n"); \
    hid_t atype = H5Tcopy(H5T_C_S1);                                  \
    CHECK_STATUS_AND_RETURN_ON_FAIL(atype, (int32_t) atype,             \
                                    "Could not copy an existing data type when creating a String attribute.\n" \
                                    "The attribute we wanted to create was '" #attribute_name"'.\n"); \
    herr_t attr_status = H5Tset_size(atype, stringlen);                 \
    CHECK_STATUS_AND_RETURN_ON_FAIL(attr_status, (int32_t) attr_status, \
                                    "Could not set the total size of a datatype when creating a String attribute.\n" \
                                    "The attribute we wanted to create was '" #attribute_name"'.\n"); \
    attr_status = H5Tset_strpad(atype, H5T_STR_NULLTERM);               \
    CHECK_STATUS_AND_RETURN_ON_FAIL(attr_status, (int32_t) attr_status, \
                                    "Could not set the padding when creating a String attribute.\n" \
                                    "The attribute we wanted to create was '" #attribute_name"'.\n"); \
    hid_t macro_attribute_id = H5Acreate(group_id, attribute_name, atype, macro_dataspace_id, H5P_DEFAULT, H5P_DEFAULT); \
    CHECK_STATUS_AND_RETURN_ON_FAIL(macro_attribute_id, (int32_t) macro_attribute_id, \
                                    "Could not create an attribute ID for string.\n" \
                                    "The attribute we wanted to create was '" #attribute_name"'.\n"); \
    attr_status = H5Awrite(macro_attribute_id, atype, attribute_value); \
    CHECK_STATUS_AND_RETURN_ON_FAIL(attr_status, (int32_t) attr_status, \
                                    "Could not write an attribute.\n"   \
                                    "The attribute we wanted to create was '" #attribute_name"'.\n"); \
    attr_status = H5Aclose(macro_attribute_id);                         \
    CHECK_STATUS_AND_RETURN_ON_FAIL(attr_status, (int32_t) attr_status,                            \
                                    "Could not close an attribute ID.\n" \
                                    "The attribute we wanted to create was '" #attribute_name"'.\n"); \
    attr_status = H5Tclose(atype);                                      \
    CHECK_STATUS_AND_RETURN_ON_FAIL(attr_status, (int32_t) attr_status, \
                                    "Could not close atype value.\n"    \
                                    "The attribute we wanted to create was '" #attribute_name"'.\n"); \
    attr_status = H5Sclose(macro_dataspace_id);                         \
    CHECK_STATUS_AND_RETURN_ON_FAIL(attr_status, (int32_t) attr_status, \
                                    "Could not close an attribute dataspace when creating a String attribute.\n" \
                                    "The attribute we wanted to create was '" #attribute_name"'.\n"); \
    }

#define CREATE_AND_WRITE_1D_ARRAY(file_id, field_name, dims, buffer, h5_dtype) { \
        if(sizeof(buffer[0]) != H5Tget_size(h5_dtype)) {                \
            fprintf(stderr,"Error: For field " #field_name", the C size = %zu does not match the hdf5 datatype size=%zu\n", \
                    sizeof(buffer[0]),  H5Tget_size(h5_dtype));         \
            return -1;                                                  \
        }                                                               \
        hid_t macro_dataspace_id = H5Screate_simple(1, dims, NULL);     \
        CHECK_STATUS_AND_RETURN_ON_FAIL(macro_dataspace_id, (int32_t) macro_dataspace_id, \
                                        "Could not create a dataspace for field " #field_name".\n" \
                                        "The dimensions of the dataspace was %d\n", (int32_t) dims[0]); \
        hid_t macro_dataset_id = H5Dcreate2(file_id, field_name, h5_dtype, macro_dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT); \
        CHECK_STATUS_AND_RETURN_ON_FAIL(macro_dataset_id, (int32_t) macro_dataset_id, \
                                        "Could not create a dataset for field " #field_name".\n" \
                                        "The dimensions of the dataset was %d\nThe file id was %d\n.", \
                                        (int32_t) dims[0], (int32_t) file_id); \
        herr_t dset_status = H5Dwrite(macro_dataset_id, h5_dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer); \
        CHECK_STATUS_AND_RETURN_ON_FAIL(dset_status, (int32_t) dset_status, \
                                        "Failed to write a dataset for field " #field_name".\n" \
                                        "The dimensions of the dataset was %d\nThe file ID was %d\n." \
                                        "The dataset ID was %d.", (int32_t) dims[0], (int32_t) file_id, \
                                        (int32_t) macro_dataset_id);    \
        dset_status = H5Dclose(macro_dataset_id);                       \
        CHECK_STATUS_AND_RETURN_ON_FAIL(dset_status, (int32_t) dset_status, \
                                        "Failed to close the dataset for field " #field_name".\n" \
                                        "The dimensions of the dataset was %d\nThe file ID was %d\n." \
                                        "The dataset ID was %d.", (int32_t) dims[0], (int32_t) file_id, \
                                        (int32_t) macro_dataset_id);    \
        dset_status = H5Sclose(macro_dataspace_id);                     \
        CHECK_STATUS_AND_RETURN_ON_FAIL(dset_status, (int32_t) dset_status, \
                                        "Failed to close the dataspace for field " #field_name".\n" \
                                        "The dimensions of the dataset was %d\nThe file ID was %d\n." \
                                        "The dataspace ID was %d.", (int32_t) dims[0], (int32_t) file_id, \
                                        (int32_t) macro_dataspace_id);  \
    }

// Unlike the binary output where we generate an array of output struct instances, the HDF5 workflow has
// a single output struct (for each snapshot) where the **properties** of the struct are arrays.
// This macro callocs (i.e., allocates and zeros) space for these inner arrays.
#define MALLOC_GALAXY_OUTPUT_INNER_ARRAY(snap_idx, field_name) {        \
        save_info->buffer_output_gals[snap_idx].field_name = malloc(save_info->buffer_size * sizeof(*(save_info->buffer_output_gals[snap_idx].field_name))); \
        if(save_info->buffer_output_gals[snap_idx].field_name == NULL) { \
            fprintf(stderr, "Could not allocate %d elements for the " #field_name" GALAXY_OUTPUT " \
                    "field for output snapshot " #snap_idx"\n", save_info->buffer_size); \
            return MALLOC_FAILURE;                                      \
        }                                                               \
    }

#define FREE_GALAXY_OUTPUT_INNER_ARRAY(snap_idx, field_name) {     \
        free(save_info->buffer_output_gals[snap_idx].field_name);  \
    }

// Externally Visible Functions //

/*
 * initialize_hdf5_galaxy_files -- create the HDF5 output file and prepare all
 * snapshot groups and extensible datasets.
 *
 * Creates "<OutputDir>/<FileNameGalaxies>_<filenr>.hdf5" and builds the group
 * hierarchy File -> "Snap_<N>" for each of the NumSnapOutputs output snapshots.
 * Inside each snapshot group, all NUM_OUTPUT_FIELDS galaxy property datasets are
 * created as rank-1 extensible datasets (initial size 0, chunk size
 * NUM_GALS_PER_BUFFER) so that save_hdf5_galaxies() can extend them on the fly.
 * Also allocates and zeros the per-snapshot galaxy buffers (buffer_output_gals).
 * All open HDF5 handles are stored in save_info for later use.
 *
 * Returns EXIT_SUCCESS, or a negative SAGE error code on failure.
 */
int32_t initialize_hdf5_galaxy_files(const int filenr, struct save_info *save_info, const struct params *run_params)
{
    char buffer[3*MAX_STRING_LEN];

    // Create the file.
    // Use 3*MAX_STRING_LEN because OutputDir and FileNameGalaxies can be MAX_STRING_LEN.  Add a bit more buffer for the filenr and '.hdf5'.
    snprintf(buffer, 3*MAX_STRING_LEN-1, "%s/%s_%d.hdf5", run_params->OutputDir, run_params->FileNameGalaxies, filenr);

    hid_t file_id = H5Fcreate(buffer, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    CHECK_STATUS_AND_RETURN_ON_FAIL(file_id, FILE_NOT_FOUND,
                                    "Can't open file %s for initialization.\n", buffer);
    save_info->file_id = file_id;

    // Generate the names, description and HDF5 data types for each of the output fields.
    // Zero the string buffers first: the Description/Units attributes are written
    // with the full MAX_STRING_LEN datatype, so any bytes beyond the snprintf'd
    // text end up in the file -- they must be defined (zero), not stack garbage.
    char field_names[NUM_OUTPUT_FIELDS][MAX_STRING_LEN];
    char field_descriptions[NUM_OUTPUT_FIELDS][MAX_STRING_LEN];
    char field_units[NUM_OUTPUT_FIELDS][MAX_STRING_LEN];
    hsize_t field_dtypes[NUM_OUTPUT_FIELDS];

    memset(field_names, 0, sizeof(field_names));
    memset(field_descriptions, 0, sizeof(field_descriptions));
    memset(field_units, 0, sizeof(field_units));

    generate_field_metadata(field_names, field_descriptions, field_units, field_dtypes);

    save_info->num_output_fields = NUM_OUTPUT_FIELDS;
    save_info->name_output_fields = malloc(NUM_OUTPUT_FIELDS * sizeof(save_info->name_output_fields[0]));
    CHECK_POINTER_AND_RETURN_ON_NULL(save_info->name_output_fields,
                                     "Failed to allocate %d elements of size %zu for save_info->name_output_fields",
                                     NUM_OUTPUT_FIELDS,
                                     sizeof(char *));

    for(int i=0;i<NUM_OUTPUT_FIELDS;i++) {
        save_info->name_output_fields[i] = malloc(MAX_STRING_LEN * sizeof(save_info->name_output_fields[i][0]));
        CHECK_POINTER_AND_RETURN_ON_NULL(save_info->name_output_fields[i],
                                         "Failed to allocate %d elements of size %zu for save_info->name_output_fields[%d]",
                                         NUM_OUTPUT_FIELDS,
                                         sizeof(char), i);
        memcpy(save_info->name_output_fields[i], field_names[i], MAX_STRING_LEN);
    }
    save_info->field_dtypes = malloc(NUM_OUTPUT_FIELDS * sizeof(save_info->field_dtypes[0]));
    CHECK_POINTER_AND_RETURN_ON_NULL(save_info->field_dtypes,
                                     "Failed to allocate %d elements of size %zu for save_info->field_dtypes",
                                     NUM_OUTPUT_FIELDS,
                                     sizeof(save_info->field_dtypes[0]));

    // We will have groups for each output snapshot, and then inside those groups, a dataset for
    // each field.
    save_info->group_ids = mymalloc(run_params->NumSnapOutputs * sizeof(save_info->group_ids[0]));
    CHECK_POINTER_AND_RETURN_ON_NULL(save_info->group_ids,
                                     "Failed to allocate %d elements of size %zu for save_info->group_ids", run_params->NumSnapOutputs,
                                     sizeof(*(save_info->group_ids)));

    // A couple of variables before we enter the loop.
    // JS 17/03/19: I've attempted to put these directly into the function calls and things blew up.
    /* Note from MS: That almost certainly means that there is a bug somewhere here (16/9/2019) */
    for(int32_t snap_idx = 0; snap_idx < run_params->NumSnapOutputs; snap_idx++) {

        hsize_t dims[1] = {0};
        hsize_t maxdims[1] = {H5S_UNLIMITED};
        hsize_t chunk_dims[1] = {NUM_GALS_PER_BUFFER};
        char full_field_name[2*MAX_STRING_LEN];

        // Create a snapshot group.
        snprintf(full_field_name, 2*MAX_STRING_LEN - 1, "Snap_%d", run_params->ListOutputSnaps[snap_idx]);
        hid_t group_id = H5Gcreate2(file_id, full_field_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        CHECK_STATUS_AND_RETURN_ON_FAIL(group_id, (int32_t) group_id,
                                        "Failed to create the %s group.\nThe file ID was %d\n", full_field_name,
                                        (int32_t) file_id);
        save_info->group_ids[snap_idx] = group_id;

        const float redshift = run_params->ZZ[run_params->ListOutputSnaps[snap_idx]];
        CREATE_SINGLE_ATTRIBUTE(group_id, "redshift", redshift, H5T_NATIVE_FLOAT);

        for(int32_t field_idx = 0; field_idx < NUM_OUTPUT_FIELDS; field_idx++) {

            // Then create each field inside.
            snprintf(full_field_name, 2*MAX_STRING_LEN - 1,"Snap_%d/%.*s", run_params->ListOutputSnaps[snap_idx],
                     MAX_STRING_LEN - 1, field_names[field_idx]);

            hid_t prop = H5Pcreate(H5P_DATASET_CREATE);
            CHECK_STATUS_AND_RETURN_ON_FAIL(prop, (int32_t) prop,
                                            "Could not create the property list for output snapshot number %d.\n", snap_idx);

            // Create a dataspace with 0 dimension.  We will extend the datasets before every write.
            hid_t dataspace_id = H5Screate_simple(1, dims, maxdims);
            CHECK_STATUS_AND_RETURN_ON_FAIL(dataspace_id, (int32_t) dataspace_id,
                                            "Could not create a dataspace for output snapshot number %d.\n"
                                            "The requested initial size was %d with an unlimited maximum upper bound.",
                                            snap_idx, (int32_t) dims[0]);

            // To increase reading/writing speed, we chunk the HDF5 file. --JS
            // MS: That is incorrect. We need a resizeable dataset, and that
            //requires chunking
            herr_t status = H5Pset_chunk(prop, 1, chunk_dims);
            CHECK_STATUS_AND_RETURN_ON_FAIL(status, (int32_t) status,
                                            "Could not set the HDF5 chunking for output snapshot number %d.  Chunk size was %d.\n",
                                            snap_idx, (int32_t) chunk_dims[0]);

            // Now create the dataset.
            hid_t dataset_id = H5Dcreate2(file_id, full_field_name, field_dtypes[field_idx], dataspace_id, H5P_DEFAULT, prop, H5P_DEFAULT);
            CHECK_STATUS_AND_RETURN_ON_FAIL(dataset_id, (int32_t) dataset_id,
                                            "Could not create the '%s' dataset.\n", full_field_name);

            // Set metadata attributes for each dataset.
            CREATE_STRING_ATTRIBUTE(dataset_id, "Description", field_descriptions[field_idx], MAX_STRING_LEN);
            CREATE_STRING_ATTRIBUTE(dataset_id, "Units", field_units[field_idx], MAX_STRING_LEN);

            status = H5Dclose(dataset_id);
            CHECK_STATUS_AND_RETURN_ON_FAIL(status, (int32_t) status,
                                            "Failed to close field number %d for output snapshot number %d\n"
                                            "The dataset ID was %d\n", field_idx, snap_idx,
                                            (int32_t) dataset_id);

            status = H5Pclose(prop);
            CHECK_STATUS_AND_RETURN_ON_FAIL(status, (int32_t) status,
                                            "Failed to close the property list for output snapshot number %d.\n", snap_idx);

            status = H5Sclose(dataspace_id);
            CHECK_STATUS_AND_RETURN_ON_FAIL(status, (int32_t) status,
                                            "Failed to close the dataspace for output snapshot number %d.\n", snap_idx);

        }
        
        // Conditionally create 2D datasets for cumulative SFH if SaveFullSFH is enabled
        if(run_params->SaveFullSFH) {
            herr_t sfh_status;  // Use different name to avoid shadowing macro's 'status'
            
            // Create 2D datasets for cumulative SFH (stellar mass formed at each snapshot)
            const char *cum_sfh_names[2] = {"SFHMassDisk", "SFHMassBulge"};
            const char *cum_sfh_descriptions[2] = {
                "Cumulative star formation history - stellar mass formed in disk at each snapshot (10^10 Msun/h)",
                "Cumulative star formation history - stellar mass formed in bulge (starbursts) at each snapshot (10^10 Msun/h)"
            };
            const char *cum_sfh_units[2] = {"10^10 Msun/h", "10^10 Msun/h"};
            
            for(int cum_idx = 0; cum_idx < 2; cum_idx++) {
                snprintf(full_field_name, 2*MAX_STRING_LEN - 1, "Snap_%d/%s", run_params->ListOutputSnaps[snap_idx], cum_sfh_names[cum_idx]);
                
                // Create 2D dataset with shape [0, SimMaxSnaps] initially, extensible in first dimension
                hsize_t dims_cum[2] = {0, (hsize_t)run_params->SimMaxSnaps};
                hsize_t maxdims_cum[2] = {H5S_UNLIMITED, (hsize_t)run_params->SimMaxSnaps};
                hsize_t chunk_dims_cum[2] = {NUM_GALS_PER_BUFFER, (hsize_t)run_params->SimMaxSnaps};
                
                hid_t prop_cum = H5Pcreate(H5P_DATASET_CREATE);
                CHECK_STATUS_AND_RETURN_ON_FAIL(prop_cum, (int32_t) prop_cum,
                                                "Could not create property list for cumulative SFH dataset %s", cum_sfh_names[cum_idx]);
                
                sfh_status = H5Pset_chunk(prop_cum, 2, chunk_dims_cum);
                CHECK_STATUS_AND_RETURN_ON_FAIL(sfh_status, (int32_t) sfh_status,
                                                "Could not set chunking for cumulative SFH dataset %s", cum_sfh_names[cum_idx]);
                
                hid_t dataspace_cum = H5Screate_simple(2, dims_cum, maxdims_cum);
                CHECK_STATUS_AND_RETURN_ON_FAIL(dataspace_cum, (int32_t) dataspace_cum,
                                                "Could not create dataspace for cumulative SFH dataset %s", cum_sfh_names[cum_idx]);
                
                hid_t dataset_cum = H5Dcreate2(file_id, full_field_name, H5T_NATIVE_FLOAT, dataspace_cum,
                                               H5P_DEFAULT, prop_cum, H5P_DEFAULT);
                CHECK_STATUS_AND_RETURN_ON_FAIL(dataset_cum, (int32_t) dataset_cum,
                                                "Could not create cumulative SFH dataset %s", cum_sfh_names[cum_idx]);
                
                // Set metadata attributes. The attribute datatype is MAX_STRING_LEN
                // wide and H5Awrite reads that many bytes from the source, so the
                // string literals must be staged in a zeroed buffer of that size --
                // passing them directly reads past the end of the literal.
                char cum_attr_buf[MAX_STRING_LEN];
                memset(cum_attr_buf, 0, sizeof(cum_attr_buf));
                snprintf(cum_attr_buf, sizeof(cum_attr_buf), "%s", cum_sfh_descriptions[cum_idx]);
                CREATE_STRING_ATTRIBUTE(dataset_cum, "Description", cum_attr_buf, MAX_STRING_LEN);
                memset(cum_attr_buf, 0, sizeof(cum_attr_buf));
                snprintf(cum_attr_buf, sizeof(cum_attr_buf), "%s", cum_sfh_units[cum_idx]);
                CREATE_STRING_ATTRIBUTE(dataset_cum, "Units", cum_attr_buf, MAX_STRING_LEN);
                CREATE_SINGLE_ATTRIBUTE(dataset_cum, "NumSnapshots", run_params->SimMaxSnaps, H5T_NATIVE_INT);
                
                sfh_status = H5Dclose(dataset_cum);
                CHECK_STATUS_AND_RETURN_ON_FAIL(sfh_status, (int32_t) sfh_status,
                                                "Failed to close cumulative SFH dataset %s", cum_sfh_names[cum_idx]);
                
                sfh_status = H5Pclose(prop_cum);
                CHECK_STATUS_AND_RETURN_ON_FAIL(sfh_status, (int32_t) sfh_status,
                                                "Failed to close property list for cumulative SFH dataset %s", cum_sfh_names[cum_idx]);
                
                sfh_status = H5Sclose(dataspace_cum);
                CHECK_STATUS_AND_RETURN_ON_FAIL(sfh_status, (int32_t) sfh_status,
                                                "Failed to close dataspace for cumulative SFH dataset %s", cum_sfh_names[cum_idx]);
            }
        }
    }

    // Now for each snapshot, we process `buffer_count` galaxies into RAM for every snapshot before
    // writing a single chunk. Unlike the binary instance where we have a single GALAXY_OUTPUT
    // struct instance per galaxy, here HDF5_GALAXY_OUTPUT is a **struct of arrays**.
    save_info->buffer_size = NUM_GALS_PER_BUFFER;
    save_info->num_gals_in_buffer = mycalloc(run_params->NumSnapOutputs, sizeof(save_info->num_gals_in_buffer[0])); // Calloced because initially no galaxies in buffer.

    CHECK_POINTER_AND_RETURN_ON_NULL(save_info->num_gals_in_buffer,
                                     "Failed to allocate %d elements of size %zu for save_info->num_gals_in_buffer", run_params->NumSnapOutputs,
                                     sizeof(save_info->num_gals_in_buffer[0]));

    save_info->buffer_output_gals = mymalloc(run_params->NumSnapOutputs * sizeof(save_info->buffer_output_gals[0]));

    CHECK_POINTER_AND_RETURN_ON_NULL(save_info->buffer_output_gals,
                                     "Failed to allocate %d elements of size %zu for save_info->buffer_output_gals", run_params->NumSnapOutputs,
                                     sizeof(save_info->buffer_output_gals[0]));

    // Now we need to malloc all the arrays **inside** the GALAXY_OUTPUT struct.
    for(int32_t snap_idx = 0; snap_idx < run_params->NumSnapOutputs; snap_idx++) {

#define SAGE_FIELD_MALLOC(dset, field, ctype, h5t, desc, unit) MALLOC_GALAXY_OUTPUT_INNER_ARRAY(snap_idx, field);
        GALAXY_OUTPUT_FIELDS(SAGE_FIELD_MALLOC)
#undef SAGE_FIELD_MALLOC
        MALLOC_GALAXY_OUTPUT_INNER_ARRAY(snap_idx, TaskForestNr);
        
        /* Conditionally allocate cumulative SFH arrays if SaveFullSFH is enabled */
        if(run_params->SaveFullSFH) {
            /* Allocate memory for cumulative SFH - SimMaxSnaps elements per galaxy */
            save_info->buffer_output_gals[snap_idx].SFHMassDisk = malloc(save_info->buffer_size * run_params->SimMaxSnaps * sizeof(float));
            save_info->buffer_output_gals[snap_idx].SFHMassBulge = malloc(save_info->buffer_size * run_params->SimMaxSnaps * sizeof(float));
            
            if(save_info->buffer_output_gals[snap_idx].SFHMassDisk == NULL ||
               save_info->buffer_output_gals[snap_idx].SFHMassBulge == NULL) {
                fprintf(stderr, "Could not allocate memory for SFH arrays (SimMaxSnaps=%d, buffer_size=%d)\n",
                        run_params->SimMaxSnaps, save_info->buffer_size);
                if(save_info->buffer_output_gals[snap_idx].SFHMassDisk != NULL) {
                    free(save_info->buffer_output_gals[snap_idx].SFHMassDisk);
                    save_info->buffer_output_gals[snap_idx].SFHMassDisk = NULL;
                }
                if(save_info->buffer_output_gals[snap_idx].SFHMassBulge != NULL) {
                    free(save_info->buffer_output_gals[snap_idx].SFHMassBulge);
                    save_info->buffer_output_gals[snap_idx].SFHMassBulge = NULL;
                }
                return MALLOC_FAILURE;
            }
        } else {
            /* Set pointers to NULL if not saving full SFH */
            save_info->buffer_output_gals[snap_idx].SFHMassDisk = NULL;
            save_info->buffer_output_gals[snap_idx].SFHMassBulge = NULL;
        }
    }

    return EXIT_SUCCESS;
}

#undef MALLOC_GALAXY_OUTPUT_INNER_ARRAY

/*
 * save_hdf5_galaxies -- buffer one tree's galaxies and flush to HDF5 when the
 * buffer is full.
 *
 * Iterates over num_gals galaxies in the current tree; each galaxy with a valid
 * output_snap_n is packed into the per-snapshot GALAXY_OUTPUT_HDF5 buffer via
 * prepare_galaxy_for_hdf5_output().  Whenever num_gals_in_buffer reaches
 * NUM_GALS_PER_BUFFER, trigger_buffer_write() extends and writes the datasets.
 * Updates the cumulative galaxy count in save_info->tot_ngals[].
 *
 * Returns EXIT_SUCCESS, or a negative SAGE error code on failure.
 */
int32_t save_hdf5_galaxies(const int64_t task_forestnr, const int32_t num_gals, struct forest_info *forest_info,
                           struct halo_data *halos, struct halo_aux_data *haloaux, struct GALAXY *halogal,
                           struct save_info *save_info, const struct params *run_params)
{
    int32_t status = EXIT_FAILURE;

    for(int32_t gal_idx = 0; gal_idx < num_gals; gal_idx++) {

        // Only processing galaxies at selected snapshots. This field was generated in `save_galaxies()`.
        if(haloaux[gal_idx].output_snap_n < 0) {
            continue;
        }

        // Add galaxies to buffer.
        int32_t snap_idx = haloaux[gal_idx].output_snap_n;
        status = prepare_galaxy_for_hdf5_output(&halogal[gal_idx], save_info, snap_idx, halos, task_forestnr,
                                                forest_info->original_treenr[task_forestnr], run_params);
        if(status != EXIT_SUCCESS) {
            return status;
        }
        save_info->num_gals_in_buffer[snap_idx]++;

        // We can't guarantee that this tree will contain enough galaxies to trigger a write.
        // Hence we need to increment this here.
        save_info->forest_ngals[snap_idx][task_forestnr]++;

        // Check to see if we need to write.
        if(save_info->num_gals_in_buffer[snap_idx] == save_info->buffer_size) {
            status = trigger_buffer_write(snap_idx, save_info->buffer_size, save_info->tot_ngals[snap_idx], save_info, run_params);
            if(status != EXIT_SUCCESS) {
                return status;
            }
        }
    }

    return EXIT_SUCCESS;
}


/*
 * finalize_hdf5_galaxy_files -- flush remaining buffered galaxies, write the
 * Header and TreeInfo groups, and close all HDF5 handles.
 *
 * Calls trigger_buffer_write() for any galaxies still in each snapshot buffer,
 * then writes the TreeInfo group (per-tree forest counts) and the Header group
 * (simulation metadata, run parameters, git ref) via write_header().  Closes
 * every open dataset, group, and file handle stored in save_info, then frees
 * the per-snapshot galaxy buffers.
 *
 * Returns EXIT_SUCCESS, or a negative SAGE error code on failure.
 */
int32_t finalize_hdf5_galaxy_files(const struct forest_info *forest_info, struct save_info *save_info,
                                   const struct params *run_params)
{

    hid_t group_id = H5Gcreate(save_info->file_id, "/TreeInfo", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHECK_STATUS_AND_RETURN_ON_FAIL(group_id, (int32_t) group_id,
                                    "Failed to create the TreeInfo group.\nThe file ID was %d\n",
                                    (int32_t) save_info->file_id);

    herr_t h5_status = H5Gclose(group_id);
    CHECK_STATUS_AND_RETURN_ON_FAIL(h5_status, (int32_t) h5_status,
                                    "Failed to close the /TreeInfo group."
                                    "The group ID was %d.\n", (int32_t) group_id);

    for(int32_t snap_idx = 0; snap_idx < run_params->NumSnapOutputs; snap_idx++) {

        // Attributes can only be 64kb in size (strict rule enforced by the HDF5 group).
        // For larger simulations, we will have so many trees, that the number of galaxies per tree
        // array (`save_info->forest_ngals`) will exceed 64kb.  Hence we will write this data to a
        // dataset rather than into an attribute.
        char field_name[MAX_STRING_LEN];
        char description[MAX_STRING_LEN];
        char unit[MAX_STRING_LEN];

        snprintf(field_name, MAX_STRING_LEN - 1, "/TreeInfo/Snap_%d", run_params->ListOutputSnaps[snap_idx]);
        group_id = H5Gcreate2(save_info->file_id, field_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        CHECK_STATUS_AND_RETURN_ON_FAIL(group_id, (int32_t) group_id,
                                        "Failed to create the '%s' group.\n"
                                        "The file ID was %d\n", field_name, (int32_t) save_info->file_id);

        h5_status = H5Gclose(group_id);
        CHECK_STATUS_AND_RETURN_ON_FAIL(h5_status, (int32_t) h5_status,
                                        "Failed to close '%s' group."
                                        "The group ID was %d.\n", field_name, (int32_t) group_id);

        // We still have galaxies remaining in the buffer. Need to write them.
        int32_t num_gals_to_write = save_info->num_gals_in_buffer[snap_idx];

        if(num_gals_to_write > 0) {
            h5_status = trigger_buffer_write(snap_idx, num_gals_to_write,
                                             save_info->tot_ngals[snap_idx], save_info, run_params);
            if(h5_status != EXIT_SUCCESS) {
                return h5_status;
            }

            for(int32_t gal_idx = 0; gal_idx < num_gals_to_write; gal_idx++) {

                // JS: We're going to be a bit sneaky here so we don't need to pass the tree number to this function.
                /* MS: 17/9/2019 -- we have to do it the difficult way! */
                int64_t tree = save_info->buffer_output_gals[snap_idx].TaskForestNr[gal_idx];
                if(tree < 0 || tree >= forest_info->nforests_this_task) {
                    fprintf(stderr,"\nError: at snap_idx = %d -> got tree = %"PRId64". num_gals_to_write = %d\n"
                            "Expecting to get tree in the range [0, %"PRId64"), where the upper limit is the number of forests on THIS task\n",
                            snap_idx, tree, num_gals_to_write, forest_info->nforests_this_task);
                    return EXIT_FAILURE;
                }
                save_info->forest_ngals[snap_idx][tree]++;
            }
        }

        // Write attributes showing how many galaxies we wrote for this snapshot.
        CREATE_SINGLE_ATTRIBUTE(save_info->group_ids[snap_idx], "num_gals", save_info->tot_ngals[snap_idx], H5T_NATIVE_LLONG);


        snprintf(field_name, MAX_STRING_LEN -  1, "TreeInfo/Snap_%d/NumGalsPerTreePerSnap", run_params->ListOutputSnaps[snap_idx]);
        snprintf(description, MAX_STRING_LEN -  1, "The number of galaxies per tree at this snapshot.");
        snprintf(unit, MAX_STRING_LEN-  1, "Unitless");

        // JS: I've tried to put this manually into the function but it keeps hanging...
        hsize_t dims[1];
        dims[0] = forest_info->nforests_this_task;;

        hid_t dataspace_id = H5Screate_simple(1, dims, NULL);
        CHECK_STATUS_AND_RETURN_ON_FAIL(dataspace_id, (int32_t) dataspace_id,
                                        "Could not create a dataspace for the number of galaxies per tree.\n"
                                        "The dimensions of the dataspace was %d\n", (int32_t) dims[0]);

        hid_t dataset_id = H5Dcreate2(save_info->file_id, field_name, H5T_NATIVE_INT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        CHECK_STATUS_AND_RETURN_ON_FAIL(dataset_id, (int32_t) dataset_id,
                                        "Could not create a dataset for the number of galaxies per tree at snapshot = %d.\n"
                                        "The dimensions of the dataset was %d\nThe file id was %d.\n",
                                        snap_idx, (int32_t) dims[0], (int32_t) save_info->file_id);

        h5_status = H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, save_info->forest_ngals[snap_idx]);
        CHECK_STATUS_AND_RETURN_ON_FAIL(h5_status, (int32_t) h5_status,
                                        "Failed to write a dataset for the number of galaxies per tree at snapshot = %d.\n"
                                        "The dimensions of the dataset was %d.\nThe file ID was %d.\n"
                                        "The dataset ID was %d.",
                                        snap_idx, (int32_t) dims[0], (int32_t) save_info->file_id,
                                        (int32_t) dataset_id);

        h5_status = H5Dclose(dataset_id);
        CHECK_STATUS_AND_RETURN_ON_FAIL(h5_status, (int32_t) h5_status,
                                        "Failed to close the dataset for the number of galaxies per tree.\n"
                                        "The dimensions of the dataset was %d\nThe file ID was %d\n."
                                        "The dataset ID was %d.", (int32_t) dims[0], (int32_t) save_info->file_id,
                                        (int32_t) dataset_id);

        h5_status = H5Sclose(dataspace_id);
        CHECK_STATUS_AND_RETURN_ON_FAIL(h5_status, (int32_t) h5_status,
                                        "Failed to close the dataspace for the number of galaxies per tree.\n"
                                        "The dimensions of the dataset was %d\nThe file ID was %d\n."
                                        "The dataset ID was %d.", (int32_t) dims[0], (int32_t) save_info->file_id,
                                        (int32_t) dataset_id);
    }
    group_id = H5Gopen2(save_info->file_id, "/TreeInfo", H5P_DEFAULT);

    /*MS: Now add in the two attributes about the ID generation scheme */
    CREATE_SINGLE_ATTRIBUTE(group_id, "FileNr_Mulfac", run_params->FileNr_Mulfac, H5T_NATIVE_LLONG);
    CREATE_SINGLE_ATTRIBUTE(group_id, "ForestNr_Mulfac", run_params->ForestNr_Mulfac, H5T_NATIVE_LLONG);

    h5_status = H5Gclose(group_id);
    CHECK_STATUS_AND_RETURN_ON_FAIL(h5_status, (int32_t) h5_status,
                                    "Failed to close the NumGalsPerTree group."
                                    "The group ID was %d.\n", (int32_t) group_id);


    // Finally let's write some header attributes here.
    // We do this here rather than in ``initialize()`` because we need the number of galaxies per tree.
    group_id = H5Gcreate(save_info->file_id, "/Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHECK_STATUS_AND_RETURN_ON_FAIL(group_id, (int32_t) group_id,
                                    "Failed to create the Header group.\nThe file ID was %d\n",
                                    (int32_t) save_info->file_id);

    int status = write_header(save_info->file_id, forest_info, run_params);
    if(status != EXIT_SUCCESS) {
        return status;
    }

    status = H5Gclose(group_id);
    CHECK_STATUS_AND_RETURN_ON_FAIL(status, (int32_t) status,
                                    "Failed to close the Header group."
                                    "The group ID was %d.\n", (int32_t) group_id);


    // Now we need to ensure we free all of the HDF5 IDs.  The hierachy is File->Groups->Datasets.
    for(int32_t snap_idx = 0; snap_idx < run_params->NumSnapOutputs; snap_idx++) {
        // Then close the group.
        status = H5Gclose(save_info->group_ids[snap_idx]);
        CHECK_STATUS_AND_RETURN_ON_FAIL(status, (int32_t) status,
                                        "Failed to close the group for output snapshot number %d\n"
                                        "The group ID was %d\n", snap_idx, (int32_t) save_info->group_ids[snap_idx]);
    }

    // Finally the file itself.
    status = H5Fclose(save_info->file_id);
    CHECK_STATUS_AND_RETURN_ON_FAIL(status, (int32_t) status,
                                    "Failed to close the HDF5 file.\nThe file ID was %d\n",
                                    (int32_t) save_info->file_id);

    myfree(save_info->group_ids);

    for(int32_t i=0;i<save_info->num_output_fields;i++) {
        free(save_info->name_output_fields[i]);
    }
    free(save_info->name_output_fields);
    free(save_info->field_dtypes);

    // Free all the other memory.
    myfree(save_info->num_gals_in_buffer);

    for(int32_t snap_idx = 0; snap_idx < run_params->NumSnapOutputs; snap_idx++) {

#define SAGE_FIELD_FREE(dset, field, ctype, h5t, desc, unit) FREE_GALAXY_OUTPUT_INNER_ARRAY(snap_idx, field);
        GALAXY_OUTPUT_FIELDS(SAGE_FIELD_FREE)
#undef SAGE_FIELD_FREE
        FREE_GALAXY_OUTPUT_INNER_ARRAY(snap_idx, TaskForestNr);
        
        /* Conditionally free full SFH arrays if they were allocated */
        /* Conditionally free SFH arrays if they were allocated */
        if(run_params->SaveFullSFH) {
            free(save_info->buffer_output_gals[snap_idx].SFHMassDisk);
            free(save_info->buffer_output_gals[snap_idx].SFHMassBulge);
        }
    }

    myfree(save_info->buffer_output_gals);

    return EXIT_SUCCESS;
}

#undef FREE_GALAXY_OUTPUT_INNER_ARRAY


/*
 * create_hdf5_master_file -- write a top-level HDF5 file linking all per-task
 * output files (Task 0 only).
 *
 * Creates "<OutputDir>/<FileNameGalaxies>.hdf5" and populates it with HDF5
 * external links to each per-task file ("<FileNameGalaxies>_<N>.hdf5") so
 * that readers can access the full catalogue through a single entry point.
 * All tasks other than Task 0 return immediately without doing any work.
 *
 * Returns EXIT_SUCCESS, or a negative SAGE error code on failure.
 */
int32_t create_hdf5_master_file(const struct params *run_params)
{
    // Only Task 0 needs to do stuff from here.
    if(run_params->ThisTask > 0) {
        return EXIT_SUCCESS;
    }

    hid_t master_file_id, group_id, root_group_id;
    char master_fname[2*MAX_STRING_LEN + 6];
    herr_t status;

    // Create the file.
    snprintf(master_fname, 2*MAX_STRING_LEN + 5, "%s/%s.hdf5", run_params->OutputDir, run_params->FileNameGalaxies);

    master_file_id = H5Fcreate(master_fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    CHECK_STATUS_AND_RETURN_ON_FAIL(master_file_id, FILE_NOT_FOUND,
                                    "Can't open file %s for master file writing.\n", master_fname);

    // We will keep track of how many galaxies were saved across all files per snapshot.
    // We do this for each snapshot in the simulation, not only those that are output, to allow easy
    // checking of which snapshots were output.
    int64_t *ngals_allfiles_snap = mycalloc(run_params->SimMaxSnaps, sizeof(*(ngals_allfiles_snap))); // Calloced because initially no galaxies.
    CHECK_POINTER_AND_RETURN_ON_NULL(ngals_allfiles_snap,
                                     "Failed to allocate %d elements of size %zu for ngals_allfiles_snaps.", run_params->SimMaxSnaps,
                                     sizeof(*(ngals_allfiles_snap)));

    // The master file will be accessed as (e.g.,) f["Core0"]["Snap_63"]["StellarMass"].
    // Hence we want to store the external links to the **root** group (i.e., "/").
    root_group_id = H5Gopen2(master_file_id, "/", H5P_DEFAULT);
    CHECK_STATUS_AND_RETURN_ON_FAIL(root_group_id, (int32_t) root_group_id,
                                    "Failed to open the root group for the master file.\nThe file ID was %d\n",
                                    (int32_t) master_file_id);

    // At this point, all the files of all other processors have been created. So iterate over the
    // number of processors and create links within this master file to those files.
    char target_fname[3*MAX_STRING_LEN];
    char core_fname[MAX_STRING_LEN];

    for(int32_t task_idx = 0; task_idx < run_params->NTasks; ++task_idx) {

        snprintf(core_fname, MAX_STRING_LEN - 1, "Core_%d", task_idx);
        snprintf(target_fname, 3*MAX_STRING_LEN - 1, "./%s_%d.hdf5", run_params->FileNameGalaxies, task_idx);

        // Make a symlink to the root of the target file.
        status = H5Lcreate_external(target_fname, "/", root_group_id, core_fname, H5P_DEFAULT, H5P_DEFAULT);
        CHECK_STATUS_AND_RETURN_ON_FAIL(status, (int32_t) status,
                                        "Failed to create an external link to file %s from the master file.\n"
                                        "The group ID was %d and the group name was %s\n", target_fname,
                                        (int32_t) root_group_id, core_fname);
    }

    // We've finished with the linking. Now let's create some attributes and datasets inside the header group.
    group_id = H5Gcreate2(master_file_id, "Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHECK_STATUS_AND_RETURN_ON_FAIL(group_id, (int32_t) group_id,
                                    "Failed to create the Header group for the master file.\nThe file ID was %d\n",
                                    (int32_t) master_file_id);

    // When we're writing the header attributes for the master file, we don't have knowledge of trees.
    // So pass a NULL pointer here instead of `forest_info`.
    status = write_header(master_file_id, NULL, run_params);
    if(status != EXIT_SUCCESS) {
        return status;
    }

    status = H5Gclose(group_id);
    CHECK_STATUS_AND_RETURN_ON_FAIL(status, (int32_t) status,
                                    "Failed to close the Header group for the master file."
                                    "The group ID was %d\n", (int32_t) group_id);

    // Finished creating links.
    status = H5Gclose(root_group_id);
    CHECK_STATUS_AND_RETURN_ON_FAIL(status, (int32_t) status,
                                    "Failed to close root group for the master file %s\n"
                                    "The group ID was %d and the file ID was %d\n", master_fname,
                                    (int32_t) root_group_id, (int32_t) master_file_id);

    // JS: Cleanup cause we're considerate programmers.
    status = H5Fclose(master_file_id);
    CHECK_STATUS_AND_RETURN_ON_FAIL(status, (int32_t) status,
                                    "Failed to close the Master HDF5 file.\nThe file ID was %d\n",
                                    (int32_t) master_file_id);

    myfree(ngals_allfiles_snap);

    return EXIT_SUCCESS;

}

// Local Functions //

/*
 * generate_field_metadata -- populate the field name, description, and HDF5
 * datatype arrays for all NUM_OUTPUT_FIELDS galaxy output properties.
 *
 * Fills field_names[], field_descriptions[], field_units[], and field_dtypes[]
 * with one entry per output field.  Field names are kept identical to the
 * binary output format so that comparison scripts (e.g., tests/sagediff.py)
 * work against both formats without modification.
 *
 * Returns EXIT_SUCCESS, or a negative SAGE error code on failure.
 */
static int32_t generate_field_metadata(char (*field_names)[MAX_STRING_LEN], char (*field_descriptions)[MAX_STRING_LEN],
                                char (*field_units)[MAX_STRING_LEN], hsize_t *field_dtypes)
{

    /* All four tables generated in output order from GALAXY_OUTPUT_FIELDS. */
    char tmp_names[NUM_OUTPUT_FIELDS][MAX_STRING_LEN] = {
#define SAGE_FIELD_NAME(dset, field, ctype, h5t, desc, unit) #dset,
        GALAXY_OUTPUT_FIELDS(SAGE_FIELD_NAME)
#undef SAGE_FIELD_NAME
    };

    char tmp_descriptions[NUM_OUTPUT_FIELDS][MAX_STRING_LEN] = {
#define SAGE_FIELD_DESC(dset, field, ctype, h5t, desc, unit) desc,
        GALAXY_OUTPUT_FIELDS(SAGE_FIELD_DESC)
#undef SAGE_FIELD_DESC
    };

    char tmp_units[NUM_OUTPUT_FIELDS][MAX_STRING_LEN] = {
#define SAGE_FIELD_UNIT(dset, field, ctype, h5t, desc, unit) unit,
        GALAXY_OUTPUT_FIELDS(SAGE_FIELD_UNIT)
#undef SAGE_FIELD_UNIT
    };

    hsize_t tmp_dtype[NUM_OUTPUT_FIELDS] = {
#define SAGE_FIELD_DTYPE(dset, field, ctype, h5t, desc, unit) h5t,
        GALAXY_OUTPUT_FIELDS(SAGE_FIELD_DTYPE)
#undef SAGE_FIELD_DTYPE
    };
    for(int32_t i = 0; i < NUM_OUTPUT_FIELDS; i++) {
        memcpy(field_names[i], tmp_names[i], MAX_STRING_LEN);
        memcpy(field_descriptions[i], tmp_descriptions[i], MAX_STRING_LEN);
        memcpy(field_units[i], tmp_units[i], MAX_STRING_LEN);
        field_dtypes[i] = tmp_dtype[i];
    }

    return EXIT_SUCCESS;
}

/*
 * prepare_galaxy_for_hdf5_output -- copy one GALAXY's properties into the
 * HDF5 struct-of-arrays output buffer for the given snapshot.
 *
 * Appends all output fields of *g (with unit conversions matching
 * prepare_galaxy_for_output() in save_gals_binary.c) into the slot at index
 * num_gals_in_buffer[output_snap_idx] within buffer_output_gals[output_snap_idx].
 * Does not advance the buffer counter -- caller is responsible.
 *
 * Returns EXIT_SUCCESS, or a negative SAGE error code on failure.
 */
static int32_t prepare_galaxy_for_hdf5_output(const struct GALAXY *g, struct save_info *save_info,
                                       const int32_t output_snap_idx,  const struct halo_data *halos,
                                       const int64_t task_forestnr,
                                       const int64_t original_treenr,
                                       const struct params *run_params)
{

    int64_t gals_in_buffer = save_info->num_gals_in_buffer[output_snap_idx];

    save_info->buffer_output_gals[output_snap_idx].SnapNum[gals_in_buffer] = g->SnapNum;

    if(g->Type < SHRT_MIN || g->Type > SHRT_MAX) {
        fprintf(stderr,"Error: Galaxy type = %d can not be represented in 2 bytes\n", g->Type);
        fprintf(stderr,"Converting galaxy type while saving from integer to short will result in data corruption");
        return EXIT_FAILURE;
    }
    save_info->buffer_output_gals[output_snap_idx].Type[gals_in_buffer] = g->Type;
    save_info->buffer_output_gals[output_snap_idx].Regime[gals_in_buffer] = g->Regime;
    save_info->buffer_output_gals[output_snap_idx].FFBRegime[gals_in_buffer] = g->FFBRegime;
    save_info->buffer_output_gals[output_snap_idx].Concentration[gals_in_buffer] = g->Concentration;

    save_info->buffer_output_gals[output_snap_idx].GalaxyIndex[gals_in_buffer] = g->GalaxyIndex;
    save_info->buffer_output_gals[output_snap_idx].CentralGalaxyIndex[gals_in_buffer] = g->CentralGalaxyIndex;

    save_info->buffer_output_gals[output_snap_idx].SAGEHaloIndex[gals_in_buffer] = g->HaloNr;
    save_info->buffer_output_gals[output_snap_idx].SAGETreeIndex[gals_in_buffer] = original_treenr;
    save_info->buffer_output_gals[output_snap_idx].SimulationHaloIndex[gals_in_buffer] = llabs(halos[g->HaloNr].MostBoundID);
    save_info->buffer_output_gals[output_snap_idx].TaskForestNr[gals_in_buffer] = task_forestnr;

    save_info->buffer_output_gals[output_snap_idx].mergeType[gals_in_buffer] = g->mergeType;
    save_info->buffer_output_gals[output_snap_idx].mergeIntoID[gals_in_buffer] = g->mergeIntoID;
    save_info->buffer_output_gals[output_snap_idx].mergeIntoSnapNum[gals_in_buffer] = g->mergeIntoSnapNum;
    save_info->buffer_output_gals[output_snap_idx].dT[gals_in_buffer] = g->dT * run_params->UnitTime_in_s / SEC_PER_MEGAYEAR;

    save_info->buffer_output_gals[output_snap_idx].Posx[gals_in_buffer] = g->Pos[0];
    save_info->buffer_output_gals[output_snap_idx].Posy[gals_in_buffer] = g->Pos[1];
    save_info->buffer_output_gals[output_snap_idx].Posz[gals_in_buffer] = g->Pos[2];

    save_info->buffer_output_gals[output_snap_idx].Velx[gals_in_buffer] = g->Vel[0];
    save_info->buffer_output_gals[output_snap_idx].Vely[gals_in_buffer] = g->Vel[1];
    save_info->buffer_output_gals[output_snap_idx].Velz[gals_in_buffer] = g->Vel[2];

    save_info->buffer_output_gals[output_snap_idx].Spinx[gals_in_buffer] = halos[g->HaloNr].Spin[0];
    save_info->buffer_output_gals[output_snap_idx].Spiny[gals_in_buffer] = halos[g->HaloNr].Spin[1];
    save_info->buffer_output_gals[output_snap_idx].Spinz[gals_in_buffer] = halos[g->HaloNr].Spin[2];

    save_info->buffer_output_gals[output_snap_idx].Len[gals_in_buffer] = g->Len;
    save_info->buffer_output_gals[output_snap_idx].Mvir[gals_in_buffer] = g->Mvir;
    save_info->buffer_output_gals[output_snap_idx].CentralMvir[gals_in_buffer] = get_virial_mass(halos[g->HaloNr].FirstHaloInFOFgroup, halos, run_params);
    save_info->buffer_output_gals[output_snap_idx].Rvir[gals_in_buffer] = get_virial_radius(g->HaloNr, halos, run_params);  // output the actual Rvir, not the maximum Rvir
    save_info->buffer_output_gals[output_snap_idx].Vvir[gals_in_buffer] = get_virial_velocity(g->HaloNr, halos, run_params);  // output the actual Vvir, not the maximum Vvir
    save_info->buffer_output_gals[output_snap_idx].VvirPeak[gals_in_buffer] = g->Vvir;  // the peak-retained value the physics uses
    save_info->buffer_output_gals[output_snap_idx].Vmax[gals_in_buffer] = g->Vmax;
    save_info->buffer_output_gals[output_snap_idx].VelDisp[gals_in_buffer] = halos[g->HaloNr].VelDisp;

    save_info->buffer_output_gals[output_snap_idx].ColdGas[gals_in_buffer] = g->ColdGas;
    save_info->buffer_output_gals[output_snap_idx].StellarMass[gals_in_buffer] = g->StellarMass;
    save_info->buffer_output_gals[output_snap_idx].BulgeMass[gals_in_buffer] = g->BulgeMass;
    save_info->buffer_output_gals[output_snap_idx].HotGas[gals_in_buffer] = g->HotGas;
    save_info->buffer_output_gals[output_snap_idx].EjectedMass[gals_in_buffer] = g->EjectedMass;
    save_info->buffer_output_gals[output_snap_idx].BlackHoleMass[gals_in_buffer] = g->BlackHoleMass;
    save_info->buffer_output_gals[output_snap_idx].ICS[gals_in_buffer] = g->ICS;
    save_info->buffer_output_gals[output_snap_idx].CGMgas[gals_in_buffer] = g->CGMgas;
    save_info->buffer_output_gals[output_snap_idx].MassLoading[gals_in_buffer] = g->MassLoading;
    save_info->buffer_output_gals[output_snap_idx].H2gas[gals_in_buffer] = g->H2gas;
    save_info->buffer_output_gals[output_snap_idx].H1gas[gals_in_buffer] = g->H1gas;

    save_info->buffer_output_gals[output_snap_idx].MetalsColdGas[gals_in_buffer] = g->MetalsColdGas;
    save_info->buffer_output_gals[output_snap_idx].MetalsStellarMass[gals_in_buffer] = g->MetalsStellarMass;
    save_info->buffer_output_gals[output_snap_idx].MetalsBulgeMass[gals_in_buffer] = g->MetalsBulgeMass;
    save_info->buffer_output_gals[output_snap_idx].MetalsHotGas[gals_in_buffer] = g->MetalsHotGas;
    save_info->buffer_output_gals[output_snap_idx].MetalsEjectedMass[gals_in_buffer] = g->MetalsEjectedMass;
    save_info->buffer_output_gals[output_snap_idx].MetalsICS[gals_in_buffer] = g->MetalsICS;
    save_info->buffer_output_gals[output_snap_idx].MetalsCGMgas[gals_in_buffer] = g->MetalsCGMgas;

    save_info->buffer_output_gals[output_snap_idx].tcool[gals_in_buffer] = g->tcool;
    save_info->buffer_output_gals[output_snap_idx].tff[gals_in_buffer] = g->tff;
    save_info->buffer_output_gals[output_snap_idx].tcool_over_tff[gals_in_buffer] = g->tcool_over_tff;
    save_info->buffer_output_gals[output_snap_idx].MachNumber[gals_in_buffer] = g->MachNumber;
    save_info->buffer_output_gals[output_snap_idx].tdeplete[gals_in_buffer] = g->tdeplete;
    save_info->buffer_output_gals[output_snap_idx].H2DepletionTime_Gyr[gals_in_buffer] = g->H2DepletionTime_Gyr;
    save_info->buffer_output_gals[output_snap_idx].RcoolToRvir[gals_in_buffer] = g->RcoolToRvir;
    save_info->buffer_output_gals[output_snap_idx].g_max[gals_in_buffer] = g->g_max;
    save_info->buffer_output_gals[output_snap_idx].r_heat[gals_in_buffer] = g->r_heat;

    float tmp_SfrDisk = 0.0;
    float tmp_SfrBulge = 0.0;
    float tmp_SfrDiskZ = 0.0;
    float tmp_SfrBulgeZ = 0.0;

    // NOTE: in Msun/yr. Divisors come from the substep count actually integrated, not STEPS;
    // see sfr_rate_divisor() in model_misc.h.
    const int sfr_norm = sfr_rate_divisor(g->SubstepsUsed);
    const int met_norm = sfr_metallicity_divisor(g->SubstepsUsed);
    for(int step = 0; step < STEPS; step++) {
        tmp_SfrDisk += g->SfrDisk[step] * run_params->UnitMass_in_g / run_params->UnitTime_in_s * SEC_PER_YEAR / SOLAR_MASS / sfr_norm;
        tmp_SfrBulge += g->SfrBulge[step] * run_params->UnitMass_in_g / run_params->UnitTime_in_s * SEC_PER_YEAR / SOLAR_MASS / sfr_norm;

        if(g->SfrDiskColdGas[step] > 0.0) {
            tmp_SfrDiskZ += g->SfrDiskColdGasMetals[step] / g->SfrDiskColdGas[step] / met_norm;
        }

        if(g->SfrBulgeColdGas[step] > 0.0) {
            tmp_SfrBulgeZ += g->SfrBulgeColdGasMetals[step] / g->SfrBulgeColdGas[step] / met_norm;
        }
    }

    save_info->buffer_output_gals[output_snap_idx].SfrDisk[gals_in_buffer] = tmp_SfrDisk;
    save_info->buffer_output_gals[output_snap_idx].SfrBulge[gals_in_buffer] = tmp_SfrBulge;
    save_info->buffer_output_gals[output_snap_idx].SfrDiskZ[gals_in_buffer] = tmp_SfrDiskZ;
    save_info->buffer_output_gals[output_snap_idx].SfrBulgeZ[gals_in_buffer] = tmp_SfrBulgeZ;
    
    // Conditionally save cumulative SFH arrays if SaveFullSFH is enabled
    if(run_params->SaveFullSFH) {
        // Save cumulative star formation history (stellar mass formed at each snapshot)
        for(int snap = 0; snap < run_params->SimMaxSnaps; snap++) {
            const int idx = gals_in_buffer * run_params->SimMaxSnaps + snap;  // Index into flattened 2D array
            save_info->buffer_output_gals[output_snap_idx].SFHMassDisk[idx] = g->SFHMassDisk[snap];
            save_info->buffer_output_gals[output_snap_idx].SFHMassBulge[idx] = g->SFHMassBulge[snap];
        }
    }

    save_info->buffer_output_gals[output_snap_idx].DiskScaleRadius[gals_in_buffer] = g->DiskScaleRadius;
    save_info->buffer_output_gals[output_snap_idx].BulgeRadius[gals_in_buffer] = g->BulgeRadius;
    save_info->buffer_output_gals[output_snap_idx].MergerBulgeRadius[gals_in_buffer] = g->MergerBulgeRadius;
    save_info->buffer_output_gals[output_snap_idx].InstabilityBulgeRadius[gals_in_buffer] = g->InstabilityBulgeRadius;
    save_info->buffer_output_gals[output_snap_idx].MergerBulgeMass[gals_in_buffer] = g->MergerBulgeMass;
    save_info->buffer_output_gals[output_snap_idx].InstabilityBulgeMass[gals_in_buffer] = g->InstabilityBulgeMass;

    if (g->Cooling > 0.0) {
        save_info->buffer_output_gals[output_snap_idx].Cooling[gals_in_buffer] = log10(g->Cooling * run_params->UnitEnergy_in_cgs / run_params->UnitTime_in_s);
    } else {
        save_info->buffer_output_gals[output_snap_idx].Cooling[gals_in_buffer] = 0.0;
    }

    if (g->Heating > 0.0) {
        save_info->buffer_output_gals[output_snap_idx].Heating[gals_in_buffer] = log10(g->Heating * run_params->UnitEnergy_in_cgs / run_params->UnitTime_in_s);
    } else {
        save_info->buffer_output_gals[output_snap_idx].Heating[gals_in_buffer] = 0.0;
    }

    save_info->buffer_output_gals[output_snap_idx].QuasarModeBHaccretionMass[gals_in_buffer] = g->QuasarModeBHaccretionMass;

    save_info->buffer_output_gals[output_snap_idx].TimeOfLastMajorMerger[gals_in_buffer] = g->TimeOfLastMajorMerger * run_params->UnitTime_in_Megayears;
    save_info->buffer_output_gals[output_snap_idx].TimeOfLastMinorMerger[gals_in_buffer] = g->TimeOfLastMinorMerger * run_params->UnitTime_in_Megayears;

    save_info->buffer_output_gals[output_snap_idx].OutflowRate[gals_in_buffer] = g->OutflowRate * run_params->UnitMass_in_g / run_params->UnitTime_in_s * SEC_PER_YEAR / SOLAR_MASS;
    save_info->buffer_output_gals[output_snap_idx].mdot_cool[gals_in_buffer] = g->mdot_cool * run_params->UnitMass_in_g / run_params->UnitTime_in_s * SEC_PER_YEAR / SOLAR_MASS;
    save_info->buffer_output_gals[output_snap_idx].mdot_stream[gals_in_buffer] = g->mdot_stream * run_params->UnitMass_in_g / run_params->UnitTime_in_s * SEC_PER_YEAR / SOLAR_MASS;

    // ICS assembly tracking
    save_info->buffer_output_gals[output_snap_idx].ICS_disrupt[gals_in_buffer] = g->ICS_disrupt;
    save_info->buffer_output_gals[output_snap_idx].ICS_accrete[gals_in_buffer] = g->ICS_accrete;
    save_info->buffer_output_gals[output_snap_idx].ICS_sum_mt[gals_in_buffer] = g->ICS_sum_mt;

    //infall properties
    if(g->Type != 0) {
        save_info->buffer_output_gals[output_snap_idx].infallMvir[gals_in_buffer] = g->infallMvir;
        save_info->buffer_output_gals[output_snap_idx].infallVvir[gals_in_buffer] = g->infallVvir;
        save_info->buffer_output_gals[output_snap_idx].infallVmax[gals_in_buffer] = g->infallVmax;
        save_info->buffer_output_gals[output_snap_idx].infallStellarMass[gals_in_buffer] = g->infallStellarMass;
        save_info->buffer_output_gals[output_snap_idx].TimeOfInfall[gals_in_buffer] = g->TimeOfInfall;
    } else {
        save_info->buffer_output_gals[output_snap_idx].infallMvir[gals_in_buffer] = 0.0;
        save_info->buffer_output_gals[output_snap_idx].infallVvir[gals_in_buffer] = 0.0;
        save_info->buffer_output_gals[output_snap_idx].infallVmax[gals_in_buffer] = 0.0;
        save_info->buffer_output_gals[output_snap_idx].infallStellarMass[gals_in_buffer] = 0.0;
        save_info->buffer_output_gals[output_snap_idx].TimeOfInfall[gals_in_buffer] = 0.0;
    }

    return EXIT_SUCCESS;
}


/*MS: 23/9/2019 Yes, there appears to be a NULL pointer dereference in the 'SIZEOF_STRUCT_FIELD' but
  the expression is a compile time constant and there is no invalid memory access. That said, C really shouold
 not allow such constructs! */
#define SIZEOF_STRUCT_FIELD(field)    (sizeof(((struct HDF5_GALAXY_OUTPUT *) NULL)->field[0]))

// We created the datasets (e.g., "Snap_43/StellarMass") with 'infinite' dimensions.
// Before we write, we must extend the current dimensions to account for the new values.
// The basic flow for this is:
// Get the dataset ID -> Extend the dataset to the new dimensions -> Get the filespace of the dataset
// -> Select a block of memory that we will add to the current filespace, this is the hyperslab.
// -> Create a dataspace that will hold the data -> Write the data to the group using the new spaces.
// Please refer to the HDF5 documentation for comprehensive explanations. I've probably butchered this...

/* Assumes 'snap_idx', 'field_idx' are set appropriately before invoking the macro */
#define EXTEND_AND_WRITE_GALAXY_DATASET(field_name) {                   \
    char full_field_name[2*MAX_STRING_LEN];                           \
    snprintf(full_field_name, 2*MAX_STRING_LEN - 1,"Snap_%d/%s", run_params->ListOutputSnaps[snap_idx], save_info->name_output_fields[field_idx]); \
    hid_t dataset_id = H5Dopen2(save_info->file_id, full_field_name, H5P_DEFAULT); \
    if(dataset_id < 0) {                                                \
        fprintf(stderr, "Could not access the " #field_name" dataset for output snapshot %d.\n", snap_idx); \
        return (int32_t) dataset_id;                                    \
    }                                                                   \
    hid_t h5_dtype = H5Dget_type(dataset_id);                           \
    if(SIZEOF_STRUCT_FIELD(field_name) != H5Tget_size(h5_dtype)) {      \
        fprintf(stderr,"Error while writing field " #field_name"\n");   \
        fprintf(stderr,"The HDF5 dataset has size %zu bytes but the struct element has size = %zu bytes\n", \
                H5Tget_size(h5_dtype), SIZEOF_STRUCT_FIELD(field_name)); \
        fprintf(stderr,"Perhaps the size of the struct item needs to be updated?\n"); \
        return -1;                                                      \
    }                                                                   \
    status = H5Dset_extent(dataset_id, new_dims);                       \
    if(status < 0) {                                                    \
        fprintf(stderr, "Could not resize the dimensions of the " #field_name" dataset for output snapshot %d.\n" \
                "The dataset ID value is %d. The new dimension values were %d\n", \
                snap_idx, (int32_t) dataset_id, (int32_t) new_dims[0]); \
        return (int32_t) status;                                        \
    }                                                                   \
    hid_t filespace = H5Dget_space(dataset_id);                         \
    if(filespace < 0) {                                                 \
        fprintf(stderr, "Could not retrieve the dataspace of the " #field_name" dataset for output snapshot %d.\n" \
                "The dataset ID value is %d.\n", snap_idx, (int32_t) dataset_id); \
        return (int32_t) filespace;                                     \
    }                                                                   \
    status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, old_dims, NULL, dims_extend, NULL); \
    if(status < 0) {                                                    \
        fprintf(stderr, "Could not select a hyperslab region to add to the filespace of the " #field_name" dataset for output snapshot %d.\n" \
                "The dataset ID value is %d.\n"                         \
                "The old dimensions were %d and we attempted to extend this by %d elements.\n", snap_idx, (int32_t) dataset_id, \
                (int32_t) old_dims[0], (int32_t) dims_extend[0]);       \
        return (int32_t) status;                                        \
    }                                                                   \
    hid_t memspace = H5Screate_simple(1, dims_extend, NULL);            \
    if(memspace < 0) {                                                  \
        fprintf(stderr, "Could not create a new dataspace for the " #field_name" dataset for output snapshot %d.\n" \
                "The length of the dataspace we attempted to created was %d.\n", snap_idx, (int32_t) dims_extend[0]); \
        return (int32_t) memspace;                                      \
    }                                                                   \
    status = H5Dwrite(dataset_id, h5_dtype, memspace, filespace, H5P_DEFAULT, (save_info->buffer_output_gals[snap_idx]).field_name); \
    CHECK_STATUS_AND_RETURN_ON_FAIL(status, (int32_t) status,           \
            "Could not write the dataset for the " #field_name" field for output snapshot %d.\n" \
            "The dataset ID value is %d.\n"                             \
            "The old dimensions were %d and we attempting to extend (and write to) this by %d elements.\n" \
            "The HDF5 datatype was #h5_dtype.\n", snap_idx, (int32_t) dataset_id, \
            (int32_t) old_dims[0], (int32_t) dims_extend[0]);           \
    status = H5Dclose(dataset_id);                                      \
    CHECK_STATUS_AND_RETURN_ON_FAIL(status, (int32_t) status,           \
                                    "Failed to close field number %d for output snapshot number %d\n" \
                                    "The dataset ID was %d\n", field_idx, snap_idx, \
                                    (int32_t) dataset_id);              \
    status = H5Sclose(memspace);                                        \
    CHECK_STATUS_AND_RETURN_ON_FAIL(status, (int32_t) status, \
                                    "Could not close the memory space for the " #field_name" dataset for output snapshot %d.\n" \
                                    "The dataset ID value is %d.\n"     \
                                    "The old dimensions were %d and we attempting to extend this by %d elements.\n", \
                                    snap_idx, (int32_t) dataset_id, (int32_t) old_dims[0], (int32_t) dims_extend[0]); \
    status = H5Tclose(h5_dtype);                                        \
    CHECK_STATUS_AND_RETURN_ON_FAIL(status, (int32_t) status,           \
                                    "Error: Failed to close the datatype for the " #field_name" dataset for output snapshot %d.\n" \
                                    "The dataset ID value is %d.\n"     \
                                    "The old dimensions were %d and we attempting to extend this by %d elements.\n", \
                                    snap_idx, (int32_t) dataset_id, (int32_t) old_dims[0], (int32_t) dims_extend[0]); \
    status = H5Sclose(filespace);                                        \
    CHECK_STATUS_AND_RETURN_ON_FAIL(status, (int32_t) status,           \
                                    "Could not close the filespace for the " #field_name" dataset for output snapshot %d.\n" \
                                    "The dataset ID value is %d.\n"     \
                                    "The old dimensions were %d and we attempting to extend this by %d elements.\n", \
                                    snap_idx, (int32_t) dataset_id, (int32_t) old_dims[0], (int32_t) dims_extend[0]); \
    field_idx++;                                                        \
}


/*
 * trigger_buffer_write -- extend each dataset for snap_idx and write
 * num_to_write buffered galaxies into the new rows.
 *
 * For each of the NUM_OUTPUT_FIELDS extensible datasets in the "Snap_<N>"
 * group: extends the dataset by num_to_write rows (H5Dset_extent), selects the
 * new hyperslab, and writes from the corresponding buffer array.  Called both
 * when the buffer reaches NUM_GALS_PER_BUFFER and by finalize_hdf5_galaxy_files()
 * to flush any remainder.  num_already_written is the offset into the dataset
 * where new rows start.
 *
 * Returns EXIT_SUCCESS, or a negative SAGE error code on failure.
 */
static int32_t trigger_buffer_write(const int32_t snap_idx, const int32_t num_to_write, const int64_t num_already_written,
                             struct save_info *save_info, const struct params *run_params)
{
    herr_t status;

    // To save the galaxies, we must first extend the size of the dataset to accomodate the new data.

    // JS 16/03/19: I've attempted to put these into the HDF5 function calls directly
    // (rather than specifying as arrays).  However, it causes errors...

    // This is the length which we will extended the dataset by.
    hsize_t dims_extend[1];
    dims_extend[0] = (hsize_t) num_to_write;

    // The previous length of the dataset.
    hsize_t old_dims[1];
    old_dims[0] = (hsize_t) num_already_written;

    // Then this is the new length.
    hsize_t new_dims[1];
    new_dims[0] = old_dims[0] + dims_extend[0];

    // This parameter is incremented in every Macro call. It is used to ensure we are
    // accessing the correct dataset.
    int32_t field_idx = 0;

    // We now need to write each property to file.  This is performed in a stack of macros because
    // it's not possible to loop through the members of a struct.
#define SAGE_FIELD_WRITE(dset, field, ctype, h5t, desc, unit) EXTEND_AND_WRITE_GALAXY_DATASET(field);
    GALAXY_OUTPUT_FIELDS(SAGE_FIELD_WRITE)
#undef SAGE_FIELD_WRITE


    // Conditionally write cumulative SFH datasets if SaveFullSFH is enabled
    if(run_params->SaveFullSFH) {
        // Write cumulative SFH datasets (SFHMassDisk, SFHMassBulge)
        const char *cum_sfh_field_names[2] = {"SFHMassDisk", "SFHMassBulge"};
        float *cum_sfh_data_ptrs[2] = {
            save_info->buffer_output_gals[snap_idx].SFHMassDisk,
            save_info->buffer_output_gals[snap_idx].SFHMassBulge
        };
        
        for(int cum_idx = 0; cum_idx < 2; cum_idx++) {
            char full_field_name[2*MAX_STRING_LEN];
            snprintf(full_field_name, 2*MAX_STRING_LEN - 1, "Snap_%d/%s",
                     run_params->ListOutputSnaps[snap_idx], cum_sfh_field_names[cum_idx]);
            
            // Open dataset
            hid_t dataset_cum = H5Dopen2(save_info->file_id, full_field_name, H5P_DEFAULT);
            CHECK_STATUS_AND_RETURN_ON_FAIL(dataset_cum, (int32_t) dataset_cum,
                                            "Could not open cumulative SFH dataset %s", cum_sfh_field_names[cum_idx]);
            
            // Get current dimensions
            hid_t space_cum = H5Dget_space(dataset_cum);
            hsize_t current_dims_cum[2];
            H5Sget_simple_extent_dims(space_cum, current_dims_cum, NULL);
            H5Sclose(space_cum);
            
            // Set new dimensions [old_ngals + num_to_write, SimMaxSnaps]
            hsize_t new_dims_cum[2] = {current_dims_cum[0] + num_to_write, (hsize_t)run_params->SimMaxSnaps};
            status = H5Dset_extent(dataset_cum, new_dims_cum);
            CHECK_STATUS_AND_RETURN_ON_FAIL(status, (int32_t) status,
                                            "Could not extend cumulative SFH dataset %s", cum_sfh_field_names[cum_idx]);
            
            // Select hyperslab in file space
            hid_t filespace_cum = H5Dget_space(dataset_cum);
            hsize_t offset_cum[2] = {current_dims_cum[0], 0};
            hsize_t count_cum[2] = {(hsize_t)num_to_write, (hsize_t)run_params->SimMaxSnaps};
            status = H5Sselect_hyperslab(filespace_cum, H5S_SELECT_SET, offset_cum, NULL, count_cum, NULL);
            CHECK_STATUS_AND_RETURN_ON_FAIL(status, (int32_t) status,
                                            "Could not select hyperslab for cumulative SFH dataset %s", cum_sfh_field_names[cum_idx]);
            
            // Create memory space
            hid_t memspace_cum = H5Screate_simple(2, count_cum, NULL);
            CHECK_STATUS_AND_RETURN_ON_FAIL(memspace_cum, (int32_t) memspace_cum,
                                            "Could not create memspace for cumulative SFH dataset %s", cum_sfh_field_names[cum_idx]);
            
            // Write data
            status = H5Dwrite(dataset_cum, H5T_NATIVE_FLOAT, memspace_cum, filespace_cum, H5P_DEFAULT, cum_sfh_data_ptrs[cum_idx]);
            CHECK_STATUS_AND_RETURN_ON_FAIL(status, (int32_t) status,
                                            "Could not write cumulative SFH dataset %s", cum_sfh_field_names[cum_idx]);
            
            // Cleanup
            H5Sclose(memspace_cum);
            H5Sclose(filespace_cum);
            H5Dclose(dataset_cum);
        }
    }

    // We've performed a write, so future galaxies will overwrite the old data.
    save_info->num_gals_in_buffer[snap_idx] = 0;
    save_info->tot_ngals[snap_idx] += num_to_write;

    return EXIT_SUCCESS;
}

#undef SIZEOF_STRUCT_FIELD
#undef EXTEND_AND_WRITE_GALAXY_DATASET

/*
 * write_header -- write run metadata into the "Header" HDF5 group hierarchy.
 *
 * Creates Header/Simulation (cosmology, box size, particle mass, snapshot
 * redshifts, git ref) and Header/Runtime (physics switches, key model
 * parameters) groups and populates them as scalar HDF5 attributes.  Called
 * once per output file from finalize_hdf5_galaxy_files().
 *
 * Returns EXIT_SUCCESS, or a negative SAGE error code on failure.
 */
static int32_t write_header(hid_t file_id, const struct forest_info *forest_info, const struct params *run_params) {

    // Inside the "Header" group, we split the attributes up inside different groups for usability.
    hid_t sim_group_id = H5Gcreate2(file_id, "Header/Simulation", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHECK_STATUS_AND_RETURN_ON_FAIL(sim_group_id, (int32_t) sim_group_id,
                                    "Failed to create the Header/Simulation group.\nThe file ID was %d\n",
                                    (int32_t) file_id);

    hid_t runtime_group_id = H5Gcreate2(file_id, "Header/Runtime", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHECK_STATUS_AND_RETURN_ON_FAIL(runtime_group_id, (int32_t) runtime_group_id,
                                    "Failed to create the Header/Runtime group.\nThe file ID was %d\n",
                                    (int32_t) file_id);

    hid_t misc_group_id = H5Gcreate2(file_id, "Header/Misc", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    CHECK_STATUS_AND_RETURN_ON_FAIL(misc_group_id, (int32_t) misc_group_id,
                                    "Failed to create the Header/Miscgroup.\nThe file ID was %d\n",
                                    (int32_t) file_id);

    // Simulation information.
    CREATE_STRING_ATTRIBUTE(sim_group_id, "SimulationDir", &run_params->SimulationDir, strlen(run_params->SimulationDir));
    CREATE_STRING_ATTRIBUTE(sim_group_id, "FileWithSnapList", &run_params->FileWithSnapList, strlen(run_params->FileWithSnapList));
    CREATE_SINGLE_ATTRIBUTE(sim_group_id, "LastSnapshotNr", run_params->LastSnapshotNr, H5T_NATIVE_INT);
    CREATE_SINGLE_ATTRIBUTE(sim_group_id, "SimMaxSnaps", run_params->SimMaxSnaps, H5T_NATIVE_INT);

    CREATE_SINGLE_ATTRIBUTE(sim_group_id, "omega_matter", run_params->Omega, H5T_NATIVE_DOUBLE);
    CREATE_SINGLE_ATTRIBUTE(sim_group_id, "omega_lambda", run_params->OmegaLambda, H5T_NATIVE_DOUBLE);
    CREATE_SINGLE_ATTRIBUTE(sim_group_id, "particle_mass", run_params->PartMass, H5T_NATIVE_DOUBLE);
    CREATE_SINGLE_ATTRIBUTE(sim_group_id, "hubble_h", run_params->Hubble_h, H5T_NATIVE_DOUBLE);
    CREATE_SINGLE_ATTRIBUTE(sim_group_id, "num_simulation_tree_files", run_params->NumSimulationTreeFiles, H5T_NATIVE_INT);
    CREATE_SINGLE_ATTRIBUTE(sim_group_id, "box_size", run_params->BoxSize, H5T_NATIVE_DOUBLE);

    // If we're writing the header attributes for the master file, we don't have knowledge of trees.
    if(forest_info != NULL) {
        CREATE_SINGLE_ATTRIBUTE(sim_group_id, "num_trees_this_file", forest_info->nforests_this_task, H5T_NATIVE_LLONG);
        CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "frac_volume_processed", forest_info->frac_volume_processed, H5T_NATIVE_DOUBLE);
    } else {
        const long long nforests_on_master_file = 0;
        CREATE_SINGLE_ATTRIBUTE(sim_group_id, "num_trees_this_file", nforests_on_master_file, H5T_NATIVE_LLONG);

        const double frac_volume_on_master = (run_params->LastFile - run_params->FirstFile + 1)/(double) run_params->NumSimulationTreeFiles;
        CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "frac_volume_processed", frac_volume_on_master, H5T_NATIVE_DOUBLE);
    }

    // Data and version information.
    CREATE_SINGLE_ATTRIBUTE(misc_group_id, "num_cores", run_params->NTasks, H5T_NATIVE_INT);
    CREATE_STRING_ATTRIBUTE(misc_group_id, "sage_data_version", &SAGE_DATA_VERSION, strlen(SAGE_DATA_VERSION));
    CREATE_STRING_ATTRIBUTE(misc_group_id, "sage_version", &SAGE_VERSION, strlen(SAGE_VERSION));
    CREATE_STRING_ATTRIBUTE(misc_group_id, "git_SHA_reference", &GITREF_STR, strlen(GITREF_STR));

    // Output file info.
    CREATE_STRING_ATTRIBUTE(runtime_group_id, "FileNameGalaxies", &run_params->FileNameGalaxies, strlen(run_params->FileNameGalaxies));
    CREATE_STRING_ATTRIBUTE(runtime_group_id, "OutputDir", &run_params->OutputDir, strlen(run_params->OutputDir));
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "FirstFile", run_params->FirstFile, H5T_NATIVE_INT);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "LastFile", run_params->LastFile, H5T_NATIVE_INT);

    // Recipe flags.
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "SFprescription", run_params->SFprescription, H5T_NATIVE_INT);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "AGNrecipeOn", run_params->AGNrecipeOn, H5T_NATIVE_INT);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "SupernovaRecipeOn", run_params->SupernovaRecipeOn, H5T_NATIVE_INT);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "ReionizationOn", run_params->ReionizationOn, H5T_NATIVE_INT);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "DiskInstabilityOn", run_params->DiskInstabilityOn, H5T_NATIVE_INT);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "CGMrecipeOn", run_params->CGMrecipeOn, H5T_NATIVE_INT);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "FIREmodeOn", run_params->FIREmodeOn, H5T_NATIVE_INT);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "FeedbackFreeModeOn", run_params->FeedbackFreeModeOn, H5T_NATIVE_INT);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "BulgeSizeOn", run_params->BulgeSizeOn, H5T_NATIVE_INT);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "H2DiskAreaOption", run_params->H2DiskAreaOption, H5T_NATIVE_INT);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "H2RadialIntegrationOn", run_params->H2RadialIntegrationOn, H5T_NATIVE_INT);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "H2RadialNBins", run_params->H2RadialNBins, H5T_NATIVE_INT);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "H2RadialRMaxFactor", run_params->H2RadialRMaxFactor, H5T_NATIVE_DOUBLE);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "CGMDensityProfile", run_params->CGMDensityProfile, H5T_NATIVE_INT);
    // CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "PrecipCriterionOn", run_params->PrecipCriterionOn, H5T_NATIVE_INT);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "RegimeRandomMode", run_params->RegimeRandomMode, H5T_NATIVE_INT);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "ConcentrationOn", run_params->ConcentrationOn, H5T_NATIVE_INT);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "RamPressureStrippingOn", run_params->RamPressureStrippingOn, H5T_NATIVE_INT);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "SaveFullSFH", run_params->SaveFullSFH, H5T_NATIVE_INT);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "TrackICSAssembly", run_params->TrackICSAssembly, H5T_NATIVE_INT);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "StarburstColdGasOn", run_params->StarburstColdGasOn, H5T_NATIVE_INT);

    // Model parameters.
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "SfrEfficiency", run_params->SfrEfficiency, H5T_NATIVE_DOUBLE);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "FFBMaxEfficiency", run_params->FFBMaxEfficiency, H5T_NATIVE_DOUBLE);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "FFBConcSigma", run_params->FFBConcSigma, H5T_NATIVE_DOUBLE);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "FFBThresholdSlope", run_params->FFBThresholdSlope, H5T_NATIVE_DOUBLE);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "FFBIgnoreRegime", run_params->FFBIgnoreRegime, H5T_NATIVE_INT);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "FFBRandomMode", run_params->FFBRandomMode, H5T_NATIVE_INT);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "MShockMsun", run_params->MShockMsun, H5T_NATIVE_DOUBLE);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "ColdStreamCeilingOn", run_params->ColdStreamCeilingOn, H5T_NATIVE_INT);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "StreamMassFactor", run_params->StreamMassFactor, H5T_NATIVE_DOUBLE);
    // CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "DiskRadiusFactor", run_params->DiskRadiusFactor, H5T_NATIVE_DOUBLE);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "FeedbackReheatingEpsilon", run_params->FeedbackReheatingEpsilon, H5T_NATIVE_DOUBLE);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "FeedbackEjectionEfficiency", run_params->FeedbackEjectionEfficiency, H5T_NATIVE_DOUBLE);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "ReIncorporationFactor", run_params->ReIncorporationFactor, H5T_NATIVE_DOUBLE);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "RadioModeEfficiency", run_params->RadioModeEfficiency, H5T_NATIVE_DOUBLE);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "QuasarModeEfficiency", run_params->QuasarModeEfficiency, H5T_NATIVE_DOUBLE);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "BlackHoleGrowthRate", run_params->BlackHoleGrowthRate, H5T_NATIVE_DOUBLE);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "ThreshMajorMerger", run_params->ThreshMajorMerger, H5T_NATIVE_DOUBLE);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "ThresholdSatDisruption", run_params->ThresholdSatDisruption, H5T_NATIVE_DOUBLE);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "FractionDisruptedToICS", run_params->FractionDisruptedToICS, H5T_NATIVE_DOUBLE);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "DynamicDisruptionSplit", run_params->DynamicDisruptionSplit, H5T_NATIVE_INT32);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "DisruptionSplitAlpha", run_params->DisruptionSplitAlpha, H5T_NATIVE_DOUBLE);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "DisruptionSplitCref", run_params->DisruptionSplitCref, H5T_NATIVE_DOUBLE);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "Yield", run_params->Yield, H5T_NATIVE_DOUBLE);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "RecycleFraction", run_params->RecycleFraction, H5T_NATIVE_DOUBLE);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "FracZleaveDisk", run_params->FracZleaveDisk, H5T_NATIVE_DOUBLE);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "Reionization_z0", run_params->Reionization_z0, H5T_NATIVE_DOUBLE);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "Reionization_zr", run_params->Reionization_zr, H5T_NATIVE_DOUBLE);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "EnergySN", run_params->EnergySN, H5T_NATIVE_DOUBLE);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "EtaSN", run_params->EtaSN, H5T_NATIVE_DOUBLE);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "RedshiftPowerLawExponent", run_params->RedshiftPowerLawExponent, H5T_NATIVE_DOUBLE);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "SNEnergyConservationOn", run_params->SNEnergyConservationOn, H5T_NATIVE_INT);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "MaxSNEnergyCoupling", run_params->MaxSNEnergyCoupling, H5T_NATIVE_DOUBLE);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "RamPressureEpsilon", run_params->RamPressureEpsilon, H5T_NATIVE_DOUBLE);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "BaryonFrac", run_params->BaryonFrac, H5T_NATIVE_DOUBLE);

    /* Numerical resolution and forest decomposition.  Not physics, but the
       results depend on them: SubstepResolution sets the integration
       resolution the model is calibrated at, and the decomposition scheme
       changes how forests are split across ranks, which the regime/FFB draws
       are sensitive to. */
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "SubstepResolution", run_params->SubstepResolution, H5T_NATIVE_DOUBLE);
    /* The enum-valued parameters need lvalues: CREATE_SINGLE_ATTRIBUTE takes the
       address of its argument, so a cast expression cannot be passed directly. */
    const int32_t forest_dist_scheme = (int32_t) run_params->ForestDistributionScheme;
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "ForestDistributionScheme", forest_dist_scheme, H5T_NATIVE_INT);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "ExponentForestDistributionScheme", run_params->Exponent_Forest_Dist_Scheme, H5T_NATIVE_DOUBLE);

    /* Input provenance: which trees produced this file, and in what format. */
    const int32_t tree_type_id = (int32_t) run_params->TreeType;
    const int32_t output_format_id = (int32_t) run_params->OutputFormat;
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "TreeType", tree_type_id, H5T_NATIVE_INT);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "OutputFormat", output_format_id, H5T_NATIVE_INT);
    CREATE_STRING_ATTRIBUTE(runtime_group_id, "TreeName", &run_params->TreeName, strlen(run_params->TreeName));

    // Misc runtime Parameters.
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "UnitLength_in_cm", run_params->UnitLength_in_cm, H5T_NATIVE_DOUBLE);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "UnitMass_in_g", run_params->UnitMass_in_g, H5T_NATIVE_DOUBLE);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "UnitVelocity_in_cm_per_s", run_params->UnitVelocity_in_cm_per_s, H5T_NATIVE_DOUBLE);

    // Redshift at each snapshot.
    hsize_t dims[1];
    dims[0] = run_params->Snaplistlen;

    CREATE_AND_WRITE_1D_ARRAY(file_id, "Header/snapshot_redshifts", dims, run_params->ZZ, H5T_NATIVE_DOUBLE);

    // Output snapshots.
    dims[0] = run_params->NumSnapOutputs;

    CREATE_AND_WRITE_1D_ARRAY(file_id, "Header/output_snapshots", dims, run_params->ListOutputSnaps, H5T_NATIVE_INT);
    CREATE_SINGLE_ATTRIBUTE(runtime_group_id, "NumOutputs", run_params->NumSnapOutputs, H5T_NATIVE_INT);

    herr_t status = H5Gclose(sim_group_id);
    CHECK_STATUS_AND_RETURN_ON_FAIL(status, (int32_t) status,
                                    "Failed to close Header/Simulation group.\n"
                                    "The group ID was %d\n", (int32_t) sim_group_id);

    status = H5Gclose(runtime_group_id);
    CHECK_STATUS_AND_RETURN_ON_FAIL(status, (int32_t) status,
                                    "Failed to close Header/Runtime group.\n"
                                    "The group ID was %d\n", (int32_t) runtime_group_id);

    status = H5Gclose(misc_group_id);
    CHECK_STATUS_AND_RETURN_ON_FAIL(status, (int32_t) status,
                                    "Failed to close Header/Misc group.\n"
                                    "The group ID was %d\n", (int32_t) misc_group_id);

    return EXIT_SUCCESS;
}

#undef CREATE_AND_WRITE_1D_ARRAY
#undef CREATE_SINGLE_ATTRIBUTE
#undef CREATE_STRING_ATTRIBUTE

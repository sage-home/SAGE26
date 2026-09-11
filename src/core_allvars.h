/*
 * core_allvars.h -- central data structures: GALAXY, run_params, halo_data,
 * and all tree-reader I/O structs.
 *
 * Every physics module and I/O file includes this header.  The GALAXY struct
 * is the main per-galaxy record; run_params holds the parameter-file values
 * and per-run state; halo_data is the on-disk lhalo binary halo record.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/* define off_t as a 64-bit long integer */
#define _FILE_OFFSET_BITS 64

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

#ifdef HDF5
#include <hdf5.h>
#endif

#include "macros.h"
#include "core_simulation.h"


enum Valid_TreeTypes
{
    /* The number of input tree types supported
       This consists of two parts, the first part
       dictates the tree kind (i.e., what the bytes mean), while
       the second part dictates the actual format on disk (i.e.,
       how to read/cast the bytes from disk) */
    lhalo_binary = 0,
    lhalo_hdf5 = 1,
    genesis_hdf5 = 2,
    consistent_trees_ascii = 3,
    consistent_trees_hdf5 = 4,
    gadget4_hdf5 = 5,
    num_tree_types
};

/* Struct for making hdf5 file reading a bit easier */
struct HDF5_METADATA_NAMES
{
    char name_NTrees[MAX_STRING_LEN];
    char name_totNHalos[MAX_STRING_LEN];
    char name_TreeNHalos[MAX_STRING_LEN];
    char name_ParticleMass[MAX_STRING_LEN];
    char name_NumSimulationTreeFiles[MAX_STRING_LEN];
};


enum Valid_OutputFormats
{
    /* The number of output formats supported by sage */
    sage_binary = 0, /* will be deprecated after version 1 release*/
    sage_hdf5 = 1,
    lhalo_binary_output = 2, /* special functionality to convert *any* supported input mergertree into a lhalo-binary format */
    num_output_format_types
};

enum Valid_Forest_Distribution_Schemes
{
    /* Determines the compute cost for each forest as a function
     of the number of halos in the forest*/
    uniform_in_forests = 0, /* returns 1 (i.e., all forests have the same cost regardless of forest size)*/
    linear_in_nhalos = 1, /* returns nhalos (i.e., bigger forests have a bigger compute cost) */
    quadratic_in_nhalos = 2, /* return nhalos^2 as the compute cost*/
    exponent_in_nhalos = 3,/* returns nhalos^exponent */
    generic_power_in_nhalos = 4, /* returns pow(nhalos, exponent) */
    num_forest_weight_types
};


/* do not use '0' as an enum since that '0' usually
   indicates 'success' on POSIX systems */
enum sage_error_types {
    /* start off with a large number */
    FILE_NOT_FOUND=1 << 12,
    SNAPSHOT_OUT_OF_RANGE,
    INVALID_OPTION_IN_PARAMS,
    OUT_OF_MEMBLOCKS,
    MALLOC_FAILURE,
    INVALID_PTR_REALLOC_REQ,
    INTEGER_32BIT_TOO_SMALL,
    NULL_POINTER_FOUND,
    FILE_READ_ERROR,
    FILE_WRITE_ERROR,
    INVALID_FILE_POINTER,
    INVALID_FILE_DESCRIPTOR,
    INVALID_VALUE_READ_FROM_FILE,
    PARSE_ERROR,
    INVALID_MEMORY_ACCESS_REQUESTED,
    HDF5_ERROR,
};


/* This structure contains the properties used within the code */
struct GALAXY
{
    int32_t   SnapNum;
    int32_t   Type;       /* 0=central; 1=satellite with subhalo; 2=orphan satellite; 3=merged (dead) */
    int32_t   Regime;     /* 0=CGM-dominated (cold-flow/precipitation); 1=hot-halo (classical); set by determine_and_store_regime() */
    int32_t   FFBRegime;  /* 0=standard SF; 1=feedback-free burst active; set by determine_and_store_ffb_regime() */
    float     FFBRandom;  /* persistent random number for sigmoid-based FFB determination (drawn at galaxy creation) */
    float     RegimeRandom; /* persistent random number for sigmoid-based CGM/Hot regime determination (drawn at galaxy creation; used when RegimeRandomMode==1) */

    int32_t   GalaxyNr;   /* index within the current forest's galaxy array */
    int32_t   CentralGal; /* index of the FOF central galaxy in the current galaxy array */
    int32_t   HaloNr;     /* index of the host halo in the halos[] array */
    long long MostBoundID;
    uint64_t GalaxyIndex;        /* unique output ID encoding file/forest/galaxy; see generate_galaxy_index() in core_save.c */
    uint64_t CentralGalaxyIndex; /* GalaxyIndex of this galaxy's FOF central */

    int32_t   mergeType;  /* 0=none; 1=minor merger; 2=major merger; 3=disk instability; 4=disrupt to ICS */
    int32_t   mergeIntoID;       /* GalaxyIndex of the merger target (output-array index before offset correction) */
    int32_t   mergeIntoSnapNum;  /* snapshot at which the merger is recorded */
    float dT;                    /* total time interval for this snapshot step [code time units] */

    /* (sub)halo properties */
    float Pos[3];        /* comoving position, copied from the host halo [Mpc/h] */
    float Vel[3];        /* peculiar velocity, copied from the host halo [km/s] */
    int   Len;           /* number of simulation particles in the host (sub)halo */
    float Mvir;          /* virial mass [10^10 Msun/h] */
    float deltaMvir;     /* change in virial mass since previous snapshot [10^10 Msun/h] */
    float Rvir;          /* virial radius [Mpc/h] */
    float Vvir;          /* virial circular velocity [km/s] */
    float Vmax;          /* maximum circular velocity of the (sub)halo [km/s] */
    float Concentration; /* NFW concentration parameter; computed if ConcentrationOn > 0 */

    /* baryonic reservoirs [all in 10^10 Msun/h] */
    float ColdGas;
    float StellarMass;
    float BulgeMass;
    float HotGas;
    float EjectedMass;
    float BlackHoleMass;
    float ICS;       /* intracluster/intragroup stellar component */
    float CGMgas;    /* CGM-regime gas reservoir (Regime==0 only) */
    float H2gas;     /* molecular hydrogen mass */
    float H1gas;     /* atomic hydrogen mass */

    /* metals [shadow each baryonic reservoir; same units] */
    float MetalsColdGas;
    float MetalsStellarMass;
    float MetalsBulgeMass;
    float MetalsHotGas;
    float MetalsEjectedMass;
    float MetalsICS;
    float MetalsCGMgas;

    /* per-substep SFR trackers; converted to Msun/yr at output */
    float SfrDisk[STEPS];              /* disk SFR per substep [10^10 Msun/h / code time] */
    float SfrBulge[STEPS];             /* bulge (starburst) SFR per substep [10^10 Msun/h / code time] */
    float SfrDiskColdGas[STEPS];       /* ColdGas at each substep, for output metallicity of disk SF [10^10 Msun/h] */
    float SfrDiskColdGasMetals[STEPS]; /* MetalsColdGas at each substep [10^10 Msun/h] */
    float SfrBulgeColdGas[STEPS];      /* ColdGas at each substep of bulge SF [10^10 Msun/h] */
    float SfrBulgeColdGasMetals[STEPS];/* MetalsColdGas at each substep of bulge SF [10^10 Msun/h] */
    int32_t SubstepsUsed;              /* effective_steps actually integrated over this snapshot. The Sfr*
                                          arrays above hold STEPS bins but accumulate one entry per substep,
                                          so the output average must divide by this, not by STEPS. Equals
                                          STEPS whenever the adaptive path does not fire (see evolve_galaxies). */

    /* full star formation history - tracks stellar mass formed at each snapshot */
    float SFHMassDisk[ABSOLUTEMAXSNAPS];   /* stellar mass formed in disk at each snapshot [10^10 Msun/h] */
    float SFHMassBulge[ABSOLUTEMAXSNAPS];  /* stellar mass formed in bulge (starbursts) at each snapshot [10^10 Msun/h] */
    float ICS_disrupt;                     /* cumulative stellar mass disrupted to ICS (assembly tracking) [10^10 Msun/h] */
    float ICS_accrete;                     /* cumulative ICS accreted from satellites (assembly tracking) [10^10 Msun/h] */
    float ICS_sum_mt;                      /* mass-weighted accumulator [10^10 Msun/h * code time]: sum of m*t at ICS deposition;
                                              mean ICS-assembly lookback = ICS_sum_mt / (ICS_disrupt + ICS_accrete) */

    /* misc */
    float DiskScaleRadius; /* exponential disk scale radius [Mpc/h] */
    float SpinSmooth[3];   /* main-branch running mean of the halo spin vector [(Mpc/h)(km/s)];
                              used only when DiskRadiusOn >= 2, carried forward with the galaxy */
    float BulgeRadius;     /* effective (half-mass) bulge radius [Mpc/h] */
    float MergTime;        /* dynamical-friction merger clock; counts down to 0 [code time units]; >999 = unset */
    double Cooling;        /* total cooling luminosity this snapshot [code energy / code time] */
    double Heating;        /* total AGN heating luminosity this snapshot [code energy / code time] */
    float r_heat;          /* AGN radio-mode heating radius [Mpc/h]; suppresses cooling gas at r < r_heat. Ratchet-only (no decay), capped at Rvir in the CGM regime. */
    float QuasarModeBHaccretionMass; /* BH mass accreted in quasar mode this snapshot [10^10 Msun/h] */
    float TimeOfLastMajorMerger;     /* lookback time to z=0 at last major merger [code time units]; -1 = never; written out in Myr */
    float TimeOfLastMinorMerger;     /* lookback time to z=0 at last minor merger [code time units]; -1 = never; written out in Myr */
    float OutflowRate;           /* SN-driven gas outflow rate [10^10 Msun/h / code time] */
    float RcoolToRvir;           /* ratio of cooling radius to virial radius at last cooling evaluation */

    /* infall properties -- values frozen at the moment a galaxy first becomes a satellite */
    float infallMvir;        /* Mvir at infall [10^10 Msun/h] */
    float infallVvir;        /* Vvir at infall [km/s] */
    float infallVmax;        /* Vmax at infall [km/s] */
    float infallStellarMass; /* StellarMass at infall [10^10 Msun/h] */
    float TimeOfInfall;      /* snapshot number at infall (stored as float); -1 = never a satellite */

    float MassLoading; /* SN mass-loading factor eta = M_ejected / M_* for the current SF episode */

    /* Cooling diagnostics (set each snapshot by the active cooling recipe) */
    float tcool;             /* cooled gas rate [Msun/Gyr] */
    float tff;               /* free-fall time at the precipitation radius [Gyr] */
    float tcool_over_tff;    /* ratio used for precipitation threshold test */
    float MachNumber;        /* inflow Mach number of the volume-filling CGM phase,
                              * Stern et al. (2019) Eq 28: t_cool/t_ff = 0.845 / Mach.
                              * h-corrected, unlike tcool_over_tff (see CHANGELOG). -1 if unset. */
    float tdeplete;          /* gas depletion timescale from the current SF episode [code time units] */
    float H2DepletionTime_Gyr; /* molecular depletion time from K13 prescription [Gyr] */

    /* bulge properties -- split by formation channel for morphology tracking */
    float MergerBulgeMass;        /* bulge mass built via mergers [10^10 Msun/h] */
    float InstabilityBulgeMass;   /* bulge mass built via disk instability [10^10 Msun/h] */
    float MergerBulgeRadius;      /* half-mass radius of merger-built bulge [Mpc/h] */
    float InstabilityBulgeRadius; /* half-mass radius of instability-built bulge [Mpc/h] */

    float mdot_cool;    /* instantaneous CGM cooling rate onto the disk [10^10 Msun/h / code time] */
    float mdot_stream;  /* cold-stream inflow rate from CGMgas [10^10 Msun/h / code time] */

    double g_max; /* maximum gravitational instability growth rate for BK25 FFB threshold (dimensionless) */
};



/* auxiliary halo data */
struct halo_aux_data
{
    int32_t DoneFlag;
    int32_t HaloFlag;
    int32_t NGalaxies;
    int FirstGalaxy;
    int output_snap_n;
};


struct lhalotree_info {
    int64_t nforests;/* number of forests to process */

    /* lhalotree format only has int32_t for nhalos per forest */
    int64_t *nhalos_per_forest;/* number of halos to read, nforests elements */

    union {
        int *fd;/* the file descriptor for each forest (i.e., which file descriptor to read this forest from) nforests elements*/
#ifdef HDF5
        hid_t *h5_fd;/* contains the HDF5 file descriptor for each forest */
#endif
    };
    off_t *bytes_offset_for_forest;/* where to start reading the files, nforests elements */

    union {
        int *open_fds;/* contains numfiles elements of open file descriptors, numfiles elements */
#ifdef HDF5
        hid_t *open_h5_fds;/* contains numfiles elements of open HDF5 file descriptors */
#endif
    };
    int32_t numfiles;/* number of unique files being processed by this task,  must be >=1 and <= lastfile - firstfile + 1 */
    int32_t unused;/* unused, but present for alignment */
};

struct ctrees_info {
    //different from totnforests; only stores forests to be processed by ThisTask when in MPI mode
    //in serial mode, ``forests_info->ctr.nforests == forests_info->totnforests``)
    union {
        int64_t nforests;
        int64_t nforests_this_task;
    };
    int64_t ntrees;

    void *column_info;/* stored as a void * to avoid including parse_ctrees.h here*/

    /* forest level quantities */
    int64_t *ntrees_per_forest;/* contains nforests elements */
    int64_t *start_treenum_per_forest;/* contains nforests elements */

    /* tree level quantities */
    int *tree_fd;/* contains ntrees elements */
    off_t *tree_offsets;/* contains ntrees elements */

    /* file level quantities */
    int *open_fds;/* contains numfiles elements of open file descriptors */
    int32_t numfiles;/* total number of files the forests are spread over (BOX_DIVISIONS^3 per Consistent trees terminology) */
    int32_t unused;/* unused, but present for alignment */
};

/* place-holder for future AHF i/o capabilities */
struct ahf_info {
    int64_t nforests;
    void *some_yet_to_be_implemented_ptr;
};

#ifdef HDF5
struct genesis_info {
    union{
        int64_t nforests;/* number of forests to process on this task */
        int64_t nforests_this_task;/* shadowed for convenience */
    };

    int64_t start_forestnum;/* Global forestnumber to start processing from */
    int64_t maxforestsize; /* max. number of halos in any one single forest on any task */
    int64_t *offset_for_global_forestnum;/* What would be the offset to add to file-local 'forestnum' to get the global forest num
                                            that is needed to access the metadata ("*foreststats*.hdf5") file  -- shape (lastfile + 1, ) */
    int64_t *halo_offset_per_snap;/* Stores the current halo offsets to read from at each snapshot -- shape (maxsnaps, ).
                                     Initialised to all 0's for every new file and incremented as forests are read in. This details
                                     adds a loop-dependency - where later forests can not be correctly processed before all
                                     preceeding forests have been processed. It had to be implemented this way because otherwise
                                     the amount of RAM required to store the matrix offsets_per_forest_per_snap (with shape
                                     '[nforests, maxsnaps]' would have been a roadblock in the future. */
    hid_t meta_fd;/* file descriptor for the metadata file*/
    hid_t *h5_fds;/* contains all the file descriptors for the individual files -- shape (lastfile + 1, ) */

    int32_t min_snapnum; /* smallest snapshot to process (inclusive, >= 0), across all forests*/
    int32_t maxsnaps;/* maxsnaps == max_snap_num + 1, largest snapshot to process across all forests */
    int32_t totnfiles;/* total number of files requested to be processed (across all tasks)*/
    int32_t numfiles;/* total number of files to process on ThisTask (>=1)*/
    int32_t start_filenum;/* Which is the first file that this task is going to process  */
    int32_t curr_filenum; /* What file is currently being worked on --
                              required to reset the halo_offset_per_snap at the beginning of every new file */

};

struct ctrees_h5_info {
    //different from totnforests; only stores forests to be processed by ThisTask when in MPI mode
    //in serial mode, ``forests_info->ctr.nforests == forests_info->totnforests``)
    union {
        int64_t nforests;
        int64_t nforests_this_task;
    };

    /* file level quantities */
    hid_t meta_fd; /* file descriptor for the metadata file */
    hid_t *h5_file_groups; /* contains all the file descriptors for the individual files -- shape (lastfile + 1, ) */
    hid_t *h5_forests_group; /* contains the file descriptors for the 'Forests' group in the SOA case */
    char snap_field_name[16]; /*  some of the Uchuu files have 'Snap_num', others have 'Snap_idx' as the snapshot field name
                               This variable contains the correct field name, as determined from the input file during forests
                              reading init */
    int8_t snap_field_is_double;/* some of the Uchuu files accidentally wrote out the snapshot field as double instead of int64_t ->
                                this flag is set during the forests reading init to correctly read the snapshot field */
    int8_t *contig_halo_props;/* Contains whether or not the halos are in contiguous order -- shape (lastfile + 1) */
    int32_t totnfiles;/* total number of files that the simulation is spread across*/
    int32_t start_filenum;/* the first file processed on this task*/
    int32_t end_filenum; /* the last file processed on this task (inclusive) */
};

struct gadget4_info {
    int64_t nforests;/* number of forests to process on this task, scalar */

    int64_t *nhalos_per_forest; /* number of halos per forest, nforests elements*/

    int32_t numfiles;/* number of unique files being processed by this task,  must be >=1 and <= lastfile - firstfile + 1 */
    hid_t *open_h5_fds;/* contains open HDF5 file descriptors,  contains numfiles elements */

    // Unlike all the other mergertree formats, in the Gadget4 mergertree format, a single forest
    // can be spread over multiple files (potentially >> 1). Therefore, we need to know the range of files
    // that the forest is spread across. The loop over files should go from
    // ``[ start_fd_index_for_forest[iforest], end_fd_index_for_forests[iforest] ]`` (inclusive).
    // Within the loop, the reading for the first file needs to start at ``offset_in_first_file_for_forests[iforest]``
    // The number of halos that should be read in a file is ``min(halos_left_in_file, num_halos_left_to_read_in_forest )``
    //      ii) end file for forest -> read min()

    int32_t *start_h5_fd_index; /* contains the  index into open_h5_fds (starting) HDF5 file descriptor for each forest
                                    filenr-based indexing into open_h5_fds -> i.e., assumes that open_h5_fds can
                                    be indexed by (at least) [start_filenum, end_filenum] , nforests elements */

    int16_t *num_files_per_forest; /* contains the number of files that the forest is split across (used to index the HDF5 file descriptor for each forest), nforests elements */
    int32_t **nhalos_per_file_per_forest; /* irregular 2-D matrix, containing the number of halos within *each* file that this forest is spread over,
                                            dimension is at least [1, nforests], and set to [num_files_for_forests[iforest], nforests]
                                            loading code for `iforest` would look like:

                                                const int32_t numfiles = num_files_for_forests[iforest];
                                                int32_t h5_fd_index = start_h5_fd_index[iforest];
                                                int64_t start_offset = offset_in_first_file_for_forests[iforest];
                                                for(int32_t i=0;i<numfiles;i++) {
                                                    const int64_t numhalos_thisfile = nhalos_per_file_per_forest[i][iforest];
                                                    assert(numhalos_thisfile > 0);
                                                    hid_t hfd = open_h5_fds[h5_fd_index];
                                                    assert(hfd > 0);
                                                    READ_PARTIAL_HALOS_HDF5(hfd, start_offset, numhalos_thisfile);
                                                    h5_fd_index++;
                                                    start_offset = 0;
                                                }
                                            */
    int64_t *offset_in_nhalos_first_file_for_forests; /* offset counted in nhalos contained in all preceeding forests,
                                                        where to start reading the forest in the first files, nforests elements */
};
#endif

struct forest_info {
    union {
        struct lhalotree_info lht;
        struct ctrees_info ctr;
        struct ahf_info ahf;
#ifdef HDF5
        struct genesis_info gen;
        struct ctrees_h5_info ctr_h5;
        struct gadget4_info gadget4;
#endif
    };

    /* Run-level quantities */
    int64_t totnforests;  // Total number of forests across **all** input tree files.
    int64_t totnhalos; //Total number of halos across **all** input tree files (if it can be calculated ahead of time, otherwise set to 0 e.g., in case of Consistent-Trees ascii)
    double frac_volume_processed; // Fraction of the simulation volume processed by **this** task.
    // We assume that each of the input tree files span the same volume. Hence by summing the
    // number of trees processed by each task from each file, we can determine the
    // fraction of the simulation volume that this task processes.  We weight this summation by the
    // number of trees in each file because some files may have more/less trees whilst still spanning the
    // same volume (e.g., a void would contain few trees whilst a dense knot would contain many).
    int32_t firstfile;//The first file processed in this run (i.e., over all tasks)
    int32_t lastfile;//The last file processed in this run (i.e., over all tasks)

    /* Task level quantities -> unique for each task */
    int64_t nforests_this_task; // Total number of forests processed by **this** task.
    int64_t nhalos_this_task;// Total number of halos to be processed by **this** task (if it can be calculated ahead of time, otherwise set to 0 e.g., in case of Consistent-Trees ascii)

    /* Forest-level quantities (per task) */
    int32_t *FileNr; // The file number that each forest needs to be read from. For formats where an individual tree may be
                     // split across multiple files (e.g., Gadget4), this field contains the starting file number (i.e., where the
                    // first halos within the tree are to be found)
    int64_t *original_treenr; // The (file-local) tree number from the original tree files.
                              // Necessary because Task N's "Tree 0" could start at the middle of a file.
};

struct save_info {
    union {
        int *save_fd; // Contains the open file to write to for each output.
#ifdef HDF5
        hid_t file_id;  // HDF5 only writes to a single file per processor.
#endif
    };

    int64_t *tot_ngals; // Number of galaxies **per snapshot**.
    int32_t **forest_ngals; // Number of galaxies **per snapshot** **per tree**; forest_ngals[snap][forest].

#ifdef HDF5
    char **name_output_fields;
    hsize_t *field_dtypes;

    hid_t *group_ids;

    int32_t num_output_fields;

    int32_t buffer_size;
    int32_t *num_gals_in_buffer;
    struct HDF5_GALAXY_OUTPUT *buffer_output_gals;
#endif

};


struct params
{
    int32_t    FirstFile;    /* first and last file for processing; only relevant for lhalotree style files (binary or hdf5) */
    int32_t    LastFile;

    char   OutputDir[MAX_STRING_LEN];
    char   FileNameGalaxies[MAX_STRING_LEN];
    char   TreeName[MAX_STRING_LEN];
    char   TreeExtension[MAX_STRING_LEN]; // If the trees are in HDF5, they will have a .hdf5 extension. Otherwise they have no extension.
    char   SimulationDir[MAX_STRING_LEN];
    char   FileWithSnapList[MAX_STRING_LEN];

    /* cosmological parameters (read from parameter file) */
    double Omega;        /* matter density parameter (z=0) */
    double OmegaLambda;  /* dark energy density parameter (z=0) */
    double PartMass;     /* N-body particle mass [10^10 Msun/h] */
    double Hubble_h;     /* dimensionless Hubble parameter h (H0 = 100 h km/s/Mpc) */
    double BoxSize;      /* simulation box side length [Mpc/h] */

    /* supernova energy and mass-loading parameters */
    double EnergySNcode; /* SN energy per unit stellar mass in code units */
    double EnergySN;     /* SN energy per event in cgs (erg) */
    double EtaSNcode;    /* SN rate per unit stellar mass in code units */
    double EtaSN;        /* number of SN per solar mass of stars formed */

    /* moving for alignment */
    int32_t NumSimulationTreeFiles;

    /* recipe flags */
    int32_t    SFprescription;
    int32_t    AGNrecipeOn;
    int32_t    SupernovaRecipeOn;
    int32_t    ReionizationOn;
    int32_t    DiskInstabilityOn;
    int32_t    CGMrecipeOn;
    int32_t    CGMDensityProfile;  // 0: uniform, 1: NFW, 2: beta-profile,
                                 // 3: Stern+21 cooling flow (rho ~ r^-1.6, evaluated at
                                 //    R_circ = 0.05 Rvir, T^(s) = 1.2 T_vir)
    // int32_t    PrecipCriterionOn; // Which factors of the Voit t_cool/t_ff precipitation rate
                                 // mdot = S((10 - r)/2) * (M_CGM - M_eq)/t_ff are applied:
                                 // 0: neither -- mdot = M_CGM / t_ff for every CGM halo
                                 // 1: both (default, the submitted rate)
                                 // 2: M_eq only -- drop the f_inflow sigmoid
                                 // 3: sigmoid only -- drop the condensation term
                                 // 4: neither, but keeping the hand-over to standard
                                 //    cooling that mode 0 skips (the 2x2 reference)
                                 // 5: SAGE16 cold accretion, mdot = M_CGM/(Rvir/Vvir)
                                 //    (= sqrt(2) x mode 0; no hand-over)
                                 // 6: nothing
    int32_t    FIREmodeOn;
    int32_t    RegimeRandomMode;     // 0: fresh random draw each snapshot (default, original behaviour); 1: use the persistent RegimeRandom assigned at galaxy creation (deterministic regime evolution driven by mass)
    int32_t    ColdStreamCeilingOn;  // Cold-stream shut-off below z_crit.
                                  // 0: hard z_crit cut for M > Mshock (published behaviour)
                                  // 1: Dekel & Birnboim (2006) eqs 39-41, smooth -- z_crit emerges
    double     StreamMassFactor;  // f in Dekel & Birnboim (2006) eqs 40-41; order a few, they use 3.
    // double     DiskRadiusFactor;  // Angular-momentum retention factor f_j multiplying the
                                  // Mo+98 disk scale radius. 1.0 = full retention (default).
    // int32_t    DiskRadiusOn;      // Disk scale radius model:
                                  // 0: published behaviour -- Mo+98 from the instantaneous halo spin,
                                  //    unbounded, and a fallback that returns 0 when Rvir == 0
                                  // 1: as 0, plus a working Rvir fallback (from Len*PartMass) and a
                                  //    [DISK_RADIUS_MIN_FRAC, DiskRadiusMaxFrac] * Rvir bound
                                  // 2: as 1, but built from a running mean of the halo spin *vector*
                                  //    over a halo dynamical time. Suppresses the snapshot-to-snapshot
                                  //    jitter in r_d (3x) and shrinks r_d by a near-uniform ~9%; does
                                  //    NOT remove the low-particle-count bias in |j|, which is
                                  //    correlated between snapshots (see docs/physics/disk_sizes.md)
    // double     DiskRadiusMaxFrac; // Ceiling on r_d / Rvir when DiskRadiusOn > 0. The default 0.15
                                  // corresponds to lambda ~ 0.21 and moves ~4% of Millennium
                                  // galaxies at z=0; set very large to disable the ceiling.
    double     GasDiskRadiusFactor; // chi: ratio of the atomic-gas scale length to the stellar/H2
                                  // scale length, applied in the HI ionisation truncation only.
                                  // 1.0 = cospatial (default, published behaviour); observed disks
                                  // have chi ~ 1.5-2.
    double     MShockMsun;   // Dekel & Birnboim (2006) virial-shock stability mass [Msun].
                             // Sets which of two baryon cycles a halo follows, so it is a
                             // physics parameter rather than a constant; exposed for the
                             // sensitivity test requested in referee Major Comment 9.
    // int32_t    PreventiveHeatingOn;  // Non-AGN preventive suppression of the cooling flow:
    //                               // 0: off (published behaviour)
    //                               // 1: halo-mass gate, hot regime (Regime==1) only
    //                               // 2: halo-mass gate, both regimes
    //                               // 3: as 1, but combined with the AGN suppression by taking the
    //                               //    stronger of the two rather than multiplying them
    //                               // 4: as 3, both regimes
    //                               // 5: Voit t_cool/t_ff ceiling on the hot-regime cooling rate
    //                               // 6: gravitational (halo-accretion) heating offset
    //                               // Modes 1/2 multiply, which double-counts at z=0 where the r_heat
    //                               // ratchet has already saturated; modes 3/4 do not.
    //                               // See preventive_suppression() in model_cooling_heating.c.
    // double     PreventiveHeatingMass;   // M_prev [Msun]: halo mass at which the cooling flow is
    //                                     // suppressed by 50%. Default 1e12.
    // double     PreventiveHeatingSlope;  // Exponent in f = 1/(1 + (Mvir/M_prev)^slope). Default 2.0.
    // double     PreventiveHeatingEfficiency; // epsilon for mode 6: fraction of the halo's accretion
                                        // energy thermalised in the corona. Default 0.02.
    int32_t    ConcentrationOn;   // 0: off, 1: Ishiyama+21 lookup table, 2: Vmax/Vvir from simulation, 3: hybrid (Vmax/Vvir, infall-frozen for satellites)
    int32_t    FeedbackFreeModeOn;  // 0: off, 1: Li+24 mass sigmoid, 2: BK25 sharp, 3: BK25 stored-c sharp, 4: BK25 log-normal c scatter, 5: Li+24 mass sharp (no sigmoid), 6: Li+24 sigmoid + H2 SF, 7: BK25 log-normal c scatter + H2 SF
    int32_t    FFBIgnoreRegime;     // 0: FFB restricted to CGM-regime (Regime=0) halos; 1: allow FFB in hot-regime halos too
    int32_t    FFBRandomMode;       // 0: draw a fresh random each snapshot (DEFAULT, published behaviour -- galaxies move in and out of FFB); 1: use the persistent FFBRandom assigned at galaxy creation (FFB status fixed per galaxy)
    int32_t    BulgeSizeOn;   // 0: off; 1: Shen+03 eq. 33; 2: Shen+03 eq. 32 two-regime; 3: Tonini+16 separate merger/instability bulges
    int32_t    H2DiskAreaOption;          // 0 = pi*r_s^2, 1 = pi*(3*r_s)^2, 2 = 2*pi*r_s^2 (central Sigma_0)
    int32_t    H2RadialIntegrationOn;     // 0: single-slab area (uses H2DiskAreaOption); 1: radial integration of exponential disk
    int32_t    H2RadialNBins;             // radial bins for integration (default 25)
    double     H2RadialRMaxFactor;        // R_max = factor * r_s (default 5.0)
    int32_t    SaveFullSFH;               // 0 = save only the snapshot-averaged SfrDisk/SfrBulge (default),
                                          // 1 = additionally save the per-snapshot SFHMassDisk/SFHMassBulge
                                          // histories. Those accumulate stellar mass, not rate, so they are
                                          // correct at any substep count (unlike the Sfr* rate bins).
    int32_t    TrackICSAssembly;          // 0 = off, 1 = track ICS_disrupt and ICS_accrete
    int32_t    StarburstColdGasOn;        // 0: starbursts use H2 (follows SFprescription); 1: all non-FFB starbursts use cold gas

    /* baryonic physics calibration parameters */
    double RecycleFraction;       /* fraction of stellar mass returned to cold gas by SN */
    double Yield;                 /* metal yield per unit stellar mass locked up */
    double FracZleaveDisk;        /* fraction of SN-enriched gas that leaves the disk (vs stays in ColdGas) */
    double ReIncorporationFactor; /* rate at which ejected gas re-accretes onto the hot halo */
    double ThreshMajorMerger;     /* mass ratio above which a merger is 'major' [dimensionless] */
    double BaryonFrac;            /* cosmic baryon fraction Omega_b/Omega_m [dimensionless] */
    double SfrEfficiency;         /* SF efficiency per dynamical time [dimensionless] */
    double FFBMaxEfficiency;      /* maximum SF efficiency in the feedback-free burst regime [dimensionless] */
    double FFBConcSigma;      // sigma_c for log-normal concentration scatter (ln c); typical ~0.2 (Jing 2000, Bullock+01)
    double FFBThresholdSlope; // exponent n in M_vir,FFB ~ ((1+z)/10)^n; -6.2 (Li+24) is the default. The
                              // normalisation is pinned at z=9, so varying this pivots the threshold about
                              // that redshift; used to test whether the slope is degenerate with alpha_FFB.
    double FeedbackReheatingEpsilon;   /* SN mass-loading: reheated mass per unit stars formed [dimensionless] */
    double FeedbackEjectionEfficiency; /* fraction of SN energy available to eject gas from the halo [dimensionless] */
    double RadioModeEfficiency;   /* radio-mode AGN heating efficiency [dimensionless] */
    double QuasarModeEfficiency;  /* quasar-mode BH accretion efficiency during mergers [dimensionless] */
    double BlackHoleGrowthRate;   /* BH growth normalisation per merger [dimensionless] */
    double Reionization_z0;       /* redshift at which the filter mass reaches its peak (Kravtsov+04 z0) */
    double Reionization_zr;       /* redshift at which reionization completes (Kravtsov+04 zr) */
    double ThresholdSatDisruption;/* satellite disrupted when Mvir/(baryonic mass) drops below this [dimensionless] */
    double FractionDisruptedToICS;  // Fraction of disrupted satellite stellar mass that goes to ICS (rest goes to BCG)
    int32_t DynamicDisruptionSplit;  // 0: fixed fraction; 1: mass-ratio f_ICL = 1-(Msub/Mhost)^alpha; 2: concentration-weighted
    double SubstepResolution;        // global multiplier on the adaptive substep count (floor STEPS and cap MAX_STEPS both scale by this); default 1.0. Runtime knob for convergence / N-invariance testing without recompiling.
    int32_t RamPressureStrippingOn;  // 1 = on (DEFAULT): Gunn & Gott (1972) ram-pressure stripping of satellite ColdGas, applied once per snapshot (see model_ram_pressure.c). 0 = off. Independent of the (always-on) analytic hot/CGM-phase stripping.
    double RamPressureEpsilon;       // order-unity prefactor on the ram pressure P_ram = eps * rho_host * v_sat^2; default 1.0. Absorbs the disk-orientation geometry uncertainty (face-on vs edge-on infall). Only used when RamPressureStrippingOn == 1.
    double DisruptionSplitAlpha;     // Base exponent for mass-ratio dependence of ICL fraction (DynamicDisruptionSplit>=1)
    double DisruptionSplitCref;      // Reference concentration for concentration weighting (DynamicDisruptionSplit=2)
    double RedshiftPowerLawExponent; /* exponent of the (1+z) term in the FIRE mass-loading scaling (Muratov+15); default 1.25 */
    int32_t SNEnergyConservationOn;  // 1 = bound the FIRE ejection energy by the supernova energy actually available (DEFAULT); 0 = off (unbounded coupling, the pre-2026 published behaviour). Only acts when FIREmodeOn == 1.
    double MaxSNEnergyCoupling;      // cap on the effective coupling eps_eff = FeedbackEjectionEfficiency * f_FIRE when SNEnergyConservationOn == 1; default 2.0, i.e. E_FB <= m_* eta_SN E_SN (all of the SN energy). 1.0 caps at half.

    // int32_t CGMsimpleInflowOn;  // 0 = off (default, published behaviour); 1 = simple CGM inflow model (mdot_stream = M_CGM / t_ff, no cooling flow, no precipitation threshold, no cold streams)   
    int32_t KarpovModeOn;  // 0 = off (default, published behaviour); 1 = Karpov+2020 supernova feedback model (mdot_outflow = eta_SN * SFR, no energy budget, no cooling flow, no precipitation threshold, no cold streams)
    /* code unit definitions (set from parameter file; all other unit fields derived from these) */
    double UnitLength_in_cm;          /* 1 code length = this many cm (default: 1 Mpc/h) */
    double UnitVelocity_in_cm_per_s;  /* 1 code velocity = this many cm/s (default: 1 km/s) */
    double UnitMass_in_g;             /* 1 code mass = this many grams (default: 10^10 Msun) */

    /* derived unit conversions (computed by core_init.c from the three above) */
    double UnitTime_in_s;
    double RhoCrit;               /* critical density in code units */
    double UnitPressure_in_cgs;
    double UnitDensity_in_cgs;
    double UnitCoolingRate_in_cgs;
    double UnitEnergy_in_cgs;
    double UnitTime_in_Megayears;
    double G;       /* gravitational constant in code units */
    double Hubble;  /* Hubble constant in code units */

    /* reionization filter-mass scale factors (Kravtsov+2004 / Gnedin 2000) */
    double a0;  /* scale factor at which the filter mass M_F reaches its peak */
    double ar;  /* scale factor at which reionization completes (used in M_F integral) */

    int32_t nsnapshots;
    int32_t LastSnapshotNr;
    int32_t SimMaxSnaps;
    int32_t NumSnapOutputs;
    int32_t Snaplistlen;
    enum Valid_TreeTypes TreeType;
    enum Valid_OutputFormats OutputFormat;

    /* The combination of  ForestDistributionScheme = generic_power_in_nhalos and
       exponent_for_forest_dist_scheme = 0.7 seems to produce good work-load
       balance across MPI on the 512 Genesis test dataset - MS 16/01/2020 */
    enum Valid_Forest_Distribution_Schemes ForestDistributionScheme;
    double Exponent_Forest_Dist_Scheme;

    /* GalaxyIndex encoding multipliers: GalaxyIndex = FileNr*FileNr_Mulfac + ForestNr*ForestNr_Mulfac + GalaxyNr */
    int64_t FileNr_Mulfac;
    int64_t ForestNr_Mulfac;

    int32_t ListOutputSnaps[ABSOLUTEMAXSNAPS];
    //Essentially creating an alias so that the indecipherable 'ZZ'
    //can be interpreted to contain 'redshift' values
    union {
        double ZZ[ABSOLUTEMAXSNAPS];
        double redshift[ABSOLUTEMAXSNAPS];
    };

    //Similarly: 'AA' contains the scale_factors corresponding to
    //each snapshot
    union {
        double AA[ABSOLUTEMAXSNAPS];
        double scale_factors[ABSOLUTEMAXSNAPS];
    };

    double *Age;

    int32_t interrupted;/* to re-print the progress-bar */

    int32_t ThisTask;
    int32_t NTasks;
};


#ifdef __cplusplus
}
#endif

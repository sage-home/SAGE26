/*
 * core_read_parameter_file.c -- parameter file parser.
 *
 * Provides read_parameter_file(), which opens and parses a SAGE parameter file
 * in the key=value format, validating that all required keys are present and
 * that no unrecognised keys appear.  After parsing, converts string-valued
 * parameters (TreeType, OutputFormat, ForestDistributionScheme) to their
 * canonical enum values and applies post-read defaults and validation.
 *
 * SAGE26 -- released under MIT (see LICENSE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h> /* for isblank()*/

#include "core_allvars.h"
#include "core_mymalloc.h"
#include "model_misc.h" /* sf_prescription_tracks_h2() for option-combination checks */

enum datatypes {
    DOUBLE = 1,
    STRING = 2,
    INT = 3
};

#define MAXTAGS          300  /* Max number of parameters */
#define MAXTAGLEN         50  /* Max number of characters in the string param tags */

/* compare_ints_descending -- qsort comparator, descending order. */
static int compare_ints_descending (const void* p1, const void* p2);

static int compare_ints_descending (const void* p1, const void* p2)
{
    int i1 = *(int*) p1;
    int i2 = *(int*) p2;
    if (i1 < i2) {
        return 1;
    } else if (i1 == i2) {
        return 0;
    } else {
        return -1;
    }
 }

/*
 * read_parameter_file -- parse a SAGE parameter file into *run_params.
 *
 * Reads key=value pairs from fname, matching each key against a hard-coded
 * table of MAXTAGS registered parameters.  After the full file is read,
 * verifies that all required parameters were set and that no unknown keys were
 * present.  Then converts TreeType/OutputFormat/ForestDistribution string
 * values to their enum equivalents and sorts ListOutputSnaps in descending
 * order.  Returns EXIT_SUCCESS or a positive error count on failure.
 */
int read_parameter_file(const char *fname, struct params *run_params)
{
    int errorFlag = 0;
    int *used_tag = NULL;
    char my_treetype[MAX_STRING_LEN], my_outputformat[MAX_STRING_LEN], my_forest_dist_scheme[MAX_STRING_LEN];
    int NParam = 0;
    char ParamTag[MAXTAGS][MAXTAGLEN + 1];
    char OrigParamTag[MAXTAGS][MAXTAGLEN + 1];
    int  ParamID[MAXTAGS];
    int  ParamRequired[MAXTAGS];
    void *ParamAddr[MAXTAGS];

    /* Ensure that all strings will be NULL terminated */
    for(int i=0;i<MAXTAGS;i++) {
        ParamTag[i][MAXTAGLEN] = '\0';
        OrigParamTag[i][MAXTAGLEN] = '\0';
    }

    NParam = 0;

#ifdef VERBOSE
    const int ThisTask = run_params->ThisTask;

    if(ThisTask == 0) {
        fprintf(stdout, "\nreading parameter file:\n\n");
    }
#endif

    /* Pre-initialize optional string parameters with defaults */
    strncpy(my_outputformat,      "sage_hdf5",              MAXTAGLEN);
    strncpy(my_forest_dist_scheme,"generic_power_in_nhalos",MAXTAGLEN);

    /* Pre-initialize optional numeric parameters with defaults.
       Required parameters are left uninitialised -- they must appear in the file. */
    run_params->NumSnapOutputs             = -1;
    run_params->ReionizationOn             = 1;
    run_params->SupernovaRecipeOn          = 1;
    run_params->DiskInstabilityOn          = 1;
    run_params->SFprescription             = 1;
    run_params->AGNrecipeOn                = 2;
    run_params->H2DiskAreaOption           = 1;
    run_params->H2RadialIntegrationOn      = 1;
    run_params->H2RadialNBins              = 25;
    run_params->H2RadialRMaxFactor         = 5.0;
    run_params->CGMrecipeOn                = 1;
    run_params->CGMDensityProfile          = 0;
    // run_params->PrecipCriterionOn          = 0; /* both factors of the precipitation rate */
    run_params->RegimeRandomMode           = 0;   /* default: fresh draw each snapshot (published behaviour); 1 makes the regime persistent per galaxy */ /* (hard-code once published)*/
    run_params->FIREmodeOn                 = 1;
    run_params->RedshiftPowerLawExponent   = 1.25;
    run_params->SNEnergyConservationOn     = 1;   /* default: on -- neither the reheating nor the ejection term may spend more than the SN energy available */ /* (hard-code once published)*/
    run_params->MaxSNEnergyCoupling        = 2.0; /* cap on eps_halo * f_FIRE: E_FB <= m_* eta_SN E_SN (the whole SN budget) */ /* (hard-code once published)*/
    run_params->FFBMaxEfficiency           = 0.2;
    run_params->FFBConcSigma               = 0.2;
    run_params->FFBThresholdSlope          = -6.2;
    run_params->ConcentrationOn            = 3;
    run_params->FeedbackFreeModeOn         = 1;
    run_params->FFBIgnoreRegime            = 1;  /* (hard-code once published)*/
    run_params->FFBRandomMode              = 0;   /* default: fresh draw each snapshot (published behaviour) -- galaxies move in and out of FFB, sustaining a transient low-z FFB population. 1 fixes each galaxy's quantile at creation, which removes both. */ /* (hard-code once published)*/
    run_params->BulgeSizeOn                = 3;
    run_params->SaveFullSFH                = 0;
    run_params->TrackICSAssembly           = 1;
    run_params->StarburstColdGasOn         = 1;
    run_params->DynamicDisruptionSplit     = 2;
    run_params->SubstepResolution          = 1.0; /* default: unscaled adaptive substeps (STEPS floor, MAX_STEPS cap) */
    run_params->RamPressureStrippingOn     = 1;   /* default: on -- Gunn & Gott (1972) ISM stripping of satellites. Set 0 for the legacy no-ISM-stripping behaviour. */
    run_params->RamPressureEpsilon         = 1.0; /* default: unscaled ram pressure P_ram = rho_host * v_sat^2 */
    run_params->ThreshMajorMerger          = 0.3;
    run_params->RecycleFraction            = 0.43;
    run_params->ReIncorporationFactor      = 0.15;
    run_params->ColdStreamCeilingOn        = 0;     /* 0 reproduces published behaviour */ /* (remove once published)*/
    run_params->StreamMassFactor           = 3.0;   /* Dekel & Birnboim (2006) adopt f = 3 */ /* (remove once published)*/
    // run_params->DiskRadiusFactor           = 1.0;   /* f_j: 1.0 reproduces published behaviour exactly */ /* (remove once published)*/
    // run_params->PreventiveHeatingOn        = 0;      /* 0 reproduces published behaviour bit-for-bit */
    // run_params->PreventiveHeatingMass      = 1.0e12; /* Msun; 50% cooling suppression at this halo mass */
    // run_params->PreventiveHeatingSlope     = 2.0;    /* f = 1/(1 + (Mvir/M_prev)^slope) */
    // run_params->PreventiveHeatingEfficiency = 0.02;  /* mode 6: accretion-energy coupling epsilon */
    // run_params->DiskRadiusOn               = 0;     /* 0 reproduces published behaviour bit-for-bit 1: + working Rvir fallback and a bound on r_d/Rvir; 2: + spin vector smoothed over a halo dynamical time (removes the low-Len bias in |j|) */ 
    // run_params->DiskRadiusMaxFrac          = 0.15;  /* ceiling on r_d/Rvir; used only when DiskRadiusOn > 0 */ 
    run_params->GasDiskRadiusFactor        = 1.0;   /* chi = 1.0: atomic disk cospatial with the stellar disk (published behaviour) */
    run_params->MShockMsun                 = 6.0e11;
    run_params->EnergySN                   = 1.0e51;
    run_params->EtaSN                      = 5.0e-3;
    run_params->Yield                      = 0.025;
    run_params->FracZleaveDisk             = 0.0;
    run_params->SfrEfficiency              = 0.05;
    run_params->FeedbackReheatingEpsilon   = 2.9;
    run_params->FeedbackEjectionEfficiency = 0.3;
    run_params->BlackHoleGrowthRate        = 0.015;
    run_params->RadioModeEfficiency        = 0.08;
    run_params->QuasarModeEfficiency       = 0.005;
    run_params->Reionization_z0            = 8.0;
    run_params->Reionization_zr            = 7.0;
    run_params->ThresholdSatDisruption     = 1.0;
    run_params->FractionDisruptedToICS     = 0.8;
    run_params->DisruptionSplitAlpha       = 0.25;
    run_params->DisruptionSplitCref        = 10.0;
    run_params->Exponent_Forest_Dist_Scheme = 0.7;

    // run_params->CGMsimpleInflowOn         = 1; /* 0: full CGM recipe, 1: simple inflow (no precipitation) */
    run_params->KarpovModeOn              = 0; /* 0: full Karpov+2023 recipe, 1: low-metallicity floor (Z/Z_sun = 0.01) for reheated and ejected gas */

/* Register a parameter: tag name, address, type, required (1) or optional with default (0) */
#define REG(tag, addr, type, req) do {         \
    strncpy(ParamTag[NParam], tag, MAXTAGLEN); \
    ParamAddr[NParam]    = (addr);             \
    ParamID[NParam]      = (type);             \
    ParamRequired[NParam]= (req);              \
    NParam++;                                  \
} while(0)

    /* ---- Required: I/O paths ---- */
    REG("FileNameGalaxies",       run_params->FileNameGalaxies,          STRING, 1);
    REG("OutputDir",              run_params->OutputDir,                  STRING, 1);
    REG("TreeType",               my_treetype,                            STRING, 1);
    REG("TreeName",               run_params->TreeName,                   STRING, 1);
    REG("SimulationDir",          run_params->SimulationDir,              STRING, 1);
    REG("FileWithSnapList",       run_params->FileWithSnapList,           STRING, 1);
    REG("LastSnapshotNr",         &(run_params->LastSnapshotNr),          INT,    1);
    REG("FirstFile",              &(run_params->FirstFile),               INT,    1);
    REG("LastFile",               &(run_params->LastFile),                INT,    1);
    REG("NumSimulationTreeFiles", &(run_params->NumSimulationTreeFiles),  INT,    1);

    /* ---- Required: cosmology and simulation units ---- */
    REG("UnitVelocity_in_cm_per_s", &(run_params->UnitVelocity_in_cm_per_s), DOUBLE, 1);
    REG("UnitLength_in_cm",         &(run_params->UnitLength_in_cm),          DOUBLE, 1);
    REG("UnitMass_in_g",            &(run_params->UnitMass_in_g),             DOUBLE, 1);
    REG("Hubble_h",                 &(run_params->Hubble_h),                  DOUBLE, 1);
    REG("Omega",                    &(run_params->Omega),                     DOUBLE, 1);
    REG("OmegaLambda",              &(run_params->OmegaLambda),               DOUBLE, 1);
    REG("BaryonFrac",               &(run_params->BaryonFrac),                DOUBLE, 1);
    REG("PartMass",                 &(run_params->PartMass),                  DOUBLE, 1);
    REG("BoxSize",                  &(run_params->BoxSize),                   DOUBLE, 1);

    /* ---- Optional: output and code settings ---- */
    REG("NumOutputs",                       &(run_params->NumSnapOutputs),             INT,    0);
    REG("OutputFormat",                     my_outputformat,                           STRING, 0);
    REG("ForestDistributionScheme",         my_forest_dist_scheme,                     STRING, 0);
    REG("ExponentForestDistributionScheme", &(run_params->Exponent_Forest_Dist_Scheme),DOUBLE, 0);

    /* ---- Optional: recipe on/off flags ---- */
    REG("ReionizationOn",        &(run_params->ReionizationOn),       INT, 0);
    REG("SupernovaRecipeOn",     &(run_params->SupernovaRecipeOn),    INT, 0);
    REG("DiskInstabilityOn",     &(run_params->DiskInstabilityOn),    INT, 0);
    REG("SFprescription",        &(run_params->SFprescription),       INT, 0);
    REG("AGNrecipeOn",           &(run_params->AGNrecipeOn),          INT, 0);
    REG("CGMrecipeOn",           &(run_params->CGMrecipeOn),          INT, 0);
    REG("CGMDensityProfile",     &(run_params->CGMDensityProfile),    INT, 0);
    // REG("PrecipCriterionOn",     &(run_params->PrecipCriterionOn),    INT, 0);
    REG("RegimeRandomMode",      &(run_params->RegimeRandomMode),     INT, 0);
    REG("FIREmodeOn",            &(run_params->FIREmodeOn),           INT, 0);
    REG("ConcentrationOn",       &(run_params->ConcentrationOn),      INT, 0);
    REG("FeedbackFreeModeOn",    &(run_params->FeedbackFreeModeOn),   INT, 0);
    REG("FFBIgnoreRegime",       &(run_params->FFBIgnoreRegime),      INT, 0);
    REG("FFBRandomMode",         &(run_params->FFBRandomMode),        INT, 0);
    REG("BulgeSizeOn",           &(run_params->BulgeSizeOn),          INT, 0);
    REG("SaveFullSFH",           &(run_params->SaveFullSFH),          INT, 0);
    REG("TrackICSAssembly",      &(run_params->TrackICSAssembly),     INT, 0);
    REG("StarburstColdGasOn",    &(run_params->StarburstColdGasOn),   INT, 0);
    REG("DynamicDisruptionSplit",&(run_params->DynamicDisruptionSplit),INT, 0);
    REG("SubstepResolution",     &(run_params->SubstepResolution),     DOUBLE, 0);
    REG("RamPressureStrippingOn",   &(run_params->RamPressureStrippingOn),   INT, 0);
    REG("RamPressureEpsilon",       &(run_params->RamPressureEpsilon),       DOUBLE, 0);
    REG("H2DiskAreaOption",      &(run_params->H2DiskAreaOption),     INT, 0);
    REG("H2RadialIntegrationOn", &(run_params->H2RadialIntegrationOn),INT, 0);
    REG("H2RadialNBins",         &(run_params->H2RadialNBins),        INT, 0);

    /* ---- Optional: model parameters ---- */
    REG("ThreshMajorMerger",          &(run_params->ThreshMajorMerger),          DOUBLE, 0);
    REG("RecycleFraction",            &(run_params->RecycleFraction),            DOUBLE, 0);
    REG("ReIncorporationFactor",      &(run_params->ReIncorporationFactor),      DOUBLE, 0);
    REG("ColdStreamCeilingOn",        &(run_params->ColdStreamCeilingOn),        INT,    0);
    REG("StreamMassFactor",           &(run_params->StreamMassFactor),           DOUBLE, 0);
    // REG("DiskRadiusFactor",           &(run_params->DiskRadiusFactor),           DOUBLE, 0);
    // REG("DiskRadiusOn",               &(run_params->DiskRadiusOn),               INT,    0);
    // REG("PreventiveHeatingOn",        &(run_params->PreventiveHeatingOn),        INT,    0);
    // REG("PreventiveHeatingMass",      &(run_params->PreventiveHeatingMass),      DOUBLE, 0);
    // REG("PreventiveHeatingSlope",     &(run_params->PreventiveHeatingSlope),     DOUBLE, 0);
    // REG("PreventiveHeatingEfficiency",&(run_params->PreventiveHeatingEfficiency),DOUBLE, 0);
    // REG("DiskRadiusMaxFrac",          &(run_params->DiskRadiusMaxFrac),          DOUBLE, 0);
    REG("GasDiskRadiusFactor",        &(run_params->GasDiskRadiusFactor),        DOUBLE, 0);
    REG("MShockMsun",                 &(run_params->MShockMsun),                 DOUBLE, 0);
    REG("EnergySN",                   &(run_params->EnergySN),                   DOUBLE, 0);
    REG("EtaSN",                      &(run_params->EtaSN),                      DOUBLE, 0);
    REG("Yield",                      &(run_params->Yield),                      DOUBLE, 0);
    REG("FracZleaveDisk",             &(run_params->FracZleaveDisk),             DOUBLE, 0);
    REG("SfrEfficiency",              &(run_params->SfrEfficiency),              DOUBLE, 0);
    REG("FeedbackReheatingEpsilon",   &(run_params->FeedbackReheatingEpsilon),   DOUBLE, 0);
    REG("FeedbackEjectionEfficiency", &(run_params->FeedbackEjectionEfficiency), DOUBLE, 0);
    REG("BlackHoleGrowthRate",        &(run_params->BlackHoleGrowthRate),        DOUBLE, 0);
    REG("RadioModeEfficiency",        &(run_params->RadioModeEfficiency),        DOUBLE, 0);
    REG("QuasarModeEfficiency",       &(run_params->QuasarModeEfficiency),       DOUBLE, 0);
    REG("Reionization_z0",            &(run_params->Reionization_z0),            DOUBLE, 0);
    REG("Reionization_zr",            &(run_params->Reionization_zr),            DOUBLE, 0);
    REG("ThresholdSatDisruption",     &(run_params->ThresholdSatDisruption),     DOUBLE, 0);
    REG("FractionDisruptedToICS",     &(run_params->FractionDisruptedToICS),     DOUBLE, 0);
    REG("DisruptionSplitAlpha",       &(run_params->DisruptionSplitAlpha),       DOUBLE, 0);
    REG("DisruptionSplitCref",        &(run_params->DisruptionSplitCref),        DOUBLE, 0);
    REG("H2RadialRMaxFactor",         &(run_params->H2RadialRMaxFactor),         DOUBLE, 0);
    REG("FFBMaxEfficiency",           &(run_params->FFBMaxEfficiency),           DOUBLE, 0);
    REG("FFBConcSigma",               &(run_params->FFBConcSigma),               DOUBLE, 0);
    REG("FFBThresholdSlope",          &(run_params->FFBThresholdSlope),          DOUBLE, 0);
    REG("RedshiftPowerLawExponent",   &(run_params->RedshiftPowerLawExponent),   DOUBLE, 0);
    REG("SNEnergyConservationOn",     &(run_params->SNEnergyConservationOn),     INT, 0);
    REG("MaxSNEnergyCoupling",        &(run_params->MaxSNEnergyCoupling),        DOUBLE, 0);
    // REG("CGMsimpleInflowOn",          &(run_params->CGMsimpleInflowOn),          INT, 0);
    REG("KarpovModeOn",               &(run_params->KarpovModeOn),               INT, 0);

#undef REG

    /* Save original tag names before the parse loop zeroes them out for duplicate detection.
       Both arrays are MAXTAGLEN+1 with index MAXTAGLEN pre-set to '\0'. */
    for(int i = 0; i < NParam; i++) {
        memcpy(OrigParamTag[i], ParamTag[i], MAXTAGLEN + 1);
    }

    used_tag = mymalloc(sizeof(int) * NParam);
    for(int i=0; i<NParam; i++) {
        used_tag[i]=1;
    }

    FILE *fd = fopen(fname, "r");
    if (fd == NULL) {
        fprintf(stderr,"Parameter file '%s' not found.\n", fname);
        return FILE_NOT_FOUND;
    }

    char buffer[MAX_STRING_LEN];
    while(fgets(&(buffer[0]), MAX_STRING_LEN, fd) != NULL) {
        char buf1[MAX_STRING_LEN], buf2[MAX_STRING_LEN];
        char fmt[MAX_STRING_LEN];
        snprintf(fmt, MAX_STRING_LEN, "%%%ds %%%ds[^\n]", MAX_STRING_LEN-1, MAX_STRING_LEN-1);
        if(sscanf(buffer, fmt, buf1, buf2) < 2) {
            continue;
        }

        if(buf1[0] == '%' || buf1[0] == '-') { /* the second condition is checking for output snapshots -- that line starts with "->" */
            continue;
        }

        /* Allowing for spaces in the filenames (but requires comments to ALWAYS start with '%' or ';') */
        int buf2len = strnlen(buf2, MAX_STRING_LEN-1);
        for(int i=0;i<buf2len;i++) {  /* BUG FIX: Changed <= to < to avoid buffer over-read */
            if(buf2[i] == '%' || buf2[i] == ';' || buf2[i] == '#') {
                int null_pos = i;
                //Ignore all preceding whitespace
                for(int j=i-1;j>=0;j--) {
                    null_pos = isblank(buf2[j]) ? j:null_pos;
                }
                buf2[null_pos] = '\0';
                break;
            }
        }
        buf2len = strnlen(buf2, MAX_STRING_LEN-1);
        while(buf2len > 0 && isblank(buf2[buf2len-1])) {
            buf2len--;
        }
        buf2[buf2len] = '\0';

        int j=-1;
        for(int i = 0; i < NParam; i++) {
            if(strncasecmp(buf1, ParamTag[i], MAX_STRING_LEN-1) == 0) {
                j = i;
                ParamTag[i][0] = 0;
                used_tag[i] = 0;
                break;
            }
        }

        if(j >= 0) {
            /* strtod/strtol instead of atof/atoi: a malformed numeric value
               (e.g. a typo like "O.05") must be a startup error, not a silent 0. */
            char *endptr = NULL;
            switch (ParamID[j])
                {
                case DOUBLE:
                    *((double *) ParamAddr[j]) = strtod(buf2, &endptr);
                    if(endptr == buf2 || *endptr != '\0') {
                        fprintf(stderr, "Error in file %s:   Value '%s' for parameter '%s' is not a valid number.\n",
                                fname, buf2, buf1);
                        errorFlag = 1;
                    }
                    break;
                case STRING:
                    snprintf(ParamAddr[j], MAX_STRING_LEN, "%s", buf2);
                    break;
                case INT:
                    *((int *) ParamAddr[j]) = (int) strtol(buf2, &endptr, 10);
                    if(endptr == buf2 || *endptr != '\0') {
                        fprintf(stderr, "Error in file %s:   Value '%s' for parameter '%s' is not a valid integer.\n",
                                fname, buf2, buf1);
                        errorFlag = 1;
                    }
                    break;
                }
        } else {
            fprintf(stderr, "Error in file %s:   Tag '%s' not allowed or multiply defined.\n", fname, buf1);
            errorFlag = 1;
        }
    }
    fclose(fd);

    const size_t outlen = strlen(run_params->OutputDir);
    if(outlen > 0 && outlen < MAX_STRING_LEN - 1) {  /* BUG FIX: Added bounds check */
        if(run_params->OutputDir[outlen - 1] != '/')
            strncat(run_params->OutputDir, "/", MAX_STRING_LEN - outlen - 1);  /* BUG FIX: Use strncat */
    }

    for(int i = 0; i < NParam; i++) {
        if(used_tag[i] && ParamRequired[i]) {
            fprintf(stderr, "Error. Missing required parameter '%s' in parameter file '%s'.\n",
                    OrigParamTag[i], fname);
            errorFlag = 1;
        }
    }

    if(errorFlag) {
        ABORT(1);
    }

#ifdef VERBOSE
    if(ThisTask == 0) {
        for(int i = 0; i < NParam; i++) {
            char valstr[MAX_STRING_LEN];
            switch(ParamID[i]) {
                case DOUBLE: snprintf(valstr, sizeof(valstr), "%g",  *((double *)ParamAddr[i])); break;
                case INT:    snprintf(valstr, sizeof(valstr), "%d",  *((int    *)ParamAddr[i])); break;
                case STRING: snprintf(valstr, sizeof(valstr), "%s",   (char    *)ParamAddr[i]);  break;
                default:     snprintf(valstr, sizeof(valstr), "?");                               break;
            }
            fprintf(stdout, "%35s\t%10s\n", OrigParamTag[i], valstr);
        }
        fprintf(stdout, "\n");
    }
#endif

    if( ! (run_params->LastSnapshotNr+1 > 0 && run_params->LastSnapshotNr+1 < ABSOLUTEMAXSNAPS) ) {
        fprintf(stderr,"LastSnapshotNr = %d should be in [0, %d) \n", run_params->LastSnapshotNr, ABSOLUTEMAXSNAPS);
        ABORT(1);
    }
    run_params->SimMaxSnaps = run_params->LastSnapshotNr + 1;

    if(!(run_params->NumSnapOutputs == -1 || (run_params->NumSnapOutputs > 0 && run_params->NumSnapOutputs <= ABSOLUTEMAXSNAPS))) {
        fprintf(stderr,"NumOutputs must be -1 or between 1 and %i\n", ABSOLUTEMAXSNAPS);
        ABORT(1);
    }

    // read in the output snapshot list
    if(run_params->NumSnapOutputs == -1) {
        run_params->NumSnapOutputs = run_params->SimMaxSnaps;
        for (int i=run_params->NumSnapOutputs-1; i>=0; i--) {
            run_params->ListOutputSnaps[i] = i;
        }
#ifdef VERBOSE
        if(ThisTask == 0) {
            fprintf(stdout, "all %d snapshots selected for output\n", run_params->NumSnapOutputs);
        }
#endif
    } else {
#ifdef VERBOSE
        if(ThisTask == 0) {
            fprintf(stdout, "%d snapshots selected for output: ", run_params->NumSnapOutputs);
        }
#endif

        // reopen the parameter file
        fd = fopen(fname, "r");

        int done = 0;
        while(!feof(fd) && !done) {
            char buf[MAX_STRING_LEN];

            /* scan down to find the line with the snapshots */
            if(fscanf(fd, "%s", buf) == 0) continue;
            if(strcmp(buf, "->") == 0) {
                // read the snapshots into ListOutputSnaps
                for(int i=0; i<run_params->NumSnapOutputs; i++) {
                    if(fscanf(fd, "%d", &(run_params->ListOutputSnaps[i])) == 1) {
#ifdef VERBOSE
                        if(ThisTask == 0) {
                            fprintf(stdout, "%d ", run_params->ListOutputSnaps[i]);
                        }
#endif
                    }
                }
                done = 1;
                break;
            }
        }

        fclose(fd);
        if(! done ) {
            fprintf(stderr,"Error: Could not properly parse output snapshots\n");
            ABORT(2);
        }
#ifdef VERBOSE
        fprintf(stdout, "\n");
#endif
    }


    if(run_params->FirstFile < 0 || run_params->LastFile < 0 || run_params->LastFile < run_params->FirstFile) {
        fprintf(stderr,"Error: FirstFile = %d and LastFile = %d must both be >=0 *AND* LastFile "
                        "should be larger than   FirstFile.\nProbably a typo in the parameter-file. "
                        "Please change to appropriate values...exiting\n",
                        run_params->FirstFile, run_params->LastFile);
        ABORT(EXIT_FAILURE);
    }

    /* sort the output snapshot numbers in descending order (in case the user didn't do that already) MS: 24th Oct, 2023 */
    qsort(run_params->ListOutputSnaps, run_params->NumSnapOutputs, sizeof(run_params->ListOutputSnaps[0]), compare_ints_descending);

    /* Check for duplicate snapshot outputs */
    int num_dup_snaps = 0;
    for(int ii=1;ii<run_params->NumSnapOutputs;ii++) {
        const int dsnap = run_params->ListOutputSnaps[ii-1] - run_params->ListOutputSnaps[ii];
        if(dsnap == 0) {
            fprintf(stderr,"Error: Found duplicate snapshots in the list of desired output snapshots\n");
            fprintf(stderr,"Duplicate value = %d in position = %d (out of %d total output snapshots requested)\n",
                            run_params->ListOutputSnaps[ii], ii, run_params->NumSnapOutputs);
            num_dup_snaps++;
        }
    }
    if(num_dup_snaps != 0) {
        fprintf(stderr,"Error: Found %d duplicate snapshots - please remove them from the parameter file and then re-run sage\n\n", num_dup_snaps);
        ABORT(EXIT_FAILURE);
    }

    /* because in the default case of 'lhalo-binary', nothing
       gets written to "treeextension", we need to
       null terminate tree-extension first  */
    run_params->TreeExtension[0] = '\0';

    // Check tree type is valid.
    if (strncmp(my_treetype, "lhalo_hdf5", MAX_STRING_LEN - 1)   == 0 ||
        strncmp(my_treetype, "genesis_hdf5", MAX_STRING_LEN - 1) == 0 ||
        strncmp(my_treetype, "gadget4_hdf5", MAX_STRING_LEN - 1) == 0
        ) {
#ifndef HDF5
        fprintf(stderr, "You have specified to use a HDF5 file but have not compiled with the HDF5 option enabled.\n");
        fprintf(stderr, "Please check your file type and compiler options.\n");
        ABORT(EXIT_FAILURE);
#endif
        // strncmp returns 0 if the two strings are equal.
        // only relevant options are HDF5 or binary files. Consistent-trees is *always* ascii (with different filename extensions)
        snprintf(run_params->TreeExtension, MAX_STRING_LEN - 1, ".hdf5");
    }

#define CHECK_VALID_ENUM_IN_PARAM_FILE(paramname, num_enum_types, enum_names, enum_values, string_value) { \
        int found = 0;                                                  \
        for(int i=0;i<num_enum_types;i++) {                             \
            if (strcasecmp(string_value, enum_names[i]) == 0) {         \
                run_params->paramname = enum_values[i];                 \
                found = 1;                                              \
                break;                                                  \
            }                                                           \
        }                                                               \
        if(found == 0) {                                                \
            fprintf(stderr, #paramname " field contains unsupported value of '%s' is not supported\n", string_value); \
            fprintf(stderr," Please choose one of the values -- \n");   \
            for(int i=0;i<num_enum_types;i++) {                         \
                fprintf(stderr, #paramname " = '%s'\n", enum_names[i]); \
            }                                                           \
            ABORT(EXIT_FAILURE);                                        \
        }                                                               \
 }

    const char tree_names[][MAXTAGLEN] = {"lhalo_hdf5", "lhalo_binary", "genesis_hdf5",
                                          "consistent_trees_ascii", "consistent_trees_hdf5",
                                          "gadget4_hdf5"};
    const enum Valid_TreeTypes tree_enums[] = {lhalo_hdf5, lhalo_binary, genesis_hdf5,
                                               consistent_trees_ascii, consistent_trees_hdf5,
                                               gadget4_hdf5};
    /* enum, not const int: BUILD_BUG_OR_ZERO declares an array of this size, and in C99
       only an integer constant expression keeps that from being a variable-length array. */
    enum { nvalid_tree_types = sizeof(tree_names)/(MAXTAGLEN*sizeof(char)) };
    BUILD_BUG_OR_ZERO((nvalid_tree_types == (int) num_tree_types), number_of_tree_types_is_incorrect);
    CHECK_VALID_ENUM_IN_PARAM_FILE(TreeType, nvalid_tree_types, tree_names, tree_enums, my_treetype);

    /* Check output data type is valid. */
#ifndef HDF5
    if(strncmp(my_outputformat, "sage_hdf5", MAX_STRING_LEN-1) == 0) {
        fprintf(stderr, "You have specified to use HDF5 output format but have not compiled with the HDF5 option enabled.\n");
        fprintf(stderr, "Please check your file type and compiler options.\n");
        ABORT(EXIT_FAILURE);
    }
#endif

    const char format_names[][MAXTAGLEN] = {"sage_binary", "sage_hdf5", "lhalo_binary_output"};
    const enum Valid_OutputFormats format_enums[] = {sage_binary, sage_hdf5, lhalo_binary_output};
    const int nvalid_format_types  = sizeof(format_names)/(MAXTAGLEN*sizeof(char));
    XRETURN(nvalid_format_types == 3, EXIT_FAILURE, "nvalid_format_types = %d should have been 3\n", nvalid_format_types);
    CHECK_VALID_ENUM_IN_PARAM_FILE(OutputFormat, nvalid_format_types, format_names, format_enums, my_outputformat);

    /* Check that the way forests are distributed over (MPI) tasks is valid */
    const char scheme_names[][MAXTAGLEN] = {"uniform_in_forests", "linear_in_nhalos", "quadratic_in_nhalos", "exponent_in_nhalos", "generic_power_in_nhalos"};
    const enum Valid_Forest_Distribution_Schemes scheme_enums[] = {uniform_in_forests, linear_in_nhalos,
                                                                   quadratic_in_nhalos, exponent_in_nhalos, generic_power_in_nhalos};
    const int nvalid_scheme_types  = sizeof(scheme_names)/(MAXTAGLEN*sizeof(char));
    XRETURN(nvalid_scheme_types == num_forest_weight_types, EXIT_FAILURE, "nvalid_format_types = %d should have been %d\n",
            nvalid_format_types, num_forest_weight_types);

    CHECK_VALID_ENUM_IN_PARAM_FILE(ForestDistributionScheme, nvalid_scheme_types, scheme_names, scheme_enums, my_forest_dist_scheme);
#undef CHECK_VALID_ENUM_IN_PARAM_FILE


    /* SF prescription must be one of the eight implemented recipes; the
       H2-tracking predicate (sf_prescription_tracks_h2) relies on this range. */
    if(run_params->SFprescription < 0 || run_params->SFprescription > 7) {
        fprintf(stderr,"Error: SFprescription = %d is not valid; it must be in [0, 7].\n", run_params->SFprescription);
        fprintf(stderr,"Please change the value for the parameter 'SFprescription' in the parameter file (%s)\n", fname);
        ABORT(EXIT_FAILURE);
    }

    /* Physics option flags: reject out-of-range values at startup rather than
       running silently with untested behaviour.  Valid ranges follow the
       dispatch chains in the physics modules (see docs/parameters.md). */
    {
        const struct { const char *name; int32_t value; int32_t min; int32_t max; } option_ranges[] = {
            {"AGNrecipeOn",            run_params->AGNrecipeOn,            0, 3},
            {"SupernovaRecipeOn",      run_params->SupernovaRecipeOn,      0, 1},
            {"ReionizationOn",         run_params->ReionizationOn,         0, 1},
            {"DiskInstabilityOn",      run_params->DiskInstabilityOn,      0, 1},
            {"CGMrecipeOn",            run_params->CGMrecipeOn,            0, 1},
            {"CGMDensityProfile",      run_params->CGMDensityProfile,      0, 3},
            // {"PrecipCriterionOn",      run_params->PrecipCriterionOn,      0, 5},
            {"FIREmodeOn",             run_params->FIREmodeOn,             0, 1},
            {"RegimeRandomMode",       run_params->RegimeRandomMode,       0, 1},
            {"ConcentrationOn",        run_params->ConcentrationOn,        0, 3},
            {"FeedbackFreeModeOn",     run_params->FeedbackFreeModeOn,     0, 7},
            {"FFBIgnoreRegime",        run_params->FFBIgnoreRegime,        0, 1},
            {"FFBRandomMode",          run_params->FFBRandomMode,          0, 1},
            {"ColdStreamCeilingOn",    run_params->ColdStreamCeilingOn,    0, 1},
            {"BulgeSizeOn",            run_params->BulgeSizeOn,            0, 3},
            // {"DiskRadiusOn",           run_params->DiskRadiusOn,           0, 2},
            // {"PreventiveHeatingOn",    run_params->PreventiveHeatingOn,    0, 6},
            {"H2DiskAreaOption",       run_params->H2DiskAreaOption,       0, 2},
            {"H2RadialIntegrationOn",  run_params->H2RadialIntegrationOn,  0, 1},
            {"SaveFullSFH",            run_params->SaveFullSFH,            0, 1},
            {"TrackICSAssembly",       run_params->TrackICSAssembly,       0, 1},
            {"StarburstColdGasOn",     run_params->StarburstColdGasOn,     0, 1},
            {"DynamicDisruptionSplit", run_params->DynamicDisruptionSplit, 0, 2},
            {"RamPressureStrippingOn", run_params->RamPressureStrippingOn, 0, 1},
            {"SNEnergyConservationOn", run_params->SNEnergyConservationOn, 0, 1},
        };
        for(size_t i = 0; i < sizeof(option_ranges) / sizeof(option_ranges[0]); i++) {
            if(option_ranges[i].value < option_ranges[i].min || option_ranges[i].value > option_ranges[i].max) {
                fprintf(stderr, "Error: %s = %d is not valid; it must be in [%d, %d].\n",
                        option_ranges[i].name, option_ranges[i].value, option_ranges[i].min, option_ranges[i].max);
                fprintf(stderr, "Please change the value for the parameter '%s' in the parameter file (%s)\n",
                        option_ranges[i].name, fname);
                ABORT(EXIT_FAILURE);
            }
        }
    }

    /* Numeric parameters that must be strictly positive for the physics to
       be well-defined. */
    if(run_params->H2RadialIntegrationOn && run_params->H2RadialNBins < 1) {
        fprintf(stderr, "Error: H2RadialNBins = %d is not valid; the radial integration needs at least 1 bin.\n",
                run_params->H2RadialNBins);
        ABORT(EXIT_FAILURE);
    }
    if(run_params->H2RadialIntegrationOn && run_params->H2RadialRMaxFactor <= 0.0) {
        fprintf(stderr, "Error: H2RadialRMaxFactor = %g is not valid; it must be > 0.\n",
                run_params->H2RadialRMaxFactor);
        ABORT(EXIT_FAILURE);
    }
    // if(run_params->PreventiveHeatingOn > 0 && run_params->PreventiveHeatingMass <= 0.0) {
    //     fprintf(stderr, "Error: PreventiveHeatingMass = %g is not valid; it must be > 0 when PreventiveHeatingOn > 0.\n",
    //             run_params->PreventiveHeatingMass);
    //     ABORT(EXIT_FAILURE);
    // }
    // if(run_params->PreventiveHeatingOn > 0 && run_params->PreventiveHeatingSlope <= 0.0) {
    //     fprintf(stderr, "Error: PreventiveHeatingSlope = %g is not valid; it must be > 0 when PreventiveHeatingOn > 0.\n",
    //             run_params->PreventiveHeatingSlope);
    //     ABORT(EXIT_FAILURE);
    // }
    // if(run_params->PreventiveHeatingOn == 6 && run_params->PreventiveHeatingEfficiency <= 0.0) {
    //     fprintf(stderr, "Error: PreventiveHeatingEfficiency = %g is not valid; it must be > 0 when PreventiveHeatingOn = 6.\n",
    //             run_params->PreventiveHeatingEfficiency);
    //     ABORT(EXIT_FAILURE);
    // }
    // if(run_params->DiskRadiusOn > 0 ) {
    //     fprintf(stderr, "Error: DiskRadiusOn = %g is not valid; it must be > 0.\n",
    //             run_params->DiskRadiusOn);
    //     ABORT(EXIT_FAILURE);
    // }
    if(run_params->GasDiskRadiusFactor <= 0.0) {
        fprintf(stderr, "Error: GasDiskRadiusFactor = %g is not valid; it must be > 0.\n",
                run_params->GasDiskRadiusFactor);
        ABORT(EXIT_FAILURE);
    }
    if(run_params->RamPressureStrippingOn && run_params->RamPressureEpsilon <= 0.0) {
        fprintf(stderr, "Error: RamPressureEpsilon = %g is not valid; it must be > 0 when RamPressureStrippingOn = 1.\n",
                run_params->RamPressureEpsilon);
        ABORT(EXIT_FAILURE);
    }
    if(run_params->SNEnergyConservationOn && run_params->MaxSNEnergyCoupling <= 0.0) {
        fprintf(stderr, "Error: MaxSNEnergyCoupling = %g is not valid; it must be > 0 when SNEnergyConservationOn = 1.\n",
                run_params->MaxSNEnergyCoupling);
        ABORT(EXIT_FAILURE);
    }
    if(run_params->SubstepResolution <= 0.0) {
        fprintf(stderr, "Error: SubstepResolution = %g is not valid; it must be > 0.\n",
                run_params->SubstepResolution);
        ABORT(EXIT_FAILURE);
    }

    /* Option combinations that would run but produce physically meaningless
       output are rejected here instead of failing silently mid-run. */
    if((run_params->FeedbackFreeModeOn == 6 || run_params->FeedbackFreeModeOn == 7)
       && !sf_prescription_tracks_h2(run_params->SFprescription)) {
        fprintf(stderr, "Error: FeedbackFreeModeOn = %d selects H2-based FFB star formation, but\n"
                        "SFprescription = %d does not track H2 (only prescriptions other than 0 and 2 do).\n"
                        "FFB bursts would form zero stars. Choose an H2-tracking SFprescription or an\n"
                        "FFB mode in [1, 5].\n",
                run_params->FeedbackFreeModeOn, run_params->SFprescription);
        ABORT(EXIT_FAILURE);
    }
    if((run_params->FeedbackFreeModeOn == 4 || run_params->FeedbackFreeModeOn == 7)
       && run_params->FFBConcSigma <= 0.0) {
        fprintf(stderr, "Error: FeedbackFreeModeOn = %d uses log-normal concentration scatter, but\n"
                        "FFBConcSigma = %g; the scatter width must be > 0 (typical ~0.2).\n",
                run_params->FeedbackFreeModeOn, run_params->FFBConcSigma);
        ABORT(EXIT_FAILURE);
    }

    /* Check that exponent supplied is non-negative (for cases where the exponent will be used) */
    if((run_params->ForestDistributionScheme == exponent_in_nhalos || run_params->ForestDistributionScheme == generic_power_in_nhalos)
       && run_params->Exponent_Forest_Dist_Scheme < 0) {
        fprintf(stderr,"Error: You have requested a power-law exponent but the exponent = %e must be greater than 0\n",
                run_params->Exponent_Forest_Dist_Scheme);
        fprintf(stderr,"Please change the value for the parameter 'ExponentForestDistributionScheme' in the parameter file (%s)\n", fname);
        ABORT(EXIT_FAILURE);
    }

    myfree(used_tag);
    return EXIT_SUCCESS;
}


#undef MAXTAGS
#undef MAXTAGLEN

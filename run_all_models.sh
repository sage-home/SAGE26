#!/bin/bash
#
# run_all_models.sh
# Runs SAGE for each model variant by modifying millennium.par,
# creating the output directory if needed, and running ./sage.
#
# Usage:
#   ./run_all_models.sh           # Run all models
#   ./run_all_models.sh --skip-existing  # Skip models that already have output
#

set -e

PAR_FILE="input/millennium.par"
BACKUP_FILE="input/millennium.par.bak"
SKIP_EXISTING=false

if [[ "${1}" == "--skip-existing" ]]; then
    SKIP_EXISTING=true
fi

# Backup the original parameter file
cp "$PAR_FILE" "$BACKUP_FILE"

# Restore on exit or interruption
trap 'echo "Restoring original parameter file..."; cp "$BACKUP_FILE" "$PAR_FILE"; rm -f "$BACKUP_FILE"' EXIT

# --- Helper functions ---

reset_par() {
    cp "$BACKUP_FILE" "$PAR_FILE"
}

set_param() {
    local param="$1"
    local value="$2"
    # Replace the numeric/integer value following the parameter name
    sed -i '' "s/^\(${param}[[:space:]]*\)[0-9.eE+-]*/\1${value}/" "$PAR_FILE"
}

set_output_dir() {
    local dir="$1"
    sed -i '' "s|^\(OutputDir[[:space:]]*\)[^%]*|\1${dir} |" "$PAR_FILE"
}

run_model() {
    local model_name="$1"
    local output_dir="output/${model_name}/"

    # Optionally skip if output already exists
    if $SKIP_EXISTING && [[ -f "${output_dir}model_0.hdf5" ]]; then
        echo "SKIPPING ${model_name} (output already exists)"
        echo ""
        return
    fi

    echo "============================================"
    echo "Running model: ${model_name}"
    echo "============================================"

    # Create output directory if needed
    mkdir -p "$output_dir"

    # Set the output directory in the par file
    set_output_dir "$output_dir"

    # Run SAGE
    ./sage "$PAR_FILE"

    echo "Completed: ${model_name}"
    echo ""
}

# ==========================================================
# FFB efficiency sweep (millennium_ffb10 .. millennium_ffb100)
# ==========================================================
for ffb in 10 20 30 40 50 60 70 80 90 100; do
    reset_par
    eff=$(awk "BEGIN{printf \"%.1f\", ${ffb}/100}")
    set_param "FFBMaxEfficiency" "$eff"
    run_model "millennium_ffb${ffb}"
done

# ==========================================================
# SF prescription variants
# ==========================================================

# KMT09
reset_par
set_param "SFprescription" "5"
run_model "millennium_kmt09"

# KD12
reset_par
set_param "SFprescription" "4"
run_model "millennium_kd12"

# K13
reset_par
set_param "SFprescription" "6"
run_model "millennium_k13"

# GD14
reset_par
set_param "SFprescription" "7"
run_model "millennium_gd14"

# No H2 (original Croton 2006 SF law)
reset_par
set_param "SFprescription" "0"
run_model "millennium_noh2"

# ==========================================================
# Feedback variants
# ==========================================================

# No feedback-free mode
reset_par
set_param "FeedbackFreeModeOn" "0"
run_model "millennium_noffb"

# C16 feedback: FIRE off, old SN feedback parameters
reset_par
set_param "FIREmodeOn" "0"
set_param "FeedbackReheatingEpsilon" "3.0"
set_param "FeedbackEjectionEfficiency" "0.3"
run_model "millennium_c16feedback"

# No CGM
reset_par
set_param "CGMrecipeOn" "0"
run_model "millennium_nocgm"

# ==========================================================
# CGM density profile variants
# ==========================================================

# Uniform
reset_par
set_param "CGMDensityProfile" "0"
run_model "millennium_uniform"

# NFW
reset_par
set_param "CGMDensityProfile" "1"
run_model "millennium_NFW"

# Beta-profile
reset_par
set_param "CGMDensityProfile" "2"
run_model "millennium_beta"

# ==========================================================
# Vanilla SAGE (original Croton et al. 2016)
# ==========================================================
reset_par
set_param "SFprescription" "0"
set_param "SfrEfficiency" "0.05"
set_param "CGMrecipeOn" "0"
set_param "FIREmodeOn" "0"
set_param "FeedbackFreeModeOn" "0"
set_param "FeedbackReheatingEpsilon" "3.0"
set_param "FeedbackEjectionEfficiency" "0.3"
run_model "millennium_vanilla"

# ==========================================================
# Fiducial model (millennium)
# ==========================================================
reset_par
mkdir -p output/millennium/
echo "============================================"
echo "Running model: millennium (fiducial)"
echo "============================================"
./sage "$PAR_FILE"
echo "Completed: millennium (fiducial)"
echo ""

echo "============================================"
echo "All models completed successfully."
echo "============================================"

# Run plotting
echo "Running paper_plots.py..."
python3 plotting/paper_plots.py
echo "Plotting complete."

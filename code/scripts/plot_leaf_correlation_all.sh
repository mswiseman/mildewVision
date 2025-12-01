#!/usr/bin/env bash

#
# -----------------------------------------------------------------------------
# Script: plot_leaf_correlation_all.sh
#
# Runs multiple leaf analysis pipelines in parallel using the active Python
# environment (expected: conda env "mildewVision"). Examples as written launch:
#   1) Disease severity and saliency map generation (plot_sal_map_leaf.py)
#   2) Disease severity and optional saliency analysis (leaf_correlation.py)
#
# For information on argparse arguments see argparse section in either 
# plot_sal_map_leaf.py or leaf_correlation.py
#
# Usage:
#   bash plot_leaf_correlation_all.sh
# -----------------------------------------------------------------------------

# Move to the directory this script lives in (no hard-coded Windows paths)
# On 309 computer that should be ~/Desktop/blackbird_scripts/
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || {
    echo "Failed to cd into script directory" >&2
    exit 1
}

# Find python from the current environment (should be mildewVision)
PYTHON="$(command -v python || true)"
if [[ -z "$PYTHON" ]]; then
    echo "python not found on PATH. Activate your conda env first:" >&2
    echo "    conda activate mildewVision" >&2
    echo "then run:" >&2
    echo "    bash $(basename "$0")" >&2
    exit 1
fi

# Limit number of concurrent jobs
MAX_JOBS=2

# Commands to run (each in its own timed Python process)
commands=(
    "time \"$PYTHON\" ../plot_sal_map_leaf.py \
        --model_type ResNet \
        --model_path ../.. \
        --dataset_path ../../data \
        --loading_epoch 59 \
        --threshold 0.7 \
        --up_threshold 0.8 \
        --down_threshold 0.3 \
        --cuda \
        --cuda_id 0 \
        --outdim 2 \
        --means 0.5410 0.6371 0.4188 \
        --stds 0.1764 0.1650 0.2326 \
        --timestamp Feb14_15-53-04_2024 \
        --dpi 2 \
        --pretrained \
        --sal_gradient \
        --sal_deeplift \
        --img_folder 6-28-2023_10dpi \
        --trays 1 \
        --pm HPM-666"

    "time \"$PYTHON\" ../leaf_correlation.py \
        --model_type ResNet \
        --model_path ../.. \
        --dataset_path ../../data \
        --loading_epoch 59 \
        --threshold 0.7 \
        --up_threshold 0.8 \
        --down_threshold 0.3 \
        --cuda \
        --cuda_id 0 \
        --outdim 2 \
        --means 0.5410 0.6371 0.4188 \
        --stds 0.1764 0.1650 0.2326 \
        --timestamp Feb14_15-53-04_2024 \
        --dpi 10 \
        --pretrained \
        --img_folder 6-28-2023_10dpi \
        --trays 1 \
        --sal_gradient \
	    --sal_deeplift \
        --pm HPM-666"
)

wait_for_free_job_slot() {
    while : ; do
        jobs_running=$(jobs -r | wc -l)
        if [[ $jobs_running -lt $MAX_JOBS ]]; then
            break
        fi
        sleep 1
    done
}

# Launch commands, respecting MAX_JOBS
for cmd in "${commands[@]}"; do
    eval "$cmd" &
    wait_for_free_job_slot
done

# Wait for all background jobs to complete
wait

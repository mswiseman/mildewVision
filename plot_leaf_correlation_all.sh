#!/usr/bin/env bash

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
        --dataset_path /e/Stacked/2025_AtTKO_HlMLO_Pheno \
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
        --img_folder 10-3-2025_2dpi \
        --trays 1 \
        --pm Gc_USC1"

    "time \"$PYTHON\" ../leaf_correlation_mw.py \
        --model_type ResNet \
        --model_path ../.. \
        --dataset_path /d/Stacked/Quintec_PM_Resistance_Screens_stained \
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
        --img_folder 11-24-2025_10dpi \
        --trays 1 \
        --sal_gradient \
        --pm various"
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

#!/usr/bin/env bash

###############################################################################
# plot_sal_map_leaf
#
# This script runs saliency map generation for one or more trays using
# plot_sal_map_leaf.py from the mildewVision pipeline. 
#
# To run: bash plot_sal_map_leaf.sh
#
# ----------------------- Available Parameters ------------------------------
#
# Core model parameters:
#   --model_type       Neural network architecture (e.g. ResNet, VGG, Inception, etc)
#   --model_path       Base directory path; auto adds ./results/[model string]
#   --loading_epoch    Epoch number of the model weights to load (part of model string)
#   --pretrained       Pretrained weights as base
#   --timestamp        Model time stamp (part of model string)
#
# Dataset parameters:
#   --dataset_path     Root directory containing stacked leaf images
#   --img_folder       Subfolder inside dataset representing image group
#   --trays            Tray number(s) to process (looped by this script)
#   --pm               Powdery mildew metadata
#   --dpi              Days post inoculation metadata
#
# Inference & output settings:
#   --threshold        Classification threshold
#   --up_threshold     If above this threshold, call a patch "infected"
#   --down_threshold   If below this, call a "healthy" patch
#   --outdim           Number of classes (either 2 or 3)
#
# Hardware options:
#   --cuda             Enable GPU acceleration
#   --cuda_id          GPU device index
#
# Normalization (must match training):
#   --means            Channel-wise model mean RGB values (from training set)
#   --stds             Channel-wise model std dev RGB values (from training set)
#
# Saliency options (Captum):
#   --sal_gradient     Enable gradient saliency maps
#   --sal_deeplift     Enable DeepLIFT saliency maps
#   --sal_gradcam      Enable Grad-CAM saliency maps
#   --sal_smoothgrad   Enable SmoothGrad saliency maps
#   --sal_thresh_method   Choose between percentile (default 95%) or fixed
#
# ---------------------------------------------------------------------------
# For all available commands, see argparse section in plot_sal_map_leaf.py
###############################################################################

# Tray numbers to process (edit as needed)
trays=("1" "2" "3" "4")

for tray in "${trays[@]}"; do
    echo "Processing tray: $tray"

    time python ../plot_sal_map_leaf.py                \
        --model_type     ResNet                        \
        --model_path     ../..                         \
        --dataset_path   /e/Stacked/2025_Pheno         \
        --loading_epoch  59                            \
        --threshold      0.7                           \
        --up_threshold   0.8                           \
        --down_threshold 0.3                           \
        --cuda                                         \
        --cuda_id       0                              \
        --outdim        2                              \
        --means         0.5410 0.6371 0.4188           \
        --stds          0.1764 0.1650 0.2326           \
        --timestamp     Feb14_15-53-04_2024            \
        --dpi           2                              \
        --pretrained                                   \
        --sal_gradient                                 \
        --sal_deeplift                                 \
        --img_folder    10-3-2025_2dpi                 \
        --trays         "$tray"                        \
        --pm            Gc_USC1

done


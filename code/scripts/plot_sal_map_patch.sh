#!/bin/bash

# Script for plotting saliency maps of various deep learning classification outputs
# Written by Michele Wiseman using Tian Qiu's scripts as templates

# Version 2 - July 8th 2023

#declare -a trays=("1") # change to tray numbers if multiple, don't include commas, include spaces (e.g. "2" "3" "4")

#for((i=0;i<4;i++))
#do
    #tray=${trays[i]}
    time python3 ../plot_sal_map_leaf.py \
                --model_type VGG                                    \
                --model_path /Users/michelewiseman/Desktop/blackbird_ml        \
                --dataset_path /Users/michelewiseman/Desktop/blackbird_ml/data                  \
                --pretrained                                               \
                --loading_epoch 95                                         \
                --mps                                                      \
                --img_folder 6-28-2023_6dpi                                \
                --up_threshold 0.7                                         \
                --down_threshold 0.3                                       \
                --outdim 2                                                 \
                --dpi 6                                                    \
                --threshold 0.2                                            \
                --trays 1                                                  \
                --means 0.49 0.58 0.33                                     \
                --stds 0.15 0.15 0.19                                      \
                --timestamp May10_16-42-59_2021
                #--save_healthy                                             \
                #--save_infected

    time python3 ../plot_sal_map_leaf.py \
                --model_type ResNet                                    \
                --model_path /Users/michelewiseman/Desktop/blackbird_ml        \
                --dataset_path /Users/michelewiseman/Desktop/blackbird_ml/data                  \
                --pretrained                                               \
                --loading_epoch 16                                         \
                --mps                                                      \
                --img_folder 6-28-2023_6dpi                                \
                --up_threshold 0.7                                         \
                --outdim 3                                                 \
                --down_threshold 0.3                                       \
                --dpi 6                                                    \
                --threshold 0.2                                            \
                --trays 1                                                  \
                --means 0.49 0.58 0.33                                    \
                --stds 0.15 0.15 0.19                                     \
                --timestamp Jul24_16-44-43_2023
                #--save_healthy                                             \
                #--save_infected
#done

# Cornell models:
# May10_16-42-59_2021 (VGG ep 95)
# May11_05-52-52_2021 (Inception3 ep 95)
# May11_00-29-56_2021 (ResNet ep 95)
# Aug14_11-22-25_2021 (DeepLab ep 94)

# Our well performing models:s
# VGG_May10_16-42-59_2021_2023 (VGG ep 99) - transfer learning from Cornell model
# Inception3_Jul02_06-12-25_2023 (ep 24) - 5 dpi mapping population
# VGG_Jul09_22-29-36_2023 (ep 49) - 10 dpi mapping population 3 class
# Inception3_Jul05_13-06-56_2023 (ep 35) - 10 dpi mapping population 3 class
# ResNet_Jul24_16-44-43_2023 (ep 16)- 10 dpi mapping population 3 class

# Downy:
# VGG_Jul11_04-22-01_2023 ep042
# Inception3_Jul11_13-43-14_2023 ep030
# ResNet_Jul11_03-59-20_2023 ep020

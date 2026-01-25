#!/bin/bash



python ../classification/inference.py                                           \
            --model_path     ../..                                              \
            --dataset_path   ../..                                              \
            --model_type ResNet                                                 \
            --HDF5                                                              \
            --amp                                                               \
            --pretrained                                                        \
            --set val                                                           \
            --cuda                                                              \
            --cuda_id 0                                                         \
            --loading_epoch 65                                                  \
            --means 0.5663 0.6596 0.4508                                        \
            --dual_head                                                         \
            --stds 0.1811 0.1667 0.2434                                         \
            --save_misclassified                                                \
            --up_threshold 0.5                                                  \
            --down_threshold 0.25                                               \
            --n_misclassified 20                                                \
            --timestamp Jan22_10-21-00_2026                                     \
            --grid_search                                                       \
            --ignore_discard

            #--inf_gate 0.2                                                     \
            # --spor_th 0.5                                                     \



# For ResNet Jan22 ep 65 model:
#  --means 0.5663 0.6596 0.4508 \
#  --stds 0.1811 0.1667 0.2434

# For ResNet Feb14 ep 59 model:
#  --means 0.5410 0.6371 0.4188 \
#  --stds 0.1764 0.1650 0.2326 \

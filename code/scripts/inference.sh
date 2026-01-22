#!/bin/bash

python ../classification/inference.py                                           \
            --model_path     ../..                                              \
            --dataset_path   ../..                                              \
            --model_type ResNet                                                 \
            --HDF5                                                              \
            --amp                                                               \
            --pretrained                                                        \
            --set test                                                          \
            --dual_head                                                         \
            --inf_gate 0.2                                                      \
            --up_threshold 0.7                                                  \
            --down_threshold 0.05                                               \
            --spor_th 0.85                                                      \
            --cuda                                                              \
            --cuda_id 0                                                         \
            --loading_epoch 43                                                  \
            --timestamp Jan21_23-29-29_2026                                     \
            --means 0.5410 0.6371 0.4188                                        \
            --stds 0.1764 0.1650 0.2326                                         \
            --save_misclassified                                                \
            --n_misclassified 10
            #            --grid_search                                                       \


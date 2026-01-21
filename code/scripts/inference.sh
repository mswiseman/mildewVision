#!/bin/bash


python3 ../classification/inference.py                                          \
            --model_path     ../..                                              \
            --dataset_path   ../../data                                         \
            --model_type ResNet                                                 \
            --HDF5                                                              \
            --pretrained                                                        \
            --set test                                                          \
            --cuda                                                              \
            --cuda_id 0                                                         \
            --loading_epoch 59                                                  \
            --timestamp Feb14_15-53-04_2024                                     \
            --save_misclassified                                                \
            --n_misclassified 10

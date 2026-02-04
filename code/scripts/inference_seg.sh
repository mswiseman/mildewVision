#!/bin/bash

# for running inference using segmentation
# ../segmentation/infer_from_folder.py is mainly for testing performance on patches
# ../segmentation/infer_from_folder.py is for inference on entire leaf disks


python ../segmentation/infer_from_folder.py                                                           \
    --model_type        DeepLab                                                                       \
    --loading_epoch     60                                                                            \
    --pretrained                                                                                      \
    --timestamp         Feb04_00-32-07_2021                                                           \
    --model_path        "../.."                                                                       \
    --cuda                                                                                            \
    --cuda_id           0                                                                             \
    --patch_folder      "../../data/segmentation/test_set/images"                                     \
    --out_mask_folder   "../../data/segmentation/test_set/masks"


python ../segmentation/infer_from_folder.py                                                           \
    --model_type        DeepLab                                                                       \
    --loading_epoch     60                                                                            \
    --pretrained                                                                                      \
    --timestamp         Feb04_00-32-07_2021                                                           \
    --model_path        "../.."                                                                       \
    --cuda                                                                                            \
    --cuda_id           0                                                                             \
    --outdim            2                                                                             \
    --in_folder         "../../data/6-28-2023_10dpi/1"                                                \
    --out_folder        "../../results/segmentation/6-28-2023_10dpi-1"                                \
    --step              112                                                                           \
    --batch_size        8

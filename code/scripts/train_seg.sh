#!/bin/bash

# note: right now run.py is hardcoded to load patches and masks from a prespecified image_dir and mask_dir
# need to add arg parse so editing code isn't necessary

python ../segmentation/run.py           \
        --root_path  /c/Users/Intel\ User/Desktop/blackbird_scripts \
        --model_type DeepLab            \
        --pretrained                    \
        --save_model                    \
        --weighted_loss                 \
        --loading_epoch 0               \
        --total_epochs 95               \
        --cuda                          \
        --optimType Adam                \
        --lr 1e-4                       \
        --weight_decay 2e-4             \
        --bsize 32                      \
        --nworker 1                     \
        --cuda_device 0

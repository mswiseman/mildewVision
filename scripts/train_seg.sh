#!/bin/bash


for((i=1;i<4;i++))
do
    python3 ../segmentation/run.py      \
                --root_path  /Users/michelewiseman/Desktop/blackbird_ml/ \
                --model_type DeepLab        \
                --pretrained                \
                --save_model                \
                --weighted_loss             \
                --loading_epoch 0           \
                --total_epochs 100          \
                --cuda                      \
                --optimType Adam            \
                --lr 1e-4                   \
                --weight_decay 2e-4         \
                --bsize 32                  \
                --nworker 2                 \
                --cv                        \
                --seg_idx $i                \
                --cuda_device 0

done


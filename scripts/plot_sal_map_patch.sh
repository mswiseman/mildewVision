#!/bin/bash

# Script for plotting saliency maps of various deep learning classification outputs
# Written by Michele Wiseman using Tian Qiu's scripts as templates

# Version 1 - February 8th 2023

#declare -a trays=("1") # change to tray numbers if multiple, don't include commas, include spaces (e.g. "2" "3" "4")

#for((i=0;i<4;i++))
#do
    #tray=${trays[i]}
    time python3 ../plot_sal_map_patch.py \
                --model_type ResNet                                    \
                --model_path /Users/michelewiseman/Desktop/blackbird_ml        \
                --dataset_path /Users/michelewiseman/Desktop/blackbird_ml/data                  \
                --pretrained                                                \
                --loading_epoch 19                                         \
                --mask_threshold 0.2                                        \
                --img_folder 6-14-2023_6dpi                                 \
                --cuda_id 0                                                 \
                --platform BlackBird                                        \
                --dpi 6                                                     \
                --trays 1                                                   \
                --mps                                                       \
                --means 0.49 0.5975 0.34                                    \
                --stds 0.14 0.14 0.18                                      \
                --outdim 2                                                  \
                --hdf5                                                      \
                --timestamp Jul26_20-03-13_2023



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

# Downy:
# VGG_Jul11_04-22-01_2023 ep042
# Inception3_Jul11_13-43-14_2023 ep030
# ResNet_Jul11_03-59-20_2023 ep020


# Text below provides parameter explanations and usage directions

# To use: 

# 1. Change parameters if desired, including tray number (at top)
# 2. Save
# 3. Ensure all zeros in front of image file names have been removed by running `bash ../common/remove_zero.sh`
# 4. Open terminal
# 5. bash plot_sal_map_leah.sh

# General Notes for above parameters:

#--model_type
#    - The type of deep learning model to train. Choices include:
#        - GoogleNet: GoogleNet is a 22-layer deep Convolutional Neural Network (CNN) introduced in 2014. It uses the inception module, which 
#            concatenates multiple filters with different sizes to form a more diverse representation of the image. The final layer of GoogleNet
#            is a softmax classifier, used for image classification. https://arxiv.org/abs/1409.4842
#        - ResNet: ResNet stands for Residual Network, introduced in 2015. ResNets are deep neural networks with hundreds or even thousands of 
#            layers, which were difficult to train before. The main idea behind ResNet is to use residual connections, where the input from one layer 
#            is added to the output of another layer. This allows the network to better preserve information and make the training of deep networks 
#            easier. https://arxiv.org/abs/1512.03385
#        - SqueezeNet: SqueezeNet is a light-weight CNN introduced in 2016, designed to have fewer parameters while still achieving high accuracy 
#            on image classification tasks. It uses the Fire module, which concatenates 1x1 and 3x3 filters, to reduce the number of parameters.
#            https://arxiv.org/abs/1602.07360
#        - DenseNet: DenseNet is a type of CNN introduced in 2016, which uses dense connections instead of residual connections. In a dense network,
#            each layer is connected to all previous layers, leading to a dense flow of information through the network. This architecture helps 
#            reduce the number of parameters, improve feature reuse, and promote feature fusion. https://arxiv.org/abs/1608.06993
#        - VGG: VGG is a very deep CNN introduced in 2014, which uses a simple architecture with only 3x3 convolutions and max-pooling. It has a 
#            large number of parameters, but it still manages to achieve good results on image classification tasks. https://arxiv.org/abs/1409.1556
#        - AlexNet: AlexNet is an 8-layer deep CNN introduced in 2012, which was one of the first deep networks to win the ImageNet Large Scale 
#            Visual Recognition Challenge (ILSVRC). AlexNet uses the ReLU activation function, max-pooling, and dropout to prevent overfitting.
#            https://dl.acm.org/doi/10.1145/3065386
# --timestamp 
#    - is the model timestamp, e.g. Feb05_22-39-19_2023
#--model_path
#    - This is essentially the root path
#    - on PC, looks like this: /c/Users/michele.wiseman/Desktop/blackbird
#--dataset_path
#    - Where the image data is housed, e.g. /d/Stacked/Deposition_Study
#--pretrained
#    - If you are using a pretrained model (e.g. Inception3, DeepLab, etc., you should have the --pretrained flag)
#--epoch
#    - The specific epoch of the model you're loading (e.g. Inception3_model_ep095 would be 95)
#--threshold
#    - Pretty sure this is masking threshold. Lower = less masking but may have more background; higher = more masking, but may lose part of leaf.
#--cuda
#    - Include this tag if you have a CUDA enabled GPU
#    - Can check by running (in python after import torch): print(torch.cuda.is_available())
#    - More info here: https://developer.nvidia.com/cuda-toolkit
#--cuda_id
#    - The ID of your CUDA-endabled GPU (if using one) in case you have more than one
#    - Default is 0
#    - To determine, run: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
#--timestamp
#    - Timestamp on the specific model being tested (e.g. Feb06_09-26-58_2023) 
#--group
#    - Optional, but provides architecture for better subdirectory organization
#    - E.g. if a specific mapping population, specify that here (--group comet_64305M)
#--img_folder 
#    - Name of your image folder (e.g. 2-5-2023_6dpi) 
#--trays
#    - Tray subdirectories in your image folder, e.g. 1.
#    - Can run multiple at a time by altering for loop at the top. 

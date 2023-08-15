#!/bin/bash

# Script for training deep learning classification models
# Originally written by Tian Qiu (https://bit.ly/3HKwcT4)
# Documentation and additional parameters written by Michele Wiseman

# Version 3, June 29th 2023

for((i=0;i<4;i++))
do


# Inception3 training
python3 ../classification/run.py      \
            --root_path  /Users/michelewiseman/Desktop/blackbird_ml \
            --model_type Inception3                      \
            --save_model                                 \
            --pretrained                                 \
            --weighted_loss                              \
            --cuda_device 0                              \
            --cuda                                       \
            --loading_epoch 0                            \
            --total_epochs 100                           \
            --outdim 2                                   \
            --scheduler                                  \
            --optim_type Adadelta                        \
            --lr 0.0001                                  \
            --weight_decay 0.02                          \
            --scheduler                                  \
            --bsize 16                                   \
            --means 0.49 0.58 0.33                       \
            --stds 0.15 0.15 0.19                        \
            --nworker 2                                  \
            --seg_idx $i                                 \
            --test_date Jul_24_2023                      \
            --cross_validation

# ResNet training
python3 ../classification/run.py      \
            --root_path  /Users/michelewiseman/Desktop/blackbird_ml \
            --model_type ResNet                          \
            --save_model                                 \
            --pretrained                                 \
            --weighted_loss                              \
            --cuda_device 0                              \
            --cuda                                       \
            --loading_epoch 0                            \
            --total_epochs 100                           \
            --optim_type SGD                             \
            --outdim 2                                   \
            --lr 0.0001                                  \
            --weight_decay 0.002                         \
            --scheduler                                  \
            --means 0.49 0.58 0.33                       \
            --stds 0.15 0.15 0.19                        \
            --bsize 16                                   \
            --nworker 2                                  \
            --seg_idx $i                                 \
            --test_date Jul_24_2023                      \
            --cross_validation

 # VGG training
 python3 ../classification/run.py      \
          --root_path  /Users/michelewiseman/Desktop/blackbird_ml \
          --model_type VGG                             \
          --save_model                                 \
          --pretrained                                 \
          --weighted_loss                              \
          --cuda_device 0                              \
          --cuda                                       \
          --loading_epoch 0                            \
          --total_epochs 100                           \
          --optim_type Adadelta                        \
          --lr 0.0001                                  \
          --weight_decay 0.002                         \
          --outdim 2                                   \
          --scheduler                                  \
          --means 0.49 0.58 0.33                       \
          --stds 0.15 0.15 0.19                        \
          --bsize 16                                   \
          --seg_idx $i                                 \
          --test_date Jul_24_2023                      \
          --cross_validation                           \
          --nworker 2

done



# To use: 
#
# 1. Change parameters if desired
# 2. Save
# 3. Open terminal
# 4. bash train.sh


#General Notes for above parameters:

#--root_path
#    - The main directory that will house subdirectories data, results, and model
#    - on PC, looks like this: /c/Users/michele.wiseman/Desktop/blackbird_ml
#    - on OSx, looks like this: /Users/michelewiseman/Desktop/blackbird_ml

#--model_type
#    - The type of deep learning model to train. Choices include:
#
#        - GoogleNet: GoogleNet is a 22-layer deep Convolutional Neural Network (CNN) introduced in 2014. It uses the inception module, which 
#            concatenates multiple filters with different sizes to form a more diverse representation of the image. The final layer of GoogleNet
#            is a softmax classifier, used for image classification. https://arxiv.org/abs/1409.4842
#
#        - ResNet: ResNet stands for Residual Network, introduced in 2015. ResNets are deep neural networks with hundreds or even thousands of 
#            layers, which were difficult to train before. The main idea behind ResNet is to use residual connections, where the input from one layer 
#            is added to the output of another layer. This allows the network to better preserve information and make the training of deep networks 
#            easier. https://arxiv.org/abs/1512.03385
#
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

#--pretrained
#    - If you are using a pretrained model (e.g. Inception3, DeepLab, etc., you should have the --pretrained flag)

#--weighted_loss
#    - Weighted loss is a loss function that assigns different weights or strengths to different errors or prediction mistakes in a model's output. 
#        The purpose of weighting the loss is to give more importance to certain errors or to penalize certain mistakes more than others, in order to 
#        better optimize the model to fit the desired target. The weights are specified as coefficients in the loss function, with larger weights given 
#        to the errors that the user wants to correct more urgently.
#    - Default if called is 1:3, but can customize in optimizer.py. 1:3 gives more weight to errors with classification of the infected class. 

#--resume
#    - To resume training on a pretrained model (often for fine tuning or if training was interupted). 

#--resume_timestamp
#    - Timestamp to resume training. 

#--save_model
#    - This flag saves your training model 
#    - You can customize when you save a model and how (e.g. file format and location) in run.py

#--seg_idx
#    - The segmentation index to be used in the cross-validation of the deep learning training script

#--loading_epoch
#    - The loading_epoch attribute represents the epoch at which the training process was previously interrupted. By starting the epoch loop from 
#        start_epoch = self.loading_epoch + 1, the code continues the training process from where it left off, rather than starting over from the 
#        beginning.
#    - Parameter customization in solver.py

#--total_epochs
#    -  One epoch is completed when all examples in the training dataset have been passed through the model once. This parameter specifies how 
#    many times you want that to happen.

#--cuda
#    - Include this tag if you have a CUDA enabled GPU
#    - Can check by running (in python after import torch): print(torch.cuda.is_available())
#    - More info here: https://developer.nvidia.com/cuda-toolkit

#--optimType
#    - Type of optimizer, can customize in solver.py:

#    - Adam (Adaptive Moment Estimation) - Adam is a gradient-based optimization algorithm that uses moving averages of the parameters to provide
#        running estimates of the second raw moments of the gradients; the moving averages are initialized with a default of 0 and are updated at 
#        each time step. Adam is often the default optimizer used in many deep learning applications due to its fast convergence and good performance
#        on a wide range of problems.

#    - Adadelta - Adadelta is an extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing learning rate. Instead of 
#        accumulating all past squared gradients, Adadelta restricts the window of accumulated past gradients to some fixed size w.

#    - RMSprop (Root Mean Squared Propagation) - RMSprop is a modification of gradient descent where the gradient is divided by a running average of
#        its recent magnitude. The idea is to ensure that the magnitude of the gradient is relatively consistent, preventing the learning rate from 
#        oscillating or becoming too large.

#    - SGD (Stochastic Gradient Descent) - SGD is the simplest deep learning optimization algorithm and involves updating the parameters in the 
#        direction of the negative gradient of the loss with respect to the parameters. SGD is highly sensitive to the learning rate and the choice of 
#        learning rate can greatly affect the performance of the model.

#--lr
#    - Learning rate
#    - By updating the learning rate, the model can be made to learn more quickly or slowly, depending on the specifics of the training process. 
#        The process of updating the learning rate after specified steps is known as scheduling the learning rate, and is commonly used in deep
#        learning to prevent overfitting or to fine-tune the training process.

#--weight_decay
#    - Weight decay is a regularization technique for reducing the magnitude of the weight values in a neural network model. It helps to prevent
#        overfitting and generalization of the model by reducing the magnitude of the weights and preventing them from becoming too large.The weight
#        decay parameter, commonly denoted as λ, is used to control the strength of the regularization. A larger value of λ results in more aggressive
#        weight decay, which encourages smaller weights, while a smaller value of λ results in less weight decay, which allows for larger weights. 
#        The weight decay term is added to the loss function and then optimized along with the other parameters of the model during training.

#--scheduler
#    - This flag sets up a learning rate scheduler for the optimizer in use.
#    - Can customize in solver.py

#--bsize = batch size
#    - If you have a small dataset and a simple model, you can start with a larger batch size, such as 128 or 256.
#    - If you have a larger dataset and a more complex model, you may need to use a smaller batch size, such as 32, to avoid running out of memory.
#    - If your model is overfitting, you may want to reduce the batch size to increase the amount of randomness in the training process.
#    - If your model is underfitting, you may want to increase the batch size to stabilize the training process and reduce the amount of randomness

#--nworker
#    - The number of dataloader workers refers to the number of parallel processes that are used to load the data in a deep learning model. The data
#        is typically loaded in batches (--bsize) to save memory and speed up training. By setting the number of dataloader workers, you specify how many 
#        processes will be used to load the data in parallel. Too few workers may slow down the data loading process, while too many workers may cause 
#        the system to become overwhelmed and lead to performance degradation. The optimal number of workers can depend on various factors, such as the 
#        available memory, the number of GPU/CPU cores, and the size of the batches (--bsize) being loaded.

#--cuda_device
#    - The ID of your CUDA-enabled GPU (if using one) in case you have more than one
#    - Default is 0
#    - To determine, run: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
#
#--means
#    - mean r g b values for you training set
#
#--stds 
#    - std dev of your r g b values for your training set 

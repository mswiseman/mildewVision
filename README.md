# Preface
Many scripts in this repository build off of [Tian Qiu's original repository](https://github.com/suptimq/Saliency_based_Grape_PM_Quantification) (used for [this paper](https://academic.oup.com/hr/article/doi/10.1093/hr/uhac187/6675613)). 

If you came here from Plant Health 2023 to familiarize yourself with computer vision, I suggest you start by checking out some simpler examples of computer vision implementation as these models and the associated code is quite complex. The company Roboflow (unaffilated) has some really awesome tutorials on getting started with computer vision, so I encourage you to check them out and work through the examples with them: 

* [Object detection](https://www.youtube.com/watch?v=wuZtUMEiKWY)<br>
* [Instance Segmentaton](https://www.youtube.com/watch?v=pFiGSrRtaU4&t=606s)<br>
* [Classification](https://www.youtube.com/watch?v=93kXzUOiYY4)<br>

This repo is still in progress, feel free to email me with any questions or clarifications: [wisemami@oregonstate.edu](mailto:wisemami@oregonstate.edu) 


# Introduction

The code in this repository primarily uses [Pytorch](https://pytorch.org/get-started/locally/) pretrained models to train and subsequently make inferences on leaf disks with or without powdery mildew. <br>

Overview of the training and inference process: <br>

<img width="836" alt="Overview" src="https://github.com/mswiseman/mildewVision/assets/33985124/6fa10b3c-fd77-43ad-80c3-115785dc5c7a">

# Image data

1 cm leaf disks were excised using ethanol disinfested leather punches and subsequently arrayed adaxial side onto up on 1% water agar plates. Image acquisition was performed using the Blackbird CNC Imaging Robot (version 2, developed by Cornell University and PrinterSys).  The Blackbird is a G-code driven CNC that positions a Nikon FT2 DLSR camera equipped with a 2.5x zoom ultra-macro lens (Venus Optics Laowa 25mm) in the X/Y position and then the camera captures images in a z-stack every 20 uM in Z-height.  The image stacking process is automated using the [stackPhotos.py](https://github.com/mswiseman/mildewVision/blob/main/blackbird_processing/stackPhotos.py) Python script. [Helicon Focus software](https://www.heliconsoft.com/software-downloads/) (Helicon Software, version 8.1) was utilized to perform the focus stacking, with the parameters set to method B (depth map radius: 1, smoothing radius: 4, and sharpness: 2). Blackbird datasheets were prepared using the [generateBlackbirdDatasheet.py](https://github.com/mswiseman/mildewVision/blob/main/blackbird_processing/generateBlackbirdDatasheet.py) script. 

# Implementation
*conda env coming soon...*

[CUDA](https://developer.nvidia.com/cuda-toolkit) is required for GPU usage; currently it's only available for PCs. Please check your GPU to figure out which version you need. If running on Apple Silicon, [MPS](https://developer.apple.com/metal/pytorch/) is necessary to take advantage of accelerated Pytorch. 

Package Requirements: 
torch torchvision tensorboard termcolor optuna pandas captum matplotlib pandas pillow scikit-learn glob2 optuna h5py hashlib opencv-python  

## Classification Training
To train your own model, you need:

1. image patches to make train/test/val .hdf5 files
   - you can make image patches using .preprocessing/[makePatches.py](https://github.com/mswiseman/mildewVision/blob/main/preprocessing/make_patches.py). It's easiest to sort these patches into different directories according to the label (e.g. if it has a dog in it, put it in the dog directory)
   - you can label these image patches in a given directory by adding a suffix to the filename using ./preprocessing/[rename_files.py](https://github.com/mswiseman/mildewVision/blob/main/preprocessing/rename_files.py)
   - you can then make train/test/val hdf5 files using ./preprocessing/[images_to_test_train_hdf5.py](https://github.com/mswiseman/mildewVision/blob/main/preprocessing/images_to_test_train_hdf5.py)

2. determine mean r/g/b values of your test/train/val sets using .preprocessing/[get_mean_std.py](https://github.com/mswiseman/mildewVision/blob/main/preprocessing/get_mean_std.py) and plug those into your ./script/[train.sh]() script under `--means` and `--stds` (super important...this dramatically effects your model performance). 

3. Customize other training parameters such as the model, learning rate, etc. See the argparse section in ./classification/run.py to see full list of customizable variables. You can start with the default values, but your model will perform much better if you try different base models and find the optimal hyperparamters (e.g. by using [Optuna](https://optuna.org/) hyperparameter engineering). 

## Segmentation Training
Coming soon...

## Testing
Coming soon...

## Inference
Coming soon...

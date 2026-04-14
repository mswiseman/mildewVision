# Preface
The Blackbird is a highthroughput phenomics imaging platform developed through collaboration of scientists and engineers at [Cornell AgriTech](https://cals.cornell.edu/cornell-agritech), the [USDA-ARS Grape Genetics Research Unit (GGRU)](https://www.ars.usda.gov/northeast-area/geneva-ny/grape-genetics-research-unit-ggru/), and [Moblanc Robotics](https://moblancrobotics.com/). Most code/scripts in this repository build off of [Tian Qiu's Grape PM Saliency mapping repository](https://github.com/suptimq/Saliency_based_Grape_PM_Quantification) (used for [this paper](https://academic.oup.com/hr/article/doi/10.1093/hr/uhac187/6675613)). 


This codebase is a continual work-in-progress as I'm always trying to improve our code, documentation, and models; alas, feel free to email me with any questions or clarifications: [wisemami@oregonstate.edu](mailto:wisemami@oregonstate.edu) 


# Introduction

The code in this repository primarily uses [Pytorch](https://pytorch.org/get-started/locally/) pretrained models to train and subsequently make inferences on leaf disks with or without powdery mildew. <br>

Overview of the training and inference process: <br>

![flow chart](aps2023_visuals/flowchart_of_modeL_training_v5.png)

# Implementation


[CUDA](https://developer.nvidia.com/cuda-toolkit) is required for GPU usage; currently it's only available for PCs. Please check your GPU to figure out which version you need. If running on Apple Silicon, [MPS](https://developer.apple.com/metal/pytorch/) is necessary to take advantage of accelerated Pytorch. <br>

**Package Requirements**: <br>
To install the required packages via conda, simply run `conda env create -f environment.yml` and then `conda activate mildewVision` to activate the environment.   <br><br>If running on **Google Colab**, check out a GPU (preferably A100 or better when training) and run: `!pip install optuna==3.1.0 termcolor` as the other packages should already be installed (as of 11/25/2025).  

![overview part 2](aps2023_visuals/overview.png)

## Classification Training
To train your own model, you need:<br>

1. A labeled image patch dataset to build the necessary train/test/val .hdf5 files
   - you can make image patches using [code/preprocessing/makePatches.py](code/preprocessing/make_patches.py). It's easiest to sort these patches into different directories according to the label (e.g. if infected, put in the "infected" directory. If not infected, put in the "healthy" directory)
   - In subsequent models, I would make patches by using the `--save_infected`, `--save_healthy` or `--save_discarded` tags when running [plot_leaf_sal.py](plot_leaf_sal.py), this way I could correct and add previously missclassified patches to my new dataset in hopes the next model iteration would learn the features better. 
   - you can then make a train/test/val hdf5 files (or k-fold splits) using [code/preprocessing/images_to_test_train_hdf5.py](code/preprocessing/images_to_test_train_hdf5.py)

2. To determine mean rgb chanel values for your test/train/val sets using [code/preprocessing/get_mean_std.py](code/preprocessing/get_mean_std.py) and plug those into your [code/scripts/train.sh](code/scripts/train.sh) script under `--means` and `--stds` (super important...this dramatically effects your model performance). 

3. Customize other training parameters such as the model, learning rate, etc. within the [code/scripts/train.sh](code/scripts/train.sh) script. See the argparse section in [code/classification/run.py](code/classification/run.py) to see full list of customizable variables. <br><br> Note: You can start with the default values, but your model will likely perform better if you try different base models and hyperparamter values (e.g. by using [Optuna](https://optuna.org/) hyperparameter optimization as shown below). Always cross-validate and test to ensure you're not overfitting though. 
![optuna](aps2023_visuals/optuna.png)


## Inference
Once you have downloaded our example [two-class hop powdery mildew model](https://drive.google.com/file/d/1-AsUV2SXraAWjazjMEYMv5mnDP6ABVmI/view?usp=drive_link) to [results/ResNet_Feb14_15-53-04_2024](results/ResNet_Feb14_15-53-04_2024), you should be able to activate the conda environment (`conda activate mildewVision`) and run either `bash ./code/scripts/plot_leaf_correlation_all.sh` or `bash ./code/scripts/plot_leaf_sal_map.sh` as a minimal working example. 

Once that's working, you can customize the argparse arguments in the [leaf_correleation_all.sh](code/scripts/plot_leaf_correlation_all.sh) bash script to run inference on multiple datasets in parallel (adjust the `MAX_JOBS` parameter according to your computational power).  In the example [here](./code/scripts/plot_leaf_correlation_all.sh), I have included commands for calling either [plot_leaf_sal_map.py](plot_sal_map_leaf.py) or  [leaf_correlation.py](leaf_correlation.py). Both code/scripts return the same .csv file that provides metadata about your run parameters, disease severity estimates, saliency metrics, etc., but [plot_leaf_sal_map.py](plot_sal_map_leaf.py) also returns visual outputs of patch disease severity as well as saliency maps (if you include the optional saliency tags, see example below). If you are running standard inference you may opt to call the [code/leaf_correlation.py](code/leaf_correlation.py) [script](code/scripts/plot_leaf_correlation_all.sh) instead as it runs 5-10x faster. 

<br>Example raw and deeplift saliency map (`--sal_deeplift`) output of from plot_sal_map.py:
![ Example raw and deeplift saliency map output](aps2023_visuals/saliency_output_example.png)

## Segmentation Training
*Coming soon...*

## Testing
*Coming soon...*

# Image data
1 cm leaf disks were excised using ethanol disinfested leather punches and subsequently arrayed adaxial side onto up on 1% water agar plates. Image acquisition was performed using the Blackbird CNC Imaging Robot (version 1 "Blackbird-Green", developed by Cornell University, USDA-ARS Grape Genetics Research Unit, and Moblanc Robotics).  The Blackbird is a G-code driven CNC that positions a Nikon Z 7II mirrorless camera equipped with a 2.5x zoom ultra-macro lens (Venus Optics Laowa 25mm) in the X/Y position and then the camera captures images in a z-stack every 200 µM in Z-height.  Blackbird datasheets can be prepared using the [generateBlackbirdDatasheet.py](code/blackbird_processing/generateBlackbirdDatasheet.py) script. The image stacking process is automated using the [stackPhotosParallel.py](code/blackbird_processing/stackPhotosParallel.py) Python script. [Helicon Focus software](https://www.heliconsoft.com/software-downloads/) (Helicon Software, version 8.1) was utilized to perform the focus stacking, with the parameters set to method B (depth map radius: 1, smoothing radius: 4, and sharpness: 2). <br><br>Example images can be viewed [here](https://app.box.com/folder/221778779975?s=cfuosvlzzldi53pbjocjmbnf2ymhrkwa).
![slow animation of mildew growth over time](aps2023_visuals/mildew_slow.gif)




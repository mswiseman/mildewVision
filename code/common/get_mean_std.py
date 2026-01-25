import os
import PIL
import h5py

import numpy as np

import torchvision.transforms as tvtrans

from pathlib import Path


"""Usage
Calculate mean and std of a dataset
"""

#directory = r'C:\Users\Intel User\Desktop\blackbird_scripts'
current_dir = Path(os.getcwd())
main_dir = Path(os.path.dirname(current_dir))
print("Main directory:", main_dir)
dataset_folder = main_dir / 'data'
print("Dataset folder:", dataset_folder)

train_set_filepath = current_dir / 'train.hdf5'
val_set_filepath = current_dir / 'val.hdf5'

# Load data
with h5py.File(train_set_filepath, 'r') as f:
    image_ds = f['images']
    train_images = image_ds[:, ]


with h5py.File(val_set_filepath, 'r') as f:
    image_ds = f['images']
    val_images = image_ds[:, ]


train_images_red = train_images[..., 0]
train_images_green = train_images[..., 1]
train_images_blue = train_images[..., 2]

val_images_red = val_images[..., 0]
val_images_green = val_images[..., 1]
val_images_blue = val_images[..., 2]

train_images_mean = (np.mean(train_images_red)/255, np.mean(
    train_images_green)/255, np.mean(train_images_blue)/255)
train_images_std = (np.std(train_images_red)/255, np.std(
    train_images_green)/255, np.std(train_images_blue)/255)

val_images_mean = (np.mean(val_images_red)/255, np.mean(
    val_images_green)/255, np.mean(val_images_blue)/255)
val_images_std = (np.std(val_images_red)/255, np.std(
    val_images_green)/255, np.std(val_images_blue)/255)

print(f'{train_images.shape[0]} training samples')
print(f'train mean {train_images_mean} std {train_images_std}')

print(f'{val_images.shape[0]} val samples')
print(f'val mean {val_images_mean} std {val_images_std}')

import h5py
import numpy as np
from pathlib import Path

'''Usage
Statistics of color factors

Written by Tian Qiu, Cornell University with modifications by Michele Wiseman, Oregon State University

To run:
python get_mean_std.py --train_set_filepath <path_to_train_set> 
'''

# Entire dataset
dataset_folder = Path('/Users/michelewiseman/Desktop/Saliency_based_Grape_PM_Quantification-main/')
train_set_filepath = dataset_folder / 'train_downy.hdf5'

# Load data
with h5py.File(train_set_filepath, 'r') as f:
    image_ds = f['images']
    train_images = image_ds[:, ]

train_images_red = train_images[..., 0]
train_images_green = train_images[..., 1]
train_images_blue = train_images[..., 2]

train_images_mean = (np.mean(train_images_red), np.mean(train_images_green), np.mean(train_images_blue))
train_images_std = (np.std(train_images_red), np.std(train_images_green), np.std(train_images_blue))

print(f'{train_images.shape[0]} training samples')
print(f'train mean {train_images_mean} std {train_images_std}')

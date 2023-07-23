import os
import PIL
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

import torchvision.transforms as tvtrans


"""Usage
Statistics of color factors

Written by Tian Qiu, Cornell University with modifications by Michele Wiseman, Oregon State University

To run:
    python stat_brightness.py --dataset_path <path_to_dataset> 

    e.g.
    
    python stat_brightness.py --dataset_path /d/Stacked/Deposition_Study/6-14-2023_6dpi/ 
"""


np.random.seed(2021)
# torch.seed(2021)

dataset_path = Path("/d/Stacked/Deposition_Study/6-14-2023_6dpi/")

trays = [1]

# Initialize lists
brightness = []
contrast = []
hue = []
saturation = []

transform = tvtrans.Compose([
    tvtrans.ColorJitter(brightness=[1.5, 1.7], contrast=[
                        0.8, 1.0], hue=[0.05, 0.1]),
])

# Loop trays
for tray in trays:
    dataset_tray_path = dataset_path / str(tray)
    patch_filenames = [x for x in os.listdir(
        dataset_tray_path) if x.endswith('.png')]
    # leaf_disk_image_filenames = [x for x in os.listdir(dataset_tray_path)]

    # # Loop leaf disk images
    # for leaf_disk_image_filename in leaf_disk_image_filenames:
    #     dataset_disk_path = dataset_tray_path / leaf_disk_image_filename
    #     patch_filenames = [x for x in os.listdir(
    #         dataset_disk_path) if x.startswith('tray')]

    # Loop image patches
    for patch_filename in patch_filenames:
        image_filepath = dataset_tray_path / patch_filename
        # image_filepath = dataset_disk_path / patch_filename

        # Load image
        img = PIL.Image.open(image_filepath)
        # img_transformed = transform(img)
        img_mode = img.mode
        h, s, v = img.convert('HSV').split()
        # img_arr = np.asarray(img)
        # img_arr_hsv = np.asarray(img_hsv)

        # Calculate brightness
        # hue_img = img_arr_hsv[..., 0]
        # saturation_img = img_arr_hsv[..., 1]
        # brightness_img = img_arr_hsv[..., 2]
        hue.append(np.mean(h))
        saturation.append(np.mean(s))
        brightness.append(np.mean(v))
        contrast.append(np.max(v) - np.min(v))

fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0, 0].hist(brightness)
axs[0, 0].set_title('Brightness')

axs[0, 1].hist(contrast)
axs[0, 1].set_title('Contrast')

axs[1, 0].hist(hue)
axs[1, 0].set_title('Hue')

axs[1, 1].hist(saturation)
axs[1, 1].set_title('Saturation')

# fig.tight_layout()
fig.suptitle('Color Factors')

plt.savefig('brightness.png')

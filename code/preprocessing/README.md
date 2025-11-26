# Preprocessing scripts

Use these scripts to prepare training data. <br>

**[1. make_patches.py](https://github.com/mswiseman/mildewVision/blob/main/preprocessing/make_patches.py)** <br>
Splits an image into 224 x 224 patches, converts the patches to rgb, and removes any patches that have black borders above the user-defined threshold.<br>

**[2. rename_files.py](https://github.com/mswiseman/mildewVision/blob/main/preprocessing/rename_files.py)** <br>
Indexes and adds a label suffix to the end of your patches. For example, if you were to run: <br>
`python rename_files.py -directory <path_to_files> -suffix _clear.png -index_start 1` <br>

then renaming would look like this:
```
- image1.png > 1_clear.png
- image2.png > 2_clear.png
- image3.png > 3_clear.png
```

**[3. images_to_test_train_hdf5.py](https://github.com/mswiseman/mildewVision/blob/main/preprocessing/images_to_test_train_hdf5.py)** <br>
Takes .png image patches, resizes them, converts them to rgb, removes blurry photos, checks for and removes duplicate patches, and then creates test/train/val hdf5 files.<br>

**[4. view_hdf5_images.py](https://github.com/mswiseman/mildewVision/blob/main/preprocessing/view_hdf5_images.py)** <br>
Randomly displays 10 indices from an hdf5 file; helpful as a sanity check.<br>

**[5. stat_brightness.py](https://github.com/mswiseman/mildewVision/blob/main/preprocessing/stat_brightness.py)** <br>
Statistics of color factors, written by Tian Qiu and modfied by me. This helps check for any inconsistencies. <br>

**[6. get_mean_std.py](https://github.com/mswiseman/mildewVision/blob/main/preprocessing/get_mean_std.py)** <br>
Calculate mean and std deviation of rgb values of your train/test/val sets. This is important for normalization.

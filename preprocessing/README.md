# Preprocessing scripts

Use these scripts to prepare training data. 

**1. make_patches.py**
Splits an image into 224 x 224 patches, converts the patches to rgb, and removes any patches that have black borders above the user-defined threshold

**2. images_to_test_train_hdf5.py**
Takes .png image patches, resizes them, converts them to rgb, removes blurry photos, checks for and removes duplicate patches, and then creates test/train/val hdf5 files.

**3. view_hdf5_images.py**
Randomly displays 10 indices from an hdf5 file; helpful as a sanity check.

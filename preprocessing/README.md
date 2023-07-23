# Preprocessing scripts

Use these scripts to prepare training data. <br><br>

**1. make_patches.py** <br>
Splits an image into 224 x 224 patches, converts the patches to rgb, and removes any patches that have black borders above the user-defined threshold<br><br>

**2. images_to_test_train_hdf5.py**<br>
Takes .png image patches, resizes them, converts them to rgb, removes blurry photos, checks for and removes duplicate patches, and then creates test/train/val hdf5 files.<br><br>

**3. view_hdf5_images.py**<br>
Randomly displays 10 indices from an hdf5 file; helpful as a sanity check.<br><br>

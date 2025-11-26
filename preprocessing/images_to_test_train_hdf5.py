import argparse
import os

import h5py
import random
import numpy as np
import hashlib
from PIL import Image
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import glob

"""
v3 Developed by Michele Wiseman of Oregon State University
11/25/2025

This script is used for preparing labeled image data to train a machine learning model.

The script has several steps:
1. Load images from the specified directory, resize them to a common size, and convert them to RGB.
2. Remove duplicate images from the dataset.
3. Calculate a threshold for determining whether an image is blurry based on the variance of its Laplacian.
4. Remove blurry images based on this threshold.
5. Randomly shuffle the remaining images.
6. Split the shuffled images into training, testing, and validation datasets.
7. Save each dataset to a separate HDF5 file.

Functions:
- find_duplicate_images_and_labels: Checks for duplicate images and labels in the dataset and removes them.
- variance_of_laplacian: Calculates the variance of the Laplacian of an image, which is used to measure the blurriness of the image.
- grayscale_and_vola: Converts an image to grayscale and calculates its variance of Laplacian.
- median_threshold: Calculates the median variance of Laplacian of a subset of the images, which is used as the threshold for removing blurry images.

Variables:
- input_dir: Directory where the input images are located.
- output_dir: Directory where the output HDF5 files are saved.
- image_size: The common size to which all images are resized.
- sample_size: The number of images used to calculate the median variance of Laplacian.
- blur_threshold_factor: A factor subtracted from the median variance of Laplacian to determine the threshold for removing blurry images.
- k_fold: Whether to perform k-fold cross validation.
- k_fold_number: The number of folds for k-fold cross validation.

To use:
python images_to_test_train_hdf5.py -sample_size 100 -blur_threshold_factor 30 -image_size 224 224 -input_dir /Users/michelewiseman/Desktop/test -output_dir /Users/michelewiseman/Desktop/test
"""

# Add argparse arguments
parser = argparse.ArgumentParser(description="Preprocessing images script")
parser.add_argument('-sample_size', type=int, default=1000, help="Number of images used to calculate the median threshold")
parser.add_argument('-blur_threshold_factor', type=int, default=100, help="Factor subtracted from the median variance of Laplacian to determine the threshold for removing blurry images")
parser.add_argument('-image_size', type=int, default=(224, 224), nargs=2, help="The size to which all images are resized")
parser.add_argument('-input_dir', type=str, default=r'C:\Users\Intel User\Desktop\Downy', help="Directory where the input images are located")
parser.add_argument('-output_dir', type=str, default=r'C:\Users\Intel User\Desktop\Downy', help="Directory where the output HDF5 files are saved")
parser.add_argument('-k_fold', action='store_true', help="Perform k-fold cross validation")
parser.add_argument('-k_fold_number', type=int, default=5, help="Number of folds for k-fold cross validation")
parser.add_argument('--use_existing_hdf5', action='store_true',
                    help="Load one or more existing HDF5 files and combine them before k-fold.")
parser.add_argument('--hdf5_files', type=str, nargs='*', default=None,
                    help="Explicit list of HDF5 files to load (overrides --hdf5_dir).")
parser.add_argument('--hdf5_dir', type=str, default=None,
                    help="Directory to scan for .hdf5 if --hdf5_files not given.")
parser.add_argument('--dedupe_loaded', action='store_true',
                    help="Run duplicate removal after loading HDF5s (hash-based on image bytes + label).")

args = parser.parse_args()

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Parameters from command line arguments
sample_size = args.sample_size
blur_threshold_factor = args.blur_threshold_factor
image_size = tuple(args.image_size)
input_dir = args.input_dir
output_dir = args.output_dir


def find_duplicate_images_and_labels(images, labels, dedupe_by='image+label'):
    """
    Remove duplicates. `dedupe_by`:
      - 'image+label' (default): treat (image bytes, label) pairs as unique
      - 'image': treat image bytes alone as unique (ignore label when deduping)
    Returns: (duplicates, images_nd, labels_nd) where labels_nd has shape (N,1)
    """
    image_dict = {}
    duplicates = []
    non_dup_images = []
    non_dup_labels = []

    # Ensure labels are integers and 2D shape (N,1)
    labels = np.asarray(labels)
    if labels.ndim == 1:
        labels = labels[:, None]
    labels = labels.astype(np.int64, copy=False)

    for i in range(len(images)):
        img = np.ascontiguousarray(images[i])  # normalize memory layout for consistent bytes
        img_hash = hashlib.md5(img.view(np.uint8)).hexdigest()

        # Make label a Python int for hashing
        label_scalar = int(np.ravel(labels[i])[0])

        key = (img_hash, label_scalar) if dedupe_by == 'image+label' else img_hash

        if key in image_dict:
            duplicates.append((i, image_dict[key]))
            continue

        image_dict[key] = i
        non_dup_images.append(images[i])
        non_dup_labels.append(label_scalar)

    print(f'Number of duplicates found: {len(duplicates)}')
    print('..................................................')

    return duplicates, np.array(non_dup_images), np.array(non_dup_labels, dtype=np.int64).reshape(-1, 1)

def load_hdf5_datasets(paths):
    imgs_list, labels_list = [], []
    for p in paths:
        with h5py.File(p, 'r') as f:
            if 'images' not in f or 'labels' not in f:
                raise KeyError(f"{p} missing 'images' or 'labels' dataset.")
            imgs = f['images'][:]
            lbls = f['labels'][:]
            imgs_list.append(imgs)
            labels_list.append(lbls)
            print(f"Loaded {imgs.shape[0]} samples from {os.path.basename(p)}")
    images = np.concatenate(imgs_list, axis=0)

    labels = np.concatenate(labels_list, axis=0)
    if labels.ndim == 1:
        labels = labels[:, None]
    elif labels.ndim == 2 and labels.shape[1] == 1:
        pass
    else:
        raise ValueError(f"Labels must be shape (N,) or (N,1), got {labels.shape}")
    labels = labels.astype(np.int64, copy=False)
    return images, labels

def gather_hdf5_paths(hdf5_files, hdf5_dir, default_dir):
    """Resolve list of .hdf5 files to load."""
    if hdf5_files:
        return hdf5_files
    search_dir = hdf5_dir or default_dir
    if not search_dir:
        raise ValueError("Provide --hdf5_files or --hdf5_dir or ensure default --output_dir is set.")
    # Prefer common names; fall back to all .hdf5
    candidates = []
    patterns = [
        os.path.join(search_dir, "train_*.hdf5"),
        os.path.join(search_dir, "val_*.hdf5"),
        os.path.join(search_dir, "test_*.hdf5"),
        os.path.join(search_dir, "train.hdf5"),
        os.path.join(search_dir, "val.hdf5"),
        os.path.join(search_dir, "test.hdf5"),
        os.path.join(search_dir, "*.hdf5"),
    ]
    seen = set()
    for pat in patterns:
        for p in glob.glob(pat):
            if p not in seen:
                candidates.append(p)
                seen.add(p)
    if not candidates:
        raise FileNotFoundError(f"No .hdf5 files found in {search_dir}")
    return candidates

def img_hashes(h5_path):
    with h5py.File(h5_path, 'r') as f:
        imgs = f['images'][:]
    # hash each image consistently
    return {hashlib.md5(np.ascontiguousarray(im).view(np.uint8)).hexdigest() for im in imgs}

#def variance_of_laplacian(image):
#    return cv2.Laplacian(image, cv2.CV_64F).var()


#def grayscale_and_vola(image):
#    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#    vol = variance_of_laplacian(gray)
#    return gray, vol


#def median_threshold(images, sample_size):
#    images_list = list(images)  # Convert numpy array to list
#    sampled_images = random.sample(images_list, sample_size)
#    vol_values = []
#    for img in sampled_images:
#        _, vol = grayscale_and_vola(np.array(img))  # Convert list to numpy array
#        vol_values.append(vol)
#    print(f'Median threshold for {sample_size} sampled images: {np.median(vol_values)}')
#    return np.median(vol_values)


if args.use_existing_hdf5:
    print("Loading datasets from existing HDF5 files...")
    hdf5_paths = gather_hdf5_paths(args.hdf5_files, args.hdf5_dir, output_dir)
    print("Files to combine:")
    for p in hdf5_paths:
        print(" -", p)
    images, labels = load_hdf5_datasets(hdf5_paths)

    # Optional de-duplication across files
    if args.dedupe_loaded:
        print('Removing duplicate images across loaded HDF5 datasets')
        _, images, labels = find_duplicate_images_and_labels(
            images, labels, dedupe_by='image+label'
        )

else:
    # ------- ORIGINAL FILESYSTEM LOADING PIPELINE -------
    print("Resizing and converting to RGB")
    print('..................................................')

    images = []
    labels = []
    for filename in os.listdir(input_dir):
        fname_lower = filename.lower()
        if fname_lower.endswith(('.png', '.jpg', '.jpeg')):
            try:
                with Image.open(os.path.join(input_dir, filename)) as img:
                    img = img.resize(image_size).convert('RGB')
                    img_arr = np.array(img)

                # Extract label from filename
                label = filename.split('_')[-1].split('.')[0].lower()
                if label == 'clear':
                    label_num = 0
                elif label == 'infected':
                    label_num = 1
                else:
                    raise ValueError(f"Unexpected label in filename: {label}")

                labels.append(label_num)
                images.append(img_arr)  # only append if label parsed OK

            except (IndexError, ValueError) as e:
                print(f"Skipping {filename}: {e}. Expected format 'image_label.png'.")

    images = np.array(images)
    labels = np.array(labels)

    # Remove duplicates
    print('Removing duplicate images')
    _, images, labels = find_duplicate_images_and_labels(images, labels)
    labels = np.expand_dims(labels, axis=-1)

# Calculate the median threshold
#print('Calculating median variance of laplacian threshold')
#threshold = median_threshold(images, sample_size)

# Remove blurry images
#excluded_images_count = 0
#non_blurry_images = []
#non_blurry_labels = []

#for i, img in enumerate(images):
#    _, vol = grayscale_and_vola(img)
#    if vol >= (threshold - blur_threshold_factor):
#        non_blurry_images.append(img)
#        non_blurry_labels.append(labels[i])
#    else:
#        excluded_images_count += 1

#print(f"Excluded {excluded_images_count} images due to blurriness")
#print(f'..................................................')

#images = np.array(non_blurry_images)
#labels = np.array(non_blurry_labels)
labels = np.expand_dims(labels, axis=-1)

# Count the labels
print('Counting labels')
print(f"Number of clear labels: {np.count_nonzero(labels == 0)}")
print(f"Number of infected labels: {np.count_nonzero(labels == 1)}")
#print(f"Number of conidiophores labels: {np.count_nonzero(labels == 2)}")
print(f'..................................................')

# Shuffling and creating hdf5
indices = np.arange(len(images))
images = images[indices]
labels = labels[indices]

if args.k_fold:
    # Build a 1-D label vector for splitting
    y1d = labels.reshape(-1) if labels.ndim > 1 else labels
    k_folds = args.k_fold_number

    # sanity check: at least k samples per class
    uniq, cnts = np.unique(y1d, return_counts=True)
    for cls, c in zip(uniq, cnts):
        if c < k_folds:
            raise ValueError(f"Class {cls} has only {c} samples, need >= n_splits={k_folds}.")

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=SEED)

    for fold, (train_idx, test_idx) in enumerate(skf.split(images, y1d)):
        X_train_full, X_test = images[train_idx], images[test_idx]
        y_train_full_1d, y_test_1d = y1d[train_idx], y1d[test_idx]

        # make validation from the training part; stratify requires 1-D
        X_train, X_val, y_train_1d, y_val_1d = train_test_split(
            X_train_full, y_train_full_1d,
            test_size=0.1, stratify=y_train_full_1d, random_state=SEED
        )

        # reshape labels back to (N,1) for saving
        y_train = y_train_1d.reshape(-1, 1)
        y_val   = y_val_1d.reshape(-1, 1)
        y_test  = y_test_1d.reshape(-1, 1)

        print(f"[Fold {fold}] Train: {X_train.shape}, {y_train.shape}")
        print(f"[Fold {fold}] Val:   {X_val.shape}, {y_val.shape}")
        print(f"[Fold {fold}] Test:  {X_test.shape}, {y_test.shape}")
        print('..................................................')

        with h5py.File(os.path.join(output_dir, f'train_{fold}.hdf5'), 'w') as f:
            f.create_dataset('images', data=X_train)
            f.create_dataset('labels', data=y_train)
        with h5py.File(os.path.join(output_dir, f'val_{fold}.hdf5'), 'w') as f:
            f.create_dataset('images', data=X_val)
            f.create_dataset('labels', data=y_val)
        with h5py.File(os.path.join(output_dir, f'test_{fold}.hdf5'), 'w') as f:
            f.create_dataset('images', data=X_test)
            f.create_dataset('labels', data=y_test)


else:
    print('Shuffling images and creating hdf5')
    np.random.shuffle(indices)
    images = images[indices]
    labels = labels[indices]
    print('Splitting into test/train/val datasets')
    num_train = int(0.8 * len(indices))
    num_test = int(0.1 * len(indices))
    num_val = len(indices) - num_train - num_test

    indices_train = indices[:num_train]
    indices_test = indices[num_train:num_train + num_test]
    indices_val = indices[num_train + num_test:]

    train_data = images[indices_train]
    train_labels = labels[indices_train]

    test_data = images[indices_test]
    test_labels = labels[indices_test]

    val_data = images[indices_val]
    val_labels = labels[indices_val]

    # Printing the shape of each split
    print(f"Training set shape: {train_data.shape}, {train_labels.shape}")
    print(f"Testing set shape: {test_data.shape}, {test_labels.shape}")
    print(f"Validation set shape: {val_data.shape}, {val_labels.shape}")
    print(f'..................................................')

    # Save the datasets to new HDF5 files
    print('Saving to HDF5 files')

    with h5py.File(os.path.join(output_dir, f'train.hdf5'), 'w') as train_f:
        train_f.create_dataset('images', data=train_data)
        train_f.create_dataset('labels', data=train_labels)

    with h5py.File(os.path.join(output_dir, f'test.hdf5'), 'w') as test_f:
        test_f.create_dataset('images', data=test_data)
        test_f.create_dataset('labels', data=test_labels)

    with h5py.File(os.path.join(output_dir, f'val.hdf5'), 'w') as val_f:
        val_f.create_dataset('images', data=val_data)
        val_f.create_dataset('labels', data=val_labels)

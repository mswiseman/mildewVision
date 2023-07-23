import argparse
import os
import cv2
import h5py
import random
import numpy as np
import hashlib
from PIL import Image
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

"""
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
parser.add_argument('-input_dir', type=str, default='/Users/michelewiseman/Desktop/downy_model', help="Directory where the input images are located")
parser.add_argument('-output_dir', type=str, default='/Users/michelewiseman/Desktop/downy_model', help="Directory where the output HDF5 files are saved")
parser.add_argument('-k_fold', action='store_true', help="Perform k-fold cross validation")
parser.add_argument('-k_fold_number', type=int, default=5, help="Number of folds for k-fold cross validation")
args = parser.parse_args()

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Parameters from command line arguments
sample_size = args.sample_size
blur_threshold_factor = args.blur_threshold_factor
image_size = args.image_size
input_dir = args.input_dir
output_dir = args.output_dir


def find_duplicate_images_and_labels(images, labels):
    image_dict = {}
    duplicates = []
    non_duplicate_images = []
    non_duplicate_labels = []

    for i in range(len(images)):
        img = images[i]
        img_hash = hashlib.md5(img.tobytes()).hexdigest()
        label = labels[i]
        if (img_hash, label) in image_dict:
            duplicates.append((i, image_dict[(img_hash, label)]))
        else:
            image_dict[(img_hash, label)] = i
            non_duplicate_images.append(img)
            non_duplicate_labels.append(label)

    print(f'Number of duplicate labels found: {len(duplicates)}')
    print(f'..................................................')

    return duplicates, np.array(non_duplicate_images), np.array(non_duplicate_labels)


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def grayscale_and_vola(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    vol = variance_of_laplacian(gray)
    return gray, vol


def median_threshold(images, sample_size):
    images_list = list(images)  # Convert numpy array to list
    sampled_images = random.sample(images_list, sample_size)
    vol_values = []
    for img in sampled_images:
        _, vol = grayscale_and_vola(np.array(img))  # Convert list to numpy array
        vol_values.append(vol)
    print(f'Median threshold for {sample_size} sampled images: {np.median(vol_values)}')
    return np.median(vol_values)


# Load, resize, and convert to RGB
print("Resizing and converting to RGB")
print('..................................................')

images = []
labels = []
for filename in os.listdir(input_dir):
    if filename.endswith('.png'):
        img = Image.open(os.path.join(input_dir, filename))
        img = img.resize(image_size).convert('RGB')
        img_arr = np.array(img)

        # Extract label from filename
        try:
            label = filename.split('_')[-1].split('.')[0]
            if label == 'clear':
                label_num = 0
            elif label == 'infected':
                label_num = 1
            elif label == 'conidiophores':
                label_num = 2
            else:
                raise ValueError(f"Unexpected label in filename: {label}")
            labels.append(label_num)
        except IndexError:
            print(f"Invalid filename {filename}, skipping file")
            print(f"File needs to be in the format: 'image_label.png'")
        except ValueError as ve:
            print(ve)
            print(f"File {filename} skipped")

        images.append(img_arr)

# Convert to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Remove duplicates
print('Removing duplicate images')
_, images, labels = find_duplicate_images_and_labels(images, labels)

# Calculate the median threshold
print('Calculating median variance of laplacian threshold')
threshold = median_threshold(images, sample_size)

# Remove blurry images
excluded_images_count = 0
non_blurry_images = []
non_blurry_labels = []

for i, img in enumerate(images):
    _, vol = grayscale_and_vola(img)
    if vol >= (threshold - blur_threshold_factor):
        non_blurry_images.append(img)
        non_blurry_labels.append(labels[i])
    else:
        excluded_images_count += 1

print(f"Excluded {excluded_images_count} images due to blurriness")
print(f'..................................................')

images = np.array(non_blurry_images)
labels = np.array(non_blurry_labels)
labels = np.expand_dims(labels, axis=-1)

# Count the labels
print('Counting labels')
print(f"Number of clear labels: {np.count_nonzero(labels == 0)}")
print(f"Number of infected labels: {np.count_nonzero(labels == 1)}")
print(f"Number of conidiophores labels: {np.count_nonzero(labels == 2)}")
print(f'..................................................')

# Shuffling and creating hdf5
print('Shuffling images and creating hdf5')
indices = np.arange(len(images))
np.random.shuffle(indices)
images = images[indices]
labels = labels[indices]

if args.k_fold:
    # Defining the number of splits for the k-fold cross-validation
    k_folds = args.k_fold_number
    kf = KFold(n_splits=k_folds)

    for fold, (train_indices, test_indices) in enumerate(kf.split(images)):
        train_data, test_data = images[train_indices], images[test_indices]
        train_labels, test_labels = labels[train_indices], labels[test_indices]

        # Do the same for validation set if you wish
        # Or split the test set into half for test and validation sets
        # val_data = ...
        # val_labels = ...

        # Print the shape of each split
        print(f"Training set shape for fold {fold}: {train_data.shape}, {train_labels.shape}")
        print(f"Testing set shape for fold {fold}: {test_data.shape}, {test_labels.shape}")
        # print(f"Validation set shape for fold {fold}: {val_data.shape}, {val_labels.shape}")
        print(f'..................................................')

        # Save the datasets to new HDF5 files
        print('Saving to HDF5 files')

        with h5py.File(os.path.join(output_dir, f'train_{fold}.hdf5'), 'w') as train_f:
            train_f.create_dataset('images', data=train_data)
            train_f.create_dataset('labels', data=train_labels)

        with h5py.File(os.path.join(output_dir, f'test_{fold}.hdf5'), 'w') as test_f:
            test_f.create_dataset('images', data=test_data)
            test_f.create_dataset('labels', data=test_labels)

        # Uncomment the following if you create validation sets
        # with h5py.File(os.path.join(output_dir, f'val_{fold}.hdf5'), 'w') as val_f:
        #     val_f.create_dataset('images', data=val_data)
        #     val_f.create_dataset('labels', data=val_labels)

    else:
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

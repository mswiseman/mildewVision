import h5py
import matplotlib.pyplot as plt
import random

'''
To run this script, you need to have the following packages installed:
    - h5py
    - matplotlib
    - random

Run this script from the command line with:
    python view_hdf5_images.py --input_hdf5 <path_to_hdf5_file>
'''


# change this to the path of your HDF5 file
input_hdf5 = "/Users/michelewiseman/Desktop/test/test.hdf5"

# Open the HDF5 file
with h5py.File(input_hdf5, 'r') as f:
    # Get the images and labels
    images = f['images']
    labels = f['labels'][:]

    # Select 10 random indices
    random_indices = random.sample(range(len(labels)), 10)

    # Get the images and labels for the selected indices
    selected_images = [images[i] for i in random_indices]
    selected_labels = [labels[i] for i in random_indices]

    # Plot the selected images with their corresponding labels
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    for i, (image, label, idx) in enumerate(zip(selected_images, selected_labels, random_indices)):
        row = i // 5
        col = i % 5
        axs[row, col].imshow(image)
        axs[row, col].set_title(f"Label: {label}, Index: {idx}")
        axs[row, col].axis('off')
    plt.tight_layout()
    plt.show()

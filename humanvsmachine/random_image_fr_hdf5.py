import os
import h5py
import numpy as np
from PIL import Image
import random

# this randomly samples images from our test dataset 

dataset_filepath = r'C:\Users\Intel User\Downloads\test.hdf5'

# Open the HDF5 file
with h5py.File(dataset_filepath, 'r') as f:
    # Read the images and labels datasets
    images = f['images']
    labels = f['labels']

    # Convert labels to a list
    labels_list = list(labels)

    # Get the indices of images of each type
    clear_indices = [i for i, label in enumerate(labels_list) if label == 0]
    infected_indices = [i for i, label in enumerate(labels_list) if label == 1]
    conidiophore_indices = [i for i, label in enumerate(labels_list) if label == 2]

    # Randomly select 34 images from each category
    selected_clear_indices = random.sample(clear_indices, min(34, len(clear_indices)))
    selected_infected_indices = random.sample(infected_indices, min(34, len(infected_indices)))
    selected_conidiophore_indices = random.sample(conidiophore_indices, min(34, len(conidiophore_indices)))

    # Save the selected images
    for image_type, selected_indices in [('clear', selected_clear_indices), ('infected', selected_infected_indices), ('conidiophore', selected_conidiophore_indices)]:
        for index in selected_indices:
            # Read the image data
            image_data = images[index]

            # Convert to PIL Image
            img = Image.fromarray(np.uint8(image_data))

            # Save the image
            img.save(os.path.join(r"C:\Users\Intel User\Desktop\practice", f'{index}_{image_type}.png'))

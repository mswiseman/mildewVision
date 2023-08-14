import os
import shutil

# This script adds an indexed suffix to the end of your file name. Be sure to change the suffix depending on the label of your data. 

directory = r'D:\clear_patches_for_train'

i = 1

for filename in os.listdir(directory):
    print("Renaming files...")
    if filename.endswith('.png'):
        new_filename = f'{i}_clear.png'
        i += 1
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)

        # Check if the new filename already exists, and if so, remove it
        if os.path.exists(new_path):
            os.remove(new_path)

        shutil.move(old_path, new_path)


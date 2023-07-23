import os
import shutil

'''
This script renames all files in a directory to a sequential index, with an optional suffix. 

For example, if you were to run: 
python rename_files.py -directory <path_to_files> -suffix _clear.png -index_start 1

then renaming would look like this:
    - image1.png > 1_clear.png
    - image2.png > 2_clear.png
    - image3.png > 3_clear.png
'''

directory = '/Users/michelewiseman/Desktop/healthy'
suffix = '_clear.png'
index_start = 1

i = index_start

for filename in os.listdir(directory):
    print("Renaming files...")
    if filename.endswith('.png'):
        new_filename = f'{i}{suffix}'
        i += 1
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)

        # Check if the new filename already exists, and if so, remove it
        if os.path.exists(new_path):
            os.remove(new_path)

        shutil.move(old_path, new_path)

import numpy as np
import imageio.v2 as imageio
import os


def create_morph_gif(image_files, output_file, steps):
    images = [imageio.imread(file) for file in image_files]

    frames = []
    for i in range(len(images)):
        for t in np.linspace(0, 1, steps):
            morphed_image = (1 - t) * images[i] + t * images[(i + 1) % len(images)]
            frames.append(morphed_image.astype(np.uint8))

    imageio.mimsave(output_file, frames, 'GIF', duration=0.1)

# Usage:
directory = '/Users/michelewiseman/Desktop/aps_visuals/'
os.chdir(directory)  # change the current working directory to the directory where the images are stored
create_morph_gif(["Picture1.png", "Picture2.png", "Picture3.png", "Picture4.png", "Picture6.png"],
                 "mildew_slow.gif", 15)

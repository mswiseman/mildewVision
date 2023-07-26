import imageio
import os
from PIL import Image, ImageSequence

def create_gif(image_files, output_file, duration, fade_duration):
    """
    Create a GIF from a list of image files, with control over the cycle speed and a fading effect
    between frames.

    Args:
        image_files: List of paths to the input image files.
        output_file: Path to the output GIF file.
        duration: Duration to display each frame, in seconds.
        fade_duration: Duration of the fade effect between frames, in seconds.
    """
  
    # Create a list to hold the frames of the GIF
    frames = []

    # Load each image file
    for filename in image_files:
        image = Image.open(filename)

        # If fade_duration is greater than 0, create a fading effect
        if fade_duration > 0:
            # Create a sequence of images that gradually fade in the next image
            fade_sequence = ImageSequence.Iterator(
                Image.blend(image, next_image, alpha)
                for alpha in np.linspace(0, 1, int(fade_duration / duration))
            )

            # Add the fade sequence to the list of frames
            frames.extend(list(fade_sequence))
        else:
            frames.append(image)

    # Create the GIF
    imageio.mimsave(output_file, frames, 'GIF', duration=duration)

# Usage:
# create_gif(["image1.png", "image2.png", "image3.png"], "output.gif", 0.5, 0.1)

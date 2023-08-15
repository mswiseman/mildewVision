import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from matplotlib.animation import FuncAnimation

# Set patch size
patch_size = (100, 100)  # example patch size, change as needed

# Load your image
im = Image.open("/Users/michelewiseman/Desktop/blackbird_ml/results/VGG_upth0.8_downth0.2_Jul09_22-29-36_2023/6-28-2023_6dpi/6_['1']_106-21326M_R1/6_106-21326M_R1_masked.png")
width, height = im.size

# Convert image to numpy array
img = np.array(im)

# Create a figure with two subplots: one for the image, one for the patches
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Set background color of the whole figure
fig.patch.set_facecolor('black')

# Or set the background color of each subplot
ax1.set_facecolor('black')
ax2.set_facecolor('black')

# Display the image
ax1.imshow(img)
ax1.axis('off')

# We'll add a red rectangle to this plot to highlight the current patch
rect = patches.Rectangle((0, 0), patch_size[0], patch_size[1], linewidth=1, edgecolor='white', facecolor='none')
ax1.add_patch(rect)

# We'll show the current patch in this plot
ax2.axis('off')


# This function updates the figure for each frame of the animation
def update(coords):
    # Unpack coords into i and j
    i, j = coords

    # Calculate the current patch coordinates
    y = i * patch_size[0]
    x = j * patch_size[1]

    # Extract the current patch
    patch = img[y:y + patch_size[1], x:x + patch_size[0]]

    # Check if the patch is black
    if np.sum(patch) == 0:
        # If the patch is black, clear ax2 and display a message
        ax2.clear()
        ax2.set_title("Black patch, no magnification", color='white')
        ax2.axis('off')
    else:
        # If the patch is not black, display it
        ax2.imshow(patch)
        ax2.set_title(f"Patch ({j}, {i})", color='white')

    # Update the position of the red rectangle
    rect.set_xy((x, y))


# Calculate total frames
total_frames = (height // patch_size[1]) * (width // patch_size[0])

# Create the animation
ani = FuncAnimation(fig, update,
                    frames=((i, j) for i in range(height // patch_size[1]) for j in range(width // patch_size[0])),
                    interval=400, save_count=total_frames)

# Save the animation as a gif
ani.save('patch_animation.gif', writer='pillow')

plt.close()

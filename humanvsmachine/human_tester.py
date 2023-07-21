import os
import random
import time
import pandas as pd
from PIL import Image, ImageTk
from tkinter import messagebox
import tkinter as tk
import argparse

# Argument parser for command line options
parser = argparse.ArgumentParser()
parser.add_argument('--practice', action='store_true')  # Add a boolean 'practice' argument
parser.add_argument('--name', type=str)  # Add a string 'name' argument
args = parser.parse_args()  # Parse the provided arguments

# Create a DataFrame to store the responses
df = pd.DataFrame(columns=['image_id', 'response', 'time_taken'])

# Get the list of image files
image_folder = r"C:\Users\Intel User\Desktop\test_set"

# If the 'practice' argument is True, change the image folder
if args.practice:
    image_folder = r"C:\Users\Intel User\Desktop\practice"

image_files = os.listdir(image_folder)  # List all files in the image folder

# Shuffle the list of image files
random.shuffle(image_files)

# Create a new Tkinter window
window = tk.Tk()

# Create a label to display the image
image_label = tk.Label(window)
image_label.pack()

# Function to handle button clicks
def handle_button_click(response):
    global start_time

    # Calculate the time taken
    time_taken = time.time() - start_time

    # Store the response in the DataFrame
    global df
    df = pd.concat([df, pd.DataFrame({
        'image_id': [image_files[i]],
        'response': [response],
        'time_taken': [time_taken],
    })], ignore_index=True)

    # Go to the next image
    next_image()

# Define the labels for the response buttons
response_labels = ["0_clear", "1_hyphae", "2_conidiophores"]
for response in [0, 1, 2]:
    button = tk.Button(window, text=response_labels[response], command=lambda response=response: handle_button_click(response))
    button.pack()

# Variable to keep track of the current image index
i = -1

# Get the name from the arguments
name = args.name

# Function to go to the next image
def next_image():
    global i, start_time

    # Increment the image index
    i += 1

    # Check if we've gone through all the images
    if i >= len(image_files):
        # Save the DataFrame to a CSV file with the name from the arguments
        df.to_csv(f'{name}_response.csv', index=False)

        # Display a completion message
        messagebox.showinfo("Complete", "All done!")

        # Close the window
        window.destroy()
        return

    # Open the next image
    img = Image.open(os.path.join(image_folder, image_files[i]))

    # Convert the image to a format Tkinter can use
    tk_img = ImageTk.PhotoImage(img)

    # Display the image
    image_label.configure(image=tk_img)
    image_label.image = tk_img

    # Get the start time
    start_time = time.time()

# Start with the first image
next_image()

# Start the Tkinter event loop
window.mainloop()

# Save the DataFrame to a CSV file
df.to_csv('responses.csv', index=False)

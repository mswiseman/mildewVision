import re
import os

model_type = ("ResNet") #"DenseNet", "VGG", "Inception3")
model_path = ("/c/Users/Intel\ User/Desktop/blackbird_scripts") # Path to the model
dataset_path = ("/e/Stacked/Mapping_Population_2023") # Path to the dataset
epoch = ("43")
threshold = ("0.7")
up_threshold = ("0.8")
down_threshold = ("0.3")
outdim = ("2")
means = ("0.5489 0.6434 0.4223")
stds = ("0.1784 0.1688 0.2417")
timestamp = ("Jan29_16-28-38_2024")

"""
This function generates the shell script commands for the a given base folder.

To run the function, you need to provide the base folder path and customize any variables below. The script will 
return a list of bash script commands for which you can save the output to a shell file (eg python generate_bash_scripts.py > leaf_correlation_all.sh).

"""

def generate_script_commands(base_folder):
    commands = []

    for root, dirs, files in os.walk(base_folder):
        # Check if there are PNG files in the current directory
        if any(file.endswith('.png') for file in files):
            #print("Found PNG files in", root)
            # Split the path to get each component
            path_parts = re.split(r'[\\/]', root)

            # Debugging print
            #print("Path parts:", path_parts)

            image_folder = None
            tray = None
            dpi = None

            # Check if the path has at least 2 parts
            if len(path_parts) >= 2:
                # Get the image folder and tray from the path
                img_folder = path_parts[-2]
                tray = path_parts[-1]

                # Debugging print
                #print("Image Folder:", img_folder, "Tray:", tray)

                # Extract dpi from the folder name (assuming it ends with '_<dpi>dpi')

                if 'dpi' in img_folder:
                    #print("True")
                    dpi = ''.join(re.findall(r'\d+', img_folder.split('_')[-1]))
                    #print("DPI:", dpi)
                    # Construct the command
                    cmd = (
                        "time python ../leaf_correlation_mw.py            \\\n"
                        f"                --model_type {model_type}              \\\n"
                        f"                --model_path {model_path}  \\\n"
                        f"                --dataset_path {dataset_path}  \\\n"
                        f"                --loading_epoch {epoch}               \\\n"
                        f"                --threshold {threshold}                  \\\n"
                        f"                --up_threshold {up_threshold}               \\\n"
                        f"                --down_threshold {down_threshold}             \\\n"
                        f"                --cuda                           \\\n"
                        f"                --cuda_id    0                   \\\n"
                        f"                --outdim {outdim}                       \\\n"
                        f"                --means {means}     \\\n"
                        f"                --stds {stds}      \\\n"
                        f"                --timestamp {timestamp}  \\\n"
                        f"                --contam_control                 \\\n"
                        f"                --dpi {dpi}                          \\\n"
                        "                --pretrained                     \\\n"
                        f"                --img_folder {img_folder}      \\\n"
                        f"                --trays {tray}                    \n"
                    )
                    commands.append(cmd)
                    #print("Command generated:", cmd)

    return commands

# Replace with your actual base directory path
commands = generate_script_commands(r"E:/Stacked/Mapping_Population_2023/")

# Print out the commands
for command in commands:
    if " --dpi 0" not in command:
        print(command)

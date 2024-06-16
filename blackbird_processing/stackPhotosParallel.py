import os
import subprocess
import tempfile
import shutil
import concurrent.futures


'''To use: 

python stackPhotosParallel.py <input_dir> <output_dir_base> <helicon_focus_path> <nconvert_path> <backup_dir_base> <max_workers>

The defaults are:
input_dir = "D:/Unstacked/"
output_dir_base = "D:/Stacked/"
helicon_focus_path = "C:/Program Files/Helicon Software/Helicon Focus 8/HeliconFocus.exe"
nconvert_path = "C:/XnView/nconvert.exe"
backup_dir_base = "D:/Backup_Permanent"
max_workers = 4 # alter based on CPU core availability

Written by Michele Wiseman of Oregon State University
October 23rd, 2023
Version 1.0
'''

input_dir = "D:/Unstacked/"
output_dir_base = "D:/Stacked/"
helicon_focus_path = "C:/Program Files/Helicon Software/Helicon Focus 8/HeliconFocus.exe"
nconvert_path = "C:/XnView/nconvert.exe"
backup_dir_base = "D:/Backup_Permanent"
max_workers = 4

def process_stack_group(input_dir, output_dir_base, helicon_focus_path, nconvert_path, backup_dir_base, experiment, date, tray, stack_group):
    ''' The function to process each stack group '''
    tray_path = os.path.join(input_dir, experiment, date, tray)
    stack_group_path = os.path.join(tray_path, stack_group)

    # Create output directory
    output_dir = os.path.join(output_dir_base, experiment, date, tray)
    os.makedirs(output_dir, exist_ok=True)

    # Create backup directory
    backup_dir = os.path.join(backup_dir_base, experiment, date, tray)
    os.makedirs(backup_dir, exist_ok=True)

    # Prepare list of source files for Helicon Focus
    source_files = [os.path.join(stack_group_path, file) for file in os.listdir(stack_group_path) if
                    file.endswith('.nef')]

    # Run Helicon Focus
    with tempfile.NamedTemporaryFile(delete=False, suffix='.lst') as f:
        temp_file_path = f.name
        f.write('\n'.join(source_files).encode())

    helicon_output_file_path = os.path.join(output_dir, f'{stack_group}.tiff')
    subprocess.run([
        helicon_focus_path, '-silent', '-i', temp_file_path, '-save:' + helicon_output_file_path,
        '-mp:1', '-rp:4', '-sp:2', '-tif:u'
    ])

    # Convert TIFF to PNG using NConvert
    png_output_file_path = os.path.join(output_dir, f'{stack_group.rsplit("_", 1)[0]}.png')
    subprocess.run([
        nconvert_path, '-out', 'png', '-o', png_output_file_path, helicon_output_file_path
    ])

    # Remove the TIFF file
    os.remove(helicon_output_file_path)

    # Move unstacked files
    shutil.move(stack_group_path, backup_dir)

    return f'Stack group {stack_group} done.'

def process_params(params):
    return process_stack_group(*params)

def main():
    input_dir = "D:/Unstacked/"
    output_dir_base = "D:/Stacked/"
    helicon_focus_path = "C:/Program Files/Helicon Software/Helicon Focus 8/HeliconFocus.exe"
    nconvert_path = "C:/XnView/nconvert.exe"
    backup_dir_base = "D:/Backup_Permanent"

    # Using concurrent.futures to parallelize the stacking jobs
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Collecting all tasks to run in parallel
        tasks = [(
                 input_dir, output_dir_base, helicon_focus_path, nconvert_path, backup_dir_base, experiment, date, tray,
                 stack_group)
                 for experiment in os.listdir(input_dir)
                 for date in os.listdir(os.path.join(input_dir, experiment))
                 for tray in os.listdir(os.path.join(input_dir, experiment, date))
                 for stack_group in os.listdir(os.path.join(input_dir, experiment, date, tray))]

        # Running tasks in parallel
        for result in executor.map(process_params, tasks):
            print(result)


if __name__ == '__main__':
    main()

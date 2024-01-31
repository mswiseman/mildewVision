This directory contains helper scripts

1. [**rename.py**](https://github.com/mswiseman/mildewVision/blob/main/common/rename.py) <br>
You can use this scrpt to change the name of a directory of files to [index_number]_[suffix]. This can be helpful when labeling additional training data.

2. [**rename_by_csv.py**](https://github.com/mswiseman/mildewVision/blob/main/common/rename_by_csv.py) <br>
Rename a list of files using a csv.<br>

3. [**remove_zeros.sh**](https://github.com/mswiseman/mildewVision/blob/main/common/remove_zeros.sh) <br>
This script removes the leading zeros in a file name e.g. 001-image.png to 1-image.png. Be sure to run test before running on entire directories.

4. [**generate_bash_scripts.py**](https://github.com/mswiseman/mildewVision/blob/main/common/generate_bash_scripts.py) <br>
This script goes through all your subdirectories and populates the required information (e.g. image date, dpi, tray #s, etc.) into your bash scripts. 

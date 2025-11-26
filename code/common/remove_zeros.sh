#!/bin/bash

# This script removes leading zeros in file names e.g. 001-image.png to 1-image.png

# For each file in the current directory
for file in *; do
    # Use parameter expansion to remove leading zeros
    new_file="${file##+(0)}"

    # If the new filename is different from the original, rename the file
    if [[ "$file" != "$new_file" ]]; then
        mv "$file" "$new_file"
    fi
done

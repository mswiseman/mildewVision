import csv
import os

# first row is the old name
# second row is the new name
# list and script must be in the same directory

# Open the CSV file and read the rows
with open('rename_list.csv', 'r') as file:
    reader = csv.reader(file)
    rename_list = list(reader)

# Loop through the rows in the CSV
for row in rename_list:
    old_name = row[0]
    new_name = row[1]
    
    # Rename the file
    os.rename(old_name, new_name)

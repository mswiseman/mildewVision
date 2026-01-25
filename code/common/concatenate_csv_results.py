import os
import pandas as pd

def find_csv_files(root_path):
    """Recursively find all CSV files in the given root path."""
    csv_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

def remove_column_from_csv(file_path, column_name):
    """Remove the specified column from the CSV file."""
    df = pd.read_csv(file_path)
    df.drop(column_name, axis=1, inplace=True, errors='ignore')
    df.dropna(how='all', inplace=True)  # Remove rows where all elements are NaN
    return df

def concatenate_csv_files(files):
    """Concatenate a list of dataframes into one."""
    return pd.concat(files, ignore_index=True)

def main():
    root_paths = ["C:/Users/Intel User/Desktop/blackbird_scripts/data/Results"]  # Update these paths as needed
    column_to_remove = '0'  # Column to remove
    output_file = 'C:/Users/Intel User/Desktop/blackbird_scripts/data/Results/concatenated_results.csv'  # Output file name

    all_csv_files = []
    for path in root_paths:
        csv_files = find_csv_files(path)
        all_csv_files.extend(csv_files)

    modified_dfs = [remove_column_from_csv(file, column_to_remove) for file in all_csv_files]
    concatenated_df = concatenate_csv_files(modified_dfs)
    concatenated_df.to_csv(output_file, index=False)

    print(f"Concatenated file created: {output_file}")

if __name__ == "__main__":
    main()

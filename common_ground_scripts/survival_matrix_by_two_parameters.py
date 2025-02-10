#!/usr/bin/env python3
"""
File: survival_matrix_by_two_parameters.py

Summary:
    Reads simulation output JSON files from a specified input directory,
    groups them by unique simulation parameter sets (excluding the two parameters
    specified by --row_key and --col_key), and for each group creates a CSV matrix.
    In each CSV:
      - Rows correspond to unique values of the parameter specified by --row_key.
      - Columns correspond to unique values of the parameter specified by --col_key.
      - Each cell contains the mean proportion of surviving information (computed as
        count_surviving_info(data["final_moving_avg"], threshold, fraction) divided by m)
        averaged over all repetitions for that (row, column) combination.
    The CSV files are formatted for easy import into R.
    
Usage Example:
    python survival_matrix_by_two_parameters.py simulation_results --output_dir csv_outputs --row_key gamma --col_key eps --threshold 0.5 --fraction 0.1
"""

import os
import json
import argparse
import numpy as np
import csv
from collections import defaultdict
from surviving_information import count_surviving_info
from tqdm import tqdm

def load_json_files(input_dir):
    """
    Load all JSON files from the given directory.
    
    Returns:
        A list of dictionaries, one per JSON file.
    """
    results = []
    for fname in tqdm(os.listdir(input_dir), desc="Loading JSON files"):
        if fname.endswith(".json"):
            path = os.path.join(input_dir, fname)
            with open(path, "r") as f:
                data = json.load(f)
                results.append(data)
    return results

def group_data(json_data, row_key, col_key):
    """
    Group simulation results by unique simulation parameter sets, excluding the two examined parameters.
    
    Args:
        json_data (list): List of simulation result dictionaries.
        row_key (str): Parameter key for rows.
        col_key (str): Parameter key for columns.
    
    Returns:
        dict: A dictionary where keys are grouping keys (tuples of sorted key-value pairs, excluding row_key and col_key)
              and values are lists of simulation results.
    """
    groups = {}
    for data in json_data:
        params = data.get("params", {})
        group_key = tuple(sorted((k, v) for k, v in params.items() if k not in {row_key, col_key}))
        groups.setdefault(group_key, []).append(data)
    return groups

def create_matrix_for_group(data_list, row_key, col_key, threshold, fraction):
    """
    Create a matrix of mean surviving proportions for a group of simulation results.
    
    For each JSON file in the group, the surviving proportion is computed as:
    
        proportion = count_surviving_info(data["final_moving_avg"], threshold, fraction) / m
    
    where m is the length of an agent’s state vector.
    Then, for each unique combination of row (row_key) and column (col_key) values,
    the function computes the mean proportion.
    
    Args:
        data_list (list): List of simulation result dictionaries in the same group.
        row_key (str): Parameter key for rows.
        col_key (str): Parameter key for columns.
        threshold (float): Survival threshold.
        fraction (float): Minimum fraction.
    
    Returns:
        tuple: (matrix, sorted_row_values, sorted_col_values)
               matrix is a dictionary of dictionaries where matrix[row][col] is the mean proportion.
    """
    cell_values = defaultdict(list)
    row_set = set()
    col_set = set()
    for data in data_list:
        params = data.get("params", {})
        row_val = params.get(row_key)
        col_val = params.get(col_key)
        if row_val is None or col_val is None:
            continue
        row_set.add(row_val)
        col_set.add(col_val)
        final_moving_avg = data.get("final_moving_avg", {})
        if not final_moving_avg:
            continue
        survived = count_surviving_info(final_moving_avg, threshold, fraction)
        m = len(next(iter(final_moving_avg.values())))
        proportion = survived / m
        cell_values[(row_val, col_val)].append(proportion)
    
    matrix = {}
    for row_val in sorted(row_set):
        row_dict = {}
        for col_val in sorted(col_set):
            values = cell_values.get((row_val, col_val), [])
            row_dict[col_val] = np.mean(values) if values else np.nan
        matrix[row_val] = row_dict
    return matrix, sorted(row_set), sorted(col_set)

def write_matrix_to_csv(matrix, row_vals, col_vals, filename, output_dir, row_key, col_key):
    """
    Write the matrix to a CSV file with clearly labeled rows and columns.
    
    The CSV file is formatted as follows:
      - The first row is a header with the first cell containing the row_key,
        followed by each column labeled as "col_key=value".
      - Each subsequent row begins with "row_key=value" followed by the computed mean proportions.
    
    Args:
        matrix (dict): Nested dictionary of mean surviving proportions.
        row_vals (list): Sorted list of unique row values.
        col_vals (list): Sorted list of unique column values.
        filename (str): Name of the CSV file.
        output_dir (str): Directory to save the CSV file.
        row_key (str): The row parameter key.
        col_key (str): The column parameter key.
    """
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header row: first cell is the row_key, then columns labeled as "col_key=value"
        header = [f"{row_key}"] + [f"{col_key}={col}" for col in col_vals]
        writer.writerow(header)
        # Write each row: first column is "row_key=value", then the computed cell values.
        for row in row_vals:
            row_data = [f"{row_key}={row}"]
            for col in col_vals:
                value = matrix[row].get(col, "")
                row_data.append(str(value))
            writer.writerow(row_data)
    print(f"CSV saved to {filepath}")

def create_filename_from_group_key(group_key):
    """
    Create a filename based on the grouping key.
    
    Args:
        group_key (tuple): Tuple of (parameter, value) pairs.
    
    Returns:
        str: Filename in the format "param1_val1_param2_val2.csv".
    """
    parts = []
    for k, v in group_key:
        parts.append(f"{k}_{v}")
    return "_".join(parts) + ".csv"

def main():
    parser = argparse.ArgumentParser(
        description="Generate CSV matrices of mean surviving information by two parameters."
    )
    parser.add_argument("input_dir", type=str, help="Directory containing simulation JSON files.")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save CSV files (default: current directory).")
    parser.add_argument("--row_key", type=str, default="gamma", help="Parameter key for rows (default: 'gamma').")
    parser.add_argument("--col_key", type=str, default="eps", help="Parameter key for columns (default: 'eps').")
    parser.add_argument("--threshold", type=float, default=0.5, help="Survival threshold (default: 0.5).")
    parser.add_argument("--fraction", type=float, default=0.1, help="Minimum fraction (default: 0.1).")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    json_data = load_json_files(args.input_dir)
    groups = group_data(json_data, args.row_key, args.col_key)
    
    for group_key, data_list in groups.items():
        matrix, row_vals, col_vals = create_matrix_for_group(data_list, args.row_key, args.col_key, args.threshold, args.fraction)
        filename = create_filename_from_group_key(group_key)
        write_matrix_to_csv(matrix, row_vals, col_vals, filename, args.output_dir, args.row_key, args.col_key)

if __name__ == "__main__":
    main()
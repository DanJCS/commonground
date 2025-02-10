#!/usr/bin/env python3
"""
File: global_agreement_timeseries.py

Summary:
    This script computes and plots the global convergence (or "agreement") of agents' state vectors over time.
    Global agreement is defined as the average of all pairwise similarities among agents' state vectors,
    where the similarity between two state vectors is computed using the normalized Euclidean distance:
    
        similarity = 1 - (||u - v|| / (2 * m))
    
    The script processes simulation result JSON files from an input directory. Each JSON file must contain
    the "records" key, which holds the recorded state vectors (typically saved every 200 timesteps, with t=0 excluded).
    The results are then grouped by the "m" value. For each recorded timestep, the script computes the mean global 
    agreement along with the 2.5th and 97.5th percentiles across repetitions.
    Finally, a time-series plot is generated—with a separate colored line for each m value—and saved as a PNG file.

Usage:
    python3 global_agreement_timeseries.py <input_dir> --output global_agreement.png --metric euclidean_direct_normalised --time_interval 200
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
from tqdm import tqdm  # for progress bars

def compute_global_agreement(state_vectors, m):
    """
    Compute the global agreement for a given set of state vectors.
    
    Global agreement is defined as the mean of all pairwise similarities among agents,
    where similarity is calculated using the normalized Euclidean distance:
    
        similarity = 1 - (||u - v|| / (2 * m))
    
    Args:
        state_vectors (dict): Dictionary with keys as agent names and values as state vectors (list or np.array).
        m (int): Length of the state vectors.
    
    Returns:
        float: The mean global agreement.
    """
    agents = list(state_vectors.keys())
    n = len(agents)
    similarities = []
    for i in range(n):
        for j in range(i + 1, n):
            vec_i = np.array(state_vectors[agents[i]])
            vec_j = np.array(state_vectors[agents[j]])
            d = np.linalg.norm(vec_i - vec_j)
            sim = 1 - (d / (2 * m))
            similarities.append(sim)
    return np.mean(similarities)

def load_json_files(input_dir):
    """
    Load all JSON simulation result files from the specified input directory.
    
    Args:
        input_dir (str): Directory containing simulation result JSON files.
    
    Returns:
        list: A list of dictionaries loaded from the JSON files.
    
    Raises:
        ValueError: If any file does not contain the "records" key.
    """
    files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    data_list = []
    for filename in tqdm(files, desc="Loading JSON files"):
        path = os.path.join(input_dir, filename)
        with open(path, 'r') as f:
            data = json.load(f)
            if "records" not in data:
                raise ValueError(f"File {filename} does not contain 'records'.")
            data_list.append(data)
    return data_list

def group_by_m(data_list):
    """
    Group simulation results by the value of m (state vector length).
    
    Args:
        data_list (list): List of simulation result dictionaries.
    
    Returns:
        dict: A dictionary with keys as m (int) and values as lists of simulation results having that m.
    """
    grouped = defaultdict(list)
    for data in data_list:
        m_val = data["params"].get("m", None)
        if m_val is not None:
            grouped[m_val].append(data)
    return grouped

def aggregate_time_series(group, time_interval):
    """
    Aggregate global agreement time-series data for a group of simulation repetitions.
    
    For each recorded timestep, compute:
      - The average global agreement.
      - The 2.5th percentile.
      - The 97.5th percentile.
    
    Args:
        group (list): List of simulation result dictionaries (all having the same m).
        time_interval (int): The interval (in timesteps) at which records are saved.
    
    Returns:
        dict: Mapping each timestep (int) to a tuple (mean, lower, upper).
    """
    # Assume each repetition has the same number of recorded timesteps.
    sample = group[0]["records"]
    num_records = len(next(iter(sample.values())))
    
    aggregated = {}
    for idx in range(num_records):
        agreement_values = []
        for data in group:
            records = data["records"]
            # Build a dictionary of state vectors for the current recorded timestep.
            state_vectors = {agent: records[agent][idx] for agent in records}
            m = len(next(iter(state_vectors.values())))
            agreement = compute_global_agreement(state_vectors, m)
            agreement_values.append(agreement)
        timestep = (idx + 1) * time_interval
        mean_val = np.mean(agreement_values)
        lower = np.percentile(agreement_values, 2.5)
        upper = np.percentile(agreement_values, 97.5)
        aggregated[timestep] = (mean_val, lower, upper)
    return aggregated

def plot_time_series(aggregated_data, output_file):
    """
    Plot the global agreement time-series for each m value.
    
    Each m group is plotted as a separate line (with a distinct color) with the mean
    global agreement over time and a shaded region between the 2.5th and 97.5th percentiles.
    
    Args:
        aggregated_data (dict): Dictionary where keys are m values and values are time-series data.
        output_file (str): Output filename for saving the plot (PNG format).
    """
    plt.figure(figsize=(10, 6))
    # Generate a color cycle based on the number of distinct m values.
    m_values = sorted(aggregated_data.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(m_values)))
    
    for m_val, color in zip(m_values, colors):
        ts_data = aggregated_data[m_val]
        timesteps = sorted(ts_data.keys())
        means = [ts_data[t][0] for t in timesteps]
        lowers = [ts_data[t][1] for t in timesteps]
        uppers = [ts_data[t][2] for t in timesteps]
        plt.plot(timesteps, means, marker='o', color=color, label=f"m={m_val}")
        plt.fill_between(timesteps, lowers, uppers, color=color, alpha=0.2)
    
    plt.xlabel("Timestep")
    plt.ylabel("Global Agreement")
    plt.title("Global Agreement Time-Series")
    plt.ylim(0, 1)
    plt.legend(title="State Vector Length (m)")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()

def main():
    """
    Main function to generate a global agreement time-series plot.
    
    This function:
      - Parses command-line arguments.
      - Loads simulation JSON files from the input directory.
      - Groups the simulation results by the value of m.
      - Aggregates global agreement statistics (mean, 2.5th, and 97.5th percentiles) for each recorded timestep.
      - Plots the time-series data with a separate line for each m value and shaded percentile intervals.
      - Saves the plot to the specified output file.
    
    Command-Line Arguments:
      input_dir (str): Directory containing simulation result JSON files.
      --output (str): Output filename for the plot (default: "global_agreement.png").
      --metric (str): Similarity metric to use (default: "euclidean_direct_normalised").
                      (Note: Only "euclidean_direct_normalised" is supported in this implementation.)
      --time_interval (int): The interval (in timesteps) at which state vectors are recorded (default: 200).
    """
    parser = argparse.ArgumentParser(description="Plot global agreement time-series from simulation JSON files.")
    parser.add_argument("input_dir", type=str, help="Directory containing simulation result JSON files.")
    parser.add_argument("--output", type=str, default="global_agreement.png", help="Output plot filename.")
    parser.add_argument("--metric", type=str, default="euclidean_direct_normalised", help="Similarity metric to use (default: euclidean_direct_normalised)")
    parser.add_argument("--time_interval", type=int, default=200, help="Interval at which records are saved (default: 200)")
    args = parser.parse_args()
    
    # Load JSON simulation results with a progress bar.
    data_list = load_json_files(args.input_dir)
    
    # Group the results by m value.
    grouped_by_m = group_by_m(data_list)
    
    aggregated_data = {}
    # Iterate over each m group with a progress bar.
    for m_val, group in tqdm(grouped_by_m.items(), desc="Aggregating groups by m"):
        aggregated_ts = aggregate_time_series(group, args.time_interval)
        aggregated_data[m_val] = aggregated_ts
        
    # Plot the aggregated global agreement time-series.
    plot_time_series(aggregated_data, args.output)
    print(f"Plot saved to {args.output}")

if __name__ == "__main__":
    main()
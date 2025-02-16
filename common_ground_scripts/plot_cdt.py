#!/usr/bin/env python3
"""
File: plot_cdt.py

Summary:
    Generates plots demonstrating the convergence of agent state vectors
    towards a centroid over time.  This script calculates the "Centroid
    Dissimilarity Trend (CDT)" – essentially, the average dissimilarity
    between each agent's state vector and the centroid (mean state vector)
    of all agents at specific time intervals.  The script processes a
    directory containing simulation output files in JSON format, groups
    the results by simulation parameters (including 'm'), and produces a
    separate plot for each unique parameter combination.  Each plot shows
    the CDT over time, averaged across multiple repetitions of the same
    parameter settings.

Dependencies:
    * Python built-ins: os, json, argparse
    * Third-party: numpy, matplotlib

Usage:
    python plot_cdt.py [<input_dir>] --interval 200 --output_dir <output_dir>

    Where:
      <input_dir> is the directory containing the .json simulation output files.
                  (Default: "simulation_results" if not provided.)
      --interval specifies the timestep interval for plotting (default: 200).
      --output_dir is directory where plots will be stored

Example Usage:
    python plot_cdt.py simulation_results --interval 200 --output_dir convergence_plots
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm


def calculate_centroid(agent_states):
    """Calculates the centroid (mean state vector) of a group of agents.

    Args:
        agent_states (dict): A dictionary where keys are agent names (strings)
                             and values are lists representing the agent's
                             state vector at a specific timestep.

    Returns:
        np.ndarray: The centroid (mean state vector) as a NumPy array.
    """
    state_vectors = np.array(list(agent_states.values()))
    return np.mean(state_vectors, axis=0)


def calculate_dissimilarity(state_vector, centroid):
    """Calculates the normalized Euclidean distance between a state vector and the centroid.

    Args:
        state_vector (np.ndarray): The agent's state vector.
        centroid (np.ndarray): The centroid (mean state vector).

    Returns:
        float: The normalized Euclidean distance.
    """
    m = len(state_vector)
    return np.linalg.norm(state_vector - centroid) / (2 * np.sqrt(m))


def process_simulation_file(filepath, interval):
    """Processes a single simulation JSON file.

    Args:
        filepath (str): Path to the JSON file.
        interval (int): Timestep interval for calculating CDT.

    Returns:
        tuple: (params, cdt_data).
               params is a dictionary of simulation parameters.
               cdt_data is a dictionary: {timestep: mean_dissimilarity}.
               Returns (None, None) if the file is invalid.
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Warning: Could not read or parse file: {filepath}")
        return None, None

    if "records" not in data or "params" not in data:
        print(f"Warning: Missing 'records' or 'params' in file: {filepath}")
        return None, None

    records = data["records"]
    params = data["params"]

    # Determine if the records are in "raw" format.
    # NOTE: This assumes that raw records are lists of lists.
    test_agent = next(iter(records))
    if isinstance(records[test_agent][0], list):
        record_type = "raw"
    else:
        print(f"Warning: Invalid record format in {filepath}")
        return None, None

    # Calculate maximum time based on the length of records for the first agent.
    max_time = len(next(iter(records.values()))) * interval

    cdt_data = {}
    for t in range(0, max_time, interval):
        timestep_index = t // interval
        if timestep_index >= len(next(iter(records.values()))):
            continue

        agent_states = {}
        for agent_name, state_history in records.items():
            if timestep_index < len(state_history):
                agent_states[agent_name] = state_history[timestep_index]
        if not agent_states:
            continue

        centroid = calculate_centroid(agent_states)
        total_dissimilarity = 0
        for agent_state in agent_states.values():
            total_dissimilarity += calculate_dissimilarity(np.array(agent_state), centroid)
        mean_dissimilarity = total_dissimilarity / len(agent_states)
        cdt_data[t] = mean_dissimilarity

    return params, cdt_data


def create_param_key(params, exclude_keys={"rep"}):
    """Creates a hashable key from simulation parameters, excluding specified keys.
    """
    filtered_params = {k: v for k, v in params.items() if k not in exclude_keys}
    return tuple(sorted(filtered_params.items()))


def plot_cdt(data_by_params, output_dir):
    """Generates and saves CDT plots.

    Args:
        data_by_params (dict): Aggregated CDT data, grouped by parameter keys.
        output_dir (str): Directory to save the plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    for param_key, grouped_data in data_by_params.items():
        param_dict = dict(param_key)
        param_str = ", ".join(f"{k}={v}" for k, v in param_dict.items())

        all_times = []
        all_means = []
        for rep_data in grouped_data:
            times = sorted(rep_data.keys())
            means = [rep_data[t] for t in times]
            all_times.append(times)
            all_means.append(means)

        # Potential Bug: If all_times is empty, max() will raise a ValueError.
        if not all_times or not any(all_times):
            print(f"Warning: No valid time points for parameters {param_str}")
            continue

        max_time = max(max(times) for times in all_times if times)
        plt.figure(figsize=(10, 6))
        for times, means in zip(all_times, all_means):
            plt.plot(times, means, marker='', linestyle='-', alpha=0.3)

        # Calculate average across repetitions
        unique_times = sorted(set(time for times in all_times for time in times))
        avg_means = []
        for t in unique_times:
            values_at_t = [rep_data[t] for rep_data in grouped_data if t in rep_data]
            avg_means.append(np.mean(values_at_t) if values_at_t else np.nan)

        plt.plot(unique_times, avg_means, marker='o', linestyle='-', color='black', label='Average')
        plt.xlabel("Timestep")
        plt.ylabel("Mean Centroid Distance (MCD)")
        plt.title(f"Convergence Demonstration\nParameters: {param_str}")
        plt.grid(True)
        plt.xlim(0, max_time)
        plt.legend()
        plt.tight_layout()

        param_filename = "_".join(f"{k}{v}" for k, v in param_dict.items())
        output_filename = os.path.join(output_dir, f"cdt_{param_filename}.png")
        plt.savefig(output_filename, dpi=150)
        plt.close()
        print(f"Saved plot to {output_filename}")


def main():
    parser = argparse.ArgumentParser(description="Generate Centroid Dissimilarity Trend plots.")
    # Make input_dir optional and assign a default value if not provided.
    parser.add_argument("input_dir", nargs="?", default="/Users/danieljung/Desktop/ABM CG/Natural results/Results/full_offline",
                        help="Directory containing simulation .json files (default: simulation_results).")
    parser.add_argument("--interval", type=int, default=200, help="Timestep interval for plotting.")
    parser.add_argument("--output_dir", type=str, default="convergence_plots",
                        help="Directory to save the generated plots.")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found.")
        return

    data_by_params = defaultdict(list)
    for filename in tqdm(os.listdir(args.input_dir), desc="Processing files"):
        if filename.endswith(".json"):
            filepath = os.path.join(args.input_dir, filename)
            params, cdt_data = process_simulation_file(filepath, args.interval)
            if params and cdt_data:
                param_key = create_param_key(params)
                data_by_params[param_key].append(cdt_data)

    plot_cdt(data_by_params, args.output_dir)


if __name__ == "__main__":
    main()
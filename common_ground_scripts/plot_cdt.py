#!/usr/bin/env python3
"""
File: plot_cdt.py

Summary:
    Generates plots demonstrating the convergence of agent state vectors
    towards a centroid over time.  This script calculates the "Centroid
    Dissimilarity Trend (CDT)" – essentially, the average dissimilarity
    between each agent's state vector and the centroid (mean state vector)
    of all agents at specific time intervals. It also calculates and plots
    the "Relative Centroid Change (RCC)" to show the rate of change of the
    centroid position. This version supports variable record increments and
    includes enhanced plotting features.

Usage:
    python plot_cdt.py [<input_dir>] --record_interval 200 --output_dir <output_dir>

    Where:
      <input_dir> is the directory containing the .json simulation output files.
      --record_interval specifies the timestep interval used when recording state vectors.
      --output_dir is the directory where plots will be stored.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm


def calculate_centroid(agent_states):
    state_vectors = np.array(list(agent_states.values()))
    return np.mean(state_vectors, axis=0)


def calculate_dissimilarity(state_vector, centroid):
    m = len(state_vector)
    return np.linalg.norm(state_vector - centroid) / (2 * np.sqrt(m))

def calculate_relative_centroid_change(centroid_current, centroid_previous):
    """
    Calculates the relative change in centroid position.  We use the
    *absolute* difference to avoid cancellation of changes in opposite
    directions.  This is normalized by the state vector length (m).

    Args:
        centroid_current (np.ndarray): Centroid at the current timestep.
        centroid_previous (np.ndarray): Centroid at the previous timestep.

    Returns:
        float: The normalized relative change in centroid position.  A smaller
               value indicates that the centroid is changing less, suggesting
               greater stability.
    """
    m = len(centroid_current)
    # Use absolute difference to get the magnitude of change.
    absolute_difference = np.sum(np.abs(centroid_previous - centroid_current))
    return absolute_difference / m

def process_simulation_file(filepath, record_interval):
    """
    Processes a single simulation JSON file, calculating both CDT and RCC.

    Returns:
        tuple: (params, cdt_data, rcc_data).
               params is a dictionary of simulation parameters.
               cdt_data is a dictionary: {timestep: mean_dissimilarity}.
               rcc_data is a dictionary: {timestep: relative_centroid_change}.
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Warning: Could not read or parse file: {filepath}")
        return None, None, None

    if "records" not in data or "params" not in data:
        print(f"Warning: Missing 'records' or 'params' in file: {filepath}")
        return None, None, None

    records = data["records"]
    params = data["params"]

    # Use the number of recorded samples and the record_interval to compute actual time.
    num_samples = len(next(iter(records.values())))

    cdt_data = {}
    rcc_data = {}
    centroid_previous = None  # Store the previous centroid

    for i in range(num_samples):
        t_actual = i * record_interval
        agent_states = {}
        for agent_name, state_history in records.items():
            if i < len(state_history):
                agent_states[agent_name] = state_history[i]
        if not agent_states:
            continue

        centroid_current = calculate_centroid(agent_states)

        total_dissimilarity = 0
        for agent_state in agent_states.values():
            total_dissimilarity += calculate_dissimilarity(np.array(agent_state), centroid_current)

        mean_dissimilarity = total_dissimilarity / len(agent_states)
        cdt_data[t_actual] = mean_dissimilarity

        # Calculate Relative Centroid Change (RCC)
        if centroid_previous is not None:
            rcc = calculate_relative_centroid_change(centroid_current, centroid_previous)
            rcc_data[t_actual] = rcc

        centroid_previous = centroid_current  # Update for the next iteration

    return params, cdt_data, rcc_data


def plot_cdt_and_rcc(data_by_params, output_dir):
    """
    Generates and saves CDT and RCC plots, grouped by 'm' value.

    Args:
        data_by_params (dict): Aggregated CDT and RCC data.
        output_dir (str): Directory to save the plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    for param_key, (cdt_list, rcc_list) in data_by_params.items():

        # Extract the 'm' value from the parameter set
        m_value = None
        for k, v in param_key:
            if k == 'm':
                m_value = v
                break
        if m_value is None:
            print("Warning: 'm' value not found in parameter set. Skipping plot.")
            continue


        #Average all cdt repetitions
        times_cdt = sorted(cdt_list.keys())
        values_cdt = [cdt_list[t] for t in times_cdt]

        # Average all rcc repetitions
        times_rcc = sorted(rcc_list.keys())
        values_rcc = [rcc_list[t] for t in times_rcc]


        plt.figure(figsize=(10, 6))

        # Plot CDT
        plt.plot(times_cdt, values_cdt, linestyle='--', color="r",marker=".", label="Mean Centroid Distance",alpha=0.5)

        # Plot RCC
        plt.plot(times_rcc, values_rcc, linestyle='--', color="b",marker=".", label="Relative Centroid Change", alpha=0.5)

        # Find and mark the convergence point (based on RMCD)
        epsilon = 0.005
        convergence_time = None
        for t, rcc_val in zip(times_rcc, values_rcc):
            if rcc_val < epsilon:
                convergence_time = t
                break

        if convergence_time is not None:

            plt.plot(
                convergence_time,
                rcc_list[convergence_time],
                marker="o",
                markersize=3,
                color="purple",
                alpha=0.5,
                label=f"Convergence at: t={convergence_time} ",
            )  # Mark the intersection

        plt.xlabel("Timestep")
        plt.ylabel("Value")  # More general y-axis label
        plt.title(f"Convergence Time-series (m={m_value})")  # Simplified title
        plt.grid(True)
        plt.ylim(-0.025, 0.255)  # Fixed y-axis limits
        plt.legend()
        plt.tight_layout()

        filename = f"Convergence_m{m_value}.png"
        outfile = os.path.join(output_dir, filename)
        plt.savefig(outfile, dpi=150)
        plt.close()
        print(f"Saved plot to {outfile}")



def main():
    parser = argparse.ArgumentParser(description="Generate CDT and RCC plots from simulation JSON files.")
    parser.add_argument("input_dir", nargs="?", default="simulation_results",
                        help="Directory containing simulation JSON files (default: simulation_results).")
    parser.add_argument("--record_interval", type=int, default=100,
                        help="Timestep interval used for recording state vectors (default: 10).")
    parser.add_argument("--output_dir", type=str, default="convergence_plots",
                        help="Directory to save the generated plots.")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    data_by_params = defaultdict(lambda: [defaultdict(float), defaultdict(float)])


    for filename in tqdm(os.listdir(args.input_dir), desc="Processing files"):
        if filename.endswith(".json"):
            filepath = os.path.join(args.input_dir, filename)
            params, cdt_data, rcc_data = process_simulation_file(filepath, args.record_interval)
            if params is None or cdt_data is None or rcc_data is None:
                continue

            param_key = tuple(sorted(params.items()))

            # Aggregate CDT data
            if param_key in data_by_params:
                existing_cdt = data_by_params[param_key][0]
                for t, value in cdt_data.items():
                    if t in existing_cdt:
                        existing_cdt[t] = (existing_cdt[t] + value) / 2.0
                    else:
                        existing_cdt[t] = value
            else:
                data_by_params[param_key][0] = cdt_data

            # Aggregate RCC data
            if param_key in data_by_params:
                existing_rcc = data_by_params[param_key][1]
                for t, value in rcc_data.items():
                    if t in existing_rcc:
                        existing_rcc[t] = (existing_rcc[t] + value) / 2.0
                    else:
                        existing_rcc[t] = value
            else:
                data_by_params[param_key][1] = rcc_data


    plot_cdt_and_rcc(data_by_params, args.output_dir)
    print(f"All convergence plots saved in '{args.output_dir}'.")

if __name__ == "__main__":
    main()
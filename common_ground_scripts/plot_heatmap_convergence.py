#!/usr/bin/env python3
"""
File: plot_heatmap_convergence.py

Summary:
    Generates heatmaps visualizing the influence of feedback reliability (epsilon)
    and silence interpretation (gamma) on the convergence time of the agent-based
    model.  Convergence time is determined by the Relative Centroid Change (RCC)
    falling below a specified threshold (epsilon) and staying below it for a
    defined stabilization period. Separate heatmaps are created for each value
    of the state vector length (m).

Dependencies:
    * Python built-ins: os, json, argparse
    * Third-party: numpy, matplotlib

Usage:
    python plot_heatmap_convergence.py <input_dir> --output_dir <output_dir> --record_interval <interval>

    Where:
      <input_dir> is the directory containing the .json simulation output files.
      --output_dir is the directory where heatmap plots will be saved.
      --record_interval specifies the timestep interval used when recording state vectors.
      --epsilon value (default: 0.01)
      --stabilization_period value (default: 3)

Example:
    python plot_heatmap_convergence.py simulation_results --output_dir heatmap_plots --record_interval 10 --epsilon 0.01
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

def calculate_centroid(agent_states):
    """Calculates the centroid (mean state vector) of agent states."""
    return np.mean(list(agent_states.values()), axis=0)

def calculate_relative_centroid_change(centroid_current, centroid_previous):
    """Calculates the relative change in centroid position (normalized)."""
    m = len(centroid_current)
    return np.sum(np.abs(centroid_previous - centroid_current)) / m

def process_simulation_file(filepath, record_interval, epsilon, stabilization_period):
    """Processes a single JSON file to extract convergence time."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Warning: Could not read or parse {filepath}")
        return None, None

    records = data.get("records")
    params = data.get("params")
    if not records or not params:
        print(f"Warning: Missing 'records' or 'params' in {filepath}")
        return None, None

    m = params.get('m')
    epsilon_val = params.get('eps')
    gamma_val = params.get('gamma')
    timesteps = params.get('timesteps')


    if m is None or epsilon_val is None or gamma_val is None or timesteps is None:
        print(f"Warning: Missing m, epsilon, gamma, or timesteps in parameters of {filepath}")
        return None, None

    num_samples = len(next(iter(records.values())))
    rcc_values = []
    centroid_previous = None

    for i in range(num_samples):
        t_actual = i * record_interval
        agent_states = {
            agent: states[i] for agent, states in records.items() if i < len(states)
        }
        if not agent_states:
            continue
        centroid_current = calculate_centroid(agent_states)
        if centroid_previous is not None:
            rcc = calculate_relative_centroid_change(centroid_current, centroid_previous)
            rcc_values.append((t_actual, rcc))
        centroid_previous = centroid_current

    # Find convergence time with stabilization
    convergence_time = None
    for i, (t_actual, rcc) in enumerate(rcc_values):
      # Check for stabilization (stays below the threshold of "epsilon")
      if rcc < epsilon and all(rcc_values[j][1] < epsilon for j in range(i,min(len(rcc_values),i+stabilization_period))):
        convergence_time = t_actual
        break

    if convergence_time is None:
        convergence_time = timesteps  # Use maximum timesteps if no convergence

    return (m, epsilon_val, gamma_val), convergence_time


def create_heatmap(data_by_m, m_value, output_dir, epsilon, record_interval):
    """Creates and saves a heatmap for a specific m value."""
    if m_value not in data_by_m:
        print(f"No data found for m={m_value}")
        return

    data = data_by_m[m_value]
    # Collect unique epsilon and gamma values
    epsilon_values = sorted(set(x[0] for x in data.keys()))
    gamma_values = sorted(set(x[1] for x in data.keys()))

    # Create a 2D array for the heatmap
    heatmap_data = np.full((len(epsilon_values), len(gamma_values)), np.nan)  # Initialize with NaN

    for i, epsilon_val in enumerate(epsilon_values):
        for j, gamma_val in enumerate(gamma_values):
            convergence_times = data.get((epsilon_val, gamma_val), [])
            if convergence_times:
              heatmap_data[i, j] = np.mean(convergence_times)  # Use average
            # No need for else, since it is initialised by NaN.

    # Create the heatmap plot
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap_data, cmap="RdYlBu", origin="lower", vmin=0, vmax=5000)  # Use "viridis_r" for inverted colormap
    plt.colorbar(label="Average Convergence Time (Timesteps)")
    plt.xticks(ticks=range(len(gamma_values)), labels=[f"{val:.2f}" for val in gamma_values], rotation=45)
    plt.yticks(ticks=range(len(epsilon_values)), labels=[f"{val:.2f}" for val in epsilon_values])
    plt.xlabel("Gamma (γ)")
    plt.ylabel("Epsilon (ε)")
    plt.title(f"Convergence Time Heatmap (m={m_value}, tol={epsilon}")
    plt.tight_layout()

    filename = f"heatmap_m{m_value}_eps{epsilon}_ri{record_interval}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Heatmap saved to {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Generate heatmaps of convergence time.")
    parser.add_argument("input_dir", type=str, help="Directory containing simulation .json files.")
    parser.add_argument("--output_dir", type=str, default="heatmap_plots",
                        help="Directory to save heatmap plots (default: heatmap_plots).")
    parser.add_argument("--record_interval", type=int, default=100,
                        help="Timestep interval for recording state vectors (default: 10).")
    parser.add_argument("--epsilon", type=float, default=0.005,
                        help="Tolerance value (epsilon) for convergence (default: 0.01).")
    parser.add_argument("--stabilization_period", type=int, default=3,
                        help="Number of consecutive intervals below epsilon for convergence (default: 3).")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    data_by_m = defaultdict(lambda: defaultdict(list))

    # Load and process simulation files
    for filename in tqdm(os.listdir(args.input_dir), desc="Processing files"):
        if filename.endswith(".json"):
            filepath = os.path.join(args.input_dir, filename)
            params, convergence_time = process_simulation_file(filepath, args.record_interval, args.epsilon, args.stabilization_period)
            if params is not None and convergence_time is not None:
                m_value, epsilon_val, gamma_val = params
                data_by_m[m_value][(epsilon_val, gamma_val)].append(convergence_time)

    # Create heatmaps for each 'm'
    for m_value in data_by_m:
        create_heatmap(data_by_m, m_value, args.output_dir, args.epsilon, args.record_interval)

if __name__ == "__main__":
    main()
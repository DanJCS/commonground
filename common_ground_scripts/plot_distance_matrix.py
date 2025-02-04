#!/usr/bin/env python3
"""
File: plot_distance_matrix.py

Summary:
    Loads a simulation result JSON file, computes the distance matrix for the final_moving_avg vectors,
    and plots a heatmap of the distance matrix using Seaborn.

Usage:
    python3 plot_distance_matrix.py path/to/simulation_result.json [--method jsd] [--temperature 1.0] [--output output.png]
    
If the --output option is omitted, the plot is displayed interactively.
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import the distance matrix builder from your codebase
from pairwise_similarity import build_distance_matrix

def plot_distance_matrix(json_file, method="jsd", temperature=1.0, output_file=None):
    # Load JSON simulation result
    with open(json_file, "r") as f:
        data = json.load(f)
    
    # Extract the final moving average from the simulation result.
    final_moving_avg = data.get("final_moving_avg", None)
    if final_moving_avg is None:
        raise ValueError("The provided JSON file does not contain a 'final_moving_avg' key.")
    
    # Compute the distance matrix and get the agent order.
    dist_matrix, agent_names = build_distance_matrix(final_moving_avg, method=method, temperature=temperature)
    
    # Optionally, inspect basic statistics:
    print("Distance Matrix Stats:")
    print("  Min:", np.min(dist_matrix))
    print("  Max:", np.max(dist_matrix))
    print("  Mean:", np.mean(dist_matrix))
    
    # Create a heatmap using Seaborn.
    plt.figure(figsize=(8, 6))
    sns.heatmap(dist_matrix, 
                xticklabels=agent_names, 
                yticklabels=agent_names,
                cmap="viridis", 
                cbar_kws={"label": "Distance"},
                square=True)
    plt.title("Pairwise Distance Matrix Heatmap")
    plt.xlabel("Agents")
    plt.ylabel("Agents")
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"Heatmap saved to {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot the distance matrix from a simulation result JSON file.")
    parser.add_argument("json_file", help="Path to the simulation result JSON file.")
    parser.add_argument("--method", type=str, default="jsd", help="Distance metric to use (e.g., jsd, euclidean, cosine).")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature for converting state vectors to probability distributions.")
    parser.add_argument("--output", type=str, default=None, help="If provided, the plot will be saved to this file instead of displayed.")
    args = parser.parse_args()
    
    if not os.path.exists(args.json_file):
        raise FileNotFoundError(f"The file {args.json_file} does not exist.")
    
    plot_distance_matrix(args.json_file, method=args.method, temperature=args.temperature, output_file=args.output)

if __name__ == "__main__":
    main()
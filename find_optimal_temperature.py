#!/usr/bin/env python3
"""
find_optimal_temperature.py

This script assists in finding an ideal temperature parameter for converting
final_moving_avg vectors to Probability Distributions (PDs) using the softmax
function. It highlights differences, especially at the peaks of the vectors,
and computes pairwise similarities using Jensen-Shannon Divergence (JSD).

Features:
1. Loads vectors from a specified .json file.
2. Selects a specified number of random vectors.
3. Applies softmax with a range of temperature values to convert vectors to PDs.
4. Computes pairwise JSD between PDs for each temperature.
5. Prints the vectors, PDs, and JSD values.
6. Offers suggestions and visualizations to aid in selecting the optimal temperature.
"""

import os
import json
import math
import random
import argparse
import itertools
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

###############################################################################
# 1. Utility Functions
###############################################################################

def softmax(vector: List[float], temperature: float) -> np.ndarray:
    """
    Applies the softmax function to a vector with a specified temperature.
    
    Args:
        vector (List[float]): The input vector.
        temperature (float): The temperature parameter.
        
    Returns:
        np.ndarray: The resulting probability distribution.
    """
    scaled_vector = np.array(vector) / temperature
    # To prevent overflow, subtract the max before exponentiating
    scaled_vector = scaled_vector - np.max(scaled_vector)
    exp_vector = np.exp(scaled_vector)
    return exp_vector / np.sum(exp_vector)

def compute_jsd(p: np.ndarray, q: np.ndarray) -> float:
    """
    Computes the Jensen-Shannon Divergence between two probability distributions.
    
    Args:
        p (np.ndarray): First probability distribution.
        q (np.ndarray): Second probability distribution.
        
    Returns:
        float: The Jensen-Shannon Divergence.
    """
    return jensenshannon(p, q)**2  # Squared JSD

def select_random_vectors(vectors: Dict[str, List[float]], num_vectors: int) -> Dict[str, List[float]]:
    """
    Selects a specified number of random vectors from the provided dictionary.
    
    Args:
        vectors (Dict[str, List[float]]): Dictionary of vectors.
        num_vectors (int): Number of vectors to select.
        
    Returns:
        Dict[str, List[float]]: Selected random vectors.
    """
    selected_keys = random.sample(list(vectors.keys()), min(num_vectors, len(vectors)))
    return {key: vectors[key] for key in selected_keys}

def plot_pd_vectors(pd_vectors: Dict[str, np.ndarray], temperature: float, output_dir: str, param_set_str: str):
    """
    Plots the PD vectors for visualization.
    
    Args:
        pd_vectors (Dict[str, np.ndarray]): Dictionary of PD vectors.
        temperature (float): The temperature used.
        output_dir (str): Directory to save the plots.
        param_set_str (str): String representation of parameter set.
    """
    plt.figure(figsize=(10,6))
    for agent, pd in pd_vectors.items():
        plt.plot(pd, label=agent, alpha=0.6)
    plt.title(f'Probability Distributions at Temperature={temperature}\n{param_set_str}')
    plt.xlabel('Dimension')
    plt.ylabel('Probability')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.05), fontsize='small')
    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f'PDs_Temperature_{temperature}.png')
    plt.savefig(plot_filename, dpi=150)
    plt.close()

def plot_jsd_vs_temperature(temperature_values: List[float], avg_jsd_values: List[float], output_dir: str, param_set_str: str):
    """
    Plots the average JSD values against temperature.
    
    Args:
        temperature_values (List[float]): List of temperature values.
        avg_jsd_values (List[float]): Corresponding average JSD values.
        output_dir (str): Directory to save the plot.
        param_set_str (str): String representation of parameter set.
    """
    plt.figure(figsize=(8,5))
    plt.plot(temperature_values, avg_jsd_values, marker='o', linestyle='-')
    plt.title(f'Average JSD vs Temperature\n{param_set_str}')
    plt.xlabel('Temperature')
    plt.ylabel('Average JSD')
    plt.grid(True)
    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f'JSD_vs_Temperature.png')
    plt.savefig(plot_filename, dpi=150)
    plt.close()

###############################################################################
# 2. Main Function
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Find the ideal temperature parameter for softmax conversion.")
    parser.add_argument(
        "json_file",
        type=str,
        help="Path to the .json file containing vectors."
    )
    parser.add_argument(
        "--num_vectors",
        type=int,
        default=5,
        help="Number of random vectors to select (default: 5)."
    )
    parser.add_argument(
        "--temperature_start",
        type=float,
        default=0.1,
        help="Starting temperature value (default: 0.1)."
    )
    parser.add_argument(
        "--temperature_end",
        type=float,
        default=2.0,
        help="Ending temperature value (default: 2.0)."
    )
    parser.add_argument(
        "--temperature_step",
        type=float,
        default=0.1,
        help="Step size for temperature values (default: 0.1)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="temperature_analysis",
        help="Directory to save output plots (default: temperature_analysis)."
    )
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load JSON data
    if not os.path.isfile(args.json_file):
        print(f"Error: File '{args.json_file}' does not exist.")
        return
    
    with open(args.json_file, "r") as f:
        data = json.load(f)
    
    if "final_x_vectors" not in data:
        print("Error: 'final_x_vectors' key not found in the JSON file.")
        return
    
    vectors = data["final_x_vectors"]
    
    # Select random vectors
    selected_vectors = select_random_vectors(vectors, args.num_vectors)
    print(f"\nSelected {len(selected_vectors)} random vectors:\n")
    for agent, vec in selected_vectors.items():
        print(f"{agent}: {vec}")
    
    # Define temperature range
    temperature_values = np.arange(args.temperature_start, args.temperature_end + args.temperature_step, args.temperature_step)
    
    # Store JSD results
    avg_jsd_per_temp = []
    
    # Iterate over temperatures
    for temp in tqdm(temperature_values, desc="Analyzing Temperatures"):
        pd_vectors = {}
        # Convert vectors to PDs
        for agent, vec in selected_vectors.items():
            pd = softmax(vec, temp)
            pd_vectors[agent] = pd
        # Compute pairwise JSD
        jsd_values = []
        agents = list(pd_vectors.keys())
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                p = pd_vectors[agents[i]]
                q = pd_vectors[agents[j]]
                jsd = compute_jsd(p, q)
                jsd_values.append(jsd)
        # Calculate average JSD for this temperature
        avg_jsd = np.mean(jsd_values) if jsd_values else float('nan')
        avg_jsd_per_temp.append(avg_jsd)
        
        # Print results for this temperature
        print(f"\nTemperature: {temp}")
        for agent, pd in pd_vectors.items():
            print(f"\n{agent} PD: {pd}")
        print("\nPairwise JSDs:")
        
        # Corrected Pairwise JSDs Printing
        agent_pairs = list(itertools.combinations(agents, 2))
        for (agent1, agent2), jsd in zip(agent_pairs, jsd_values):
            print(f"JSD({agent1}, {agent2}) = {jsd:.4f}")
        
        # Optionally, plot PD vectors for this temperature
        param_set_str = ", ".join([f"{k}={v}" for k, v in data["params"].items()])
        plot_pd_vectors(pd_vectors, temp, args.output_dir, param_set_str)
    
    # Plot average JSD vs Temperature
    plot_jsd_vs_temperature(temperature_values, avg_jsd_per_temp, args.output_dir, param_set_str)
    
    # Summary Suggestions
    print("\n===========================================")
    print("Temperature Analysis Complete!")
    print(f"Plots saved in the directory: {args.output_dir}")
    print("\nSuggestions for Selecting Optimal Temperature:")
    print("1. **Stability in JSD:** Look for a temperature where the average JSD stabilizes or changes minimally with further temperature increases.")
    print("2. **Highlighting Peaks:** Choose a temperature that sharpens the PDs sufficiently to highlight differences at the peaks without making them too sparse.")
    print("3. **Visual Inspection:** Review the 'Probability Distributions' plots to ensure that peaks are prominent and differences are visible.")
    print("4. **Balance:** Aim for a balance where the PDs are neither too uniform (high temperature) nor too sharp (low temperature).")
    print("5. **Domain Knowledge:** Incorporate your understanding of the data and what level of differentiation is meaningful for your specific application.")
    print("===========================================\n")

###############################################################################
# 3. Entry Point
###############################################################################

if __name__ == "__main__":
    main()
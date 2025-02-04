#!/usr/bin/env python3
"""
File: plot_violin.py

Summary:
    Reads a simulation result from a JSON file (with a "final_moving_avg" key) and
    generates a violin plot that visualizes the distribution of each dimension in the agents' state vectors.
    
    Each agent's state vector is assumed to be a list of certainty values (e.g., between 0 and 1).
    The script melts the data into a "long" format so that Seaborn can plot one violin for each dimension.
    
Usage:
    python3 plot_violin.py path/to/simulation_result.json --output output_plot.png

If the --output option is omitted, the plot is shown interactively.
"""

import json
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_violin(json_file, output_file=None):
    # Read the JSON file.
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Get the final moving averages.
    # Expecting final_moving_avg to be a dict mapping agent names to a list (state vector).
    final_moving_avg = data.get("final_moving_avg", None)
    if final_moving_avg is None:
        print("Error: The JSON file does not contain a 'final_moving_avg' key.")
        return
    
    # Convert the dictionary to a DataFrame.
    # Each row corresponds to one agent; columns correspond to dimensions.
    df = pd.DataFrame.from_dict(final_moving_avg, orient="index")
    # Rename columns to more descriptive names: Info_1, Info_2, ... Info_m
    num_dims = df.shape[1]
    df.columns = [f"Info_{i+1}" for i in range(num_dims)]
    
    # Melt the DataFrame into a long format.
    # The resulting DataFrame will have three columns: 'Agent', 'Information', and 'Certainty'
    df_long = df.reset_index().melt(id_vars="index", var_name="Information", value_name="Certainty")
    df_long = df_long.rename(columns={"index": "Agent"})
    
    # Create the violin plot.
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Information", y="Certainty", data=df_long, palette="Set3")
    plt.title("Distribution of Agent State Vector Certainty Values")
    plt.xlabel("Piece of Information (Dimension)")
    plt.ylabel("Certainty Level")
    plt.tight_layout()
    
    # Either save the plot to a file or show it interactively.
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Generate a violin plot of agent state vector distributions from a simulation result JSON.")
    parser.add_argument("json_file", help="Path to the simulation result JSON file")
    parser.add_argument("--output", help="Path to save the output plot (e.g., plot.png). If omitted, the plot is displayed interactively.")
    args = parser.parse_args()
    
    plot_violin(args.json_file, args.output)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
File: survival_per_parameter.py

Summary:
    This script analyzes the effect of one "interesting parameter" on the proportion
    of surviving information from simulation results produced by sweep.py.
    
    For each JSON file in the input directory, the script:
      - Reads the simulation result (which includes "final_moving_avg" and "params").
      - Uses the survival criterion (from surviving_information.py) to count how many
        pieces of information survived.
      - Computes the proportion of surviving information by dividing the count by m.
    
    Then, the script groups results by the interesting parameter. If a secondary parameter
    is provided, the script groups further by the secondary parameter and plots a separate line
    for each secondary value. In either case, it computes the average proportion and a 95% confidence
    interval (error bar) from the multiple repetitions.
    
    Finally, the script generates a line plot with:
      - x-axis: the values of the interesting parameter,
      - y-axis: the average proportion of surviving information,
      - error bars: 95% confidence intervals.
    
Usage:
    # Without secondary parameter:
    python3 survival_per_parameter.py --input_dir sweep_results --interesting sigma
    
    # With secondary parameter (e.g., m):
    python3 survival_per_parameter.py --input_dir sweep_results --interesting sigma --secondary m --output survival_vs_sigma.png

Inputs:
    - --input_dir: Directory containing JSON simulation outputs.
    - --interesting: The parameter whose effect is to be analyzed (e.g., "sigma").
    - --secondary: (Optional) A secondary parameter to further separate runs (e.g., "m").
    - --threshold: Survival threshold (passed to count_surviving_info, default 0.5).
    - --fraction: Minimum fraction (default 0.1).
    - --output: (Optional) Filename for saving the plot; if not provided, the plot is shown interactively.
    - --confidence: Confidence level (default 0.95).
    
Outputs:
    A line plot showing the average proportion of surviving information (with 95% CI error bars)
    versus the interesting parameter. If a secondary parameter is provided, one line is plotted per
    distinct secondary value.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from statistics import mean, stdev
import math

# Import the survival counting function from surviving_information.py
from surviving_information import count_surviving_info

def parse_json_files(input_dir):
    """
    Load all JSON files from the given directory.
    
    Returns:
        A list of dictionaries, one per JSON file.
    """
    results = []
    for fname in os.listdir(input_dir):
        if fname.endswith(".json"):
            path = os.path.join(input_dir, fname)
            with open(path, "r") as f:
                try:
                    data = json.load(f)
                    results.append(data)
                except Exception as e:
                    print(f"Error loading {fname}: {e}")
    return results

def aggregate_survival_data(json_data, interesting_param, secondary_param=None,
                            threshold=0.5, fraction=0.1):
    """
    Aggregates the proportion of surviving information (survived/m) for each
    value of the interesting parameter, and if provided, further groups by the secondary parameter.
    
    Args:
        json_data: List of JSON simulation results.
        interesting_param: The key (in params) whose different values will be on the x-axis.
        secondary_param: (Optional) A secondary grouping parameter.
        threshold: Survival threshold for count_surviving_info.
        fraction: Minimum fraction for count_surviving_info.
    
    Returns:
        A nested dictionary if secondary_param is provided:
          data[secondary_value][interesting_value] = list of proportions
        Otherwise, a dictionary:
          data[interesting_value] = list of proportions
    """
    aggregated = defaultdict(lambda: defaultdict(list)) if secondary_param else defaultdict(list)
    
    for entry in json_data:
        params = entry.get("params", {})
        if interesting_param not in params:
            continue  # skip if the interesting parameter is missing
        
        interesting_val = params[interesting_param]
        m = params.get("m", None)
        if m is None:
            continue  # cannot compute proportion without knowing m
        
        # Count surviving pieces using existing criterion.
        survived = count_surviving_info(entry.get("final_moving_avg", {}), 
                                        survival_threshold=threshold, 
                                        min_fraction=fraction)
        prop = survived / m  # proportion of surviving pieces
        
        if secondary_param:
            sec_val = params.get(secondary_param, None)
            if sec_val is not None:
                aggregated[sec_val][interesting_val].append(prop)
        else:
            aggregated[interesting_val].append(prop)
            
    return aggregated

def compute_mean_ci(values, confidence=0.95):
    """
    Compute the mean and 95% confidence interval for a list of values.
    Returns (mean, lower_bound, upper_bound).
    If only one value is present, returns (value, value, value).
    """
    n = len(values)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    mean_val = mean(values)
    if n == 1:
        return (mean_val, mean_val, mean_val)
    s = stdev(values)
    # For 95% confidence using normal approximation, z is approximately 1.96.
    z = 1.96 if confidence == 0.95 else 1.96  # For now, fix to 95%
    margin = z * (s / math.sqrt(n))
    return (mean_val, mean_val - margin, mean_val + margin)

def plot_survival_vs_parameter(aggregated, interesting_param, secondary_param=None, confidence=0.95, output_file=None):
    """
    Plots a line plot of the average proportion of surviving information versus the interesting parameter.
    If secondary_param is provided, a separate line is drawn for each secondary value.
    
    Args:
        aggregated: Nested dictionary of the form:
            If secondary_param is provided:
                data[sec_val][interesting_val] = list of proportions
            Else:
                data[interesting_val] = list of proportions
        interesting_param: The name of the interesting parameter (for labeling the x-axis).
        secondary_param: (Optional) The name of the secondary parameter (for multiple lines).
        confidence: Confidence level for error bars (default 0.95).
        output_file: (Optional) If provided, the plot is saved to this file; otherwise, shown interactively.
    """
    plt.figure(figsize=(8, 6))
    
    if secondary_param:
        for sec_val, inner_dict in aggregated.items():
            x_vals = sorted(inner_dict.keys())
            y_means = []
            y_err_lower = []
            y_err_upper = []
            for x in x_vals:
                mean_val, low, high = compute_mean_ci(inner_dict[x], confidence)
                y_means.append(mean_val)
                y_err_lower.append(mean_val - low)
                y_err_upper.append(high - mean_val)
            # Plot errorbar with label indicating secondary parameter value.
            plt.errorbar(x_vals, y_means, yerr=[y_err_lower, y_err_upper], label=f"{secondary_param}={sec_val}",
                         marker="o", capsize=5, linestyle="-")
    else:
        x_vals = sorted(aggregated.keys())
        y_means = []
        y_err_lower = []
        y_err_upper = []
        for x in x_vals:
            mean_val, low, high = compute_mean_ci(aggregated[x], confidence)
            y_means.append(mean_val)
            y_err_lower.append(mean_val - low)
            y_err_upper.append(high - mean_val)
        plt.errorbar(x_vals, y_means, yerr=[y_err_lower, y_err_upper],
                     marker="o", capsize=5, linestyle="-")
    
    plt.xlabel(interesting_param)
    plt.ylabel("Proportion of Surviving Information")
    plt.title(f"Surviving Information vs {interesting_param}")
    plt.ylim(-0.1,1)
    if secondary_param:
        plt.legend(title=secondary_param)
    plt.grid(True)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Plot the effect of an interesting parameter on the proportion of surviving information."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing JSON simulation outputs (from sweep.py)")
    parser.add_argument("--interesting", type=str, required=True,
                        help="The parameter of interest (e.g., sigma)")
    parser.add_argument("--secondary", type=str, default=None,
                        help="(Optional) A secondary parameter for grouping (e.g., m)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Survival threshold (default: 0.5)")
    parser.add_argument("--fraction", type=float, default=0.1,
                        help="Minimum fraction (default: 0.1)")
    parser.add_argument("--confidence", type=float, default=0.95,
                        help="Confidence level for error bars (default: 0.95)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename for the plot (if not provided, the plot is shown interactively)")
    args = parser.parse_args()

    # Load all JSON simulation results.
    json_data = parse_json_files(args.input_dir)
    if not json_data:
        print("No JSON files found in the specified directory.")
        return

    # Aggregate survival data.
    aggregated = aggregate_survival_data(
        json_data,
        interesting_param=args.interesting,
        secondary_param=args.secondary,
        threshold=args.threshold,
        fraction=args.fraction
    )

    # Plot the results.
    plot_survival_vs_parameter(
        aggregated,
        interesting_param=args.interesting,
        secondary_param=args.secondary,
        confidence=args.confidence,
        output_file=args.output
    )

if __name__ == "__main__":
    main()

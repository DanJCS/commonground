#!/usr/bin/env python3
"""
File: analyze_two_params.py

Summary:
    Analyzes how two chosen ABM parameters (param1 and param2) jointly affect the number
    of surviving pieces of information by the end of each simulation run. This script
    groups runs by all other parameters (the "major group") and, for each major group,
    produces a 2D heatmap plotting param1 on the X-axis and param2 on the Y-axis.
    Each cell in the heatmap shows the mean count (and variance) of surviving information.

Key Functions:
    * load_results(input_dir):
        Loads .json simulation files, returning a list of dictionaries.
    * make_major_key(params, param1, param2):
        Extracts and sorts all parameter entries except param1 and param2,
        treating them as a unique "major group" identifier.
    * main():
        - Reads command-line arguments to locate input files, param1, param2, etc.
        - Builds nested dictionaries: major_group -> {(param1_val, param2_val) -> [survival_counts]}.
        - For each major group, constructs a heatmap that visualizes the mean (and variance) of survived info
          for each combination of param1 and param2.

Dependencies:
    * Python built-ins: os, json, argparse, collections (defaultdict)
    * Third-party: matplotlib, numpy
    * Internal:
        - Surviving_information (import count_surviving_info)

Usage:
    python analyze_two_params.py <input_dir> <param1> <param2> <output_dir> \
        [--threshold 0.5] [--fraction 0.1]

    Where:
      <input_dir> contains the .json result files.
      <param1> is the name of the first ABM parameter for the X-axis.
      <param2> is the name of the second ABM parameter for the Y-axis.
      <output_dir> is where heatmaps will be saved.
      --threshold and --fraction control the survival logic (optional).
"""

# Advice on potential redundancy with plot_surviving_by_m.py:
"""
- **plot_surviving_by_m.py** focuses on a single parameter (m) against the
  number of surviving pieces, generating a line plot with confidence intervals.
- **analyze_two_params.py** allows you to pick any two parameters (including “m” 
  as one of them) and produces 2D heatmaps.

They each serve slightly different purposes:
- `plot_surviving_by_m.py` specifically builds a line plot for m vs. survival counts,
  possibly grouping by other parameters in the background.
- `analyze_two_params.py` is more general, letting you pick any two parameters 
  and produce a heatmap.

Hence, `analyze_two_params.py` does not necessarily make `plot_surviving_by_m.py` 
redundant. While you could analyze “m” as one of the two parameters in the heatmap, 
you might lose the specialized features of `plot_surviving_by_m.py` 
(such as line plots, linear fits, and direct analysis on a single dimension).
In other words, they can complement each other rather than replace one another.
"""


import os
import json
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple

# Import the survival counting function from the helper module
from surviving_information import count_surviving_info

###############################################################################
# 1. Helper: load_results
###############################################################################
def load_results(input_dir: str) -> List[dict]:
    """
    Loads all .json files from input_dir and returns a list of dictionaries.
    Each dict is expected to have at least:
       {
          "params": {...},
          "final_moving_avg": {...}
       }
    """
    results = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(input_dir, filename)
            with open(filepath, "r") as f:
                data = json.load(f)
                results.append(data)
    return results

###############################################################################
# 2. make_major_key
###############################################################################
def make_major_key(params: dict, param1: str, param2: str) -> Tuple[Tuple[str, str], ...]:
    """
    Build a tuple of (key, value) for all params EXCEPT param1 and param2, sorted by key name.
    This key identifies the 'major group'.
    """
    exclude = {param1, param2}
    filtered = {k: v for k, v in params.items() if k not in exclude}
    return tuple(sorted(filtered.items()))

###############################################################################
# 3. Main script
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Analyze how two parameters interplay in surviving info.")
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing simulation .json results."
    )
    parser.add_argument(
        "param1",
        type=str,
        help="First parameter to analyze for the heatmap (X-axis)."
    )
    parser.add_argument(
        "param2",
        type=str,
        help="Second parameter to analyze for the heatmap (Y-axis)."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory where plots will be saved."
    )
    # Optional thresholds for count_surviving_info
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Survival threshold (default=0.5)."
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.1,
        help="Minimum fraction of agents required for survival (default=0.1)."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load the results
    all_data = load_results(args.input_dir)

    # 2. Create a nested dictionary structure:
    #    major_groups[ major_key ][ (param1_val, param2_val) ] -> list of survival counts
    major_groups = defaultdict(lambda: defaultdict(list))

    for entry in all_data:
        if "params" not in entry or "final_moving_avg" not in entry:
            continue

        params = entry["params"]
        final_moving_avg = entry["final_moving_avg"]

        # Identify major group key
        major_key = make_major_key(params, args.param1, args.param2)

        # Minor group key: the param1 and param2 values
        val1 = params.get(args.param1, None)
        val2 = params.get(args.param2, None)
        minor_key = (val1, val2)

        # Compute survival count for this run using the imported function
        survived_count = count_surviving_info(
            final_moving_avg,
            survival_threshold=args.threshold,
            min_fraction=args.fraction
        )

        # Store the result
        major_groups[major_key][minor_key].append(survived_count)

    # 3. For each major group, we want a heatmap of param1 vs param2
    for major_key, minor_dict in major_groups.items():
        # Collect unique values for param1 and param2
        all_pairs = list(minor_dict.keys())
        param1_vals = sorted(set(p[0] for p in all_pairs if p[0] is not None))
        param2_vals = sorted(set(p[1] for p in all_pairs if p[1] is not None), reverse=True)

        # Build a 2D matrix (heatmap_data) for mean survival
        # and another for variance (optional)
        heatmap_data = np.zeros((len(param2_vals), len(param1_vals)), dtype=float)
        heatmap_var = np.zeros((len(param2_vals), len(param1_vals)), dtype=float)

        for r_idx, val2 in enumerate(param2_vals):
            for c_idx, val1 in enumerate(param1_vals):
                counts = minor_dict.get((val1, val2), [])
                if counts:
                    arr = np.array(counts, dtype=float)
                    heatmap_data[r_idx, c_idx] = arr.mean()
                    heatmap_var[r_idx, c_idx] = arr.var()
                else:
                    heatmap_data[r_idx, c_idx] = np.nan
                    heatmap_var[r_idx, c_idx] = np.nan

        # 4. Plot the heatmap
        fig, ax = plt.subplots(figsize=(10, 10))
        cax = ax.imshow(
            heatmap_data,
            aspect="auto",
            origin="upper",
            cmap="viridis"
        )
        fig.colorbar(cax, ax=ax, label="Mean Surviving Info")

        ax.set_title(f"{args.param1} vs {args.param2} Surviving Info Count")
        ax.set_xlabel(args.param1)
        ax.set_ylabel(args.param2)

        ax.set_xticks(range(len(param1_vals)))
        ax.set_xticklabels(param1_vals, rotation=45, ha="right")
        ax.set_yticks(range(len(param2_vals)))
        ax.set_yticklabels(param2_vals)

        # Insert mean/variance text in each cell
        for r_idx, val2 in enumerate(param2_vals):
            for c_idx, val1 in enumerate(param1_vals):
                mean_val = heatmap_data[r_idx, c_idx]
                var_val = heatmap_var[r_idx, c_idx]
                if not np.isnan(mean_val):
                    text_str = f"{mean_val:.2f}\n({var_val:.2f})"
                    ax.text(
                        c_idx, r_idx, text_str,
                        ha="center", va="center",
                        color="white",
                        fontsize=9
                    )

        plt.tight_layout()
        outfile = os.path.join(
            args.output_dir,
            f"heatmap_{abs(hash(major_key))}.png"
        )
        plt.savefig(outfile, dpi=150)
        plt.close()

    print("All heatmaps have been saved successfully.")

if __name__ == "__main__":
    main()

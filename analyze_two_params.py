import os
import json
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table
from collections import defaultdict

###############################################################################
# 1. count_surviving_info (reusing from your existing logic)
###############################################################################
def count_surviving_info(final_moving_avg, survival_threshold=0.5, min_fraction=0.1):
    """
    Given final_moving_avg (dict: agent_name -> list of length m),
    determine how many 'pieces of information' survive.

    Survival criterion:
      - At least `min_fraction` of agents must have final_moving_avg[i] > `survival_threshold`
      - i is the index of the piece of info in each agent's vector.

    Returns an integer: the count of survived pieces.
    """
    agent_names = list(final_moving_avg.keys())
    num_agents = len(agent_names)
    if num_agents == 0:
        return 0  # avoid divide-by-zero

    # Assume each agent has a vector of length m
    m = len(final_moving_avg[agent_names[0]])

    survived_count = 0
    cutoff_count = int(min_fraction * num_agents)

    for i in range(m):
        above_threshold = sum(
            1
            for agent in agent_names
            if final_moving_avg[agent][i] > survival_threshold
        )
        if above_threshold >= cutoff_count:
            survived_count += 1

    return survived_count

###############################################################################
# 2. Helper: load JSON results
###############################################################################
def load_results(input_dir):
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
# 3. Group by "major" (excluding param1, param2)
###############################################################################
def make_major_key(params, param1, param2):
    """
    Build a tuple of (key, value) for all params EXCEPT param1 and param2, sorted by key name.
    This key identifies the 'major group'.
    """
    exclude = {param1, param2}
    filtered = {k: v for k, v in params.items() if k not in exclude}
    return tuple(sorted(filtered.items()))

###############################################################################
# 4. Main script
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
        params = entry["params"]
        final_moving_avg = entry["final_moving_avg"]

        # Identify major group key
        major_key = make_major_key(params, args.param1, args.param2)

        # Minor group key: the param1 and param2 values
        val1 = params.get(args.param1, None)
        val2 = params.get(args.param2, None)
        minor_key = (val1, val2)

        # Compute survival count for this run
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
        # and another for variance
        heatmap_data = np.zeros((len(param2_vals), len(param1_vals)), dtype=float)
        heatmap_var = np.zeros((len(param2_vals), len(param1_vals)), dtype=float)

        for r_idx, val2 in enumerate(param2_vals):
            for c_idx, val1 in enumerate(param1_vals):
                counts = minor_dict.get((val1, val2), [])
                if len(counts) > 0:
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

        # 5. Add a small table below the plot to display major group parameters
        # Convert major_key (a tuple of (key, value)) into a table:
        major_dict = dict(major_key)
        # Build table data with a header row, e.g. ("Param", "Value")
        table_data = [("Parameters", "Value")] + list(major_dict.items())

        # Create the table; 'bbox' adjusts the position below the plot
        the_table = ax.table(
            cellText=table_data,
            loc="bottom",
            cellLoc="center",
            colLabels=None,
            bbox=[0.0, -0.6, 1.0, 0.5]  # tweak as needed (x, y, width, height)
        )
        # Adjust font sizes
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(8)

        # Increase bottom margin to make room for the table
        plt.subplots_adjust(bottom=0.3)

        # 6. Save the figure
        filename_parts = [f"{k}{v}" for k, v in major_key]
        major_str = "_".join(filename_parts) if filename_parts else "default"
        outfile = os.path.join(args.output_dir, f"heatmap_{major_str}.png")
        plt.tight_layout()
        plt.savefig(outfile, dpi=150)
        plt.close()

    print("All heatmaps have been saved successfully.")


###############################################################################
# Execute main
###############################################################################
if __name__ == "__main__":
    main()
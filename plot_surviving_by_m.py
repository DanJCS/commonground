import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
from typing import Dict, List
from surviving_information import count_surviving_info  # We'll reuse your function

# -------------------------------------------------------------------
# 1) Load JSON results from disk
# -------------------------------------------------------------------
def load_simulation_results(results_dir: str) -> List[dict]:
    """
    Loads all .json simulation result files from `results_dir`.
    Returns a list of dictionaries, each containing simulation output.
    """
    all_results = []
    for filename in os.listdir(results_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, "r") as f:
                data = json.load(f)
                all_results.append(data)
    return all_results

# -------------------------------------------------------------------
# 2) Group data by all parameters except 'm'
# -------------------------------------------------------------------
def make_param_key_excl_m(params: dict, exclude_keys=None) -> frozenset:
    """
    Build a hashable key of (param, value) pairs, excluding 'm'
    and any other excluded keys provided.
    """
    if exclude_keys is None:
        exclude_keys = set()
    # Always exclude 'm'
    exclude_keys = exclude_keys.union({'m'})
    
    filtered = {k: v for k, v in params.items() if k not in exclude_keys}
    # Sort to keep consistent ordering
    return frozenset(sorted(filtered.items()))

# -------------------------------------------------------------------
# 3) Parse each result, compute survival count, store in a data structure
# -------------------------------------------------------------------
def compute_survival_counts(
    results_dir: str,
    survival_threshold=0.5,
    min_fraction=0.1
) -> Dict[frozenset, Dict[int, List[int]]]:
    """
    1. Loads each .json result.
    2. Uses `count_surviving_info(final_moving_avg, survival_threshold, min_fraction)`
       to compute how many pieces of information survived in each simulation.
    3. Groups data by (all params except m) -> { m -> [list_of_counts_for_each_rep] }.

    Returns:
       grouped_data: {
         param_key_excl_m (frozenset): {
           m_value (int): [counts_across_repetitions],
           ...
         },
         ...
       }
    """
    all_data = load_simulation_results(results_dir)
    grouped_data = defaultdict(lambda: defaultdict(list))

    for entry in all_data:
        params = entry["params"]
        final_moving_avg = entry["final_moving_avg"]

        # Compute how many pieces survived
        survived_count = count_surviving_info(
            final_moving_avg,
            survival_threshold=survival_threshold,
            min_fraction=min_fraction
        )

        param_key = make_param_key_excl_m(params)
        m_value = params["m"]  # We'll group by m_value within each param_key
        grouped_data[param_key][m_value].append(survived_count)

    return grouped_data

# -------------------------------------------------------------------
# 4) Compute mean & 90% CI for each m in each group
# -------------------------------------------------------------------
def summarize_survival_data(grouped_data: Dict[frozenset, Dict[int, List[int]]]):
    """
    For each unique param_key_excl_m, compute:
      - sorted list of m
      - mean survival
      - 90% CI for each m

    Returns: {
      param_key_excl_m: {
        "m_values": [...],
        "mean_survival": [...],
        "lower_ci": [...],
        "upper_ci": [...]
      }
    }
    """
    z_90 = 1.645  # z-score for ~90% confidence (normal approximation)
    results_for_plot = {}

    for param_key, m_dict in grouped_data.items():
        m_values_sorted = sorted(m_dict.keys())
        mean_list = []
        lower_ci_list = []
        upper_ci_list = []

        for m_val in m_values_sorted:
            counts = np.array(m_dict[m_val], dtype=float)
            mean_val = np.mean(counts)
            std_val = np.std(counts, ddof=1) if len(counts) > 1 else 0.0
            n = len(counts)

            # Confidence interval margin
            if n > 1:
                margin = z_90 * (std_val / math.sqrt(n))
            else:
                margin = 0.0

            mean_list.append(mean_val)
            lower_ci_list.append(mean_val - margin)
            upper_ci_list.append(mean_val + margin)

        results_for_plot[param_key] = {
            "m_values": m_values_sorted,
            "mean_survival": mean_list,
            "lower_ci": lower_ci_list,
            "upper_ci": upper_ci_list
        }

    return results_for_plot

# -------------------------------------------------------------------
# 5) Plot (m vs. mean survival) with 90% CI, separate plot per param_key_excl_m
# -------------------------------------------------------------------
def plot_survival_vs_m(
    results_for_plot: Dict[frozenset, dict], 
    output_dir="plots"
):
    """
    Creates a separate plot for each param_key_excl_m.
    X-axis: m
    Y-axis: mean survival
    Error bars: 90% CI
    Plot title: display relevant params (excl 'm')
    """
    os.makedirs(output_dir, exist_ok=True)

    for param_key_excl_m, data_dict in results_for_plot.items():
        m_values = data_dict["m_values"]
        mean_vals = data_dict["mean_survival"]
        lower_ci = data_dict["lower_ci"]
        upper_ci = data_dict["upper_ci"]

        # Reconstruct param_dict for labeling
        param_dict = dict(param_key_excl_m)
        title_str = ", ".join(f"{k}={v}" for k, v in sorted(param_dict.items()) if k in {"eps", "gamma", "zeta"})
        
        # Prepare error bars
        y = np.array(mean_vals)
        yerr_lower = y - np.array(lower_ci)
        yerr_upper = np.array(upper_ci) - y
        yerr = [yerr_lower, yerr_upper]
        
        # Linear fit
        coefficients = np.polyfit(m_values, mean_vals, 1)  # Linear fit (degree 1)
        linear_fit = np.poly1d(coefficients)
        fit_line = linear_fit(m_values)
        
        plt.figure(figsize=(6,4))
        plt.errorbar(m_values, mean_vals, yerr=yerr, fmt='o-', capsize=4)
        plt.plot(m_values, fit_line, "--", color="red", label="Linear Fit")
        # Add the linear fit equation to the plot
        equation_text = f"y = {coefficients[0]:.2f}x + {coefficients[1]:.2f}"
        plt.text(
            0.05, 0.95, equation_text, transform=plt.gca().transAxes,
            fontsize=10, verticalalignment='top', color="red"
        )
        plt.title(f"Survival vs. m\n{title_str}", fontsize=10)
        plt.xlabel("m (Length of x-vector)")
        plt.ylabel("Mean # of Surviving Pieces")
        plt.ylim(bottom=0)
        plt.grid(True)

        # Save figure
        filename = _filename_from_param_dict(param_dict)
        plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=150)
        plt.close()

def _filename_from_param_dict(param_dict: dict) -> str:
    """
    Generates a simple filename from param_dict, only including 'eps', 'gamma', and 'zeta'.
    Example: "eps0.5_gamma-1.0_zeta0"
    """
    keys_of_interest = {"eps", "gamma", "zeta"}
    parts = []
    for k, v in sorted(param_dict.items()):
        if k in keys_of_interest:
            parts.append(f"{k}{v}")
    return "_".join(parts)

# -------------------------------------------------------------------
# 6) Main: orchestrates the entire process
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Plot survival vs. m with 90% CI.")
    parser.add_argument(
        "results_dir",
        type=str,
        help="Directory containing .json result files."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save the plots."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Survival threshold for final_moving_avg (default: 0.5)."
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.1,
        help="Minimum fraction of agents required for survival (default: 0.1)."
    )

    args = parser.parse_args()

    # 1. Load data, compute survival counts
    grouped_data = compute_survival_counts(
        results_dir=args.results_dir,
        survival_threshold=args.threshold,
        min_fraction=args.fraction
    )

    # 2. Summarize results (m-values, mean survival, 90% CI)
    results_for_plot = summarize_survival_data(grouped_data)

    # 3. Generate and save plots
    plot_survival_vs_m(results_for_plot, output_dir=args.output_dir)

    print(f"Done! Plots saved in '{args.output_dir}' folder.")
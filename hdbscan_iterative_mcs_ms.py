#!/usr/bin/env python3

"""
hdbscan_iterative_mcs_ms.py

An iterative approach to choosing HDBSCAN hyperparameters (mcs, ms)
based on mean #clusters rather than silhouette scores:

Step 1:
  - Fix ms = 1.
  - Sweep mcs in a given range, compute #clusters for each ABM parameter set.
  - Plot "mcs vs. #clusters" (with confidence intervals).
  - Prompt user to input a chosen mcs for each param set.

Step 2:
  - For each param set, fix the user-chosen mcs.
  - Sweep ms in a given range, compute #clusters, plot "ms vs. #clusters".
  - Prompt user to pick an ms (or just produce the plot so user can choose).

NOTE: This is a template. You'll likely refine the user interface, the ranges,
      and how you store user-chosen hyperparams, etc.
"""

import os
import json
import math
import numpy as np
import hdbscan
from multiprocessing import Pool
from tqdm import tqdm
from typing import Dict, List, Tuple
from statistics import mean, stdev
from math import sqrt

import matplotlib.pyplot as plt

from pairwise_similarity import load_final_moving_avg, build_distance_matrix


###############################################################################
# 1. Basic clustering: compute #clusters ignoring noise
###############################################################################
def compute_num_clusters(dist_matrix: np.ndarray, min_cluster_size: int, min_samples: int) -> int:
    """
    Run HDBSCAN on a precomputed NxN distance matrix and return the number
    of clusters (excluding noise).
    """
    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )
    labels = clusterer.fit_predict(dist_matrix)
    # exclude noise
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    return len(unique_labels)

###############################################################################
# 2. Confidence Interval helper
###############################################################################
def confidence_interval(data, confidence=0.90):
    """
    Basic normal-based CI. For a more robust approach, consider bootstrapping.
    """
    if len(data) <= 1:
        return (float("nan"), float("nan"))
    m = mean(data)
    s = stdev(data)
    z = 1.645  # ~90% z-score
    half_width = z * (s / math.sqrt(len(data)))
    return (m - half_width, m + half_width)

###############################################################################
# 3. Grouping by ABM params
###############################################################################
def make_param_key(data: Dict) -> Tuple:
    """
    Convert data["params"] into a sorted tuple that uniquely identifies
    the ABM parameters (excluding repetition keys, etc.).
    """
    # Exclude keys that are not truly part of the ABM definition
    exclude = {"rep"}
    abm_params = data["params"]
    filtered = {k: v for k, v in abm_params.items() if k not in exclude}
    return tuple(sorted(filtered.items()))

###############################################################################
# 4. Worker for computing #clusters for a single (param_key, json_file)
#    given fixed (mcs, ms)
###############################################################################
def process_file_for_clusters(args) -> Tuple[Tuple, int]:
    """
    Worker function for parallel usage.

    :param args: (param_key, json_file, input_dir, method, temperature,
                  min_cluster_size, min_samples)
    :return: (param_key, n_clusters) for that run
    """
    (param_key, json_file, input_dir, method, temperature,
     min_cluster_size, min_samples) = args

    fullpath = os.path.join(input_dir, json_file)
    fmavg = load_final_moving_avg(fullpath)
    if fmavg is None:
        return (param_key, None)

    dist_mat, _ = build_distance_matrix(fmavg, method=method, temperature=temperature)
    n_clust = compute_num_clusters(dist_mat, min_cluster_size, min_samples)
    return (param_key, n_clust)

###############################################################################
# 5. Plotting: #clusters vs. hyperparameter
###############################################################################
def plot_clusters_vs_hyperparam(x_vals, mean_clusters, ci_low, ci_high,
                                hyperparam_name="mcs", outfile="clusters_vs_param.png",
                                param_key_str=""):
    """
    Creates a simple line plot: X-axis = hyperparam, Y-axis = mean #clusters,
    with error bars for the CI. Saves to outfile.
    """
    plt.figure(figsize=(8,6))
    plt.plot(x_vals, mean_clusters, marker="o", color="blue", label="Mean #clusters")

    # Add error bars
    for x, m, lo, hi in zip(x_vals, mean_clusters, ci_low, ci_high):
        plt.vlines(x, lo, hi, color="blue", alpha=0.5)

    plt.xlabel(hyperparam_name)
    plt.ylabel("Mean #clusters")
    plt.title(f"#Clusters vs. {hyperparam_name}\n{param_key_str}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

###############################################################################
# 6. Main iterative approach
###############################################################################
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory with .json files")
    parser.add_argument("--output_dir", help="Where to save output plots", default="iterative_hdbscan_plots")
    parser.add_argument("--method", default="jsd",
                        help="Distance metric: jsd, euclidean, or cosine")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of parallel processes")
    # Ranges for searching
    parser.add_argument("--mcs_range", nargs="+", type=int,
                        default=[20, 30, 40, 50],
                        help="Range of mcs to test in step 1 (fix ms=1)")
    parser.add_argument("--ms_range", nargs="+", type=int,
                        default=[5, 10, 15, 20],
                        help="Range of ms to test in step 2 (with user-chosen mcs)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Identify all .json files and group them by param_key
    file_list = [f for f in os.listdir(args.input_dir) if f.endswith(".json")]
    file_list = sorted(file_list)
    param_groups = {}
    for fname in file_list:
        fpath = os.path.join(args.input_dir, fname)
        with open(fpath, "r") as f:
            data = json.load(f)
        if "params" not in data:
            continue

        pk = make_param_key(data)
        param_groups.setdefault(pk, []).append(fname)

    print(f"Found {len(param_groups)} unique ABM parameter sets.")
    print(f"Step 1: We'll fix ms=1 and vary mcs in: {args.mcs_range}")
    print(f"Step 2: We'll fix the user-chosen mcs and vary ms in: {args.ms_range}")

    ###########################################################################
    # STEP 1: For each param_key, test multiple mcs (ms=1) => #clusters
    ###########################################################################
    # We'll do a single parallel pass for each param_key & mcs.
    step1_tasks = []
    for pk, fnames in param_groups.items():
        for mcs_val in args.mcs_range:
            for fname in fnames:
                step1_tasks.append((pk, fname, args.input_dir, args.method, args.temperature, mcs_val, 1))

    results_step1 = []
    print(f"\n[Step 1] Computing #clusters for ms=1, for each mcs in {args.mcs_range}")
    with Pool(processes=args.n_jobs) as pool:
        with tqdm(total=len(step1_tasks), desc="Step1 mcs loop") as pbar:
            for (param_key, n_clust) in pool.imap_unordered(process_file_for_clusters, step1_tasks):
                results_step1.append((param_key, n_clust))
                pbar.update(1)

    # Aggregation: results_step1 is a list of (param_key, n_clust).
    # But we also need to know which mcs was used. So let's revise slightly:
    # We'll store it as (param_key, mcs_val, n_clust).
    # So let's fix that by storing tasks in a dictionary to recover mcs_val.

    # Actually, let's store them in the same place to keep it simpler:
    # We'll create a dictionary that maps (pk, mcs_val) -> list of n_clusters
    # Then compute the mean and CI.
    step1_map = {}  # (pk, mcs_val) -> list of n_clusters
    # But we need to combine results from both loops. Let's store them carefully.

    # We'll reconstruct (pk, mcs_val) from the tasks array to unify the data:
    pk_mcs_list = []  # parallel to step1_tasks
    idx = 0

    # We'll re-run that same loop but store (pk, mcs_val) in an array so we can match them
    step1_tasks_args = []
    for pk, fnames in param_groups.items():
        for mcs_val in args.mcs_range:
            for fname in fnames:
                step1_tasks_args.append((pk, mcs_val))

    # Now let's fix the parallel loop to store the results in the same iteration order
    results_step1.clear()
    with Pool(processes=args.n_jobs) as pool:
        with tqdm(total=len(step1_tasks), desc="Step1 mcs loop") as pbar:
            imap_it = pool.imap_unordered(process_file_for_clusters, step1_tasks)
            for i, outval in enumerate(imap_it):
                (param_key, n_clust) = outval
                pk_mcs = step1_tasks_args[i]  # (pk, mcs_val)
                results_step1.append((param_key, pk_mcs[1], n_clust))
                pbar.update(1)

    # Now, build the dictionary
    # results_step1 => list of (param_key, mcs_val, n_clust)
    for (pk, mcs_val, n_clust) in results_step1:
        if n_clust is None:
            continue
        step1_map.setdefault((pk, mcs_val), []).append(n_clust)

    # For each param_key, produce a plot of "mcs vs mean #clusters"
    user_chosen_mcs = {}
    for pk in param_groups.keys():
        x_vals = []
        means = []
        ci_low = []
        ci_high = []
        for mcs_val in args.mcs_range:
            clist = step1_map.get((pk, mcs_val), [])
            if len(clist) == 0:
                x_vals.append(mcs_val)
                means.append(float("nan"))
                ci_low.append(float("nan"))
                ci_high.append(float("nan"))
            else:
                mu = mean(clist)
                lowc, highc = confidence_interval(clist, 0.90)
                x_vals.append(mcs_val)
                means.append(mu)
                ci_low.append(lowc)
                ci_high.append(highc)

        # Build a string version of pk for the title
        pk_str = ", ".join(f"{k}={v}" for k, v in pk)
        outfile = os.path.join(args.output_dir, f"{hash(pk)}_mcs_plot.png")
        plot_clusters_vs_hyperparam(x_vals, means, ci_low, ci_high,
                                    hyperparam_name="mcs", outfile=outfile,
                                    param_key_str=pk_str)

    print("\n[Step 1] All 'mcs vs #clusters' plots saved.")
    print("Please inspect the plots in", args.output_dir)
    print("Now you will be prompted to input the chosen mcs for each param set.\n")

    # Prompt user for each param set
    for pk in param_groups.keys():
        pk_str = ", ".join(f"{k}={v}" for k, v in pk)
        valid_range = ", ".join(map(str, args.mcs_range))
        print(f"ABM Param Set: {pk_str}")
        chosen = None
        while chosen is None:
            user_input = input(f"Enter your chosen mcs (options: {valid_range}): ")
            try:
                val = int(user_input)
                if val not in args.mcs_range:
                    print("Value not in range! Please choose from the listed range.")
                else:
                    chosen = val
            except:
                print("Invalid integer input. Please try again.")
        user_chosen_mcs[pk] = chosen
    print("\nUser-chosen mcs values stored.\n")

    ###########################################################################
    # STEP 2: For each param_key, we fix the chosen mcs, vary ms in args.ms_range
    ###########################################################################
    step2_tasks = []
    for pk, fnames in param_groups.items():
        chosen_mcs_val = user_chosen_mcs[pk]
        for ms_val in args.ms_range:
            for fname in fnames:
                step2_tasks.append((pk, fname, args.input_dir, args.method,
                                    args.temperature, chosen_mcs_val, ms_val))

    results_step2 = []
    print(f"[Step 2] Using chosen mcs from step 1, now varying ms in {args.ms_range}.\n")
    # We'll do the same approach: store results => (pk, ms_val, n_clust)
    pk_ms_list = []
    for pk, fnames in param_groups.items():
        chosen_mcs_val = user_chosen_mcs[pk]
        for ms_val in args.ms_range:
            for fname in fnames:
                pk_ms_list.append((pk, ms_val))

    step2_map = {}  # (pk, ms_val) -> list of n_clusters

    with Pool(processes=args.n_jobs) as pool:
        with tqdm(total=len(step2_tasks), desc="Step2 ms loop") as pbar:
            imap_it = pool.imap_unordered(process_file_for_clusters, step2_tasks)
            for i, outval in enumerate(imap_it):
                (param_key, n_clust) = outval
                pk_ms = pk_ms_list[i]  # (pk, ms_val)
                results_step2.append((param_key, pk_ms[1], n_clust))
                pbar.update(1)

    # Build dict
    for (pk, ms_val, n_clust) in results_step2:
        if n_clust is None:
            continue
        step2_map.setdefault((pk, ms_val), []).append(n_clust)

    # For each param set, produce plot "ms vs #clusters"
    user_chosen_ms = {}
    for pk in param_groups.keys():
        chosen_mcs_val = user_chosen_mcs[pk]
        x_vals = []
        means = []
        ci_low = []
        ci_high = []
        for ms_val in args.ms_range:
            clist = step2_map.get((pk, ms_val), [])
            if len(clist) == 0:
                x_vals.append(ms_val)
                means.append(float("nan"))
                ci_low.append(float("nan"))
                ci_high.append(float("nan"))
            else:
                mu = mean(clist)
                lowc, highc = confidence_interval(clist, 0.90)
                x_vals.append(ms_val)
                means.append(mu)
                ci_low.append(lowc)
                ci_high.append(highc)

        pk_str = ", ".join(f"{k}={v}" for k, v in pk)
        outfile = os.path.join(args.output_dir, f"{hash(pk)}_ms_plot.png")
        plot_clusters_vs_hyperparam(x_vals, means, ci_low, ci_high,
                                    hyperparam_name="ms", outfile=outfile,
                                    param_key_str=f"{pk_str}\n(mcs={chosen_mcs_val})")

    print("\n[Step 2] All 'ms vs #clusters' plots saved.")
    print("Please inspect these new plots in", args.output_dir)
    print("Now you will be prompted to input the chosen ms for each param set.\n")

    for pk in param_groups.keys():
        pk_str = ", ".join(f"{k}={v}" for k, v in pk)
        chosen_mcs_val = user_chosen_mcs[pk]
        valid_range = ", ".join(map(str, args.ms_range))
        print(f"ABM Param Set: {pk_str} (chosen mcs={chosen_mcs_val})")
        chosen = None
        while chosen is None:
            user_input = input(f"Enter your chosen ms (options: {valid_range}): ")
            try:
                val = int(user_input)
                if val not in args.ms_range:
                    print("Value not in range! Please choose from the listed range.")
                else:
                    chosen = val
            except:
                print("Invalid integer input. Please try again.")
        user_chosen_ms[pk] = chosen

    print("\nFinal user-chosen hyperparameters:")
    for pk in param_groups.keys():
        mcs_val = user_chosen_mcs[pk]
        ms_val = user_chosen_ms[pk]
        pk_str = ", ".join(f"{k}={v}" for k, v in pk)
        print(f"  ParamSet: {pk_str} => mcs={mcs_val}, ms={ms_val}")

    print("\nDone! You can now use these chosen hyperparams in your final analysis.\n")


if __name__ == "__main__":
    main()
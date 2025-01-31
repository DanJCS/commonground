#!/usr/bin/env python3
"""
File: analyze_clusters.py

Summary:
    Performs cluster analysis on simulation results using HDBSCAN.  The script loads 
    precomputed distance matrices from JSON files, applies HDBSCAN clustering with 
    specified hyperparameters, and analyzes the resulting cluster structures. It 
    calculates the number of clusters, optionally groups the results by an ABM parameter,
    and can generate plots to visualize the relationship between cluster characteristics
    and ABM parameters.

Key Functions:
    * compute_num_clusters(dist_mat, min_cluster_size, min_samples):
        Calculates the number of clusters (excluding noise) from a distance matrix
        using HDBSCAN with given hyperparameters.
    * process_single_file(args):
        Worker function for parallel processing. Loads a JSON file, computes the distance
        matrix, performs clustering, and returns the number of clusters and grouping value.
    * main():
        Handles command-line arguments, loads data, distributes tasks for parallel 
        processing, aggregates results, and optionally generates plots.

Dependencies:
    * Python built-ins: os, json, math, multiprocessing, statistics
    * Third-party: numpy, hdbscan, tqdm
    * Internal: pairwise_similarity, plot_utils

Usage:
    python analyze_clusters.py <input_dir> [--method jsd/euclidean/cosine] 
                                 [--temperature 1.0] 
                                 [--min_cluster_size 50] 
                                 [--min_samples 10] 
                                 [--n_jobs 4] 
                                 [--group_by_param gamma]

    Where:
        - <input_dir> is the directory containing the JSON files with distance matrices.
        - --method specifies the distance metric used (default: jsd).
        - --temperature sets the softmax temperature (default: 1.0).
        - --min_cluster_size and --min_samples are HDBSCAN hyperparameters.
        - --n_jobs controls the number of parallel processes.
        - --group_by_param specifies an ABM parameter for grouping results.
"""

import os
import json
import math
import numpy as np
import hdbscan
from multiprocessing import Pool
from tqdm import tqdm
from typing import Dict
from statistics import mean, stdev
from math import sqrt

from pairwise_similarity import load_final_moving_avg, build_distance_matrix
import plot_utils

###############################################################################
# 1. HDBSCAN: compute number of clusters ignoring noise
###############################################################################
def compute_num_clusters(dist_mat, min_cluster_size, min_samples):
    """
    Run HDBSCAN on a precomputed distance matrix and return the number of clusters
    (excluding noise points labeled as -1).
    """
    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )
    labels = clusterer.fit_predict(dist_mat)
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    return len(unique_labels)

###############################################################################
# 2. Worker function for parallel cluster analysis on a single .json file
###############################################################################
def process_single_file(args):
    """
    A worker function to:
      1. Load a .json file (which has final_moving_avg).
      2. Compute the NxN distance matrix.
      3. Run HDBSCAN with (min_cluster_size, min_samples).
      4. Determine #clusters (excluding noise).
      5. Return (group_val, n_clusters) for further aggregation.

    :param args: tuple(
        json_filename, input_dir, method, temperature,
        min_cluster_size, min_samples, group_by_param
    )
    :return: (group_val, num_clusters)
    """
    (json_filename, input_dir, method, temperature,
     min_cluster_size, min_samples, group_by_param) = args

    path = os.path.join(input_dir, json_filename)
    if not os.path.exists(path):
        return (None, None)

    # Load the JSON
    with open(path, "r") as f:
        data = json.load(f)

    fmavg = data.get("final_moving_avg", None)
    if not fmavg:
        return (None, None)

    # Build distance matrix
    dist_mat, agent_names = build_distance_matrix(
        fmavg, method=method, temperature=temperature
    )

    # Compute #clusters ignoring noise
    n_clust = compute_num_clusters(dist_mat, min_cluster_size, min_samples)

    # If grouping is requested, read data["params"][group_by_param]
    group_val = None
    if group_by_param and "params" in data:
        group_val = data["params"].get(group_by_param, None)

    return (group_val, n_clust)

###############################################################################
# 3. Main script
###############################################################################
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory with .json files")
    parser.add_argument("--method", default="jsd",
                        help="Distance metric: jsd, euclidean, or cosine")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--min_cluster_size", type=int, default=50)
    parser.add_argument("--min_samples", type=int, default=10)
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of parallel processes to use")
    parser.add_argument("--group_by_param", type=str, default=None,
                        help="ABM param name to group runs (e.g. gamma)")
    args = parser.parse_args()

    file_list = [f for f in os.listdir(args.input_dir) if f.endswith(".json")]
    file_list = sorted(file_list)

    # Prepare a dict or list for storing results across many .json files
    grouping = {}  # { group_val -> list of num_clusters }

    # Build a list of tasks for parallelization
    tasks = []
    for json_filename in file_list:
        tasks.append((
            json_filename,          # name of the .json file
            args.input_dir,         # directory
            args.method,            # distance metric
            args.temperature,       # softmax temperature
            args.min_cluster_size,  # HDBSCAN
            args.min_samples,       # HDBSCAN
            args.group_by_param     # used to group results
        ))

    print(f"Analyzing clusters for {len(tasks)} JSON files using {args.n_jobs} processes.")

    # Parallel loop with a progress bar
    results = []
    with Pool(processes=args.n_jobs) as pool:
        with tqdm(total=len(tasks), desc="Analyzing clusters") as pbar:
            for (group_val, n_clust) in pool.imap_unordered(process_single_file, tasks):
                if group_val is not None and n_clust is not None:
                    grouping.setdefault(group_val, []).append(n_clust)
                pbar.update(1)

    # Summarize results:
    #   For each group_val, compute mean + confidence interval
    final_stats = {}
    for gv, arr in grouping.items():
        arr = [x for x in arr if x is not None]
        if len(arr) == 0:
            continue
        mu = mean(arr)
        if len(arr) > 1:
            s = stdev(arr)
            z = 1.645  # 90% z
            halfw = z * (s / sqrt(len(arr)))
            ci_low, ci_high = mu - halfw, mu + halfw
        else:
            ci_low, ci_high = (mu, mu)
        final_stats[gv] = (mu, ci_low, ci_high)

    # Optional: Plot how #clusters changes with group_by_param
    if args.group_by_param:
        # Sort by param (if numeric)
        try:
            sorted_keys = sorted(final_stats.keys(), key=lambda x: float(x) if x is not None else -math.inf)
        except:
            sorted_keys = sorted(final_stats.keys())

        x_vals = []
        y_means = []
        y_low = []
        y_high = []
        for k in sorted_keys:
            mu, lci, hci = final_stats[k]
            x_vals.append(float(k) if k is not None else 0.0)
            y_means.append(mu)
            y_low.append(lci)
            y_high.append(hci)

        # Plot with a function from plot_utils
        plot_utils.plot_clusters_vs_parameter(
            x_vals, y_means, y_low, y_high,
            param_name=args.group_by_param,
            outfile="clusters_vs_param.png"
        )

    print("Done! Final stats by group value:", final_stats)

if __name__ == "__main__":
    main()
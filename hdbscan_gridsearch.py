#!/usr/bin/env python3
"""
hdbscan_gridsearch.py

Perform a grid search over HDBSCAN hyperparameters (e.g., min_cluster_size, min_samples)
FOR EACH UNIQUE SET OF ABM PARAMETERS, to find those that maximize
an internal clustering metric (e.g., Silhouette).

Additionally:
- We record how many clusters were detected for each (mcs, ms).
- We identify which (mcs, ms) combos yield NaN silhouette scores, and print them.

Uses multiprocessing.Pool + tqdm for parallelization & progress bars.
"""

import os
import math
import json
import numpy as np
import hdbscan
from multiprocessing import Pool
from tqdm import tqdm
from typing import List, Tuple, Dict
from statistics import mean, stdev
from math import sqrt

from sklearn.metrics import silhouette_score
from pairwise_similarity import load_final_moving_avg, build_distance_matrix


###############################################################################
# 1. Clustering & Silhouette Helpers
###############################################################################
def compute_silhouette_for_run(dist_matrix: np.ndarray,
                               min_cluster_size: int,
                               min_samples: int) -> Tuple[float, int]:
    """
    Given a precomputed NxN distance matrix, run HDBSCAN(metric='precomputed')
    and compute:
      - silhouette score (ignoring noise points, label == -1),
      - number of clusters (excluding noise).

    Returns (sil_score, n_clusters). If there's only one (or zero) valid
    clusters, the silhouette score is NaN.
    """
    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )
    labels = clusterer.fit_predict(dist_matrix)

    # Count clusters (excluding label = -1)
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    n_clusters = len(unique_labels)

    # If zero or one cluster => silhouette is undefined
    mask = (labels != -1)
    if n_clusters < 2:
        return float("nan"), n_clusters

    # Otherwise, compute silhouette on the subset that isn't noise
    subset_dist = dist_matrix[mask][:, mask]
    subset_labels = labels[mask]
    sil = silhouette_score(subset_dist, subset_labels, metric="precomputed")
    return sil, n_clusters


def confidence_interval(data, confidence=0.90):
    """
    Basic normal-based confidence interval. For a more robust approach,
    consider bootstrapping. This is just a simple template.
    """
    if len(data) <= 1:
        return (float("nan"), float("nan"))
    m = mean(data)
    s = stdev(data)
    z = 1.645  # ~90% z-score
    half_width = z * (s / sqrt(len(data)))
    return (m - half_width, m + half_width)

###############################################################################
# 2. Worker Function
###############################################################################
def process_gridsearch_task(args) -> Tuple[str, str, Dict[Tuple[int, int], Dict[str, float]]]:
    """
    A worker function for parallel processing of a single .json file
    that belongs to one ABM parameter set.

    :param args: (param_key, json_file, input_dir, method, temperature, param_grid)
    :return:
        (param_key, json_file, { (mcs, ms): {'sil': float, 'n_clusters': int} })

    For each (mcs, ms), we'll record:
        'sil' -> silhouette score (possibly NaN)
        'n_clusters' -> how many clusters (excluding noise)
    """
    param_key, json_file, input_dir, method, temperature, param_grid = args
    fullpath = os.path.join(input_dir, json_file)

    fmavg = load_final_moving_avg(fullpath)
    if fmavg is None:
        # Return empty results to indicate missing data
        return (param_key, json_file, {})

    dist_mat, _ = build_distance_matrix(fmavg, method=method, temperature=temperature)

    results = {}
    for (mcs, ms) in param_grid:
        sil, n_clusters = compute_silhouette_for_run(dist_mat, mcs, ms)
        results[(mcs, ms)] = {
            'sil': sil,
            'n_clusters': n_clusters
        }

    return (param_key, json_file, results)

###############################################################################
# 3. Utility: get_param_key
###############################################################################
def get_param_key(data):
    """
    Given a JSON's data with data["params"], build a unique key for the ABM params
    that we care about. E.g., ignore repetition or derived fields.

    Example approach:
      - param_key = tuple of sorted((key, value) for key, value in data["params"].items())
      - Omit 'rep' if it exists, or any fields that are not truly part of the ABM parameter set.
    """
    abm_params = data["params"]
    # Exclude anything you consider "non-ABM" or "repetition" parameters:
    # e.g. "rep", "seed", etc. if they exist.
    exclude_keys = {"rep"}  # Add more if needed
    filtered = {k: v for k, v in abm_params.items() if k not in exclude_keys}
    # Build a sorted tuple so it's hashable & consistent
    param_key = tuple(sorted(filtered.items()))
    return param_key

###############################################################################
# 4. Main Script
###############################################################################
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory with .json files")
    parser.add_argument("--method", default="jsd",
                        help="Distance metric: jsd, euclidean, or cosine")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of parallel processes to use")
    args = parser.parse_args()

    # Define your search space:
    min_cluster_sizes = [20,30,40,50,60,70,80,90,100]
    min_samples_vals  = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    param_grid = [(mcs, ms) for mcs in min_cluster_sizes for ms in min_samples_vals]

    # 1. Gather all .json in input_dir & group by ABM param_key
    file_list = [f for f in os.listdir(args.input_dir) if f.endswith(".json")]
    file_list = sorted(file_list)

    # Build a mapping: param_key -> list of json_filenames
    param_groups = {}
    for fname in file_list:
        fpath = os.path.join(args.input_dir, fname)
        with open(fpath, "r") as f:
            data = json.load(f)
        if "params" not in data:
            continue

        pk = get_param_key(data)  # a sorted tuple of (k,v)
        param_groups.setdefault(pk, []).append(fname)

    # 2. Build a single list of tasks for parallel processing
    tasks = []
    for pk, fnames in param_groups.items():
        for f_ in fnames:
            tasks.append((pk, f_, args.input_dir, args.method, args.temperature, param_grid))

    # 3. Parallel execution
    results_for_all_files = []
    print(f"Running HDBSCAN grid search for each of {len(param_groups)} unique ABM param sets, "
          f"spread across {len(tasks)} total JSON files. Using {args.n_jobs} processes.")

    with Pool(processes=args.n_jobs) as pool:
        with tqdm(total=len(tasks), desc="Grid Search") as pbar:
            for res in pool.imap_unordered(process_gridsearch_task, tasks):
                results_for_all_files.append(res)
                pbar.update(1)

    # 4. Aggregate results by param_key
    #    aggregated[param_key][(mcs, ms)] -> list of dicts: {'sil': float, 'n_clusters': int}
    aggregated = {}
    for (param_key, json_file, param_sil_dict) in results_for_all_files:
        if not param_sil_dict:
            continue
        if param_key not in aggregated:
            aggregated[param_key] = {}
        for (mcs, ms), info_ in param_sil_dict.items():
            aggregated[param_key].setdefault((mcs, ms), []).append(info_)

    # 5. For each param_key, find best hyperparams
    #    We'll also keep track of how many runs had NaN, average cluster counts, etc.
    final_results = {}  
    for pk, grid_dict in aggregated.items():
        summary = {}
        best_key = None
        best_mean_sil = float("-inf")

        for (mcs_ms), info_list in grid_dict.items():
            # info_list is a list of dicts, each = { 'sil': float, 'n_clusters': int }
            sil_values = [i['sil'] for i in info_list]
            n_clusters_list = [i['n_clusters'] for i in info_list]

            # Filter out NaNs for the mean silhouette
            valid_sil = [v for v in sil_values if not math.isnan(v)]
            if len(valid_sil) > 0:
                mu_sil = mean(valid_sil)
            else:
                mu_sil = float("nan")

            # Also compute how many times it was NaN vs total
            num_runs = len(sil_values)
            num_nan = sum(1 for v in sil_values if math.isnan(v))

            # Compute average #clusters
            mu_clusters = mean(n_clusters_list) if len(n_clusters_list) else 0.0
            (low_ci, high_ci) = confidence_interval(valid_sil, confidence=0.90)

            summary[mcs_ms] = {
                "mean_sil": mu_sil,
                "num_runs": num_runs,
                "num_nan": num_nan,
                "ci_low": low_ci,
                "ci_high": high_ci,
                "mean_clusters": mu_clusters
            }

            # Track best by highest mean silhouette
            if not math.isnan(mu_sil) and mu_sil > best_mean_sil:
                best_mean_sil = mu_sil
                best_key = mcs_ms

        final_results[pk] = {
            "best": best_key,
            "best_mean": best_mean_sil,
            "grid": summary
        }

    # 6. Print or save the final results
    for pk, info in final_results.items():
        param_str = ", ".join(f"{k}={v}" for k, v in pk)
        print(f"\n=== ABM Parameter Set: {param_str} ===")

        best_mcs_ms = info["best"]
        best_m = info["best_mean"]

        if best_mcs_ms is not None and not math.isnan(best_m):
            print(f"  Best hyperparams => min_cluster_size={best_mcs_ms[0]}, "
                  f"min_samples={best_mcs_ms[1]}, silhouette={best_m:.4f}")
        else:
            print("  No valid clusters found for this parameter set (all silhouette scores were NaN).")

        # Optionally print a table for each (mcs, ms) to see how many NaNs, etc.
        grid_info = info["grid"]
        for (mcs_ms), stats_ in grid_info.items():
            mcs, ms = mcs_ms
            if math.isnan(stats_["mean_sil"]):
                # Show combos that produce NaN silhouette
                print(f"    (mcs={mcs}, ms={ms}): All runs had NaN silhouette. "
                      f"Avg #clusters={stats_['mean_clusters']:.1f}, #runs={stats_['num_runs']}, #NaN={stats_['num_nan']}")
            else:
                # Show combos that partially or fully produce non-NaN
                mean_sil = stats_["mean_sil"]
                n_runs = stats_["num_runs"]
                n_nan = stats_["num_nan"]
                mean_clusters = stats_["mean_clusters"]
                print(f"    (mcs={mcs}, ms={ms}): mean_sil={mean_sil:.4f}, #clusters={mean_clusters:.1f}, "
                      f"#runs={n_runs}, #NaN={n_nan}")

    print("\nGrid search complete for all unique param sets.")

if __name__ == "__main__":
    main()
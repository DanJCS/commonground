#!/usr/bin/env python3
"""
File: pairwise_similarity.py

Summary:
    Computes pairwise distance (or similarity) matrices among agents (nodes) based on their
    final state vectors, which are converted to probability distributions via softmax. The
    script supports multiple distance metrics (Jensen-Shannon Divergence, Euclidean, 1 - Cosine)
    and can process multiple simulation JSON files in parallel.

Key Functions:
    * load_final_moving_avg(json_file: str) -> dict:
        Loads a JSON file and returns the 'final_moving_avg' data as a dictionary
        { agent_name -> np.array([...]) }. Returns None if missing.

    * softmax(x, temperature=1.0, epsilon=1e-15) -> np.ndarray:
        Converts an array of values into a probability distribution
        (optionally temperature-scaled).

    * build_distance_matrix(final_moving_avg, method="jsd", temperature=1.0) -> (np.ndarray, list):
        Given the final_moving_avg dict, first applies softmax to each agent’s vector,
        then computes an NxN pairwise distance matrix using the specified method:
          - "jsd" for Jensen-Shannon Divergence
          - "euclidean" for Euclidean distance
          - "cosine" for 1 - Cosine similarity
        Returns (distance_matrix, sorted_agent_names).

    * process_single_file(args) -> (str, np.ndarray, list):
        A parallelizable worker function that:
          1) Loads a single JSON result.
          2) Builds the NxN distance matrix using build_distance_matrix.
          3) Optionally saves the matrix as a .npy file in the output directory.
        Returns the filename, distance matrix, and agent name list.

    * precompute_distance_for_directory(input_dir, method="jsd", temperature=1.0,
                                        output_dir=None, n_jobs=1) -> list:
        Iterates over all .json files in input_dir, computing and optionally saving
        NxN distance matrices in parallel. Returns a list of (filename, matrix, agent_names)
        for all processed files.

Output Format:
    - If 'output_dir' is provided, each computed NxN distance matrix is saved as a .npy file
      (e.g., my_sim_result_jsd_dist.npy).
    - The script also returns a Python list of tuples:
        (json_filename, distance_matrix, agent_names)
      for each JSON file it processes.

Dependencies:
    * Python built-ins: os, json, math
    * Third-party: numpy, multiprocessing, tqdm
    * It is commonly used after simulations to transform agents' final states into
      pairwise distance matrices for further clustering or analysis.

Usage:
    # Command-line usage example:
    python pairwise_similarity.py <input_dir> [--method jsd] [--temperature 1.0] [--output_dir ./dist_mats] [--n_jobs 4]

    # This will scan <input_dir> for .json files, compute NxN distance matrices using
    # the chosen metric, save them in ./dist_mats (if provided), and output relevant info.
"""


import os
import json
import math
import numpy as np
from typing import Dict, Tuple, List
from multiprocessing import Pool
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon

###############################################################################
# 1. Loading final_moving_avg from JSON
###############################################################################
def load_final_moving_avg(json_file: str) -> Dict[str, np.ndarray]:
    """
    Load a single JSON file (one simulation result) and extract 'final_moving_avg'.

    Returns a dict: { agent_name: np.array([...]) }
    or None if JSON is missing the required fields.
    """
    if not os.path.exists(json_file):
        print(f"[Warning] File not found: {json_file}")
        return None

    with open(json_file, "r") as f:
        data = json.load(f)

    if "final_moving_avg" not in data:
        print(f"[Warning] 'final_moving_avg' missing in {json_file}")
        return None

    fmavg = {}
    for agent_name, vec in data["final_moving_avg"].items():
        fmavg[agent_name] = np.array(vec, dtype=float)
    return fmavg

###############################################################################
# 2. Softmax utility (with temperature)
###############################################################################
def softmax(x, temperature=0.25, epsilon=1e-15):
    """
    Compute the softmax of vector x with temperature T.

    If x is all zeros, we return a uniform distribution.
    
    (T = 1/beta)
    """
    if temperature <= 0:
        raise ValueError("Temperature must be > 0.")

    scaled_x = x / temperature
    max_val = np.max(scaled_x)
    exps = np.exp(scaled_x - max_val)
    sum_exps = np.sum(exps)

    if sum_exps < epsilon:
        return np.ones_like(x) / len(x)
    else:
        return exps / sum_exps

###############################################################################
# 3. Distance/Similarity metrics on probability vectors
###############################################################################
def _kl_divergence(p, q, epsilon=1e-10):
    # Only clip the lower bound so we don't take log of zero
    p_safe = np.clip(p, epsilon, None)
    q_safe = np.clip(q, epsilon, None)

    # *Optionally* renormalize if you want to be extra safe:
    p_safe /= p_safe.sum()
    q_safe /= q_safe.sum()

    return np.sum(p_safe * np.log(p_safe / q_safe))

def js_divergence(p, q, epsilon=1e-10):
    """Jensen–Shannon Divergence."""
    m = 0.5 * (p + q)
    val = 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)
    
    if val < 0:
        print(f"WARNING: Negative value encountered in JSD calculation: {val}")

    val = max(val, 0.0)
    return math.sqrt(val)

def euclidean_distance(p, q):
    """Euclidean distance."""
    return np.linalg.norm(p - q)

def one_minus_cosine(p, q, epsilon=1e-15):
    """
    1 - cosine_similarity -> a proper distance (lower => more similar).
    """
    p_norm = np.linalg.norm(p) + epsilon
    q_norm = np.linalg.norm(q) + epsilon
    cos_sim = np.dot(p, q) / (p_norm * q_norm)
    return 1.0 - cos_sim

###############################################################################
# 4. Build NxN distance matrix
###############################################################################
def build_distance_matrix(final_moving_avg, method="jsd", temperature=1.0):
    """
    Given final_moving_avg: { agent_name: np.array([...]) },
    return an NxN matrix of distances among the agents.

    Steps:
      1. Sort agent names so the matrix is consistent.
      2. Convert each agent's vector into a probability distribution
         (softmax) if method in {jsd, euclidean, cosine} (assuming that’s desired).
      3. Fill NxN using the chosen method of distance.
    """
    agent_names = sorted(final_moving_avg.keys())
    n = len(agent_names)

    # Step 1: compute probability distributions (softmax)
    prob_distributions = {}
    for name in agent_names:
        vec = final_moving_avg[name]
        prob_distributions[name] = softmax(vec, temperature=temperature)

    # Step 2: fill NxN
    dist_mat = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            p_i = prob_distributions[agent_names[i]]
            p_j = prob_distributions[agent_names[j]]

            if method.lower() == "jsd":
                dist = js_divergence(p_i, p_j)
            elif method.lower() == "euclidean":
                dist = euclidean_distance(p_i, p_j)
            elif method.lower() == "cosine":
                dist = one_minus_cosine(p_i, p_j)
            else:
                raise ValueError(f"Unknown method {method}")

            dist_mat[i, j] = dist
            dist_mat[j, i] = dist

    return dist_mat, agent_names

###############################################################################
# 5. Function to process a single JSON file (for parallel usage)
###############################################################################
def process_single_file(args):
    """
    Worker function for parallel distance computation.
    :param args: (json_filename, input_dir, method, temperature, output_dir)
    :return: (json_filename, distance_matrix, agent_names)
    """
    json_filename, input_dir, method, temperature, output_dir = args
    fullpath = os.path.join(input_dir, json_filename)

    fmavg = load_final_moving_avg(fullpath)
    if fmavg is None:
        # Return (None, None, None) or something similar to indicate failure
        return (json_filename, None, None)

    dist_mat, agents = build_distance_matrix(fmavg, method=method, temperature=temperature)

    # Optionally save .npy
    if output_dir:
        base = os.path.splitext(json_filename)[0]
        npy_path = os.path.join(output_dir, f"{base}_{method}_dist.npy")
        np.save(npy_path, dist_mat)

    return (json_filename, dist_mat, agents)

###############################################################################
# 6. Parallel routine for a directory of JSONs (using multiprocessing + tqdm)
###############################################################################
def precompute_distance_for_directory(
    input_dir: str,
    method="jsd",
    temperature=1.0,
    output_dir=None,
    n_jobs=1
):
    """
    For each .json in input_dir, compute the distance matrix and optionally save
    it as a .npy. Parallelize across files with multiprocessing.

    :param input_dir: path to directory of JSON results
    :param method: 'jsd', 'euclidean', or 'cosine'
    :param temperature: float
    :param output_dir: where to save .npy, if None, skip saving
    :param n_jobs: number of parallel processes to use
    :return: list of (json_filename, distance_matrix, agent_names)
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    file_list = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    file_list = sorted(file_list)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    tasks = []
    for json_filename in file_list:
        tasks.append((json_filename, input_dir, method, temperature, output_dir))

    results = []
    print(f"Computing Distances for {len(tasks)} JSON files using {n_jobs} processes.")
    with Pool(processes=n_jobs) as pool:
        # We'll mimic the approach in run_parallel_simulation.py
        with tqdm(total=len(tasks), desc="Computing Distances") as pbar:
            for res in pool.imap_unordered(process_single_file, tasks):
                results.append(res)
                pbar.update(1)

    return results

###############################################################################
# 7. Main CLI usage
###############################################################################
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory with .json files")
    parser.add_argument("--method", default="jsd", help="jsd, euclidean, or cosine")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel processes")
    args = parser.parse_args()

    precompute_distance_for_directory(
        args.input_dir,
        method=args.method,
        temperature=args.temperature,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs
    )
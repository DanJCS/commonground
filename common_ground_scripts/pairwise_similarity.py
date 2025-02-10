#!/usr/bin/env python3
"""
File: pairwise_similarity.py

Summary:
    Computes pairwise distance (or similarity) matrices among agents (nodes) based on their
    final state vectors. Depending on the chosen metric, the state vectors may be first converted
    into probability distributions via softmax. Supported metrics include:
      - Jensen-Shannon Divergence ("jsd")
      - Euclidean distance ("euclidean")
      - 1 - Cosine similarity ("cosine")
      - Direct normalized Euclidean similarity ("euclidean_direct_normalised")
        <-- This new metric operates directly on raw state vectors without softmax conversion.
        
    The script can process multiple simulation JSON files in parallel.
    
Key Functions:
    * load_final_moving_avg(json_file: str) -> dict:
        Loads a JSON file and returns the 'final_moving_avg' data as a dictionary
        { agent_name -> np.array([...]) }. Returns None if missing.
        
    * softmax(x, temperature=1.0, epsilon=1e-15) -> np.ndarray:
        Converts an array of values into a probability distribution (optionally temperature-scaled).
        
    * build_distance_matrix(final_moving_avg, method="jsd", temperature=1.0) -> (np.ndarray, list):
        Given the final_moving_avg dict, first either converts each agent’s vector to a probability distribution
        (using softmax) or, if method is "euclidean_direct_normalised", uses the raw vectors.
        Then computes an NxN pairwise matrix using the specified method.
        
    * process_single_file(args) -> (str, np.ndarray, list):
        A parallelizable worker function that:
          1) Loads a single JSON result.
          2) Builds the NxN distance matrix using build_distance_matrix.
          3) Optionally saves the matrix as a .npy file in the output directory.
        Returns the filename, distance matrix, and agent name list.
        
    * precompute_distance_for_directory(...):
        Iterates over all .json files in the given directory and computes (and optionally saves)
        the NxN distance matrices in parallel.
        
Usage Example:
    python pairwise_similarity.py <input_dir> [--method jsd] [--temperature 1.0] [--output_dir ./dist_mats] [--n_jobs 4]
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
    x = np.array(x, dtype=float)
    scaled_x = x / temperature
    max_val = np.max(scaled_x)
    exps = np.exp(scaled_x - max_val)
    return exps / np.sum(exps)

###############################################################################
# 3. Distance/Similarity metrics on probability vectors
###############################################################################
def _kl_divergence(p, q, epsilon=1e-10):
    p_safe = np.clip(p, epsilon, None)
    q_safe = np.clip(q, epsilon, None)
    p_safe /= p_safe.sum()
    q_safe /= q_safe.sum()
    return np.sum(p_safe * np.log(p_safe / q_safe))

def js_divergence(p, q, epsilon=1e-10):
    """Jensen–Shannon Divergence."""
    m = 0.5 * (p + q)
    val = 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)
    if val < 0:
        print(f"WARNING: Negative value encountered in JSD calculation: {val}")
    return math.sqrt(max(val, 0.0))

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

def _metric_euclidean_direct_normalised(u, v):
    """
    Computes the direct Euclidean distance between two agents' state vectors,
    normalizes it by dividing by 2*m, and returns the similarity as 1 - normalized distance.
    """
    d = np.linalg.norm(u.state_vector - v.state_vector)
    m = len(u.state_vector)
    d_norm = d / (2 * m)
    similarity = 1 - d_norm
    return similarity

# Dictionary to hold metric functions.
try:
    _METRICS
except NameError:
    _METRICS = {}
_METRICS["euclidean_direct_normalised"] = _metric_euclidean_direct_normalised

###############################################################################
# 4. Build NxN distance matrix
###############################################################################
def build_distance_matrix(final_moving_avg, method="jsd", temperature=1.0):
    """
    Given final_moving_avg: { agent_name: np.array([...]) },
    return an NxN matrix of distances (or similarities) among the agents.
    
    For most methods, each agent's state vector is first converted to a probability distribution
    via softmax. However, if the chosen method is "euclidean_direct_normalised", raw state vectors
    are used directly.
    """
    agent_names = sorted(final_moving_avg.keys())
    n = len(agent_names)
    
    # Determine whether to use raw state vectors or softmax probabilities.
    if method.lower() == "euclidean_direct_normalised":
        vectors = {}
        for name in agent_names:
            vec = final_moving_avg[name]
            try:
                vec = np.array(vec, dtype=float)
            except Exception as e:
                raise ValueError(f"Cannot convert vector for agent '{name}' to a numpy array: {e}")
            vectors[name] = vec
    else:
        prob_distributions = {}
        for name in agent_names:
            vec = final_moving_avg[name]
            try:
                vec = np.array(vec, dtype=float)
            except Exception as e:
                raise ValueError(f"Cannot convert vector for agent '{name}' to a numpy array: {e}")
            prob_distributions[name] = softmax(vec, temperature=temperature)
    
    # Initialize the distance matrix.
    dist_mat = np.zeros((n, n), dtype=float)
    
    # Compute pairwise distances/similarities.
    for i in range(n):
        for j in range(i + 1, n):
            if method.lower() == "euclidean_direct_normalised":
                p_i = vectors[agent_names[i]]
                p_j = vectors[agent_names[j]]
                # Compute normalized Euclidean similarity directly.
                d = np.linalg.norm(p_i - p_j)
                m_val = len(p_i)
                d_norm = d / (2 * m_val)
                sim = 1 - d_norm
                dist = sim
            else:
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
        return (json_filename, None, None)

    dist_mat, agents = build_distance_matrix(fmavg, method=method, temperature=temperature)

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
    For each .json in input_dir, compute the distance matrix and optionally save it as a .npy file.
    Parallelizes the computation using multiprocessing.
    
    :param input_dir: Directory containing JSON result files.
    :param method: 'jsd', 'euclidean', 'cosine', or 'euclidean_direct_normalised'
    :param temperature: Temperature parameter for softmax conversion.
    :param output_dir: Directory to save .npy files (if None, skip saving).
    :param n_jobs: Number of parallel processes.
    :return: List of tuples (json_filename, distance_matrix, agent_names) for each JSON file.
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
    print(f"Computing distances for {len(tasks)} JSON files using {n_jobs} processes.")
    with Pool(processes=n_jobs) as pool:
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
    parser.add_argument("input_dir", help="Directory with JSON files")
    parser.add_argument("--method", default="jsd", help="Distance metric: jsd, euclidean, cosine, or euclidean_direct_normalised")
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
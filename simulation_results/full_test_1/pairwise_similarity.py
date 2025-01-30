#!/usr/bin/env python3

import os
import json
import numpy as np
import math

###############################################################################
# 1. Loading final_moving_avg from JSON
###############################################################################
def load_final_moving_avg(json_file):
    """
    Load a JSON file (one simulation result) and extract the 'final_moving_avg'.

    Expects the JSON to have a structure like:
        {
          "params": {...},
          "final_moving_avg": {
            "Agent_0": [...],
            "Agent_1": [...],
            ...
          }
          ... possibly other keys ...
        }
    Returns a dict: { agent_name: np.array([...]), ... }
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
def softmax(x, temperature=1.0, epsilon=1e-15):
    """
    Compute the softmax of vector x with temperature T = temperature.

    If x is all zeros (extremely unlikely, but let's handle it anyway),
    we return a uniform distribution of the same length.
    """
    # Avoid dividing by zero or negative temperature
    if temperature <= 0:
        raise ValueError("Temperature must be > 0.")

    # Exponentiate (x / T) in a numerically stable way
    scaled_x = x / temperature
    # Shift by max for numerical stability
    max_val = np.max(scaled_x)
    exps = np.exp(scaled_x - max_val)
    sum_exps = np.sum(exps)

    if sum_exps < epsilon:
        # If sum of exps is tiny, fallback to uniform
        return np.ones_like(x) / len(x)
    else:
        return exps / sum_exps

###############################################################################
# 3. Distance/Similarity metrics on probability vectors
###############################################################################
def _kl_divergence(p, q, epsilon=1e-15):
    """
    Kullback–Leibler divergence KL(p || q), with small-epsilon smoothing
    to avoid log(0). Here p and q are probability distributions.
    KL is sum( p[i] * log( p[i]/q[i] ) ), ignoring terms where p[i] = 0.
    """
    # Smooth p and q by epsilon to avoid division by zero or log(0)
    p_safe = np.clip(p, epsilon, 1.0)
    q_safe = np.clip(q, epsilon, 1.0)

    return np.sum(p_safe * np.log(p_safe / q_safe))

def js_divergence(p, q, epsilon=1e-15):
    """
    Compute the Jensen–Shannon Divergence between probability distributions p and q.
    JSD(p,q) = 0.5*KL(p || m) + 0.5*KL(q || m),  m = 0.5*(p+q).
    We often take the square root to get a distance metric, but we’ll
    return the raw divergence here for clarity.
    """
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m, epsilon=epsilon) + 0.5 * _kl_divergence(q, m, epsilon=epsilon)

def euclidean_distance(p, q):
    """Euclidean distance between two vectors p and q."""
    return np.linalg.norm(p - q)

def cosine_similarity(p, q, epsilon=1e-15):
    """
    Cosine similarity between two vectors p and q, in [−1,1].
    If any vector is near zero-length, we add epsilon to avoid div-by-zero.
    """
    p_norm = np.linalg.norm(p) + epsilon
    q_norm = np.linalg.norm(q) + epsilon
    return np.dot(p, q) / (p_norm * q_norm)

###############################################################################
# 4. Main function to compute pairwise “distance/similarity”
###############################################################################
def compute_pairwise_similarity(
    final_moving_avg,
    method="jsd",
    temperature=1.0
):
    """
    Given a dict: { agent_name: np.array([...]), ... }, compute pairwise
    (distance or similarity) for all unique pairs.

    Steps:
      1. Convert each agent's final_moving_avg into a probability distribution
         using softmax with the specified temperature.
      2. For each unique pair (i, j) of agents, compute:
         - Jensen–Shannon Divergence (default),
         - or Euclidean distance,
         - or Cosine similarity,
         all on the probability distributions.

    Returns:
      A dictionary keyed by (agent_i, agent_j) -> float
      or possibly an NxN matrix if you prefer. Here we’ll return a dict
      for easy parsing in other scripts.
    """
    agent_names = sorted(final_moving_avg.keys())
    # Step 1: softmax each agent’s vector
    prob_distributions = {}
    for name in agent_names:
        vec = final_moving_avg[name]
        prob_distributions[name] = softmax(vec, temperature=temperature)

    # Step 2: compute pairwise results
    pairwise_results = {}
    n = len(agent_names)
    for i in range(n):
        for j in range(i + 1, n):
            a_i = agent_names[i]
            a_j = agent_names[j]
            p_i = prob_distributions[a_i]
            p_j = prob_distributions[a_j]

            if method.lower() == "jsd":
                val = js_divergence(p_i, p_j)
            elif method.lower() == "euclidean":
                val = euclidean_distance(p_i, p_j)
            elif method.lower() == "cosine":
                val = cosine_similarity(p_i, p_j)
            else:
                raise ValueError(f"Unknown method '{method}'. Use 'jsd', 'euclidean', or 'cosine'.")

            # Store in dict, using a tuple of agent names as key
            pairwise_results[(a_i, a_j)] = val

    return pairwise_results

###############################################################################
# 5. Example usage or “if __name__ == '__main__':” if you want a CLI
###############################################################################
if __name__ == "__main__":
    # This is just an illustrative usage example.

    # Suppose we have a single JSON file:
    example_json = "results_20250114_212518_rep2.json"
    fmavg = load_final_moving_avg(example_json)
    if fmavg is None:
        print("No final_moving_avg found. Exiting.")
        exit(1)

    # Compute pairwise similarities using default (Jensen–Shannon)
    jsd_results = compute_pairwise_similarity(fmavg, method="jsd", temperature=1.0)

    # Print or do something with the results
    for (agent_i, agent_j), divergence_value in jsd_results.items():
        print(f"JSD({agent_i}, {agent_j}) = {divergence_value:.6f}")

    # If you want to do Euclidean or Cosine instead:
    # euc_results = compute_pairwise_similarity(fmavg, method="euclidean", temperature=1.0)
    # cos_results = compute_pairwise_similarity(fmavg, method="cosine", temperature=1.0)
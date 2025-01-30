
import os
import json
import argparse

def load_simulation_results(results_dir):
    """
    Load all .json simulation result files from `results_dir`.
    Returns a list of dictionaries (each one is the loaded JSON).
    """
    all_results = []
    for filename in os.listdir(results_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, "r") as f:
                data = json.load(f)
                all_results.append(data)
    return all_results

def make_param_key(params_dict):
    """
    Create a 'key' from the params dictionary, excluding fields 
    that are not relevant to grouping (e.g., 'graph_file').
    We sort items so the key is consistent even if dict order changes.

    Returns a tuple of (key, value) pairs, suitable as a dictionary key.
    """
    # Exclude anything you don’t want in the param grouping
    EXCLUDE_KEYS = {"graph_file"}
    # If some results store "G" or other objects in 'params', exclude them too if needed

    filtered = {k: v for k, v in params_dict.items() if k not in EXCLUDE_KEYS}
    # Sort by key name so the tuple is consistent
    return tuple(sorted(filtered.items()))

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

    # For each piece of info i, count how many agents have final_moving_avg[i] > threshold
    survived_count = 0
    cutoff_count = int(min_fraction * num_agents)

    for i in range(m):
        above_threshold = 0
        for agent in agent_names:
            if final_moving_avg[agent][i] > survival_threshold:
                above_threshold += 1

        if above_threshold >= cutoff_count:
            survived_count += 1

    return survived_count

def analyze_survival_across_results(
    results_dir, 
    survival_threshold=0.5, 
    min_fraction=0.1
):
    """
    1. Loads all .json files in `results_dir`
    2. Groups results by unique parameter sets
    3. For each result, counts how many pieces of information survive
    4. Accumulates these survival counts in a list under each parameter set

    Returns a dict: { param_key: [list_of_survival_counts_for_each_rep] }
    """
    all_data = load_simulation_results(results_dir)
    param_to_survival = {}

    for entry in all_data:
        params = entry["params"]
        param_key = make_param_key(params)

        final_moving_avg = entry["final_moving_avg"]
        survived_info_count = count_surviving_info(
            final_moving_avg, 
            survival_threshold=survival_threshold, 
            min_fraction=min_fraction
        )

        # Insert into param_to_survival
        if param_key not in param_to_survival:
            param_to_survival[param_key] = []
        param_to_survival[param_key].append(survived_info_count)

    return param_to_survival

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Analyze survival across results.")
    parser.add_argument(
        "results_dir",
        type=str,
        help="Directory containing .json result files."
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

    # Analyze survival
    survival_data = analyze_survival_across_results(
        results_dir=args.results_dir,
        survival_threshold=args.threshold,
        min_fraction=args.fraction
    )

    # Sort the parameter sets by (m, eps, gamma) in ascending order
    def sort_key(pk):
        param_dict = dict(pk)
        return (
            param_dict.get("m", float("inf")),
            param_dict.get("eps", float("inf")),
            param_dict.get("gamma", float("inf"))
        )

    sorted_param_keys = sorted(survival_data.keys(), key=sort_key)

    # Print out the aggregated results in sorted order
    for param_key in sorted_param_keys:
        param_dict = dict(param_key)
        survival_counts = survival_data[param_key]
        average_surviving = (
            sum(survival_counts) / len(survival_counts)
            if survival_counts else 0
        )
        print("\nParameter Set:", param_dict)
        print("Number of Surviving Pieces of Information per run:", survival_counts)
        print("Average Surviving:", average_surviving)
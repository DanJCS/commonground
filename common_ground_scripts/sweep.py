#!/usr/bin/env python3
"""
File: sweep.py

Summary:
    Runs a parameter sweep based on the parameter grid defined in config.py.
    For each combination of parameters (only the swept ones appear in the filename),
    the script runs the simulation (using natural_simulation.run_simulation_with_params)
    and saves the result as a JSON file in the designated output directory.
    
    Usage:
        python3 sweep.py --reps 5 --output_dir sweep_results --n_jobs 4
"""

import os
import json
import argparse
from itertools import product
from datetime import datetime
from multiprocessing import Pool
from tqdm import tqdm

# Import constant simulation parameters and the parameter grid from config.py.
from config import SIM_PARAMS, PARAMETER_GRID, resolve_param
# Import the simulation run function.
from natural_simulation import run_simulation_with_params

def get_swept_keys(param_grid):
    """
    Returns a set of parameter keys that are being swept (i.e., those with more than one value).
    """
    return {key for key, values in param_grid.items() if len(values) > 1}

def generate_output_filename(combination_dict, rep, output_dir):
    """
    Generate an output filename based on the swept parameters and the repetition index.
    Only include parameters whose values vary (i.e. those with more than one value in the grid).
    
    Example filename:
       eps0.5_gamma1_rep0.json
    """
    swept_keys = get_swept_keys(PARAMETER_GRID)
    parts = []
    # Sort the keys for consistent ordering.
    for key in sorted(combination_dict.keys()):
        if key in swept_keys:
            parts.append(f"{key}{combination_dict[key]}")
    param_str = "_".join(parts)
    filename = f"{param_str}_rep{rep}.json"
    return os.path.join(output_dir, filename)

def run_single_simulation(args_tuple):
    """
    Worker function for a single simulation run.
    
    Args:
        args_tuple: (combination_dict, rep)
        
    Returns:
        (combination_dict, rep, simulation_result)
    """
    combination_dict, rep = args_tuple
    # Merge constant parameters from SIM_PARAMS with the current swept combination.
    params = SIM_PARAMS.copy()
    params.update(combination_dict)
    sim_result = run_simulation_with_params(params, rep)
    return (combination_dict, rep, sim_result)

def main():
    parser = argparse.ArgumentParser(
        description="Run a parameter sweep simulation based on the parameter grid in config.py."
    )
    parser.add_argument("--reps", type=int, default=1,
                        help="Number of repetitions per parameter combination (default: 1)")
    parser.add_argument("--output_dir", type=str, default="sweep_results",
                        help="Directory to save simulation JSON results (default: sweep_results)")
    parser.add_argument("--n_jobs", type=int, default=4,
                        help="Number of parallel processes to use (default: 4)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build list of all parameter combinations from PARAMETER_GRID.
    keys = list(PARAMETER_GRID.keys())
    grid_values = [PARAMETER_GRID[key] for key in keys]
    # Each combination is a tuple of values corresponding to keys.
    combinations = list(product(*grid_values))
    # Convert each combination into a dictionary.
    combination_dicts = [dict(zip(keys, comb)) for comb in combinations]
    
    # Build tasks: for each parameter combination and each repetition.
    tasks = []
    for combination in combination_dicts:
        for rep in range(args.reps):
            tasks.append((combination, rep))
    
    total_tasks = len(tasks)
    print(f"Running parameter sweep with {total_tasks} tasks using {args.n_jobs} processes...")
    
    # Run the simulations in parallel.
    with Pool(processes=args.n_jobs) as pool:
        for combo_dict, rep, sim_result in tqdm(pool.imap_unordered(run_single_simulation, tasks),
                                                  total=total_tasks,
                                                  desc="Running simulations"):
            # Generate output filename based on the swept parameters and rep index.
            output_filename = generate_output_filename(combo_dict, rep, args.output_dir)
            with open(output_filename, "w") as f:
                json.dump(sim_result, f, indent=4)
    
    print(f"Parameter sweep complete. Results saved in directory: {args.output_dir}")

if __name__ == "__main__":
    main()

"""
File: run_parallel_simulation.py

Summary:
    Orchestrates parallel parameter sweeps for the ABM, distributing simulations across multiple CPU cores.
    Leverages 'run_simulation_with_params' from 'natural_simulation.py' to run each simulation.

Key Functions:
    * run_single_sim(arg): Wrapper for parallel execution.
    * parameter_sweep_parallel(parameter_grid, repetitions, output_dir="results", processes=None)
      - Generates tasks for each param combination x repetition, runs them in parallel.

Dependencies:
    * Python built-ins: os, json, math, random, itertools, datetime, argparse
    * Third-party: multiprocessing, tqdm
    * Internal: natural_simulation (run_simulation_with_params), CG_source

Usage:
    python run_parallel_simulation.py input_dir output_dir --processes 8 --repetitions 50

"""
import os
import json
import math
import random
import pickle
import networkx as nx
import numpy as np
import argparse

from itertools import product
from datetime import datetime
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from CG_source import *
from natural_simulation import run_simulation_with_params


def run_single_sim(arg):
    """
    Wrapper function for parallel execution.
    arg is a tuple: (params, rep, output_dir)
    """
    params, rep, output_dir = arg
    result = run_simulation_with_params(params, rep)

    # Save intermediate results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/results_{timestamp}_rep{rep}.json"
    with open(filename, "w") as f:
        json.dump(result, f, indent=4)
    return filename  # Return the filename or any other info if desired


def parameter_sweep_parallel(parameter_grid, repetitions, output_dir="results", processes=None):
    """
    Parallelized parameter sweep. Distributes simulation runs across multiple CPU cores.
    
    :param parameter_grid: Dictionary of parameter ranges to sweep.
    :param repetitions: Number of repetitions per parameter combination.
    :param output_dir: Directory to save the results.
    :param processes: Number of parallel processes to use (default: use all available).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a list of parameter-combination dictionaries
    param_combinations = list(product(*[[(key, value) for value in values] 
                                        for key, values in parameter_grid.items()]))
    
    # Prepare list of tasks (each repetition of each parameter set)
    tasks = []
    for param_tuple in param_combinations:
        params = {k: v for k, v in param_tuple}
        for rep in range(repetitions):
            tasks.append((params, rep, output_dir))
    
    # Default to using all cores if processes=None; otherwise use the user-specified number
    if processes is None:
        processes = 9  # e.g., 10 on your Apple M1 Pro

    print(f"Starting parallel sweep with {processes} processes ...")

    # Run tasks in parallel
    with Pool(processes=processes) as pool:
        with tqdm(total=len(tasks), desc="Running Simulations") as pbar:
            for _ in pool.imap_unordered(run_single_sim, tasks):
                pbar.update()

    print(f"Parameter sweep completed! Results saved to {output_dir}.")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run parallel simulations for parameter sweeps.")
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing input data (e.g., precomputed graphs)."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save the simulation results."
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Number of parallel processes to use (default: all available cores)."
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=50,
        help="Number of repetitions per parameter combination (default: 50)."
    )

    args = parser.parse_args()

    # Example parameter grid (this can also be loaded from input_dir if required)
    parameter_grid = {
        "n": [500],
        "m": [10],
        "timesteps": [5000],
        "bi": [7],
        "bj": [7],
        "a": [0.5],
        "alpha": [0.1],
        "eps": [0.1,0.2,0.3,0.4,0.5],
        "sigma_coeff": [10],
        "zeta": [0],
        "eta": [0.5],
        "gamma": [0],
        "metric_method": ["pdiff"],
        "alpha_dist": ["static"]
    }

    # Run the parameter sweep
    parameter_sweep_parallel(
        parameter_grid=parameter_grid,
        repetitions=args.repetitions,
        output_dir=args.output_dir,
        processes=args.processes
    )
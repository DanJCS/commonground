"""
File: natural_simulation.py

Summary:
    Provides the main simulation driver for the ABM.
    Includes functions to create agents, load random graphs, and run full simulations.

Key Functions:
    * create_agent_list(n, m, alpha_dist, a, alpha, bi, bj, eps)
    * load_random_graph(graph_dir)
    * run_simulation_with_params(params, repetition_index)
    * parameter_sweep(parameter_grid, repetitions, output_dir)

Dependencies:
    * CG_source (import Config, Agent, RecordBook, beta_extended)
    * Python built-ins: os, random, json, pickle
    * Third-party: numpy, networkx

Usage:
    Run directly: python natural_simulation.py
    or import its functions in another script:

    from natural_simulation import run_simulation_with_params
    results = run_simulation_with_params(params, 0)

"""
import math
import os
import random
import json
import pickle

import networkx as nx
import numpy as np

from itertools import product
from datetime import datetime
from CG_source import *

def create_agent_list(n, m, alpha_dist, a, alpha, bi, bj, eps):
    Agents = []
    info_list = [beta_extended(n) for _ in range(m)]  # Initialize state vectors
    if alpha_dist == "beta":
        alpha_list = np.random.beta(3.5, 14, n)
    elif alpha_dist == "static":
        alpha_list = [alpha] * n
    elif alpha_dist == "uniform":
        alpha_list = np.random.uniform(0, 1, n)
    for i in range(n):
        agent = Agent(
            f"Agent_{i}",
            a,
            alpha_list[i],
            bi,
            bj,
            eps,
            np.array([info_list[_][i] for _ in range(m)]),
            0,
        )
        Agents.append(agent)
    return Agents

def load_random_graph(graph_dir="graphs_library"):
    """
    Randomly selects one of the pickle files in `graph_dir` 
    and loads it as a NetworkX Graph.

    Returns:
        G: The loaded NetworkX graph.
        chosen_file: The filename of the selected graph.
    """
    # List all .pkl files in the directory
    file_list = [f for f in os.listdir(graph_dir) if f.endswith(".pkl")]
    if not file_list:
        raise FileNotFoundError(f"No .pkl files found in {graph_dir}")

    # Pick one file at random
    chosen_file = random.choice(file_list)
    path = os.path.join(graph_dir, chosen_file)

    # Load (unpickle) the NetworkX graph
    with open(path, "rb") as f:
        G = pickle.load(f)

    print(f"Loaded graph from: {chosen_file}")
    return G, chosen_file

def run_simulation_with_params(params, repetition_index):
    print(f"Running simulation with parameters: {params}, repetition: {repetition_index}")
    
    if "graph" not in params or params["graph"] is None:
        # Load a random graph from your library
        G, graph_file = load_random_graph("singular_graph")
    else:
        # Use the provided graph
        G = params["graph"]
        graph_file = "Provided Graph"
        
    config = Config(
        n=params["n"],
        m=params["m"],
        timesteps=params["timesteps"],
        bi=params["bi"],
        bj=params["bj"],
        a=params["a"],
        alpha=params["alpha"],
        eps=params["eps"],
        sigma_coeff=params["sigma_coeff"],
        zeta=params["zeta"],
        eta=params["eta"],
        gamma=params["gamma"],
        G=G,
    )
    agentlist = create_agent_list(
    config.n, config.m, params["alpha_dist"], config.a, config.alpha, config.bi, config.bj, config.eps)
    recordbook = RecordBook(agentlist, config)
    
    # Create dictionary of neighbours for a faster look-up
    neighbors_dict = {node: list(config.G.neighbors(node)) for node in config.G.nodes()}

    for t in range(config.timesteps):
        for i, agent in enumerate(agentlist):
            recordbook.record_agent_state(agent)
            neigh_list = neighbors_dict[i]  # Already a list
            if neigh_list and random.random() < agent.a:
                j = random.choice(neigh_list)
                agent.update_agent_t(
                    agentlist[j],
                    sigma=config.sigma,
                    gamma=config.gamma,
                    zeta=config.zeta,
                    eta=config.eta
                )
                agent.reset_accepted()
                agent.update_probabilities()
                agentlist[j].update_probabilities()
    for u,v in config.G.edges():
        metric = calc_metric(agentlist[u],agentlist[v], method=params["metric_method"])
        recordbook.metric_by_edge[(u,v)][t] = metric
    recordbook.compute_moving_average(avg_window=int(config.timesteps * 0.04))    
    final_x_vectors = {agent.name: agent.state_vector.tolist() for agent in agentlist}
    final_moving_avg = {
        agent.name: [v.tolist() if isinstance(v, np.ndarray) else v for v in recordbook.movingavg[agent.name]]
        for agent in agentlist
    }
    return {
        "params": params,
        "repetition": repetition_index,
        "graph_file": graph_file,
        "final_x_vectors": final_x_vectors,
        "final_moving_avg": final_moving_avg
    }

def parameter_sweep(parameter_grid, repetitions, output_dir="results"):
    """
    Runs a parameter sweep based on the provided parameter grid.

    :param parameter_grid: Dictionary of parameter ranges to sweep.
    :param repetitions: Number of repetitions per parameter combination.
    :param output_dir: Directory to save the results.
    """
    os.makedirs(output_dir, exist_ok=True)
    param_combinations = list(product(*[[(key, value) for value in values] for key, values in parameter_grid.items()]))

    for param_tuple in param_combinations:
        params = {key: value for key, value in param_tuple}
        
        for rep in range(repetitions):
        
            result = run_simulation_with_params(params, rep)

            # Save intermediate results to JSON
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{output_dir}/results_{timestamp}_rep{rep}.json"
            with open(filename, "w") as f:
                json.dump(result, f, indent=4)
    print(f"Parameter sweep completed! Results saved to {output_dir}.")

# Example usage
if __name__ == "__main__":
    # Define parameter ranges
    parameter_grid = {
        "n": [500],
        "m": [4,8,12,20],
        "timesteps": [5000],
        "bi": [7],
        "bj": [7],
        "a": [0.5],
        "alpha": [0.2],
        "eps": [0.1,0.5,0.9],
        "sigma_coeff": [10],
        "zeta": [0],
        "eta": [0.5],
        "gamma": [-1.0,0.0,1.0],
        # "graph": [nx.powerlaw_cluster_graph(500, 5, 0.3)],  # Example graph
        "metric_method": ["pdiff"],
        "alpha_dist": ["static"]
    }

    repetitions = 10
    parameter_sweep(parameter_grid, repetitions, output_dir="sweep_results")
#!/usr/bin/env python3
"""
File: natural_simulation.py

Summary:
    Provides the main simulation driver for the ABM.
    Includes functions to create agents, load random graphs, and run full simulations.
    The simulation output includes:
      - "final_x_vectors": final state vectors for each agent (as lists)
      - "final_moving_avg": the moving average state vectors (converted to lists)
      - Optionally, "records": the raw time-series of state vectors (if include_records=True)
    The sigma parameter is now defined independently (and passed directly).
    
Usage:
    Run directly:
        python natural_simulation.py
    or import its functions:
        from natural_simulation import run_simulation_with_params
        result = run_simulation_with_params(params, repetition_index, include_records=True)
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
    file_list = [f for f in os.listdir(graph_dir) if f.endswith(".pkl")]
    if not file_list:
        raise FileNotFoundError(f"No .pkl files found in {graph_dir}")
    chosen_file = random.choice(file_list)
    path = os.path.join(graph_dir, chosen_file)
    with open(path, "rb") as f:
        G = pickle.load(f)
    print(f"Loaded graph from: {chosen_file}")
    return G, chosen_file

def run_simulation_with_params(params, repetition_index, include_records=False):
    """
    Runs the simulation with the provided parameters and repetition index.
    
    Args:
        params: A dictionary of simulation parameters.
        repetition_index: An integer specifying which repetition is being run.
        include_records: Optional boolean (default False). If True, the output will include
                         the raw time-series records under the "records" key.
    
    Returns:
        A dictionary with keys:
          "params", "repetition", "graph_file", "final_x_vectors", "final_moving_avg"
          and, if include_records is True, "records".
    """
    print(f"Running simulation with parameters: {params}, repetition: {repetition_index}")
    
    if "graph" not in params or params["graph"] is None:
        # Load a random graph from your library in the "singular_graph" folder.
        G, graph_file = load_random_graph("singular_graph")
    else:
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
        sigma=params["sigma"],   # sigma is now independent
        zeta=params["zeta"],
        eta=params["eta"],
        gamma=params["gamma"],
        G=G,
    )
    agentlist = create_agent_list(config.n, config.m, params["alpha_dist"], config.a, config.alpha, config.bi, config.bj, config.eps)
    recordbook = RecordBook(agentlist, config)
    
    # Create a dictionary of neighbours for faster lookup.
    neighbors_dict = {node: list(config.G.neighbors(node)) for node in config.G.nodes()}
    
    for t in range(config.timesteps):
        for i, agent in enumerate(agentlist):
            recordbook.record_agent_state(agent)
            neigh_list = neighbors_dict[i]
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
    # Record metrics along graph edges (for potential future use)
    for u, v in config.G.edges():
        metric = calc_metric(agentlist[u], agentlist[v], method=params["metric_method"])
        recordbook.metric_by_edge[(u, v)][t] = metric
        
    # Compute moving average with window = T * 0.04
    recordbook.compute_moving_average(avg_window=int(config.timesteps * 0.04))
    
    # Convert final_x_vectors and final_moving_avg to lists.
    final_x_vectors = {agent.name: agent.state_vector.tolist() for agent in agentlist}
    final_moving_avg = {
        agent.name: [v.tolist() if isinstance(v, np.ndarray) else v for v in recordbook.movingavg[agent.name]]
        for agent in agentlist
    }
    result = {
        "params": params,
        "repetition": repetition_index,
        "graph_file": graph_file,
        "final_x_vectors": final_x_vectors,
        "final_moving_avg": final_moving_avg,
    }
    
    if include_records:
        # Convert raw records to lists.
        raw_records = {
            agent.name: [state.tolist() if isinstance(state, np.ndarray) else state for state in recordbook.records[agent.name]]
            for agent in agentlist
        }
        result["records"] = raw_records
        
    return result

def parameter_sweep(parameter_grid, repetitions, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    from itertools import product
    param_combinations = list(product(*[[(key, value) for value in values] for key, values in parameter_grid.items()]))
    for param_tuple in param_combinations:
        params = {key: value for key, value in param_tuple}
        for rep in range(repetitions):
            result = run_simulation_with_params(params, rep)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{output_dir}/results_{timestamp}_rep{rep}.json"
            with open(filename, "w") as f:
                json.dump(result, f, indent=4)
    print(f"Parameter sweep completed! Results saved to {output_dir}.")

if __name__ == "__main__":
    from itertools import product
    main()
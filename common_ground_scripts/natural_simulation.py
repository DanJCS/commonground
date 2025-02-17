#!/usr/bin/env python3
"""
File: natural_simulation.py

Summary:
    Provides the main simulation driver for the ABM.
    Includes functions to create agents, load random graphs, and run full simulations.
    The simulation output includes:
      - "final_x_vectors": final state vectors for each agent (as lists)
      - "final_moving_avg": the moving average vector at the final timestep.
      - Optionally, if include_records is True, "records": the time-series of moving average state vectors.
    The sigma parameter is now defined independently (and passed directly).

Usage:
    Run directly:
        python natural_simulation.py
    or import its functions:
        from natural_simulation import run_simulation_with_params
        result = run_simulation_with_params(params, repetition_index, include_records=True, record_interval=10)
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


### CHANGED: Added new parameter "record_interval" to allow recording at defined intervals.
def run_simulation_with_params(params, repetition_index, include_records=True, record_interval=10):
    """
    Runs the simulation with the provided parameters and repetition index.

    New Parameters:
      - include_records (bool): If True, record the moving average at intervals.
      - record_interval (int): The timestep interval at which to record the moving average.

    Returns:
        A dictionary with keys:
          "params", "repetition", "graph_file", "final_x_vectors", "final_moving_avg"
          and, if include_records is True, "records" containing the time-series of moving averages.

        Note: "final_moving_avg" always contains the moving average vector at the final timestep.
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
        sigma=params["sigma"],  # sigma is now independent
        zeta=params["zeta"],
        eta=params["eta"],
        gamma=params["gamma"],
        G=G,
    )
    agentlist = create_agent_list(config.n, config.m, params["alpha_dist"], config.a, config.alpha, config.bi,
                                  config.bj, config.eps)
    recordbook = RecordBook(agentlist, config)

    # Simulation loop.
    for t in range(config.timesteps):
        for i, agent in enumerate(agentlist):
            neigh_list = list(config.G.neighbors(i))
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
        ### CHANGED: Record the state only at specified intervals when include_records is True.
        if include_records and (record_interval is None or t % record_interval == 0):
            for agent in agentlist:
                recordbook.record_agent_state(agent)

    # After simulation, compute moving averages.
    if include_records:
        # Determine effective number of recorded samples
        num_samples = len(recordbook.records[agentlist[0].name])
        # Adjust avg_window relative to record_interval.
        avg_window = max(int((config.timesteps * 0.04) / record_interval), 1)
        recordbook.compute_moving_average_series(avg_window)
    else:
        # Even if not recording time-series, we compute the final moving average using the full record.
        # Here we record every state to compute the final moving average.
        for agent in agentlist:
            recordbook.record_agent_state(agent)
        avg_window = max(int((config.timesteps * 0.04)), 1)
        recordbook.compute_moving_average_series(avg_window)

    # Prepare final outputs.
    final_x_vectors = {agent.name: agent.state_vector.tolist() for agent in agentlist}
    ### CHANGED: final_moving_avg now stores only the last computed moving average vector.
    final_moving_avg = {
        agent.name: (recordbook.movingavg[agent.name][-1].tolist() if isinstance(recordbook.movingavg[agent.name][-1],
                                                                                 np.ndarray)
                     else recordbook.movingavg[agent.name][-1])
        for agent in agentlist
    }

    result = {
        "params": params,
        "repetition": repetition_index,
        "graph_file": graph_file,
        "final_x_vectors": final_x_vectors,
        "final_moving_avg": final_moving_avg,
    }

    ### CHANGED: If include_records is True, store the entire time-series under the "records" key.
    if include_records:
        result["records"] = {
            agent.name: [v.tolist() if isinstance(v, np.ndarray) else v for v in recordbook.movingavg[agent.name]]
            for agent in agentlist
        }

    return result


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


if __name__ == "__main__":
    from itertools import product

    main()
#!/usr/bin/env python3
"""
File: natural_simulation.py

Summary:
    Provides the main simulation driver for the ABM.
    The simulation output includes:
      - "final_x_vectors": final state vectors for each agent (as lists)
      - "final_moving_avg": the final moving average vector (a sliding-window average).
      - Optionally, if include_records is True, "records": the raw state vectors over time.

Usage:
    python natural_simulation.py
    # or
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
from CG_source import Config, Agent, RecordBook, beta_extended


### CHANGED: Modified docstring to reflect that "records" now stores raw state vectors.
def run_simulation_with_params(params, repetition_index, include_records=True, record_interval=10):
    """
    Runs the simulation with the provided parameters and repetition index.

    Args:
        params (dict): Simulation parameters, including n, m, timesteps, etc.
        repetition_index (int): Repetition index for tracking multiple runs.
        include_records (bool): If True, store the raw state vectors at specified intervals.
        record_interval (int): Timestep interval at which to record raw state vectors.

    Returns:
        dict: {
          "params": {...},                 # The parameters used
          "repetition": int,              # The repetition index
          "graph_file": str,              # Which graph was used
          "final_x_vectors": {...},       # Final state vectors as lists
          "final_moving_avg": {...},      # Final moving average (sliding window)
          "records": {...} (optional)     # Raw state vectors if include_records=True
        }
    """
    print(f"Running simulation with parameters: {params}, repetition: {repetition_index}")

    # If no graph is provided, load from 'singular_graph' folder
    if "graph" not in params or params["graph"] is None:
        G, graph_file = load_random_graph("singular_graph")
    else:
        G = params["graph"]
        graph_file = "Provided Graph"

    # Create config and agent list
    config = Config(
        n=params["n"],
        m=params["m"],
        timesteps=params["timesteps"],
        bi=params["bi"],
        bj=params["bj"],
        a=params["a"],
        alpha=params["alpha"],
        eps=params["eps"],
        sigma=params["sigma"],
        zeta=params["zeta"],
        eta=params["eta"],
        gamma=params["gamma"],
        G=G,
    )
    agentlist = create_agent_list(
        config.n,
        config.m,
        params.get("alpha_dist", "static"),
        config.a,
        config.alpha,
        config.bi,
        config.bj,
        config.eps,
    )
    recordbook = RecordBook(agentlist, config)

    # Simulation loop
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

        # CHANGED: If include_records, store the *raw* state vectors at intervals
        if include_records and (record_interval is None or t % record_interval == 0):
            for agent in agentlist:
                recordbook.record_agent_state(agent)

    # After the loop, compute the final moving average
    # We want a sliding-window average of size int(timesteps*0.04)
    avg_window = max(int(config.timesteps * 0.04), 1)

    ### If we never recorded raw states (include_records=False),
    ### we still need at least one record to compute a final average:
    if not include_records:
        for agent in agentlist:
            recordbook.record_agent_state(agent)

    recordbook.compute_moving_average_series(avg_window)

    # Prepare final outputs
    final_x_vectors = {
        agent.name: agent.state_vector.tolist() for agent in agentlist
    }

    # The "final_moving_avg" is the last sliding-window average in recordbook.movingavg
    final_moving_avg = {}
    for agent in agentlist:
        final_array = recordbook.movingavg[agent.name][-1]
        if isinstance(final_array, np.ndarray):
            final_moving_avg[agent.name] = final_array.tolist()
        else:
            final_moving_avg[agent.name] = final_array

    result = {
        "params": params,
        "repetition": repetition_index,
        "graph_file": graph_file,
        "final_x_vectors": final_x_vectors,
        "final_moving_avg": final_moving_avg,
    }

    ### CHANGED: If include_records is True, store the *raw* recorded states in "records"
    if include_records:
        result["records"] = {
            agent.name: [
                arr.tolist() if isinstance(arr, np.ndarray) else arr
                for arr in recordbook.records[agent.name]
            ]
            for agent in agentlist
        }

    return result


def load_random_graph(graph_dir="graphs_library"):
    """
    Randomly selects one of the pickle files in `graph_dir`
    and loads it as a NetworkX Graph.
    """
    import os
    import random
    import pickle

    file_list = [f for f in os.listdir(graph_dir) if f.endswith(".pkl")]
    if not file_list:
        raise FileNotFoundError(f"No .pkl files found in {graph_dir}")
    chosen_file = random.choice(file_list)
    path = os.path.join(graph_dir, chosen_file)
    with open(path, "rb") as f:
        G = pickle.load(f)
    print(f"Loaded graph from: {chosen_file}")
    return G, chosen_file


def create_agent_list(n, m, alpha_dist, a, alpha, bi, bj, eps):
    """
    Creates a list of Agent objects, each with an initial state_vector drawn from beta_extended.
    alpha_dist can be 'beta', 'static', or 'uniform' for alpha assignment.
    """
    Agents = []
    info_list = [beta_extended(n) for _ in range(m)]
    if alpha_dist == "beta":
        alpha_list = np.random.beta(3.5, 14, n)
    elif alpha_dist == "static":
        alpha_list = [alpha] * n
    elif alpha_dist == "uniform":
        alpha_list = np.random.uniform(0, 1, n)
    else:
        alpha_list = [alpha] * n  # fallback

    for i in range(n):
        agent = Agent(
            name=f"Agent_{i}",
            a=a,
            alpha=alpha_list[i],
            bi=bi,
            bj=bj,
            eps=eps,
            state_vector=np.array([info_list[idx][i] for idx in range(m)]),
            local_similarity=0.0
        )
        Agents.append(agent)
    return Agents


if __name__ == "__main__":
    # For direct usage: python natural_simulation.py
    # Typically you'd import run_simulation_with_params into other scripts (e.g., run.py).
    pass
#!/usr/bin/env python3
"""
internal_energy_stability.py

Runs the ABM in a step-by-step manner to track and plot the internal energy
(mean variance of agents' x-vectors) at each (or some) timesteps. Then detects 
when the simulation becomes "stable" based on minimal internal-energy changes.
"""

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import from your existing scripts:
# Adjust these import paths as needed so that Python can resolve them correctly.
from CG_source import Config, Agent, RecordBook, beta_extended
from natural_simulation import load_random_graph, create_agent_list
# If needed, from run_parallel_simulation import run_single_sim  # or anything else

###############################################################################
# 1. Helper: Compute Internal Energy
###############################################################################
def compute_internal_energy(agentlist):
    """
    Compute the mean variance of all agents' x-vectors in the current state.

    :param agentlist: List of Agent objects, each with a 'state_vector'.
    :return: float representing the system's internal energy.
    """
    if not agentlist:
        return 0.0

    variances = []
    for agent in agentlist:
        vec_var = np.var(agent.state_vector)
        variances.append(vec_var)

    return float(np.mean(variances))


###############################################################################
# 2. Helper: Detect Stability
###############################################################################
def detect_stability(energy_values, timesteps, stability_threshold, check_window):
    """
    Determine the first timestep at which the relative change in internal energy
    stays below `stability_threshold` for `check_window` consecutive intervals.

    :param energy_values: List of internal energy values in the order they were recorded.
    :param timesteps: Corresponding list of timesteps at which each energy was recorded.
    :param stability_threshold: Max allowable relative change in energy for stability.
    :param check_window: Number of consecutive intervals that must satisfy the threshold.
    :return: stable_timestep (int) or -1 if not found.
    """
    if len(energy_values) < check_window + 1:
        return -1

    # Calculate the relative changes
    changes = []
    for i in range(1, len(energy_values)):
        e_curr = energy_values[i]
        e_prev = energy_values[i - 1]
        if e_curr == 0.0:
            rel_change = 0.0
        else:
            rel_change = abs(e_curr - e_prev) / abs(e_curr)
        changes.append(rel_change)

    # We look for a run of `check_window` consecutive intervals < stability_threshold
    # `changes[i]` is the change from energy_values[i] to energy_values[i+1]
    for start_idx in range(len(changes) - check_window + 1):
        window_slice = changes[start_idx : start_idx + check_window]
        if all(x < stability_threshold for x in window_slice):
            # The stable point is the next recorded timestep after that window
            # Specifically, it means from index `start_idx` to `start_idx + check_window - 1`
            # are all stable intervals, so the earliest stable is timesteps[start_idx + check_window].
            return timesteps[start_idx + check_window]

    return -1


###############################################################################
# 3. Main: Running ABM Step-by-Step, Tracking Internal Energy
###############################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Track internal energy over time to detect ABM stability."
    )
    # Possibly pass a JSON with ABM parameters or define them inline
    parser.add_argument("--params_json", type=str, default=None,
                        help="Path to a JSON file specifying ABM parameters.")
    parser.add_argument("--output_dir", type=str, default="ie_stability_results",
                        help="Directory to save plots and optional debug data.")
    parser.add_argument("--max_timesteps", type=int, default=1000,
                        help="Number of timesteps to run the ABM.")
    parser.add_argument("--record_interval", type=int, default=10,
                        help="Record internal energy every N timesteps.")
    parser.add_argument("--stability_threshold", type=float, default=0.01,
                        help="Max relative change in energy for calling it 'stable'.")
    parser.add_argument("--check_window", type=int, default=5,
                        help="Number of consecutive intervals for stability detection.")
    parser.add_argument("--save_debug_json", action="store_true",
                        help="Save timestep->energy data as JSON.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 3A. Load or define parameters
    # -------------------------------------------------------------------------
    if args.params_json and os.path.isfile(args.params_json):
        with open(args.params_json, "r") as f:
            params = json.load(f)
        print(f"Loaded parameters from {args.params_json}: {params}")
    else:
        # Fallback or example default
        # You can adapt these values to match your desired run:
        params = {
            "n": 100,
            "m": 10,
            "timesteps": args.max_timesteps,  # link to your choice
            "bi": 7,
            "bj": 7,
            "a": 0.5,
            "alpha": 0.1,
            "eps": 0.2,
            "sigma_coeff": 10,
            "zeta": 0,
            "eta": 0.5,
            "gamma": 0.0,
            "alpha_dist": "static",  # from natural_simulation logic
            "graph": None,           # let it load a random graph
        }

    # If a specific graph is not given, load a random graph (like in `natural_simulation.py`).
    if params.get("graph") is None:
        G, graph_file = load_random_graph("graphs_library")
        params["graph"] = G
    else:
        graph_file = "Provided Graph"

    # Create the config object from CG_source.py's `Config` class
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
        G=params["graph"],
    )

    # Create the agent list (from natural_simulation.py)
    agentlist = create_agent_list(
        n=config.n,
        m=config.m,
        alpha_dist=params["alpha_dist"],
        a=config.a,
        alpha=config.alpha,
        bi=config.bi,
        bj=config.bj,
        eps=config.eps,
    )

    # Build adjacency data for efficiency
    neighbors_dict = {node: list(config.G.neighbors(node)) for node in config.G.nodes()}

    # -------------------------------------------------------------------------
    # 3B. Run the Simulation Step-by-Step and Track Internal Energy
    # -------------------------------------------------------------------------
    recorded_timesteps = []
    energy_values = []

    max_timesteps = config.timesteps
    record_interval = args.record_interval

    for t in range(max_timesteps):
        # Update each agent in some random order or fixed order
        # This is a direct adaptation from `run_simulation_with_params` in natural_simulation.py.
        # But we do it step-by-step so we can measure internal energy.

        for i, agent in enumerate(agentlist):
            neigh_list = neighbors_dict[i]
            if neigh_list and np.random.random() < agent.a:
                j = np.random.choice(neigh_list)
                agent.update_agent_t(
                    receiver=agentlist[j],
                    sigma=config.sigma,
                    gamma=config.gamma,
                    zeta=config.zeta,
                    eta=config.eta
                )
                agent.reset_accepted()
                agent.update_probabilities()
                agentlist[j].update_probabilities()

        # At the end of the timestep, record internal energy if t % record_interval == 0
        if t % record_interval == 0:
            ie_val = compute_internal_energy(agentlist)
            recorded_timesteps.append(t)
            energy_values.append(ie_val)

    # One final measurement at last timestep if not caught by interval
    if (max_timesteps - 1) not in recorded_timesteps:
        final_ie = compute_internal_energy(agentlist)
        recorded_timesteps.append(max_timesteps - 1)
        energy_values.append(final_ie)

    # -------------------------------------------------------------------------
    # 3C. Detect Stability
    # -------------------------------------------------------------------------
    stable_timestep = detect_stability(
        energy_values=energy_values,
        timesteps=recorded_timesteps,
        stability_threshold=args.stability_threshold,
        check_window=args.check_window
    )

    if stable_timestep < 0:
        print("No stable point detected under given threshold/window.")
    else:
        print(f"System reached stability at timestep: {stable_timestep}")

    # -------------------------------------------------------------------------
    # 3D. Save Debug JSON if requested
    # -------------------------------------------------------------------------
    if args.save_debug_json:
        debug_data = {
            "params": params,
            "graph_file": graph_file,
            "recorded_timesteps": recorded_timesteps,
            "energy_values": energy_values,
            "stable_timestep": stable_timestep,
        }
        debug_filename = f"IE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_debug.json"
        debug_path = os.path.join(args.output_dir, debug_filename)
        with open(debug_path, "w") as f:
            json.dump(debug_data, f, indent=4)
        print(f"Saved debug data to {debug_path}")

    # -------------------------------------------------------------------------
    # 3E. Plot the Internal Energy Over Time
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(recorded_timesteps, energy_values, marker="o", label="Internal Energy")
    plt.xlabel("Timestep")
    plt.ylabel("Mean Variance (Internal Energy)")
    plt.title("Internal Energy Over Time")

    if stable_timestep > 0:
        plt.axvline(stable_timestep, color="red", linestyle="--", label="Stable Point")
        plt.text(stable_timestep, max(energy_values) * 0.9,
                 f"Stability @ {stable_timestep}",
                 color="red", ha="center")

    plt.legend()
    out_plot = os.path.join(args.output_dir, "internal_energy_stability.png")
    plt.savefig(out_plot, dpi=150)
    plt.close()

    print(f"Plot saved to {out_plot}")
    print("Done.")


###############################################################################
# 4. Entry Point
###############################################################################
if __name__ == "__main__":
    main()
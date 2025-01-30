import math
import os
import random
from collections import defaultdict
from itertools import combinations

import community as community_louvain
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from CG_source import *

coeff = 10.0 # Multiplicative factor to adjust sigma based on m
n = 100 # Total number of agents
m = 10 # m is the length of the x vector (state vector)
timesteps = 5000 # Total timesteps
bi = 7 # Rationality of agents (sending) - high bi favours high value information (high value element in x vector). Loglinear learning.
bj = 7 # Rationality of agents (receiving)
a = 0.5 # Probability of activation
alpha = 0.2 # How easily influenced the agent is: higher alpha leads to larger changes per timestep
eps = 0.1 # Probability of NOT providing a response back to the sender
sigma = 1 / (coeff * m) # Decay factor - simulates loss of memory (or something of that sort). Adjusted based on m and coeff.

AVG_WINDOW = int(timesteps * 0.05) # Breaks into larger chunks of timesteps for moving average analysis

# ** x vector represents an agent's perceived common ground. Can be interpreted as their "beliefs".
# Each element can be interpreted as the level of certainty of a particular information - bounded in (-1,1)
zeta = 0 # {-1,0,1} indicator for whether the certainty of the receiver decreases, does not change or increases upon rejecting
eta = 0.5 # Determines how receiver handles rejection response. Certainty tends to eta upon rejection.
gamma = -1.0 # How sender handles absence of response. (-1,+1). -1 is the same as a rejection. +1 is the same as acceptance.
# 0 means will not affect sender's belief.

BINWIDTH = 0.1 # binwidth for histogram

## Graph parameters
numedge, triprob = 5, 0.5
## Distribution parameters
INPUT_ALPHA = "static"  # "static", "uniform", "beta" - How the alpha parameter is distributed across the population

HK = nx.powerlaw_cluster_graph(n, numedge, triprob)

metric_method = "pdiff" # Method to measure the pairwise similarity of perceived common ground (or "beliefs")

## Creating a list of agents to feed into the simulation
def create_agent_list(n, INPUT_ALPHA): 
    Agents = []
    info_list = [beta_extended(n) for _ in range(m)] # Uses beta distribution to initialise perceived common ground
    if INPUT_ALPHA == "beta":
        alpha_list = np.random.beta(3.5, 14, n)
    elif INPUT_ALPHA == "static":
        alpha_list = [alpha] * n
    elif INPUT_ALPHA == "uniform":
        alpha_list = np.random.beta(1, 1, n)
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

config = Config(
            n, m, timesteps, bi, bj, a, alpha, eps, sigma, zeta, eta, gamma, HK
        )

# Create n number of agents (in a list)
agentlist = create_agent_list(n, "static")
# Create a record book to keep track of all agents
recordbook = RecordBook(agentlist, config)
# Metric records
metric_by_edge = {}
for _, (u, v) in enumerate(HK.edges()):
    metric_by_edge[(u, v)] = np.zeros(timesteps)



# Simulation
for t in range(timesteps):
    for i, agent in enumerate(agentlist):
        # Record state vector at T=t
        recordbook.record_agent_state(agent)
        # Randomly select one neighbour as receiver(j)
        j = random.choice(list(HK.neighbors(i)))
        # If p < activity threshold:
        if random.random() < agent.a:
            # i communicates to j (main simulation)
            agent.update_agent_t(
                agentlist[j], sigma, gamma=gamma, zeta=zeta, eta=eta
            )
            print(f"---TIMESTEP: {t}------------------------------")
            print(f"Sender:\n{agent}")
            print(f"Receiver:\n{agentlist[j]}")
            recordbook.acceptrecord[t][1] += 1
            if agent.accepted == 1:
                print("Information Accepted")
                recordbook.acceptrecord[t][0] += 1
            else:
                print("Information Rejected")
            print()
            agent.reset_accepted()
            agent.update_probabilities()
            agentlist[j].update_probabilities()
    for index, (u, v) in enumerate(HK.edges()):
        metric = calc_metric(agentlist[u], agentlist[v], method=metric_method)
        metric_by_edge[(u, v)][t] = metric
        recordbook.metric_by_edge[(u, v)][t] = metric
        HK[u][v]["weight"] = metric
for agent in agentlist:
    recordbook.record_agent_state(agent)
recordbook.compute_moving_average(AVG_WINDOW)
recordbook.add_movingavg_metric(AVG_WINDOW)
for _, (u, v) in enumerate(recordbook.metric_movingavg):
    HK[u][v]["weight"] = recordbook.metric_movingavg[(u, v)][timesteps - 1]
    
# Save and print the final moving average x-vectors
output_file = os.path.join(os.getcwd(), "final_moving_avg_vectors.txt")

with open(output_file, "w") as file:
    file.write("Final Moving Average State Vectors\n")
    file.write("=" * 50 + "\n")

    print("\nFinal Moving Average State Vectors:")
    print("=" * 50)
    for agent_name, moving_avg_records in recordbook.movingavg.items():
        # Extract the final moving average vector
        final_moving_avg = moving_avg_records[timesteps - 1]

        # Print the agent's name and their moving average vector
        print(f"{agent_name}: {final_moving_avg}")
        file.write(f"{agent_name}: {final_moving_avg}\n")
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

coeff = 10 # Multiplicative factor to adjust sigma based on m
n = 100 # Total number of agents
timesteps = 5000 # Total timesteps
bi = 7 # Rationality of agents (sending) - high bi favours high value information (high value element in x vector). Loglinear learning.
bj = 7 # Rationality of agents (receiving)
a = 0.5 # Probability of activation
alpha = 0.2 # How easily influenced the agent is: higher alpha leads to larger changes per timestep
eps = 0.1 # Probability of NOT providing a response back to the sender
sigma = 0 # Decay factor - simulates loss of memory (or something of that sort). Adjusted based on m and coeff.

AVG_WINDOW = int(timesteps * 0.05) # Breaks into larger chunks of timesteps for moving average analysis

m_values = [4, 8, 12, 16] # m is the length of the x vector (state vector)
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

## Parameter sweeping
for trial in range(10):
    for m in m_values:
        output_dir = f"Offline-m({m})-Trial:{trial+1}"
        sigma = 1 / (m * coeff) # sigma is updated based on m and coeff
        os.makedirs(output_dir, exist_ok=True)

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
                recordbook.add_record(agent)
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
            recordbook.add_record(agent)
        recordbook.add_movingavg(AVG_WINDOW)
        recordbook.add_movingavg_metric(AVG_WINDOW)
        for _, (u, v) in enumerate(recordbook.metric_movingavg):
            HK[u][v]["weight"] = recordbook.metric_movingavg[(u, v)][timesteps - 1]

        # Plot moving average
        plt.figure()  # Start a new figure
        recordbook.plot_movingavg(dir=output_dir, savefig=True)
        plt.close()  # Close the figure

        # Plot acceptance rate
        plt.figure()  # Start a new figure
        recordbook.plot_acceptance_rate(dir=output_dir, savefig=True)
        plt.close()  # Close the figure

        # Plot global agreement
        plt.figure()  # Start a new figure
        plt.plot(np.arange(timesteps), recordbook.global_metric())
        plt.title("Global agreement plot")
        plt.ylim(0, 1.15)
        plt.ylabel("Total sum of pairwise metric")
        plt.xlabel("Timesteps")
        plt.savefig(os.path.join(output_dir, "Global Agreement"))
        plt.close()  # Close the figure

        # Metric progression
        plt.figure()
        recordbook.metric_progression(dir=output_dir, savefig=True)
        plt.close()

        # numerical output Louvain's
        partition = community_louvain.best_partition(HK, weight="weight")
        textfile = os.path.join(output_dir, f"Louvains output m={m}")
        with open(textfile, "w") as file:
            print(partition, file=file)

        # Louvain output
        recordbook.louvain_grid(dir=output_dir, savefig=True)

        mean_xvec = recordbook.get_mean_xvec()
        print(recordbook.get_variance(mean_xvec))

        # Surviving m
        recordbook.surviving_info(
            f"Variance: {np.sum(recordbook.get_variance(mean_xvec))}",
            dir=output_dir,
            savefig=True,
        )

        mean_xvec = recordbook.get_mean_xvec()
        print(recordbook.get_variance(mean_xvec))

        # Louvain's algorithm
        partition = community_louvain.best_partition(HK, weight="weight")

        communities = defaultdict(list)

        for node_id, community_id in partition.items():
            communities[community_id].append(node_id)

        community_vectors = {}
        for community_id, nodes in communities.items():
            state_vectors = [agentlist[node_id].state_vector for node_id in nodes]
            mean_vector = np.mean(state_vectors, axis=0)
            community_vectors[community_id] = mean_vector
            print(f"Community {community_id}: Mean State Vector = {mean_vector}")

            # Initialize the super-network
        super_G = nx.Graph()

        # Add nodes (communities) to the super-network
        super_G.add_nodes_from(community_vectors.keys())

        def compute_similarity(u, v):
            send_u = np.exp(bi * u) / np.sum(np.exp(bi * u))
            send_v = np.exp(bi * v) / np.sum(np.exp(bi * v))
            metric = 1 - math.sqrt(np.sum(np.square(send_u - send_v))) / math.sqrt(2)
            return metric

        # Add weighted edges between all pairs of communities
        for (comm1, vec1), (comm2, vec2) in combinations(community_vectors.items(), 2):
            weight = compute_similarity(vec1, vec2)
            super_G.add_edge(comm1, comm2, weight=weight)
        plt.figure(figsize=(10, 10))
        super_pos = nx.spring_layout(super_G)
        edge_weights = [super_G[u][v]["weight"] for u, v in super_G.edges()]
        nx.draw(
            super_G,
            super_pos,
            with_labels=True,
            node_color="lightblue",
            edge_color="gray",
            node_size=500,
            width=2,
        )
        nx.draw_networkx_edge_labels(
            super_G, super_pos, edge_labels=nx.get_edge_attributes(super_G, "weight")
        )
        plt.title("Super-Network of Communities")
        plt.savefig(os.path.join(output_dir, "Supernetwork 1"))
        plt.close()
        plt.clf()

        super_partition = community_louvain.best_partition(
            super_G, weight="weight", resolution=1.20
        )

        print(super_partition)

        # Get the list of super-community IDs
        super_community_ids = list(set(super_partition.values()))
        num_super_communities = len(super_community_ids)

        # Create a color map with enough distinct colors
        colors = plt.cm.get_cmap("tab10", num_super_communities)
        color_map = {}

        for idx, super_comm_id in enumerate(super_community_ids):
            color_map[super_comm_id] = colors(idx)

        # Assign colors to the nodes based on their super-community assignment
        node_colors = [color_map[super_partition[node]] for node in super_G.nodes()]

        # Step 2: Draw the Super-Network with Color-Coded Nodes

        plt.figure(figsize=(8, 8))
        nx.draw_networkx_nodes(
            super_G,
            super_pos,
            node_size=700,
            node_color=node_colors,
            cmap=plt.cm.get_cmap("tab10", num_super_communities),
        )
        nx.draw_networkx_edges(super_G, super_pos, edge_color="gray")
        nx.draw_networkx_labels(super_G, super_pos, font_size=12, font_weight="bold")

        # # Optionally, draw edge labels for weights
        # edge_labels = nx.get_edge_attributes(super_G, "weight")
        # formatted_edge_labels = {
        #     edge: f"{weight:.2f}" for edge, weight in edge_labels.items()
        # }
        # nx.draw_networkx_edge_labels(super_G, super_pos, edge_labels=formatted_edge_labels)

        # Calculate the average edge weight
        total_edge_weight = sum(
            data["weight"] for _, _, data in super_G.edges(data=True)
        )
        number_of_edges = super_G.number_of_edges()
        average_edge_weight = (
            total_edge_weight / number_of_edges if number_of_edges > 0 else 0
        )

        # Create legend handles for the groups
        legend_handles = [
            Patch(color=color_map[super_comm_id], label=f"Group {super_comm_id}")
            for super_comm_id in super_community_ids
        ]

        # Create a custom legend handle for the average edge weight
        average_weight_handle = Line2D(
            [0],
            [0],
            color="black",
            lw=0,
            label=f"Avg Edge Weight: {average_edge_weight:.2f}",
        )

        # Combine all legend handles
        all_handles = legend_handles + [average_weight_handle]

        # Add the legend to the plot
        plt.legend(handles=all_handles, loc="best", title="Groups")
        plt.title(f"Grouped Super-Network - Offline, m={m}")
        plt.axis("off")  # Turn off the axis
        plt.savefig(os.path.join(output_dir, "Supernetwork 2"))
        plt.close()
        plt.clf()

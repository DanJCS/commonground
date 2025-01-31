"""
File: CG_source.py

Summary:
    CG_source.py provides the foundational classes and core functions for the agent-based model (ABM).
    It defines:
        * A variety of metrics (_metric_normdiff, _metric_absdiff, etc.) used to measure differences or similarities between agents' state vectors.
        * A Config class to store all simulation parameters (number of agents, timesteps, graph structure, etc.).
        * An Agent class that models an individual participant in the ABM, including how it updates or sends information.
        * A RecordBook class that records simulation data (agent states, acceptance rates, similarity metrics over time, etc.).
    It also includes helper functions for computing probability distributions, updating agent states, and recording simulation results.

Dependencies:
    * Python built-ins: math, os, random, logging
    * Third-party: numpy, networkx, community (community_louvain), matplotlib

Usage:
    This script is not typically called directly via the command line. Instead, import it into other scripts.

Example:
    from CG_source import Config, Agent, RecordBook
    config = Config(n=100, m=10, timesteps=1000, ...)  # supply needed params
    agent = Agent(...)

"""
import math
import os
import random
import logging

import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def beta_extended(size, alpha=5, beta=5):
    """
    Generate random samples from a Beta distribution (alpha, beta),
    then transform them from [0, 1] range to [-1, 1].

    :param size: Number of samples to draw (int or tuple of ints)
    :param alpha: Alpha parameter for the Beta distribution (float > 0)
    :param beta: Beta parameter for the Beta distribution (float > 0)
    :return: A NumPy array of shape `size` with values in [-1, 1].
    """
    # Optional: Validate input
    if alpha <= 0 or beta <= 0:
        raise ValueError("Alpha and Beta must be positive.")
    
    regular = np.random.beta(alpha, beta, size)
    return regular * 2.0 - 1.0

def _metric_normdiff(u, v):
    vec_diff = u.state_vector - v.state_vector
    norm = np.linalg.norm(vec_diff)  # sqrt of sum of squares
    return (2 * np.sqrt(len(u.state_vector))) - norm

def _metric_normdiff_normalised(u, v):
    vec_diff = u.state_vector - v.state_vector
    norm = np.linalg.norm(vec_diff)
    return 1.0 - norm / (2.0 * np.sqrt(len(u.state_vector)))

def _metric_absdiff(u, v):
    return np.sum(np.abs(u.state_vector - v.state_vector))

def _metric_absdiff_normalised(u, v):
    return np.sum(np.abs(u.state_vector - v.state_vector)) / (2.0 * len(u.state_vector))

def _metric_percagree(u, v):
    n_total = 0
    n_agreed = 0
    for i, uval in enumerate(u.state_vector):
        vval = v.state_vector[i]
        # Count only if at least one of them is above 0.5 in magnitude
        if abs(uval) > 0.5 or abs(vval) > 0.5:
            n_total += 1
        # If both are above 0.5 in magnitude and have the same sign => agreed
        if abs(uval) > 0.5 and abs(vval) > 0.5 and (uval * vval > 0):
            n_agreed += 1
    return n_agreed / n_total if n_total else 0.0

def _metric_cossim(u, v):
    numerator = np.dot(u.state_vector, v.state_vector)
    denominator = np.linalg.norm(u.state_vector) * np.linalg.norm(v.state_vector)
    cossim = numerator / denominator
    return (1.0 + cossim) / 2.0

def _metric_pdiff(u, v):
    diff = u.p_send - v.p_send
    return 1.0 - (np.linalg.norm(diff) / math.sqrt(2.0))

_METRICS = {
    "normdiff": _metric_normdiff,
    "normdiff_normalised": _metric_normdiff_normalised,
    "absdiff": _metric_absdiff,
    "absdiff_normalised": _metric_absdiff_normalised,
    "percagree": _metric_percagree,
    "cossim": _metric_cossim,
    "pdiff": _metric_pdiff,
}

def calc_metric(u, v, method="normdiff"):
    """
    Calculate a pairwise metric between two agents' state_vectors or p_sends.

    Supported methods:
      - normdiff
      - normdiff_normalised
      - absdiff
      - absdiff_normalised
      - percagree
      - cossim
      - pdiff

    :param u: Agent U
    :param v: Agent V
    :param method: String specifying which metric to compute
    :return: A floating-point metric value.
    """
    if method not in _METRICS:
        raise ValueError(f"Unknown metric method '{method}'")
    return _METRICS[method](u, v)


### Class: Config
class Config:
    """
    A container for all parameters required to configure the simulation.

    Attributes:
        n (int): Number of agents in the simulation.
        m (int): Length of each agent's state vector (dimensionality of the 'information').
        timesteps (int): Number of time steps in the simulation.
        bi (float): Rationality (sending behavior) parameter for the agents.
        bj (float): Rationality (receiving behavior) parameter for the agents.
        a (float): Probability of an agent becoming 'active' to send info at each timestep.
        alpha (float): Influence factor (how easily agents update their state vectors).
        eps (float): Probability that the sender does NOT receive an acknowledgment back.
        sigma (float): Decay factor, controls some memory or certainty decay in updates.
        zeta (float): Parameter influencing how rejection changes an agent's certainty.
        eta (float): Parameter controlling how the receiver adapts upon rejection.
        gamma (float): Parameter controlling how the sender adapts if no acknowledgement is received.
        G (nx.Graph): A NetworkX graph representing the agent connections (social network).
    """

    def __init__(
        self,
        n: int,
        m: int,
        timesteps: int,
        bi: float,
        bj: float,
        a: float,
        alpha: float,
        eps: float,
        sigma_coeff: float,
        zeta: float,
        eta: float,
        gamma: float,
        G: nx.Graph,
    ) -> None:
        # Basic validation
        if n <= 0:
            raise ValueError("Number of agents (n) must be positive.")
        if m <= 0:
            raise ValueError("Length of the state vector (m) must be positive.")
        if timesteps <= 0:
            raise ValueError("Number of timesteps must be positive.")
        if not (0.0 <= a <= 1.0):
            raise ValueError("Activation probability (a) should be in [0,1].")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("Alpha value out of [0,1].")
        if not zeta in [-1,0,1]:
            raise ValueError("Zeta must be one of {-1,0,+1}")
        # Add more checks if needed (e.g., alpha in [0,1], etc.)

        self.n = n
        self.m = m
        self.timesteps = timesteps
        self.bi = bi
        self.bj = bj
        self.a = a
        self.alpha = alpha
        self.eps = eps
        self.sigma = 1 / (self.m * sigma_coeff)
        self.zeta = zeta
        self.eta = eta
        self.gamma = gamma
        self.G = G

    def __repr__(self) -> str:
        """
        Returns a developer-friendly string representation of the Config object.
        """
        return (
            f"Config(n={self.n}, m={self.m}, timesteps={self.timesteps}, "
            f"bi={self.bi}, bj={self.bj}, a={self.a}, alpha={self.alpha}, "
            f"eps={self.eps}, sigma={self.sigma}, zeta={self.zeta}, "
            f"eta={self.eta}, gamma={self.gamma}, G=Graph<{len(self.G.nodes())} nodes>)"
        )


### Class: Agent
class Agent:
    """
    Represents an agent with a state_vector of beliefs, and probabilities for sending/accepting info.

    :param name: Unique identifier (string)
    :param a: Probability of becoming active at each timestep (0 <= a <= 1)
    :param alpha: Influence factor controlling how quickly the agent updates its state vector
    :param bi: Rationality factor for sending (loglinear scaling)
    :param bj: Rationality factor for receiving (logistic scaling)
    :param eps: Probability the sender does NOT receive acknowledgement
    :param state_vector: 1D NumPy array representing the agent's beliefs or "common ground"
    :param local_similarity: Additional metric or info for the agent (float)
    """
    def __init__(
        self,
        name: str,
        a: float,
        alpha: float,
        bi: float,
        bj: float,
        eps: float,
        state_vector: np.ndarray,
        local_similarity: float
    ):
        # Optional: Validate certain parameters
        if not (0.0 <= a <= 1.0):
            raise ValueError("Activation probability (a) should be in [0, 1].")
        # etc...

        self.name = name
        self.a = a
        self.alpha = alpha
        self.bi = bi
        self.bj = bj
        self.eps = eps
        self.state_vector = state_vector
        self.local_similarity = local_similarity
        self.p_send = None
        self.p_accept = None

        self.zeta = 0
        self.accepted = 0

        # Initialize probabilities
        self.update_probabilities()

        # Use logging instead of print, if desired
        logging.debug(f"Agent {self.name} has been created with initial state {self.state_vector}")
        # Or, if you prefer to keep the print:
        # print(f"{self.name} has been created")

    def __str__(self):
        formatting = {"float_kind": lambda x: "%.2f" % x}
        return (
            f"{self.name}\n"
            f"{np.array2string(self.state_vector, precision=2)} <-- State Vector\n"
            f"{np.array2string(self.p_send, formatter=formatting)} <-- P_Send\n"
            f"{np.array2string(self.p_accept, precision=2)} <-- P_Accept"
        )

    def print_xvec(self):
        print(f"{np.array2string(self.state_vector, precision=3)} <-- {self.name}")

    def introduce_agent(self):
        card = {
            "Name": self.name,
            "State Vector": [f"{x:.2f}" for x in self.state_vector],
            "Local Similarity": f"{self.local_similarity:.2f}",
        }
        print(card)

    def set_zeta(self, indicator):
        try:
            if indicator not in (0, 1):
                raise ValueError("Indicator not in {0, 1}")
            self.zeta = indicator
        except ValueError as e:
            print(f"Error: {e}")

    def update_agent_t(self, receiver, sigma: float, gamma: float = 0.0, zeta: float = 0.0, eta: float = 0.5) -> None:
        """
        Attempt to send information to a `receiver`. The chosen piece of information is updated
        based on acceptance or rejection, and the sender (self) also updates its own state vector.

        :param receiver: Another Agent instance, who will possibly accept or reject the info
        :param sigma: Decay factor for the receiver's update on other elements
        :param gamma: Value used when no acknowledgement is received; defaults to 0.0
        :param zeta: Parameter controlling how rejection modifies an agent's certainty
        :param eta: Controls how the receiver adapts upon rejection
        """
        yi_t = np.random.choice(range(len(self.state_vector)), p=self.p_send)

        # Check acceptance
        accepted_flag = (random.random() < receiver.p_accept[yi_t])
        z_j = 1 if accepted_flag else -1
        z_ij = 1 if accepted_flag else -1
        self.accepted = 1 if accepted_flag else -1

        # Update receiver's chosen piece of info
        # Breaking down the formula for clarity
        # partA = ((1 + z_j) / 2) * (1 - receiver.state_vector[yi_t])
        # partB = ((1 - z_j) / 2) * eta * (zeta**2) * (zeta - receiver.state_vector[yi_t])
        # receiver.state_vector[yi_t] += receiver.alpha * (partA + partB)

        receiver.state_vector[yi_t] += receiver.alpha * (
            (1 + z_j) / 2 * (1 - receiver.state_vector[yi_t])
            + (1 - z_j) / 2 * eta * (zeta ** 2) * (zeta - receiver.state_vector[yi_t])
        )

        # Check acknowledgement
        if random.random() < self.eps:
            z_ij = gamma

        # Update sender's chosen piece of info
        self.state_vector[yi_t] = (1 - self.alpha) * self.state_vector[yi_t] + self.alpha * z_ij

        # Decay other elements in receiver's vector
        for i in range(len(receiver.state_vector)):
            if i != yi_t:
                receiver.state_vector[i] = (
                    (1 - receiver.alpha * sigma) * receiver.state_vector[i]
                    - sigma * receiver.alpha
                )

    def reset_accepted(self):
        self.accepted = 0

    def update_probabilities(self): # update the current probability of selection (loglinear learning rule)
        """
        Recalculate the loglinear 'p_send' and logistic 'p_accept' probabilities
        based on the current state vector and rationality parameters.
        """
        # For numerical stability, you might do something like:
        # x = self.bi * self.state_vector
        # x_max = np.max(x)
        # exp_values = np.exp(x - x_max)
        # self.p_send = exp_values / np.sum(exp_values)
        # but the direct approach is usually fine if the values won't explode.
        
        numerator = np.exp(self.bi * self.state_vector)
        self.p_send = numerator / np.sum(numerator)

        # logistic for p_accept
        self.p_accept = 1.0 / (1.0 + np.exp(-self.bj * self.state_vector))


### End Class: Agent


### Class: RecordBook
class RecordBook:
    def __init__(self, agentlist, config):
        self.n = config.n
        self.m = config.m
        self.timesteps = config.timesteps
        self.bi = config.bi
        self.bj = config.bj
        self.a = config.a
        self.alpha = config.alpha
        self.eps = config.eps
        self.sigma = config.sigma
        self.zeta = config.zeta
        self.eta = config.eta
        self.gamma = config.gamma
        self.records = {agent.name: [] for agent in agentlist}
        self.agentlist = agentlist
        self.movingavg = {agent.name: [] for agent in agentlist}
        self.acceptrecord = {t: [0, 0] for t in range(config.timesteps)}
        self.G = config.G
        self.metric_by_edge = {
            (u, v): np.zeros(self.timesteps) for _, (u, v) in enumerate(self.G.edges())
        }
        self.metric_movingavg = {
            (u, v): np.zeros(self.timesteps) for _, (u, v) in enumerate(self.G.edges())
        }

    def record_agent_state(self, agent: Agent) -> None:
        """
        Record the current state vector of the given agent.

        This method appends a copy of `agent.state_vector` to `self.records[agent.name]`,
        effectively creating a time-series record of how the agent's beliefs change
        over the course of the simulation.

        :param agent: The Agent instance whose current state we want to store.
        :type agent: Agent
        :return: None. The agent's state is appended to the RecordBook's 'records' dict.
        :rtype: None
        """
        # Storing a copy so future modifications to the agent’s state do not affect past records
        self.records[agent.name].append(agent.state_vector.copy())

    def compute_moving_average_series(self, avg_window: int) -> None:
        """
        Compute and store the moving average of each agent's state vectors over a sliding
        window of size `avg_window`. The resulting arrays are appended to
        `self.movingavg[agent.name]`.

        For each timestep `t`:
        - If `t >= avg_window`, we sum the agent's last `avg_window` recorded vectors
            (from `self.records[agent.name][t - avg_window : t]`) and then divide by
            `avg_window`.
        - If `t < avg_window`, we store a zero vector of the same length as the state vector
            (because we haven't accumulated `avg_window` data points yet).

        :param avg_window: Number of timesteps used to compute the sliding average (window size).
        :type avg_window: int
        :return: None (results are stored in `self.movingavg`).
        :rtype: None
        """
        for agent in self.agentlist:
            # agent.name is the key in both self.records and self.movingavg
            for t in range(self.timesteps):
                if t >= avg_window:
                    # Optional: Use NumPy sum for performance if each record entry is a NumPy array.
                    slice_data = self.records[agent.name][t - avg_window : t]  # list of np.ndarrays
                    running_total = np.sum(slice_data, axis=0)  # sum along axis=0
                    self.movingavg[agent.name].append(running_total / avg_window)
                else:
                    # For the early timesteps where we don't have avg_window data,
                    # we store a zero vector of the same shape as agent.state_vector
                    zero_vector = np.zeros(len(agent.state_vector))
                    self.movingavg[agent.name].append(zero_vector)
                    
    def compute_moving_average(self, avg_window: int) -> None:
        """
    Compute and store the moving average of each agent's state vectors.
    Only the latest moving average is stored for each agent.
    """
        for agent in self.agentlist:
            if len(self.records[agent.name]) >= avg_window:
                slice_data = self.records[agent.name][-avg_window:]  # Last `avg_window` records
                running_total = np.sum(slice_data, axis=0)  # Sum along axis 0
                self.movingavg[agent.name] = running_total / avg_window  # Replace instead of append
            else:
                zero_vector = np.zeros(len(agent.state_vector))
                self.movingavg[agent.name] = zero_vector

    def add_movingavg_metric(self, avg_window):
        for _, (u, v) in enumerate(self.G.edges()):
            for t in range(self.timesteps):
                if t >= avg_window:
                    running_total = sum(self.metric_by_edge[(u, v)][t - avg_window : t])
                    self.metric_movingavg[(u, v)][t] = running_total / avg_window

    def retrieve_starting_avg(self):
        temp = np.zeros(len(self.agentlist[0].state_vector))
        for agent in self.agentlist:
            temp += self.records[agent.name][0]
        temp /= len(self.agentlist)
        return temp

    def return_acceptance_rates(self):
        temp = np.zeros(self.timesteps)
        for t in range(self.timesteps):
            temp[t] = self.acceptrecord[t][0] / self.acceptrecord[t][1]
        return temp

    def plot_acceptance_rate(self, dir="None", savefig=False, showfig=False):
        temp = np.zeros(self.timesteps)
        for t in range(self.timesteps):
            temp[t] = self.acceptrecord[t][0] / self.acceptrecord[t][1]
        plt.figure(figsize=(12, 6))
        plt.plot(np.arange(0, self.timesteps), temp)
        plt.ylim(-0.1, 1.1)
        plt.title("Acceptance Rate")
        if savefig:
            plt.savefig(os.path.join(dir, "Acceptance rate.png"))
        if showfig:
            plt.draw()
            plt.pause(3)
            plt.close()

    def plot_infodist(self, t, dir="None", savefig=False, BINWIDTH=0.1):
        infos = [
            [self.records[agent.name][t][_] for agent in self.agentlist]
            for _ in range(self.m)
        ]
        for i, info in enumerate(infos):
            bins = np.arange(-1.1, 1.1 + BINWIDTH, BINWIDTH)
            plt.hist(info, bins=bins)
            plt.xlim(-1.1, 1.1)
            plt.xlabel("Information Value")
            plt.ylabel("Frequency")
            plt.title(f"Distribution of information, m={i} at t={t}")
            plt.draw()
            if savefig:
                plt.savefig(os.path.join(dir, f"InfoDist_{i}_fn.png"))
            plt.pause(1)
            plt.clf()
        plt.close()

    def plot_movingavg(self, dir="None", savefig=False, showfig=False):
        for info_index in range(self.m):
            plt.figure(figsize=(15, 9))
            for agentname in self.movingavg:
                moving_avg_values = [
                    record[info_index] for record in self.movingavg[agentname]
                ]
                plt.plot(range(self.timesteps), moving_avg_values, alpha=0.5)
            plt.title(f"Moving Average (Information {info_index})")
            plt.xlabel("Timesteps")
            plt.ylabel(f"Information {info_index} Moving Average")
            plt.ylim((-1.1, 1.1))
            if savefig:
                plt.savefig(os.path.join(dir, f"Info_{info_index}_plot.png"))
            if showfig:
                plt.show()
                plt.close()

    def global_metric(self):
        global_metric = np.zeros(self.timesteps)
        for t in range(self.timesteps):
            sum = 0
            for key in self.metric_by_edge.keys():
                sum += self.metric_by_edge[key][t]
            global_metric[t] = sum / len(self.metric_by_edge)
        return global_metric

    def get_mean_xvec(self):
        mean_xvec = np.zeros(self.m)
        for agent in self.agentlist:
            mean_xvec += self.movingavg[agent.name][self.timesteps - 1]
        return mean_xvec / self.n

    def get_variance(self, mean_xvec):
        Sum = np.zeros(self.m)
        for agent in self.agentlist:
            difference_vector = np.abs(
                self.movingavg[agent.name][self.timesteps - 1] - mean_xvec
            )
            Sum += difference_vector
        Sum = Sum
        return Sum

    def metric_progression(self, numedge=10, dir="None", savefig=False, showfig=False):

        random_edges = random.sample(list(self.G.edges), numedge)
        for edge in random_edges:
            plt.plot(self.metric_by_edge[edge])
            plt.plot(self.metric_movingavg[edge])
            plt.title(
                f"{self.agentlist[edge[0]].name} and {self.agentlist[edge[1]].name}"
            )
            plt.draw()
            if savefig:
                plt.savefig(os.path.join(dir, f"{edge} MP.png"))
            if showfig:
                plt.show()
                plt.close()
            plt.clf()

    def louvain_grid(self, dir="None", savefig=False, showfig=False):
        communities = community_louvain.best_partition(self.G, weight="weight")
        sorted_nodes = sorted(communities.items())
        community_assignments = [community for node, community in sorted_nodes]
        n = int(
            np.ceil(np.sqrt(len(community_assignments)))
        )  # Choose a grid size that fits all nodes
        padded_assignments = community_assignments + [-1] * (
            n * n - len(community_assignments)
        )
        grid = np.array(padded_assignments).reshape(n, n)
        cmap = plt.cm.get_cmap("tab10", np.max(community_assignments) + 1)
        plt.figure(figsize=(8, 8))
        custom_cmap = ListedColormap(
            [
                "#e6194B",  # Red
                "#3cb44b",  # Green
                "#ffe119",  # Yellow
                "#4363d8",  # Blue
                "#f58231",  # Orange
                "#911eb4",  # Purple
                "#42d4f4",  # Cyan
                "#f032e6",  # Magenta
                "#bfef45",  # Lime
                "#fabebe",
            ]
        )
        plt.imshow(grid, cmap=custom_cmap, vmin=0, vmax=9)
        for i in range(n):
            for j in range(n):
                node_index = i * n + j
                if node_index < len(community_assignments):
                    plt.text(
                        j,
                        i,
                        str(sorted_nodes[node_index][0]),
                        ha="center",
                        va="center",
                        color="black",
                    )
        plt.grid(which="both", color="gray", linestyle="-", linewidth=1)
        plt.xticks([])
        plt.yticks([])
        cbar = plt.colorbar(ticks=range(10))
        cbar.ax.set_yticklabels(range(1, 11))
        cbar.ax.set_ylabel("Community")
        plt.draw()
        if savefig:
            plt.savefig(os.path.join(dir, f"{self.m}_louvain_grid.png"))
        if showfig:
            plt.show()
            plt.close()
        plt.clf()

    def surviving_info(self, title, dir="None", savefig=False, showfig=False):
        # Create an empty grid for storing survival information
        grid_data = np.zeros((self.m, self.timesteps // 100))

        # Fill in the grid data
        for t_index, t in enumerate(range(0, self.timesteps, 100)):
            for info in range(self.m):
                tally = 0
                for agent in self.agentlist:
                    tally += self.movingavg[agent.name][t][info]
                avg_value = tally / self.n
                yellowness = max(avg_value, 0)
                yellowness = (np.exp(3 * yellowness) - 1) / (np.exp(3) - 1)
                grid_data[info, t_index] = max(tally / self.n, 0)
                
        # Plot the grid
        plt.figure(figsize=(50, self.m))
        plt.imshow(grid_data, cmap="cividis", aspect="auto", interpolation="none")

        # Set labels and ticks
        plt.title(title)
        plt.xlabel("Timesteps")
        plt.ylabel("m informations")
        plt.xticks(
            ticks=range(self.timesteps // 100), labels=range(0, self.timesteps, 100)
        )
        plt.yticks(ticks=range(self.m))

        plt.grid(False)  # Turn off the default grid
        plt.gca().set_yticks([y + 0.5 for y in range(self.m)], minor=True)
        plt.grid(which="minor", color="black", linestyle="-", linewidth=1)
        plt.draw()  

        # Handle saving and showing the figure
        if savefig:
            plt.savefig(os.path.join(dir, f"m{self.m}surviving_information_plot.png"))

        if showfig:
            plt.show()
            plt.close()  # Close the plot to free memory
        plt.clf()  # Clear the figure to prevent overlap with future plots

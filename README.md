# Common Ground Agent-Based Model (ABM) - Computational Simulation of Information Grounding

This repository contains Python scripts and utilities for running an Agent-Based Model (ABM) designed to simulate how agents in a social network establish common ground—mutually accepted information—through a dynamic process of communication and belief updating.  The model explores how network structure, agent rationality, feedback mechanisms, and memory decay influence the emergence and stability of common ground.

## Model Overview

The ABM simulates a population of agents interacting within a network.  Each agent maintains a "state vector" representing their certainty about various pieces of information.  Grounding is modeled as a probabilistic interaction where agents exchange information, assess acceptance based on their internal belief states, and update their beliefs according to the interaction's outcome (acceptance, rejection, or non-response).

### Agent Representation

*   **`state_vector`:**  A numerical array (NumPy array) representing the agent's certainty about different pieces of information.  Values range from -1 (complete certainty the information is *not* part of the common ground) to +1 (complete certainty it *is* part of the common ground).  The length of this vector is determined by the parameter `m`.
*   **`p_send`:**  A probability distribution derived from the `state_vector` using a log-linear (softmax) function, scaled by the agent's rationality parameter `bi`. It represents the probability of the agent choosing to send each piece of information.
*   **`p_accept`:** A probability distribution derived from the `state_vector` using a logistic function, scaled by the agent's rationality parameter `bj`. It represents the probability of the agent accepting each piece of information.
* **`alpha`:**  The "influence factor," controlling how much an agent's `state_vector` changes after an interaction.
*   **Initialization:** Agents' state vectors are initialized with values drawn from a Beta distribution, transformed to the range [-1, 1], to represent initial uncertainty or bias.

### Network Structure

The ABM uses the NetworkX library to represent the social network. By default, it uses a `powerlaw_cluster_graph`, which is a type of scale-free network with a high clustering coefficient, often found in real-world social networks. A dictionary, `neighbors_dict`, is used to provide efficient lookup of each agent's neighbors.

### Interaction Process (Grounding - Computational Details)

At each timestep, the following interaction process occurs:

1.  **Sender Selection:**  Each agent has a probability `a` (activation probability) of being selected as a sender (denoted as agent `i`).

2.  **Receiver Selection:** If an agent is selected as a sender, it randomly chooses one of its neighbors (denoted as agent `j`) from the network.

3.  **Information Selection:** The sender (`i`) selects a piece of information (`yi_t`) from its `state_vector` to transmit. The selection is probabilistic, based on the `p_send` distribution:

    ```
    P[yi(t) = k] = exp(bi * xi,k(t)) / sum(exp(bi * xi,l(t)))  for l = 1 to m
    ```

    where `k` is the index of the information piece, `xi,k(t)` is the sender's certainty about information `k` at time `t`, and `bi` is the sender's rationality parameter.

4.  **Acceptance Decision:** The receiver (`j`) decides whether to accept or reject the information based on its `p_accept` distribution:

    ```
    P[zj(t) = 1] = 1 / (1 + exp(-bj * xj,yi(t)(t)))
    ```

    where `zj(t) = 1` indicates acceptance, `zj(t) = -1` indicates rejection, `xj,yi(t)(t)` is the receiver's certainty about the received information at time `t`, and `bj` is the receiver's rationality parameter.

5.  **Feedback:** The receiver sends feedback (`zij`) to the sender:

    *   With probability `1 - eps`, the receiver truthfully reveals its decision: `zij(t) = zj(t)`.
    *   With probability `eps`, the receiver provides no feedback, and `zij(t) = gamma`.  The parameter `gamma` represents the sender's interpretation of silence.

6.  **State Vector Update:** Both the sender (`i`) and receiver (`j`) update their `state_vector` at the index corresponding to the transmitted information (`yi_t`). The update rules are:

    ```
    xi,yi(t)(t+1) = (1 - alpha) * xi,yi(t)(t) + alpha * zij(t)
    xj,yi(t)(t+1) = (1 - alpha) * xj,yi(t)(t) + alpha * f(zj(t), xj,yi(t)(t))
    ```
    where `alpha` is the influence factor. The function `f(z, x)` depends on the parameter `zeta`:
    
    -   If `zeta = 0`: `f(z, x) = 1 if z = 1, else x`
    -   If `zeta != 0`:
        `f(z, x) = (1 if z = 1) else (eta * (zeta**2) * (zeta - x))`

    The parameter `eta` controls the magnitude of change when the receiver rejects a grounding attempt and zeta !=0

7. **Decay:** Elements of the *receiver's* state vector *other than* the communicated piece of information (`yi_t`) decay:
    ```
    xj,l(t+1) = (1 - alpha * sigma) * xj,l(t) - sigma * alpha   for all l != yi(t)
    ```
     where sigma is the decay rate.

### Moving Average

The ABM also calculates a moving average of the agents' state vectors over a sliding window of size `int(timesteps * 0.04)`. This provides a smoothed representation of the agents' beliefs over time, reducing the impact of short-term fluctuations.

## Parameters

| Parameter       | Data Type      | Default Value | Function                                                                                                                                                                  |
| :-------------- | :------------- | :------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `n`             | int            | 500           | Number of agents in the simulation.                                                                                                                                       |
| `m`             | int            | 2             | Length of the state vector (number of pieces of information).                                                                                                             |
| `timesteps`     | int            | 5000          | Number of timesteps in the simulation.                                                                                                                                    |
| `bi`            | float          | 5.0           | Rationality parameter for the sender (used in the log-linear `p_send` calculation).                                                                                       |
| `bj`            | float          | 5.0           | Rationality parameter for the receiver (used in the logistic `p_accept` calculation).                                                                                     |
| `a`             | float          | 0.5           | Activation probability.  Probability of an agent being selected as a sender in each timestep.                                                                             |
| `alpha`         | float          | 0.2           | Influence factor. Determines the magnitude of state vector updates after an interaction (Step 6).                                                                         |
| `eps`           | float          | 0.1           | Probability that the receiver does *not* provide feedback.                                                                                                                |
| `sigma`         | float          | 0.05          | Decay factor. Controls the rate at which certainty about non-communicated information decreases in the receiver's state vector (Step 7).                                  |
| `zeta`          | int            | 0             | Determines how the receiver's certainty changes upon rejection.  If 0, certainty doesn't change on rejection. If -1 or 1, certainty decreases or increases, respectively. |
| `eta`           | float          | 0.5           | Determines the magnitude of change when `zeta` is not 0.                                                                                                                  |
| `gamma`         | float          | -1.0          | Sender's interpretation of non-response.  The value assigned to `zij` when the receiver doesn't provide feedback.                                                         |
| `G`             | NetworkX Graph | None          | The network graph representing connections between agents. If None, a power-law cluster graph is generated.                                                               |
| `metric_method` | string         | "pdiff"       | Determines the method of calculating the distance/similarity between agents.                                                                                              |
| `alpha_dist`    | string         | "static"      | The distribution from which the alpha value is drawn. Can be "static", "uniform" or "beta".                                                                               |

## Repository Structure

```
danjcs-commonground/
├── common_ground_scripts/
│   ├── README.md                     # This README file
│   ├── CG_source.py                  # Defines core classes (`Agent`, `Config`, `RecordBook`) and functions.
│   ├── analyze_clusters.py           # Performs cluster analysis (HDBSCAN) on simulation results.
│   ├── analyze_two_params.py         # Generates heatmaps to analyze the interplay of two parameters on surviving info.
│   ├── cluster_analysis.py           # Exports clustered networks to GEXF for Gephi visualization.
│   ├── config.py                     # Defines simulation parameters and parameter grid for sweeps.
│   ├── generate_graph_library.py     # Generates power-law cluster graphs.
│   ├── interpreter.py                # Produces scatter plots of agent state vectors.
│   ├── natural_simulation.py         # Main simulation engine. Runs the ABM.
│   ├── pairwise_similarity.py        # Computes pairwise distance/similarity matrices.
│   ├── plot_distance_matrix.py       # Plots heatmaps of the pairwise distance matrix.
│   ├── plot_surviving_by_m.py        # Plots surviving information vs. state vector length (m).
│   ├── plot_utils.py                 # Plotting utility functions.
│   ├── plot_violin.py                # Generates violin/box plots of state vector value distributions.
│   ├── run.py                        # Runs a single simulation.
│   ├── run_parallel_simulation.py    # Runs multiple simulations in parallel.
│   ├── sweep.py                      # Runs a parameter sweep.
│   ├── survival_per_parameter.py     # Analyzes the effect of a parameter on surviving information.
│   ├── surviving_information.py      # Calculates which pieces of information "survive".
└── utility_scripts/
    ├── find_beta.py                  # Analyzes how p_send changes with different beta values.
    ├── find_optimal_temperature.py   # Finds optimal softmax temperature via Jensen-Shannon Divergence.
    └── hdbscan_gridsearch.py         # Grid search over HDBSCAN hyperparameters.
```

## Getting Started

### 1. Dependencies

- Python 3.7+
- Libraries: `numpy`, `matplotlib`, `networkx`, `hdbscan`, `scipy`, `scikit-learn`, `tqdm`, `python-louvain`, `pandas`, `seaborn`
- Install via:

  ```bash
  pip install numpy matplotlib networkx hdbscan scikit-learn scipy tqdm python-louvain pandas seaborn
  ```

### 2. Generating Graphs (Optional)

If you need to generate random power-law cluster graphs:

```bash
cd common_ground_scripts
python generate_graph_library.py
```

This creates pickled graph files in the `graphs_library_high_cluster` directory (by default).

### 3. Running Simulations

*   **Single Run:**

    ```bash
    python common_ground_scripts/run.py  # Uses parameters from config.py
    ```

*   **Parameter Sweep (Parallel):**

    ```bash
    python common_ground_scripts/run_parallel_simulation.py <input_dir> <output_dir> --processes <num_processes> --repetitions <num_repetitions>
    ```

    *   `<input_dir>`: Directory containing input data (can be empty if using the default graph generation).
    *   `<output_dir>`: Directory to save simulation results.
    *   `--processes`:  Number of parallel processes to use (defaults to the number of CPU cores if not provided).
    *   `--repetitions`: Number of simulation runs for each parameter combination.
  The parameter grid is defined in `common_ground_scripts/config.py`.

### 4. Post-Processing & Analysis (Examples)
- **analyze_clusters.py**
  ```bash
  python common_ground_scripts/analyze_clusters.py <input_dir> --method jsd --min_cluster_size 50 --min_samples 10 --n_jobs 4
  ```

- **analyze_two_params.py**
  ```bash
  python common_ground_scripts/analyze_two_params.py <input_dir> param1 param2 <output_dir> --threshold 0.5 --fraction 0.1
  ```

- **surviving_information.py**
  ```bash
    python common_ground_scripts/surviving_information.py <results_dir> --threshold 0.5 --fraction 0.1
  ```

## Script Interdependencies

-   **`CG_source.py`:**  Foundation for all other scripts.  Defines `Config`, `Agent`, `RecordBook`, and helper functions.
-   **`natural_simulation.py`:** Core simulation driver.  Uses `CG_source.py`.  Called by `run.py`, `run_parallel_simulation.py`, `sweep.py`, and `simulation_record.py`.
-   **`run_parallel_simulation.py`:** Uses `natural_simulation.py` for parallel execution.
-   **`sweep.py`:** Uses `natural_simulation.py` and `config.py` for parameter sweeps.
-   **`pairwise_similarity.py`:**  Computes distance/similarity matrices.  Used by `analyze_clusters.py`, `hdbscan_gridsearch.py`, and `plot_distance_matrix.py`.
-   **`analyze_clusters.py`:** Uses `pairwise_similarity.py` and `plot_utils.py`.
-   **`hdbscan_gridsearch.py`:**  Uses `pairwise_similarity.py`.
-   **`analyze_two_params.py`:**  Uses `surviving_information.py`.
-   **`plot_surviving_by_m.py`:** Uses `surviving_information.py`.
-   **`surviving_information.py`:**  Provides functions for analyzing information survival.
-   **`plot_utils.py`:**  General plotting utilities.
- **`interpreter.py`**: Used for visualizing simulation output at specific timesteps.
-   **`plot_distance_matrix.py`:** Uses `pairwise_similarity.py`.
-   **`plot_violin.py`:**  Generates violin plots.
-  **`utility_scripts/*`**:  Contains helper scripts for specific analyses (e.g., finding optimal parameters).

## Future Directions

*   **Network Dynamics:** Explore adding mechanisms for dynamic network rewiring, where agent connections change over time based on interaction outcomes.
*   **Heterogeneous Agents:** Extend the model to allow for agents with different initial belief distributions (beyond the current Beta distribution) or learning parameters (e.g., different `alpha` values).
* **Visualisation of simulation**: Enhance visualisations to enable better tracking of model dynamics.
* **Sensitivity analysis**: Perform sensitivity analyses on the different model parameters.
* **Expanded Graph Support**: Support more graph types.

---

fin.

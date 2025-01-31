# Common Ground Agent-Based Model (ABM)

This repository contains Python scripts and utilities for running an Agent-Based Model (ABM) designed to study how common ground—mutually accepted information among agents—emerges and evolves over time. The scripts support simulations, parallel parameter sweeps, post-simulation analyses, and visualizations. Common ground forms through a process called "grounding," where agents exchange information dynamically, refining their shared understanding of what is collectively accepted.

### Overview of the Model

Our ABM simulates 500 agents interacting within a Holme-Kim network over T timesteps. Each agent maintains a state vector (sometimes referred to as an x-vector), representing its perception of the common ground. At each timestep, an agent may be selected as a sender (we will denote as $i$) with some probability. The sender then chooses one of its neighbors as a receiver (we will denote as $j$) and transmits a piece of information. The receiver can either accept or reject the information. Depending on a probability factor, the receiver may also provide feedback to the sender regarding the acceptance decision, leading both agents to update their state vectors accordingly.

As interactions continue, state vectors among neighboring agents gradually converge. When this convergence reaches a sufficient threshold, we consider that a common ground has been established among those agents.


### Parameters

| Variable      | Function                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $n$           | Number of agents in the system. $n=500$ for our research paper.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| $m$           | The length of the state vector (i.e. the complexity/depth of common ground)                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| $\beta$       | The level of rationality of the agent. As $\beta \to \infty$, agents strictly select the highest certainty information.<br>This parameter is also used in the acceptance equation. This is to be discussed further. <br>Currently, $\beta=7$, which makes agents exhibit highly rational behaviour.                                                                                                                                                                                                                                                     |
| $a$           | Activation probability. At each timestep, any agent has $a$ probability of acting as a sender. Currently, $a=0.5$.                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| $\alpha$      | The degree of influence of each response. In other words, it determines how quickly agents adopt a new level of certainty, based on the outcome of the grounding attempt. Currently, $\alpha=0.2$. We may discuss using a lower value.<br>**It seems to inversely correlate with the number of surviving information.**                                                                                                                                                                                                                                 |
| $\varepsilon$ | Determines the probability of providing a response back to the sender. Following a grounding attempt by the sender, $i$, the receiver, $j$ has $1-\varepsilon$ probability of revealing to $i$ whether or not the grounding attempt was successful.<br>This is one-half of our variables of interest.                                                                                                                                                                                                                                                   |
| $\gamma$      | Determines how the lack of response is processed by $i$. It is the equilibrium point to which $i$'s level of certainty devolves upon repeated silence.                                                                                                                                                                                                                                                                                                                                                                                                  |
| $\sigma$      | Determines the decay rate. Simulates the decaying of memory, as well as to signify that any information that has been "left out" for a long period of time is considered to be obsolete. Given a fixed number of timesteps and that each agent may only select one piece of information per timestep, the amount of time taken to cycle through the common ground is proportional to $m$, effectively making the progress "slower". Thus, it makes sense to also lower the parameter which reduces the certainty level per timestep by a fixed portion. |
| $\zeta$       | $\{ -1,0,1 \}$, indicates whether $j$'s certainty decreases, increases or remains the same after rejecting a grounding attempt.                                                                                                                                                                                                                                                                                                                                                                                                                         |
| $\eta$        | Determines the magnitude of change when $\zeta \neq 0$                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |

---

## Project Objectives

The core goal of this project is to simulate a network of agents (participants in a social interaction) and track how their shared information—**“common ground”**—becomes established. By modeling how beliefs, information, and acceptance thresholds shift, we can investigate which pieces of information “survive” in the population, how stable consensus forms, and how social network structures affect these processes.

In brief:

- **Simulate Agents** that update their beliefs based on interactions.
- **Explore “Survival”** of specific pieces of information, measuring how likely they are to become part of the population-wide common ground. We are yet to conclusively define what exactly constitutes "survival". This will be done in the sequel.
- **Study Network Influence** by generating or loading various network structures (e.g., power-law cluster graphs).
- **Benchmark** different ABM parameters (like agent rationality, network connectivity, memory decay, etc.) and see their effects on emergent outcomes. In particular, we are interested in the gamma and epsilon parameters.

### Immediate tasks:

- To determine a stable hyperparameter values (mcs and ms) for HDBSCAN
- Use HDBSCAN to partition the agents to clusters that share a common ground
- Visualise the partitioned network using Pyvis. Colour-code different clusters and study where/how many clusters occur
- Conduct analyses to study how well-separated the clusters are.

---

## Repository Structure

```
danjcs-commonground/
├── common_ground_scripts/
│   ├── README.md
│   ├── CG_source.py
│   ├── analyze_clusters.py
│   ├── analyze_two_params.py
│   ├── generate_graph_library.py
│   ├── natural_simulation.py
│   ├── pairwise_similarity.py
│   ├── plot_surviving_by_m.py
│   ├── plot_utils.py
│   ├── run_parallel_simulation.py
│   └── surviving_information.py
└── utility_scripts/
    ├── find_optimal_temperature.py
    └── hdbscan_gridsearch.py
```

1. **common\_ground\_scripts/**

   - **CG\_source.py**\
     Defines the **foundational classes** and **core functions** for the ABM:

     - `Config` (stores ABM parameters).
     - `Agent` (represents each agent’s beliefs, sending behavior, etc.).
     - `RecordBook` (records simulation data).\
       Many scripts in this folder depend on `CG_source.py`.

   - **analyze\_clusters.py**\
     Clusters final agent states using **HDBSCAN** to see how many clusters of agent beliefs form under different hyperparameters. Relies on **pairwise\_similarity.py** (distance matrices) and **plot\_utils.py** (visualization).

   - **analyze\_two\_params.py**\
     Analyzes **two** chosen parameters simultaneously, generating heatmaps to visualize how different pairs of parameter values affect “surviving” information. Depends on `surviving_information.py` to count which pieces of information survived.

   - **generate\_graph\_library.py**\
     Generates and pickles multiple **power-law cluster graphs**. Used for producing a library of random network structures to feed into the simulations.

   - **natural\_simulation.py**\
     The **main simulation driver**. Contains methods for creating agents (`create_agent_list`), loading random graphs (`load_random_graph`), and running single or multiple ABM simulations (`run_simulation_with_params` & `parameter_sweep`). Depends heavily on `CG_source.py` for agent/config definitions.

   - **pairwise\_similarity.py**\
     Builds NxN distance (or similarity) matrices of agent states (often final states) using various metrics (Jensen-Shannon Divergence, Euclidean, Cosine). Used by scripts like **analyze\_clusters.py** and others that require distance computations for clustering or analysis.

   - **plot\_surviving\_by\_m.py**\
     Loads simulation outputs and produces **line plots** illustrating how many pieces of information “survive” as the dimension `m` (length of the agent’s state vector) varies, grouped by other parameters.

   - **plot\_utils.py**\
     A small collection of **plotting utilities** used for HDBSCAN grid search plots and for comparing cluster metrics vs. ABM parameters.

   - **run\_parallel\_simulation.py**\
     Orchestrates **parallel** parameter sweeps of the ABM. Uses `multiprocessing.Pool` to run multiple simulations concurrently, each with different parameters. Depends on `natural_simulation.py` and `CG_source.py`.

   - **surviving\_information.py**\
     Provides functions to **analyze which pieces of information** survive in the final agent states. It loads JSON outputs from simulations, checks a threshold, and tallies how many pieces are shared by at least a specified fraction of agents.

2. **utility\_scripts/**

   - **find\_optimal\_temperature.py**\
     Helps in determining a suitable temperature parameter (for softmax conversions) by computing Jensen-Shannon Divergence (JSD) across a range of temperatures, enabling a user to pick a sweet spot.

   - **hdbscan\_gridsearch.py**\
     Performs a more general **grid search** over HDBSCAN hyperparameters (min\_cluster\_size, min\_samples), computing silhouette scores. Summarizes which hyperparameter set yields the best average clustering performance for each ABM parameter set. Uses `pairwise_similarity.py` for distance matrices.

---

## Script Interdependencies

- **CG\_source.py**: Foundational script. Nearly every other script relies on **Agent**, **Config**, or **RecordBook** definitions from here.
- **natural\_simulation.py**: Main simulation driver. Imports `Config`, `Agent`, `RecordBook` from `CG_source.py`.
- **run\_parallel\_simulation.py**: Orchestrates simulation parameter sweeps in parallel, calling `run_simulation_with_params` from `natural_simulation.py`.
- **surviving\_information.py**: Summarizes how many pieces of info survive in the final states. It’s used by:
  - **analyze\_two\_params.py** (importing `count_surviving_info`)
  - **plot\_surviving\_by\_m.py** (importing `count_surviving_info`)
- **pairwise\_similarity.py**: Builds pairwise distance matrices from final agent states. Depended on by:
  - **analyze\_clusters.py** and **hdbscan\_gridsearch.py**, which use these distance matrices for HDBSCAN clustering.
- **plot\_utils.py**: Visualization helpers for cluster analysis. Called by **analyze\_clusters.py**, **hdbscan\_gridsearch.py**, etc.
- **find\_optimal\_temperature.py**: Independent script that can be used after simulations to explore how temperature scaling affects distribution shape and Jensen-Shannon Divergence across agent vectors.

---

## Getting Started

### 1. Dependencies

- Python 3.7+
- Required libraries:
  - `numpy`, `matplotlib`, `networkx`, `hdbscan`, `scipy`, `scikit-learn`, `tqdm`, `python-louvain`
- Install dependencies via:
  ```bash
  pip install numpy matplotlib networkx hdbscan scikit-learn scipy tqdm python-louvain
  ```

### 2. Generating Graphs (Optional)

If you need random power-law cluster graphs for your simulations, run:

```bash
cd common_ground_scripts
python generate_graph_library.py
```

This will produce pickled graph files in the specified output directory (default `graphs_library_high_cluster`).

### 3. Running a Basic Simulation

- You can directly call:

  ```bash
  python natural_simulation.py
  ```

  which, in its current form, may run a parameter sweep defined at the bottom of the script.

- For large parameter sweeps, **run\_parallel\_simulation.py** is typically used:

  ```bash
  python run_parallel_simulation.py <input_dir> <output_dir> --processes 8 --repetitions 50
  ```

  Here, `<input_dir>` can contain pickled graphs if needed, or you can define a parameter grid within the script.

### 4. Post-Processing & Analysis

- **analyze\_clusters.py**

  ```bash
  python analyze_clusters.py <input_dir> --method jsd --min_cluster_size 50 --min_samples 10 --n_jobs 4
  ```

  Finds how many clusters result from each simulation’s final state, possibly grouped by an ABM parameter.

- **analyze\_two\_params.py**

  ```bash
  python analyze_two_params.py <input_dir> param1 param2 <output_dir> --threshold 0.5 --fraction 0.1
  ```

  Produces heatmaps illustrating how two chosen parameters (e.g., `alpha`, `gamma`) jointly affect survival of information.

- **plot\_surviving\_by\_m.py**

  ```bash
  python plot_surviving_by_m.py <results_dir> <output_dir> [--threshold 0.5] [--fraction 0.1]
  ```

  Generates a line plot of survival vs. `m`, grouping by other parameters.

- **pairwise\_similarity.py**

  ```bash
  python pairwise_similarity.py <input_dir> --method jsd --temperature 1.0 --output_dir dist_matrices --n_jobs 4
  ```

  Creates NxN distance matrices from each JSON result, saves them as `.npy` in `dist_matrices`, for subsequent clustering or other analysis.

- **surviving\_information.py**

  ```bash
  python surviving_information.py <results_dir> --threshold 0.5 --fraction 0.1
  ```

  Prints how many pieces of information survived in each run, grouped by parameters.

- **find\_optimal\_temperature.py** (in `utility_scripts/`)

  ```bash
  python find_optimal_temperature.py my_result.json
  ```

  Explores how the softmax temperature influences distribution shapes by computing Jensen-Shannon Divergence at various temperatures.

- **hdbscan\_gridsearch.py** (in `utility_scripts/`)

  ```bash
  python hdbscan_gridsearch.py <input_dir> --method jsd --temperature 1.0 --n_jobs 4
  ```

  Performs a grid search over HDBSCAN hyperparameters across all simulation results in `<input_dir>`.

---

## Future Directions

- **Modular Enhancements**: We will continue refining the ABM (e.g., adding new update rules or agent features).
- **Expanded Graph Support**: Possibly create more custom graphs or import real social network data.
- **Visualization**: Enhance plotting scripts to produce interactive dashboards or more sophisticated visual analytics.
- **Documentation**: This README (v1) will continue to be updated as more features are introduced.

---

fin.



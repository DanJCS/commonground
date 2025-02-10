# Common Ground Agent-Based Model (ABM)

This repository contains Python scripts and utilities for running an Agent-Based Model (ABM) designed to study how common ground—mutually accepted information among agents—emerges and evolves over time. The scripts support simulations, parallel parameter sweeps, post-simulation analyses, and visualizations. Common ground forms through a process called "grounding," where agents exchange information dynamically, refining their shared understanding of what is collectively accepted.

*Common ground is the set of mutually accepted information and provides a context to the individuals engaged in a social activity

### Overview of the Model

Our ABM simulates 500 agents interacting within a Holme-Kim network over T timesteps. Each agent maintains a state vector (sometimes referred to as an x-vector), representing its perception of the common ground. At each timestep, an agent may be selected as a sender (we will denote as $i$) with some probability. The sender then chooses one of its neighbors as a receiver (we will denote as $j$) and transmits a piece of information. The receiver can either accept or reject the information. Depending on a probability factor, the receiver may also provide feedback to the sender regarding the acceptance decision, leading both agents to update their state vectors accordingly.

As interactions continue, state vectors among neighboring agents gradually converge. When this convergence reaches a sufficient threshold, we consider that a common ground has been established among those agents.


### Parameters

| Variable      | Function                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $n$           | Number of agents in the system. $n=500$ for our research paper.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| $m$           | The length of the state vector (i.e. the complexity/depth of common ground)                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| $\beta$       | The level of rationality of the agent. As $\beta \to \infty$, agents strictly select the highest certainty information.<br>This parameter is also used in the acceptance equation. This is to be discussed further. <br>Currently, $\beta=4$, which allows agents to favour high-certainty information.                                                                                                                                                                                                                                              |
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

---

## Repository Structure

```
danjcs-commonground/
├── common_ground_scripts/
│   ├── README.md                     # This README file
│   ├── CG_source.py                  # Foundational classes & functions (Config, Agent, RecordBook)
│   ├── analyze_clusters.py           # Clusters final agent states using HDBSCAN
│   ├── analyze_two_params.py         # Generates heatmaps to analyze the interplay of two parameters on surviving info
│   ├── cluster_analysis.py           # Exports clustered networks (with HDBSCAN) to GEXF for Gephi visualization
│   ├── config.py                     # Defines simulation parameters and parameter grid for sweeps
│   ├── generate_graph_library.py     # Generates and pickles power-law cluster graphs for simulation input
│   ├── interpreter.py                # Produces scatter plots of agent state vectors at selected timesteps
│   ├── natural_simulation.py         # Main simulation driver (creates agents, runs simulations, computes moving averages)
│   ├── pairwise_similarity.py        # Builds NxN distance/similarity matrices from final agent states
│   ├── plot_distance_matrix.py       # Plots heatmaps of the pairwise distance matrix using Seaborn
│   ├── plot_surviving_by_m.py        # Plots how surviving information changes with state vector length (m)
│   ├── plot_utils.py                 # Contains plotting utility functions (e.g., for grid search and cluster vs. parameter plots)
│   ├── plot_violin.py                # Generates violin/box plots of the distributions of agent state vector values
│   ├── run.py                        # Single-run simulation driver (using parameters from config.py)
│   ├── run_parallel_simulation.py    # Orchestrates parallel parameter sweeps using multiprocessing
│   ├── simulation_record.py          # Runs a simulation and records the full time series of agent state vectors
│   ├── sweep.py                      # Runs a parameter sweep based on the grid defined in config.py
│   ├── survival_per_parameter.py     # Analyzes the effect of a chosen parameter on the proportion of surviving info
│   └── surviving_information.py      # Computes and aggregates counts of surviving pieces of information
└── utility_scripts/
├── find_beta.py                  # Analyzes how p_send changes with different beta (rationality) values
├── find_optimal_temperature.py   # Helps identify an optimal softmax temperature via Jensen-Shannon Divergence
└── hdbscan_gridsearch.py         # Performs a grid search over HDBSCAN hyperparameters for optimal clustering
```

### Detailed Script Overview

**common_ground_scripts/README.md**  
- This file (the README) provides an overall description of the project, model overview, parameter definitions, objectives, repository structure, and instructions for running simulations and analyses.

**common_ground_scripts/CG_source.py**  
- Provides the foundational classes and helper functions for the ABM.  
- Defines the `Config` class (for storing simulation parameters), the `Agent` class (which models individual agents including their state vectors and probability distributions for sending/accepting information), and the `RecordBook` class (which records the evolution of agent states and various metrics during simulations).  
- Includes multiple metric functions to compare agents’ state vectors.

**common_ground_scripts/analyze_clusters.py**  
- Loads simulation result JSON files and computes pairwise distance matrices between agents’ final state vectors.  
- Applies HDBSCAN clustering (using user-specified hyperparameters) to the distance matrices to determine the number of clusters formed (ignoring noise).  
- Optionally groups results by a chosen ABM parameter and generates plots to visualize the clustering behavior.  
- Supports parallel processing for efficiency.

**common_ground_scripts/analyze_two_params.py**  
- Examines the joint influence of two selected ABM parameters on the survival of information.  
- Loads simulation results, computes survival counts using criteria defined in `surviving_information.py`, and groups results by the remaining parameters.  
- Generates 2D heatmaps where one parameter is plotted on the X-axis and the other on the Y-axis, with each cell showing the mean (and variance) of surviving information counts.

**common_ground_scripts/cluster_analysis.py**  
- Performs clustering analysis on simulation outputs by applying HDBSCAN to the final moving average state vectors.  
- Exports the underlying network (with nodes annotated with cluster labels and color-coded using a Matplotlib colormap) to a GEXF file for visualization in tools like Gephi.  
- Also creates a text summary detailing the number of clusters (excluding noise), the number of noise nodes, and a mapping of clusters to colors.

**common_ground_scripts/config.py**  
- Defines the default simulation parameters (e.g., number of agents, state vector length, timesteps, and behavioral parameters) for a single-run simulation.  
- Also defines a parameter grid used for parameter sweeps.  
- Provides a helper function to resolve parameter values (which can be static or callable).

**common_ground_scripts/generate_graph_library.py**  
- Generates a library of power-law cluster graphs using NetworkX, then pickles (saves) them for later use in simulations.  
- Allows customization of graph properties such as the number of nodes, clustering factor, and connection probability.

**common_ground_scripts/interpreter.py**  
- Interprets simulation result JSON files to extract agent state vectors.  
- Generates scatter plots of these state vectors at user-specified timesteps, supporting both raw and moving average data, and can also apply softmax transformation.  
- Facilitates visual exploration of the simulation’s evolution over time.

**common_ground_scripts/natural_simulation.py**  
- Serves as the main simulation driver for the ABM.  
- Creates agents with initialized state vectors, loads or generates a network graph, and simulates interactions over many timesteps.  
- Records the evolution of agent states (both raw and moving average) and outputs the final results (and optionally the full time series) as a JSON file.

**common_ground_scripts/pairwise_similarity.py**  
- Converts final agent state vectors into probability distributions using a softmax function (with an adjustable temperature).  
- Computes NxN pairwise distance (or similarity) matrices using methods such as Jensen-Shannon Divergence, Euclidean distance, or cosine similarity.  
- Supports parallel processing and can optionally save the computed matrices as `.npy` files.

**common_ground_scripts/plot_distance_matrix.py**  
- Loads a simulation result JSON file and uses `pairwise_similarity.py` to compute the distance matrix for the final moving average state vectors.  
- Creates a heatmap of the pairwise distances using Seaborn, and either displays the plot interactively or saves it to a file.

**common_ground_scripts/plot_surviving_by_m.py**  
- Analyzes how the number of surviving pieces of information changes with the length of the state vector (`m`).  
- Loads simulation outputs, computes survival counts (using criteria from `surviving_information.py`), and groups the data by other parameters.  
- Generates line plots with 90% confidence intervals (and an optional linear fit) showing survival trends as `m` varies.

**common_ground_scripts/plot_utils.py**  
- Contains a collection of plotting utility functions used across the project.  
- Includes functions for plotting heatmaps (e.g., from HDBSCAN grid search results) and line plots that relate ABM parameters to clustering metrics.

**common_ground_scripts/plot_violin.py**  
- Reads a simulation result JSON file and extracts the final moving average state vectors.  
- Generates violin (or box) plots to visualize the distribution of certainty values for each dimension of the state vector using Seaborn.

**common_ground_scripts/run.py**  
- A simple, single-run simulation driver that loads simulation parameters from `config.py`.  
- Runs the simulation (via `natural_simulation.py`) and saves the final simulation result as a JSON file.  
- Allows the user to specify a repetition index and output filename.

**common_ground_scripts/run_parallel_simulation.py**  
- Orchestrates parallel parameter sweeps by distributing simulation runs across multiple CPU cores using Python’s multiprocessing.  
- Uses `run_simulation_with_params` from `natural_simulation.py` to run each simulation concurrently, saving each result as a JSON file.

**common_ground_scripts/simulation_record.py**  
- Runs a single simulation while recording the full time series of agent state vectors.  
- Outputs a JSON file that includes simulation parameters, final state vectors, moving averages, and the complete raw time series data.  
- Intended for detailed post-simulation analyses.

**common_ground_scripts/sweep.py**  
- Executes a parameter sweep based on the parameter grid defined in `config.py`.  
- For each combination of parameters and for each repetition, runs a simulation and saves the output as a JSON file.  
- Facilitates systematic exploration of the model’s behavior across different parameter settings.

**common_ground_scripts/survival_per_parameter.py**  
- Analyzes how a chosen “interesting parameter” affects the proportion of surviving information.  
- Loads simulation results, computes survival counts, and calculates the proportion of surviving pieces relative to the state vector length (`m`).  
- Groups results (and can further group by a secondary parameter) and generates a line plot with 95% confidence intervals.

**common_ground_scripts/surviving_information.py**  
- Provides functions to determine which pieces of information “survive” in the final agent state vectors.  
- Uses a survival criterion based on a threshold and a minimum fraction of agents.  
- Aggregates survival counts across simulation runs and supports grouping by parameter sets.

**utility_scripts/find_beta.py**  
- Analyzes how the agent’s sending probability distribution (`p_send`) changes as the beta (rationality) parameter varies.  
- Creates a dummy agent, computes `p_send` for a range of beta values, and prints the results with values rounded to three significant figures.

**utility_scripts/find_optimal_temperature.py**  
- Helps determine an optimal softmax temperature for converting state vectors into probability distributions.  
- Loads a JSON file, selects random state vectors, applies softmax over a range of temperatures, and computes pairwise Jensen-Shannon Divergence (JSD).  
- Produces plots of probability distributions and average JSD versus temperature, along with suggestions for selecting the optimal temperature.

**utility_scripts/hdbscan_gridsearch.py**  
- Performs a grid search over HDBSCAN hyperparameters (min_cluster_size and min_samples) for each unique ABM parameter set found in the simulation results.  
- Computes silhouette scores and cluster counts for each parameter combination, aggregates the results, and prints a summary of the best hyperparameters based on the mean silhouette score.

---

## Script Interdependencies

- **CG_source.py:**  
  This foundational script defines key classes—such as `Config`, `Agent`, and `RecordBook`—as well as various helper functions and metrics for comparing agent state vectors. Nearly every other script depends on these definitions for simulation setup and agent behavior.

- **natural_simulation.py:**  
  Serves as the core simulation driver. It imports definitions from `CG_source.py` to create agents, initialize the network, simulate interactions over timesteps, and record both raw and moving average state vectors. Many scripts (including single-run, parallel, and sweep drivers) invoke functions from this module to run simulations.

- **run_parallel_simulation.py:**  
  Leverages `natural_simulation.py` (and by extension `CG_source.py`) to run multiple simulations concurrently via Python’s multiprocessing. It enables parallel parameter sweeps and efficiently generates numerous simulation outputs.

- **sweep.py:**  
  Uses the parameter grid defined in `config.py` to systematically explore different simulation configurations. For each combination, it calls `run_simulation_with_params` from `natural_simulation.py` and saves the output, facilitating a comprehensive parameter sweep.

- **simulation_record.py:**  
  Similar to `run.py`, but with added functionality to record the complete time series (raw state vectors) for each agent. This detailed output is useful for in-depth post-simulation analysis.

- **pairwise_similarity.py:**  
  Converts agents’ final state vectors into probability distributions (using a softmax function with an adjustable temperature) and computes NxN distance or similarity matrices using metrics such as Jensen-Shannon Divergence, Euclidean distance, or cosine similarity. Several analysis and visualization scripts depend on these matrices.

- **analyze_clusters.py:**  
  Imports functions from `pairwise_similarity.py` to build distance matrices and then applies HDBSCAN clustering to determine how many clusters are formed in the final agent states (ignoring noise). It also uses helper functions from `plot_utils.py` for visualization.

- **hdbscan_gridsearch.py:**  
  Performs a grid search over HDBSCAN hyperparameters (e.g., min_cluster_size and min_samples) for each unique set of simulation parameters. It uses the distance matrices computed by `pairwise_similarity.py` and aggregates clustering performance (e.g., silhouette scores) to identify optimal settings.

- **analyze_two_params.py:**  
  Focuses on the joint effect of two selected ABM parameters on the survival of information. It relies on `surviving_information.py` to compute survival counts from simulation outputs and then generates 2D heatmaps to visualize the interplay between the two parameters.

- **plot_surviving_by_m.py:**  
  Analyzes how the number of surviving pieces of information changes with the state vector length (`m`). It uses functions from `surviving_information.py` to compute survival counts and then plots these counts (with confidence intervals) as a function of `m`.

- **surviving_information.py:**  
  Provides core functions to determine which pieces of information “survive” in the final agent state vectors based on a threshold and a minimum fraction criterion. This module is imported by analysis scripts (e.g., `analyze_two_params.py` and `plot_surviving_by_m.py`) to quantify information survival.

- **plot_utils.py:**  
  Contains generic plotting helper functions used across multiple analysis scripts (such as heatmaps for grid searches and line plots for cluster versus parameter relationships).

- **interpreter.py:**  
  Loads simulation output JSON files and generates scatter plots of agent state vectors at specified timesteps. It supports both raw data and moving average data, with optional softmax transformation, to aid in visual inspection of simulation dynamics.

- **plot_distance_matrix.py:**  
  Uses `pairwise_similarity.py` to compute a distance matrix from the final moving average state vectors and visualizes this matrix as a heatmap using Seaborn.

- **plot_violin.py:**  
  Reads simulation results and generates violin or box plots to visualize the distribution of certainty values across each dimension of the agent state vectors.

- **run.py:**  
  A simple driver script that runs a single simulation using parameters defined in `config.py`. It calls `run_simulation_with_params` from `natural_simulation.py` and saves the final results as a JSON file.

- **survival_per_parameter.py:**  
  Analyzes how a chosen “interesting parameter” (and optionally a secondary parameter) affects the proportion of surviving information. It imports functions from `surviving_information.py` to compute survival counts and then generates a line plot (with 95% confidence intervals) to summarize the impact.

- **utility_scripts/find_beta.py:**  
  A utility script that creates a dummy agent to analyze how the sending probability distribution (`p_send`) changes over a range of beta (rationality) values. It prints the computed probabilities rounded to three significant figures.

- **utility_scripts/find_optimal_temperature.py:**  
  Assists in selecting an optimal softmax temperature for converting state vectors into probability distributions. It loads simulation outputs, applies softmax over a range of temperatures, computes pairwise Jensen-Shannon Divergence among the resulting distributions, and produces corresponding plots and suggestions.

- **utility_scripts/hdbscan_gridsearch.py:**  
  Executes a grid search over HDBSCAN hyperparameters for each unique ABM parameter set found in the simulation results. It aggregates metrics (such as silhouette scores and cluster counts) to help identify the best clustering settings.

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



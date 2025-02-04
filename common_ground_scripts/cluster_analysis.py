#!/usr/bin/env python3
"""
File: cluster_analysis.py

Summary:
    Performs clustering analysis using HDBSCAN on simulation results.
    It loads JSON files, computes clusters from the final moving average vectors,
    and then exports the network with nodes annotated with cluster labels to a GEXF file,
    which can be imported into Gephi for further visualization.
    The output file names are of the format:
      [num_clusters]cls_[eps]_[gamma]_[rep]_graph.gexf
    Additionally, for each JSON file a .txt summary is produced with:
      - The number of clusters detected (ignoring noise)
      - The number of noise points
      - A mapping of each cluster to its assigned color
    This updated version uses a Matplotlib colormap (tab20) to assign distinct colors to each cluster.
"""

import os
import json
import argparse
from multiprocessing import Pool
from tqdm import tqdm
import hdbscan
import networkx as nx
import pickle

from pairwise_similarity import build_distance_matrix

# Import Matplotlib colormap utilities
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def compute_clusters(data, mcs, ms):
    """
    Computes HDBSCAN clusters for the given data.
    Returns a dictionary mapping agent names to their cluster labels.
    """
    final_moving_avg = data.get("final_moving_avg", None)
    if not final_moving_avg:
        return None

    dist_mat, agent_names = build_distance_matrix(final_moving_avg, method="jsd", temperature=0.05)
    clusterer = hdbscan.HDBSCAN(metric="precomputed", min_cluster_size=mcs, min_samples=ms)
    labels = clusterer.fit_predict(dist_mat)
    return {str(agent_names[i]): label for i, label in enumerate(labels)}


def export_graph(data, clusters, output_filename, input_dir):
    """
    Exports the network to a GEXF file for Gephi.
    Attempts to obtain the graph from data["params"]["graph"].
    If not available, it checks for data["graph_file"] and tries to load the pickled graph
    from the hard-wired subdirectory 'singular_graph' (located in the same directory as this script).
    If neither is available, an empty graph is used.
    
    The computed cluster labels are added as a node attribute "cluster".
    Additionally, the "viz_color" attribute is assigned using the Matplotlib tab20 colormap.
    
    Returns a summary dictionary with:
      - num_clusters: number of non-noise clusters detected
      - noise_count: number of nodes labeled as noise (label -1)
      - cluster_colors: mapping from cluster label (>= 0) to its assigned color (hex string)
    """
    params = data.get("params", {})
    if "graph" in params:
        G = params["graph"]
    elif "graph_file" in data:
        graph_file = data["graph_file"]
        script_dir = os.path.dirname(os.path.realpath(__file__))
        graph_dir = os.path.join(script_dir, "graphs_library_50")
        graph_file_path = os.path.join(graph_dir, graph_file)
        if os.path.exists(graph_file_path):
            with open(graph_file_path, "rb") as f:
                G = pickle.load(f)
        else:
            print(f"Graph file '{graph_file_path}' not found. Using an empty graph.")
            G = nx.Graph()
    else:
        print("No graph found in simulation result. Using an empty graph.")
        G = nx.Graph()

    # Ensure G is a NetworkX graph
    if not isinstance(G, nx.Graph):
        G = nx.Graph(G)

    # Get a colormap with 20 distinct colors
    cmap = cm.get_cmap("tab20")
    cluster_colors = {}  # mapping from cluster label (>=0) to its color
    noise_count = 0

    # Add cluster labels and assign colors using the colormap
    for node in G.nodes():
        # Assume node IDs are stored as strings matching those in clusters.
        cluster_label = clusters.get(str(node), -1)
        G.nodes[node]["cluster"] = cluster_label
        if cluster_label < 0:
            assigned_color = "#CCCCCC"  # default grey for noise
            noise_count += 1
        else:
            norm_val = (cluster_label % 20) / 20.0  # normalize label to [0,1]
            rgba = cmap(norm_val)
            assigned_color = mcolors.to_hex(rgba)
            # Record the color for this cluster if not already recorded
            if cluster_label not in cluster_colors:
                cluster_colors[cluster_label] = assigned_color
        G.nodes[node]["viz_color"] = assigned_color

    # Compute the number of non-noise clusters
    num_clusters = len(cluster_colors)

    # Export the graph in GEXF format (which Gephi can read)
    nx.write_gexf(G, output_filename)
    print(f"Exported graph to {output_filename}")

    return {
        "num_clusters": num_clusters,
        "noise_count": noise_count,
        "cluster_colors": cluster_colors
    }


def write_summary(summary_filename, summary):
    """
    Writes a text summary to the given file.
    The summary dictionary should contain:
      - num_clusters: int
      - noise_count: int
      - cluster_colors: dict mapping cluster label to color hex string
    """
    with open(summary_filename, "w") as f:
        f.write(f"Number of clusters detected (excluding noise): {summary['num_clusters']}\n")
        f.write(f"Number of noise nodes (label -1): {summary['noise_count']}\n")
        f.write("Cluster color mapping:\n")
        for cl, color in sorted(summary["cluster_colors"].items()):
            f.write(f"  Cluster {cl}: {color}\n")
    print(f"Wrote summary to {summary_filename}")


def process_file(args_tuple):
    """
    Processes a single JSON file.
    Constructs the output filename using the detected number of clusters,
    the simulation parameters eps, gamma, and repetition.
    Exports the graph to a GEXF file and writes a corresponding summary .txt file.
    """
    json_file, mcs, ms, output_dir, input_dir = args_tuple
    with open(json_file, "r") as f:
        data = json.load(f)

    clusters = compute_clusters(data, mcs, ms)
    if clusters is not None:
        # Count the number of clusters (ignoring noise labeled as -1)
        num_clusters = len({label for label in clusters.values() if label != -1})
        # Retrieve eps, gamma, and repetition from JSON (or use "NA" if not present)
        eps_val = data.get("params", {}).get("eps", "NA")
        gamma_val = data.get("params", {}).get("gamma", "NA")
        rep_val = data.get("repetition", "NA")
        # Construct the output filename using the new naming convention
        base_filename = f"{num_clusters}cls_{eps_val}_{gamma_val}_{rep_val}_graph.gexf"
        output_filename = os.path.join(output_dir, base_filename)
        # Export the graph and retrieve summary info
        summary = export_graph(data, clusters, output_filename, input_dir)
        # Create a matching text file for the summary
        summary_filename = os.path.splitext(output_filename)[0] + ".txt"
        write_summary(summary_filename, summary)


def main():
    parser = argparse.ArgumentParser(description="Analyze and export clusters to Gephi with summary information.")
    parser.add_argument("input_dir", help="Directory with .json files")
    parser.add_argument("--mcs", type=int, default=50, help="min_cluster_size for HDBSCAN")
    parser.add_argument("--ms", type=int, default=10, help="min_samples for HDBSCAN")
    parser.add_argument("--n_jobs", type=int, default=4, help="Number of processes")
    args = parser.parse_args()

    output_dir = "cluster_viz"
    os.makedirs(output_dir, exist_ok=True)

    json_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith(".json")]
    tasks = [(f, args.mcs, args.ms, output_dir, args.input_dir) for f in json_files]

    with Pool(processes=args.n_jobs) as pool:
        with tqdm(total=len(tasks), desc="Processing files") as pbar:
            for _ in pool.imap_unordered(process_file, tasks):
                pbar.update(1)


if __name__ == "__main__":
    main()
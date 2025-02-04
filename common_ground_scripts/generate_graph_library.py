"""
File: generate_graph_library.py

Summary:
    Generates a library of power-law cluster graphs via NetworkX, then pickles each for reuse.

Key Function:
    * generate_graph_library(output_dir="graphs_library_high_cluster", library_size=500, n=500, m=6, p=0.9)
      - Creates random power-law cluster graphs and saves them in `output_dir`.

Dependencies:
    * Python built-ins: os, pickle
    * Third-party: networkx

Usage:
    python generate_graph_library.py
    # or
    from generate_graph_library import generate_graph_library
    generate_graph_library("my_graphs", 100, 100, 5, 0.3)

"""
import os
import pickle
import networkx as nx

def generate_graph_library(
    output_dir="graphs_library_high_cluster",
    library_size=500,
    n=50,
    m=6,
    p=0.3
):
    """
    Generates a library of power-law cluster graphs and saves them via pickle.
    """
    os.makedirs(output_dir, exist_ok=True)

    for i in range(library_size):
        # Use a varying seed for reproducibility
        seed = i  
        G = nx.powerlaw_cluster_graph(n, m, p, seed=seed)

        filename = os.path.join(output_dir, f"graph_{i}.pkl")
        with open(filename, "wb") as f:
            pickle.dump(G, f)
        print(f"Saved graph_{i}.pkl")

if __name__ == "__main__":
    generate_graph_library(
        output_dir="graphs_library_50",
        library_size=1,
        n=50,
        m=6,
        p=0.3
    )
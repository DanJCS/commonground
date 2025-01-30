#!/usr/bin/env python3
"""
plot_utils.py

Utility functions for plotting results from HDBSCAN grid search,
and cluster analysis vs. ABM parameters.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_hdbscan_gridsearch(param_grid, summary, outfile="hdbscan_gridsearch.png"):
    """
    param_grid: list of (mcs, ms)
    summary: dict => { (mcs, ms): { 'mean': X, 'count': Y, 'ci_low': A, 'ci_high': B } }
    """
    # Example approach: transform data into 2D arrays for a heatmap
    mcs_vals = sorted(list(set(mcs for (mcs, ms) in param_grid)))
    ms_vals  = sorted(list(set(ms for (mcs, ms) in param_grid)))

    # Build a 2D array silhouette_means[i, j]
    silhouette_means = np.zeros((len(mcs_vals), len(ms_vals)), dtype=float)
    silhouette_means[:] = np.nan

    for (mcs, ms), stats in summary.items():
        i = mcs_vals.index(mcs)
        j = ms_vals.index(ms)
        silhouette_means[i, j] = stats["mean"]

    # Plot a heatmap
    fig, ax = plt.subplots(figsize=(8,6))
    cax = ax.imshow(silhouette_means, origin="lower", cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(ms_vals)))
    ax.set_yticks(range(len(mcs_vals)))
    ax.set_xticklabels(ms_vals)
    ax.set_yticklabels(mcs_vals)
    ax.set_xlabel("min_samples")
    ax.set_ylabel("min_cluster_size")
    ax.set_title("HDBSCAN Grid Search (Mean Silhouette)")
    fig.colorbar(cax, ax=ax, label="Mean Silhouette")

    # Optionally overlay text
    for i in range(len(mcs_vals)):
        for j in range(len(ms_vals)):
            val = silhouette_means[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white")

    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

def plot_clusters_vs_parameter(x_vals, y_means, y_low, y_high,
                               param_name="gamma", outfile="clusters_vs_param.png"):
    """
    Simple plot: X = param values, Y = mean #clusters, with error bars for CI.
    """
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(x_vals, y_means, marker="o", color="blue", label="Mean #clusters")

    # Add error bars
    # We'll do a classic approach: vertical line from y_low to y_high
    # and a marker at the mean
    for x, m, lo, hi in zip(x_vals, y_means, y_low, y_high):
        ax.vlines(x, lo, hi, color="blue", alpha=0.5)

    ax.set_xlabel(param_name)
    ax.set_ylabel("Mean #Clusters")
    ax.set_title(f"#Clusters vs. {param_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
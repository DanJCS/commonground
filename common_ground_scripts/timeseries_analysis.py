#!/usr/bin/env python3
"""
File: timeseries_analysis.py

Summary:
    Runs a single simulation (with m fixed to 2) and produces time-series scatter plots 
    of the agents' moving average state vectors. For each timestep provided via the 
    --timesteps argument (a comma-separated list), a scatter plot is generated showing 
    the agents' positions in ℝ². The plot displays a dotted red boundary for the region 
    [-1,1]×[-1,1]. All agent points are plotted in grey.
    
    Optionally, if a softmax temperature is provided via --softmax, an additional set 
    of plots is produced that shows the softmax-converted state vectors. These are plotted 
    in the [0,1]×[0,1] space, with an appropriately scaled boundary.
    
Usage:
    python3 timeseries_analysis.py --timesteps 0,1000,2000,3000 [--softmax 1.0] [--output_dir plots]

Inputs:
    --timesteps: Comma-separated list of timesteps at which to generate scatter plots.
    --softmax: (Optional) Softmax temperature value. If provided, additional plots will be generated.
    --output_dir: (Optional) Directory where the plots will be saved. Defaults to "timeseries_plots".

Outputs:
    For each timestep, two types of scatter plots are generated:
      - A plot of the raw moving average state vectors in ℝ² (with m = 2), with the axis limited to [-1,1].
      - If --softmax is provided, a plot of the softmax-transformed state vectors in ℝ²,
        with the axis limited to [0,1].
    Each plot is saved as a PNG file in the output directory with a filename that includes the timestep
    (and "softmax" if applicable).

Dependencies:
    - Uses the simulation driver function from natural_simulation.py.
    - Assumes simulation outputs follow the existing format, including a "final_moving_avg" key.
    - Uses matplotlib for plotting and the softmax function from pairwise_similarity.py.
"""

import os
import json
import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

# Import softmax from pairwise_similarity.py (our version converts a vector using temperature)
from pairwise_similarity import softmax

# Import the simulation driver function from natural_simulation.py
from natural_simulation import run_simulation_with_params

def parse_timesteps(ts_str):
    """
    Parses a comma-separated string of timesteps into a sorted list of integers.
    """
    try:
        return sorted([int(ts.strip()) for ts in ts_str.split(",")])
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid timesteps format: {e}")

def plot_scatter(points, title, xlim, ylim, output_path):
    """
    Plots a scatter plot for the given 2D points.
    
    Args:
        points: List of (x, y) tuples.
        title: Title for the plot.
        xlim: Tuple (xmin, xmax) for x-axis limits.
        ylim: Tuple (ymin, ymax) for y-axis limits.
        output_path: File path where the plot will be saved.
    """
    plt.figure(figsize=(6,6))
    # Plot points in grey.
    xs, ys = zip(*points) if points else ([], [])
    plt.scatter(xs, ys, color="grey")
    plt.title(title)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    # Draw a red dotted boundary (dotted line) for the domain.
    plt.plot([xlim[0], xlim[1], xlim[1], xlim[0], xlim[0]], 
             [ylim[0], ylim[0], ylim[1], ylim[1], ylim[0]],
             linestyle=":", color="red")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Run a single simulation (with m=2) and generate time-series scatter plots "
                    "of the agents' moving average state vectors."
    )
    parser.add_argument("--timesteps", type=parse_timesteps, required=True,
                        help="Comma-separated list of timesteps to plot (e.g., 0,1000,2000,3000)")
    parser.add_argument("--softmax", type=float, default=None,
                        help="(Optional) If provided, also plot softmax-converted state vectors using this temperature")
    parser.add_argument("--output_dir", type=str, default="timeseries_plots",
                        help="Directory to save the plots (default: timeseries_plots)")
    # We use a command-line argument to allow setting a repetition index (optional).
    parser.add_argument("--rep", type=int, default=0, help="Repetition index for the simulation (default: 0)")
    args = parser.parse_args()

    # Create output directory if needed.
    os.makedirs(args.output_dir, exist_ok=True)

    # For this timeseries analysis, we want to fix m=2.
    # We set up a parameter dictionary directly.
    params = {
        "n": 500,
        "m": 2,  # fixed for 2D plotting
        "timesteps": 5000,
        "bi": 5.0,
        "bj": 5.0,
        "a": 0.5,
        "alpha": 0.2,
        "eps": 0.1,
        "sigma": 0.2,
        "zeta": 0,
        "eta": 0.5,
        "gamma": -1.0,
        "metric_method": "pdiff",
        "alpha_dist": "static"
    }

    print("Running simulation for time-series analysis with parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    print(f"Repetition index: {args.rep}")

    # Run the simulation. (This returns a dict with keys including "final_moving_avg".)
    result = run_simulation_with_params(params, args.rep)
    
    # Extract the time-series moving average data.
    # It is assumed that final_moving_avg is a dictionary:
    #   { agent_name: [list of moving averages over time] }
    final_moving_avg = result.get("final_moving_avg", {})
    if not final_moving_avg:
        print("No final_moving_avg data found in simulation result.")
        return

    # For each requested timestep, produce a scatter plot.
    for ts in args.timesteps:
        points = []
        for agent, series in final_moving_avg.items():
            if ts < len(series):
                # Each series entry is a 2-element vector.
                point = series[ts]
                # Ensure point is a tuple of floats.
                point = tuple(float(x) for x in point)
                points.append(point)
        title = f"State Vectors at Timestep {ts}"
        output_path = os.path.join(args.output_dir, f"state_ts{ts}.png")
        # Plot in raw space: x,y in [-1,1] (with a margin, say 10% extra)
        plot_scatter(points, title, (-1.1, 1.1), (-1.1, 1.1), output_path)

        # If softmax conversion is requested, plot an additional scatter.
        if args.softmax is not None:
            softmax_points = []
            for agent, series in final_moving_avg.items():
                if ts < len(series):
                    vec = series[ts]
                    # Apply softmax conversion using the given temperature.
                    # Importing softmax from pairwise_similarity handles conversion.
                    soft_vec = softmax(vec, temperature=args.softmax)
                    # The result is a probability distribution over 2 items, so range ~[0,1].
                    softmax_points.append(tuple(float(x) for x in soft_vec))
            title_soft = f"Softmax (T={args.softmax}) at Timestep {ts}"
            output_path_soft = os.path.join(args.output_dir, f"state_ts{ts}_softmax.png")
            # Plot in softmax space: we set x and y axis from 0 to 1 (with a small margin).
            plot_scatter(softmax_points, title_soft, (-0.1, 1.1), (-0.1, 1.1), output_path_soft)

if __name__ == "__main__":
    main()

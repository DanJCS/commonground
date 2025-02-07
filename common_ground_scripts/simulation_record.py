#!/usr/bin/env python3
"""
File: simulation_record.py

Summary:
    Runs a single simulation and saves the full time series of agent state vectors.
    The output JSON file will include:
      - "params": simulation parameters
      - "repetition": repetition index
      - "final_x_vectors": final agent state vectors (as before)
      - "final_moving_avg": the moving average state vectors computed (using avg_window = T*0.04)
      - "records": the raw time-series state vectors for each agent at every timestep
  
Usage:
    python3 simulation_record.py [--rep REP] [--output OUTPUT_FILENAME]

Note:
    To record the full raw time series, a minimal modification is required in natural_simulation.py.
    After computing the moving average, modify the return statement to include:
        "records": recordbook.records
    This ensures that the raw state vectors for each agent are stored.
"""

import argparse
import json
from datetime import datetime
from natural_simulation import run_simulation_with_params

def main():
    parser = argparse.ArgumentParser(
        description="Run a single simulation and record full state vector time series."
    )
    parser.add_argument("--rep", type=int, default=0, help="Repetition index (default: 0)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON filename (default: timestamped filename)")
    args = parser.parse_args()

    # Use the simulation parameters as defined in natural_simulation.py or modify here if needed.
    # (For example, you could hard-code or load from another configuration.)
    # In this script, we use the parameters already set inside natural_simulation.py.
    # If desired, you could modify these parameters here.
    params = {
        "n": 500,
        "m": 2,              # For easier visualization in 2D (this analysis uses m=2)
        "timesteps": 5000,
        "bi": 4.0,
        "bj": 4.0,
        "a": 0.5,
        "alpha": 0.2,
        "eps": 0.1,
        "sigma": 0.02,       # sigma is now independent
        "zeta": 0,
        "eta": 0.5,
        "gamma": 1.0,
        "metric_method": "pdiff",
        "alpha_dist": "static"
    }

    print("Running simulation with parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print(f"Repetition index: {args.rep}")

    result = run_simulation_with_params(params, args.rep)

    # IMPORTANT: Ensure that natural_simulation.py is modified so that its final return includes:
    #    "records": recordbook.records
    # This is necessary to capture the full raw time-series data.

    # Build output filename with a timestamp if not provided.
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"simulation_record_{timestamp}_rep{args.rep}.json"
    else:
        output_filename = args.output

    with open(output_filename, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Simulation complete. Results saved to {output_filename}")

if __name__ == "__main__":
    main()

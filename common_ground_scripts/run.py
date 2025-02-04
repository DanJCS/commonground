#!/usr/bin/env python3
"""
File: run.py

Summary:
    Main single-run simulation driver.
    It loads simulation parameters from config.py, resolves them into actual values,
    runs the simulation using the function from natural_simulation.py, and saves the simulation output as a JSON file.
    
Usage:
    python3 run.py [--rep REPETITION] [--output OUTPUT_FILENAME]
    
Example:
    python3 run.py --rep 0 --output my_simulation_result.json
"""

import json
import argparse
from datetime import datetime

from config import SIM_PARAMS  # Import simulation parameters
from natural_simulation import run_simulation_with_params  # Existing simulation driver

def main():
    parser = argparse.ArgumentParser(
        description="Run a single simulation with parameters defined in config.py."
    )
    parser.add_argument("--rep", type=int, default=0, 
                        help="Repetition index (default: 0)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename for the simulation result (default: timestamped JSON file)")
    args = parser.parse_args()

    # Use the parameters from config.py
    params = SIM_PARAMS

    print("Running simulation with parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print(f"Repetition: {args.rep}")

    # Run the simulation (this function is defined in natural_simulation.py)
    result = run_simulation_with_params(params, args.rep)

    # Build output filename with a timestamp if not provided.
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"simulation_result_{timestamp}_rep{args.rep}.json"
    else:
        output_filename = args.output

    # Save simulation results to JSON
    with open(output_filename, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Simulation complete. Results saved to {output_filename}")

if __name__ == "__main__":
    main()

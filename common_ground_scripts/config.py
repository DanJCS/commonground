#!/usr/bin/env python3
"""
File: config.py

Summary:
    Defines simulation parameters and grid definitions for a single-run simulation.
    Each parameter is defined as either a static value or as a callable so that later
    it can be defined as a distribution or a function of other parameters.
    
Usage:
    Import SIM_PARAMS and PARAMETER_GRID from this module.
"""

import numpy as np

# Helper to resolve a parameter: if it is callable, call it; otherwise, use it directly.
def resolve_param(param):
    return param() if callable(param) else param

# Parameter definitions (each can be a static value or a function)
n_agents = 500
m_dimension = 2
timesteps = 5000
beta_parameter = 5.0
activation_probability = 0.5
alpha_value = 0.2
epsilon_value = 0.1
sigma = 0.05
zeta_value = 0
eta_value = 0.5
gamma_value = -1.0
metric_method = "pdiff"
alpha_distribution_type = "static"  # could be "static", "beta", or "uniform"

# Build a simulation parameters dictionary.
SIM_PARAMS = {
    "n": resolve_param(n_agents),
    "m": resolve_param(m_dimension),
    "timesteps": resolve_param(timesteps),
    "bi": resolve_param(beta_parameter),
    "bj": resolve_param(beta_parameter),
    "a": resolve_param(activation_probability),
    "alpha": resolve_param(alpha_value),
    "eps": resolve_param(epsilon_value),
    "sigma": resolve_param(sigma),
    "zeta": resolve_param(zeta_value),
    "eta": resolve_param(eta_value),
    "gamma": resolve_param(gamma_value),
    "metric_method": resolve_param(metric_method),
    "alpha_dist": resolve_param(alpha_distribution_type),
}

# Define a parameter grid (all values here are static; you can extend these if needed)
PARAMETER_GRID = {
    "n": [resolve_param(n_agents)],
    "m": [2,3,4],
    "timesteps": [resolve_param(timesteps)],
    "bi": [resolve_param(beta_parameter)],
    "bj": [resolve_param(beta_parameter)],
    "a": [resolve_param(activation_probability)],
    "alpha": [resolve_param(alpha_value)],
    "eps": [0.1,0.3,0.5,0.7,0.9],
    "sigma": [0.05],
    "zeta": [resolve_param(zeta_value)],
    "eta": [resolve_param(eta_value)],
    "gamma": [-1.0,-0.8,-0.6,-0.4,-0.2,0.0],
    "metric_method": [resolve_param(metric_method)],
    "alpha_dist": [resolve_param(alpha_distribution_type)],
}

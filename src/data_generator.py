"""
Data generation module for topology-driven DSGRN PINN.

Generates trajectory data by integrating the ODE system defined by
a NetworkTopology with given L, U, T, gamma, and steepness parameters.
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats.qmc import LatinHypercube
from typing import Tuple

from network_parser import NetworkTopology
from ode_builder import build_ode_rhs_np


def generate_ics_lhs(
    n_samples: int,
    n_dims: int = 2,
    bounds=None,
    seed: int = 42
) -> np.ndarray:
    """
    Generate initial conditions via Latin Hypercube Sampling.

    Args:
        n_samples: number of IC points
        n_dims: number of dimensions (= number of nodes)
        bounds: per-dimension bounds, one of:
            - (lower, upper): same for all dims
            - list of (lower, upper) per dim
            - list of upper bounds (lower=0 assumed)
        seed: random seed

    Returns:
        (n_samples, n_dims) array
    """
    sampler = LatinHypercube(d=n_dims, seed=seed)
    samples = sampler.random(n=n_samples)

    if bounds is None:
        bounds = [(0.0, 6.0)] * n_dims
    elif isinstance(bounds, (tuple, list)) and len(bounds) == 2 and not isinstance(bounds[0], (tuple, list)):
        bounds = [(bounds[0], bounds[1])] * n_dims
    elif isinstance(bounds, (list, np.ndarray)):
        if not isinstance(bounds[0], (tuple, list)):
            bounds = [(0.0, b) for b in bounds]

    ics = np.zeros((n_samples, n_dims))
    for d in range(n_dims):
        lo, hi = bounds[d]
        ics[:, d] = samples[:, d] * (hi - lo) + lo

    return ics


def generate_trajectories(
    topology: NetworkTopology,
    L: np.ndarray,
    U: np.ndarray,
    T: np.ndarray,
    gamma: np.ndarray,
    approx_type: str = 'hill',
    steepness: np.ndarray = None,
    n_traj: int = 20,
    t_span: Tuple[float, float] = (0.0, 10.0),
    n_points: int = 200,
    seed: int = 42,
    ics: np.ndarray = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate trajectory data from the topology-driven ODE.

    Args:
        topology: NetworkTopology instance
        L, U, T: 2D arrays (n_nodes, n_nodes) indexed [source, target]
        gamma: 1D array (n_nodes,)
        approx_type: 'hill' or 'ramp'
        steepness: 2D array (n_nodes, n_nodes), Hill exponent or ramp width
        n_traj: number of trajectories
        t_span: integration time interval
        n_points: number of time points per trajectory
        seed: random seed for IC generation
        ics: optional pre-computed initial conditions (n_traj, n_nodes)

    Returns:
        (data_df, ics) where data_df has columns: traj_id, t, x0..x_{n-1}, ic0..ic_{n-1}
        and ics is the (n_traj, n_nodes) array of initial conditions used
    """
    L = np.asarray(L, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    if steepness is None:
        steepness = np.full_like(L, 50.0)
    steepness = np.asarray(steepness, dtype=np.float64)

    n_nodes = topology.n_nodes

    # Compute trapping box and generate ICs
    box = topology.trapping_box(U, gamma)
    if ics is None:
        ic_bounds = [(0.0, b) for b in box]
        ics = generate_ics_lhs(n_traj, n_dims=n_nodes, bounds=ic_bounds, seed=seed)

    # Build ODE RHS
    rhs_fn = build_ode_rhs_np(topology, L, U, T, gamma, approx_type, steepness)

    # Integration time grid
    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    # State column names
    x_cols = [f'x{i}' for i in range(n_nodes)]
    ic_cols = [f'ic{i}' for i in range(n_nodes)]

    all_data = []
    for traj_id, ic in enumerate(ics):
        sol = solve_ivp(
            fun=rhs_fn,
            t_span=t_span,
            y0=ic,
            method='LSODA',
            t_eval=t_eval,
            rtol=1e-8,
            atol=1e-10,
        )
        if sol.status == 0:
            row = {'traj_id': traj_id, 't': sol.t}
            for i in range(n_nodes):
                row[x_cols[i]] = sol.y[i]
                row[ic_cols[i]] = ic[i]
            all_data.append(pd.DataFrame(row))
        else:
            print(f"Warning: solver failed for trajectory {traj_id}, status={sol.status}")

    if all_data:
        result_df = pd.concat(all_data, ignore_index=True)
    else:
        cols = {'traj_id': pd.Series(dtype=int), 't': pd.Series(dtype=float)}
        for c in x_cols + ic_cols:
            cols[c] = pd.Series(dtype=float)
        result_df = pd.DataFrame(cols)

    return result_df, ics

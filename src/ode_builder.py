"""
ODE builder for DSGRN regulatory networks.

Constructs the ODE right-hand side from a NetworkTopology by recursively
evaluating the logic tree at each node. Provides both numpy (for solve_ivp)
and torch (for PINN physics residual) backends.
"""

import numpy as np
import torch

from network_parser import InputEdge, LogicNode, NetworkTopology
from sigma_functions import (
    sigma_plus_hill_np, sigma_minus_hill_np,
    sigma_plus_ramp_np, sigma_minus_ramp_np,
    sigma_plus_hill_torch, sigma_minus_hill_torch,
    sigma_plus_ramp_torch, sigma_minus_ramp_torch,
)

# Map (approx_type, sign, backend) -> sigma function
_SIGMA_NP = {
    ('hill', '+'): sigma_plus_hill_np,
    ('hill', '-'): sigma_minus_hill_np,
    ('ramp', '+'): sigma_plus_ramp_np,
    ('ramp', '-'): sigma_minus_ramp_np,
}

_SIGMA_TORCH = {
    ('hill', '+'): sigma_plus_hill_torch,
    ('hill', '-'): sigma_minus_hill_torch,
    ('ramp', '+'): sigma_plus_ramp_torch,
    ('ramp', '-'): sigma_minus_ramp_torch,
}


# ---------------------------------------------------------------------------
# Numpy backend (for data generation)
# ---------------------------------------------------------------------------

def build_ode_rhs_np(topology, L, U, T, gamma, approx_type, steepness):
    """
    Build a numpy ODE right-hand side function from a NetworkTopology.

    Args:
        topology: NetworkTopology instance
        L, U, T: 2D arrays of shape (n_nodes, n_nodes), indexed [source, target]
        gamma: 1D array of shape (n_nodes,), decay rates
        approx_type: 'hill' or 'ramp'
        steepness: 2D array of shape (n_nodes, n_nodes), Hill exponent d or ramp width h

    Returns:
        Callable f(t, y) -> dy/dt suitable for scipy.integrate.solve_ivp
    """
    L = np.asarray(L, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    steepness = np.asarray(steepness, dtype=np.float64)

    def rhs(t, y):
        dy = np.zeros(topology.n_nodes)
        for i in range(topology.n_nodes):
            logic_val = _eval_tree_np(
                topology.node_logic[i], y, L, U, T, steepness, approx_type
            )
            dy[i] = -gamma[i] * y[i] + logic_val
        return dy

    return rhs


def _eval_tree_np(node, y, L, U, T, steepness, approx_type):
    """Recursively evaluate a logic tree node (numpy, scalar y)."""
    if isinstance(node, InputEdge):
        s, t = node.source, node.target
        sigma_fn = _SIGMA_NP[(approx_type, node.sign)]
        return sigma_fn(y[s], T[s, t], L[s, t], U[s, t], steepness[s, t])

    if isinstance(node, LogicNode):
        vals = [_eval_tree_np(c, y, L, U, T, steepness, approx_type)
                for c in node.children]
        if node.op == 'sum':
            return sum(vals)
        elif node.op == 'product':
            result = 1.0
            for v in vals:
                result *= v
            return result

    raise ValueError(f"Unknown node type: {type(node)}")


# ---------------------------------------------------------------------------
# Torch backend (for PINN physics residual)
# ---------------------------------------------------------------------------

def compute_rhs_torch(topology, y, L, U, T, gamma, approx_type, steepness):
    """
    Compute ODE right-hand side in torch (batched).

    Args:
        topology: NetworkTopology instance
        y: (N, n_nodes) tensor of states
        L, U, T, steepness: (n_nodes, n_nodes) tensors
        gamma: (n_nodes,) tensor of decay rates
        approx_type: 'hill' or 'ramp'

    Returns:
        (N, n_nodes) tensor of dy/dt
    """
    N = y.shape[0]
    n = topology.n_nodes
    rhs = torch.zeros(N, n, device=y.device, dtype=y.dtype)

    for i in range(n):
        logic_val = _eval_tree_torch(
            topology.node_logic[i], y, L, U, T, steepness, approx_type
        )
        rhs[:, i] = -gamma[i] * y[:, i] + logic_val

    return rhs


def _eval_tree_torch(node, y, L, U, T, steepness, approx_type):
    """Recursively evaluate a logic tree node (torch, batched). Returns (N,) tensor."""
    if isinstance(node, InputEdge):
        s, t = node.source, node.target
        sigma_fn = _SIGMA_TORCH[(approx_type, node.sign)]
        return sigma_fn(y[:, s], T[s, t], L[s, t], U[s, t], steepness[s, t])

    if isinstance(node, LogicNode):
        vals = [_eval_tree_torch(c, y, L, U, T, steepness, approx_type)
                for c in node.children]
        if node.op == 'sum':
            result = vals[0]
            for v in vals[1:]:
                result = result + v
            return result
        elif node.op == 'product':
            result = vals[0]
            for v in vals[1:]:
                result = result * v
            return result

    raise ValueError(f"Unknown node type: {type(node)}")

"""
PINN models for DSGRN regulatory network parameter recovery.

Implements:
1. Sine: Sinusoidal activation function
2. SIREN: Sinusoidal Representation Network
3. DSGRNPinn: Physics-informed neural network for learning L, U, T, d per edge
"""

import torch
import torch.nn as nn
import numpy as np

from network_parser import NetworkTopology
from ode_builder import compute_rhs_torch


class Sine(nn.Module):
    """Sinusoidal activation: sin(omega0 * x)."""

    def __init__(self, omega0: float = 30.0):
        super().__init__()
        self.omega0 = omega0

    def forward(self, x):
        return torch.sin(self.omega0 * x)


class SIREN(nn.Module):
    """
    Sinusoidal Representation Network.

    Uses sine activations throughout for learning high-frequency features.

    Args:
        in_dim: input dimension
        out_dim: output dimension
        hidden_dim: hidden units per layer
        n_layers: number of hidden layers
        omega0: frequency parameter for sine activations
    """

    def __init__(self, in_dim, out_dim, hidden_dim=64, n_layers=4, omega0=30.0):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * n_layers + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(Sine(omega0))
        self.net = nn.Sequential(*layers)
        self._init_weights(omega0)

    def _init_weights(self, omega0):
        with torch.no_grad():
            for i, layer in enumerate(self.net):
                if isinstance(layer, nn.Linear):
                    n_in = layer.weight.size(1)
                    if i == 0:
                        layer.weight.uniform_(-1 / n_in, 1 / n_in)
                    else:
                        bound = np.sqrt(6 / n_in) / omega0
                        layer.weight.uniform_(-bound, bound)

    def forward(self, x):
        return self.net(x)


class DSGRNPinn(nn.Module):
    """
    Physics-informed neural network for DSGRN regulatory network parameter recovery.

    Learns per-edge parameters L, U, T, d (Hill exponent) from trajectory data.
    Uses reparameterization to enforce 0 < L < T < U.

    Args:
        topology: NetworkTopology defining the network structure
        gamma: decay rates per node (fixed, not learned)
        hidden_dim: SIREN hidden units
        n_layers: SIREN hidden layers
        omega0: SIREN frequency
        approx_type: 'hill' or 'ramp'
        init_steepness: initial Hill exponent / ramp width
        init_L, init_U, init_T: optional 2D arrays for parameter initialization
    """

    def __init__(
        self,
        topology: NetworkTopology,
        gamma,
        hidden_dim: int = 64,
        n_layers: int = 4,
        omega0: float = 30.0,
        approx_type: str = 'hill',
        init_steepness: float = 4.0,
        init_L=None,
        init_U=None,
        init_T=None,
    ):
        super().__init__()
        self.topology = topology
        self.approx_type = approx_type
        n_nodes = topology.n_nodes
        n_edges = topology.n_edges

        # SIREN: (t, ic0, ..., ic_{n-1}) -> (x0, ..., x_{n-1})
        self.net = SIREN(
            in_dim=1 + n_nodes,
            out_dim=n_nodes,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            omega0=omega0,
        )

        # Gamma: fixed decay rates
        self.register_buffer('gamma', torch.tensor(gamma, dtype=torch.float32))

        # Precompute edge index tensors for _scatter_to_matrix
        src_idx = [e[0] for e in topology.edge_list]
        tgt_idx = [e[1] for e in topology.edge_list]
        self.register_buffer('_edge_src', torch.tensor(src_idx, dtype=torch.long))
        self.register_buffer('_edge_tgt', torch.tensor(tgt_idx, dtype=torch.long))

        # Per-edge parameter initialization
        if init_L is not None:
            init_L = np.asarray(init_L)
            init_T_arr = np.asarray(init_T)
            init_U = np.asarray(init_U)
            L_flat = np.array([init_L[s, t] for s, t in topology.edge_list])
            T_flat = np.array([init_T_arr[s, t] for s, t in topology.edge_list])
            U_flat = np.array([init_U[s, t] for s, t in topology.edge_list])
        else:
            L_flat = np.ones(n_edges) * 1.0
            T_flat = np.ones(n_edges) * 2.0
            U_flat = np.ones(n_edges) * 3.0

        # Reparameterization: L = exp(log_L), T = L + exp(log_dTL), U = T + exp(log_dUT)
        L_flat_safe = np.maximum(L_flat, 1e-3)
        delta_TL = np.maximum(T_flat - L_flat_safe, 1e-6)
        delta_UT = np.maximum(U_flat - T_flat, 1e-6)

        self.log_L = nn.Parameter(
            torch.log(torch.tensor(L_flat_safe, dtype=torch.float32))
        )
        self.log_delta_TL = nn.Parameter(
            torch.log(torch.tensor(delta_TL, dtype=torch.float32))
        )
        self.log_delta_UT = nn.Parameter(
            torch.log(torch.tensor(delta_UT, dtype=torch.float32))
        )

        # Per-edge steepness (Hill exponent d or ramp width h)
        self.log_d = nn.Parameter(
            torch.log(torch.tensor([init_steepness] * n_edges, dtype=torch.float32))
        )

    # -- Parameter accessors --

    def _get_params(self):
        """Compute all flat parameter tensors once (avoids redundant exp calls)."""
        L = torch.exp(self.log_L)
        T = L + torch.exp(self.log_delta_TL)
        U = T + torch.exp(self.log_delta_UT)
        d = torch.exp(self.log_d)
        return L, T, U, d

    def _scatter_to_matrix(self, flat):
        """Map flat (n_edges,) tensor to (n_nodes, n_nodes) matrix."""
        n = self.topology.n_nodes
        mat = torch.zeros(n, n, device=flat.device, dtype=flat.dtype)
        mat[self._edge_src, self._edge_tgt] = flat
        return mat

    def get_learned_params(self):
        """Return learned parameters as numpy 2D arrays."""
        with torch.no_grad():
            L, T, U, d = self._get_params()
            return {
                'L': self._scatter_to_matrix(L).cpu().numpy(),
                'U': self._scatter_to_matrix(U).cpu().numpy(),
                'T': self._scatter_to_matrix(T).cpu().numpy(),
                'd': self._scatter_to_matrix(d).cpu().numpy(),
            }

    def get_learned_params_flat(self):
        """Return learned parameters as flat numpy arrays (one value per edge)."""
        with torch.no_grad():
            L, T, U, d = self._get_params()
            return {
                'L': L.cpu().numpy(),
                'U': U.cpu().numpy(),
                'T': T.cpu().numpy(),
                'd': d.cpu().numpy(),
            }

    # -- Forward / physics --

    def forward(self, t, ic):
        """
        Predict states from time and initial conditions.

        Args:
            t: (N, 1)
            ic: (N, n_nodes)

        Returns:
            (N, n_nodes) predicted states
        """
        inputs = torch.cat([t, ic], dim=1)
        return self.net(inputs)

    def compute_physics_residual(self, t, ic, x_pred=None):
        """
        Compute ODE residual: dx/dt - f(x).

        Args:
            t: (N, 1) time tensor
            ic: (N, n_nodes) initial condition tensor
            x_pred: unused, kept for API compatibility

        Returns:
            (N, n_nodes) tensor of residuals
        """
        n_nodes = self.topology.n_nodes

        # Forward pass with fresh autograd tape on t
        t_ad = t.clone().requires_grad_(True)
        inputs = torch.cat([t_ad, ic], dim=1)
        x = self.net(inputs)

        # Time derivatives via autograd
        dx_dt_parts = []
        for i in range(n_nodes):
            dx_i = torch.autograd.grad(
                x[:, i].sum(), t_ad,
                create_graph=True, retain_graph=True
            )[0]
            dx_dt_parts.append(dx_i)
        dx_dt = torch.cat(dx_dt_parts, dim=1)

        # Reconstruct parameter matrices (single computation)
        L_flat, T_flat, U_flat, d_flat = self._get_params()
        L = self._scatter_to_matrix(L_flat)
        U = self._scatter_to_matrix(U_flat)
        T = self._scatter_to_matrix(T_flat)
        d = self._scatter_to_matrix(d_flat)

        # ODE right-hand side
        rhs = compute_rhs_torch(
            self.topology, x, L, U, T, self.gamma, self.approx_type, d
        )

        return dx_dt - rhs

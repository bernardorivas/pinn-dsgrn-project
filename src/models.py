"""
Models module for PINN discontinuous ODE steepness learning.

Implements:
1. Sine: Sinusoidal activation function
2. SIREN: Sinusoidal Representation Network
3. DiscontinuousPINN: Physics-informed neural network for learning ODE steepness
"""

import torch
import torch.nn as nn
import numpy as np


class Sine(nn.Module):
    """
    Sinusoidal activation function.

    Applies sin(omega0 * x) as an activation function, which is beneficial for
    learning high-frequency features (like discontinuities) in coordinate-based
    neural networks.

    Args:
        omega0: Frequency multiplier for the sine function (default 30.0)
    """

    def __init__(self, omega0: float = 30.0):
        super().__init__()
        self.omega0 = omega0

    def forward(self, x):
        """Apply sinusoidal activation."""
        return torch.sin(self.omega0 * x)


class SIREN(nn.Module):
    """
    Sinusoidal Representation Network (SIREN).

    A periodic activation network designed for coordinate-based learning tasks.
    Uses sine activations throughout the network to enable learning of high-frequency
    features, making it well-suited for approximating discontinuities and sharp
    transitions in ODEs.

    Reference:
        Sitzmann, V., Martel, J., Bergman, A., Lindell, D., & Wetzstein, G. (2020).
        Implicit neural representations with levels of experts.

    Args:
        in_dim: Input dimension (e.g., 3 for [t, ic0, ic1])
        out_dim: Output dimension (e.g., 2 for [x0, x1])
        hidden_dim: Number of hidden units per layer (default 64)
        n_layers: Number of hidden layers (default 4)
        omega0: Frequency parameter for sine activations (default 30.0)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 4,
        omega0: float = 30.0
    ):
        super().__init__()

        # Build network layers
        layers = []
        dims = [in_dim] + [hidden_dim] * n_layers + [out_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # All but last layer get sine activation
                layers.append(Sine(omega0))

        self.net = nn.Sequential(*layers)
        self._init_weights(omega0)

    def _init_weights(self, omega0):
        """
        Initialize weights according to SIREN paper specifications.

        For the first layer, weights are uniformly distributed in [-1/n_in, 1/n_in].
        For subsequent layers, weights are uniformly distributed in
        [-sqrt(6/n_in)/omega0, sqrt(6/n_in)/omega0].

        Args:
            omega0: Frequency parameter (used for scaling subsequent layers)
        """
        with torch.no_grad():
            for i, layer in enumerate(self.net):
                if isinstance(layer, nn.Linear):
                    num_input = layer.weight.size(1)
                    if i == 0:
                        # First layer initialization
                        layer.weight.uniform_(-1 / num_input, 1 / num_input)
                    else:
                        # Subsequent layers (non-first)
                        bound = np.sqrt(6 / num_input) / omega0
                        layer.weight.uniform_(-bound, bound)

    def forward(self, x):
        """Forward pass through the SIREN network."""
        return self.net(x)


class DiscontinuousPINN(nn.Module):
    """
    Physics-informed neural network for learning steepness in discontinuous ODEs.

    This network learns to approximate the solution of a 2D ODE system while also
    learning the steepness parameters (Hill exponent n or piecewise width h) of
    smooth approximations to Heaviside step functions.

    The network architecture:
    - Input: (t, ic0, ic1) where t is time and ic are initial conditions
    - Output: (x0, x1) predicted state values
    - Uses SIREN with sine activations for high-frequency features

    The ODE system is:
        x0' = -x0 + U[0] + H(T[0] - x1) * (L[0] - U[0])
        x1' = -x1 + U[1] + H(T[1] - x0) * (L[1] - U[1])

    Where H(·) is approximated by either a Hill function or piecewise linear ramp.

    Args:
        hidden_dim: Number of hidden units per layer (default 64)
        n_layers: Number of hidden layers (default 4)
        omega0: Frequency parameter for SIREN (default 30.0)
        U: Steady-state values when Heaviside = 0 (default (5.0, 5.0))
        L: Steady-state values when Heaviside = 1 (default (1.0, 1.0))
        T: Threshold values for Heaviside terms (default (3.0, 3.0))
        approx_type: Type of approximation ('hill' or 'piecewise', default 'hill')
        init_steepness: Initial value for steepness parameter (default 4.0)
            - For Hill: initial n value
            - For piecewise: initial 1/h value
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 4,
        omega0: float = 30.0,
        U: tuple = (5.0, 5.0),
        L: tuple = (1.0, 1.0),
        T: tuple = (3.0, 3.0),
        approx_type: str = 'hill',
        init_steepness: float = 4.0
    ):
        super().__init__()

        # SIREN network: (t, ic0, ic1) -> (x0, x1)
        self.net = SIREN(
            in_dim=3,        # time + 2 initial conditions
            out_dim=2,       # x0, x1
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            omega0=omega0
        )

        # ODE parameters (fixed, not learned)
        self.register_buffer('U', torch.tensor(U, dtype=torch.float32))
        self.register_buffer('L', torch.tensor(L, dtype=torch.float32))
        self.register_buffer('T', torch.tensor(T, dtype=torch.float32))

        # Learnable steepness parameters
        self.approx_type = approx_type
        if approx_type == 'hill':
            # For Hill approximation: learn n (Hill exponent) for each term
            # Log-parameterized for positivity constraint
            self.log_steepness = nn.Parameter(
                torch.log(torch.tensor([init_steepness, init_steepness], dtype=torch.float32))
            )
        elif approx_type == 'piecewise':
            # For piecewise approximation: learn h (half-width) for each term
            # Parameterized as 1/h initially, then log-transformed
            self.log_steepness = nn.Parameter(
                torch.log(torch.tensor([1.0 / init_steepness, 1.0 / init_steepness], dtype=torch.float32))
            )
        else:
            raise ValueError(f"Unknown approx_type: {approx_type}. Must be 'hill' or 'piecewise'.")

    def get_steepness_params(self):
        """
        Get current steepness parameters.

        Returns:
            torch.Tensor: Steepness values [param_0, param_1]
                - For Hill: returns n values
                - For piecewise: returns h values
        """
        return torch.exp(self.log_steepness)

    def hill_approx(self, x, theta, n):
        """
        Hill function approximation of H(theta - x).

        Computes: theta^n / (theta^n + x^n)

        This smooth approximation to the Heaviside step H(theta - x) is symmetric
        around x = theta and has steepness controlled by exponent n.

        Args:
            x: Input value(s)
            theta: Threshold parameter
            n: Hill exponent (controls steepness)

        Returns:
            Approximation of H(theta - x) in [0, 1]
        """
        eps = 1e-12  # Numerical stability
        x_pos = torch.clamp(x, min=eps)
        theta_n = theta ** n
        x_n = x_pos ** n
        return theta_n / (theta_n + x_n)

    def piecewise_approx(self, x, theta, h):
        """
        Piecewise linear ramp approximation of H(theta - x).

        Implements a linear transition region:
        - Returns 1.0 for x <= theta - h
        - Linear interpolation for theta - h < x < theta + h
        - Returns 0.0 for x >= theta + h

        The transition region has width 2h, allowing flexible learning of the
        steepness.

        Args:
            x: Input value(s)
            theta: Center of transition region
            h: Half-width of transition (total width = 2h)

        Returns:
            Piecewise linear approximation of H(theta - x) in [0, 1]
        """
        left = theta - h
        right = theta + h

        # Linear interpolation in [left, right]
        slope = -1.0 / (2.0 * h + 1e-12)  # Ensure non-zero denominator
        linear = 1.0 + slope * (x - left)

        return torch.where(
            x <= left,
            torch.ones_like(x),
            torch.where(x >= right, torch.zeros_like(x), linear)
        )

    def forward(self, t, ic):
        """
        Forward pass: predict state from time and initial conditions.

        Args:
            t: (N, 1) tensor of time points
            ic: (N, 2) tensor of initial conditions [ic0, ic1]

        Returns:
            x_pred: (N, 2) tensor of predicted states [x0, x1]
        """
        # Concatenate inputs: [t, ic0, ic1]
        inputs = torch.cat([t, ic], dim=1)

        # Neural network prediction
        x_pred = self.net(inputs)

        return x_pred

    def compute_physics_residual(self, t, ic, x_pred):
        """
        Compute ODE physics residual: dx/dt - f(x, t).

        This method computes the residual of the ODE system by:
        1. Computing time derivatives via automatic differentiation
        2. Evaluating the ODE right-hand side at predicted states
        3. Computing the difference (residual)

        The residual should be zero when the predictions satisfy the ODE.

        Args:
            t: (N, 1) tensor of time points
            ic: (N, 2) tensor of initial conditions
            x_pred: (N, 2) tensor of predicted states (from forward pass)

        Returns:
            residual: (N, 2) tensor of ODE residuals [r0, r1]
                where r0 = dx0/dt - f0(x0, x1) and r1 = dx1/dt - f1(x0, x1)
        """
        # Enable gradient tracking for time derivative computation
        t_ad = t.clone().requires_grad_(True)
        inputs = torch.cat([t_ad, ic], dim=1)
        x = self.net(inputs)

        # Compute time derivatives using automatic differentiation
        dx0_dt = torch.autograd.grad(
            x[:, 0].sum(), t_ad,
            create_graph=True, retain_graph=True
        )[0]
        dx1_dt = torch.autograd.grad(
            x[:, 1].sum(), t_ad,
            create_graph=True, retain_graph=True
        )[0]

        # Extract states (maintain shape for broadcasting)
        x0, x1 = x[:, 0:1], x[:, 1:2]

        # Get steepness parameters
        params = self.get_steepness_params()

        # Compute approximate Heaviside terms based on approximation type
        if self.approx_type == 'hill':
            # Hill approximation: H(T[0] - x1) and H(T[1] - x0)
            h_01 = self.hill_approx(x1, self.T[0], params[0])
            h_10 = self.hill_approx(x0, self.T[1], params[1])
        elif self.approx_type == 'piecewise':
            # Piecewise linear approximation
            h_01 = self.piecewise_approx(x1, self.T[0], params[0])
            h_10 = self.piecewise_approx(x0, self.T[1], params[1])
        else:
            raise ValueError(f"Unknown approx_type: {self.approx_type}")

        # Right-hand side of ODE: dx/dt = f(x)
        rhs_x0 = -x0 + self.U[0] + h_01 * (self.L[0] - self.U[0])
        rhs_x1 = -x1 + self.U[1] + h_10 * (self.L[1] - self.U[1])

        # Residual: dx/dt - rhs
        residual_x0 = dx0_dt - rhs_x0
        residual_x1 = dx1_dt - rhs_x1

        return torch.cat([residual_x0, residual_x1], dim=1)

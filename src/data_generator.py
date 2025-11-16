"""
Data generation module for PINN discontinuous ODE steepness learning.

Generates trajectory data for three approximation types:
- Heaviside (discontinuous true vector field)
- Hill (steep Hill function approximation, n=50)
- Piecewise (steep piecewise linear approximation, h=0.01)

Core system of ODEs:
x0' = -x0 + U[0] + H(T[0] - x1) * (L[0] - U[0])
x1' = -x1 + U[1] + H(T[1] - x0) * (L[1] - U[1])
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats.qmc import LatinHypercube
from typing import Tuple, Callable


def hill_decreasing(x: np.ndarray, theta: float, n: float) -> np.ndarray:
    """
    Hill function approximation of H(theta - x).

    Approximates the Heaviside step function with a smooth Hill-type curve:
    Hill(theta - x) = theta^n / (theta^n + x^n)

    This function has the property that it approaches 1 for x << theta
    and 0 for x >> theta, with steepness controlled by exponent n.

    Args:
        x: Input values (array or scalar)
        theta: Threshold parameter (location of steepest slope)
        n: Hill exponent (controls steepness; higher n = steeper transition)

    Returns:
        Array of Hill function values with same shape as x
    """
    eps = 1e-12  # Prevent division by zero/underflow
    x_clamp = np.maximum(x, eps)
    theta_n = theta ** n
    x_n = x_clamp ** n
    return theta_n / (theta_n + x_n)


def piecewise_ramp(x: np.ndarray, theta: float, h: float) -> np.ndarray:
    """
    Piecewise linear ramp approximation of H(theta - x).

    Approximates the Heaviside step with a linear ramp transition:
    - Returns 1.0 for x <= theta - h
    - Linear interpolation in [theta - h, theta + h]
    - Returns 0.0 for x >= theta + h

    The parameter h controls the width of the transition region (total width = 2h).

    Args:
        x: Input values (array or scalar)
        theta: Center of transition region
        h: Half-width of the transition region (total width = 2h)

    Returns:
        Array of piecewise ramp values with same shape as x
    """
    left = theta - h
    right = theta + h

    return np.where(
        x <= left,
        1.0,
        np.where(x >= right, 0.0, (right - x) / (2 * h))
    )


def smooth_heaviside(x: np.ndarray, theta: float, eps: float = 1e-6) -> np.ndarray:
    """
    Smoothed Heaviside approximation using tanh for numerical solver stability.

    Provides a smooth approximation of H(theta - x) via the hyperbolic tangent:
    smooth_heaviside(x, theta, eps) = 0.5 * (1.0 + tanh((theta - x) / eps))

    This is used when generating data with 'heaviside' data_type because ODE solvers
    require smooth right-hand sides. The parameter eps controls the transition width
    (smaller eps = sharper transition = more challenging for solvers).

    Args:
        x: Input values (array or scalar)
        theta: Threshold parameter
        eps: Smoothness parameter (default 1e-6 for nearly discontinuous behavior)

    Returns:
        Array of smoothed Heaviside values in [0, 1]
    """
    return 0.5 * (1.0 + np.tanh((theta - x) / eps))


def generate_ics_lhs(
    n_samples: int,
    bounds: Tuple[float, float] = (0.0, 6.0),
    seed: int = 42
) -> np.ndarray:
    """
    Generate initial conditions using Latin Hypercube Sampling.

    Ensures stratified coverage of the initial condition space by partitioning
    the domain into n_samples × n_samples cells and sampling one point from each.
    This is more efficient than random sampling for exploring high-dimensional spaces.

    Args:
        n_samples: Number of initial condition pairs to generate
        bounds: Tuple (lower, upper) for the bounds of each coordinate
        seed: Random seed for reproducibility

    Returns:
        Array of shape (n_samples, 2) with initial conditions [ic0, ic1]
        Values uniformly distributed in [bounds[0], bounds[1]]²
    """
    sampler = LatinHypercube(d=2, seed=seed)
    samples = sampler.random(n=n_samples)
    # Scale from [0, 1]² to [bounds[0], bounds[1]]²
    ic_samples = samples * (bounds[1] - bounds[0]) + bounds[0]
    return ic_samples


def ode_rhs(
    t: float,
    y: np.ndarray,
    approx_fn: Callable,
    params: np.ndarray,
    U: Tuple[float, float],
    L: Tuple[float, float],
    T: Tuple[float, float]
) -> np.ndarray:
    """
    Right-hand side of the ODE system.

    Computes the time derivatives for the two-dimensional ODE system:
    x0' = -x0 + U[0] + H(T[0] - x1) * (L[0] - U[0])
    x1' = -x1 + U[1] + H(T[1] - x0) * (L[1] - U[1])

    The Heaviside terms are approximated using the provided approximation function,
    which allows for different steepness behaviors.

    Args:
        t: Current time (not used in this autonomous system, but required by solve_ivp)
        y: Current state [x0, x1]
        approx_fn: Approximation function for H(theta - x), with signature approx_fn(x, theta, param)
        params: Parameters for the approximation function (array of length 2)
        U: Steady-state values when Heaviside = 0
        L: Steady-state values when Heaviside = 1
        T: Threshold values for Heaviside terms

    Returns:
        Array [dx0/dt, dx1/dt] of the same shape as y
    """
    x0, x1 = y

    # Compute approximate Heaviside terms
    # H(T[0] - x1) controls x0 dynamics
    h_01 = approx_fn(T[0] - x1, T[0], params[0])
    # H(T[1] - x0) controls x1 dynamics
    h_10 = approx_fn(T[1] - x0, T[1], params[1])

    # ODE right-hand side
    dx0 = -x0 + U[0] + h_01 * (L[0] - U[0])
    dx1 = -x1 + U[1] + h_10 * (L[1] - U[1])

    return np.array([dx0, dx1])


def generate_trajectories(
    data_type: str,
    n_traj: int = 10,
    t_span: Tuple[float, float] = (0.0, 5.0),
    n_points: int = 200,
    U: Tuple[float, float] = (5.0, 5.0),
    L: Tuple[float, float] = (1.0, 1.0),
    T: Tuple[float, float] = (3.0, 3.0),
    hill_n: float = 50.0,
    pw_h: float = 0.01,
    ic_bounds: Tuple[float, float] = (0.0, 6.0),
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate trajectories from the ODE system using different approximation functions.

    This is the main data generation function. It creates trajectory data by integrating
    the ODE system from multiple initial conditions. The system dynamics are controlled
    by the choice of approximation function (Heaviside, Hill, or piecewise).

    System parameters are fixed across all generated data:
    - U (steady-state when H=0): default (5.0, 5.0)
    - L (steady-state when H=1): default (1.0, 1.0)
    - T (thresholds for H): default (3.0, 3.0)

    The approximation function is chosen based on data_type:
    - 'heaviside': smooth_heaviside with eps=1e-6 (nearly discontinuous)
    - 'hill': hill_decreasing with n=hill_n (default 50.0)
    - 'piecewise': piecewise_ramp with h=pw_h (default 0.01)

    Args:
        data_type: Type of vector field ('heaviside', 'hill', 'piecewise')
        n_traj: Number of trajectories to generate
        t_span: (t_start, t_end) for integration
        n_points: Number of time points per trajectory (uniform grid)
        U: Steady-state values when H=0
        L: Steady-state values when H=1
        T: Threshold values for Heaviside terms
        hill_n: Hill exponent for 'hill' data type (default 50.0)
        pw_h: Half-width of piecewise transition for 'piecewise' type (default 0.01)
        ic_bounds: Bounds for Latin hypercube initial condition sampling
        seed: Random seed for reproducibility (for both LHS and solver)

    Returns:
        pd.DataFrame with columns:
        - 'traj_id': Trajectory index (0 to n_traj-1)
        - 't': Time points
        - 'x0': x0 solution values
        - 'x1': x1 solution values
        - 'ic0': Initial condition for x0
        - 'ic1': Initial condition for x1
        - 'data_type': String identifier of the data type

    Raises:
        ValueError: If data_type is not one of 'heaviside', 'hill', 'piecewise'
    """
    if data_type not in ['heaviside', 'hill', 'piecewise']:
        raise ValueError(
            f"data_type must be 'heaviside', 'hill', or 'piecewise'; got {data_type}"
        )

    # Select approximation function and parameters based on data_type
    if data_type == 'heaviside':
        # Use smooth_heaviside with fixed eps=1e-6
        approx_fn = lambda x, theta, param: smooth_heaviside(x, theta, eps=1e-6)
        params = np.array([1.0, 1.0])  # Not used for heaviside, just placeholders
    elif data_type == 'hill':
        # Use hill_decreasing with ground truth n=hill_n for both Heaviside terms
        approx_fn = hill_decreasing
        params = np.array([hill_n, hill_n])
    else:  # data_type == 'piecewise'
        # Use piecewise_ramp with ground truth h=pw_h for both terms
        approx_fn = piecewise_ramp
        params = np.array([pw_h, pw_h])

    # Generate initial conditions using Latin Hypercube Sampling
    ics = generate_ics_lhs(n_samples=n_traj, bounds=ic_bounds, seed=seed)

    # Time grid for solution evaluation (uniform spacing)
    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    # Store all trajectory data
    all_data = []

    # Integrate ODE from each initial condition
    for traj_id, ic in enumerate(ics):
        # Define ODE wrapper with fixed parameters
        def ode_wrapper(t, y):
            return ode_rhs(t, y, approx_fn, params, U, L, T)

        # Solve ODE
        solution = solve_ivp(
            fun=ode_wrapper,
            t_span=t_span,
            y0=ic,
            method='LSODA',      # Adaptive stiff solver
            t_eval=t_eval,
            rtol=1e-6,           # Relative tolerance
            atol=1e-9,           # Absolute tolerance
            dense_output=False
        )

        # Extract solution (only if successful)
        if solution.status == 0:  # Status 0 = solution completed successfully
            # Create DataFrame for this trajectory
            traj_data = pd.DataFrame({
                'traj_id': traj_id,
                't': solution.t,
                'x0': solution.y[0],
                'x1': solution.y[1],
                'ic0': ic[0],
                'ic1': ic[1],
                'data_type': data_type
            })
            all_data.append(traj_data)
        else:
            # Warn if solver failed, but continue
            print(f"Warning: ODE solver failed for trajectory {traj_id}, status={solution.status}")

    # Concatenate all trajectories
    if all_data:
        result_df = pd.concat(all_data, ignore_index=True)
    else:
        # Return empty DataFrame with correct schema if no trajectories succeeded
        result_df = pd.DataFrame({
            'traj_id': pd.Series(dtype=int),
            't': pd.Series(dtype=float),
            'x0': pd.Series(dtype=float),
            'x1': pd.Series(dtype=float),
            'ic0': pd.Series(dtype=float),
            'ic1': pd.Series(dtype=float),
            'data_type': pd.Series(dtype=str)
        })

    return result_df

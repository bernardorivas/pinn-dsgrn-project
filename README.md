# PINN-DSGRN: Topology-Driven Parameter Recovery

Physics-Informed Neural Networks for recovering ODE parameters (L, U, T, Hill exponent d) from trajectory data, guided by DSGRN network topology.

## Core Idea

Given a DSGRN network specification and trajectory data generated under smooth (Hill/ramp) approximations to discontinuous ODEs, train a PINN to recover the per-edge parameters that produced the data. The recovered parameters are validated against the DSGRN parameter index and Morse graph.

**ODE system** for network with n nodes:

```
dy_i/dt = -gamma_i * y_i + logic_tree_i(y)
```

where `logic_tree_i` recursively evaluates sums and products of sigma functions (Hill or piecewise-linear approximations to Heaviside), each parameterized by per-edge L (lower), U (upper), T (threshold), and d (steepness).

## Installation

```bash
pip install -r requirements.txt
```

Optional: install [DSGRN](https://github.com/marciogameiro/DSGRN) and dsgrn_utils for parameter index computation and Morse graph validation. The core pipeline works without them.

## Usage

### Stage 1: Data Generation

Generate DSGRN-compatible parameters and trajectory data across a Hill exponent sweep:

```bash
python scripts/run_pipeline.py
```

This parses the network specification, generates well-separated threshold parameters for target DSGRN parameter indices, and integrates the ODE system for each Hill exponent value. Output goes to `results/hill_sweep/{run_id}/`.

### Stage 2: PINN Recovery

Train PINNs to recover per-edge parameters from Stage 1 trajectory data:

```bash
python scripts/run_pinn_recovery.py
```

Output (recovered parameters, training history, comparison plots, re-simulated trajectories) goes to `results/pinn_recovery/{campaign_id}/`.

Configuration for both stages is via constants at the top of each script.

## Project Structure

```
├── src/
│   ├── network_parser.py      # Parse net_spec -> NetworkTopology with logic trees
│   ├── sigma_functions.py     # Dual-backend (numpy/torch) Hill and ramp functions
│   ├── ode_builder.py         # Build ODE RHS from topology via recursive tree eval
│   ├── data_generator.py      # Latin hypercube ICs + solve_ivp trajectory generation
│   ├── models.py              # DSGRNPinn: SIREN + learnable per-edge L/U/T/d
│   ├── trainer.py             # Training loop (data + physics + IC loss), early stopping
│   ├── dsgrn_interface.py     # DSGRN parameter index, Morse graph, T generation
│   ├── pinn_recovery.py       # Recovery pipeline orchestration and analysis
│   ├── utils.py               # Visualization and run ID utilities
│   └── experiment_runner.py   # Alternative YAML-based orchestration
├── scripts/
│   ├── run_pipeline.py            # Stage 1 orchestrator
│   ├── run_pinn_recovery.py       # Stage 2 orchestrator
│   ├── generate_dsgrn_params.py   # Standalone parameter generation
│   └── hill_sweep.py              # Standalone Hill exponent sweep
├── notebooks/                 # Exploration and analysis
├── results/                   # All pipeline outputs
├── configs/                   # (reserved for future config files)
└── requirements.txt
```

## PINN Architecture

**SIREN** (Sinusoidal Representation Network) with sine activations:
- Input: `(t, ic_0, ..., ic_{n-1})`
- Output: `(x_0, ..., x_{n-1})`
- Learnable per-edge parameters with reparameterization enforcing `0 < L < T < U`

**Loss:** `w_data * ||x_pred - x_true||^2 + w_phys * ||dx/dt - f(x)||^2 + w_ic * ||x(0) - ic||^2`

## Dependencies

- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- SciPy >= 1.10.0
- Pandas >= 2.0.0
- Matplotlib >= 3.7.0
- Seaborn >= 0.12.0
- PyYAML >= 6.0
- tqdm >= 4.65.0
- DSGRN (optional)

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

Configuration is via constants at the top of each script.

## Project Structure

```
├── src/
│   ├── network_parser.py      # Parse net_spec -> NetworkTopology with logic trees
│   ├── sigma_functions.py     # Dual-backend (numpy/torch) Hill and ramp functions
│   ├── ode_builder.py         # Build ODE RHS from topology via recursive tree eval
│   ├── data_generator.py      # Latin hypercube ICs + solve_ivp trajectory generation
│   ├── dsgrn_interface.py     # DSGRN parameter index, Morse graph, T generation
│   └── utils.py               # Visualization and run ID utilities
├── scripts/
│   ├── run_pipeline.py            # Stage 1 orchestrator
│   ├── generate_dsgrn_params.py   # Standalone parameter generation
│   └── hill_sweep.py              # Standalone Hill exponent sweep
├── results/                   # All pipeline outputs
└── requirements.txt
```

## Dependencies

- NumPy >= 1.24.0
- SciPy >= 1.10.0
- Pandas >= 2.0.0
- Matplotlib >= 3.7.0
- Seaborn >= 0.12.0
- DSGRN (optional)

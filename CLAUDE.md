# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Project

```bash
# Activate virtual environment
source .venv/bin/activate

# Generate DSGRN parameters + Hill coefficient sweep trajectories
python scripts/run_pipeline.py
# Output: results/hill_sweep/{run_id}/{par_index}/

# Individual scripts
python scripts/generate_dsgrn_params.py
python scripts/hill_sweep.py
```

Configuration is via hardcoded constants at the top of each script (no config files).

## Architecture

Data generation pipeline driven by DSGRN network topology.

Parse DSGRN `net_spec` string into `NetworkTopology` with logic trees -> generate DSGRN-compatible parameters (L, U, T) for target parameter indices -> integrate ODE system across a sweep of Hill exponents -> save trajectory CSVs.

**ODE dynamics:** `dy_i/dt = -gamma_i * y_i + logic_tree_i(y)`, where `logic_tree_i` recursively evaluates sums/products of sigma functions (Hill or ramp approximations to Heaviside).

## Key Conventions

- **Imports:** All `src/` modules use bare imports (e.g., `from network_parser import parse_net_spec`). Scripts prepend `src/` to `sys.path` at runtime.
- **Node indexing:** `net_spec` uses 1-based IDs; internally everything is 0-based. Conversion happens in `parse_net_spec()`.
- **Parameter arrays:** L, U, T are `(n_nodes, n_nodes)` indexed as `[source, target]`. Only entries corresponding to actual edges are nonzero. gamma is `(n_nodes,)`. Steepness d is `(n_nodes, n_nodes)` or scalar.
- **Dual backends:** `sigma_functions.py` provides numpy versions (for `solve_ivp` data generation) and torch versions (for PINN autograd). The `ode_builder.py` dispatches to the correct backend based on `_np` vs `_torch` suffix.
- **Run IDs:** Auto-incremented via `utils.next_run_id()` (e.g., 001, 002, ...).
- **No tests directory.** Validation is done visually via phase portraits, nullcline plots, and parameter comparison figures.

## DSGRN Dependency

DSGRN and dsgrn_utils are optional. `dsgrn_interface.py` degrades gracefully (returns None/-1) when unavailable. Core ODE generation works without DSGRN.

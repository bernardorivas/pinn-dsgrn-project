"""
Full pipeline: generate DSGRN parameters + figures + hill sweep for multiple par_indices.

Usage:
    python scripts/run_pipeline.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from network_parser import topology_slug, parse_net_spec
from dsgrn_interface import generate_dsgrn_figures
from generate_dsgrn_params import run_param_generation
from hill_sweep import run_sweep
from utils import next_run_id

# ==================== Configuration ====================

NET_SPEC = "1 : 1+2\n2 : (~1)2"
PAR_INDICES = [752, 1552, 652]
GAMMA = np.array([1.0, 1.0])

# Auto-T generation (T=None for all par_indices)
MIN_SPACING = 2.5       # within-source minimum gap
MIN_T = 2.0             # minimum threshold value
MAX_T = 7.0             # maximum threshold value
GLOBAL_MIN_SPACING = 1.0  # cross-source minimum gap between any pair of T values
U_MARGIN = 5.0
MAX_ITER = 500_000
SEED = 100

# Hill sweep settings
D_UNIFORM = [2, 4, 8, 12, 16, 20, 30, 40, 50]
N_TRAJ = 50
T_SPAN = (0.0, 10.0)
N_POINTS = 300

# ===========================================================


def main():
    topology = parse_net_spec(NET_SPEC)
    network_id = topology_slug(topology)

    # Single run_id for this pipeline invocation
    sweep_base = ROOT / "results" / "hill_sweep"
    run_id = next_run_id(sweep_base)

    print(f"Network: {NET_SPEC!r}")
    print(f"Network ID: {network_id}")
    print(f"Par indices: {PAR_INDICES}")
    print(f"Run ID: {run_id}")
    print()

    for par_index in PAR_INDICES:
        print(f"{'='*60}")
        print(f"par_index = {par_index}")
        print(f"{'='*60}")

        # 1. Generate parameters (skip if already exist)
        params_dir = ROOT / "results" / "dsgrn_parameters" / network_id
        edges_path = params_dir / f"{par_index}_parameters_edges.csv"
        if edges_path.exists():
            print(f"\n[1] Parameters already exist at {edges_path}")
        else:
            pi_seed = SEED + par_index
            print(f"\n[1] Generating parameters (seed={pi_seed})...")
            run_param_generation(
                NET_SPEC, par_index, network_id,
                T=None, gamma=GAMMA,
                min_spacing=MIN_SPACING, min_T=MIN_T, max_T=MAX_T,
                U_margin=U_MARGIN, max_iter=MAX_ITER, seed=pi_seed,
                global_min_spacing=GLOBAL_MIN_SPACING,
            )

        # 2. Generate DSGRN figures (par_index-only, outside run_id)
        dsgrn_figs_dir = ROOT / "results" / "dsgrn_figs" / str(par_index)
        if not (dsgrn_figs_dir / "morse_graph.png").exists():
            print("\n[2] Generating DSGRN figures...")
            generate_dsgrn_figures(NET_SPEC, par_index, dsgrn_figs_dir)
        else:
            print(f"\n[2] DSGRN figures already exist at {dsgrn_figs_dir}")

        # 3. Run hill sweep (under run_id)
        print("\n[3] Running hill sweep...")
        sweep_dir = sweep_base / run_id / str(par_index)
        run_sweep(
            par_index, NET_SPEC, network_id,
            d_values=D_UNIFORM, n_traj=N_TRAJ, t_span=T_SPAN,
            n_points=N_POINTS, seed=SEED, output_dir=sweep_dir,
        )

        print()

    print(f"Pipeline complete. Results in results/hill_sweep/{run_id}/")


if __name__ == "__main__":
    main()

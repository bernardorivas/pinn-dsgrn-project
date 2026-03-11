"""
Generate DSGRN-compatible parameters for a given network and par_index.

Saves parameter CSVs to results/dsgrn_parameters/{network_id}/.
Configure the variables below, then run:
    python scripts/generate_dsgrn_params.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from network_parser import parse_net_spec, topology_slug
from dsgrn_interface import generate_parameters, compute_parameter_index

# ==================== Configuration ====================

NET_SPEC = "1 : 1+2\n2 : (~1)2"
PAR_INDICES = [752, 1552, 652]

# None = auto-generate slug from topology; or set e.g. "2node_toggle"
NETWORK_ID = None

# Decay rates (one per node). None = all ones.
GAMMA = np.array([1.0, 1.0])

# Threshold matrix. None = auto-generate well-separated values.
T = None

# Rejection sampling settings
U_MARGIN = 5.0
MAX_ITER = 100_000
SEED = 42

# Auto-T generation settings (only used if T is None)
MIN_SPACING = 1.5  # minimum gap between consecutive thresholds on same source
MIN_T = 1.0        # minimum threshold value
MAX_T = 6.0        # maximum threshold value

# ===========================================================


def run_param_generation(net_spec, par_index, network_id, T=None, gamma=None,
                         min_spacing=1.5, min_T=1.0, max_T=6.0,
                         U_margin=5.0, max_iter=100_000, seed=42,
                         global_min_spacing=0.0):
    """
    Generate and save parameters for a single par_index.

    Returns params dict with keys: L, U, T, gamma.
    """
    topology = parse_net_spec(net_spec)

    params = generate_parameters(
        net_spec, par_index,
        T=T, gamma=gamma,
        min_spacing=min_spacing, min_T=min_T, max_T=max_T,
        U_margin=U_margin, max_iter=max_iter, seed=seed,
        global_min_spacing=global_min_spacing,
    )

    # Verify par_index
    verified_idx = compute_parameter_index(
        net_spec, params['L'], params['U'], params['T']
    )
    print(f"  Verified par_index: {verified_idx}", end="")
    if verified_idx == par_index:
        print(" (match)")
    else:
        print(f" (MISMATCH, expected {par_index})")
        raise RuntimeError(f"par_index mismatch: got {verified_idx}, expected {par_index}")

    # Save
    out_dir = ROOT / "results" / "dsgrn_parameters" / network_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for edge in topology.edges:
        s, t = edge.source, edge.target
        rows.append({
            'source': s, 'target': t,
            'L': params['L'][s, t],
            'U': params['U'][s, t],
            'T': params['T'][s, t],
        })
    edges_df = pd.DataFrame(rows)
    gamma_df = pd.DataFrame({
        'node': range(len(params['gamma'])),
        'gamma': params['gamma'],
    })

    edges_path = out_dir / f"{par_index}_parameters_edges.csv"
    gamma_path = out_dir / f"{par_index}_parameters_gamma.csv"
    edges_df.to_csv(edges_path, index=False)
    gamma_df.to_csv(gamma_path, index=False)

    print(f"  Saved to {out_dir}/: {edges_path.name}, {gamma_path.name}")
    return params


def main():
    topology = parse_net_spec(NET_SPEC)
    network_id = NETWORK_ID or topology_slug(topology)

    print(f"Network: {NET_SPEC!r}")
    print(f"Network ID: {network_id}")
    print()

    for par_index in PAR_INDICES:
        print(f"=== par_index={par_index} ===")
        run_param_generation(
            NET_SPEC, par_index, network_id,
            T=T, gamma=GAMMA,
            min_spacing=MIN_SPACING, min_T=MIN_T, max_T=MAX_T,
            U_margin=U_MARGIN, max_iter=MAX_ITER, seed=SEED,
        )
        print()


if __name__ == "__main__":
    main()

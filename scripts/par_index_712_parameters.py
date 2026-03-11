"""
Generate constrained parameters for par_index=712 with small trapping box.

Fixes T values at 1/3 and 2/3, uses U_MARGIN=2.0 to reach par_index=712,
then selects the sample with the smallest trapping box among all hits.

Saves parameters + d=50 trajectory data in hill_sweep format
so run_pinn_recovery.py can consume it directly.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from network_parser import parse_net_spec, topology_slug
from data_generator import generate_trajectories, generate_ics_lhs
from dsgrn_interface import dsgrn_available
from utils import plot_phase_portrait, next_run_id

# ---------- Configuration ----------
NET_SPEC = "1 : 1+2\n2 : (~1)2"
TARGET_PAR_INDEX = 712

# Fixed thresholds
T_FIXED = np.array([
    [2/3, 1/3],
    [1/3, 2/3],
])

U_MARGIN = 2.0      # max offset above T for U sampling

# Trajectory generation
D_VALUE = 50
N_TRAJ = 50
T_SPAN = (0.0, 10.0)
N_POINTS = 300
GAMMA = np.array([1.0, 1.0])
SEED = 42

# Search budget
N_SEARCH = 500_000


def sample_LU(topology, rng):
    """Sample one (L, U) pair with fixed T."""
    n = topology.n_nodes
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    eps = 1e-3
    for s, t in topology.edge_list:
        L[s, t] = rng.uniform(eps, T_FIXED[s, t])
        U[s, t] = rng.uniform(T_FIXED[s, t] + eps, T_FIXED[s, t] + U_MARGIN)
    return L, U


def save_parameters(output_dir, topology, L, U, T, gamma):
    """Save L, U, T, gamma as edge CSV + gamma CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for edge in topology.edges:
        s, t = edge.source, edge.target
        rows.append({
            'source': s, 'target': t,
            'L': L[s, t], 'U': U[s, t], 'T': T[s, t],
        })
    pd.DataFrame(rows).to_csv(output_dir / "parameters_edges.csv", index=False)
    pd.DataFrame({'node': range(len(gamma)), 'gamma': gamma}).to_csv(
        output_dir / "parameters_gamma.csv", index=False)


def main():
    if not dsgrn_available():
        print("ERROR: DSGRN is required for par_index computation.")
        sys.exit(1)

    import DSGRN
    from DSGRN.ParameterFromSample import par_index_from_sample

    topology = parse_net_spec(NET_SPEC)
    network_id = topology_slug(topology)

    network = DSGRN.Network(NET_SPEC)
    pg = DSGRN.ParameterGraph(network)
    print(f"Network: {NET_SPEC!r}")
    print(f"Parameter graph size: {pg.size()}")
    print(f"Target par_index: {TARGET_PAR_INDEX}")
    print(f"T_FIXED:\n{T_FIXED}")
    print(f"U_MARGIN: {U_MARGIN}\n")

    rng = np.random.default_rng(SEED)

    # Search: collect ALL hits, keep the one with smallest trapping box
    print(f"Searching {N_SEARCH} samples for par_index={TARGET_PAR_INDEX}...")
    best_L, best_U = None, None
    best_box_vol = float('inf')
    n_hits = 0

    for i in range(N_SEARCH):
        L_cand, U_cand = sample_LU(topology, rng)
        idx = par_index_from_sample(pg, L_cand, U_cand, T_FIXED)
        if idx == TARGET_PAR_INDEX:
            n_hits += 1
            box = topology.trapping_box(U_cand, GAMMA)
            vol = box[0] * box[1]
            if vol < best_box_vol:
                best_box_vol = vol
                best_L, best_U = L_cand.copy(), U_cand.copy()

        if (i + 1) % 100_000 == 0:
            print(f"  {i+1}/{N_SEARCH} searched, {n_hits} hits so far"
                  + (f", best vol={best_box_vol:.3f}" if n_hits > 0 else ""))

    if n_hits == 0:
        print(f"ERROR: par_index={TARGET_PAR_INDEX} not found in {N_SEARCH} samples.")
        sys.exit(1)

    print(f"\n  Total hits: {n_hits} ({100*n_hits/N_SEARCH:.2f}%)")

    L_found, U_found = best_L, best_U

    # Verify
    verified_idx = par_index_from_sample(pg, L_found, U_found, T_FIXED)
    assert verified_idx == TARGET_PAR_INDEX

    box = topology.trapping_box(U_found, GAMMA)
    print(f"  Best trapping box: B0={box[0]:.4f}, B1={box[1]:.4f} (vol={box[0]*box[1]:.4f})")
    print(f"  L:\n{L_found}")
    print(f"  U:\n{U_found}")
    print(f"  T:\n{T_FIXED}")

    # Save parameters
    # 1. Standard DSGRN parameter store
    dsgrn_params_dir = ROOT / "results" / "dsgrn_parameters" / network_id
    save_parameters(dsgrn_params_dir, topology, L_found, U_found, T_FIXED, GAMMA)
    rows = []
    for edge in topology.edges:
        s, t = edge.source, edge.target
        rows.append({
            'source': s, 'target': t,
            'L': L_found[s, t], 'U': U_found[s, t], 'T': T_FIXED[s, t],
        })
    pd.DataFrame(rows).to_csv(
        dsgrn_params_dir / f"{TARGET_PAR_INDEX}_parameters_edges.csv", index=False)
    pd.DataFrame({'node': range(len(GAMMA)), 'gamma': GAMMA}).to_csv(
        dsgrn_params_dir / f"{TARGET_PAR_INDEX}_parameters_gamma.csv", index=False)
    print(f"  Saved DSGRN params to {dsgrn_params_dir}")

    # 2. Hill sweep directory
    sweep_base = ROOT / "results" / "hill_sweep"
    run_id = next_run_id(sweep_base)
    sweep_dir = sweep_base / run_id / str(TARGET_PAR_INDEX)
    save_parameters(sweep_dir, topology, L_found, U_found, T_FIXED, GAMMA)
    print(f"  Sweep run_id: {run_id}")
    print(f"  Sweep dir: {sweep_dir}")

    # Generate trajectories for d=D_VALUE
    print(f"\nGenerating {N_TRAJ} trajectories with d={D_VALUE}...")
    steepness = np.full((topology.n_nodes, topology.n_nodes), float(D_VALUE))
    ic_bounds = [(0.0, b) for b in box]
    ics = generate_ics_lhs(N_TRAJ, n_dims=topology.n_nodes, bounds=ic_bounds, seed=SEED)

    data_df, _ = generate_trajectories(
        topology, L_found, U_found, T_FIXED, GAMMA,
        approx_type='hill',
        steepness=steepness,
        n_traj=N_TRAJ,
        t_span=T_SPAN,
        n_points=N_POINTS,
        ics=ics,
    )

    traj_dir = sweep_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    traj_path = traj_dir / f"d{D_VALUE}.csv"
    data_df.to_csv(traj_path, index=False)
    print(f"  Saved {traj_path}")

    # Phase portrait
    phase_dir = sweep_dir / "phase_portraits"
    phase_dir.mkdir(parents=True, exist_ok=True)
    png_path = phase_dir / f"d{D_VALUE}.png"
    plot_phase_portrait(
        topology, L_found, U_found, T_FIXED, GAMMA,
        approx_type='hill',
        steepness=steepness,
        trajectories=data_df,
        ics=None,
        trapping_box=box,
        n_grid=30,
        save_path=str(png_path),
    )
    print(f"  Saved {png_path}")

    # Report trajectory ranges
    x0_vals = data_df['x0'].values
    x1_vals = data_df['x1'].values
    print(f"\n  Trajectory ranges:")
    print(f"    x0: [{x0_vals.min():.4f}, {x0_vals.max():.4f}]")
    print(f"    x1: [{x1_vals.min():.4f}, {x1_vals.max():.4f}]")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  par_index: {TARGET_PAR_INDEX}")
    print(f"  sweep_id:  {run_id}")
    print(f"  B0={box[0]:.4f}, B1={box[1]:.4f}")
    print(f"  x0 range: [{x0_vals.min():.4f}, {x0_vals.max():.4f}]")
    print(f"  x1 range: [{x1_vals.min():.4f}, {x1_vals.max():.4f}]")
    print(f"\nTo run PINN recovery:")
    print(f"  In run_pinn_recovery.py, set:")
    print(f"    SWEEP_ID = '{run_id}'")
    print(f"    RECOVERY_TARGETS = {{{TARGET_PAR_INDEX}: [{D_VALUE}]}}")


if __name__ == "__main__":
    main()

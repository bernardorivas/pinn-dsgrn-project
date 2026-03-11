"""
PINN parameter recovery pipeline.

Core logic: load sweep data, train PINN, analyze results, aggregate across d values.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from network_parser import parse_net_spec
from models import DSGRNPinn
from trainer import train_pinn
from data_generator import generate_trajectories
from dsgrn_interface import compare_dynamics, generate_dsgrn_figures
from utils import (
    plot_training_curves,
    plot_phase_portrait_comparison,
    plot_parameter_comparison,
    plot_recovery_vs_d,
    plot_mae_vs_d,
)


def load_sweep_data(sweep_base, sweep_id, par_index, d_value, net_spec):
    """
    Load trajectory CSV and ground-truth parameters for a single (par_index, d_value).

    Args:
        sweep_base: root directory containing sweep runs (e.g. results/hill_sweep)
        sweep_id: run ID string (e.g. "003")
        par_index: DSGRN parameter index
        d_value: Hill coefficient used in sweep
        net_spec: DSGRN network specification string

    Returns:
        (data_df, L_gt, U_gt, T_gt, gamma, topology)
    """
    sweep_dir = Path(sweep_base) / sweep_id / str(par_index)
    traj_path = sweep_dir / "trajectories" / f"d{d_value}.csv"
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory CSV not found: {traj_path}")

    data_df = pd.read_csv(traj_path)

    edges_df = pd.read_csv(sweep_dir / "parameters_edges.csv")
    gamma_df = pd.read_csv(sweep_dir / "parameters_gamma.csv")

    topology = parse_net_spec(net_spec)
    n = topology.n_nodes

    L_gt = np.zeros((n, n))
    U_gt = np.zeros((n, n))
    T_gt = np.zeros((n, n))
    for _, row in edges_df.iterrows():
        s, t = int(row['source']), int(row['target'])
        L_gt[s, t] = row['L']
        U_gt[s, t] = row['U']
        T_gt[s, t] = row['T']

    gamma = gamma_df['gamma'].values

    # Reconstruct ic columns if missing (fallback for older sweeps)
    ic_cols = [f'ic{i}' for i in range(n)]
    if not all(c in data_df.columns for c in ic_cols):
        for i in range(n):
            data_df[f'ic{i}'] = data_df.groupby('traj_id')[f'x{i}'].transform('first')

    return data_df, L_gt, U_gt, T_gt, gamma, topology


def run_single_pinn(topology, gamma, data_df, pinn_config, train_config, seed):
    """
    Create and train a DSGRNPinn model on trajectory data.

    Args:
        topology: NetworkTopology instance
        gamma: 1D array of decay rates
        data_df: DataFrame with t, x0..x_{n-1}, ic0..ic_{n-1}
        pinn_config: dict with hidden_dim, n_layers, omega0, approx_type, init_steepness
        train_config: dict with device, max_epochs, lr, patience, loss_weights
        seed: random seed

    Returns:
        trainer result dict (final_loss, final_params, converged_epoch, history, model_state)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = DSGRNPinn(
        topology=topology,
        gamma=gamma,
        hidden_dim=pinn_config['hidden_dim'],
        n_layers=pinn_config['n_layers'],
        omega0=pinn_config['omega0'],
        approx_type=pinn_config['approx_type'],
        init_steepness=pinn_config['init_steepness'],
    )

    result = train_pinn(
        model, data_df,
        device=train_config['device'],
        max_epochs=train_config['max_epochs'],
        lr=train_config['lr'],
        patience=train_config['patience'],
        loss_weights=train_config['loss_weights'],
    )
    return result


def analyze_run(result, topology, data_df, L_gt, U_gt, T_gt, gamma,
                net_spec, d_gt, approx_type, run_dir, save_checkpoint=True):
    """
    Post-process a single PINN run: save artifacts, compute metrics, generate plots.

    Args:
        result: dict from train_pinn
        topology: NetworkTopology
        data_df: training data DataFrame
        L_gt, U_gt, T_gt: ground-truth parameter matrices
        gamma: decay rates
        net_spec: DSGRN network specification string
        d_gt: ground-truth Hill coefficient (scalar, uniform across edges)
        approx_type: 'hill' or 'ramp'
        run_dir: output directory for this run
        save_checkpoint: whether to save model_checkpoint.pt

    Returns:
        comparison dict (summary row)
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # -- Save training history --
    has_history = result['history'] and result['history'].get('epoch')
    if has_history:
        history_df = pd.DataFrame(result['history'])
        history_df.to_csv(run_dir / "training_history.csv", index=False)

    # -- Recovered parameters --
    params = result['final_params']
    L_hat = np.asarray(params['L'])
    U_hat = np.asarray(params['U'])
    T_hat = np.asarray(params['T'])
    d_hat = np.asarray(params['d'])

    rec_params = {
        'L': L_hat.tolist(),
        'U': U_hat.tolist(),
        'T': T_hat.tolist(),
        'd': d_hat.tolist(),
    }
    with open(run_dir / "recovered_params.json", 'w') as f:
        json.dump(rec_params, f, indent=2)

    # -- Checkpoint --
    if save_checkpoint:
        torch.save(result['model_state'], run_dir / "model_checkpoint.pt")

    # -- Resimulate with recovered parameters --
    n = topology.n_nodes
    ic_cols = [f'ic{i}' for i in range(n)]
    traj_ids = sorted(data_df['traj_id'].unique())
    ics = data_df.groupby('traj_id')[ic_cols].first().loc[traj_ids].values
    t_span = (float(data_df['t'].min()), float(data_df['t'].max()))
    n_points = len(data_df[data_df['traj_id'] == traj_ids[0]])

    resim_df, _ = generate_trajectories(
        topology, L_hat, U_hat, T_hat, gamma,
        approx_type=approx_type,
        steepness=d_hat,
        n_traj=len(traj_ids),
        t_span=t_span,
        n_points=n_points,
        ics=ics,
    )
    resim_df.to_csv(run_dir / "resimulated_trajectories.csv", index=False)

    # -- Per-edge MAE / MRE --
    edge_list = topology.edge_list
    n_edges = len(edge_list)
    L_ae, U_ae, T_ae = [], [], []
    L_re, U_re, T_re = [], [], []
    for s, t in edge_list:
        L_ae.append(abs(L_hat[s, t] - L_gt[s, t]))
        U_ae.append(abs(U_hat[s, t] - U_gt[s, t]))
        T_ae.append(abs(T_hat[s, t] - T_gt[s, t]))
        L_re.append(abs(L_hat[s, t] - L_gt[s, t]) / max(L_gt[s, t], 1e-8))
        U_re.append(abs(U_hat[s, t] - U_gt[s, t]) / max(U_gt[s, t], 1e-8))
        T_re.append(abs(T_hat[s, t] - T_gt[s, t]) / max(T_gt[s, t], 1e-8))
    L_mae = float(np.mean(L_ae))
    U_mae = float(np.mean(U_ae))
    T_mae = float(np.mean(T_ae))
    L_mre = float(np.mean(L_re))
    U_mre = float(np.mean(U_re))
    T_mre = float(np.mean(T_re))

    # -- DSGRN comparison --
    dynamics = compare_dynamics(net_spec, L_gt, U_gt, T_gt, L_hat, U_hat, T_hat)

    rec_par_index = dynamics['rec_index']
    if rec_par_index >= 0:
        dsgrn_dir = run_dir / "dsgrn_learned"
        try:
            generate_dsgrn_figures(net_spec, rec_par_index, dsgrn_dir)
        except Exception as e:
            print(f"  Warning: DSGRN figure generation failed: {e}")

    # -- Plots --
    if has_history:
        plot_training_curves(
            history_df, topology=topology,
            save_path=str(run_dir / "training_curves.png"),
            title=f"Training (d_gt={d_gt})",
        )

    d_gt_matrix = np.full((topology.n_nodes, topology.n_nodes), float(d_gt))
    params_true = {
        'L': L_gt, 'U': U_gt, 'T': T_gt, 'gamma': gamma,
        'steepness': d_gt_matrix, 'approx_type': approx_type,
    }
    params_rec = {
        'L': L_hat, 'U': U_hat, 'T': T_hat, 'gamma': gamma,
        'steepness': d_hat, 'approx_type': approx_type,
    }
    plot_phase_portrait_comparison(
        topology, params_true, params_rec,
        trajectories=data_df,
        save_path=str(run_dir / "phase_comparison.png"),
    )

    plot_parameter_comparison(
        L_gt, U_gt, T_gt, L_hat, U_hat, T_hat,
        save_path=str(run_dir / "parameter_comparison.png"),
    )

    # -- Comparison JSON --
    comparison = {
        'd_value': float(d_gt),
        'final_loss': float(result['final_loss']),
        'converged_epoch': int(result['converged_epoch']),
        'gt_par_index': int(dynamics['gt_index']),
        'rec_par_index': int(dynamics['rec_index']),
        'same_region': bool(dynamics['same_region']),
        'L_mae': L_mae,
        'U_mae': U_mae,
        'T_mae': T_mae,
        'L_mre': L_mre,
        'U_mre': U_mre,
        'T_mre': T_mre,
        'gt_morse_str': dynamics['gt_morse_str'],
        'rec_morse_str': dynamics['rec_morse_str'],
    }
    with open(run_dir / "comparison.json", 'w') as f:
        json.dump(comparison, f, indent=2)

    return comparison


def run_recovery(sweep_base, sweep_id, par_index, d_value, net_spec,
                 pinn_config, train_config, output_dir, seed,
                 save_checkpoint=True):
    """
    End-to-end recovery for a single (par_index, d_value).

    Returns:
        comparison summary dict
    """
    print(f"\n--- par_index={par_index}, d={d_value} ---")

    data_df, L_gt, U_gt, T_gt, gamma, topology = load_sweep_data(
        sweep_base, sweep_id, par_index, d_value, net_spec,
    )

    result = run_single_pinn(
        topology, gamma, data_df, pinn_config, train_config, seed,
    )

    run_dir = Path(output_dir) / str(par_index) / f"d{d_value}"
    comparison = analyze_run(
        result, topology, data_df, L_gt, U_gt, T_gt, gamma,
        net_spec, d_value, pinn_config['approx_type'], run_dir,
        save_checkpoint=save_checkpoint,
    )

    print(f"  loss={comparison['final_loss']:.4e}  "
          f"same_region={comparison['same_region']}  "
          f"L_mae={comparison['L_mae']:.4f}  "
          f"U_mae={comparison['U_mae']:.4f}  "
          f"T_mae={comparison['T_mae']:.4f}")

    return comparison


def cross_d_summary(campaign_dir, par_index, d_values):
    """
    Aggregate comparison.json across d values for one par_index.

    Generates cross_d_summary.csv, recovery_vs_d.png, mae_vs_d.png.
    """
    par_dir = Path(campaign_dir) / str(par_index)

    rows = []
    for d in d_values:
        comp_path = par_dir / f"d{d}" / "comparison.json"
        if comp_path.exists():
            with open(comp_path) as f:
                rows.append(json.load(f))
        else:
            print(f"  Warning: {comp_path} not found, skipping")

    if not rows:
        print(f"  No comparison data for par_index={par_index}")
        return

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(par_dir / "cross_d_summary.csv", index=False)
    print(f"  Saved cross_d_summary.csv ({len(rows)} rows)")

    plot_recovery_vs_d(summary_df, save_path=str(par_dir / "recovery_vs_d.png"))
    plot_mae_vs_d(summary_df, save_path=str(par_dir / "mae_vs_d.png"))
    print(f"  Saved recovery_vs_d.png, mae_vs_d.png")

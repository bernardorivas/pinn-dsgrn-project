"""
Two-pipeline experiment runner for DSGRN PINN parameter recovery.

Stage 1 (Data Generation): Generates trajectory data from known parameters,
    produces before-training visualizations, saves student data package.
Stage 2 (Parameter Recovery): Trains PINN(s) to recover L, U, T, d,
    re-simulates with recovered parameters, compares via DSGRN.
"""

import json
import yaml
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

from network_parser import parse_net_spec
from data_generator import generate_trajectories, generate_ics_lhs
from models import DSGRNPinn
from trainer import train_pinn
from dsgrn_interface import dsgrn_available, compute_parameter_index, compare_dynamics
from utils import (
    plot_initial_conditions,
    plot_trajectories_timeseries,
    plot_phase_portrait,
    plot_trajectories_comparison,
    plot_phase_portrait_comparison,
    plot_parameter_comparison,
    plot_training_curves,
    plot_morse_graph_comparison,
)


def run_experiment_suite(config_path: str = 'configs/experiment_config.yaml'):
    """
    Run the full two-stage experimental pipeline.

    Returns:
        pd.DataFrame with summary results across all runs
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    topology = parse_net_spec(config['network']['net_spec'])
    n_nodes = topology.n_nodes

    L = np.array(config['parameters']['L'], dtype=np.float64)
    U = np.array(config['parameters']['U'], dtype=np.float64)
    T = np.array(config['parameters']['T'], dtype=np.float64)
    gamma = np.array(config['parameters']['gamma'], dtype=np.float64)

    dg = config['data_generation']
    steepness = np.array(dg['steepness'], dtype=np.float64)
    approx_type_data = dg['approx_type']

    pinn_cfg = config['pinn']
    train_cfg = config['training']
    out_cfg = config['output']

    exp_name = config.get('experiment_name', 'experiment')
    base_dir = Path(out_cfg['results_dir']) / exp_name
    before_dir = base_dir / 'before'
    after_dir = base_dir / 'after'
    before_dir.mkdir(parents=True, exist_ok=True)
    after_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # STAGE 1: DATA GENERATION
    # ==================================================================
    print("=" * 60)
    print("STAGE 1: DATA GENERATION")
    print("=" * 60)

    # Save full config snapshot
    with open(before_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Ground truth DSGRN analysis
    gt_index = -1
    if dsgrn_available():
        gt_index = compute_parameter_index(config['network']['net_spec'], L, U, T)
        print(f"Ground truth DSGRN parameter index: {gt_index}")
    (before_dir / 'par_index.txt').write_text(str(gt_index))

    # Trapping box
    box = topology.trapping_box(U, gamma)
    print(f"Trapping box: {box}")
    with open(before_dir / 'trapping_box.json', 'w') as f:
        json.dump({'bounds': box.tolist()}, f, indent=2)

    # Generate trajectories
    data, ics = generate_trajectories(
        topology=topology, L=L, U=U, T=T, gamma=gamma,
        approx_type=approx_type_data, steepness=steepness,
        n_traj=dg['n_traj'], t_span=tuple(dg['t_span']),
        n_points=dg['n_points'], seed=dg['seed'],
    )
    data.to_csv(before_dir / 'trajectories.csv', index=False)
    np.savetxt(before_dir / 'initial_conditions.csv', ics, delimiter=',',
               header=','.join(f'ic{i}' for i in range(n_nodes)), comments='')
    print(f"Generated {len(ics)} trajectories, {len(data)} data points")

    # Before-training visualizations
    if n_nodes == 2:
        plot_initial_conditions(
            ics, box, T=T,
            save_path=str(before_dir / 'plot_initial_conditions.png'))
        plot_trajectories_timeseries(
            data, n_nodes=n_nodes,
            save_path=str(before_dir / 'plot_timeseries.png'))
        plot_phase_portrait(
            topology, L, U, T, gamma, approx_type_data, steepness,
            trajectories=data, ics=ics, trapping_box=box,
            save_path=str(before_dir / 'plot_phase_portrait.png'))

    # Student package
    student_dir = before_dir / 'student_package'
    student_dir.mkdir(exist_ok=True)
    data.to_csv(student_dir / 'trajectories.csv', index=False)
    with open(student_dir / 'network.yaml', 'w') as f:
        yaml.dump({
            'net_spec': config['network']['net_spec'],
            'gamma': config['parameters']['gamma'],
        }, f)

    # ==================================================================
    # STAGE 2: PARAMETER RECOVERY
    # ==================================================================
    print("\n" + "=" * 60)
    print("STAGE 2: PARAMETER RECOVERY")
    print("=" * 60)

    n_runs = train_cfg['n_runs']
    runs_dir = after_dir / 'runs'
    runs_dir.mkdir(exist_ok=True)

    all_results = []
    pbar = tqdm(total=n_runs, desc="Training runs")

    for run_id in range(n_runs):
        run_dir = runs_dir / f'run_{run_id:03d}'
        run_dir.mkdir(exist_ok=True)

        seed = train_cfg['base_seed'] + run_id
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create model
        model = DSGRNPinn(
            topology=topology,
            gamma=gamma.tolist(),
            hidden_dim=pinn_cfg['hidden_dim'],
            n_layers=pinn_cfg['n_layers'],
            omega0=pinn_cfg['omega0'],
            approx_type=pinn_cfg['approx_type'],
            init_steepness=pinn_cfg['init_steepness'],
        )

        # Train
        result = train_pinn(
            model=model,
            data=data,
            device=train_cfg['device'],
            max_epochs=train_cfg['max_epochs'],
            lr=train_cfg['lr'],
            patience=train_cfg['patience'],
            log_interval=train_cfg['log_interval'],
            loss_weights=train_cfg['loss_weights'],
            save_history=True,
        )

        # Save training history
        if result['history'] is not None:
            hist_df = pd.DataFrame(result['history'])
            hist_df.to_csv(run_dir / 'training_history.csv', index=False)
            plot_training_curves(
                hist_df, topology=topology,
                save_path=str(run_dir / 'plot_training_curves.png'),
                title=f'Run {run_id}')

        # Save model checkpoint
        if out_cfg.get('save_checkpoints', True):
            torch.save(result['model_state'], run_dir / 'model_checkpoint.pt')

        # Recovered parameters
        rec = result['final_params']
        with open(run_dir / 'recovered_params.json', 'w') as f:
            json.dump({k: v.tolist() for k, v in rec.items()}, f, indent=2)

        # Re-simulate with recovered parameters
        data_rec, _ = generate_trajectories(
            topology=topology,
            L=rec['L'], U=rec['U'], T=rec['T'], gamma=gamma,
            approx_type=pinn_cfg['approx_type'],
            steepness=rec['d'],
            n_traj=len(ics), t_span=tuple(dg['t_span']),
            n_points=dg['n_points'], seed=dg['seed'],
            ics=ics,
        )
        data_rec.to_csv(run_dir / 'resimulated_trajectories.csv', index=False)

        # After-training visualizations
        if n_nodes == 2:
            plot_trajectories_comparison(
                data, data_rec, n_nodes=n_nodes,
                save_path=str(run_dir / 'plot_timeseries_comparison.png'))

            params_true = {'L': L, 'U': U, 'T': T, 'gamma': gamma,
                           'steepness': steepness, 'approx_type': approx_type_data}
            params_rec = {'L': rec['L'], 'U': rec['U'], 'T': rec['T'],
                          'gamma': gamma, 'steepness': rec['d'],
                          'approx_type': pinn_cfg['approx_type']}
            plot_phase_portrait_comparison(
                topology, params_true, params_rec, trajectories=data,
                save_path=str(run_dir / 'plot_phase_comparison.png'))

            plot_parameter_comparison(
                L, U, T, rec['L'], rec['U'], rec['T'],
                save_path=str(run_dir / 'plot_parameter_comparison.png'))

        # DSGRN comparison
        comparison = {'gt_index': gt_index, 'rec_index': -1, 'same_region': False}
        if dsgrn_available():
            comparison = compare_dynamics(
                config['network']['net_spec'], L, U, T,
                rec['L'], rec['U'], rec['T'])
            (run_dir / 'recovered_par_index.txt').write_text(str(comparison['rec_index']))
            plot_morse_graph_comparison(
                comparison.get('gt_morse_str', 'N/A'),
                comparison.get('rec_morse_str', 'N/A'),
                save_path=str(run_dir / 'plot_morse_comparison.png'))

        # Parameter errors
        L_err = np.abs(rec['L'] - L)
        U_err = np.abs(rec['U'] - U)
        T_err = np.abs(rec['T'] - T)

        # Only compute errors for edges that exist
        edge_mask = np.zeros_like(L, dtype=bool)
        for s, t in topology.edge_list:
            edge_mask[s, t] = True

        comparison['L_mae'] = float(L_err[edge_mask].mean())
        comparison['U_mae'] = float(U_err[edge_mask].mean())
        comparison['T_mae'] = float(T_err[edge_mask].mean())

        L_gt_masked = L[edge_mask]
        U_gt_masked = U[edge_mask]
        T_gt_masked = T[edge_mask]
        comparison['L_mre'] = float((L_err[edge_mask] / (np.abs(L_gt_masked) + 1e-12)).mean())
        comparison['U_mre'] = float((U_err[edge_mask] / (np.abs(U_gt_masked) + 1e-12)).mean())
        comparison['T_mre'] = float((T_err[edge_mask] / (np.abs(T_gt_masked) + 1e-12)).mean())

        with open(run_dir / 'comparison.json', 'w') as f:
            json.dump({k: v for k, v in comparison.items()
                       if isinstance(v, (int, float, bool, str))}, f, indent=2)

        # Collect summary row
        row = {
            'run_id': run_id,
            'seed': seed,
            'final_loss': result['final_loss'],
            'converged_epoch': result['converged_epoch'],
            'same_dsgrn_region': comparison.get('same_region', False),
            'gt_par_index': gt_index,
            'rec_par_index': comparison.get('rec_index', -1),
            'L_mae': comparison.get('L_mae', np.nan),
            'U_mae': comparison.get('U_mae', np.nan),
            'T_mae': comparison.get('T_mae', np.nan),
        }
        # Add per-edge recovered values
        for prefix in ('L', 'U', 'T', 'd'):
            for i, (s, t) in enumerate(topology.edge_list):
                row[f'rec_{prefix}_{s}_{t}'] = float(rec[prefix][s, t])

        all_results.append(row)
        pbar.update(1)
        pbar.set_postfix(loss=f"{result['final_loss']:.3e}")

    pbar.close()

    # Save summary
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(after_dir / 'summary.csv', index=False)

    print(f"\nResults saved to {base_dir}")
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(summary_df[['run_id', 'final_loss', 'L_mae', 'U_mae', 'T_mae',
                       'same_dsgrn_region']].to_string(index=False))

    return summary_df

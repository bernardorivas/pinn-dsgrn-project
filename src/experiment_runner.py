"""
Experiment module.

Implements the main pipeline:
- Loads configuration from YAML
- Generates or loads training data
- Runs full pipeline (3 data types x 2 approximation types x N runs)
- Trains models and collects results
- Computes and displays summary statistics
"""

import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from data_generator import generate_trajectories
from models import DiscontinuousPINN
from trainer import train_pinn


def run_experiment_suite(config_path: str = 'configs/experiment_config.yaml'):
    """
    Run full experimental suite:
    - 3 data types × 2 approximation types × N runs

    This function orchestrates the complete experimental workflow:
    1. Loads configuration from YAML file
    2. Creates output directories for results and figures
    3. Generates or loads trajectory data for each data type
    4. Trains PINN models with different approximation types and random seeds
    5. Saves training histories and final results
    6. Computes and displays summary statistics

    Args:
        config_path: Path to YAML configuration file
                    (default 'configs/experiment_config.yaml')

    Returns:
        pd.DataFrame: Results dataframe with columns:
            - data_type: Type of data ('heaviside', 'hill', 'piecewise')
            - approx_type: Approximation type ('hill', 'piecewise')
            - run_id: Run index (0 to n_runs-1)
            - seed: Random seed used
            - final_loss: Best loss achieved during training
            - converged_epoch: Epoch at which best loss was achieved
            - param_0: First learned steepness parameter
            - param_1: Second learned steepness parameter
            - param_mean: Mean of learned steepness parameters
            - param_std: Standard deviation of learned steepness parameters
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create output directories
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    (results_dir / 'training_curves').mkdir(exist_ok=True)
    (results_dir / 'figures').mkdir(exist_ok=True)

    # Generate or load data
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)

    data_types = ['heaviside', 'hill', 'piecewise']
    approx_types = ['hill', 'piecewise']
    n_runs = config['training']['n_runs']

    all_results = []

    # Progress tracking
    total_experiments = len(data_types) * len(approx_types) * n_runs
    pbar = tqdm(total=total_experiments, desc="Experiments")

    for data_type in data_types:
        # Generate/load data
        data_path = data_dir / f'{data_type}_trajectories.csv'

        if not data_path.exists():
            print(f"Generating {data_type} data...")
            data = generate_trajectories(
                data_type=data_type,
                **config['data_generation']
            )
            data.to_csv(data_path, index=False)
        else:
            data = pd.read_csv(data_path)

        for approx_type in approx_types:
            for run_id in range(n_runs):
                # Set seed for reproducibility
                seed = config['training']['base_seed'] + run_id
                torch.manual_seed(seed)
                np.random.seed(seed)

                # Create model
                model = DiscontinuousPINN(
                    approx_type=approx_type,
                    **config['model']
                )

                # Train
                result = train_pinn(
                    model=model,
                    data=data,
                    device=config['training']['device'],
                    max_epochs=config['training']['max_epochs'],
                    lr=config['training']['lr'],
                    patience=config['training']['patience'],
                    log_interval=config['training']['log_interval'],
                    loss_weights=config['training']['loss_weights'],
                    save_history=True
                )

                # Save training curve
                if result['history'] is not None:
                    hist_df = pd.DataFrame(result['history'])
                    hist_path = (
                        results_dir / 'training_curves' /
                        f'{data_type}_{approx_type}_run{run_id:03d}.csv'
                    )
                    hist_df.to_csv(hist_path, index=False)

                # Record results
                all_results.append({
                    'data_type': data_type,
                    'approx_type': approx_type,
                    'run_id': run_id,
                    'seed': seed,
                    'final_loss': result['final_loss'],
                    'converged_epoch': result['converged_epoch'],
                    'param_0': result['final_params'][0],
                    'param_1': result['final_params'][1],
                    'param_mean': result['final_params'].mean(),
                    'param_std': result['final_params'].std()
                })

                pbar.update(1)
                pbar.set_postfix({
                    'data': data_type,
                    'approx': approx_type,
                    'run': run_id,
                    'loss': f"{result['final_loss']:.3e}"
                })

    pbar.close()

    # Save aggregated results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_dir / 'experiment_results.csv', index=False)

    print(f"\nResults saved to {results_dir / 'experiment_results.csv'}")

    # Compute and display summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    summary = results_df.groupby(['data_type', 'approx_type']).agg({
        'param_0': ['mean', 'std', 'min', 'max'],
        'param_1': ['mean', 'std', 'min', 'max'],
        'final_loss': ['mean', 'std']
    }).round(4)

    print(summary)

    return results_df

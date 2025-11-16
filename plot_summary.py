#!/usr/bin/env python3
"""Generate summary bar plots of learned parameters from experiment results."""

import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils import plot_parameter_summary_bars, plot_parameter_scatter_split, plot_all_training_curves

def main():
    # Load results
    results_path = Path('results/experiment_results.csv')

    if not results_path.exists():
        print(f"Error: {results_path} not found")
        print("Please run experiments first using run_experiments.py")
        return

    results = pd.read_csv(results_path)
    print(f"Loaded {len(results)} experiment results")
    print(f"Data types: {sorted(results['data_type'].unique())}")
    print(f"Approx types: {sorted(results['approx_type'].unique())}")

    # Create output directory
    figures_dir = Path('results/figures')
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Generate bar plot
    output_path = figures_dir / 'parameter_summary_bars.png'
    print(f"\nGenerating summary bar plot...")
    plot_parameter_summary_bars(results, save_path=output_path)
    print(f"Saved to: {output_path}")

    # Generate scatter plots (split by approx_type)
    print(f"\nGenerating scatter plots...")

    # Hill approximation
    hill_path = figures_dir / 'parameter_scatter_hill.png'
    plot_parameter_scatter_split(results, approx_type='hill', save_path=hill_path)
    print(f"Saved to: {hill_path}")

    # Piecewise approximation
    piecewise_path = figures_dir / 'parameter_scatter_piecewise.png'
    plot_parameter_scatter_split(results, approx_type='piecewise', save_path=piecewise_path)
    print(f"Saved to: {piecewise_path}")

    # Generate training curves for each (data_type, approx_type) combination
    print(f"\nGenerating training curves for each experiment type...")

    data_types = sorted(results['data_type'].unique())
    approx_types = sorted(results['approx_type'].unique())

    for data_type in data_types:
        for approx_type in approx_types:
            curves_path = figures_dir / f'training_curves_{data_type}_{approx_type}.png'
            plot_all_training_curves(
                data_type=data_type,
                approx_type=approx_type,
                results_dir='results/training_curves',
                save_path=curves_path
            )

if __name__ == '__main__':
    main()

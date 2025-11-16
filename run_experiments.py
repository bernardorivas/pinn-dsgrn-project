#!/usr/bin/env python3
"""
Main script to run full experimental suite.

Usage:
    python run_experiments.py
    python run_experiments.py --config configs/custom_config.yaml
"""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from experiment_runner import run_experiment_suite

def main():
    parser = argparse.ArgumentParser(
        description='Run PINN experiments for learning discontinuous ODE steepness'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/experiment_config.yaml',
        help='Path to configuration YAML file'
    )

    args = parser.parse_args()

    print("="*80)
    print("PINN Discontinuous ODE Steepness Learning")
    print("="*80)
    print(f"Config: {args.config}\n")

    # Run experiments
    results_df = run_experiment_suite(config_path=args.config)

    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETE")
    print("="*80)
    print(f"Total runs: {len(results_df)}")
    print(f"Results saved to: results/experiment_results.csv")
    print(f"Training curves saved to: results/training_curves/")
    print("\nRun 'jupyter notebook notebooks/analysis.ipynb' for detailed analysis.")

if __name__ == '__main__':
    main()

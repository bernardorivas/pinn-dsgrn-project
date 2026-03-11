"""
PINN parameter recovery orchestrator.

Usage:
    python scripts/run_pinn_recovery.py
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils import next_run_id
from pinn_recovery import run_recovery, cross_d_summary

# ==================== Configuration ====================

SWEEP_ID = "001"
NET_SPEC = "1 : 1+2\n2 : (~1)2"

# Which (par_index, d_values) to run
RECOVERY_TARGETS = {
    752: [32],
}

# PINN model config
HIDDEN_DIM = 128
N_LAYERS = 5
OMEGA0 = 30.0
INIT_STEEPNESS = 4.0
APPROX_TYPE = 'hill'

# Training config
BASE_SEED = 2000
MAX_EPOCHS = 50000
LR = 1e-3
PATIENCE = 3000
DEVICE = 'mps'
LOSS_WEIGHTS = {'data': 1.0, 'physics': 1.0, 'ic': 1.0}
SAVE_CHECKPOINTS = True

# ===========================================================


def main():
    sweep_base = ROOT / "results" / "hill_sweep"
    recovery_base = ROOT / "results" / "pinn_recovery"
    campaign_id = next_run_id(recovery_base)
    campaign_dir = recovery_base / campaign_id
    campaign_dir.mkdir(parents=True, exist_ok=True)

    pinn_config = {
        'hidden_dim': HIDDEN_DIM,
        'n_layers': N_LAYERS,
        'omega0': OMEGA0,
        'approx_type': APPROX_TYPE,
        'init_steepness': INIT_STEEPNESS,
    }
    train_config = {
        'device': DEVICE,
        'max_epochs': MAX_EPOCHS,
        'lr': LR,
        'patience': PATIENCE,
        'loss_weights': LOSS_WEIGHTS,
    }

    # Save config
    config = {
        'sweep_id': SWEEP_ID,
        'net_spec': NET_SPEC,
        'recovery_targets': {str(k): v for k, v in RECOVERY_TARGETS.items()},
        'pinn': pinn_config,
        'training': train_config,
        'base_seed': BASE_SEED,
        'save_checkpoints': SAVE_CHECKPOINTS,
    }
    with open(campaign_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Campaign: {campaign_id}")
    print(f"Sweep source: {SWEEP_ID}")
    print(f"Output: {campaign_dir}")
    print()

    all_summaries = []

    for par_index, d_values in RECOVERY_TARGETS.items():
        print(f"{'='*60}")
        print(f"par_index = {par_index}, d_values = {d_values}")
        print(f"{'='*60}")

        for d_value in d_values:
            seed = BASE_SEED + par_index * 100 + d_value
            summary = run_recovery(
                sweep_base=sweep_base,
                sweep_id=SWEEP_ID,
                par_index=par_index,
                d_value=d_value,
                net_spec=NET_SPEC,
                pinn_config=pinn_config,
                train_config=train_config,
                output_dir=campaign_dir,
                seed=seed,
                save_checkpoint=SAVE_CHECKPOINTS,
            )
            all_summaries.append({'par_index': par_index, **summary})

        print(f"\nCross-d summary for par_index={par_index}:")
        cross_d_summary(campaign_dir, par_index, d_values)

    # Print final summary table
    if all_summaries:
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        header = f"{'par_idx':>8} {'d':>4} {'loss':>10} {'epoch':>6} {'same':>5} {'L_mae':>8} {'U_mae':>8} {'T_mae':>8}"
        print(header)
        print('-' * len(header))
        for s in all_summaries:
            print(
                f"{s['par_index']:>8} "
                f"{s['d_value']:>4.0f} "
                f"{s['final_loss']:>10.4e} "
                f"{s['converged_epoch']:>6d} "
                f"{'Y' if s['same_region'] else 'N':>5} "
                f"{s['L_mae']:>8.4f} "
                f"{s['U_mae']:>8.4f} "
                f"{s['T_mae']:>8.4f}"
            )

    print(f"\nCampaign {campaign_id} complete. Results: {campaign_dir}")


if __name__ == "__main__":
    main()

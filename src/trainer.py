"""
Training module for DSGRN PINN parameter recovery.

Implements the training loop with early stopping, loss computation,
and dynamic parameter tracking for variable-size networks.
"""

import torch
import pandas as pd
import numpy as np
from typing import Optional


def compute_losses(model, t, ic, x_true, loss_weights):
    """
    Compute combined PINN loss (data + physics + IC).

    Args:
        model: nn.Module with forward() and compute_physics_residual()
        t: (N, 1) time tensor
        ic: (N, n_nodes) initial condition tensor
        x_true: (N, n_nodes) ground truth state tensor
        loss_weights: dict with keys 'data', 'physics', 'ic'

    Returns:
        (total_loss, loss_dict)
    """
    x_pred = model(t, ic)

    loss_data = torch.mean((x_pred - x_true) ** 2)

    residual = model.compute_physics_residual(t, ic, x_pred)
    loss_physics = torch.mean(residual ** 2)

    t0_mask = (t <= t.min() + 1e-6).squeeze()
    if t0_mask.any():
        loss_ic = torch.mean((x_pred[t0_mask] - ic[t0_mask]) ** 2)
    else:
        loss_ic = torch.tensor(0.0, device=t.device)

    total_loss = (
        loss_weights['data'] * loss_data
        + loss_weights['physics'] * loss_physics
        + loss_weights['ic'] * loss_ic
    )

    loss_dict = {
        'data': loss_data.item(),
        'physics': loss_physics.item(),
        'ic': loss_ic.item(),
    }
    return total_loss, loss_dict


def train_pinn(
    model,
    data: pd.DataFrame,
    device: str = 'mps',
    max_epochs: int = 20000,
    lr: float = 1e-3,
    patience: int = 1000,
    log_interval: int = 200,
    loss_weights: Optional[dict] = None,
    save_history: bool = True,
) -> dict:
    """
    Train a DSGRNPinn model with early stopping.

    Args:
        model: DSGRNPinn instance
        data: DataFrame with columns t, x0..x_{n-1}, ic0..ic_{n-1}
        device: 'mps', 'cuda', or 'cpu'
        max_epochs: maximum training epochs
        lr: learning rate
        patience: early stopping patience
        log_interval: print interval
        loss_weights: loss component weights
        save_history: whether to record training history

    Returns:
        dict with: final_loss, final_params, converged_epoch,
                   history, model_state
    """
    if loss_weights is None:
        loss_weights = {'data': 1.0, 'physics': 1.0, 'ic': 1.0}

    n_nodes = model.topology.n_nodes
    edge_list = model.topology.edge_list

    model = model.to(device)

    # Prepare tensors
    x_cols = [f'x{i}' for i in range(n_nodes)]
    ic_cols = [f'ic{i}' for i in range(n_nodes)]

    t = torch.tensor(data['t'].values, dtype=torch.float32).reshape(-1, 1).to(device)
    ic = torch.tensor(data[ic_cols].values, dtype=torch.float32).to(device)
    x_true = torch.tensor(data[x_cols].values, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Build history keys
    param_keys = []
    for prefix in ('L', 'U', 'T', 'd'):
        for s, tgt in edge_list:
            param_keys.append(f'{prefix}_{s}_{tgt}')

    history = {'epoch': [], 'loss_total': [], 'loss_data': [],
               'loss_physics': [], 'loss_ic': []}
    for k in param_keys:
        history[k] = []

    best_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()

        total_loss, loss_dict = compute_losses(model, t, ic, x_true, loss_weights)
        total_loss.backward()
        optimizer.step()

        params_flat = model.get_learned_params_flat()

        if epoch % log_interval == 0:
            print(
                f"Epoch {epoch:5d} | "
                f"Loss: {total_loss.item():.4e} | "
                f"Data: {loss_dict['data']:.4e} | "
                f"Phys: {loss_dict['physics']:.4e} | "
                f"IC: {loss_dict['ic']:.4e} | "
                f"L: {np.round(params_flat['L'], 3)} | "
                f"U: {np.round(params_flat['U'], 3)} | "
                f"T: {np.round(params_flat['T'], 3)} | "
                f"d: {np.round(params_flat['d'], 3)}"
            )

        if save_history:
            history['epoch'].append(epoch)
            history['loss_total'].append(total_loss.item())
            history['loss_data'].append(loss_dict['data'])
            history['loss_physics'].append(loss_dict['physics'])
            history['loss_ic'].append(loss_dict['ic'])
            for prefix in ('L', 'U', 'T', 'd'):
                for i, (s, tgt) in enumerate(edge_list):
                    history[f'{prefix}_{s}_{tgt}'].append(
                        float(params_flat[prefix][i])
                    )

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            patience_counter = 0
            best_state = {
                'epoch': epoch,
                'model_state': {k: v.clone() for k, v in model.state_dict().items()},
                'params': model.get_learned_params(),
            }
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state['model_state'])
        final_params = best_state['params']
        converged_epoch = best_state['epoch']
    else:
        final_params = model.get_learned_params()
        converged_epoch = epoch

    return {
        'final_loss': best_loss,
        'final_params': final_params,
        'converged_epoch': converged_epoch,
        'history': history if save_history else None,
        'model_state': best_state['model_state'] if best_state else model.state_dict(),
    }

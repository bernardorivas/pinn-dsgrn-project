"""
Training module for PINN discontinuous ODE steepness learning.

Implements:
1. compute_losses: Combined loss function (data + physics + IC)
2. train_pinn: Main training loop with early stopping
"""

import torch
import pandas as pd
import numpy as np
from models import DiscontinuousPINN


def compute_losses(
    model: DiscontinuousPINN,
    t: torch.Tensor,
    ic: torch.Tensor,
    x_true: torch.Tensor,
    loss_weights: dict
) -> tuple[torch.Tensor, dict]:
    """
    Compute combined loss function for PINN training.

    Combines three loss components:
    1. Data loss: MSE between predicted and ground truth states
    2. Physics loss: MSE of ODE residual (enforces physics constraints)
    3. IC loss: Soft constraint that predictions match initial conditions at t≈0

    Args:
        model: DiscontinuousPINN model instance
        t: (N, 1) tensor of time points
        ic: (N, 2) tensor of initial conditions [ic0, ic1]
        x_true: (N, 2) tensor of ground truth states [x0, x1]
        loss_weights: dict with keys 'data', 'physics', 'ic' specifying loss weights

    Returns:
        total_loss: (scalar) weighted combination of all loss components
        loss_dict: dict with keys 'data', 'physics', 'ic' containing individual loss values
    """
    # Forward pass: predict states
    x_pred = model(t, ic)

    # 1. Data loss: MSE between prediction and ground truth
    loss_data = torch.mean((x_pred - x_true) ** 2)

    # 2. Physics loss: ODE residual
    residual = model.compute_physics_residual(t, ic, x_pred)
    loss_physics = torch.mean(residual ** 2)

    # 3. Initial condition loss (soft constraint)
    # Find points at t=0 (or very close: t <= t.min() + 1e-6)
    t0_mask = (t <= t.min() + 1e-6).squeeze()
    if t0_mask.any():
        ic_pred = x_pred[t0_mask]
        ic_target = ic[t0_mask]
        loss_ic = torch.mean((ic_pred - ic_target) ** 2)
    else:
        loss_ic = torch.tensor(0.0, device=t.device)

    # Total weighted loss
    total_loss = (
        loss_weights['data'] * loss_data +
        loss_weights['physics'] * loss_physics +
        loss_weights['ic'] * loss_ic
    )

    loss_dict = {
        'data': loss_data.item(),
        'physics': loss_physics.item(),
        'ic': loss_ic.item()
    }

    return total_loss, loss_dict


def train_pinn(
    model: DiscontinuousPINN,
    data: pd.DataFrame,
    device: str = 'mps',
    max_epochs: int = 10000,
    lr: float = 1e-3,
    patience: int = 500,
    log_interval: int = 100,
    loss_weights: dict = None,
    save_history: bool = True
) -> dict:
    """
    Train PINN model with early stopping.

    Implements the main training loop for the Physics-Informed Neural Network,
    including loss computation, backpropagation, early stopping, and history tracking.

    Args:
        model: DiscontinuousPINN model to train
        data: pandas DataFrame with columns ['t', 'ic0', 'ic1', 'x0', 'x1']
        device: Device to train on ('mps', 'cuda', or 'cpu', default 'mps')
        max_epochs: Maximum number of training epochs (default 10000)
        lr: Learning rate for Adam optimizer (default 1e-3)
        patience: Early stopping patience (stop if no improvement for this many epochs, default 500)
        log_interval: Interval for printing training progress (default 100)
        loss_weights: dict with keys 'data', 'physics', 'ic' for loss weighting.
                     If None, defaults to {'data': 1.0, 'physics': 1.0, 'ic': 1.0}
        save_history: If True, save training history; otherwise return None for history (default True)

    Returns:
        dict with keys:
            - 'final_loss': float, best loss achieved during training
            - 'final_params': np.ndarray, shape (2,), learned steepness parameters [param_0, param_1]
            - 'converged_epoch': int, epoch at which best loss was achieved
            - 'history': dict with training history (if save_history=True, else None)
                Keys: 'epoch', 'loss_total', 'loss_data', 'loss_physics', 'loss_ic', 'param_0', 'param_1'
    """
    # Set default loss weights
    if loss_weights is None:
        loss_weights = {'data': 1.0, 'physics': 1.0, 'ic': 1.0}

    # Move model to device
    model = model.to(device)

    # Prepare data tensors
    t = torch.tensor(data['t'].values, dtype=torch.float32).reshape(-1, 1).to(device)
    ic = torch.tensor(
        data[['ic0', 'ic1']].values,
        dtype=torch.float32
    ).to(device)
    x_true = torch.tensor(
        data[['x0', 'x1']].values,
        dtype=torch.float32
    ).to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Initialize training history
    history = {
        'epoch': [],
        'loss_total': [],
        'loss_data': [],
        'loss_physics': [],
        'loss_ic': [],
        'param_0': [],
        'param_1': []
    }

    # Early stopping variables
    best_loss = float('inf')
    patience_counter = 0
    best_state = None

    # Training loop
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()

        # Compute loss
        total_loss, loss_dict = compute_losses(
            model, t, ic, x_true, loss_weights
        )

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Get current steepness parameters
        current_params = model.get_steepness_params().detach().cpu().numpy()

        # Logging every log_interval epochs
        if epoch % log_interval == 0:
            print(
                f"Epoch {epoch:5d} | "
                f"Loss: {total_loss.item():.4e} | "
                f"Data: {loss_dict['data']:.4e} | "
                f"Phys: {loss_dict['physics']:.4e} | "
                f"IC: {loss_dict['ic']:.4e} | "
                f"Params: [{current_params[0]:.3f}, {current_params[1]:.3f}]"
            )

        # Save history (only if save_history=True)
        if save_history:
            history['epoch'].append(epoch)
            history['loss_total'].append(total_loss.item())
            history['loss_data'].append(loss_dict['data'])
            history['loss_physics'].append(loss_dict['physics'])
            history['loss_ic'].append(loss_dict['ic'])
            history['param_0'].append(current_params[0])
            history['param_1'].append(current_params[1])

        # Early stopping check
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            patience_counter = 0
            best_state = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'params': current_params.copy()
            }
        else:
            patience_counter += 1

        # Stop if patience exceeded
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model state
    if best_state is not None:
        model.load_state_dict(best_state['model_state'])
        final_params = best_state['params']
        converged_epoch = best_state['epoch']
    else:
        # Fallback: use current params (should not happen in practice)
        final_params = current_params.copy()
        converged_epoch = epoch

    return {
        'final_loss': best_loss,
        'final_params': final_params,
        'converged_epoch': converged_epoch,
        'history': history if save_history else None
    }

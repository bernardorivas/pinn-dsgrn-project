"""Visualization and analysis utilities for DSGRN PINN experiments."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def next_run_id(base_dir):
    """Find the next auto-incrementing run ID (001, 002, ...) under base_dir."""
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return "001"
    existing = [d.name for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    if not existing:
        return "001"
    return f"{max(int(x) for x in existing) + 1:03d}"


def plot_initial_conditions(ics, trapping_box, T=None, save_path=None):
    """
    Plot initial conditions on 2D phase plane with trapping box.

    Args:
        ics: (n_traj, 2) array of initial conditions
        trapping_box: (2,) array of upper bounds [B0, B1]
        T: optional (n_nodes, n_nodes) threshold matrix for drawing threshold lines
        save_path: path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(ics[:, 0], ics[:, 1], s=60, c='steelblue', edgecolors='k', zorder=5)

    # Trapping box
    B0, B1 = trapping_box
    rect = plt.Rectangle((0, 0), B0, B1, fill=False, edgecolor='red',
                          linewidth=2, linestyle='--', label='Trapping box')
    ax.add_patch(rect)

    # Threshold lines from T matrix (draw T[src, tgt] as lines on the axis of src)
    if T is not None:
        T = np.asarray(T)
        n = T.shape[0]
        colors = ['green', 'purple', 'orange', 'brown']
        for s in range(min(n, 2)):
            for t in range(min(n, 2)):
                if T[s, t] > 0:
                    c = colors[(s * n + t) % len(colors)]
                    if s == 0:
                        ax.axvline(T[s, t], color=c, linestyle=':', alpha=0.7,
                                   label=f'T[{s},{t}]={T[s,t]:.2f}')
                    else:
                        ax.axhline(T[s, t], color=c, linestyle=':', alpha=0.7,
                                   label=f'T[{s},{t}]={T[s,t]:.2f}')

    ax.set_xlabel('$x_0$', fontsize=12)
    ax.set_ylabel('$x_1$', fontsize=12)
    ax.set_title('Initial Conditions', fontsize=14)
    ax.set_xlim(-0.2, B0 * 1.1)
    ax.set_ylim(-0.2, B1 * 1.1)
    ax.legend(fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_trajectories_timeseries(data, n_nodes=2, save_path=None):
    """
    Plot time series x_i(t) subplots, one line per trajectory.

    Args:
        data: DataFrame with traj_id, t, x0, x1, ...
        n_nodes: number of state variables
        save_path: path to save figure
    """
    fig, axes = plt.subplots(1, n_nodes, figsize=(7 * n_nodes, 5))
    if n_nodes == 1:
        axes = [axes]

    for i in range(n_nodes):
        ax = axes[i]
        for tid in data['traj_id'].unique():
            traj = data[data['traj_id'] == tid]
            ax.plot(traj['t'], traj[f'x{i}'], alpha=0.6, linewidth=1)
        ax.set_xlabel('$t$', fontsize=12)
        ax.set_ylabel(f'$x_{i}$', fontsize=12)
        ax.set_title(f'$x_{i}(t)$', fontsize=14)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Trajectory Time Series', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_phase_portrait(topology, L, U, T, gamma, approx_type, steepness,
                        trajectories=None, ics=None, trapping_box=None,
                        n_grid=25, save_path=None):
    """
    Plot 2D phase portrait with streamplot, optional trajectory overlay.

    Only works for 2-node systems.
    """
    from ode_builder import build_ode_rhs_np

    L = np.asarray(L)
    U = np.asarray(U)
    T = np.asarray(T)
    gamma = np.asarray(gamma)
    steepness = np.asarray(steepness)

    if trapping_box is None:
        trapping_box = topology.trapping_box(U, gamma)

    B0, B1 = trapping_box[0], trapping_box[1]
    rhs_fn = build_ode_rhs_np(topology, L, U, T, gamma, approx_type, steepness)

    x0_grid = np.linspace(0.01, B0 * 1.05, n_grid)
    x1_grid = np.linspace(0.01, B1 * 1.05, n_grid)
    X0, X1 = np.meshgrid(x0_grid, x1_grid)

    DX0 = np.zeros_like(X0)
    DX1 = np.zeros_like(X1)
    for i in range(n_grid):
        for j in range(n_grid):
            dy = rhs_fn(0, [X0[i, j], X1[i, j]])
            DX0[i, j] = dy[0]
            DX1[i, j] = dy[1]

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.streamplot(x0_grid, x1_grid, DX0, DX1, color='#555555',
                  linewidth=0.8, density=1.5, arrowsize=1.2)

    # Nullclines on a finer grid
    n_null = 500
    x0_fine = np.linspace(0.01, B0 * 1.05, n_null)
    x1_fine = np.linspace(0.01, B1 * 1.05, n_null)
    X0f, X1f = np.meshgrid(x0_fine, x1_fine)
    DX0f = np.zeros_like(X0f)
    DX1f = np.zeros_like(X1f)
    for i in range(n_null):
        for j in range(n_null):
            dy = rhs_fn(0, [X0f[i, j], X1f[i, j]])
            DX0f[i, j] = dy[0]
            DX1f[i, j] = dy[1]
    c0 = ax.contour(X0f, X1f, DX0f, levels=[0], colors='#e41a1c', linewidths=1.5)
    c1 = ax.contour(X0f, X1f, DX1f, levels=[0], colors='#377eb8', linewidths=1.5)
    # Add legend proxy lines (avoids matplotlib version issues with collections)
    ax.plot([], [], color='#e41a1c', linewidth=1.5, label="$x_0' = 0$")
    ax.plot([], [], color='#377eb8', linewidth=1.5, label="$x_1' = 0$")

    if trajectories is not None:
        for tid in trajectories['traj_id'].unique():
            traj = trajectories[trajectories['traj_id'] == tid]
            ax.plot(traj['x0'], traj['x1'], 'k-', alpha=0.5, linewidth=1)

    # Threshold lines
    n_nodes = topology.n_nodes
    for s in range(min(n_nodes, 2)):
        for t_idx in range(min(n_nodes, 2)):
            if T[s, t_idx] > 0:
                if s == 0:
                    ax.axvline(T[s, t_idx], color='white', linestyle='--', alpha=0.7)
                else:
                    ax.axhline(T[s, t_idx], color='white', linestyle='--', alpha=0.7)

    ax.set_xlabel('$x_0$', fontsize=12)
    ax.set_ylabel('$x_1$', fontsize=12)
    ax.set_title('Phase Portrait', fontsize=14)
    ax.set_xlim(0, B0 * 1.05)
    ax.set_ylim(0, B1 * 1.05)
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_trajectories_comparison(data_true, data_rec, n_nodes=2, save_path=None):
    """
    Overlay ground truth (solid) and recovered (dashed) time series.
    """
    fig, axes = plt.subplots(1, n_nodes, figsize=(7 * n_nodes, 5))
    if n_nodes == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors
    traj_ids = sorted(data_true['traj_id'].unique())

    for i in range(n_nodes):
        ax = axes[i]
        for k, tid in enumerate(traj_ids):
            c = colors[k % len(colors)]
            gt = data_true[data_true['traj_id'] == tid]
            ax.plot(gt['t'], gt[f'x{i}'], '-', color=c, alpha=0.7, linewidth=1.5)
            if data_rec is not None:
                rc = data_rec[data_rec['traj_id'] == tid]
                if len(rc) > 0:
                    ax.plot(rc['t'], rc[f'x{i}'], '--', color=c, alpha=0.7, linewidth=1.5)
        ax.set_xlabel('$t$', fontsize=12)
        ax.set_ylabel(f'$x_{i}$', fontsize=12)
        ax.set_title(f'$x_{i}(t)$: solid=GT, dashed=recovered', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Trajectory Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_phase_portrait_comparison(topology, params_true, params_rec,
                                   trajectories=None, save_path=None):
    """Side-by-side phase portraits: ground truth vs recovered."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, params, title in [
        (axes[0], params_true, 'Ground Truth'),
        (axes[1], params_rec, 'Recovered'),
    ]:
        _draw_streamplot_on_ax(ax, topology, params, trajectories)
        ax.set_title(title, fontsize=14)

    plt.suptitle('Phase Portrait Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save_or_show(fig, save_path)


def _draw_streamplot_on_ax(ax, topology, params, trajectories=None, n_grid=25):
    """Helper to draw a streamplot on a given axes."""
    from ode_builder import build_ode_rhs_np

    L, U, T = np.asarray(params['L']), np.asarray(params['U']), np.asarray(params['T'])
    gamma = np.asarray(params['gamma'])
    steepness = np.asarray(params['steepness'])
    approx_type = params['approx_type']

    box = topology.trapping_box(U, gamma)
    B0, B1 = box[0], box[1]
    rhs_fn = build_ode_rhs_np(topology, L, U, T, gamma, approx_type, steepness)

    x0g = np.linspace(0.01, B0 * 1.05, n_grid)
    x1g = np.linspace(0.01, B1 * 1.05, n_grid)
    X0, X1 = np.meshgrid(x0g, x1g)
    DX0, DX1 = np.zeros_like(X0), np.zeros_like(X1)
    for i in range(n_grid):
        for j in range(n_grid):
            dy = rhs_fn(0, [X0[i, j], X1[i, j]])
            DX0[i, j], DX1[i, j] = dy[0], dy[1]

    ax.streamplot(x0g, x1g, DX0, DX1, color='#555555',
                  linewidth=0.8, density=1.2, arrowsize=1.0)

    # Nullclines on a finer grid
    n_null = 500
    x0f = np.linspace(0.01, B0 * 1.05, n_null)
    x1f = np.linspace(0.01, B1 * 1.05, n_null)
    X0f, X1f = np.meshgrid(x0f, x1f)
    DX0f, DX1f = np.zeros_like(X0f), np.zeros_like(X1f)
    for i in range(n_null):
        for j in range(n_null):
            dy = rhs_fn(0, [X0f[i, j], X1f[i, j]])
            DX0f[i, j], DX1f[i, j] = dy[0], dy[1]
    c0 = ax.contour(X0f, X1f, DX0f, levels=[0], colors='#e41a1c', linewidths=1.5)
    c1 = ax.contour(X0f, X1f, DX1f, levels=[0], colors='#377eb8', linewidths=1.5)
    # Add legend proxy lines (avoids matplotlib version issues with collections)
    ax.plot([], [], color='#e41a1c', linewidth=1.5, label="$x_0' = 0$")
    ax.plot([], [], color='#377eb8', linewidth=1.5, label="$x_1' = 0$")

    if trajectories is not None:
        for tid in trajectories['traj_id'].unique():
            tr = trajectories[trajectories['traj_id'] == tid]
            ax.plot(tr['x0'], tr['x1'], 'k-', alpha=0.4, linewidth=0.8)

    ax.legend(fontsize=8)

    ax.set_xlabel('$x_0$', fontsize=12)
    ax.set_ylabel('$x_1$', fontsize=12)
    ax.set_xlim(0, B0 * 1.05)
    ax.set_ylim(0, B1 * 1.05)


def plot_parameter_comparison(L_true, U_true, T_true, L_hat, U_hat, T_hat,
                              save_path=None):
    """Heatmap comparison of ground truth vs recovered parameter matrices."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    names = ['L', 'U', 'T']
    true_mats = [np.asarray(L_true), np.asarray(U_true), np.asarray(T_true)]
    hat_mats = [np.asarray(L_hat), np.asarray(U_hat), np.asarray(T_hat)]

    for j, (name, gt, rec) in enumerate(zip(names, true_mats, hat_mats)):
        axes[0, j].imshow(np.ones_like(gt), cmap='gray_r', vmin=0, vmax=1, aspect='auto')
        axes[0, j].set_title(f'{name} (GT)', fontsize=12)
        _annotate_heatmap(axes[0, j], gt, force_black=True)

        diff = rec - gt
        vmax = max(abs(diff.min()), abs(diff.max()), 0.01)
        im1 = axes[1, j].imshow(diff, cmap='RdBu_r', aspect='auto',
                                 vmin=-vmax, vmax=vmax)
        axes[1, j].set_title(f'{name} (Rec - GT)', fontsize=12)
        plt.colorbar(im1, ax=axes[1, j], fraction=0.046)
        _annotate_heatmap(axes[1, j], diff)

    plt.suptitle('Parameter Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save_or_show(fig, save_path)


def _annotate_heatmap(ax, mat, force_black=False):
    """Annotate heatmap cells with values."""
    absmax = np.abs(mat).max() if np.abs(mat).max() > 0 else 1.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if force_black:
                color = 'black'
            else:
                color = 'white' if abs(mat[i, j]) > 0.5 * absmax else 'black'
            ax.text(j, i, f'{mat[i, j]:.3f}', ha='center', va='center',
                    fontsize=10, color=color)


def plot_training_curves(history_df, topology=None, save_path=None,
                         title="Training Curves"):
    """
    Plot loss and parameter evolution during training.

    Creates a 2-row figure: top row for losses, bottom row for parameters.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss curves
    axes[0, 0].semilogy(history_df['epoch'], history_df['loss_total'], label='Total')
    axes[0, 0].semilogy(history_df['epoch'], history_df['loss_data'], label='Data')
    axes[0, 0].semilogy(history_df['epoch'], history_df['loss_physics'], label='Physics')
    axes[0, 0].semilogy(history_df['epoch'], history_df['loss_ic'], label='IC')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].set_title('Loss Components')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history_df['epoch'], history_df['loss_total'])
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Total Loss')
    axes[0, 1].set_title('Total Loss (Linear)')
    axes[0, 1].grid(True, alpha=0.3)

    # Parameter evolution: find all L_*, U_*, T_*, d_* columns
    param_cols = [c for c in history_df.columns if c.startswith(('L_', 'U_', 'T_', 'd_'))]

    lut_cols = [c for c in param_cols if not c.startswith('d_')]
    d_cols = [c for c in param_cols if c.startswith('d_')]

    for c in lut_cols:
        axes[1, 0].plot(history_df['epoch'], history_df[c], label=c, alpha=0.7)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_title('L, U, T Evolution')
    axes[1, 0].legend(fontsize=8, ncol=2)
    axes[1, 0].grid(True, alpha=0.3)

    for c in d_cols:
        axes[1, 1].semilogy(history_df['epoch'], history_df[c], label=c, alpha=0.7)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Steepness (log)')
    axes[1, 1].set_title('Hill Exponent d Evolution')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_morse_graph_comparison(gt_str, rec_str, save_path=None):
    """Display ground truth vs recovered Morse graph strings side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, label, text in [(axes[0], 'Ground Truth', gt_str),
                            (axes[1], 'Recovered', rec_str)]:
        ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=10,
                transform=ax.transAxes, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title(f'Morse Graph ({label})', fontsize=14)
        ax.axis('off')
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_recovery_vs_d(cross_d_df, save_path=None):
    """Bar + line plot: same_region (bool) and final_loss vs d_value."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    d_vals = cross_d_df['d_value'].values

    colors = ['#2ecc71' if s else '#e74c3c' for s in cross_d_df['same_region']]
    bar_width = 0.5 * np.min(np.diff(d_vals)) if len(d_vals) > 1 else 1.0
    ax1.bar(d_vals, cross_d_df['same_region'].astype(int), color=colors,
            alpha=0.5, width=bar_width, label='Same DSGRN region')
    ax1.set_ylabel('Same Region (0/1)')
    ax1.set_xlabel('Hill Coefficient $d$')
    ax1.set_ylim(-0.1, 1.5)

    ax2 = ax1.twinx()
    ax2.semilogy(d_vals, cross_d_df['final_loss'], 'ko-', label='Final loss')
    ax2.set_ylabel('Final Loss (log)')

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper right')

    ax1.set_title('Recovery vs Hill Coefficient', fontsize=14)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_mae_vs_d(cross_d_df, save_path=None):
    """Line plot: L_mae, U_mae, T_mae vs d_value."""
    fig, ax = plt.subplots(figsize=(10, 6))

    d_vals = cross_d_df['d_value'].values
    ax.plot(d_vals, cross_d_df['L_mae'], 'o-', label='L MAE')
    ax.plot(d_vals, cross_d_df['U_mae'], 's-', label='U MAE')
    ax.plot(d_vals, cross_d_df['T_mae'], '^-', label='T MAE')

    ax.set_xlabel('Hill Coefficient $d$')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Parameter MAE vs Hill Coefficient', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path)


def _save_or_show(fig, save_path):
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

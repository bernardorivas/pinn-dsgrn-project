"""Utility functions for visualization and analysis of PINN training results."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_curves(
    history_df: pd.DataFrame,
    save_path: str = None,
    title: str = "Training Curves"
):
    """
    Plot loss evolution and parameter evolution during training.

    Creates a 2x2 subplot figure showing:
    - (0,0): Semilogy plot of all loss components vs epoch
    - (0,1): Linear scale plot of total loss vs epoch
    - (1,0): Linear plot of steepness parameters vs epoch
    - (1,1): Semilogy plot of steepness parameters vs epoch

    Parameters
    ----------
    history_df : pd.DataFrame
        Training history with columns: ['epoch', 'loss_total', 'loss_data',
        'loss_physics', 'loss_ic', 'param_0', 'param_1']
    save_path : str, optional
        Path to save the figure. If None, displays the plot.
    title : str, default="Training Curves"
        Title for the figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss curves (semilogy)
    axes[0, 0].semilogy(history_df['epoch'], history_df['loss_total'], label='Total')
    axes[0, 0].semilogy(history_df['epoch'], history_df['loss_data'], label='Data')
    axes[0, 0].semilogy(history_df['epoch'], history_df['loss_physics'], label='Physics')
    axes[0, 0].semilogy(history_df['epoch'], history_df['loss_ic'], label='IC')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].set_title('Loss Components')
    axes[0, 0].grid(True, alpha=0.3)

    # Total loss only (linear scale)
    axes[0, 1].plot(history_df['epoch'], history_df['loss_total'])
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Total Loss')
    axes[0, 1].set_title('Total Loss (Linear Scale)')
    axes[0, 1].grid(True, alpha=0.3)

    # Parameter evolution (linear scale)
    axes[1, 0].plot(history_df['epoch'], history_df['param_0'], label='Param 0')
    axes[1, 0].plot(history_df['epoch'], history_df['param_1'], label='Param 1')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Steepness Parameter')
    axes[1, 0].legend()
    axes[1, 0].set_title('Learned Steepness Parameters')
    axes[1, 0].grid(True, alpha=0.3)

    # Parameter evolution (log scale)
    axes[1, 1].semilogy(history_df['epoch'], history_df['param_0'], label='Param 0')
    axes[1, 1].semilogy(history_df['epoch'], history_df['param_1'], label='Param 1')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Steepness Parameter (log)')
    axes[1, 1].legend()
    axes[1, 1].set_title('Learned Steepness (Log Scale)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_parameter_distributions(
    results_df: pd.DataFrame,
    save_path: str = None
):
    """
    Plot histogram of learned parameters grouped by (data_type, approx_type).

    Creates a grid of histograms with rows corresponding to data types and
    columns corresponding to approximation types. For each subplot, displays
    overlaid histograms of param_0 and param_1 with means and confidence bands.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe with columns: ['data_type', 'approx_type', 'param_0',
        'param_1', 'final_loss', ...]
    save_path : str, optional
        Path to save the figure. If None, displays the plot.
    """
    data_types = results_df['data_type'].unique()
    approx_types = results_df['approx_type'].unique()

    fig, axes = plt.subplots(
        len(data_types), len(approx_types),
        figsize=(6 * len(approx_types), 5 * len(data_types))
    )

    # Reshape axes for single row/column case
    if len(data_types) == 1:
        axes = axes.reshape(1, -1)
    if len(approx_types) == 1:
        axes = axes.reshape(-1, 1)

    for i, data_type in enumerate(data_types):
        for j, approx_type in enumerate(approx_types):
            ax = axes[i, j]

            subset = results_df[
                (results_df['data_type'] == data_type) &
                (results_df['approx_type'] == approx_type)
            ]

            # Skip empty subsets
            if len(subset) == 0:
                continue

            # Plot both parameters as histograms
            ax.hist(
                subset['param_0'], bins=20, alpha=0.6,
                label='Param 0', color='blue', edgecolor='black'
            )
            ax.hist(
                subset['param_1'], bins=20, alpha=0.6,
                label='Param 1', color='red', edgecolor='black'
            )

            # Compute statistics
            mu0, sigma0 = subset['param_0'].mean(), subset['param_0'].std()
            mu1, sigma1 = subset['param_1'].mean(), subset['param_1'].std()

            # Plot mean lines and confidence bands for param_0
            ax.axvline(mu0, color='blue', linestyle='--', linewidth=2, label=f'$\\mu_0$={mu0:.2f}')
            ax.axvspan(mu0 - sigma0, mu0 + sigma0, alpha=0.2, color='blue')

            # Plot mean lines and confidence bands for param_1
            ax.axvline(mu1, color='red', linestyle='--', linewidth=2, label=f'$\\mu_1$={mu1:.2f}')
            ax.axvspan(mu1 - sigma1, mu1 + sigma1, alpha=0.2, color='red')

            # Formatting
            ax.set_xlabel('Steepness Parameter')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Data: {data_type} | Approx: {approx_type}')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_parameter_summary_bars(
    results_df: pd.DataFrame,
    save_path: str = None
):
    """
    Plot bar charts showing mean ± std for param_0 and param_1 across all
    (data_type, approx_type) combinations.

    Creates a 1x2 subplot figure with:
    - Left panel: param_0 means with std error bars
    - Right panel: param_1 means with std error bars

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe with columns: ['data_type', 'approx_type', 'param_0',
        'param_1', ...]
    save_path : str, optional
        Path to save the figure. If None, displays the plot.
    """
    # Compute summary statistics
    summary = compute_confidence_intervals(results_df)
    
    # Create labels for x-axis: data_type/approx_type
    summary['label'] = summary['data_type'] + '\n' + summary['approx_type']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot param_0
    ax0 = axes[0]
    x_pos = np.arange(len(summary))
    ax0.bar(x_pos, summary['param_0_mean'], yerr=summary['param_0_std'],
            capsize=5, alpha=0.7, color='steelblue', edgecolor='black', linewidth=1.5)
    ax0.set_xticks(x_pos)
    ax0.set_xticklabels(summary['label'], rotation=0, ha='center')
    ax0.set_ylabel('Parameter Value', fontsize=12)
    ax0.set_title('param_0: Mean ± Std', fontsize=14, fontweight='bold')
    ax0.grid(True, alpha=0.3, axis='y')
    
    # Annotate with exact values
    for i, (mean, std) in enumerate(zip(summary['param_0_mean'], summary['param_0_std'])):
        ax0.text(i, mean + std + 0.05 * mean, f'{mean:.2f}±{std:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    # Plot param_1
    ax1 = axes[1]
    ax1.bar(x_pos, summary['param_1_mean'], yerr=summary['param_1_std'],
            capsize=5, alpha=0.7, color='coral', edgecolor='black', linewidth=1.5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(summary['label'], rotation=0, ha='center')
    ax1.set_ylabel('Parameter Value', fontsize=12)
    ax1.set_title('param_1: Mean ± Std', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Annotate with exact values
    for i, (mean, std) in enumerate(zip(summary['param_1_mean'], summary['param_1_std'])):
        ax1.text(i, mean + std + 0.05 * mean, f'{mean:.2f}±{std:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Learned Parameters Across Experiment Types', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()



def plot_parameter_scatter_split(
    results_df: pd.DataFrame,
    approx_type: str,
    save_path: str = None
):
    """
    Create scatterplot showing individual runs and mean±std for a specific
    approximation type.

    Creates a 1x2 subplot figure with:
    - Left panel: param_0 values across data types
    - Right panel: param_1 values across data types

    Each subplot shows all individual run values as scatter points with the
    mean as a larger marker and std as error bars.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe with columns: ['data_type', 'approx_type', 'param_0',
        'param_1', ...]
    approx_type : str
        Which approximation type to plot ('hill' or 'piecewise')
    save_path : str, optional
        Path to save the figure. If None, displays the plot.
    """
    # Filter for specific approx_type
    subset = results_df[results_df['approx_type'] == approx_type].copy()
    
    if len(subset) == 0:
        print(f"No data found for approx_type='{approx_type}'")
        return
    
    # Get unique data types and sort them
    data_types = sorted(subset['data_type'].unique())
    n_types = len(data_types)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Define colors for each data type
    colors = {'heaviside': '#2E86AB', 'hill': '#A23B72', 'piecewise': '#F18F01'}
    
    for param_idx, param_name in enumerate(['param_0', 'param_1']):
        ax = axes[param_idx]
        
        for i, data_type in enumerate(data_types):
            # Get data for this data_type
            data = subset[subset['data_type'] == data_type][param_name].values
            
            # Add jitter to x-position for visibility
            n_points = len(data)
            jitter = np.random.uniform(-0.15, 0.15, n_points)
            x_pos = np.ones(n_points) * i + jitter
            
            # Plot individual runs
            color = colors.get(data_type, 'gray')
            ax.scatter(x_pos, data, alpha=0.5, s=50, color=color, 
                      edgecolors='black', linewidth=0.5, label=f'{data_type} (runs)')
            
            # Compute and plot mean and std
            mean_val = data.mean()
            std_val = data.std()
            
            # Mean as larger diamond
            ax.scatter([i], [mean_val], marker='D', s=200, color=color,
                      edgecolors='black', linewidth=2, zorder=10,
                      label=f'{data_type} ($\\mu$={mean_val:.2f})')
            
            # Std as error bar
            ax.errorbar([i], [mean_val], yerr=[std_val], fmt='none',
                       ecolor='black', capsize=8, capthick=2, linewidth=2, zorder=9)
        
        # Formatting - create labels with ground truth parameters
        label_map = {
            'heaviside': 'heaviside',
            'hill': 'hill (n=50)',
            'piecewise': 'piecewise (h=0.01)'
        }
        x_labels = [label_map.get(dt, dt) for dt in data_types]
        
        ax.set_xticks(range(n_types))
        ax.set_xticklabels(x_labels, fontsize=11)
        ax.set_ylabel('Parameter Value', fontsize=12)
        ax.set_title(f'{param_name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xlim(-0.5, n_types - 0.5)
        
        # Legend - only show mean values to avoid clutter
        handles, labels = ax.get_legend_handles_labels()
        # Keep only the mean (diamond) entries
        mean_handles = [h for h, l in zip(handles, labels) if '$\\mu$=' in l]
        mean_labels = [l for l in labels if '$\\mu$=' in l]
        ax.legend(mean_handles, mean_labels, loc='best', fontsize=9)
    
    plt.suptitle(f'Learned Parameters - {approx_type.capitalize()} Approximation', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()



def plot_all_training_curves(
    data_type: str,
    approx_type: str,
    results_dir: str = 'results/training_curves',
    save_path: str = None
):
    """
    Plot all training curves for a specific (data_type, approx_type) combination.

    Creates a 1x3 subplot figure showing:
    - Left: Training loss vs epoch
    - Middle: param_0 vs epoch
    - Right: param_1 vs epoch

    All runs (typically 20) are shown as semi-transparent lines.

    Parameters
    ----------
    data_type : str
        Data type ('heaviside', 'hill', or 'piecewise')
    approx_type : str
        Approximation type ('hill' or 'piecewise')
    results_dir : str
        Path to directory containing training curve CSV files
    save_path : str, optional
        Path to save the figure. If None, displays the plot.
    """
    from pathlib import Path
    
    # Collect all curve files for this specific combination
    curves_path = Path(results_dir)
    pattern = f"{data_type}_{approx_type}_run*.csv"
    curve_files = sorted(curves_path.glob(pattern))
    
    if len(curve_files) == 0:
        print(f"No training curve files found matching {pattern} in {results_dir}")
        return
    
    print(f"Loading {len(curve_files)} training curves for {data_type}/{approx_type}...")
    
    # Load all data
    all_dfs = []
    for file_path in curve_files:
        df = pd.read_csv(file_path)
        all_dfs.append(df)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Define color based on data type
    color_map = {
        'heaviside': '#2E86AB',
        'hill': '#A23B72',
        'piecewise': '#F18F01'
    }
    color = color_map.get(data_type, 'gray')
    
    # Plot all runs
    for i, df in enumerate(all_dfs):
        # Loss plot
        axes[0].semilogy(df['epoch'], df['loss_total'], 
                       color=color, alpha=0.4, linewidth=1.5)
        
        # param_0 plot
        axes[1].plot(df['epoch'], df['param_0'],
                    color=color, alpha=0.4, linewidth=1.5)
        
        # param_1 plot
        axes[2].plot(df['epoch'], df['param_1'],
                    color=color, alpha=0.4, linewidth=1.5)
    
    # Format loss subplot
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Training Loss (log scale)', fontsize=12)
    axes[0].set_title('Training Loss Evolution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Format param_0 subplot
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('param_0 Value', fontsize=12)
    axes[1].set_title('param_0 Evolution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Format param_1 subplot
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('param_1 Value', fontsize=12)
    axes[2].set_title('param_1 Evolution', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    # Create title with ground truth info
    if data_type == 'hill':
        data_label = f'{data_type} (n=50)'
    elif data_type == 'piecewise':
        data_label = f'{data_type} (h=0.01)'
    else:
        data_label = data_type
    
    title = f'Training Curves: {data_label} data → {approx_type} approximation ({len(all_dfs)} runs)'
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved to: {save_path}")
    else:
        plt.show()


def compute_confidence_intervals(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute \mu \pm \sigma confidence intervals for each (data_type, approx_type) group.

    Groups the results by data_type and approx_type, then computes summary
    statistics including means, standard deviations, confidence interval bounds,
    number of runs, and loss statistics.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe with columns: ['data_type', 'approx_type', 'param_0',
        'param_1', 'final_loss', ...]

    Returns
    -------
    pd.DataFrame
        Summary dataframe with columns:
        - data_type, approx_type: grouping keys
        - param_0_mean, param_0_std: mean and std of param_0
        - param_0_ci_lower, param_0_ci_upper: \mu - \sigma and \mu + \sigma bounds
        - param_1_mean, param_1_std: mean and std of param_1
        - param_1_ci_lower, param_1_ci_upper: \mu - \sigma and \mu + \sigma bounds
        - n_runs: number of runs in the group
        - mean_final_loss, std_final_loss: statistics of final loss
    """
    summary = []

    for (data_type, approx_type), group in results_df.groupby(['data_type', 'approx_type']):
        mu0, sigma0 = group['param_0'].mean(), group['param_0'].std()
        mu1, sigma1 = group['param_1'].mean(), group['param_1'].std()

        summary.append({
            'data_type': data_type,
            'approx_type': approx_type,
            'param_0_mean': mu0,
            'param_0_std': sigma0,
            'param_0_ci_lower': mu0 - sigma0,
            'param_0_ci_upper': mu0 + sigma0,
            'param_1_mean': mu1,
            'param_1_std': sigma1,
            'param_1_ci_lower': mu1 - sigma1,
            'param_1_ci_upper': mu1 + sigma1,
            'n_runs': len(group),
            'mean_final_loss': group['final_loss'].mean(),
            'std_final_loss': group['final_loss'].std()
        })

    return pd.DataFrame(summary)

# PINN for Learning Steepness in Discontinuous ODEs

## Project Overview

This project uses Physics-Informed Neural Networks (PINNs) to learn steepness parameters when approximating discontinuous ODEs with smooth/piecewise functions.

**Core Question:** Given trajectory data from different vector field types (discontinuous Heaviside, steep Hill function, steep piecewise linear), can we learn the steepness parameter (Hill exponent `n` or piecewise width `h`) that best fits the data through PINN optimization?

## Systems

The project focuses on a two-dimensional discontinuous ODE system:

```
x_0' = -x_0 + U_1 + H(T_2 - x_1)(L_1 - U_1)
x_1' = -x_1 + U_2 + H(T_1 - x_0)(L_2 - U_2)
```

Where `H(·)` is the Heaviside function and parameters are `U = (5, 5)`, `L = (1, 1)`, `T = (3, 3)`.

**Approximations to Learn:**

1. **Hill function:** `H(\theta - x) \approx \theta^n / (\theta^n + x^n)` → learn `n`
2. **Piecewise linear:** Ramp with width `2h` → learn `h`

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
pinn_discontinuous/
├── src/
│   ├── __init__.py
│   ├── data_generator.py      # Generate trajectories from 3 vector field types
│   ├── models.py              # PINN architecture (SIREN-based)
│   ├── trainer.py             # Training loop with loss computation
│   ├── experiment_runner.py   # Batch experiments + hyperparameter management
│   └── utils.py               # Metrics, plotting, checkpointing
├── configs/
│   └── experiment_config.yaml # All hyperparameters
├── data/                      # Generated trajectories
│   ├── heaviside_trajectories.csv
│   ├── hill_trajectories.csv
│   └── piecewise_trajectories.csv
├── results/                   # Outputs
│   ├── experiment_results.csv # Main results table
│   ├── training_curves/       # Loss/parameter evolution per run
│   └── figures/               # Analysis plots
├── notebooks/
│   └── analysis.ipynb         # Posterior statistical analysis
├── requirements.txt
├── README.md
└── run_experiments.py         # Main entry point
```

## Usage

### Running Experiments

Run the full experimental suite (120 experiments by default: 3 data types × 2 approximation types × 20 runs):

```bash
python run_experiments.py
```

Or with a custom configuration:

```bash
python run_experiments.py --config configs/custom_config.yaml
```

### Configuration

Edit `configs/experiment_config.yaml` to customize:

- Number of trajectories and time points
- Model architecture (hidden layers, SIREN frequency)
- Training hyperparameters (learning rate, epochs, patience)
- Loss function weights
- Device selection ('mps', 'cuda', or 'cpu')

### Analysis

After experiments complete, run the analysis notebook:

```bash
jupyter notebook notebooks/analysis.ipynb
```

## Experimental Design

### Three Data Sources

1. **Heaviside (discontinuous):** True vector field with step functions
2. **Hill (steep continuous):** Vector field with `n = 50` (nearly discontinuous)
3. **Piecewise (steep continuous):** Vector field with `h = 0.01` (narrow transition)

### Two Approximation Types

1. **Hill approximation:** Learn 2 parameters `n = [n_0, n_1]`
2. **Piecewise approximation:** Learn 2 parameters `h = [h_0, h_1]`

### Experiment Matrix

```
3 data types × 2 approximation types × 20 random seeds = 120 training runs
```

For each (data_type, approx_type) pair:

- Compute mean \mu and standard deviation \sigma of learned parameters
- Report confidence interval [\mu - \sigma, \mu + \sigma]
- Analyze robustness and convergence patterns

## Model Architecture

The project uses a **SIREN (Sinusoidal Representation Network)** architecture:

- Periodic activation functions good for high-frequency features
- Input: (t, ic_0, ic_1) → time and initial conditions
- Output: (x_0, x_1) → predicted states
- Learnable steepness parameters (log-parameterized for positivity)

## Loss Function

Combined loss:

- **Data loss:** MSE between prediction and ground truth trajectories
- **Physics loss:** ODE residual via automatic differentiation
- **IC loss:** Initial condition constraint (soft penalty)

## Expected Outputs

1. **Main Results Table** (`results/experiment_results.csv`):
   - 120 rows with final parameters, losses, convergence epochs
   - Statistical summary printed to console

2. **Training Curves** (`results/training_curves/*.csv`):
   - Per-run evolution of losses and parameters

3. **Analysis Figures** (`results/figures/`):
   - Parameter distributions with confidence intervals
   - Loss vs parameter correlations
   - Convergence time histograms

## Success Criteria

1. All 120 experiments complete without errors
2. Gathered learned parameters, training data, etc.
3. Confidence intervals [\mu - \sigma, \mu + \sigma] have reasonable width (\sigma/\mu < 0.3)
4. Learned params are validated by DSGRN (or analytical) bounds

## Dependencies

- PyTorch ≥ 2.0.0
- NumPy ≥ 1.24.0
- SciPy ≥ 1.10.0
- Pandas ≥ 2.0.0
- Matplotlib ≥ 3.7.0
- Seaborn ≥ 0.12.0
- PyYAML ≥ 6.0
- tqdm ≥ 4.65.0
- Jupyter ≥ 1.0.0

## License

[To be determined]

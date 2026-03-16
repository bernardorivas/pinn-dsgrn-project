"""
Scatter plots of timeseries data: phase portraits (x0 vs x1) and
individual time series (t vs x0, t vs x1).

ICs marked with '*' in a single color; trajectory points marked with 'o',
one color per trajectory. No connecting lines.

Output structure:
  results/timeseries_plots/{par_index}/d{d_value}_phase.png
  results/timeseries_plots/{par_index}/d{d_value}_x0.png
  results/timeseries_plots/{par_index}/d{d_value}_x1.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT / "data"
OUT_DIR = PROJECT / "results" / "timeseries_plots"

IC_COLOR = "black"
IC_MARKER = "*"
IC_SIZE = 60
TRAJ_COLOR = "C0"
TRAJ_MARKER = "o"
TRAJ_SIZE = 6
TRAJ_ALPHA = 0.4


def plot_par_index(par_index: str):
    par_dir = DATA_DIR / par_index
    out_dir = OUT_DIR / par_index
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(par_dir.glob("*.csv"))
    for csv_path in csv_files:
        prefix = csv_path.stem  # e.g. "01"
        df = pd.read_csv(csv_path)

        traj_ids = df["traj_id"].unique()

        # --- Phase portrait: x0 vs x1 ---
        fig, ax = plt.subplots(figsize=(6, 6))
        for tid in traj_ids:
            sub = df[df["traj_id"] == tid]
            ic = sub.iloc[[0]]
            traj = sub.iloc[1:]
            ax.scatter(traj["x0"], traj["x1"], c=TRAJ_COLOR, marker=TRAJ_MARKER,
                       s=TRAJ_SIZE, alpha=TRAJ_ALPHA, linewidths=0)
            ax.scatter(ic["x0"], ic["x1"], c=IC_COLOR, marker=IC_MARKER,
                       s=IC_SIZE, zorder=5)
        ax.set_xlabel("x0")
        ax.set_ylabel("x1")
        ax.set_title(f"par {par_index}  |  {prefix}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_phase.png", dpi=150)
        plt.close(fig)

        # --- x0 vs t ---
        fig, ax = plt.subplots(figsize=(8, 4))
        for tid in traj_ids:
            sub = df[df["traj_id"] == tid]
            ic = sub.iloc[[0]]
            traj = sub.iloc[1:]
            ax.scatter(traj["t"], traj["x0"], c=TRAJ_COLOR, marker=TRAJ_MARKER,
                       s=TRAJ_SIZE, alpha=TRAJ_ALPHA, linewidths=0)
            ax.scatter(ic["t"], ic["x0"], c=IC_COLOR, marker=IC_MARKER,
                       s=IC_SIZE, zorder=5)
        ax.set_xlabel("t")
        ax.set_ylabel("x0")
        ax.set_title(f"par {par_index}  |  {prefix}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_x0.png", dpi=150)
        plt.close(fig)

        # --- x1 vs t ---
        fig, ax = plt.subplots(figsize=(8, 4))
        for tid in traj_ids:
            sub = df[df["traj_id"] == tid]
            ic = sub.iloc[[0]]
            traj = sub.iloc[1:]
            ax.scatter(traj["t"], traj["x1"], c=TRAJ_COLOR, marker=TRAJ_MARKER,
                       s=TRAJ_SIZE, alpha=TRAJ_ALPHA, linewidths=0)
            ax.scatter(ic["t"], ic["x1"], c=IC_COLOR, marker=IC_MARKER,
                       s=IC_SIZE, zorder=5)
        ax.set_xlabel("t")
        ax.set_ylabel("x1")
        ax.set_title(f"par {par_index}  |  {prefix}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_x1.png", dpi=150)
        plt.close(fig)

        print(f"  {par_index}/{prefix} done")


def main():
    par_indices = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())
    print(f"Parameter indices: {par_indices}")
    for par_index in par_indices:
        print(f"Plotting {par_index} ...")
        plot_par_index(par_index)
    print(f"Output in {OUT_DIR}")


if __name__ == "__main__":
    main()

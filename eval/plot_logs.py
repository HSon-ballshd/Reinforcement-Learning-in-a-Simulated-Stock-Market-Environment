"""
Read training CSV logs and generate visualisation plots.

Usage:
    python -m eval.plot_logs
    python -m eval.plot_logs --exp h3
    python -m eval.plot_logs --all
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------
def load_csv(path: Path) -> dict[str, list]:
    """Parse a training CSV into a dict of column name -> values."""
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    result: dict[str, list] = {k: [] for k in rows[0].keys()}
    for row in rows:
        for k, v in row.items():
            try:
                result[k].append(float(v))
            except ValueError:
                result[k].append(v)
    return result


def load_logs(exp: str, seed: int) -> dict:
    """Load logs for a given experiment and seed."""
    path = Path(f"outputs/{exp}_seed{seed}_log.csv")
    if path.exists():
        return load_csv(path)
    # Fallback: scan for newest matching file
    matches = sorted(Path("outputs").glob(f"{exp}_*_seed{seed}_log.csv"))
    if matches:
        return load_csv(matches[-1])
    return {}


def load_agent_logs(exp: str, seeds: list[int]) -> dict[str, dict]:
    """Load logs for multiple seeds; return {seed -> logs}."""
    return {s: load_logs(exp, s) for s in seeds}


def rolling_mean(arr: list[float], window: int) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if len(a) < window:
        return a
    pad = np.concatenate([[a[0]] * (window - 1), a])
    return np.convolve(pad, np.ones(window) / window, mode="valid")


# ------------------------------------------------------------------
# Per-run plot (2×2)
# ------------------------------------------------------------------
def plot_single_run(logs: dict, title: str, output_path: Path) -> None:
    steps = [int(s) for s in logs.get("step", [])]
    n_steps = steps[-1] if steps else 1

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # 1. Loss
    ax = axes[0, 0]
    loss = logs.get("loss", [])
    if loss:
        x = np.arange(len(loss))
        ax.plot(x[::max(1, len(loss)//500)],
                np.array(loss)[::max(1, len(loss)//500)],
                color="tab:blue", linewidth=0.8)
        ax.set_xlabel("train_step")
        ax.set_ylabel("TD Loss")
        ax.set_title("Loss vs Training Step")
    else:
        ax.text(0.5, 0.5, "No loss data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Loss vs Training Step")

    # 2. Epsilon
    ax = axes[0, 1]
    eps = logs.get("epsilon", [])
    if eps:
        ax.plot(eps, color="tab:orange", linewidth=1.0)
        ax.set_xlabel("training_step")
        ax.set_ylabel("ε")
        ax.set_title("Epsilon Decay")
        ax.set_ylim(0.0, 1.05)
        ax.set_xlim(0, n_steps)
    else:
        ax.text(0.5, 0.5, "No epsilon data", ha="center", va="center", transform=ax.transAxes)

    # 3. Episode return (rolling mean)
    ax = axes[1, 0]
    ep_ret = logs.get("episode_return", [])
    if ep_ret:
        smoothed = rolling_mean(ep_ret, 20)
        ax.plot(smoothed, color="tab:green", linewidth=1.2)
        ax.set_xlabel("episode")
        ax.set_ylabel("Episode Return (clipped)")
        ax.set_title("Episode Return (20-ep rolling mean)")
    else:
        ax.text(0.5, 0.5, "No episode data", ha="center", va="center", transform=ax.transAxes)

    # 4. Eval return at checkpoint
    ax = axes[1, 1]
    eval_ret = logs.get("eval_return_pct", [])
    if eval_ret and steps:
        ax.plot(steps, eval_ret, color="tab:red", marker="o", markersize=3, linewidth=1.2)
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xlabel("training_step")
        ax.set_ylabel("Mean Eval Return (%)")
        ax.set_title("Eval Return vs Training Step")
        ax.set_xlim(0, n_steps)
    else:
        ax.text(0.5, 0.5, "No eval data", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {output_path}")


# ------------------------------------------------------------------
# H3 comparison overlay
# ------------------------------------------------------------------
def plot_h3_comparison(
    dqn_logs: dict[str, dict],
    ra_logs:  dict[str, dict],
    output_path: Path,
) -> None:
    """Overlay eval-return curves for all DQN and RA-DQN seeds."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for seed, logs in dqn_logs.items():
        steps = [int(s) for s in logs.get("step", [])]
        ev = logs.get("eval_return_pct", [])
        if ev and steps:
            ax.plot(steps, ev, label=f"DQN seed={seed}", marker="o", markersize=3)

    for seed, logs in ra_logs.items():
        steps = [int(s) for s in logs.get("step", [])]
        ev = logs.get("eval_return_pct", [])
        if ev and steps:
            ax.plot(steps, ev, label=f"RA-DQN seed={seed}", marker="s", markersize=3)

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Eval Return (%)")
    ax.set_title("H3: Plain DQN vs Regime-Aware DQN — Eval Return")
    ax.legend(framealpha=0.6)
    ax.set_xlim(0, None)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {output_path}")


# ------------------------------------------------------------------
# Summary table
# ------------------------------------------------------------------
def plot_summary_table(results: dict, output_path: Path) -> None:
    """Render a plain-text summary table as a PNG."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    # Build table data
    rows = [["Experiment", "Metric", "Value"]]
    for exp_name, exp_data in results.items():
        for metric, value in exp_data.items():
            rows.append([exp_name, metric, f"{value:.2f}" if isinstance(value, float) else str(value)])

    table = ax.table(
        cellText=rows[1:],
        colLabels=rows[0],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {output_path}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training logs from CSV files")
    parser.add_argument("--exp", choices=["h1", "h3"], default=None,
                        help="Which experiment to plot (default: plot all found)")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 123, 456],
                        help="Seed(s) to plot")
    parser.add_argument("--all", action="store_true",
                        help="Plot everything found in outputs/")
    args = parser.parse_args()

    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid": True,
        "grid.alpha": 0.4,
        "font.size": 10,
    })

    seeds = args.seeds

    if args.exp in (None, "h1"):
        for seed in seeds:
            logs = load_logs("h1", seed)
            if logs:
                plot_single_run(
                    logs,
                    title=f"H1 — DQN Training Progress (seed={seed})",
                    output_path=Path(f"outputs/h1_dqn_seed{seed}.png"),
                )

    if args.exp in (None, "h3"):
        # DQN logs (from H1 checkpoints re-eval)
        dqn_logs = {s: load_logs("h3_dqn", s) for s in seeds}
        ra_logs  = {s: load_logs("h3_ra",  s) for s in seeds}

        # Per-seed RA-DQN plots
        for seed in seeds:
            if ra_logs[seed]:
                plot_single_run(
                    ra_logs[seed],
                    title=f"H3 — RA-DQN Training Progress (seed={seed})",
                    output_path=Path(f"outputs/h3_ra_dqn_seed{seed}.png"),
                )

        # H3 comparison overlay (only seeds that have both DQN and RA logs)
        common_seeds = [s for s in seeds if dqn_logs.get(s) or ra_logs.get(s)]
        if common_seeds:
            plot_h3_comparison(
                {s: dqn_logs[s] for s in common_seeds if dqn_logs.get(s)},
                {s: ra_logs[s]  for s in common_seeds if ra_logs.get(s)},
                output_path=Path("outputs/h3_comparison.png"),
            )

    print("\nDone. Plots saved to outputs/")


if __name__ == "__main__":
    main()

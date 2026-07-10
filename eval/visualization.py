"""
Training-progress visualisations for the eval harness.

Produces:
  - One 2×2 figure per training run (loss, epsilon, episode-return, eval-return)
  - A combined comparison figure for H3 (plain DQN vs regime-aware DQN)

All plots are saved to `outputs/` as PNG files.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any

# Style — whitegrid, no border on top/right axes
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})


# ------------------------------------------------------------------
# Single-run plots
# ------------------------------------------------------------------
def plot_training_run(
    logs: dict[str, list[float]],
    title: str,
    output_path: Path,
    eval_every: int = 5_000,
    total_steps: int | None = None,
) -> None:
    """
    Render a 2×2 subplot figure for one training run.

    Args:
        logs: dict with keys:
              - loss            (sampled every train_step that had enough replay)
              - epsilon         (one value per training step)
              - episode_return  (one value per completed episode)
              - eval_returns    (one value per eval_every checkpoint)
        title: figure title
        output_path: where to save the PNG
        eval_every: steps between eval checkpoints (used to build x-axis)
        total_steps: total training steps (for the eval-return x-axis)
    """
    loss_vals     = logs.get("loss", [])
    epsilon_vals  = logs.get("epsilon", [])
    ep_ret_vals   = logs.get("episode_return", [])
    eval_vals     = logs.get("eval_returns", [])

    n_steps = total_steps or len(epsilon_vals)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # ---- 1. Loss ----
    ax = axes[0, 0]
    if loss_vals:
        # Downsample for plotting (every 100th point keeps the line readable)
        step_axis = np.arange(len(loss_vals))
        plot_y = loss_vals[:: max(1, len(loss_vals) // 500)]
        plot_x = step_axis[:: max(1, len(loss_vals) // 500)]
        ax.plot(plot_x, plot_y, color="tab:blue", linewidth=0.8)
        ax.set_xlabel("train_step")
        ax.set_ylabel("TD Loss")
        ax.set_title("Loss vs Training Step")
        ax.set_xlim(0, len(loss_vals))
    else:
        ax.text(0.5, 0.5, "No loss data yet", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Loss vs Training Step")

    # ---- 2. Epsilon ----
    ax = axes[0, 1]
    x_eps = np.arange(len(epsilon_vals))
    ax.plot(x_eps, epsilon_vals, color="tab:orange", linewidth=1.0)
    ax.set_xlabel("training_step")
    ax.set_ylabel("ε")
    ax.set_title("Epsilon Decay")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(0, n_steps)

    # ---- 3. Episode Return (clipped) ----
    ax = axes[1, 0]
    if ep_ret_vals:
        # Rolling mean over 20 episodes to smooth the curve
        smoothed = _rolling_mean(ep_ret_vals, 20)
        ax.plot(smoothed, color="tab:green", linewidth=1.2)
        ax.set_xlabel("episode")
        ax.set_ylabel("Episode Return (clipped)")
        ax.set_title("Episode Return (20-ep rolling mean)")
        ax.set_xlim(0, len(ep_ret_vals))
    else:
        ax.text(0.5, 0.5, "No episodes yet", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Episode Return (20-ep rolling mean)")

    # ---- 4. Eval Return at checkpoint ----
    ax = axes[1, 1]
    if eval_vals:
        x_eval = np.arange(eval_every, n_steps + 1, eval_every)[: len(eval_vals)]
        ax.plot(x_eval, eval_vals, color="tab:red", marker="o", markersize=3, linewidth=1.2)
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xlabel("training_step")
        ax.set_ylabel("Mean Eval Return (%)")
        ax.set_title("Eval Return vs Training Step")
        ax.set_xlim(0, n_steps)
    else:
        ax.text(0.5, 0.5, "No eval data yet", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Eval Return vs Training Step")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved plot → {output_path}")


def _rolling_mean(arr: list[float], window: int) -> np.ndarray:
    a = np.array(arr, dtype=np.float32)
    if len(a) < window:
        return a
    pad = np.concatenate([[a[0]] * (window - 1), a])
    return np.convolve(pad, np.ones(window) / window, mode="valid")


# ------------------------------------------------------------------
# H3 comparison plot
# ------------------------------------------------------------------
def plot_h3_comparison(
    dqn_logs: dict[str, list[float]],
    ra_logs:  dict[str, list[float]],
    dqn_title: str = "DQN",
    ra_title:  str = "DQN+Regime",
    output_path: Path | str = "outputs/h3_comparison.png",
    eval_every: int = 5_000,
    total_steps: int | None = None,
) -> None:
    """
    Side-by-side eval-return comparison for plain DQN vs regime-aware DQN.

    Overlays both agents' eval-return curves on a single axes so the
    convergence behaviour is directly visible.
    """
    output_path = Path(output_path)
    n_steps = total_steps or 20_000

    dqn_eval = np.array(dqn_logs.get("eval_returns", []), dtype=np.float32)
    ra_eval  = np.array(ra_logs.get("eval_returns",  []), dtype=np.float32)

    x_dqn = np.arange(eval_every, n_steps + 1, eval_every)[: len(dqn_eval)]
    x_ra  = np.arange(eval_every, n_steps + 1, eval_every)[: len(ra_eval)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_dqn, dqn_eval, label=dqn_title, color="tab:blue",   marker="o", markersize=3)
    ax.plot(x_ra,  ra_eval,  label=ra_title,  color="tab:purple", marker="s", markersize=3)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Eval Return (%)")
    ax.set_title("H3: Plain DQN vs Regime-Aware DQN — Eval Return Comparison")
    ax.legend(framealpha=0.6)
    ax.set_xlim(0, n_steps)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved comparison plot → {output_path}")

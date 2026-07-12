"""
Training-progress visualisations for the eval harness.

Produces:
  - One 2×2 figure per training run (loss, epsilon, episode-return, eval-return)
  - A combined comparison figure for H3 (plain DQN vs regime-aware DQN)
  - H3 evaluation summary: 2×2 panel figure from h3_eval results

All plots are saved to `outputs/` as PNG files.
"""

from __future__ import annotations

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.gridspec import GridSpec
from pathlib import Path

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

# H3 plot palette — blue for plain DQN, purple for RA-DQN
_C = {
    "dqn": "#1f77b4",   # tab:blue
    "ra":  "#7e2c8a",   # deep purple — visually distinct from training purple
}


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
    print(f"  Saved plot -> {output_path}")


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
    print(f"  Saved comparison plot -> {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# H3 Evaluation Summary Plot (scientific report figure)
# ══════════════════════════════════════════════════════════════════════════════

def plot_h3_eval_summary(
    summary_json: Path | str,
    runs_csv: Path | str,
    output_path: Path | str | None = None,
) -> None:
    """
    Render a 2x2 panel figure from h3_eval results for a scientific report.

    Panel layout
    ------------
    (0,0)  Return distributions  — violin + box + scatter overlay, log-scale y
    (0,1)  Sharpe-like ratio     — violin + box
    (1,0)  Trading activity       — grouped bars: buys / sells / holds
    (1,1)  Risk profile          — max drawdown bars + legend / verdict text

    Parameters
    ----------
    summary_json : path to h3_eval_summary_{ts}.json
    runs_csv     : path to h3_eval_runs_{ts}.csv
    output_path  : if None, auto-generates outputs/h3_eval_fig_{ts}.png
    """
    summary_json = Path(summary_json)
    runs_csv     = Path(runs_csv)
    if output_path is None:
        ts = summary_json.stem.replace("h3_eval_summary_", "")
        output_path = summary_json.parent / f"h3_eval_fig_{ts}.png"
    else:
        output_path = Path(output_path)

    with open(summary_json) as f:
        summary = json.load(f)

    import csv as _csv
    runs = []
    with open(runs_csv) as f:
        for row in _csv.DictReader(f):
            # regime_time is stored as JSON string in the CSV
            row["total_return_pct"]   = float(row["total_return_pct"])
            row["final_value"]        = float(row["final_value"])
            row["n_buys"]             = int(row["n_buys"])
            row["n_sells"]            = int(row["n_sells"])
            row["n_holds"]            = int(row["n_holds"])
            row["n_trades"]           = int(row["n_trades"])
            row["sharpe_like"]        = float(row["sharpe_like"])
            row["max_drawdown_pct"]   = float(row["max_drawdown_pct"])
            runs.append(row)

    dqn_runs = [r for r in runs if r["agent_type"] == "DQN"]
    ra_runs  = [r for r in runs if r["agent_type"] == "RA-DQN"]

    tests = summary.get("statistical_tests", {}).get("return_pct", {})
    wt    = tests.get("welch_t", {})
    cfg   = summary.get("config", {})
    verdict = "PASS" if summary.get("h3_pass") else "FAIL"

    # ── Data arrays ────────────────────────────────────────────────────────────
    dqn_ret  = np.array([r["total_return_pct"] for r in dqn_runs])
    ra_ret   = np.array([r["total_return_pct"] for r in ra_runs])
    dqn_sharpe = np.array([r["sharpe_like"]    for r in dqn_runs])
    ra_sharpe  = np.array([r["sharpe_like"]    for r in ra_runs])
    dqn_mdd    = np.array([r["max_drawdown_pct"] for r in dqn_runs])
    ra_mdd     = np.array([r["max_drawdown_pct"]  for r in ra_runs])
    dqn_buys   = np.array([r["n_buys"]  for r in dqn_runs], dtype=float)
    ra_buys    = np.array([r["n_buys"]  for r in ra_runs],  dtype=float)
    dqn_sells  = np.array([r["n_sells"] for r in dqn_runs], dtype=float)
    ra_sells   = np.array([r["n_sells"] for r in ra_runs],  dtype=float)
    dqn_holds  = np.array([r["n_holds"] for r in dqn_runs], dtype=float)
    ra_holds   = np.array([r["n_holds"] for r in ra_runs],  dtype=float)

    # ── Figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 10))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle(
        f"H3 Evaluation: Regime-Aware DQN vs Plain DQN  |  "
        f"{cfg.get('n_eval_seeds', 25)} eval seeds × {cfg.get('n_train_seeds', 3)} train seeds  |  "
        f"Verdict: {verdict}",
        fontsize=12, fontweight="bold", y=0.98,
    )

    # -------------------------------------------------------------------------
    # Panel (0,0): Return distributions — violin + box + scatter
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])

    positions = [1, 2]
    data_ret  = [dqn_ret, ra_ret]

    # Clip negative values for log scale; replace <= 0 with a tiny positive floor
    _clip = lambda a: np.where(a > 0, a, 1e-4)

    parts = ax.violinplot(
        [_clip(dqn_ret), _clip(ra_ret)],
        positions=positions,
        showmeans=False, showmedians=False, showextrema=False,
    )
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(_C["dqn"] if i == 0 else _C["ra"])
        body.set_alpha(0.25)
    for key in ["cbars", "cmins", "cmaxes"]:
        for line in parts.get(key, []):
            line.set_visible(False)

    bp = ax.boxplot(
        data_ret,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        manage_ticks=False,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color="grey"),
        capprops=dict(color="grey"),
        flierprops=dict(marker="o", markersize=2, alpha=0.4,
                        markerfacecolor="grey"),
    )
    bp["boxes"][0].set_facecolor(_C["dqn"])
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(_C["ra"])
    bp["boxes"][1].set_alpha(0.7)

    # Scatter overlay — jittered
    rng = np.random.default_rng(0)
    for i, (arr, pos) in enumerate(zip(data_ret, positions)):
        jitter = rng.uniform(-0.18, 0.18, len(arr))
        ax.scatter(pos + jitter, arr, s=12, alpha=0.35, zorder=5,
                   color=_C["dqn"] if i == 0 else _C["ra"], edgecolors="none")

    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels(["DQN", "RA-DQN"])
    ax.set_ylabel("Total Return (%)  [log scale]")
    ax.set_title("(A) Return Distribution")
    ax.set_xlim(0.3, 2.7)
    ax.yaxis.set_major_formatter(_PercentFormatter())

    # Annotation: means
    for arr, pos, col in [(dqn_ret, 1, _C["dqn"]), (ra_ret, 2, _C["ra"])]:
        ax.axhline(np.mean(arr), xmin=(pos - 0.6) / 2.4,
                   xmax=(pos + 0.6) / 2.4,
                   color=col, linewidth=1.5, linestyle="--", zorder=4, alpha=0.9)
        ax.text(pos + 0.32, np.mean(arr),
                f"μ={_fmt_num(np.mean(arr))}",
                fontsize=8, va="center", color=col)

    # -------------------------------------------------------------------------
    # Panel (0,1): Sharpe-like ratio violin
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])

    parts = ax.violinplot([dqn_sharpe, ra_sharpe], positions=[1, 2],
                          showmeans=True, showmedians=True, showextrema=False)
    colors_s = [_C["dqn"], _C["ra"]]
    for i, (body, col) in enumerate(zip(parts["bodies"], colors_s)):
        body.set_facecolor(col)
        body.set_alpha(0.5)
    parts["cmeans"].set_color(["#555", "#888"])
    parts["cmeans"].set_linestyle("--")
    parts["cmeans"].set_linewidth(1.2)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(2)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["DQN", "RA-DQN"])
    ax.set_ylabel("Sharpe-Like Ratio")
    ax.set_title("(B) Risk-Adjusted Return")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlim(0.3, 2.7)

    # -------------------------------------------------------------------------
    # Panel (1,0): Trading activity — grouped bars
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])

    x   = np.arange(3)          # 0=Buys, 1=Sells, 2=Holds
    w   = 0.35
    dqn_means = [dqn_buys.mean(), dqn_sells.mean(), dqn_holds.mean()]
    ra_means  = [ra_buys.mean(),  ra_sells.mean(),  ra_holds.mean()]
    dqn_stds  = [dqn_buys.std(),  dqn_sells.std(),  dqn_holds.std()]
    ra_stds   = [ra_buys.std(),   ra_sells.std(),   ra_holds.std()]
    labels = ["Buys", "Sells", "Holds"]

    ax.bar(x - w/2, dqn_means, w, label="DQN", color=_C["dqn"], alpha=0.8,
           yerr=dqn_stds, capsize=3, error_kw={"linewidth": 1})
    ax.bar(x + w/2, ra_means,  w, label="RA-DQN", color=_C["ra"], alpha=0.8,
           yerr=ra_stds,  capsize=3, error_kw={"linewidth": 1})

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Count per Episode")
    ax.set_title("(C) Trading Activity")
    ax.legend(framealpha=0.6, fontsize=9)

    # -------------------------------------------------------------------------
    # Panel (1,1): Max drawdown + stat verdict text
    # -------------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])

    x_mdd = [1, 2]
    mdd_means = [dqn_mdd.mean(), ra_mdd.mean()]
    mdd_stds  = [dqn_mdd.std(),  ra_mdd.std()]

    bars = ax.bar(x_mdd, mdd_means, width=0.5, color=[_C["dqn"], _C["ra"]],
                  alpha=0.8, yerr=mdd_stds, capsize=4,
                  error_kw={"linewidth": 1})
    ax.set_xticks(x_mdd)
    ax.set_xticklabels(["DQN", "RA-DQN"])
    ax.set_ylabel("Max Drawdown (%)")
    ax.set_title("(D) Risk Profile — Max Drawdown")
    ax.set_ylim(0, max(mdd_means) * 1.35)

    for bar, mean in zip(bars, mdd_means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(mdd_stds) + 1.5,
                f"{mean:.1f}%", ha="center", va="bottom", fontsize=9)

    # Statistical verdict text box
    p_val  = wt.get("p_value", float("nan"))
    t_val  = wt.get("t_stat",  float("nan"))
    d_val  = wt.get("cohens_d", float("nan"))
    n1, n2 = wt.get("n1", 0), wt.get("n2", 0)
    ci_lo  = tests.get("bootstrap_ci", {}).get("ci_lower", float("nan"))
    ci_hi  = tests.get("bootstrap_ci", {}).get("ci_upper", float("nan"))

    stat_lines = [
        f"Welch t-test (n={n1},{n2}):",
        f"  t = {t_val:.3f},  p = {p_val:.4f}",
        f"  Cohen's d = {d_val:.3f}",
        f"Bootstrap 95% CI:",
        f"  [{ci_lo:.0f}% , {ci_hi:.0f}%]",
    ]
    ax.text(0.98, 0.02, "\n".join(stat_lines),
            transform=ax.transAxes, fontsize=7.5, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="grey", alpha=0.85),
            family="monospace")

    # ── Save ────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved H3 eval figure -> {output_path}")


def _fmt_num(x: float) -> str:
    """Compact numeric label for plot annotations."""
    if x >= 1e9:   return f"{x/1e9:.1f}B"
    if x >= 1e6:    return f"{x/1e6:.1f}M"
    if x >= 1e3:    return f"{x/1e3:.1f}K"
    if x <= -1e9:   return f"{-x/1e9:.1f}B"
    if x <= -1e6:   return f"{-x/1e6:.1f}M"
    if x <= -1e3:   return f"{-x/1e3:.1f}K"
    if x == 0:      return "0"
    return f"{x:.1f}"


class _PercentFormatter(matplotlib.ticker.Formatter):
    """Matplotlib axis formatter that turns large numbers into compact K/M notation."""
    def __call__(self, x, pos=None):
        return _fmt_num(x)


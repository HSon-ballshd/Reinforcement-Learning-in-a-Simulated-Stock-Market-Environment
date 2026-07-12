"""
H1 Evaluation Scientific Report Figure

Generates a 2×2 panel figure from h1_eval results:

Panel layout
-------------
(A) Mean return comparison  — grouped bar chart with 95% bootstrap CI
(B) Per-run return scatter — jittered scatter coloured by agent type
(C) Sharpe-like ratio      — violin + box for risk-adjusted return
(D) Statistical verdict    — table of Welch t-test results per comparison

Usage (called automatically by h1_eval.py):
    from eval.h1_visualization import plot_h1_eval_summary
    plot_h1_eval_summary("outputs/h1_eval_summary_*.json",
                          "outputs/h1_eval_runs_*.csv")
"""

from __future__ import annotations

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.gridspec import GridSpec
from pathlib import Path


# ── Palette ──────────────────────────────────────────────────────
_C = {
    "DQN":           "#1f77b4",   # tab:blue
    "Random":        "#7f7f7f",   # grey
    "BuyAndHold":    "#ff7f0e",   # orange
    "MeanReversion": "#2ca02c",   # green
}


# ── Helpers ────────────────────────────────────────────────────────
def _fmt_num(x: float) -> str:
    if x >= 1e9:   return f"{x/1e9:.1f}B"
    if x >= 1e6:   return f"{x/1e6:.1f}M"
    if x >= 1e3:   return f"{x/1e3:.1f}K"
    if x <= -1e9:  return f"{-x/1e9:.1f}B"
    if x <= -1e6:  return f"{-x/1e6:.1f}M"
    if x <= -1e3:  return f"{-x/1e3:.1f}K"
    return f"{x:.0f}"


def _bootstrap_ci(data: np.ndarray, n_bootstrap: int = 10_000, ci: float = 0.95):
    rng = np.random.default_rng(42)
    means = [np.mean(rng.choice(data, size=len(data), replace=True))
             for _ in range(n_bootstrap)]
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (ci + (1 - ci) / 2) * 100)
    return float(lo), float(hi)


# ── Main ─────────────────────────────────────────────────────────
def plot_h1_eval_summary(
    summary_json: Path | str,
    runs_csv: Path | str,
    output_path: Path | str | None = None,
) -> None:
    summary_json = Path(summary_json)
    runs_csv     = Path(runs_csv)
    if output_path is None:
        ts = summary_json.stem.replace("h1_eval_summary_", "")
        output_path = summary_json.parent / f"h1_eval_fig_{ts}.png"
    else:
        output_path = Path(output_path)

    with open(summary_json) as f:
        summary = json.load(f)

    cfg        = summary["config"]
    agg        = summary["aggregate"]
    stat_tests = summary["statistical_tests"]
    verdict    = "PASS" if summary["h1_pass"] else "FAIL"
    agents     = list(agg.keys())

    # ── Load runs CSV ──────────────────────────────────────────────
    import csv as _csv
    runs = []
    with open(runs_csv) as f:
        for row in _csv.DictReader(f):
            row = dict(row)
            row["total_return_pct"] = float(row["total_return_pct"])
            row["final_value"]      = float(row["final_value"])
            row["sharpe_like"]     = float(row["sharpe_like"])
            row["max_drawdown_pct"]= float(row["max_drawdown_pct"])
            row["n_buys"]          = int(row["n_buys"])
            row["n_sells"]         = int(row["n_sells"])
            row["n_holds"]         = int(row["n_holds"])
            runs.append(row)

    # Per-agent data arrays
    ret_by_agent = {a: np.array([r["total_return_pct"] for r in runs
                                 if r["agent_type"] == a]) for a in agents}
    sharpe_by_agent = {a: np.array([r["sharpe_like"] for r in runs
                                    if r["agent_type"] == a]) for a in agents}
    mdd_by_agent = {a: np.array([r["max_drawdown_pct"] for r in runs
                                  if r["agent_type"] == a]) for a in agents}

    n_train = cfg["n_train_seeds"]
    n_eval  = cfg["n_eval_seeds"]

    # ── Figure ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 9))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)
    fig.suptitle(
        f"H1 Evaluation: DQN vs Baselines  |  "
        f"{n_train} train seeds × {n_eval} eval seeds  |  Verdict: {verdict}",
        fontsize=12, fontweight="bold", y=0.98,
    )

    # ────────────────────────────────────────────────────────────────
    # Panel A: Mean return — grouped bar + bootstrap CI
    # ────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])

    x    = np.arange(len(agents))
    w    = 0.55
    means = [agg[a]["mean"] for a in agents]
    stds  = [agg[a]["std"]  for a in agents]
    colors = [_C.get(a, "#888") for a in agents]

    # Bootstrap CI for mean (more accurate than ±std for small n)
    ci_lo_list, ci_hi_list = [], []
    for a in agents:
        lo, hi = _bootstrap_ci(ret_by_agent[a])
        ci_lo_list.append(means[agents.index(a)] - lo)
        ci_hi_list.append(hi - means[agents.index(a)])

    bars = ax.bar(x, means, w, color=colors, alpha=0.82, zorder=3)
    ax.errorbar(x, means,
                yerr=[ci_lo_list, ci_hi_list],
                fmt="none", color="black", capsize=5, linewidth=1.5, zorder=4)

    # Value labels — position above bar
    max_h = max(means)
    for i, (bar, m) in enumerate(zip(bars, means)):
        y_label = bar.get_height() + ci_hi_list[i] + max_h * 0.01
        ax.text(bar.get_x() + bar.get_width() / 2, y_label,
                f"{_fmt_num(m)}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(agents, fontsize=9)
    ax.set_ylabel("Mean Return (%)")
    ax.set_title("(A) Mean Return vs Baselines")
    ax.grid(axis="y", alpha=0.4, zorder=0)

    # ────────────────────────────────────────────────────────────────
    # Panel B: Per-run scatter — jittered, coloured
    # ────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])

    rng = np.random.default_rng(42)
    for i, agent in enumerate(agents):
        arr = ret_by_agent[agent]
        jitter = rng.uniform(-0.2, 0.2, len(arr))
        ax.scatter(np.full_like(arr, i) + jitter, arr,
                   s=40, alpha=0.55, zorder=5,
                   color=_C.get(agent, "#888"),
                   edgecolors="none", label=agent)

    ax.set_xticks(x)
    ax.set_xticklabels(agents, fontsize=9)
    ax.set_ylabel("Total Return (%)")
    ax.set_title("(B) Per-Run Returns")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # Legend — DQN highlighted
    handles, labels = ax.get_legend_handles_labels()
    dqn_idx = labels.index("DQN") if "DQN" in labels else 0
    ax.legend([handles[dqn_idx]] + [h for j, h in enumerate(handles) if j != dqn_idx],
              [labels[dqn_idx]] + [l for j, l in enumerate(labels) if j != dqn_idx],
              fontsize=8, framealpha=0.7, loc="upper left")

    # ────────────────────────────────────────────────────────────────
    # Panel C: Sharpe-like ratio violin
    # ────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])

    positions = list(range(len(agents)))
    parts = ax.violinplot(
        [sharpe_by_agent[a] for a in agents],
        positions=positions,
        showmeans=True, showmedians=True, showextrema=False,
    )
    for i, (body, agent) in enumerate(zip(parts["bodies"], agents)):
        body.set_facecolor(_C.get(agent, "#888"))
        body.set_alpha(0.35)

    parts["cmeans"].set_color(["#555"] * len(agents))
    parts["cmeans"].set_linestyle("--")
    parts["cmeans"].set_linewidth(1.2)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(2)
    for key in ["cbars", "cmins", "cmaxes"]:
        for line in parts.get(key, []):
            line.set_visible(False)

    ax.set_xticks(positions)
    ax.set_xticklabels(agents, fontsize=9)
    ax.set_ylabel("Sharpe-Like Ratio")
    ax.set_title("(C) Risk-Adjusted Return")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # ────────────────────────────────────────────────────────────────
    # Panel D: Statistical verdict table
    # ────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")

    # Build table rows
    row_lines: list[str] = []
    row_lines.append("Welch t-test (DQN vs each baseline)")
    row_lines.append("─" * 44)
    row_lines.append(f"{'':20s} {'t':>8s} {'p':>8s} {'d':>7s}  Verdict")
    row_lines.append("─" * 44)

    for name in ["Random", "BuyAndHold", "MeanReversion"]:
        st = stat_tests[name]
        detail = summary["h1_pass_details"][name]
        mark = "✓" if detail else "✗"
        t_str  = f"{st['t_stat']:+.3f}"
        p_str  = f"{st['p_value']:.4f}"
        d_str  = f"{st['cohens_d']:+.3f}"
        row_lines.append(f"  vs {name:17s} {t_str:>8s} {p_str:>8s} {d_str:>7s}   {mark}")

    row_lines.append("─" * 44)
    overall = "PASS" if summary["h1_pass"] else "FAIL"
    row_lines.append(f"  H1 overall:          {overall}")

    ax.text(0.02, 0.98, "\n".join(row_lines),
             transform=ax.transAxes, fontsize=9.5,
             va="top", ha="left", family="monospace",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="white",
                       edgecolor="grey", alpha=0.9))

    # Big verdict
    vcolor = "#2ca02c" if overall == "PASS" else "#d62728"
    ax.text(0.98, 0.05,
            f"H1: {overall}",
            transform=ax.transAxes, fontsize=18, fontweight="bold",
            ha="right", va="bottom", color=vcolor,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=vcolor,
                      edgecolor="none", alpha=0.15))

    # ── Save ───────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved H1 eval figure -> {output_path}")

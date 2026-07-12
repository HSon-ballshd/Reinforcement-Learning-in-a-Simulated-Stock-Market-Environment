"""
H2 Evaluation Scientific Report Figure

Generates a 2×2 panel figure from h2_eval results:

Panel layout
-------------
(A) Per-model accuracy vs random baseline  — horizontal bars with Wilson 95% CI
(B) Precision / Recall / F1 per regime class (stacked grouped bars)
(C) Class distribution in test set         — normalized bar chart
(D) Verdict text box                       — statistical significance summary

Usage (called automatically by h2_eval.py):
    from eval.h2_visualization import plot_h2_eval_summary
    plot_h2_eval_summary("outputs/h2_eval_summary_*.json",
                          "outputs/h2_eval_models_*.csv")
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
    "accent":  "#d62728",   # red  — for random baseline
    "selected":"#2ca02c",   # green — best/selected model
    "neutral": "#7f7f7f",   # grey  — other models
    "bars": [
        "#1f77b4",  # Stable
        "#ff7f0e",  # Bull
        "#9467bd",  # Bear
        "#17becf",  # Chaotic
    ],
}

# ── Helpers ────────────────────────────────────────────────────────
def _fmt_pct(x: float) -> str:
    """Compact % label: 0.31 → '31%', 0.3125 → '31.2%'."""
    return f"{x * 100:.1f}%"


class _PctFormatter(matplotlib.ticker.Formatter):
    """Matplotlib ticker: float → compact K/M or plain float."""
    def __call__(self, x, pos=None):
        if x >= 1e3:   return f"{x/1e3:.0f}K"
        if x >= 1e6:   return f"{x/1e6:.0f}M"
        return f"{x:.2f}"


# ── Main function ─────────────────────────────────────────────────
def plot_h2_eval_summary(
    summary_json: Path | str,
    models_csv: Path | str,
    output_path: Path | str | None = None,
) -> None:
    summary_json = Path(summary_json)
    models_csv   = Path(models_csv)
    if output_path is None:
        ts = summary_json.stem.replace("h2_eval_summary_", "")
        output_path = summary_json.parent / f"h2_eval_fig_{ts}.png"
    else:
        output_path = Path(output_path)

    with open(summary_json) as f:
        summary = json.load(f)

    cfg       = summary["config"]
    models    = summary["models"]          # dict[model_name] → metrics
    best_name = summary["best_model"]
    baseline  = cfg["baseline_random"]
    n_test    = cfg["n_test"]
    verdict   = "PASS" if summary["h2_pass"] else "FAIL"
    p_val     = summary.get("one_sided_p", float("nan"))
    class_dist = summary.get("class_distribution_test", {})
    n_classes  = cfg["n_classes"]
    regime_names = summary.get("regime_names", {})

    # Sort models: selected first, then by accuracy descending
    sorted_names = sorted(
        models.keys(),
        key=lambda n: (0 if n == best_name else 1, -models[n]["test_accuracy"])
    )

    # ── Figure ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 9))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)
    fig.suptitle(
        f"H2 Evaluation: Regime Classifier Accuracy  |  "
        f"{n_test} test samples  |  Verdict: {verdict}",
        fontsize=12, fontweight="bold", y=0.98,
    )

    # ────────────────────────────────────────────────────────────────
    # Panel A: Per-model accuracy bars with Wilson 95% CI
    # ────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])

    n_models = len(sorted_names)
    y_pos = np.arange(n_models)
    bar_h = 0.55

    for idx, name in enumerate(sorted_names):
        res     = models[name]
        acc     = res["test_accuracy"]
        ci_lo   = res["wilson_ci_lo"]
        ci_hi   = res["wilson_ci_hi"]
        is_best = (name == best_name)

        color = _C["selected"] if is_best else _C["neutral"]

        # Bar
        ax.barh(y_pos[idx], acc, height=bar_h, color=color,
                alpha=0.85, zorder=3)

        # CI error bars
        err_lo = acc - ci_lo
        err_hi = ci_hi - acc
        ax.errorbar(acc, y_pos[idx], xerr=[[err_lo], [err_hi]],
                    fmt="none", color="black", capsize=4, linewidth=1.2,
                    zorder=4)

        # Label inside bar (if wide enough) or to the right
        label = f"{acc:.1%}"
        if acc - ci_lo > 0.06:
            ax.text(acc - 0.005, y_pos[idx], label,
                    va="center", ha="right", fontsize=8.5, color="white",
                    fontweight="bold", zorder=5)
        else:
            ax.text(ci_hi + 0.005, y_pos[idx], label,
                    va="center", ha="left", fontsize=8.5,
                    color=color, fontweight="bold", zorder=5)

    # Random baseline — dashed vertical line
    ax.axvline(baseline, color=_C["accent"], linewidth=2.0,
               linestyle="--", zorder=2, label="Random (25%)")
    ax.text(baseline + 0.003, n_models - 0.6,
            f"Random\n{baseline:.1%}",
            va="top", ha="left", fontsize=8, color=_C["accent"])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xlabel("Test Accuracy")
    ax.set_title("(A) Per-Model Accuracy vs Random Baseline")
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis="x", alpha=0.4, zorder=0)
    # Legend marker
    ax.plot([], [], color=_C["selected"],  linewidth=2, label=f"Selected ({best_name})")
    ax.plot([], [], color=_C["neutral"],  linewidth=2, label="Other")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.7)

    # ────────────────────────────────────────────────────────────────
    # Panel B: Precision / Recall / F1 per class (stacked grouped)
    # ────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])

    # Keys are integers in-code but JSON saves them as strings — normalise
    raw_cm = models[best_name]["per_class"]
    class_metrics = {int(k): v for k, v in raw_cm.items()}
    classes  = list(range(n_classes))
    metrics  = ["precision", "recall", "f1"]
    metric_labels = ["Precision", "Recall", "F1-Score"]

    x = np.arange(n_classes)
    w = 0.25

    for mi, (m_label, m_key) in enumerate(zip(metric_labels, metrics)):
        vals = [class_metrics[c][m_key] for c in classes]
        bars = ax.bar(x + (mi - 1) * w, vals, w,
                      label=m_label, alpha=0.8, zorder=3)

    # Random baseline for each metric (≈ 1/4 = 0.25, dashed)
    ax.axhline(baseline, color=_C["accent"], linewidth=1.5,
               linestyle="--", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([regime_names.get(str(i), f"Class {i}") for i in classes],
                       fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title(f"(B) Per-Class Metrics — {best_name}")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(fontsize=8, loc="upper right", framealpha=0.7)
    ax.grid(axis="y", alpha=0.4, zorder=0)

    # ────────────────────────────────────────────────────────────────
    # Panel C: Class distribution in test set
    # ────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])

    labels   = [regime_names.get(str(i), f"Class {i}") for i in range(n_classes)]
    counts   = [class_dist.get(lbl, 0) for lbl in labels]
    total    = sum(counts)
    props    = [c / total for c in counts]
    bar_cols = _C["bars"]

    bars = ax.bar(labels, props, color=bar_cols, alpha=0.8, zorder=3)

    for bar, pct, cnt in zip(bars, props, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{pct:.1%}\n({cnt})",
                ha="center", va="bottom", fontsize=8)

    ax.axhline(baseline, color=_C["accent"], linewidth=1.5,
               linestyle="--", alpha=0.7, label=f"Random ({baseline:.1%})")
    ax.set_ylabel("Proportion")
    ax.set_title("(C) Test Set Class Distribution")
    ax.set_ylim(0, max(props) * 1.3)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis="y", alpha=0.4, zorder=0)

    # ────────────────────────────────────────────────────────────────
    # Panel D: Verdict + statistical summary
    # ────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")

    best_res = models[best_name]
    acc      = best_res["test_accuracy"]
    lift     = acc - baseline
    ci_lo    = best_res["wilson_ci_lo"]
    ci_hi    = best_res["wilson_ci_hi"]

    lines = [
        "Statistical Significance",
        "─" * 26,
        f"Model:        {best_name}",
        f"Test accuracy: {acc:.1%}",
        f"               [{ci_lo:.1%}, {ci_hi:.1%}]",
        f"Baseline:     {baseline:.1%} (random)",
        f"Lift:         +{lift:.1%}",
        "",
        "Binomial Test (H0: acc <= 25%):",
        f"  n_correct = {summary['n_correct']}",
        f"  n_test    = {summary['n_test']}",
        f"  p-value   = {p_val:.2e}",
        "",
        f"Verdict:  {verdict}",
        f"{'Classifier significantly beats random' if verdict == 'PASS' else 'No significant improvement over random'}.",
    ]

    # Color verdict line
    ax.text(0.05, 0.95, "\n".join(lines[:-2]),
            transform=ax.transAxes, fontsize=9.5,
            va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="white",
                      edgecolor="grey", alpha=0.9))

    # Big verdict label
    vcolor = "#2ca02c" if verdict == "PASS" else "#d62728"
    ax.text(0.95, 0.05,
            f"H2: {verdict}",
            transform=ax.transAxes, fontsize=18, fontweight="bold",
            ha="right", va="bottom", color=vcolor,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=vcolor,
                      edgecolor="none", alpha=0.15))

    # ── Save ───────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved H2 eval figure -> {output_path}")

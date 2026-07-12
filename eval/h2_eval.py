"""
H2 Evaluation: Regime Classifier Accuracy vs Random Baseline

Hypothesis H2: Regime classifier accuracy > 25% (random guess for 4 classes).

This script:
  1. Loads all trained models + scaler from models/
  2. Loads the regime dataset and performs a fresh train/val/test split
     (same random_state=42 as original pipeline to match the splits)
  3. Computes per-model accuracy, confidence intervals, per-class metrics
  4. Saves outputs + generates a scientific-report figure

Usage:
    python -m eval.h2_eval
"""

from __future__ import annotations

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.gridspec import GridSpec

# Project imports
from eval.classifiers.regime_classifier import (
    RegimeClassifierPipeline,
    N_CLASSES,
    REGIME_NAMES,
)
from eval.h2_visualization import plot_h2_eval_summary


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
OUTPUT_DIR  = Path("outputs")
MODEL_DIR   = Path("models")
DATA_PATH   = Path("data/regime_dataset.parquet")
ts          = datetime.now().strftime("%Y%m%d_%H%M%S")


# ------------------------------------------------------------------
# Bootstrap Wilson score CI
# ------------------------------------------------------------------
def wilson_ci(n_correct: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for accuracy."""
    p = n_correct / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n_total)) / n_total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def bootstrap_ci(arr: np.ndarray, stat: callable = np.mean,
                 n_bootstrap: int = 10_000, ci: float = 0.95) -> tuple[float, float]:
    """Bootstrap percentile CI for a statistic."""
    rng = np.random.default_rng(42)
    vals = [stat(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_bootstrap)]
    lo  = np.percentile(vals, (1 - ci) / 2 * 100)
    hi  = np.percentile(vals, (ci + (1 - ci) / 2) * 100)
    return float(lo), float(hi)


def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray
                               ) -> dict[int, dict]:
    """Precision / recall / F1 per regime class."""
    from sklearn.metrics import precision_recall_fscore_support
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(REGIME_NAMES), average=None, zero_division=0
    )
    return {
        i: {"precision": float(prec[i]),
            "recall":    float(rec[i]),
            "f1":        float(f1[i])}
        for i in range(N_CLASSES)
    }


# ------------------------------------------------------------------
# Load models and dataset
# ------------------------------------------------------------------
def load_models() -> tuple[object, dict, object]:
    """Load best_model, all_models, and scaler from MODEL_DIR."""
    with open(MODEL_DIR / "best_model.pkl",  "rb") as f:
        best_model = pickle.load(f)
    with open(MODEL_DIR / "all_models.pkl",  "rb") as f:
        all_models = pickle.load(f)
    with open(MODEL_DIR / "scaler.pkl",     "rb") as f:
        scaler = pickle.load(f)
    return best_model, all_models, scaler


def load_dataset() -> pd.DataFrame:
    """Load regime dataset, drop NaN rows."""
    df = pd.read_parquet(DATA_PATH)
    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped}/{before} NaN rows.")
    return df


# ------------------------------------------------------------------
# Main evaluation
# ------------------------------------------------------------------
def run_h2_eval() -> dict:
    print("\n=== H2 Evaluation: Regime Classifier Accuracy ===")

    # ── Load data ────────────────────────────────────────────────
    df = load_dataset()
    X = df[RegimeClassifierPipeline.FEATURE_COLS].values
    y = df["macro_regime"].values

    print(f"  Dataset: {len(df)} samples")

    # ── Same split as original pipeline (random_state=42) ───────
    # 60% train+val, 20% test
    from sklearn.model_selection import train_test_split
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # 60% of remaining → train (75% of total), 20% of remaining → val (25% of total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, random_state=42, stratify=y_tv
    )
    print(f"  Train: {len(y_train)}  Val: {len(y_val)}  Test: {len(y_test)}")

    # ── Load trained models ──────────────────────────────────────
    best_model, all_models, scaler = load_models()
    model_names = list(all_models.keys())

    # Re-fit scaler on this split's train set (same as original)
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # ── Per-model metrics on held-out TEST set ──────────────────
    baseline_random = 1.0 / N_CLASSES
    model_results   = {}

    for name, model in all_models.items():
        y_pred_test = model.predict(X_test_s)

        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_test, y_pred_test)

        # Wilson 95% CI
        n_correct = int(np.sum(y_pred_test == y_test))
        ci_lo, ci_hi = wilson_ci(n_correct, len(y_test))

        # Bootstrap mean CI for comparison
        preds = (y_pred_test == y_test).astype(float)
        bci_lo, bci_hi = bootstrap_ci(preds)

        # Per-class metrics
        per_class = compute_per_class_metrics(y_test, y_pred_test)

        model_results[name] = {
            "test_accuracy":   float(acc),
            "wilson_ci_lo":    ci_lo,
            "wilson_ci_hi":    ci_hi,
            "bootstrap_ci_lo": bci_lo,
            "bootstrap_ci_hi": bci_hi,
            "n_correct":       n_correct,
            "n_test":          len(y_test),
            "per_class":       per_class,
        }

    # Identify best model (same criterion as original: val accuracy)
    val_results = {}
    for name, model in all_models.items():
        y_pred_val = model.predict(X_val_s)
        from sklearn.metrics import accuracy_score
        val_results[name] = float(accuracy_score(y_val, y_pred_val))

    best_name      = max(val_results, key=val_results.get)
    best_test_acc  = model_results[best_name]["test_accuracy"]

    print(f"\n  Baseline (random): {baseline_random:.4f}")
    print(f"  Best model (by val): {best_name} = {best_test_acc:.4f}")
    for name, res in model_results.items():
        marker = " <- SELECTED" if name == best_name else ""
        print(f"    {name}: {res['test_accuracy']:.4f}  "
              f"95% CI [{res['wilson_ci_lo']:.4f}, {res['wilson_ci_hi']:.4f}]{marker}")

    # ── Statistical significance: best vs random ─────────────────
    from scipy.stats import binomtest
    n_correct = model_results[best_name]["n_correct"]
    n_total   = model_results[best_name]["n_test"]
    binom_res = binomtest(n_correct, n_total, p=baseline_random, alternative="greater")
    one_sided_p = binom_res.pvalue
    h2_pass = one_sided_p < 0.05 and best_test_acc > baseline_random

    print(f"\n  Binomial test (H0: accuracy <= {baseline_random:.4f}):")
    print(f"    n_correct={n_correct}, n_test={n_total}")
    print(f"    p = {one_sided_p:.2e}")
    print(f"    H2 verdict: {'PASS' if h2_pass else 'FAIL'} (acc > 25%, p < 0.05)")

    # ── Class distribution in test set ─────────────────────────
    class_counts = {}
    for i in range(N_CLASSES):
        class_counts[REGIME_NAMES[i]] = int(np.sum(y_test == i))
    total_test = len(y_test)

    # ── Build summary dict ──────────────────────────────────────
    summary = {
        "timestamp": ts,
        "config": {
            "n_total":      int(len(df)),
            "n_train":      int(len(y_train)),
            "n_val":        int(len(y_val)),
            "n_test":       int(len(y_test)),
            "n_classes":    N_CLASSES,
            "baseline_random": baseline_random,
        },
        "models":      model_results,
        "val_scores":  val_results,
        "best_model":  best_name,
        "best_test_accuracy": best_test_acc,
        "h2_pass":     h2_pass,
        "one_sided_p": float(one_sided_p),
        "n_correct":   n_correct,
        "n_test":      n_total,
        "class_distribution_test": class_counts,
        "regime_names": {str(k): v for k, v in REGIME_NAMES.items()},
    }

    # ── Save JSON ────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(exist_ok=True)
    json_path = OUTPUT_DIR / f"h2_eval_summary_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {json_path}")

    # ── Save per-model CSV ───────────────────────────────────────
    csv_path = OUTPUT_DIR / f"h2_eval_models_{ts}.csv"
    import csv as _csv
    with open(csv_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["model", "test_accuracy", "wilson_ci_lo", "wilson_ci_hi",
                    "n_correct", "n_test", "is_selected"])
        for name, res in model_results.items():
            w.writerow([
                name,
                f"{res['test_accuracy']:.6f}",
                f"{res['wilson_ci_lo']:.6f}",
                f"{res['wilson_ci_hi']:.6f}",
                res["n_correct"],
                res["n_test"],
                name == best_name,
            ])
    print(f"  Saved: {csv_path}")

    # ── Generate figure ──────────────────────────────────────────
    fig_path = OUTPUT_DIR / f"h2_eval_fig_{ts}.png"
    plot_h2_eval_summary(
        summary_json=json_path,
        models_csv=csv_path,
        output_path=fig_path,
    )
    print(f"  Saved: {fig_path}")

    # ── Console summary ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("H2 EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Dataset:    {len(df):,} samples  ({len(y_train):,} train / "
          f"{len(y_val):,} val / {len(y_test):,} test)")
    print(f"  Baseline:   {baseline_random:.1%} (random guess)")
    print(f"  Best model: {best_name} = {best_test_acc:.1%}  "
          f"[{model_results[best_name]['wilson_ci_lo']:.1%}, "
          f"{model_results[best_name]['wilson_ci_hi']:.1%}]")
    print(f"  Lift:       +{(best_test_acc - baseline_random):.1%} over random")
    print(f"  Binom p:    {one_sided_p:.2e}")
    print(f"  H2 verdict: {'PASS' if h2_pass else 'FAIL'}  "
          f"(accuracy > 25% AND p < 0.05)")
    print("=" * 60)

    return summary


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    run_h2_eval()

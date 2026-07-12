"""
H1 Evaluation: DQN vs Baselines — Return Comparison

Hypothesis H1: DQN beats Random and Buy-and-Hold on total return.

Uses the SAME trained DQN models already saved in models/
and the SAME train/eval seed split as eval_harness:
  - Train seeds: [42, 123, 456]
  - Eval seeds:  [789, 1024, 2048, 4096, 8192]

Runs all 4 agents (DQN, Random, BuyAndHold, MeanReversion) on every
eval seed, then saves outputs + generates a scientific-report figure.

Usage:
    python -m eval.h1_eval
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.gridspec import GridSpec

# Project imports
from eval.agents.dqn import DQNAgent
from eval.agents.baselines import (
    RandomAgent,
    BuyAndHoldAgent,
    MeanReversionAgent,
)
from eval.h1_visualization import plot_h1_eval_summary


# ------------------------------------------------------------------
# Config — mirrors eval_harness train/eval split
# ------------------------------------------------------------------
TRAIN_SEEDS    = [42, 123, 456]
EVAL_SEEDS    = [789, 1024, 2048, 4096, 8192]
EPISODE_STEPS = 500
INITIAL_CASH  = 10_000.0
MODEL_DIR     = Path("models")
OUTPUT_DIR    = Path("outputs")
ts            = datetime.now().strftime("%Y%m%d_%H%M%S")


# ------------------------------------------------------------------
# Per-run metrics
# ------------------------------------------------------------------
@dataclass
class RunMetrics:
    agent_type: str
    train_seed: int
    eval_seed: int
    total_return_pct: float
    final_value: float
    sharpe_like: float
    max_drawdown_pct: float
    n_buys: int
    n_sells: int
    n_holds: int


def _sharpe_like(returns: list[float]) -> float:
    """Annualised Sharpe-like ratio (annualisation = sqrt(252) omitted — single episode)."""
    if len(returns) < 2:
        return 0.0
    r = np.array(returns)
    return float(np.mean(r) / (np.std(r) + 1e-12))


def _max_drawdown(pv_series: list[float]) -> float:
    """Maximum drawdown as a positive percentage."""
    if not pv_series:
        return 0.0
    peak = pv_series[0]
    worst = 0.0
    for v in pv_series:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > worst:
            worst = dd
    return float(worst * 100)


def _eval_episode(
    agent,
    eval_seed: int,
    agent_type: str,
    train_seed: int | None = None,
) -> RunMetrics:
    """Run one episode and collect all metrics."""
    from sim.market_sim import CookieClickerMarket
    from eval.env.trading_env import TradingEnv

    market = CookieClickerMarket(n_stocks=1, seed=eval_seed)
    env    = TradingEnv(
        market,
        initial_cash=INITIAL_CASH,
        max_steps=EPISODE_STEPS,
        seed=eval_seed,
    )

    obs   = env.reset()
    agent.reset()
    if hasattr(agent, "_env") or hasattr(agent, "set_env"):
        agent._env = env

    n_buys = n_sells = n_holds = 0
    portfolio_vals = [env._portfolio_value()]
    step_rewards  = []

    for _ in range(EPISODE_STEPS):
        action = agent.select_action(obs, {"portfolio_value": portfolio_vals[-1]})
        obs, reward, done, info = env.step(action)
        portfolio_vals.append(env._portfolio_value())
        step_rewards.append(reward)

        if action == 0:   n_holds += 1
        elif action == 1: n_buys  += 1
        elif action == 2: n_sells += 1

        if done:
            break

    final_val     = portfolio_vals[-1]
    total_return  = (final_val - INITIAL_CASH) / INITIAL_CASH * 100
    mdd           = _max_drawdown(portfolio_vals)
    sharpe        = _sharpe_like(step_rewards)

    return RunMetrics(
        agent_type=agent_type,
        train_seed=train_seed if train_seed is not None else 0,
        eval_seed=eval_seed,
        total_return_pct=float(total_return),
        final_value=float(final_val),
        sharpe_like=sharpe,
        max_drawdown_pct=mdd,
        n_buys=n_buys, n_sells=n_sells, n_holds=n_holds,
    )


# ------------------------------------------------------------------
# Statistical helpers
# ------------------------------------------------------------------
def _welch_t(a: np.ndarray, b: np.ndarray
             ) -> tuple[float, float, float, int, int]:
    """Welch's t-test: returns (t_stat, p_value, cohens_d, n1, n2)."""
    from scipy.stats import ttest_ind
    t, p = ttest_ind(a, b, equal_var=False)
    pooled_std = np.sqrt((np.var(a) + np.var(b)) / 2)
    d = (np.mean(a) - np.mean(b)) / (pooled_std + 1e-12)
    return float(t), float(p), float(d), len(a), len(b)


def _bootstrap_ci(a: np.ndarray, b: np.ndarray,
                  n_bootstrap: int = 10_000, ci: float = 0.95
                  ) -> tuple[float, float]:
    """Bootstrap 95% CI of mean(b) - mean(a)."""
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(n_bootstrap):
        ma = rng.choice(a, size=len(a), replace=True)
        mb = rng.choice(b, size=len(b), replace=True)
        diffs.append(np.mean(mb) - np.mean(ma))
    lo = np.percentile(diffs, (1 - ci) / 2 * 100)
    hi = np.percentile(diffs, (ci + (1 - ci) / 2) * 100)
    return float(lo), float(hi)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def run_h1_eval() -> dict:
    print("\n=== H1 Evaluation: DQN vs Baselines ===")
    print(f"  Train seeds: {TRAIN_SEEDS}")
    print(f"  Eval seeds:  {EVAL_SEEDS}")

    all_runs: list[dict] = []
    agent_returns: dict[str, list[float]] = {
        "DQN": [], "Random": [], "BuyAndHold": [], "MeanReversion": [],
    }

    # ── DQN: 3 train seeds × 5 eval seeds = 15 runs ──────────────
    for train_s in TRAIN_SEEDS:
        ckpt = MODEL_DIR / f"dqn_agent_{train_s}.pkl"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Trained DQN not found: {ckpt}\n"
                "Run eval_harness first to train DQN agents."
            )
        dqn = DQNAgent.load(ckpt)
        print(f"  Loaded DQN (train_seed={train_s})")

        for eval_s in EVAL_SEEDS:
            m = _eval_episode(dqn, eval_s, "DQN", train_s)
            all_runs.append(asdict(m))
            agent_returns["DQN"].append(m.total_return_pct)

    print(f"  DQN: {len(agent_returns['DQN'])} runs, "
          f"mean={np.mean(agent_returns['DQN']):.2f}%")

    # ── Baselines: 5 eval seeds each = 5 runs ─────────────────────
    baseline_configs = [
        ("Random",      lambda: RandomAgent(seed=42)),
        ("BuyAndHold",  lambda: BuyAndHoldAgent()),
        ("MeanReversion", lambda: MeanReversionAgent()),
    ]

    for name, factory in baseline_configs:
        for eval_s in EVAL_SEEDS:
            agent = factory()
            if name == "Random":
                agent = RandomAgent(seed=eval_s)
            m = _eval_episode(agent, eval_s, name)
            all_runs.append(asdict(m))
            agent_returns[name].append(m.total_return_pct)
        print(f"  {name}: {len(agent_returns[name])} runs, "
              f"mean={np.mean(agent_returns[name]):.2f}%")

    # ── Aggregate ───────────────────────────────────────────────────
    agg: dict[str, dict] = {}
    for name, rets in agent_returns.items():
        arr = np.array(rets)
        agg[name] = {
            "mean":     float(np.mean(arr)),
            "std":      float(np.std(arr)),
            "median":   float(np.median(arr)),
            "min":      float(np.min(arr)),
            "max":      float(np.max(arr)),
            "n_runs":   len(rets),
        }

    print(f"\n  Aggregate results:")
    for name, a in agg.items():
        print(f"    {name:15s}: {a['mean']:>12.2f}% ± {a['std']:>10.2f}%")

    # ── Statistical tests: DQN vs each baseline ─────────────────────
    dqn_arr = np.array(agent_returns["DQN"])
    stat_tests: dict[str, dict] = {}
    h1_pass_details: dict[str, bool] = {}

    for name in ["Random", "BuyAndHold", "MeanReversion"]:
        other = np.array(agent_returns[name])
        t_stat, p_val, d_val, n1, n2 = _welch_t(dqn_arr, other)
        ci_lo, ci_hi = _bootstrap_ci(dqn_arr, other)
        stat_tests[name] = {
            "t_stat":   t_stat, "p_value": p_val,
            "cohens_d": d_val, "n1": n1, "n2": n2,
            "ci_lower": ci_lo, "ci_upper": ci_hi,
        }
        # One-sided: DQN mean > baseline mean AND p/2 < 0.05 AND t > 0
        one_sided_p = p_val / 2 if p_val == p_val else float("nan")
        h1_pass_details[name] = (
            bool(dqn_arr.size) and bool(other.size)
            and np.mean(dqn_arr) > np.mean(other)
            and one_sided_p < 0.05
            and t_stat > 0
        )

    # H1 overall: DQN beats BOTH Random AND BuyAndHold significantly
    h1_pass = h1_pass_details["Random"] and h1_pass_details["BuyAndHold"]

    print(f"\n  Statistical tests (DQN vs baselines):")
    for name, st in stat_tests.items():
        verdict = "PASS" if h1_pass_details[name] else "FAIL"
        print(f"    vs {name:15s}: t={st['t_stat']:+.3f}, p={st['p_value']:.4f}, "
              f"d={st['cohens_d']:+.3f}  [{verdict}]")

    # ── Build summary ──────────────────────────────────────────────
    summary = {
        "timestamp": ts,
        "config": {
            "train_seeds":    TRAIN_SEEDS,
            "eval_seeds":     EVAL_SEEDS,
            "n_train_seeds":  len(TRAIN_SEEDS),
            "n_eval_seeds":   len(EVAL_SEEDS),
            "n_dqn_runs":    len(agent_returns["DQN"]),
            "n_baseline_runs": len(EVAL_SEEDS),
            "episode_steps":  EPISODE_STEPS,
            "initial_cash":   INITIAL_CASH,
        },
        "aggregate": agg,
        "statistical_tests": stat_tests,
        "h1_pass":       h1_pass,
        "h1_pass_details": h1_pass_details,
    }

    # ── Save runs CSV ─────────────────────────────────────────────
    OUTPUT_DIR.mkdir(exist_ok=True)
    csv_path = OUTPUT_DIR / f"h1_eval_runs_{ts}.csv"
    import csv as _csv
    with open(csv_path, "w", newline="") as f:
        cols = list(all_runs[0].keys())   # already dicts — not dataclass instances
        w = _csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_runs)
    print(f"\n  Saved: {csv_path}")

    # ── Save summary JSON ─────────────────────────────────────────
    json_path = OUTPUT_DIR / f"h1_eval_summary_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {json_path}")

    # ── Generate figure ───────────────────────────────────────────
    fig_path = OUTPUT_DIR / f"h1_eval_fig_{ts}.png"
    plot_h1_eval_summary(
        summary_json=json_path,
        runs_csv=csv_path,
        output_path=fig_path,
    )
    print(f"  Saved: {fig_path}")

    # ── Console summary ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("H1 EVALUATION SUMMARY")
    print("=" * 65)
    print(f"  Train seeds : {TRAIN_SEEDS}")
    print(f"  Eval seeds  : {EVAL_SEEDS}")
    print(f"  Episode len : {EPISODE_STEPS} ticks  |  Initial cash: ${INITIAL_CASH:,.0f}")
    print()
    print(f"  {'Agent':15s}  {'Mean Return %':>15s}  {'Std':>10s}  {'Median':>10s}  {'Runs':>5s}")
    print(f"  {'-'*15}  {'-'*15}  {'-'*10}  {'-'*10}  {'-'*5}")
    for name, a in agg.items():
        marker = " *" if name == "DQN" else ""
        print(f"  {name:15s}  {a['mean']:>+15.2f}  {a['std']:>+10.2f}  "
              f"{a['median']:>+10.2f}  {a['n_runs']:>5d}{marker}")
    print()
    print(f"  Statistical significance (Welch t-test, DQN vs baselines):")
    for name, st in stat_tests.items():
        detail = h1_pass_details[name]
        sym = "✓" if detail else "✗"
        print(f"    vs {name:15s}: t={st['t_stat']:+.3f}, p={st['p_value']:.4f}, "
              f"d={st['cohens_d']:+.3f}  [{sym}]")
    print()
    print(f"  H1 verdict: {'PASS' if h1_pass else 'FAIL'}  "
          f"(DQN > Random AND DQN > BuyAndHold, both p < 0.05)")
    print("=" * 65)

    return summary


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    run_h1_eval()

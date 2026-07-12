"""
H3 — Regime-Aware DQN vs Plain DQN: Comprehensive Evaluation Script

Runs both DQN and RA-DQN agents across N eval seeds and records:
  - Per-run:   trades (buy/sell/hold count), final portfolio value, return %,
               Sharpe-like ratio, max drawdown, time-in-regime breakdown,
               price range (min/max), regime transitions.
  - Aggregate: mean, std, median, IQR, min, max for each metric.
  - Statistical: Welch t-test, Mann-Whitney U, effect size (Cohen's d),
                 95% bootstrap CI on mean return difference.

Outputs:
  outputs/h3_eval_runs_{ts}.csv   — one row per (agent, seed, eval_seed)
  outputs/h3_eval_summary_{ts}.json — aggregate stats + p-values
  outputs/h3_eval_table_{ts}.txt  — LaTeX-ready summary table

Usage:
    python -m eval.h3_eval
    python -m eval.h3_eval --eval-seeds 25 --retrain
    python -m eval.h3_eval --agents dqn          # only plain DQN
    python -m eval.h3_eval --agents ra            # only RA-DQN
    python -m eval.h3_eval --no-plot
"""

from __future__ import annotations

import json
import csv
import pickle
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field

import numpy as np
from scipy import stats
from tqdm import tqdm

# Project imports
from sim.market_sim import CookieClickerMarket
from eval.env.trading_env import TradingEnv
from eval.agents.dqn import DQNAgent, train_dqn
from eval.agents.dqn_regime import RegimeAwareDQNAgent
from eval.visualization import plot_h3_eval_summary

# ── Globals ────────────────────────────────────────────────────────────────────
INITIAL_CASH    = 10_000.0
EPISODE_STEPS   = 500
TX_COST_PCT     = 0.001
N_TRAIN_SEEDS   = 3
OUTPUT_DIR      = Path("outputs")
MODEL_DIR       = Path("models")
REGIME_NAMES    = ["Stable", "Bull", "Bear", "Chaotic"]


# ══════════════════════════════════════════════════════════════════════════════
# Per-run metrics
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RunMetrics:
    """All metrics collected from a single episode."""
    agent_type: str
    train_seed: int
    eval_seed: int
    total_return_pct: float
    final_value: float
    n_buys: int = 0
    n_sells: int = 0
    n_holds: int = 0
    n_trades: int = 0
    sharpe_like: float = 0.0
    max_drawdown_pct: float = 0.0
    min_portfolio_value: float = 0.0
    price_start: float = 0.0
    price_end: float = 0.0
    price_min: float = 0.0
    price_max: float = 0.0
    regime_time: list[int] = field(default_factory=list)
    regime_transitions: int = 0


def _js_mode_to_macro(mode: int) -> int:
    """Map 6-class JS mode to 4-class macro regime."""
    if mode == 0:          return 0   # Stable
    if mode in (1, 4):     return 1   # Bull / Strong Bull
    if mode in (2, 5):     return 2   # Bear / Strong Bear (5 is merged into Bear)
    return 3  # Chaotic (modes 3)


def _eval_episode(
    agent,
    eval_seed: int,
    agent_type: str,
    train_seed: int,
    steps: int = EPISODE_STEPS,
    initial_cash: float = INITIAL_CASH,
    record_regime: bool = False,
) -> RunMetrics:
    """Run one episode and collect all metrics."""
    market = CookieClickerMarket(n_stocks=1, seed=eval_seed)
    env    = TradingEnv(
        market,
        initial_cash=initial_cash,
        max_steps=steps,
        transaction_cost_pct=TX_COST_PCT,
        seed=eval_seed,
    )

    agent._env = env
    agent.reset()

    obs         = env.reset()
    price       = market.stocks[0]['price']
    price_start = price_end = price_min = price_max = price

    portfolio_values: list[float] = [env._portfolio_value()]
    step_rewards: list[float]     = []
    prev_regime: int | None        = None
    regime_time  = [0, 0, 0, 0]
    regime_transitions = 0
    n_buys = n_sells = n_holds = 0

    info = {
        'portfolio_value': env._portfolio_value(),
        'cash': env.cash,
        'holdings': env.holdings,
        'price': price,
        'step': 0,
    }

    for _ in range(steps):
        action = agent.select_action(obs, info)
        obs, reward, done, info = env.step(action)

        step_rewards.append(reward)
        pv = env._portfolio_value()
        portfolio_values.append(pv)

        price = market.stocks[0]['price']
        price_end   = price
        price_min   = min(price_min, price)
        price_max   = max(price_max, price)

        if action == TradingEnv.BUY:
            n_buys += 1
        elif action == TradingEnv.SELL:
            n_sells += 1
        else:
            n_holds += 1

        if record_regime:
            true_regime = market.stocks[0]['mode']
            macro = _js_mode_to_macro(true_regime)
            regime_time[macro] += 1
            if prev_regime is not None and true_regime != prev_regime:
                regime_transitions += 1
            prev_regime = true_regime

        if done:
            break

    final_value   = env._portfolio_value()
    total_return  = (final_value - initial_cash) / initial_cash * 100.0
    n_trades      = n_buys + n_sells

    if len(step_rewards) > 1 and np.std(step_rewards) > 1e-10:
        sharpe = (np.mean(step_rewards) / np.std(step_rewards)) * np.sqrt(len(step_rewards))
    else:
        sharpe = 0.0

    peak = -np.inf
    max_dd = 0.0
    min_pv  = initial_cash
    for pv_ in portfolio_values:
        if pv_ > peak:
            peak = pv_
        dd = (peak - pv_) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
        min_pv = min(min_pv, pv_)
    max_dd_pct = max_dd * 100.0

    return RunMetrics(
        agent_type=agent_type,
        train_seed=train_seed,
        eval_seed=eval_seed,
        total_return_pct=total_return,
        final_value=final_value,
        n_buys=n_buys,
        n_sells=n_sells,
        n_holds=n_holds,
        n_trades=n_trades,
        sharpe_like=sharpe,
        max_drawdown_pct=max_dd_pct,
        min_portfolio_value=min_pv,
        price_start=price_start,
        price_end=price_end,
        price_min=price_min,
        price_max=price_max,
        regime_time=regime_time,
        regime_transitions=regime_transitions,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Statistical helpers
# ══════════════════════════════════════════════════════════════════════════════

def _welch_t(a: list[float], b: list[float]) -> dict:
    if not a or not b or len(a) < 2 or len(b) < 2:
        return {"t_stat": float('nan'), "p_value": float('nan'),
                "cohens_d": float('nan'), "n1": len(a), "n2": len(b)}
    t, p = stats.ttest_ind(a, b, equal_var=False)
    n1, n2 = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt((va + vb) / 2)
    d = (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 1e-12 else float('nan')
    return {"t_stat": float(t), "p_value": float(p), "cohens_d": float(d),
            "n1": n1, "n2": n2}


def _mann_whitney(a: list[float], b: list[float]) -> dict:
    if not a or not b:
        return {"U": float('nan'), "p_value": float('nan'), "rank_biserial_r": float('nan')}
    u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    r = 1 - (2 * u) / (len(a) * len(b))
    return {"U": float(u), "p_value": float(p), "rank_biserial_r": float(r)}


def _bootstrap_ci(a: list[float], b: list[float], n_resamples: int = 10_000,
                   ci: float = 0.95) -> dict:
    if not a or not b:
        return {"mean_diff": float('nan'), "ci_lower": float('nan'),
                "ci_upper": float('nan'), "n_resamples": n_resamples, "ci_level": ci}
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    diffs = []
    for _ in range(n_resamples):
        ma = np.random.choice(a, size=len(a), replace=True)
        mb = np.random.choice(b, size=len(b), replace=True)
        diffs.append(np.mean(ma) - np.mean(mb))
    lo = (1 - ci) / 2
    hi = 1 - lo
    return {
        "mean_diff":  float(np.mean(diffs)),
        "ci_lower":   float(np.percentile(diffs, lo * 100)),
        "ci_upper":   float(np.percentile(diffs, hi * 100)),
        "n_resamples": n_resamples,
        "ci_level":   ci,
    }


def _summary_stats(values: list[float]) -> dict:
    a = np.array(values, dtype=float)
    return {
        "n":       len(a),
        "mean":    float(np.mean(a)),
        "std":     float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "median":  float(np.median(a)),
        "q1":      float(np.percentile(a, 25)),
        "q3":      float(np.percentile(a, 75)),
        "iqr":     float(np.percentile(a, 75) - np.percentile(a, 25)),
        "min":     float(np.min(a)),
        "max":     float(np.max(a)),
        "se":      float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0,
    }


def _aggregate_runs(runs: list[RunMetrics]) -> dict:
    if not runs:
        return {}
    out: dict = {}
    for field_ in [
        "total_return_pct", "final_value", "n_buys", "n_sells",
        "n_holds", "n_trades", "sharpe_like", "max_drawdown_pct",
        "min_portfolio_value",
    ]:
        vals = [getattr(r, field_) for r in runs]
        out[field_] = _summary_stats(vals)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Classifier closure builder
# ══════════════════════════════════════════════════════════════════════════════

def make_classify_fn(agent, clf, scaler):
    def classify(_obs) -> int:
        env   = agent._env
        stock = env.market.stocks[0]
        vals  = stock['vals']
        price = stock['price']

        tick_rets = [
            (vals[i] - vals[i+1]) / (vals[i+1] + 1e-8)
            for i in range(min(len(vals) - 1, 64))
        ]
        def trets(n): return tick_rets[:n]

        ret_1  = tick_rets[0] if len(tick_rets) >= 1 else 0.0
        ret_5  = (vals[0] - vals[5]) / vals[5] if len(vals) > 5 else 0.0
        ret_10 = (vals[0] - vals[10]) / vals[10] if len(vals) > 10 else 0.0
        ret_20 = (vals[0] - vals[20]) / vals[20] if len(vals) > 20 else 0.0

        rstd_5  = float(np.std(trets(5)))  if len(tick_rets) >= 5  else 0.0
        rstd_20 = float(np.std(trets(20))) if len(tick_rets) >= 20 else 0.0
        rstd_ratio = rstd_5 / (rstd_20 + 1e-8)

        mean_20 = float(np.mean(vals[:20])) if len(vals) >= 20 else float(np.mean(vals))
        mean_rev_z = (price - mean_20) / (rstd_20 * mean_20 + 1e-8) if rstd_20 > 1e-8 else 0.0

        def dir_cons(n):
            t = trets(n)
            return float(sum(1 for r in t if r > 0) / len(t)) if t else 0.5
        dir_5  = dir_cons(5)
        dir_20 = dir_cons(20)

        t5 = trets(5)
        drift_est_5 = float(np.mean([abs(r) * (1 if r > 0 else -1) for r in t5])) if t5 else 0.0

        def jump_count(n, rstd):
            return float(sum(1 for r in trets(n) if abs(r) > rstd)) if rstd > 1e-8 else 0.0
        jc_5  = jump_count(5,  rstd_5)
        jc_20 = jump_count(20, rstd_20)

        max_ret_5 = float(max((abs(r) for r in trets(5)), default=0.0))

        def trend_str(ret, rstd):
            return float(ret / (rstd + 1e-8)) if rstd > 1e-8 else 0.0
        trend_5  = trend_str(ret_5,  rstd_5)
        trend_20 = trend_str(ret_20, rstd_20)

        mom_div = 1.0 if ret_5 * ret_20 < 0 else 0.0
        vol_reg  = rstd_5 / (rstd_20 + 1e-8)

        base = np.array([
            ret_1, ret_5, ret_10, ret_20,
            rstd_5, rstd_20, rstd_ratio, mean_rev_z,
            dir_5, dir_20, drift_est_5,
            jc_5, jc_20, max_ret_5,
            trend_5, trend_20, mom_div, vol_reg,
        ], dtype=np.float32)

        x = scaler.transform(base.reshape(1, -1))
        return int(clf.predict(x)[0])
    return classify


# ══════════════════════════════════════════════════════════════════════════════
# RA-DQN training loop (cleaned up from eval_harness.py)
# ══════════════════════════════════════════════════════════════════════════════

def _train_ra(
    agent: RegimeAwareDQNAgent,
    market_seed: int,
    n_steps: int,
    eval_every: int,
    max_episode_steps: int | None,
    eval_steps: int,
    initial_cash: float,
    train_seeds: list[int] | None = None,
    verbose: bool = False,
    log_path: Path | None = None,
) -> dict:
    logs: dict = {"loss": [], "epsilon": [],
                  "episode_return": [], "eval_returns": []}
    best_eval = -np.inf

    market = CookieClickerMarket(n_stocks=1, seed=market_seed)
    env    = TradingEnv(market, initial_cash=initial_cash,
                        max_steps=max_episode_steps, seed=market_seed)
    agent.set_env(env)

    episode_return = 0.0
    agent.reset()
    obs = env.reset()

    eval_seed_pool = [42, 123, 456, 789, 1024]
    if train_seeds:
        eval_seed_pool = [s for s in eval_seed_pool if s not in train_seeds]

    csv_file  = None
    csv_writer = None
    if log_path:
        csv_file   = open(log_path, "w", newline="")
        csv_writer = csv.DictWriter(csv_file,
            fieldnames=["step", "loss", "epsilon", "episode_return", "eval_return_pct"])
        csv_writer.writeheader()

    iterator = tqdm(range(n_steps), desc="RA-DQN", unit="step",
                     disable=not verbose)

    try:
        for step in iterator:
            action      = agent._epsilon_greedy(obs, training=True)
            next_obs, reward, done, _ = env.step(action)
            episode_return += reward
            agent.store(obs, action, reward, next_obs, done)
            obs = next_obs

            loss = agent.train_step()
            if loss is not None:
                logs["loss"].append(float(loss))
            logs["epsilon"].append(agent.epsilon)

            if done:
                logs["episode_return"].append(float(episode_return))
                episode_return = 0.0
                env.reset()
                agent.reset()
                obs = env.reset()

            if (step + 1) % eval_every == 0:
                ev_returns = []
                for s in eval_seed_pool[:3]:
                    m = CookieClickerMarket(n_stocks=1, seed=s)
                    e = TradingEnv(m, initial_cash=initial_cash,
                                   max_steps=eval_steps, seed=s)
                    agent.reset()
                    o = e.reset()
                    info_d = {'portfolio_value': initial_cash, 'cash': initial_cash,
                              'holdings': 0.0, 'price': m.stocks[0]['price'], 'step': 0}
                    dn = False
                    while not dn:
                        a = agent.select_action(o, info_d)
                        o, _, dn, info_d = e.step(a)
                    final_val = e._portfolio_value()
                    ev_returns.append((final_val - initial_cash) / initial_cash * 100.0)
                mean_ev = float(np.mean(ev_returns))
                logs["eval_returns"].append(mean_ev)
                if mean_ev > best_eval:
                    best_eval = mean_ev
                iterator.set_postfix(epsilon=f"{agent.epsilon:.3f}",
                                     eval_ret=f"{mean_ev:.2f}%")

                if csv_writer is not None:
                    last_loss = logs["loss"][-1] if logs["loss"] else ""
                    last_ep   = logs["episode_return"][-1] if logs["episode_return"] else ""
                    csv_writer.writerow({
                        "step": step + 1, "loss": last_loss,
                        "epsilon": agent.epsilon,
                        "episode_return": last_ep,
                        "eval_return_pct": mean_ev,
                    })
                    csv_file.flush()
    finally:
        if csv_file is not None:
            csv_file.close()

    return {"logs": logs, "best_eval": float(best_eval)}


# ══════════════════════════════════════════════════════════════════════════════
# Main evaluation
# ══════════════════════════════════════════════════════════════════════════════

def run_h3_eval(
    eval_seeds: list[int] | None = None,
    train_seeds: list[int] | None = None,
    agents: list[str] | None = None,
    retrain: bool = False,
    n_steps: int = 20_000,
    verbose: bool = True,
) -> dict:
    # NOTE: deliberately disjoint from eval_harness.py's train=[42,123,456] and eval=[789,1024,2048,4096,8192]
    # Also disjoint from the RA-DQN eval_seed_pool=[42,123,456,789,1024] used during training.
    if eval_seeds is None:
        eval_seeds = [
            # Primes (none overlap with eval_harness seeds)
            11, 13, 17, 19, 23, 29, 31, 37, 41, 43,
            47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
            # Larger non-overlapping integers
            5000, 6000, 7000, 8000, 9000,
        ]

    if train_seeds is None:
        train_seeds = [42, 123, 456]

    if agents is None:
        agents = ["dqn", "ra"]

    n_eval  = len(eval_seeds)
    n_train = len(train_seeds)

    # Load classifier
    clf_path   = MODEL_DIR / "best_model.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"
    if not clf_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            "Classifier not found. Run H2 first: python -m eval.eval_harness --h2"
        )
    clf    = pickle.load(open(clf_path,   "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))

    # Load / train agents
    dqn_agents: list[DQNAgent] = []
    ra_agents:  list[RegimeAwareDQNAgent] = []

    for seed in train_seeds:
        if "dqn" in agents:
            dqn_ckpt = MODEL_DIR / f"dqn_agent_{seed}.pkl"
            if dqn_ckpt.exists() and not retrain:
                dqn = DQNAgent.load(dqn_ckpt)
                print(f"  [DQN seed={seed}] loaded from {dqn_ckpt}")
            else:
                print(f"  [DQN seed={seed}] training {n_steps} steps...")
                dqn = DQNAgent(obs_dim=8, n_actions=3, min_replay_size=500,
                               batch_size=32, seed=seed)
                train_dqn(
                    dqn, market_seed=seed, n_steps=n_steps,
                    eval_every=5_000, max_episode_steps=EPISODE_STEPS,
                    eval_steps=500, initial_cash=INITIAL_CASH,
                    train_seeds=train_seeds, verbose=True,
                    log_path=OUTPUT_DIR / f"h3_eval_dqn_seed{seed}_log.csv",
                )
                dqn.save(dqn_ckpt)
                print(f"  [DQN seed={seed}] done, saved to {dqn_ckpt}")
            dqn_agents.append(dqn)

        if "ra" in agents:
            ra_ckpt = MODEL_DIR / f"ra_dqn_agent_{seed}.pkl"
            if ra_ckpt.exists() and not retrain:
                ra = RegimeAwareDQNAgent.load(ra_ckpt)
                ra.set_classifier(make_classify_fn(ra, clf, scaler))
                print(f"  [RA-DQN seed={seed}] loaded from {ra_ckpt}")
            else:
                print(f"  [RA-DQN seed={seed}] training {n_steps} steps...")
                ra = RegimeAwareDQNAgent(obs_dim=8, n_actions=3, n_regimes=4,
                                         min_replay_size=500, batch_size=32, seed=seed)
                ra.set_classifier(make_classify_fn(ra, clf, scaler))
                _train_ra(
                    ra, market_seed=seed, n_steps=n_steps,
                    eval_every=5_000, max_episode_steps=EPISODE_STEPS,
                    eval_steps=500, initial_cash=INITIAL_CASH,
                    train_seeds=train_seeds, verbose=True,
                    log_path=OUTPUT_DIR / f"h3_eval_ra_seed{seed}_log.csv",
                )
                ra.save(ra_ckpt)
                print(f"  [RA-DQN seed={seed}] done, saved to {ra_ckpt}")
            ra_agents.append(ra)

    # Run episodes
    all_runs: list[dict] = []
    total_steps = (len(dqn_agents) + len(ra_agents)) * n_eval
    pbar = tqdm(total=total_steps,
                desc=f"H3 Eval ({n_eval} seeds x {len(dqn_agents + ra_agents)} agents)",
                disable=not verbose)

    for idx, seed in enumerate(train_seeds):
        if "dqn" in agents:
            dqn = dqn_agents[idx]
            for eval_s in eval_seeds:
                m = _eval_episode(dqn, eval_s, "DQN", seed)
                all_runs.append(asdict(m))
                pbar.update(1)

        if "ra" in agents:
            ra = ra_agents[idx]
            for eval_s in eval_seeds:
                m = _eval_episode(ra, eval_s, "RA-DQN", seed, record_regime=True)
                all_runs.append(asdict(m))
                pbar.update(1)

    pbar.close()

    dqn_runs = [RunMetrics(**r) for r in all_runs if r["agent_type"] == "DQN"]
    ra_runs  = [RunMetrics(**r) for r in all_runs if r["agent_type"] == "RA-DQN"]

    dqn_agg = _aggregate_runs(dqn_runs)
    ra_agg  = _aggregate_runs(ra_runs)

    dqn_returns = [r.total_return_pct for r in dqn_runs]
    ra_returns  = [r.total_return_pct for r in ra_runs]

    stats_tests: dict = {}
    if dqn_returns and ra_returns:
        stats_tests["return_pct"] = {
            "welch_t":       _welch_t(dqn_returns, ra_returns),
            "mann_whitney":  _mann_whitney(dqn_returns, ra_returns),
            "bootstrap_ci":   _bootstrap_ci(dqn_returns, ra_returns),
        }

    # Sharpe comparison
    dqn_has_sharpe = bool(dqn_agg) and "sharpe_like" in dqn_agg
    ra_has_sharpe  = bool(ra_agg)  and "sharpe_like" in ra_agg
    if dqn_has_sharpe and dqn_agg["sharpe_like"]["std"] > 0 \
       and ra_has_sharpe and ra_agg["sharpe_like"]["std"] > 0:
        stats_tests["sharpe_like"] = {
            "welch_t": _welch_t(
                [r.sharpe_like for r in dqn_runs],
                [r.sharpe_like for r in ra_runs],
            )
        }

    # Regime exposure (RA-DQN ground-truth)
    regime_exposure: dict | None = None
    if ra_runs:
        total_ticks = sum(sum(r.regime_time) for r in ra_runs)
        if total_ticks > 0:
            regime_exposure = {
                name: float(sum(r.regime_time[i] for r in ra_runs) / total_ticks)
                for i, name in enumerate(REGIME_NAMES)
            }

    # Require both: RA-DQN mean > DQN mean AND p < 0.05 (one-sided, RA better)
    wt = stats_tests.get("return_pct", {}).get("welch_t", {})
    p_val = wt.get("p_value", float('nan'))
    # Welch's t is two-sided; for one-sided RA > DQN, divide p by 2 and check t > 0
    t_val = wt.get("t_stat", float('nan'))
    one_sided_p = p_val / 2 if p_val == p_val else float('nan')
    h3_pass = (
        bool(ra_returns) and bool(dqn_returns)
        and np.mean(ra_returns) > np.mean(dqn_returns)
        and one_sided_p < 0.05
        and t_val > 0
    )

    return {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "train_seeds":   train_seeds,
            "eval_seeds":    eval_seeds,
            "n_train_seeds": n_train,
            "n_eval_seeds":  n_eval,
            "total_runs":    len(all_runs),
            "episode_steps": EPISODE_STEPS,
            "initial_cash":  INITIAL_CASH,
            "tx_cost_pct":   TX_COST_PCT,
        },
        "runs": all_runs,
        "aggregate": {
            "DQN":    dqn_agg,
            "RA-DQN": ra_agg,
        },
        "regime_exposure": regime_exposure,
        "statistical_tests": stats_tests,
        "h3_pass": h3_pass,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Output helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_outputs(result: dict, ts: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Per-run CSV
    csv_path = OUTPUT_DIR / f"h3_eval_runs_{ts}.csv"
    runs = result["runs"]
    if runs:
        fieldnames = list(runs[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in runs:
                row = dict(row)
                for k, v in row.items():
                    if isinstance(v, list):
                        row[k] = json.dumps(v)
                writer.writerow(row)
    print(f"  Per-run CSV  -> {csv_path}")

    # 2. Summary JSON
    json_path = OUTPUT_DIR / f"h3_eval_summary_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Summary JSON -> {json_path}")

    # 3. LaTeX table
    txt_path = OUTPUT_DIR / f"h3_eval_table_{ts}.txt"
    _write_latex_table(result, txt_path)
    print(f"  LaTeX table  -> {txt_path}")


def _fmt(val: float) -> str:
    """Format a float for display; show NaN literally."""
    if val != val:   # NaN check
        return "NaN"
    return f"{val:.2f}"


def _write_latex_table(result: dict, path: Path) -> None:
    agg   = result["aggregate"]
    tests = result["statistical_tests"].get("return_pct", {})
    has_dqn = "DQN" in agg and bool(agg["DQN"])
    has_ra  = "RA-DQN" in agg and bool(agg["RA-DQN"])

    if has_dqn and has_ra:
        header = (
            r"\begin{tabular}{lrrrrr}" "\n"
            r"\hline\hline" "\n"
            r"Metric & \multicolumn{2}{c}{DQN} & \multicolumn{2}{c}{RA-DQN} & \pvalue\\" "\n"
            r"        & Mean & Std & Mean & Std & \\" "\n"
            r"\hline"
        )
    elif has_dqn:
        header = (
            r"\begin{tabular}{lrrr}" "\n"
            r"\hline\hline" "\n"
            r"Metric & \multicolumn{2}{c}{DQN} \\" "\n"
            r"        & Mean & Std \\" "\n"
            r"\hline"
        )
    elif has_ra:
        header = (
            r"\begin{tabular}{lrrr}" "\n"
            r"\hline\hline" "\n"
            r"Metric & \multicolumn{2}{c}{RA-DQN} \\" "\n"
            r"        & Mean & Std \\" "\n"
            r"\hline"
        )
    else:
        header = r"\begin{tabular}{l}" "\n" r"\hline\hline" "\n" r"Metric \\" "\n" r"\hline"

    lines = [header]

    metrics = [
        ("Total Return (%)",          "total_return_pct", False, False),
        ("Final Value ($)",            "final_value",       False, False),
        ("Sharpe-Like Ratio",         "sharpe_like",      False, False),
        ("Max Drawdown (%)",          "max_drawdown_pct", False, False),
        ("Num Buys",                  "n_buys",           False, False),
        ("Num Sells",                 "n_sells",          False, False),
        ("Num Trades",                "n_trades",         False, False),
        ("Median Return (%)",          "total_return_pct", False, True),
    ]

    for label, key, _, is_median in metrics:
        row = [f"{label:<25}"]
        if has_dqn:
            s = agg["DQN"].get(key, {})
            if is_median:
                row.append(f"& {_fmt(s.get('median', float('nan'))):>8}")
                row.append(f"& {_fmt(s.get('iqr', float('nan'))):>8}")
            else:
                row.append(f"& {_fmt(s.get('mean', float('nan'))):>8}")
                row.append(f"& {_fmt(s.get('std',  float('nan'))):>8}")
        if has_ra:
            s = agg["RA-DQN"].get(key, {})
            if is_median:
                row.append(f"& {_fmt(s.get('median', float('nan'))):>8}")
                row.append(f"& {_fmt(s.get('iqr', float('nan'))):>8}")
            else:
                row.append(f"& {_fmt(s.get('mean', float('nan'))):>8}")
                row.append(f"& {_fmt(s.get('std',  float('nan'))):>8}")
        if has_dqn and has_ra:
            p = tests.get("welch_t", {}).get("p_value", float('nan'))
            row.append(f"& ${p:.4f}$")
        lines.append(" ".join(row) + r" \\")

    wt = tests.get("welch_t", {})
    bi = tests.get("bootstrap_ci", {})
    verdict_str = "PASS (RA-DQN $>$ DQN)" if result["h3_pass"] else "FAIL (DQN $\\geq$ RA-DQN)"

    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\begin{tablenotes}",
        rf"\item H3 verdict: {verdict_str} (requires mean diff $>0$ and one-sided $p<0.05$)",
        (r"\item Welch's t-test (two-sided): $t="
         + f"{_fmt(wt.get('t_stat', float('nan')))}$, "
         + f"$p={_fmt(wt.get('p_value', float('nan')))}$, "
         + f"Cohen's $d={_fmt(wt.get('cohens_d', float('nan')))}$"),
        (r"\item Bootstrap 95\% CI on mean return difference: "
         + f"[{_fmt(bi.get('ci_lower', float('nan')))}\% , "
         + f"{_fmt(bi.get('ci_upper', float('nan')))}\%]"),
        r"\end{tablenotes}",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def print_summary(result: dict) -> None:
    agg   = result["aggregate"]
    tests = result["statistical_tests"].get("return_pct", {})
    cfg   = result["config"]
    has_dqn = "DQN" in agg and bool(agg["DQN"])
    has_ra  = "RA-DQN" in agg and bool(agg["RA-DQN"])

    print("\n" + "=" * 75)
    print("H3 EVALUATION SUMMARY")
    print("=" * 75)
    print(f"  Train seeds : {cfg['train_seeds']}")
    print(f"  Eval seeds  : {cfg['n_eval_seeds']} seeds  "
          f"(total {cfg['total_runs']} runs)")
    print(f"  Episode len : {cfg['episode_steps']} ticks  "
          f"  Initial cash: ${cfg['initial_cash']:,.0f}")

    # Total Return
    if has_dqn or has_ra:
        print("\n  [Total Return %]")
        for agent in ["DQN", "RA-DQN"]:
            s = agg.get(agent, {}).get("total_return_pct")
            if s:
                print(f"    {agent:<8} {s['mean']:>14,.2f} +/- {s['std']:>12,.2f}  "
                      f"[{s['min']:>12,.2f} , {s['max']:>12,.2f}]  "
                      f"median={s['median']:>12,.2f}")

    # Statistical tests
    if has_dqn and has_ra:
        print("\n  [Statistical Tests -- Return %]")
        wt = tests.get("welch_t", {})
        pv = wt.get("p_value", float('nan'))
        print(f"    Welch's t-test:  t = {_fmt(wt.get('t_stat', float('nan')))},  "
              f"p = {pv:.4f},  Cohen's d = {_fmt(wt.get('cohens_d', float('nan')))}")
        mw = tests.get("mann_whitney", {})
        print(f"    Mann-Whitney U:  U = {_fmt(mw.get('U', float('nan')))},   "
              f"p = {_fmt(mw.get('p_value', float('nan')))},  "
              f"r = {_fmt(mw.get('rank_biserial_r', float('nan')))}")
        bi = tests.get("bootstrap_ci", {})
        print(f"    Bootstrap 95% CI:  [{_fmt(bi.get('ci_lower', float('nan')))}\% , "
              f"{_fmt(bi.get('ci_upper', float('nan')))}\%]")

    # Risk metrics
    if has_dqn or has_ra:
        print("\n  [Sharpe-Like Ratio]")
        for agent in ["DQN", "RA-DQN"]:
            s = agg.get(agent, {}).get("sharpe_like")
            if s:
                print(f"    {agent:<8} {s['mean']:>8.4f} +/- {s['std']:>7.4f}")

        print("\n  [Max Drawdown %]")
        for agent in ["DQN", "RA-DQN"]:
            s = agg.get(agent, {}).get("max_drawdown_pct")
            if s:
                print(f"    {agent:<8} {s['mean']:>8.2f} +/- {s['std']:>7.2f}")

    # Trading activity
    if has_dqn or has_ra:
        print("\n  [Trading Activity]")
        print(f"    {'Agent':<8}  {'#Buys':>6}  {'#Sells':>6}  {'#Trades':>7}  "
              f"{'BuyRatio':>8}  {'FinalVal':>14}")
        for agent in ["DQN", "RA-DQN"]:
            s = agg.get(agent, {})
            if not s:
                continue
            buys   = s.get("n_buys",   {}).get('mean', 0)
            sells  = s.get("n_sells",  {}).get('mean', 0)
            trades = s.get("n_trades", {}).get('mean', 0)
            ratio  = buys / (trades + 1e-9)
            fv     = s.get("final_value", {}).get('mean', 0)
            print(f"    {agent:<8}  {buys:>6.1f}  {sells:>6.1f}  {trades:>7.1f}  "
                  f"{ratio:>8.2f}  ${fv:>13,.0f}")

    # Regime exposure
    if result["regime_exposure"]:
        print("\n  [RA-DQN Regime Exposure (ground-truth)]")
        for name, frac in result["regime_exposure"].items():
            print(f"    {name:<10} {frac*100:6.1f}%")

    # Verdict
    h3_pass = result["h3_pass"]
    verdict = "PASS (RA-DQN > DQN)" if h3_pass else "FAIL (DQN >= RA-DQN)"
    print(f"\n  H3 verdict: {verdict}")
    print("=" * 75)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="H3 Comprehensive Evaluation: RA-DQN vs Plain DQN")
    parser.add_argument(
        "--eval-seeds", type=int, default=25,
        help="Number of eval seeds (default: 25). Pass 0 to use built-in default 25.")
    parser.add_argument(
        "--train-seeds", type=int, nargs="+",
        default=[42, 123, 456],
        help="Training seeds (default: 42 123 456).")
    parser.add_argument(
        "--agents", choices=["dqn", "ra", "both"], default="both",
        help="Which agents to evaluate (default: both).")
    parser.add_argument(
        "--retrain", action="store_true",
        help="Retrain agents even if checkpoints exist.")
    parser.add_argument(
        "--steps", type=int, default=20_000,
        help="DQN training steps (default: 20 000).")
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress tqdm progress bars.")
    args = parser.parse_args()

    agent_map = {"dqn": ["dqn"], "ra": ["ra"], "both": ["dqn", "ra"]}

    # Resolve eval seeds — always disjoint from eval_harness (train=[42,123,456], eval=[789,1024,2048,4096,8192])
    if args.eval_seeds == 0:
        eval_seeds = None   # use built-in 25 defaults
    else:
        n = args.eval_seeds
        pool = [
            11, 13, 17, 19, 23, 29, 31, 37, 41, 43,
            47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
            5000, 6000, 7000, 8000, 9000,
        ]
        eval_seeds = pool[:n] if n <= len(pool) else (pool * (n // len(pool) + 1))[:n]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_display = len(eval_seeds) if eval_seeds else 25
    print(f"\n=== H3 Evaluation ({n_display} eval seeds) ===")

    result = run_h3_eval(
        eval_seeds=eval_seeds,
        train_seeds=args.train_seeds,
        agents=agent_map[args.agents],
        retrain=args.retrain,
        n_steps=args.steps,
        verbose=not args.quiet,
    )

    save_outputs(result, ts)
    plot_h3_eval_summary(
        summary_json=OUTPUT_DIR / f"h3_eval_summary_{ts}.json",
        runs_csv=OUTPUT_DIR / f"h3_eval_runs_{ts}.csv",
    )
    print_summary(result)

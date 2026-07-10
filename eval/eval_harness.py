"""
Evaluation harness — runs experiments H1, H2, H3 and produces results tables.

H1: DQN beats Random and Buy-and-Hold on return.
H2: Regime classifier beats random guessing (1/6 ≈ 16.7%).
H3: DQN + regime beats plain DQN.

Usage:
    python -m eval.eval_harness
"""

from __future__ import annotations

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict

# Project imports
from sim.market_sim import CookieClickerMarket
from sim.market_sim.dataset import generate_regime_dataset
from eval.env.trading_env import TradingEnv
from eval.agents.baselines import (
    BaseAgent,
    RandomAgent,
    BuyAndHoldAgent,
    MeanReversionAgent,
    evaluate_agent,
)
from eval.agents.dqn import DQNAgent
from eval.agents.dqn_regime import RegimeAwareDQNAgent


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
@dataclass
class EvalConfig:
    n_episodes: int           = 10
    episode_steps: int        = 1000
    seeds: list[int]          = field(default_factory=lambda: [42, 123, 456, 789, 1024])
    initial_cash: float       = 10_000.0
    dataset_n_ticks: int       = 5000
    dqn_n_steps: int          = 20_000      # training steps (keep small for CI/demo)
    dqn_eval_every: int       = 5_000
    output_dir: Path           = Path("outputs")


# ------------------------------------------------------------------
# H2 — Regime classifier accuracy
# ------------------------------------------------------------------
def run_h2(config: EvalConfig) -> dict:
    """
    Experiment 2: Train regime classifiers and measure accuracy.

    Trains LogReg, RandomForest, MLP on the generated dataset.
    Reports per-model accuracy vs random baseline (1/6).
    """
    from eval.classifiers.regime_classifier import RegimeClassifierPipeline

    print("\n=== H2: Regime Classifier Accuracy ===")

    # Generate dataset if not present
    data_path = Path("data/regime_dataset.parquet")
    if not data_path.exists():
        print("Generating regime dataset...")
        generate_regime_dataset(
            n_ticks=config.dataset_n_ticks,
            seed=0,
        )

    pipeline = RegimeClassifierPipeline(data_path=data_path, out_dir="models")
    scores = pipeline.run()

    baseline = 1.0 / 6
    results = {
        "baseline_random": baseline,
        "models": {},
    }
    for name, acc in scores.items():
        results["models"][name] = {
            "accuracy": float(acc),
            "vs_random": float(acc - baseline),
            "pass": acc > baseline,
        }
        print(f"  {name}: {acc:.4f} (Δ {acc - baseline:+.4f} vs random)")

    # Best model
    best_name = max(scores, key=scores.get)
    print(f"  Best: {best_name} @ {scores[best_name]:.4f}")

    return results


# ------------------------------------------------------------------
# H1 — Trading return comparison
# ------------------------------------------------------------------
def run_h1(config: EvalConfig) -> dict:
    """
    Experiment 1: Compare DQN vs baselines (Random, Buy-and-Hold, Mean-Reversion).

    Hypothesis H1: DQN beats Random and Buy-and-Hold on return.
    """
    print("\n=== H1: Trading Return Comparison ===")

    results = {}

    # Baselines
    for name, agent_cls in [
        ("Random",         RandomAgent),
        ("BuyAndHold",     BuyAndHoldAgent),
        ("MeanReversion",  MeanReversionAgent),
    ]:
        print(f"  Running {name}...")
        returns = []
        for seed in config.seeds:
            agent = agent_cls() if agent_cls != RandomAgent else RandomAgent(seed=seed)
            r = evaluate_agent(
                agent,
                market_seed=seed,
                n_steps=config.episode_steps,
                initial_cash=config.initial_cash,
            )
            returns.append(r['total_return_pct'])
        results[name] = {
            "mean_return": float(np.mean(returns)),
            "std_return":  float(np.std(returns)),
            "min_return":  float(np.min(returns)),
            "max_return":  float(np.max(returns)),
            "n_runs":      len(returns),
        }
        print(f"    {name}: {np.mean(returns):.2f}% ± {np.std(returns):.2f}%")

    # DQN (pre-trained or train now)
    dqn_ckpt = Path("models/dqn_agent.pkl")
    if dqn_ckpt.exists():
        print("  Loading trained DQN from models/dqn_agent.pkl ...")
        dqn = DQNAgent.load(dqn_ckpt)
    else:
        print(f"  Training DQN for {config.dqn_n_steps} steps (first seed only)...")
        dqn = DQNAgent(
            obs_dim=8,
            n_actions=3,
            min_replay_size=500,
            batch_size=32,
            seed=0,
        )
        from eval.agents.dqn import train_dqn
        train_result = train_dqn(
            dqn,
            market_seed=0,
            n_steps=config.dqn_n_steps,
            eval_every=config.dqn_eval_every,
            max_episode_steps=config.episode_steps,
            eval_steps=500,
            initial_cash=config.initial_cash,
        )
        dqn.save("models/dqn_agent.pkl")
        print(f"    DQN training done. Best eval return: {train_result['best_eval']:.2f}%")

    # Evaluate DQN
    print("  Evaluating DQN across seeds...")
    dqn_returns = []
    for seed in config.seeds:
        market = CookieClickerMarket(n_stocks=1, seed=seed)
        env    = TradingEnv(
            market,
            initial_cash=config.initial_cash,
            max_steps=config.episode_steps,
            seed=seed,
        )
        dqn.reset()
        obs = env.reset()
        total_ret = 0.0
        done = False
        while not done:
            action = dqn.select_action(obs, {})
            obs, reward, done, _ = env.step(action)
            total_ret += reward
        dqn_returns.append(total_ret * 100.0)

    results["DQN"] = {
        "mean_return": float(np.mean(dqn_returns)),
        "std_return":  float(np.std(dqn_returns)),
        "min_return":  float(np.min(dqn_returns)),
        "max_return":  float(np.max(dqn_returns)),
        "n_runs":      len(dqn_returns),
    }
    print(f"    DQN: {np.mean(dqn_returns):.2f}% ± {np.std(dqn_returns):.2f}%")

    # H1 verdict
    dqn_mean = results["DQN"]["mean_return"]
    h1_pass = (
        dqn_mean > results["Random"]["mean_return"]
        and dqn_mean > results["BuyAndHold"]["mean_return"]
    )
    results["h1_pass"] = h1_pass
    print(f"\n  H1 verdict: {'PASS ✓' if h1_pass else 'FAIL ✗'} (DQN vs baselines)")

    return results


# ------------------------------------------------------------------
# H3 — Regime-aware DQN vs plain DQN
# ------------------------------------------------------------------
def run_h3(config: EvalConfig) -> dict:
    """
    Experiment 3: Regime-aware DQN vs plain DQN.

    Hypothesis H3: DQN + regime beats plain DQN.
    """
    print("\n=== H3: Regime-Aware DQN vs Plain DQN ===")

    # Load trained plain DQN
    dqn_ckpt = Path("models/dqn_agent.pkl")
    if not dqn_ckpt.exists():
        print("  Warning: models/dqn_agent.pkl not found. Run H1 first.")
        return {"error": "DQN checkpoint not found. Run H1 first."}

    dqn = DQNAgent.load(dqn_ckpt)

    # Load regime classifier
    clf_path = Path("models/best_model.pkl")
    if not clf_path.exists():
        print("  Warning: models/best_model.pkl not found. Run H2 first.")
        return {"error": "Classifier not found. Run H2 first."}

    clf = pickle.load(open(clf_path, "rb"))
    scaler = pickle.load(open(Path("models/scaler.pkl"), "rb"))

    def classify(obs: np.ndarray) -> int:
        """Wrap scaler + classifier for RegimeAwareDQNAgent."""
        x = scaler.transform(obs.reshape(1, -1))
        return int(clf.predict(x)[0])

    # Train regime-aware DQN
    ra_dqn_ckpt = Path("models/ra_dqn_agent.pkl")
    if ra_dqn_ckpt.exists():
        print("  Loading trained regime-aware DQN...")
        ra_dqn = RegimeAwareDQNAgent.load(ra_dqn_ckpt)
        ra_dqn.set_classifier(classify)
    else:
        print(f"  Training regime-aware DQN for {config.dqn_n_steps} steps...")
        ra_dqn = RegimeAwareDQNAgent(
            obs_dim=8,
            n_actions=3,
            min_replay_size=500,
            batch_size=32,
            seed=0,
        )
        ra_dqn.set_classifier(classify)

        from eval.agents.dqn import train_dqn as _train_dqn
        # Reuse same training loop but with regime agent
        _train_ra(
            ra_dqn,
            market_seed=0,
            n_steps=config.dqn_n_steps,
            eval_every=config.dqn_eval_every,
            max_episode_steps=config.episode_steps,
            eval_steps=500,
            initial_cash=config.initial_cash,
        )
        ra_dqn.save("models/ra_dqn_agent.pkl")

    # Evaluate both
    def _eval_agent(agent, seeds):
        returns = []
        for seed in seeds:
            market = CookieClickerMarket(n_stocks=1, seed=seed)
            env    = TradingEnv(
                market,
                initial_cash=config.initial_cash,
                max_steps=config.episode_steps,
                seed=seed,
            )
            agent.reset()
            obs = env.reset()
            total_ret = 0.0
            done = False
            while not done:
                action = agent.select_action(obs, {})
                obs, reward, done, _ = env.step(action)
                total_ret += reward
            returns.append(total_ret * 100.0)
        return returns

    print("  Evaluating plain DQN...")
    dqn_returns = _eval_agent(dqn, config.seeds)
    print("  Evaluating regime-aware DQN...")
    ra_returns  = _eval_agent(ra_dqn, config.seeds)

    results = {
        "DQN": {
            "mean_return": float(np.mean(dqn_returns)),
            "std_return":  float(np.std(dqn_returns)),
            "n_runs":      len(dqn_returns),
        },
        "DQN_Regime": {
            "mean_return": float(np.mean(ra_returns)),
            "std_return":  float(np.std(ra_returns)),
            "n_runs":      len(ra_returns),
        },
    }
    print(f"    DQN:         {np.mean(dqn_returns):.2f}% ± {np.std(dqn_returns):.2f}%")
    print(f"    DQN+Regime:  {np.mean(ra_returns):.2f}% ± {np.std(ra_returns):.2f}%")

    h3_pass = results["DQN_Regime"]["mean_return"] > results["DQN"]["mean_return"]
    results["h3_pass"] = h3_pass
    print(f"\n  H3 verdict: {'PASS ✓' if h3_pass else 'FAIL ✗'} (DQN+Regime vs DQN)")

    return results


# ------------------------------------------------------------------
# Regime-aware DQN training loop (subset of train_dqn)
# ------------------------------------------------------------------
def _train_ra(
    agent: RegimeAwareDQNAgent,
    market_seed: int,
    n_steps: int,
    eval_every: int,
    max_episode_steps: int | None,
    eval_steps: int,
    initial_cash: float,
) -> dict:
    logs = {"loss": [], "epsilon": []}
    best_eval = -np.inf

    market = CookieClickerMarket(n_stocks=1, seed=market_seed)
    env    = TradingEnv(
        market,
        initial_cash=initial_cash,
        max_steps=max_episode_steps,
        seed=market_seed,
    )

    episode_return = 0.0
    agent.reset()
    obs = env.reset()

    for step in range(n_steps):
        action    = agent._epsilon_greedy(obs, training=True)
        next_obs, reward, done, info = env.step(action)
        episode_return += reward

        agent.store(obs, action, reward, next_obs, done)
        obs = next_obs

        loss = agent.train_step()
        if loss is not None:
            logs["loss"].append(float(loss))

        logs["epsilon"].append(agent.epsilon)

        if done:
            env.reset()
            agent.reset()
            obs = env.reset()

        if (step + 1) % eval_every == 0:
            # quick eval
            ev_returns = []
            for s in [market_seed + i for i in range(3)]:
                m = CookieClickerMarket(n_stocks=1, seed=s)
                e = TradingEnv(m, initial_cash=initial_cash, max_steps=eval_steps, seed=s)
                agent.reset()
                o = e.reset()
                ret = 0.0
                dn = False
                while not dn:
                    a = agent.select_action(o, {})
                    o, r, dn, _ = e.step(a)
                    ret += r
                ev_returns.append(ret * 100.0)
            mean_ev = float(np.mean(ev_returns))
            if mean_ev > best_eval:
                best_eval = mean_ev

    return {"logs": logs, "best_eval": float(best_eval)}


# ------------------------------------------------------------------
# Full run + reporting
# ------------------------------------------------------------------
def run_all(config: EvalConfig | None = None) -> dict:
    """Run H1, H2, H3 in sequence and return combined results."""
    if config is None:
        config = EvalConfig()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        "config": asdict(config),
        "timestamp": ts,
        "h2": run_h2(config),
        "h1": run_h1(config),
        "h3": run_h3(config),
    }

    # Save JSON
    out_path = config.output_dir / f"results_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print("\nH1 — Trading Returns:")
    print(f"  {'Agent':<15} {'Mean %':>10} {'Std %':>10}")
    for name, stats in results["h1"].items():
        if name in ("h1_pass",):
            continue
        print(f"  {name:<15} {stats['mean_return']:>10.2f} {stats['std_return']:>10.2f}")
    print(f"  H1 (DQN beats baselines): {'PASS ✓' if results['h1'].get('h1_pass') else 'FAIL ✗'}")

    h2 = results.get("h2", {})
    if "models" in h2:
        print("\nH2 — Regime Classifier Accuracy:")
        print(f"  Baseline (random): {h2['baseline_random']:.4f}")
        for name, stats in h2["models"].items():
            mark = "✓" if stats["pass"] else ""
            print(f"  {name:<15} {stats['accuracy']:.4f}  (Δ {stats['vs_random']:+.4f}) {mark}")

    h3 = results.get("h3", {})
    if "DQN" in h3 and "DQN_Regime" in h3:
        print("\nH3 — Regime-Aware DQN:")
        print(f"  {'DQN':<15} {results['h3']['DQN']['mean_return']:>10.2f}%")
        print(f"  {'DQN+Regime':<15} {results['h3']['DQN_Regime']['mean_return']:>10.2f}%")
        print(f"  H3 (DQN+Regime beats DQN): {'PASS ✓' if h3.get('h3_pass') else 'FAIL ✗'}")

    return results


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run H1/H2/H3 experiments")
    parser.add_argument("--h1", action="store_true", help="Run H1 only")
    parser.add_argument("--h2", action="store_true", help="Run H2 only")
    parser.add_argument("--h3", action="store_true", help="Run H3 only")
    parser.add_argument("--steps", type=int, default=20_000, help="DQN training steps")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    args = parser.parse_args()

    config = EvalConfig(
        dqn_n_steps=args.steps,
        seeds=args.seeds,
    )

    if args.h1:
        run_h1(config)
    elif args.h2:
        run_h2(config)
    elif args.h3:
        run_h3(config)
    else:
        run_all(config)

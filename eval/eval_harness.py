"""
Evaluation harness — runs experiments H1, H2, H3 and produces results tables.

H1: DQN beats Random and Buy-and-Hold on return.
H2: Regime classifier beats random guessing (1/4 = 25.0%).
H3: DQN + regime beats plain DQN.

Usage:
    python -m eval.eval_harness
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — avoids tkinter conflicts on Windows

import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict

# Project imports
from eval.visualization import plot_training_run
from sim.market_sim import CookieClickerMarket
from tqdm import tqdm
from sim.market_sim.dataset import generate_regime_dataset
from eval.env.trading_env import TradingEnv
from eval.agents.baselines import (
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
    episode_steps: int        = 500       # 1000 ticks → ~15M% compounded return; 500 → ~1100%
    seeds: list[int]          = field(default_factory=lambda: [42, 123, 456, 789, 1024, 2048, 4096, 8192])
    initial_cash: float       = 10_000.0
    dataset_n_ticks: int       = 20_000
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
    Three-way split: 60% train / 20% val / 20% test.
    Model selection on val; final accuracy reported on held-out test set.
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
    result = pipeline.run()

    baseline = 1.0 / 4

    # Final reported score is test accuracy of the best model
    best_name  = result["best_name"]
    best_test_acc = result["test_scores"][best_name]

    results = {
        "baseline_random": baseline,
        "models": {},
        "best_test_accuracy": float(best_test_acc),
        "best_model": best_name,
        "h2_pass": best_test_acc > baseline,
    }

    print(f"\n  Baseline (random): {baseline:.4f}")
    print(f"  Best model on held-out test: {best_name} = {best_test_acc:.4f}")
    print(f"  H2: {results['h2_pass']} (test accuracy > {baseline:.4f})")

    return results


# ------------------------------------------------------------------
# H1 — Trading return comparison
# ------------------------------------------------------------------
def run_h1(config: EvalConfig) -> dict:
    """
    Experiment 1: Compare DQN vs baselines (Random, Buy-and-Hold, Mean-Reversion).

    Train/eval split: DQN trains on the FIRST N-2 seeds, evaluates on the LAST 2.
    Baselines are evaluated on the SAME held-out seeds for a fair comparison.
    H1: DQN beats Random and Buy-and-Hold on return.
    """
    print("\n=== H1: Trading Return Comparison ===")

    train_seeds = config.seeds[:-5]
    eval_seeds  = config.seeds[-5:]
    print(f"  Train seeds: {train_seeds}")
    print(f"  Eval seeds:  {eval_seeds}")

    results = {}
    from eval.agents.dqn import _eval_agent as dqn_eval, train_dqn

    # Train one DQN per seed; evaluate each on every eval seed
    all_dqn_returns: list[float] = []
    train_logs = None

    for seed in train_seeds:
        dqn_ckpt = Path(f"models/dqn_agent_{seed}.pkl")
        if dqn_ckpt.exists():
            print(f"  Loading trained DQN (seed={seed}) ...")
            dqn = DQNAgent.load(dqn_ckpt)
        else:
            print(f"  Training DQN for {config.dqn_n_steps} steps on seed={seed}...")
            dqn = DQNAgent(
                obs_dim=8, n_actions=3,
                min_replay_size=500, batch_size=32,
                seed=seed,
            )
            train_result = train_dqn(
                dqn,
                market_seed=seed,
                n_steps=config.dqn_n_steps,
                eval_every=config.dqn_eval_every,
                max_episode_steps=config.episode_steps,
                eval_steps=500,
                initial_cash=config.initial_cash,
                train_seeds=train_seeds,
                log_path=config.output_dir / f"h1_seed{seed}_log.csv",
            )
            dqn.save(dqn_ckpt)
            train_logs = train_result["logs"]
            print(f"    DQN (seed={seed}) done. Best eval return: {train_result['best_eval']:.2f}%")

        seed_returns = dqn_eval(dqn, eval_seeds, config.episode_steps, config.initial_cash)
        all_dqn_returns.extend(seed_returns)
        print(f"    DQN (seed={seed}): {np.mean(seed_returns):.2f}% ± {np.std(seed_returns):.2f}%")

    # Plot first agent's training progress
    if train_logs:
        plot_path = config.output_dir / "h1_dqn_training.png"
        plot_training_run(
            logs=train_logs,
            title="H1 — DQN Training Progress (seed={})".format(train_seeds[0]),
            output_path=plot_path,
            eval_every=config.dqn_eval_every,
            total_steps=config.dqn_n_steps,
        )

    results["DQN"] = {
        "mean_return": float(np.mean(all_dqn_returns)),
        "std_return":  float(np.std(all_dqn_returns)),
        "min_return":  float(np.min(all_dqn_returns)),
        "max_return":  float(np.max(all_dqn_returns)),
        "n_runs":      len(all_dqn_returns),
        "train_seeds": train_seeds,
        "eval_seeds":  eval_seeds,
    }
    print(f"  DQN (aggregate {len(all_dqn_returns)} runs): "
          f"{np.mean(all_dqn_returns):.2f}% ± {np.std(all_dqn_returns):.2f}%")

    # Baselines — also evaluated on the SAME held-out eval_seeds
    for name, agent_cls in [
        ("Random",         RandomAgent),
        ("BuyAndHold",     BuyAndHoldAgent),
        ("MeanReversion",  MeanReversionAgent),
    ]:
        print(f"  Running {name} on held-out seeds...")
        returns = []
        for seed in eval_seeds:
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

    Train/eval split matches H1: train on first N-2 seeds, evaluate on last 2.
    Both DQN variants are trained from scratch on the same train seeds and
    evaluated on the same held-out eval seeds.
    H3: DQN + regime beats plain DQN.
    """
    print("\n=== H3: Regime-Aware DQN vs Plain DQN ===")

    train_seeds = config.seeds[:-5]
    eval_seeds  = config.seeds[-5:]
    print(f"  Train seeds: {train_seeds}")
    print(f"  Eval seeds:  {eval_seeds}")

    # Load regime classifier
    clf_path = Path("models/best_model.pkl")
    scaler_path = Path("models/scaler.pkl")
    if not clf_path.exists() or not scaler_path.exists():
        print("  Warning: models/best_model.pkl or scaler.pkl not found. Run H2 first.")
        return {"error": "Classifier not found. Run H2 first."}

    clf   = pickle.load(open(clf_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))


    # --- Plain DQN (load all trained checkpoints from H1) ---
    print(f"  Loading plain DQNs from H1 checkpoints...")
    from eval.agents.dqn import _eval_agent as dqn_eval
    all_dqn_returns: list[float] = []
    for seed in train_seeds:
        dqn_ckpt = Path(f"models/dqn_agent_{seed}.pkl")
        if not dqn_ckpt.exists():
            print(f"  ERROR: {dqn_ckpt} not found. Run H1 first.")
            return {"error": "H1 checkpoints missing. Run H1 first."}
        dqn = DQNAgent.load(dqn_ckpt)
        seed_returns = dqn_eval(dqn, eval_seeds, config.episode_steps, config.initial_cash)
        all_dqn_returns.extend(seed_returns)
        print(f"    DQN (seed={seed}): {np.mean(seed_returns):.2f}% ± {np.std(seed_returns):.2f}%")
    print(f"  DQN aggregate: {np.mean(all_dqn_returns):.2f}% ± {np.std(all_dqn_returns):.2f}% ({len(all_dqn_returns)} runs)")

    # --- Regime-aware DQN ---
    # classify closure: uses clf/scaler (from outer scope) and agent._env (set at call time)
    def make_classify_fn(agent, clf, scaler):
        def classify(obs: np.ndarray) -> int:
            """Classify regime using scaler + clf on 18 features."""
            env = agent._env
            stock = env.market.stocks[0]
            vals  = stock['vals']
            price = stock['price']

            tick_rets = [
                (vals[i] - vals[i+1]) / (vals[i+1] + 1e-8)
                for i in range(min(len(vals) - 1, 64))
            ]
            def trets(n): return tick_rets[:n]

            ret_1  = tick_rets[0] if len(tick_rets) >= 1 else 0.0
            ret_5  = (vals[0] - vals[4]) / vals[4] if len(vals) > 4 else 0.0
            ret_10 = (vals[0] - vals[9]) / vals[9] if len(vals) > 9 else 0.0
            ret_20 = (vals[0] - vals[19]) / vals[19] if len(vals) > 19 else 0.0

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

    # --- Regime-aware DQN (one per train seed for fair comparison) ---
    all_ra_returns: list[float] = []
    ra_train_logs = None

    for seed in train_seeds:
        ra_ckpt = Path(f"models/ra_dqn_agent_{seed}.pkl")
        if ra_ckpt.exists():
            print(f"  Loading trained RA-DQN (seed={seed}) ...")
            ra_dqn = RegimeAwareDQNAgent.load(ra_ckpt)
            ra_dqn.set_classifier(make_classify_fn(ra_dqn, clf, scaler))
        else:
            print(f"  Training RA-DQN for {config.dqn_n_steps} steps on seed={seed}...")
            ra_dqn = RegimeAwareDQNAgent(
                obs_dim=8, n_actions=3,
                min_replay_size=500, batch_size=32,
                seed=seed,
            )
            ra_dqn.set_classifier(make_classify_fn(ra_dqn, clf, scaler))
            ra_result = _train_ra(
                ra_dqn,
                market_seed=seed,
                n_steps=config.dqn_n_steps,
                eval_every=config.dqn_eval_every,
                max_episode_steps=config.episode_steps,
                eval_steps=500,
                initial_cash=config.initial_cash,
                train_seeds=train_seeds,
                verbose=True,
                log_path=config.output_dir / f"h3_ra_seed{seed}_log.csv",
            )
            ra_dqn.save(ra_ckpt)
            ra_train_logs = ra_result["logs"]
            print(f"    RA-DQN (seed={seed}) done. Best eval: {ra_result['best_eval']:.2f}%")

        seed_returns = dqn_eval(ra_dqn, eval_seeds, config.episode_steps, config.initial_cash)
        all_ra_returns.extend(seed_returns)
        print(f"    RA-DQN (seed={seed}): {np.mean(seed_returns):.2f}% ± {np.std(seed_returns):.2f}%")

    if ra_train_logs:
        plot_path = config.output_dir / "h3_ra_dqn_training.png"
        plot_training_run(
            logs=ra_train_logs,
            title="H3 — RA-DQN Training Progress (seed={})".format(train_seeds[0]),
            output_path=plot_path,
            eval_every=config.dqn_eval_every,
            total_steps=config.dqn_n_steps,
        )

    print(f"  RA-DQN aggregate: {np.mean(all_ra_returns):.2f}% ± {np.std(all_ra_returns):.2f}% "
          f"({len(all_ra_returns)} runs)")

    results = {
        "DQN": {
            "mean_return": float(np.mean(all_dqn_returns)),
            "std_return":  float(np.std(all_dqn_returns)),
            "n_runs":      len(all_dqn_returns),
            "train_seeds": train_seeds,
            "eval_seeds":  eval_seeds,
        },
        "DQN_Regime": {
            "mean_return": float(np.mean(all_ra_returns)),
            "std_return":  float(np.std(all_ra_returns)),
            "n_runs":      len(all_ra_returns),
            "train_seeds": train_seeds,
            "eval_seeds":  eval_seeds,
        },
        "h3_pass": float(np.mean(all_ra_returns)) > float(np.mean(all_dqn_returns)),
    }
    h3_pass = results["h3_pass"]
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
    train_seeds: list[int] | None = None,
    verbose: bool = False,
    log_path: Path | None = None,
) -> dict:
    """
    Train a regime-aware DQN agent.

    Args:
        train_seeds: seeds to use for eval during training (must NOT include
                      the training seed to avoid leakage).
    """
    import sys
    import csv
    logs = {"loss": [], "epsilon": [], "episode_return": [], "eval_returns": []}
    best_eval = -np.inf

    market = CookieClickerMarket(n_stocks=1, seed=market_seed)
    env    = TradingEnv(
        market,
        initial_cash=initial_cash,
        max_steps=max_episode_steps,
        seed=market_seed,
    )
    agent.set_env(env)

    episode_return = 0.0
    agent.reset()
    obs = env.reset()

    # Default eval seeds — don't include the training seed
    eval_seed_pool = [42, 123, 456, 789, 1024]
    if train_seeds:
        eval_seed_pool = [s for s in eval_seed_pool if s not in train_seeds]

    # Open CSV log for incremental writes
    csv_file = None
    csv_writer = None
    if log_path:
        csv_file = open(log_path, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=["step", "loss", "epsilon", "episode_return", "eval_return_pct"])
        csv_writer.writeheader()

    iterator = tqdm(range(n_steps), desc="RA-DQN", unit="step", disable=not verbose)

    try:
        for step in iterator:
            action    = agent._epsilon_greedy(obs, training=True)
            next_obs, reward, done, info = env.step(action)
            episode_return += reward

            # store() with no regime args → infers internally via _infer_regime,
            # which calls the injected classifier closure (uses agent._env for features)
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
                # quick eval on held-out seeds only
                ev_returns = []
                for s in eval_seed_pool[:3]:
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
                logs["eval_returns"].append(mean_ev)
                if mean_ev > best_eval:
                    best_eval = mean_ev
                iterator.set_postfix(epsilon=f"{agent.epsilon:.3f}", eval_ret=f"{mean_ev:.2f}%")

                # Incrementally write to CSV
                if csv_writer is not None:
                    last_loss = logs["loss"][-1] if logs["loss"] else ""
                    last_ep  = logs["episode_return"][-1] if logs["episode_return"] else ""
                    csv_writer.writerow({
                        "step": step + 1,
                        "loss": last_loss,
                        "epsilon": agent.epsilon,
                        "episode_return": last_ep,
                        "eval_return_pct": mean_ev,
                    })
                    csv_file.flush()

    finally:
        if csv_file is not None:
            csv_file.close()

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

    print("\nH1 — Trading Returns (eval on held-out seeds):")
    print(f"  {'Agent':<15} {'Mean %':>10} {'Std %':>10}")
    for name in ("Random", "BuyAndHold", "MeanReversion", "DQN"):
        if name in results["h1"]:
            stats = results["h1"][name]
            print(f"  {name:<15} {stats['mean_return']:>10.2f} {stats['std_return']:>10.2f}")
    print(f"  H1 (DQN > Random & BuyAndHold): {'PASS ✓' if results['h1'].get('h1_pass') else 'FAIL ✗'}")

    h2 = results.get("h2", {})
    if "best_test_accuracy" in h2:
        print("\nH2 — Regime Classifier Accuracy (held-out test set):")
        print(f"  Baseline (random): {h2['baseline_random']:.4f}")
        print(f"  Best model: {h2['best_model']} = {h2['best_test_accuracy']:.4f}")
        print(f"  H2 (classifier > random): {'PASS ✓' if h2.get('h2_pass') else 'FAIL ✗'}")

    h3 = results.get("h3", {})
    if "DQN" in h3 and "DQN_Regime" in h3:
        print("\nH3 — Regime-Aware DQN (eval on held-out seeds):")
        print(f"  {'DQN':<15} {h3['DQN']['mean_return']:>10.2f}%")
        print(f"  {'DQN+Regime':<15} {h3['DQN_Regime']['mean_return']:>10.2f}%")
        print(f"  H3 (DQN+Regime > DQN): {'PASS ✓' if h3.get('h3_pass') else 'FAIL ✗'}")

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
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024, 2048, 4096, 8192])
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

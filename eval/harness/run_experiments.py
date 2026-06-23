"""Experiment runner: Exp 1 (traders), Exp 2 (classifiers), Exp 3 (ablation)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sim.market_sim import CookieClickerMarket, generate_regime_dataset
# FEATURE_COLS lives in eval.classifier.regime_classifier
from eval.env.trading_env import TradingEnv
from eval.baselines.heuristics import (
 BuyAndHoldTrader,
 MeanReversionTrader,
 RandomTrader,
)
from eval.classifier.regime_classifier import (
 FEATURE_COLS,
 fit_one,
 load_dataset,
 make_logreg,
 make_mlp,
 make_random_forest,
 save_classifier,
)
from eval.agents.dqn import DQNAgent, DQNConfig
from eval.agents.train import evaluate_dqn, train_dqn, _build_obs


def _env_factory(seed: int, tmax: int) -> callable:
 def make() -> TradingEnv:
  market = CookieClickerMarket(seed=seed)
  return TradingEnv(market=market, tmax=tmax)
 return make


def _eval_heuristic(env: TradingEnv, trader) -> dict:
 obs = env.reset()
 trader.reset()
 net_worths = [env.portfolio.cash + env.portfolio.shares * env.market.current_prices[0]]
 for t in range(env.tmax):
  action = trader.act(obs, env.portfolio_features())
  obs, _reward, done, info = env.step(action)
  net_worths.append(info["net_worth"])
  if done:
   break
 net_worths = np.asarray(net_worths)
 peak = np.maximum.accumulate(net_worths)
 drawdown = (peak - net_worths) / np.maximum(peak, 1e-9)
 rets = np.diff(net_worths) / np.maximum(net_worths[:-1], 1e-9)
 return {
  "return_pct": float((net_worths[-1] / net_worths[0]) - 1.0),
  "max_drawdown": float(drawdown.max()),
  "final_net_worth": float(net_worths[-1]),
  "sharpe_like": float(rets.mean() / rets.std() * np.sqrt(len(rets))) if rets.std() > 1e-9 else 0.0,
  "win_rate": float((rets > 0).mean()) if len(rets) > 0 else 0.0,
  "equity_curve": net_worths,
 }


def experiment_1_traders(ticks: int, seeds: int, out_path: Path) -> pd.DataFrame:
 print("\n=== Experiment 1: Trading Strategy Comparison ===")
 rows = []
 traders = {
  "Random": RandomTrader,
  "BuyAndHold": BuyAndHoldTrader,
  "MeanReversion": MeanReversionTrader,
 }
 for seed in range(seeds):
  for name, cls in traders.items():
   env_factory = _env_factory(seed=seed * 1000 + 7, tmax=ticks)
   env = env_factory()
   if cls is RandomTrader:
    trader = cls(seed=seed)
   else:
    trader = cls()
   metrics = _eval_heuristic(env, trader)
   rows.append({"seed": seed, "trader": name, **{k: v for k, v in metrics.items() if k != "equity_curve"}})
   print(f" seed={seed} {name:14s} return={metrics['return_pct']:+.3%} dd={metrics['max_drawdown']:.3f}")
 df = pd.DataFrame(rows)
 df.to_csv(out_path, index=False)
 print(f" -> wrote {out_path}")
 return df


def experiment_1_dqn(ticks: int, seeds: int, episodes: int, out_path: Path) -> pd.DataFrame:
 print("\n=== Experiment 1 (DQN): Training + Eval ===")
 rows = []
 all_curves = {}
 for seed in range(seeds):
  env_factory = _env_factory(seed=seed * 1000 + 7, tmax=ticks)
  cfg = DQNConfig(input_dim=11, warmup=200, eps_decay_steps=episodes * ticks // 2)
  agent = DQNAgent(cfg, use_regime=False)
  print(f" seed={seed} training DQN for {episodes} episodes ...")
  t0 = time.time()
  _ = train_dqn(env_factory, agent, n_episodes=episodes, episode_length=ticks, verbose=True)
  dt = time.time() - t0
  env_factory = _env_factory(seed=seed * 1000 + 7, tmax=ticks)
  metrics = evaluate_dqn(env_factory, agent, episode_length=ticks)
  rows.append({"seed": seed, "trader": "DQN", **{k: v for k, v in metrics.items() if k != "equity_curve"}})
  all_curves[seed] = metrics["equity_curve"]
  print(f" seed={seed} DQN return={metrics['return_pct']:+.3%} trades={metrics['n_trades']} ({dt:.1f}s)")
 df = pd.DataFrame(rows)
 df.to_csv(out_path, index=False)
 np.savez(out_path.with_suffix(".npz"), **{f"seed_{k}": v for k, v in all_curves.items()})
 print(f" -> wrote {out_path}")
 return df


def experiment_2_classifiers(dataset_path: Path, out_path: Path) -> pd.DataFrame:
 print("\n=== Experiment 2: Regime Classifier Performance ===")
 df = load_dataset(dataset_path)
 # Time-ordered split: 70% train, 30% test
 n = len(df)
 split = int(0.7 * n)
 df_train = df.iloc[:split].reset_index(drop=True)
 df_test = df.iloc[split:].reset_index(drop=True)
 rows = []
 for name, factory in [
  ("LogReg", make_logreg),
  ("RandomForest", make_random_forest),
  ("MLP", make_mlp),
 ]:
  model = fit_one(name, factory(), df_train, df_test)
  rows.append({"classifier": name, "accuracy": model.accuracy, "f1_macro": model.f1_macro})
  print(f" {name:14s} accuracy={model.accuracy:.3f} f1={model.f1_macro:.3f}")
  if name == "MLP":
   # Save the best one for the DQN+regime variant
   save_classifier(model, "data/regime_classifier.joblib")
   print(" -> saved data/regime_classifier.joblib")
 df = pd.DataFrame(rows)
 df.to_csv(out_path, index=False)
 print(f" -> wrote {out_path}")
 return df


def experiment_3_ablation(ticks: int, seeds: int, episodes: int, out_path: Path) -> pd.DataFrame:
 print("\n=== Experiment 3: DQN vs DQN+Regime ===")
 # Load saved classifier
 from eval.classifier.regime_classifier import load_classifier
 clf = load_classifier("data/regime_classifier.joblib")

 def regime_fn(obs: np.ndarray) -> np.ndarray:
  # obs shape (n_stocks, 8) per contract
  flat = obs.reshape(-1).reshape(1, -1) if obs.ndim == 1 else obs.reshape(1, -1)
  proba = clf.pipeline.predict_proba(flat)[0]
  # If only 5 classes predicted, pad to 6
  if len(proba) < 6:
   padded = np.zeros(6, dtype=np.float64)
   padded[: len(proba)] = proba
   return padded
  return proba

 rows = []
 all_curves = {}
 for use_regime in (False, True):
  variant = "DQN+Regime" if use_regime else "DQN"
  in_dim = 11 + (6 if use_regime else 0)
  for seed in range(seeds):
   env_factory = _env_factory(seed=seed * 1000 + 7, tmax=ticks)
   cfg = DQNConfig(input_dim=in_dim, warmup=200, eps_decay_steps=episodes * ticks // 2)
   agent = DQNAgent(cfg, use_regime=use_regime)
   print(f" variant={variant} seed={seed} training {episodes} episodes ...")
   t0 = time.time()
   _ = train_dqn(
    env_factory,
    agent,
    n_episodes=episodes,
    episode_length=ticks,
    regime_prob_fn=regime_fn if use_regime else None,
    verbose=True,
   )
   dt = time.time() - t0
   env_factory = _env_factory(seed=seed * 1000 + 7, tmax=ticks)
   metrics = evaluate_dqn(
    env_factory,
    agent,
    episode_length=ticks,
    regime_prob_fn=regime_fn if use_regime else None,
   )
   rows.append({
    "seed": seed,
    "variant": variant,
    **{k: v for k, v in metrics.items() if k != "equity_curve"},
   })
   all_curves[f"{variant}_{seed}"] = metrics["equity_curve"]
   print(f" {variant} seed={seed} return={metrics['return_pct']:+.3%} ({dt:.1f}s)")
 df = pd.DataFrame(rows)
 df.to_csv(out_path, index=False)
 np.savez(out_path.with_suffix(".npz"), **all_curves)
 print(f" -> wrote {out_path}")
 return df


def main() -> None:
 parser = argparse.ArgumentParser()
 parser.add_argument("--ticks", type=int, default=1000)
 parser.add_argument("--seeds", type=int, default=2)
 parser.add_argument("--episodes", type=int, default=15)
 parser.add_argument("--n-ticks-dataset", type=int, default=3000)
 parser.add_argument("--skip-data", action="store_true")
 args = parser.parse_args()

 out_dir = Path("results")
 out_dir.mkdir(exist_ok=True)
 data_dir = Path("data")
 data_dir.mkdir(exist_ok=True)

 # 0. Generate dataset (used by Exp 2 + Exp 3)
 dataset_path = data_dir / "regime_dataset.parquet"
 if not dataset_path.exists() and not args.skip_data:
  print(f"\n=== Generating regime dataset ({args.n_ticks_dataset} ticks) ===")
  generate_regime_dataset(
   n_ticks=args.n_ticks_dataset,
   n_stocks=1,
   seed=42,
   out_path=dataset_path,
  )
  print(f" -> wrote {dataset_path}")
 else:
  print(f"\n=== Using existing dataset: {dataset_path} ===")

 t0 = time.time()
 df1 = experiment_1_traders(args.ticks, args.seeds, out_dir / "exp1_heuristics.csv")
 df1_dqn = experiment_1_dqn(args.ticks, args.seeds, args.episodes, out_dir / "exp1_dqn.csv")
 df2 = experiment_2_classifiers(dataset_path, out_dir / "exp2_classifiers.csv")
 df3 = experiment_3_ablation(args.ticks, args.seeds, args.episodes, out_dir / "exp3_ablation.csv")
 dt = time.time() - t0

 print("\n=== Summary ===")
 print(f" Exp 1 heuristics: mean return by trader")
 print(df1.groupby("trader")[["return_pct", "max_drawdown", "sharpe_like"]].mean().to_string())
 print(f" Exp 1 DQN:")
 print(df1_dqn.groupby("trader")[["return_pct", "max_drawdown", "sharpe_like"]].mean().to_string())
 print(f" Exp 2 classifiers:")
 print(df2.to_string(index=False))
 print(f" Exp 3 ablation:")
 print(df3.groupby("variant")[["return_pct", "max_drawdown", "sharpe_like"]].mean().to_string())
 print(f"\nTotal time: {dt:.1f}s")


if __name__ == "__main__":
 main()
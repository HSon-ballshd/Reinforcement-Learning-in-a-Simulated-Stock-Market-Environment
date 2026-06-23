"""Train a DQN agent on the trading environment."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch

from eval.agents.dqn import DQNAgent, DQNConfig
from eval.env.trading_env import TradingEnv


def _build_obs(env: TradingEnv, regime_probs: Optional[np.ndarray] = None) -> np.ndarray:
 market_obs = env._last_obs # (n_stocks, 8)
 portfolio = env.portfolio_features()
 parts = [market_obs.reshape(-1), portfolio]
 if regime_probs is not None:
  parts.append(np.asarray(regime_probs, dtype=np.float64))
 return np.concatenate(parts).astype(np.float32)


def train_dqn(
 env_factory: Callable[[], TradingEnv],
 agent: DQNAgent,
 n_episodes: int = 50,
 episode_length: int = 1000,
 regime_prob_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
 verbose: bool = True,
) -> list:
 """Train ``agent`` for ``n_episodes`` of ``episode_length`` ticks each.

 Returns a list of per-episode total reward.
 """
 episode_returns = []
 for ep in range(n_episodes):
  env = env_factory()
  obs = env.reset()
  agent.reset()
  regime_probs = regime_prob_fn(obs) if regime_prob_fn is not None else None
  state = _build_obs(env, regime_probs)
  ep_return = 0.0
  for t in range(episode_length):
   action = agent.act(state)
   obs_next, reward, done, info = env.step(action)
   regime_probs_next = regime_prob_fn(obs_next) if regime_prob_fn is not None else None
   next_state = _build_obs(env, regime_probs_next)
   agent.push(state, action, reward, next_state, float(done))
   loss = agent.train_step()
   state = next_state
   ep_return += float(reward)
   if done:
    break
  episode_returns.append(ep_return)
  if verbose and (ep + 1) % max(1, n_episodes // 10) == 0:
   print(
    f" ep {ep+1:3d}/{n_episodes} return={ep_return:+.3f} "
    f"eps={agent.epsilon:.3f} loss={agent.mean_recent_loss:.4f}"
   )
 return episode_returns


def evaluate_dqn(
 env_factory: Callable[[], TradingEnv],
 agent: DQNAgent,
 episode_length: int = 1000,
 regime_prob_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> dict:
 """Run one greedy episode; return summary metrics."""
 env = env_factory()
 obs = env.reset()
 regime_probs = regime_prob_fn(obs) if regime_prob_fn is not None else None
 state = _build_obs(env, regime_probs)
 net_worths = [env.portfolio.cash + env.portfolio.shares * env.market.current_prices[0]]
 trades = 0
 for t in range(episode_length):
  action = agent.act(state, greedy=True)
  if action != 0:
   trades += 1
  obs, reward, done, info = env.step(action)
  regime_probs = regime_prob_fn(obs) if regime_prob_fn is not None else None
  state = _build_obs(env, regime_probs)
  net_worths.append(info["net_worth"])
  if done:
   break
 net_worths = np.asarray(net_worths)
 peak = np.maximum.accumulate(net_worths)
 drawdown = (peak - net_worths) / np.maximum(peak, 1e-9)
 return {
  "return_pct": float((net_worths[-1] / net_worths[0]) - 1.0),
  "max_drawdown": float(drawdown.max()),
  "final_net_worth": float(net_worths[-1]),
  "n_trades": int(trades),
  "sharpe_like": _sharpe_like(net_worths),
  "win_rate": _win_rate(net_worths),
  "equity_curve": net_worths,
 }


def _sharpe_like(equity: np.ndarray) -> float:
 rets = np.diff(equity) / np.maximum(equity[:-1], 1e-9)
 if rets.std() < 1e-9:
  return 0.0
 return float(rets.mean() / rets.std() * np.sqrt(len(rets)))


def _win_rate(equity: np.ndarray) -> float:
 rets = np.diff(equity)
 if len(rets) == 0:
  return 0.0
 return float((rets > 0).sum() / len(rets))
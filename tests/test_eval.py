"""Smoke tests for the eval side (env, baselines, classifier, DQN)."""
import numpy as np
import pandas as pd
import pytest

from sim.market_sim import generate_regime_dataset
from sim.market_sim.simulator import CookieClickerMarket, N_MODES
from eval.env.trading_env import BUY, HOLD, SELL, TradingEnv
from eval.baselines.heuristics import (
 BuyAndHoldTrader,
 MeanReversionTrader,
 RandomTrader,
)
from eval.classifier.regime_classifier import (
 fit_one,
 make_logreg,
 make_mlp,
)
from eval.agents.dqn import DQNAgent, DQNConfig
from eval.agents.train import _build_obs, evaluate_dqn, train_dqn


# --- env ------------------------------------------------------------------

def test_env_reset_and_step() -> None:
 market = CookieClickerMarket(seed=0)
 env = TradingEnv(market=market, tmax=10)
 obs = env.reset()
 assert obs.shape == (1, 8)
 obs2, reward, done, info = env.step(HOLD)
 assert obs2.shape == (1, 8)
 assert isinstance(reward, float)
 assert isinstance(done, bool)
 assert "net_worth" in info


def test_env_buy_changes_portfolio() -> None:
 market = CookieClickerMarket(seed=0)
 env = TradingEnv(market=market, tmax=5)
 env.reset()
 env.step(BUY)
 assert env.portfolio.shares > 0


# --- baselines ------------------------------------------------------------

def test_random_trader_acts() -> None:
 market = CookieClickerMarket(seed=0)
 env = TradingEnv(market=market, tmax=50)
 env.reset()
 t = RandomTrader(seed=0)
 actions = [t.act(env._last_obs, env.portfolio_features()) for _ in range(50)]
 assert set(actions).issubset({HOLD, BUY, SELL})


def test_buy_and_hold_buys_once() -> None:
 market = CookieClickerMarket(seed=0)
 env = TradingEnv(market=market, tmax=20)
 env.reset()
 t = BuyAndHoldTrader()
 actions = [t.act(env._last_obs, env.portfolio_features()) for _ in range(20)]
 assert actions[0] == BUY
 assert all(a == HOLD for a in actions[1:])


def test_mean_reversion_trader_acts() -> None:
 market = CookieClickerMarket(seed=0)
 env = TradingEnv(market=market, tmax=200)
 env.reset()
 t = MeanReversionTrader()
 actions = [t.act(env._last_obs, env.portfolio_features()) for _ in range(200)]
 assert any(a == BUY for a in actions) or any(a == SELL for a in actions)


# --- classifier -----------------------------------------------------------

def test_classifier_beats_random(tmp_path) -> None:
 ds = tmp_path / "ds.parquet"
 generate_regime_dataset(n_ticks=1500, n_stocks=1, seed=0, out_path=ds)
 df = pd.read_parquet(ds)
 split = int(0.7 * len(df))
 df_train, df_test = df.iloc[:split], df.iloc[split:]
 for factory in (make_logreg, make_mlp):
  m = fit_one("m", factory(), df_train, df_test)
  assert m.accuracy > 1.0 / N_MODES, m.accuracy


# --- DQN ------------------------------------------------------------------

def test_dqn_obs_dimensions() -> None:
 market = CookieClickerMarket(seed=0)
 env = TradingEnv(market=market, tmax=10)
 env.reset()
 obs = _build_obs(env, regime_probs=None)
 assert obs.shape == (11,), obs.shape


def test_dqn_trains_one_episode() -> None:
 def factory():
  return TradingEnv(market=CookieClickerMarket(seed=0), tmax=50)
 agent = DQNAgent(DQNConfig(input_dim=11, warmup=10, eps_decay_steps=20), use_regime=False)
 returns = train_dqn(factory, agent, n_episodes=2, episode_length=50, verbose=False)
 assert len(returns) == 2
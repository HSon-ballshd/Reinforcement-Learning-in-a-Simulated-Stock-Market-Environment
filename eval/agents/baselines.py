"""
Heuristic trading baselines.

Three agents for comparison:
    RandomAgent     — random action each step.
    BuyAndHoldAgent — BUY at step 0, never sell.
    MeanReversionAgent — BUY when price < rolling mean, SELL when price > rolling mean.

All agents implement the same interface:
    select_action(observation, info) -> int
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod

from eval.env.trading_env import TradingEnv


# ------------------------------------------------------------------
# Base class
# ------------------------------------------------------------------
class BaseAgent(ABC):
    """Minimal agent interface matching TradingEnv.action_space."""

    @abstractmethod
    def select_action(self, observation: np.ndarray, info: dict) -> int:
        """
        Args:
            observation: shape (8,) from TradingEnv._get_obs().
            info:       dict from TradingEnv.step(), includes 'price', 'holdings', etc.

        Returns:
            Action: TradingEnv.HOLD (0), BUY (1), or SELL (2).
        """
        ...

    def reset(self) -> None:
        """Called at the start of each episode. Override if needed."""
        pass


# ------------------------------------------------------------------
# RandomAgent
# ------------------------------------------------------------------
class RandomAgent(BaseAgent):
    """Uniform-random action each step."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def select_action(self, observation: np.ndarray, info: dict) -> int:
        return int(self._rng.integers(0, 3))


# ------------------------------------------------------------------
# BuyAndHoldAgent
# ------------------------------------------------------------------
class BuyAndHoldAgent(BaseAgent):
    """
    BUY on the first step, HOLD forever after.

    Represents a naive "index fund" baseline.
    """

    def __init__(self) -> None:
        self._bought = False

    def reset(self) -> None:
        self._bought = False

    def select_action(self, observation: np.ndarray, info: dict) -> int:
        if not self._bought:
            self._bought = True
            return TradingEnv.BUY
        return TradingEnv.HOLD


# ------------------------------------------------------------------
# MeanReversionAgent
# ------------------------------------------------------------------
class MeanReversionAgent(BaseAgent):
    """
    Mean-reversion strategy:
        BUY  when current price < rolling_mean_5
        SELL when current price > rolling_mean_5 * (1 + threshold)
        HOLD otherwise

    Features (from observation, index 0 = price, 4 = rolling_mean_5):
        obs[0] = price
        obs[4] = rolling_mean_5
    """

    def __init__(self, threshold: float = 0.01, seed: int | None = None) -> None:
        """
        Args:
            threshold: price must be this fraction below rolling mean to BUY,
                      and above mean*(1+threshold) to SELL. Default 1 %.
        """
        self.threshold = threshold
        self._rng      = np.random.default_rng(seed)

    def select_action(self, observation: np.ndarray, info: dict) -> int:
        price       = observation[0]
        rolling_avg = observation[4]

        if not np.isfinite(price) or not np.isfinite(rolling_avg) or rolling_avg == 0:
            return TradingEnv.HOLD

        lo = rolling_avg * (1.0 - self.threshold)
        hi = rolling_avg * (1.0 + self.threshold)

        holdings = info.get('holdings', 0)

        if price <= lo and holdings <= 0:
            # Only BUY if price is cheap AND we don't already hold shares.
            # Without the holdings check, MeanReversion repeatedly BUY-attempts
            # every tick while holding, wasting compute on 0-share orders.
            return TradingEnv.BUY
        elif price >= hi and holdings > 0:
            return TradingEnv.SELL
        else:
            return TradingEnv.HOLD


# ------------------------------------------------------------------
# Convenience: run one agent over an episode and return total return
# ------------------------------------------------------------------
def evaluate_agent(
    agent: BaseAgent,
    market_seed: int = 42,
    n_steps: int = 1000,
    initial_cash: float = 10_000.0,
) -> dict:
    """
    Run a single episode and collect metrics.

    Args:
        agent:       Instance of a BaseAgent subclass.
        market_seed: Seed for the CookieClickerMarket RNG.
        n_steps:     Episode length.
        initial_cash: Starting cash.

    Returns:
        dict with keys: total_return_pct, final_value, n_buys, n_sells, n_holds
    """
    from sim.market_sim import CookieClickerMarket
    from eval.env.trading_env import TradingEnv

    market = CookieClickerMarket(n_stocks=1, seed=market_seed)
    env    = TradingEnv(
        market,
        initial_cash=initial_cash,
        max_steps=n_steps,
        seed=market_seed,
    )

    obs   = env.reset()
    agent.reset()

    # Allow agents that need env access (e.g. RegimeAwareDQNAgent) to reach it
    if hasattr(agent, '_env') or hasattr(agent, 'set_env'):
        agent._env = env

    n_buys = n_sells = n_holds = 0
    info = {
        'portfolio_value': env._portfolio_value(),
        'cash': env.cash,
        'holdings': env.holdings,
        'price': env.market.stocks[0]['price'],
        'step': 0,
    }

    for _ in range(n_steps):
        action = agent.select_action(obs, info)
        obs, _, done, info = env.step(action)

        if action == TradingEnv.BUY:
            n_buys += 1
        elif action == TradingEnv.SELL:
            n_sells += 1
        else:
            n_holds += 1

        if done:
            break

    final_value     = env._portfolio_value()
    total_return   = (final_value - initial_cash) / initial_cash * 100.0

    return {
        'total_return_pct': total_return,
        'final_value':      final_value,
        'n_buys':           n_buys,
        'n_sells':          n_sells,
        'n_holds':          n_holds,
    }


def compare_baselines(
    seeds: list[int] | None = None,
    n_steps: int = 1000,
    initial_cash: float = 10_000.0,
) -> dict:
    """
    Run all three baselines across multiple seeds and return summary stats.

    Returns:
        dict of {agent_name: {mean_return, std_return, ...}}
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024]

    agents = {
        'Random':          RandomAgent(),
        'BuyAndHold':      BuyAndHoldAgent(),
        'MeanReversion':   MeanReversionAgent(threshold=0.05),
    }

    results = {name: [] for name in agents}

    for seed in seeds:
        for name, agent in agents.items():
            r = evaluate_agent(
                agent,
                market_seed=seed,
                n_steps=n_steps,
                initial_cash=initial_cash,
            )
            results[name].append(r)

    summary = {}
    for name, runs in results.items():
        returns = [r['total_return_pct'] for r in runs]
        summary[name] = {
            'mean_return_pct': float(np.mean(returns)),
            'std_return_pct':  float(np.std(returns)),
            'min_return_pct':  float(np.min(returns)),
            'max_return_pct':  float(np.max(returns)),
            'n_runs':          len(runs),
        }

    return summary

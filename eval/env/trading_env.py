"""
TradingEnv — Gym-style environment wrapping CookieClickerMarket.

Implements CONTRACT.md §3:
    Actions: 0=hold, 1=buy, 2=sell
    Reward:  portfolio delta − transaction-cost penalty
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

from sim.market_sim import CookieClickerMarket


class TradingEnv:
    """
    Single-stock trading environment.

    State tracks cash + holdings so the agent can accumulate or draw down
    a portfolio.  Observations are the market's observable features
    (no regime disclosure).
    """

    # ------------------------------------------------------------------
    # Action space
    # ------------------------------------------------------------------
    HOLD = 0
    BUY  = 1
    SELL  = 2

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def __init__(
        self,
        market: CookieClickerMarket,
        initial_cash: float = 10_000.0,
        max_position_frac: float = 1.0,
        transaction_cost_pct: float = 0.001,
        max_steps: int | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            market:       An initialised CookieClickerMarket instance.
            initial_cash: Starting cash (default 10 000).
            max_position_frac: Maximum fraction of portfolio in the stock (0–1).
            transaction_cost_pct: Flat % cost per trade (default 0.1 %).
            max_steps:   Episode length cap; None = run forever (or until
                          the market simulation ends).
            seed:         RNG seed for any stochastic elements.
        """
        self.market               = market
        self.initial_cash         = initial_cash
        self.max_position_frac    = max_position_frac
        self.transaction_cost_pct = transaction_cost_pct
        self.max_steps            = max_steps
        self._rng                 = np.random.default_rng(seed)

        # Portfolio state (reset on each episode)
        self.cash:      float = 0.0
        self.holdings:   float = 0.0   # shares
        self.total_value: float = 0.0

        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        """Begin a new episode.  Returns the initial observation."""
        self.market.reset()

        self.cash       = self.initial_cash
        self.holdings   = 0.0
        self._step_count = 0

        # Compute initial portfolio value (price after warm-up ticks)
        price       = self.market.stocks[0]['price']
        self.total_value = self.cash + self.holdings * price

        return self._get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one action.

        Returns (obs, reward, done, info).
        """
        self._step_count += 1

        prev_value = self._portfolio_value()

        # Execute action
        self._execute(action)

        # Advance the market by one tick
        self.market.tick()

        # New portfolio value
        curr_value = self._portfolio_value()
        reward     = curr_value - prev_value

        # Transaction-cost penalty
        if action in (self.BUY, self.SELL):
            cost = self.transaction_cost_pct * curr_value
            reward -= cost

        # Normalise by CURRENT portfolio value to keep rewards bounded
        # (dividing by initial_cash lets returns explode in bull markets)
        denom = curr_value if curr_value > 0 else self.initial_cash
        reward = reward / denom

        # Clip to [-1, 1] to prevent gradient explosions from extreme steps
        reward = float(np.clip(reward, -1.0, 1.0))

        obs  = self._get_obs()
        done = self._is_done()
        info = {
            'portfolio_value': curr_value,
            'cash':            self.cash,
            'holdings':        self.holdings,
            'price':           self.market.stocks[0]['price'],
            'action':          action,
            'step':            self._step_count,
        }

        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Render (no-op for now — can be extended with Matplotlib)
    # ------------------------------------------------------------------
    def render(self, mode: str = 'human') -> None:
        """Stub renderer.  Satisfies the CONTRACT.md interface."""
        pass

    # ------------------------------------------------------------------
    # Extended features (for regime classifier — not exposed to agent)
    # ------------------------------------------------------------------
    def _get_extended_features(self) -> np.ndarray:
        """
        Return the 6 engineered features needed by the regime classifier.

        These are NOT part of the agent's observation — they are only used
        by the harness classify() wrapper injected into RegimeAwareDQNAgent.
        """
        stock = self.market.stocks[0]
        vals  = stock['vals']
        price = stock['price']

        # --- match dataset.py feature computations ---
        # return_1, return_5, return_20, rolling_mean_5/20, rolling_std_20, momentum_5_20
        # are already available from the market's get_observation but we recompute
        # from vals to keep the extended feature set self-consistent.
        return_1   = (vals[0] - vals[1]) / vals[1] if len(vals) > 1 else 0.0
        return_5   = (vals[0] - vals[4]) / vals[4] if len(vals) > 4 else 0.0
        return_20  = (vals[0] - vals[19]) / vals[19] if len(vals) > 19 else 0.0
        mean_20    = float(np.mean(vals[:20]))  if len(vals) >= 20 else float(np.mean(vals))
        std_20     = float(np.std(vals[:20]))   if len(vals) >= 20 else float(np.std(vals))

        drift_proxy = 0.6 * return_1 + 0.4 * return_5

        if len(vals) >= 5:
            std_5 = float(np.std(
                [(vals[i] - vals[i+1]) / (vals[i+1] + 1e-8)
                 for i in range(min(4, len(vals)-1))]
            ))
            vol_ratio = std_5 / (std_20 + 1e-8)
        else:
            vol_ratio = 0.0

        mean_reversion_signal = (
            (price - mean_20) / (std_20 + 1e-8) if std_20 > 1e-8 else 0.0
        )

        if len(vals) >= 3:
            tick_rets = [(vals[i] - vals[i+1]) / (vals[i+1] + 1e-8)
                         for i in range(min(5, len(vals)-1))]
            directional_consistency = sum(1 for r in tick_rets if r > 0) / len(tick_rets)
        else:
            directional_consistency = 0.5

        if len(vals) >= 5:
            tret = [(vals[i] - vals[i+1]) / (vals[i+1] + 1e-8)
                    for i in range(min(19, len(vals)-1))]
            sharpe_proxy = float(np.mean(tret)) / (float(np.std(tret)) + 1e-8)
        else:
            sharpe_proxy = 0.0

        momentum_divergence = 1.0 if return_1 * return_20 < 0 else 0.0

        return np.array([
            drift_proxy,
            vol_ratio,
            mean_reversion_signal,
            directional_consistency,
            sharpe_proxy,
            momentum_divergence,
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        """Return the market observation (reveal=False — agent cannot see mode)."""
        obs = self.market.get_observation(reveal=False)   # shape (1, 8)
        return obs.squeeze(0).astype(np.float32)          # shape (8,)

    def _portfolio_value(self) -> float:
        price = self.market.stocks[0]['price']
        return self.cash + self.holdings * price

    def _execute(self, action: int) -> None:
        """Apply BUY or SELL; HOLD is a no-op."""
        price = self.market.stocks[0]['price']
        if price <= 0:
            return  # Defensive: avoid divide-by-zero

        if action == self.BUY:
            # Invest up to max_position_frac of current portfolio value
            max_invest   = self._portfolio_value() * self.max_position_frac
            invest_amount = min(self.cash, max_invest)
            if invest_amount > 0:
                shares        = invest_amount / price
                self.holdings += shares
                self.cash    -= invest_amount

        elif action == self.SELL:
            # Sell up to max_position_frac of current holdings
            max_sell    = self.holdings * self.max_position_frac
            sell_shares = min(self.holdings, max_sell)
            if sell_shares > 0:
                proceeds      = sell_shares * price
                self.holdings -= sell_shares
                self.cash    += proceeds

    def _is_done(self) -> bool:
        if self.max_steps is not None and self._step_count >= self.max_steps:
            return True
        # Safety floor: if cash goes negative the episode is invalid
        if self.cash < 0:
            return True
        return False

    # ------------------------------------------------------------------
    # Convenience properties matching Gym convention
    # ------------------------------------------------------------------
    @property
    def observation_space_shape(self) -> tuple:
        return (8,)   # matches CookieClickerMarket.get_observation(reveal=False)

    @property
    def action_space_n(self) -> int:
        return 3      # HOLD, BUY, SELL

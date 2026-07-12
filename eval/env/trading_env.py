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

        # Transaction-cost penalty — deducted from both reward AND cash
        if action in (self.BUY, self.SELL):
            cost = self.transaction_cost_pct * curr_value
            reward -= cost
            self.cash -= cost
            self.cash = max(self.cash, 0.0)  # clamp tiny negative floats from float rounding

        # Normalise by initial cash so reward is always a small fraction (e.g. 0.01 = 1%
        # of portfolio per tick), keeping it bounded regardless of compounding gains.
        reward = reward / self.initial_cash

        obs  = self._get_obs()
        done = self._is_done()
        info = {
            'portfolio_value': curr_value,
            'cash':            self.cash,
            'holdings':        self.holdings,
            'price':           self.market.stocks[0]['price'],
            'action':          action,
            'step':            self._step_count,
            'transaction_cost': cost if action in (self.BUY, self.SELL) else 0.0,
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
        Return the 11 extended features needed by the regime classifier.

        These are NOT part of the agent's 8-dim observation — they are only used
        by the harness classify() wrapper injected into RegimeAwareDQNAgent.
        """
        stock = self.market.stocks[0]
        vals  = stock['vals']
        price = stock['price']

        # Tick-return list (i=0 is most recent)
        tick_rets = [
            (vals[i] - vals[i+1]) / (vals[i+1] + 1e-8)
            for i in range(min(len(vals) - 1, 64))
        ]
        def trets(n): return tick_rets[:n]

        # Rolling volatility
        rstd_5  = float(np.std(trets(5)))  if len(tick_rets) >= 5  else 0.0
        rstd_20 = float(np.std(trets(20))) if len(tick_rets) >= 20 else 0.0
        rstd_ratio = rstd_5 / (rstd_20 + 1e-8)

        # Rolling mean for mean-reversion z-score
        mean_20 = float(np.mean(vals[:20])) if len(vals) >= 20 else float(np.mean(vals))

        # Mean-reversion z-score
        mean_rev_z = (price - mean_20) / (rstd_20 * mean_20 + 1e-8) if rstd_20 > 1e-8 else 0.0

        # Directional consistency
        def dir_cons(n):
            t = trets(n)
            return float(sum(1 for r in t if r > 0) / len(t)) if t else 0.5
        dir_5  = dir_cons(5)
        dir_20 = dir_cons(20)

        # Drift estimate (sign-weighted mean return)
        t5 = trets(5)
        drift_est_5 = float(np.mean([abs(r) * (1 if r > 0 else -1) for r in t5])) if t5 else 0.0

        # Jump counts
        def jump_count(n, rstd):
            return float(sum(1 for r in trets(n) if abs(r) > rstd)) if rstd > 1e-8 else 0.0
        jc_5  = jump_count(5,  rstd_5)
        jc_20 = jump_count(20, rstd_20)

        # Max tick return in last 5
        max_ret_5 = float(max((abs(r) for r in trets(5)), default=0.0))

        # Trend strength
        ret_5  = trets(5)[0]  if len(tick_rets) >= 5  else 0.0
        ret_20 = trets(20)[0] if len(tick_rets) >= 20 else 0.0
        trend_5  = float(ret_5  / (rstd_5  + 1e-8)) if rstd_5  > 1e-8 else 0.0
        trend_20 = float(ret_20 / (rstd_20 + 1e-8)) if rstd_20 > 1e-8 else 0.0

        # Momentum divergence
        mom_div = 1.0 if ret_5 * ret_20 < 0 else 0.0

        # Vol regime
        vol_reg = rstd_5 / (rstd_20 + 1e-8)

        return np.array([
            rstd_5, rstd_20, rstd_ratio, mean_rev_z,
            dir_5, dir_20, drift_est_5,
            jc_5, jc_20, max_ret_5,
            trend_5, trend_20, mom_div, vol_reg,
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
        # Safety floor: if cash goes genuinely negative (not just float rounding), episode is invalid
        if self.cash < -1e-8:
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

"""Heuristic baseline traders (Exp 1: Random, Buy-and-Hold, Mean-Reversion)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from eval.env.trading_env import BUY, HOLD, SELL


class Trader(Protocol):
 def act(self, obs: np.ndarray, portfolio_state: np.ndarray) -> int: ...
 def reset(self) -> None: ...


@dataclass
class RandomTrader:
 seed: int = 0
 _buy_prob: float = 1.0 / 3.0
 _sell_prob: float = 1.0 / 3.0
 _rng: np.random.Generator = None

 def __post_init__(self) -> None:
  if self._rng is None:
   self._rng = np.random.default_rng(self.seed)

 def reset(self) -> None:
  pass

 def act(self, obs: np.ndarray, portfolio_state: np.ndarray) -> int:
  r = float(self._rng.random())
  if r < self._buy_prob:
   return BUY
  if r < self._buy_prob + self._sell_prob:
   return SELL
  return HOLD


@dataclass
class BuyAndHoldTrader:
 """Buy on the first call, then hold forever."""
 _bought: bool = False

 def reset(self) -> None:
  self._bought = False

 def act(self, obs: np.ndarray, portfolio_state: np.ndarray) -> int:
  if not self._bought:
   self._bought = True
   return BUY
  return HOLD


@dataclass
class MeanReversionTrader:
 """Buy when short-term return is sharply negative, sell when positive.
 Threshold tuned so a typical Chaotic market triggers a few trades per 1k ticks.
 """
 threshold: float = 0.04
 _cooldown: int = 5
 _steps_since_trade: int = 0

 def reset(self) -> None:
  self._steps_since_trade = self._cooldown

 def act(self, obs: np.ndarray, portfolio_state: np.ndarray) -> int:
  # obs layout per CONTRACT: [price, return_1, return_5, return_20, ...]
  ret_5 = float(obs[0, 2]) if obs.ndim == 2 else float(obs[2])
  if self._steps_since_trade < self._cooldown:
   self._steps_since_trade += 1
   return HOLD
  if ret_5 < -self.threshold:
   self._steps_since_trade = 0
   return BUY
  if ret_5 > self.threshold:
   self._steps_since_trade = 0
   return SELL
  return HOLD
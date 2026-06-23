"""Cookie Clicker stock-market simulator (Python port).

Faithful port of `minigameMarket.js` (lines 763-877) with `dragonBoost = 0`.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

STABLE = 0
BULLISH = 1
BEARISH = 2
STRONG_BULL = 3
STRONG_BEAR = 4
CHAOTIC = 5
N_MODES = 6

_MODE_WEIGHTS = np.array([0, 1, 1, 2, 2, 3, 4, 5], dtype=np.float64)
_POST_RESET_TICKS = 15
_VAL_HISTORY_LEN = 64


@dataclass
class _StockState:
 stock_id: int
 val: float
 d: float
 mode: int
 dur: int
 vals: deque = field(default_factory=lambda: deque(maxlen=_VAL_HISTORY_LEN))

 def resting_val(self, bank_level: int) -> float:
  return 10.0 + 10.0 * self.stock_id + (bank_level - 1)


class CookieClickerMarket:
 def __init__(
  self,
  n_stocks: int = 1,
  bank_level: int = 1,
  seed: Optional[int] = None,
  seconds_per_tick: int = 60,
 ) -> None:
  if n_stocks < 1:
   raise ValueError("n_stocks must be >= 1")
  self.n_stocks = n_stocks
  self.bank_level = bank_level
  self.seconds_per_tick = seconds_per_tick
  self.rng = np.random.default_rng(seed)
  self._stocks: list[_StockState] = []
  self._ticks_run: int = 0
  self._records: list[dict] = []
  self.reset()

 def reset(self) -> None:
  self._stocks = []
  for i in range(self.n_stocks):
   mode = self._choose_mode()
   dur = int(np.floor(10 + self.rng.random() * 690))
   resting = 10.0 + 10.0 * i + (self.bank_level - 1)
   d = self.rng.random() * 0.2 - 0.1
   stock = _StockState(
    stock_id=i,
    val=resting,
    d=d,
    mode=mode,
    dur=dur,
    vals=deque([resting], maxlen=_VAL_HISTORY_LEN),
   )
   self._stocks.append(stock)
  self._ticks_run = 0
  self._records = []
  for _ in range(_POST_RESET_TICKS):
   self._tick_once(record=False)

 def tick(self) -> None:
  self._ticks_run += 1
  self._tick_once(record=True)

 def _tick_once(self, record: bool) -> None:
  globD = 0.0
  globP = float(self.rng.random())
  if self.rng.random() < 0.1:
   globD = float((self.rng.random() - 0.5) * 2)
  for stock in self._stocks:
   stock.last = 0
   stock.d *= 0.97
   r1 = float(self.rng.random())
   r2 = float(self.rng.random())
   if stock.mode == STABLE:
    stock.d *= 0.95
    stock.d += 0.05 * (r1 - 0.5)
   elif stock.mode == BULLISH:
    stock.d *= 0.99
    stock.d += 0.05 * (r1 - 0.1)
   elif stock.mode == BEARISH:
    stock.d *= 0.99
    stock.d -= 0.05 * (r1 - 0.1)
   elif stock.mode == STRONG_BULL:
    stock.d += 0.15 * (r1 - 0.1)
    stock.val += r2 * 5
   elif stock.mode == STRONG_BEAR:
    stock.d -= 0.15 * (r1 - 0.1)
    stock.val -= r2 * 5
   elif stock.mode == CHAOTIC:
    stock.d += 0.3 * (r1 - 0.5)
   resting = stock.resting_val(self.bank_level)
   stock.val += (resting - stock.val) * 0.01
   if globD != 0 and self.rng.random() < globP:
    stock.val -= (1 + stock.d * (self.rng.random() ** 3) * 7) * globD
    stock.val -= globD * (1 + self.rng.random() ** 3 * 7)
    stock.d += globD * (1 + self.rng.random() * 4)
    stock.dur = 0
   stock.val += ((self.rng.random() - 0.5) * 2) ** 11 * 3
   stock.d += 0.1 * (self.rng.random() - 0.5)
   if self.rng.random() < 0.15:
    stock.val += (self.rng.random() - 0.5) * 3
   if self.rng.random() < 0.03:
    stock.val += (self.rng.random() - 0.5) * 10
   if self.rng.random() < 0.1:
    stock.d += (self.rng.random() - 0.5) * 0.3
   if stock.mode == CHAOTIC:
    if self.rng.random() < 0.5:
     stock.val += (self.rng.random() - 0.5) * 10
    if self.rng.random() < 0.2:
     stock.d = (self.rng.random() - 0.5) * 2
   if stock.mode == STRONG_BULL:
    if self.rng.random() < 0.3:
     stock.d += (self.rng.random() - 0.5) * 0.1
     stock.val += (self.rng.random() - 0.7) * 10
    if self.rng.random() < 0.03:
     stock.mode = STRONG_BEAR
   if stock.mode == STRONG_BEAR:
    if self.rng.random() < 0.3:
     stock.d += (self.rng.random() - 0.5) * 0.1
     stock.val += (self.rng.random() - 0.3) * 10
   ceiling = 100 + (self.bank_level - 1) * 3
   if stock.val > ceiling and stock.d > 0:
    stock.d *= 0.9
   stock.val += stock.d
   if stock.val < 5:
    stock.val += (5 - stock.val) * 0.5
    if stock.d < 0:
     stock.d *= 0.95
   stock.val = max(stock.val, 1.0)
   stock.vals.append(stock.val)
   stock.dur -= 1
   if stock.dur <= 0:
    stock.dur = int(np.floor(10 + self.rng.random() * 690))
    stock.mode = self._choose_mode()
   if record:
    self._records.append({
     "tick": self._ticks_run,
     "stock_id": stock.stock_id,
     "price": stock.val,
     "mode": stock.mode,
     "d": stock.d,
     "dur": stock.dur,
    })

 def _choose_mode(self) -> int:
  # JS Game.choose picks uniformly from the 8 entries of [0,1,1,2,2,3,4,5].
  return int(self.rng.choice(_MODE_WEIGHTS))

 def history(self) -> pd.DataFrame:
  return pd.DataFrame(
   self._records, columns=["tick", "stock_id", "price", "mode", "d", "dur"]
  )

 def get_observation(self, *, reveal: bool = False) -> np.ndarray:
  rows = [self._features(stock, reveal=reveal) for stock in self._stocks]
  return np.asarray(rows, dtype=np.float64)

 @staticmethod
 def _features(stock: _StockState, *, reveal: bool) -> list[float]:
  vals = list(stock.vals)
  n = len(vals)
  price = vals[0]

  def safe_div(a: float, b: float) -> float:
   return float(a) / float(b) if b != 0 else 0.0

  def ret(k: int) -> float:
   if n <= k:
    return 0.0
   return safe_div(vals[0] - vals[k], vals[k])

  def rmean(k: int) -> float:
   window = vals[: min(k, n)]
   return float(np.mean(window)) if window else 0.0

  def rstd(k: int) -> float:
   window = vals[: min(k, n)]
   return float(np.std(window, ddof=0)) if len(window) >= 2 else 0.0

  momentum = safe_div(rmean(5) - rmean(20), rmean(20)) if rmean(20) else 0.0
  feats = [price, ret(1), ret(5), ret(20), rmean(5), rmean(20), rstd(20), momentum]
  if reveal:
   one_hot = [0.0] * N_MODES
   one_hot[stock.mode] = 1.0
   feats.extend(one_hot)
  return feats

 @property
 def current_prices(self) -> np.ndarray:
  return np.asarray([s.val for s in self._stocks], dtype=np.float64)

 @property
 def current_modes(self) -> np.ndarray:
  return np.asarray([s.mode for s in self._stocks], dtype=np.int64)

 @property
 def ticks(self) -> int:
  return self._ticks_run
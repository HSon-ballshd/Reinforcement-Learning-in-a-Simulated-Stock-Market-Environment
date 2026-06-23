"""Regime dataset generation (CONTRACT.md Sec.2).

Runs the simulator for ``n_ticks`` and dumps observable features + the
hidden ``mode`` label to a parquet file. The dataset is the ONLY allowed
source of labels for training the regime classifier.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from .simulator import CookieClickerMarket


def generate_regime_dataset(
 n_ticks: int = 5000,
 n_stocks: int = 1,
 seed: int = 0,
 out_path: Union[str, Path] = "data/regime_dataset.parquet",
) -> Path:
 """Run the simulator and write (X, y) pairs to ``out_path``.

 The output parquet has columns matching CONTRACT.md Sec.2:

 [tick, stock_id, price, return_1, return_5, return_20,
  rolling_mean_5, rolling_mean_20, rolling_std_20, momentum_5_20,
  mode] # mode is the label y
 """
 out_path = Path(out_path)
 out_path.parent.mkdir(parents=True, exist_ok=True)

 market = CookieClickerMarket(n_stocks=n_stocks, seed=seed)
 for _ in range(n_ticks):
  market.tick()

 df = market.history()
 # Reorder + add feature columns matching the contract's schema.
 df = df.copy()
 df["return_1"] = 0.0
 df["return_5"] = 0.0
 df["return_20"] = 0.0
 df["rolling_mean_5"] = 0.0
 df["rolling_mean_20"] = 0.0
 df["rolling_std_20"] = 0.0
 df["momentum_5_20"] = 0.0

 for stock_id in range(n_stocks):
  sub = df[df["stock_id"] == stock_id].sort_values("tick").reset_index(drop=True)
  vals = sub["price"].astype(float).tolist()

  def ret(k: int) -> list[float]:
   out = []
   for i in range(len(vals)):
    if i + k >= len(vals):
     out.append(0.0)
    else:
     p, p_prev = vals[i], vals[i + k]
     out.append((p - p_prev) / p_prev if p_prev else 0.0)
   return out

  def rmean(k: int) -> list[float]:
   out = []
   csum = 0.0
   for i, v in enumerate(vals):
    csum += v
    if i >= k:
     csum -= vals[i - k]
    denom = min(k + 1, i + 1)
    out.append(csum / denom)
   return out

  def rstd(k: int) -> list[float]:
   out = []
   for i in range(len(vals)):
    win = vals[max(0, i - k) : i + 1]
    if len(win) < 2:
     out.append(0.0)
    else:
     m = sum(win) / len(win)
     out.append((sum((x - m) ** 2 for x in win) / len(win)) ** 0.5)
   return out

  m5 = rmean(5)
  m20 = rmean(20)
  sub["return_1"] = ret(1)
  sub["return_5"] = ret(5)
  sub["return_20"] = ret(20)
  sub["rolling_mean_5"] = m5
  sub["rolling_mean_20"] = m20
  sub["rolling_std_20"] = rstd(20)
  sub["momentum_5_20"] = [
   ((m5[i] - m20[i]) / m20[i]) if m20[i] else 0.0 for i in range(len(vals))
  ]

  df.loc[sub.index, sub.columns] = sub.values

 cols = [
  "tick",
  "stock_id",
  "price",
  "return_1",
  "return_5",
  "return_20",
  "rolling_mean_5",
  "rolling_mean_20",
  "rolling_std_20",
  "momentum_5_20",
  "mode",
 ]
 df = df[cols]
 df.to_parquet(out_path, index=False)
 return out_path
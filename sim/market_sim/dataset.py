"""
Dataset generation for regime classification.

Generates a parquet file with (X, y) pairs where X is observable features
and y is the hidden regime (mode).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union
from .simulator import CookieClickerMarket


def generate_regime_dataset(
    n_ticks: int = 5000,
    n_stocks: int = 1,
    seed: int = 0,
    out_path: Union[str, Path] = "data/regime_dataset.parquet",
) -> Path:
    """
    Run the simulator for n_ticks and generate a parquet dataset.

    Each row represents one tick for one stock, with observable features
    and the ground-truth regime (mode) as the label.

    Args:
        n_ticks: Number of ticks to simulate. Default 5000.
        n_stocks: Number of stocks. Default 1 (per SPEC.md v1).
        seed: RNG seed for reproducibility. Default 0.
        out_path: Output parquet file path. Default "data/regime_dataset.parquet".

    Returns:
        Path object pointing to the created parquet file.

    Output columns (13 features + mode label):
        [tick, stock_id, price,
         return_1, return_5, return_20,
         rolling_mean_5, rolling_mean_20, rolling_std_20, momentum_5_20,
         drift_proxy, vol_ratio, mean_reversion_signal,
         directional_consistency, sharpe_proxy, momentum_divergence,
         mode]                  # mode is the label (y)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Create simulator
    market = CookieClickerMarket(
        n_stocks=n_stocks,
        bank_level=1,
        seed=seed,
    )

    records = []

    # Run simulation
    for tick_idx in range(n_ticks):
        # Market history for this tick
        for stock_id in range(n_stocks):
            stock = market.stocks[stock_id]
            vals = stock['vals']  # vals[0] = current price, vals[1] = 1-tick ago, ...
            rolling_mean_20 = float(np.mean(vals[:20])) if len(vals) >= 20 else float(np.mean(vals))

            price = stock['price']

            # ---- Compute tick-return list for all windowed features ----
            # tick_returns[i] = (vals[i] - vals[i+1]) / vals[i+1]
            # i=0 is most recent tick
            tick_rets_all = [
                (vals[i] - vals[i+1]) / (vals[i+1] + 1e-8)
                for i in range(min(len(vals) - 1, 64))
            ]
            def trets(n):  return tick_rets_all[:n]
            def trets_up_to(n): return tick_rets_all[:min(n, len(tick_rets_all))]

            # Multi-window returns
            ret_1   = float(tick_rets_all[0])  if len(tick_rets_all) >= 1 else 0.0
            ret_5   = float((vals[0] - vals[4]) / (vals[4] + 1e-8)) if len(vals) > 4 else 0.0
            ret_10  = float((vals[0] - vals[9]) / (vals[9] + 1e-8)) if len(vals) > 9 else 0.0
            ret_20  = float((vals[0] - vals[19]) / (vals[19] + 1e-8)) if len(vals) > 19 else 0.0

            # Rolling volatility (std of tick-returns)
            rstd_5  = float(np.std(trets(5)))   if len(tick_rets_all) >= 5  else 0.0
            rstd_20 = float(np.std(trets(20)))  if len(tick_rets_all) >= 20 else 0.0
            rstd_ratio = rstd_5 / (rstd_20 + 1e-8)

            # Mean-reversion z-score: how far is price from 20-tick mean?
            mean_rev_z = (price - rolling_mean_20) / (rstd_20 * rolling_mean_20 + 1e-8) if rstd_20 > 1e-8 else 0.0

            # Directional consistency (% up-ticks)
            def dir_cons(n):
                t = trets(n)
                return float(sum(1 for r in t if r > 0) / len(t)) if t else 0.5
            dir_5  = dir_cons(5)
            dir_20 = dir_cons(20)

            # Drift estimate: sign-weighted mean return (amplifies big moves)
            t5 = trets(5)
            drift_est_5 = float(np.mean([abs(r) * (1 if r > 0 else -1) for r in t5])) if t5 else 0.0

            # Jump detection: count of |tick_return| > 1 * rolling_std
            def jump_count(n, rstd):
                return float(sum(1 for r in trets(n) if abs(r) > rstd)) if rstd > 1e-8 else 0.0
            jc_5  = jump_count(5,  rstd_5)
            jc_20 = jump_count(20, rstd_20)

            # Max tick return in last 5 — Strong Bull/Bear have big spikes
            max_ret_5 = float(max((abs(r) for r in trets(5)), default=0.0))

            # Trend strength (return / volatility — like Sharpe, but no mean subtraction)
            def trend_str(ret, rstd):
                return float(ret / (rstd + 1e-8)) if rstd > 1e-8 else 0.0
            trend_5  = trend_str(ret_5,  rstd_5)
            trend_20 = trend_str(ret_20, rstd_20)

            # Momentum divergence: short and long returns disagree in sign
            mom_div = float(ret_1 * ret_20 < 0)

            # Vol regime: recent vol vs long vol
            vol_reg = rstd_5 / (rstd_20 + 1e-8)

            # Mode is stored in the stock dict
            mode = stock['mode']

            records.append({
                'tick': market.tick_count - 1,
                'stock_id': stock_id,
                'price': float(price),
                'return_1':   ret_1,
                'return_5':   ret_5,
                'return_10':  ret_10,
                'return_20':  ret_20,
                'rolling_std_5':   rstd_5,
                'rolling_std_20':  rstd_20,
                'rolling_std_ratio': rstd_ratio,
                'mean_reversion_z': mean_rev_z,
                'directional_consistency_5':  dir_5,
                'directional_consistency_20': dir_20,
                'drift_estimate_5':   drift_est_5,
                'jump_count_5':   jc_5,
                'jump_count_20':  jc_20,
                'max_tick_return_5': max_ret_5,
                'trend_strength_5':  trend_5,
                'trend_strength_20': trend_20,
                'momentum_divergence': mom_div,
                'vol_regime_5':   vol_reg,
                'mode': int(mode),
            })

        # Advance market
        market.tick()

    # Create DataFrame and save
    df = pd.DataFrame(records)
    df.to_parquet(out_path, index=False, engine='pyarrow')

    return out_path

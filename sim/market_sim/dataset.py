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
        # Get observations (including revealed mode)
        obs_revealed = market.get_observation(reveal=True)

        # Market history for this tick
        for stock_id in range(n_stocks):
            stock = market.stocks[stock_id]
            vals = stock['vals']  # vals[0] = current price, vals[1] = 1-tick ago, ...

            # Extract base features from observation
            obs_features = obs_revealed[stock_id]
            price = obs_features[0]
            return_1 = obs_features[1]
            return_5 = obs_features[2]
            return_20 = obs_features[3]
            rolling_mean_5 = obs_features[4]
            rolling_mean_20 = obs_features[5]
            rolling_std_20 = obs_features[6]
            momentum_5_20 = obs_features[7]

            # ---- New regime-discriminating features ----

            # drift_proxy: EMA return (α=0.6) as proxy for hidden drift d.
            # Strong Bull has strong positive drift, Strong Bear strong negative,
            # Stable ~0, others intermediate.
            if len(vals) >= 3:
                drift_proxy = 0.6 * return_1 + 0.4 * return_5
            else:
                drift_proxy = 0.0

            # vol_ratio: short-term / long-term volatility.
            # Chaotic has high vol at all timeframes (ratio ~1).
            # Stable has low vol (ratio < 1 as mean reversion dominates).
            if len(vals) >= 5:
                std_5  = np.std([(vals[i] - vals[i+1]) / (vals[i+1] + 1e-8)
                                 for i in range(min(4, len(vals)-1))])
                vol_ratio = std_5 / (rolling_std_20 + 1e-8)
            else:
                vol_ratio = 0.0

            # mean_reversion_signal: z-score of price vs long-term mean.
            # Trending regimes (Strong Bull/Bear) are far from mean;
            # Stable oscillates around it.
            mean_reversion_signal = (
                (price - rolling_mean_20) / (rolling_std_20 + 1e-8)
                if rolling_std_20 > 1e-8 else 0.0
            )

            # directional_consistency: % of positive tick-returns in last 5.
            # Stable is ~50%, Strong Bull is high, Strong Bear is low.
            if len(vals) >= 3:
                tick_returns = [(vals[i] - vals[i+1]) / (vals[i+1] + 1e-8)
                                for i in range(min(5, len(vals)-1))]
                directional_consistency = float(
                    sum(1 for r in tick_returns if r > 0) / len(tick_returns)
                )
            else:
                directional_consistency = 0.5

            # sharpe_proxy: mean tick-return / std over last 20 ticks.
            # Strong Bull/Bear have high |Sharpe|; Stable and Chaotic have low.
            if len(vals) >= 5:
                tick_rets = [(vals[i] - vals[i+1]) / (vals[i+1] + 1e-8)
                             for i in range(min(19, len(vals)-1))]
                mean_ret = np.mean(tick_rets)
                std_ret  = np.std(tick_rets)
                sharpe_proxy = mean_ret / (std_ret + 1e-8)
            else:
                sharpe_proxy = 0.0

            # momentum_divergence: return_1 and return_20 disagree in sign.
            # Chaotic flips sign often; Stable and Strong regimes are consistent.
            momentum_divergence = float(return_1 * return_20 < 0)

            # Mode is stored in the stock dict
            mode = stock['mode']

            records.append({
                'tick': market.tick_count - 1,
                'stock_id': stock_id,
                'price': float(price),
                'return_1': float(return_1),
                'return_5': float(return_5),
                'return_20': float(return_20),
                'rolling_mean_5': float(rolling_mean_5),
                'rolling_mean_20': float(rolling_mean_20),
                'rolling_std_20': float(rolling_std_20),
                'momentum_5_20': float(momentum_5_20),
                'drift_proxy': float(drift_proxy),
                'vol_ratio': float(vol_ratio),
                'mean_reversion_signal': float(mean_reversion_signal),
                'directional_consistency': float(directional_consistency),
                'sharpe_proxy': float(sharpe_proxy),
                'momentum_divergence': float(momentum_divergence),
                'mode': int(mode),
            })

        # Advance market
        market.tick()

    # Create DataFrame and save
    df = pd.DataFrame(records)
    df.to_parquet(out_path, index=False, engine='pyarrow')

    return out_path

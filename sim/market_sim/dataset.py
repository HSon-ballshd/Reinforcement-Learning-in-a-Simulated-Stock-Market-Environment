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
    
    Output columns:
        [tick, stock_id, price, return_1, return_5, return_20,
         rolling_mean_5, rolling_mean_20, rolling_std_20, momentum_5_20,
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
            
            # Extract features from observation
            # Format: [price, return_1, return_5, return_20,
            #          rolling_mean_5, rolling_mean_20, rolling_std_20, momentum_5_20,
            #          mode_onehot_0, ..., mode_onehot_5]
            obs_features = obs_revealed[stock_id]
            price = obs_features[0]
            return_1 = obs_features[1]
            return_5 = obs_features[2]
            return_20 = obs_features[3]
            rolling_mean_5 = obs_features[4]
            rolling_mean_20 = obs_features[5]
            rolling_std_20 = obs_features[6]
            momentum_5_20 = obs_features[7]
            
            # Mode is stored in the stock dict
            mode = stock['mode']
            
            records.append({
                'tick': market.tick_count - 1,  # Adjust for 0-indexing
                'stock_id': stock_id,
                'price': float(price),
                'return_1': float(return_1),
                'return_5': float(return_5),
                'return_20': float(return_20),
                'rolling_mean_5': float(rolling_mean_5),
                'rolling_mean_20': float(rolling_mean_20),
                'rolling_std_20': float(rolling_std_20),
                'momentum_5_20': float(momentum_5_20),
                'mode': int(mode),
            })
        
        # Advance market
        market.tick()
    
    # Create DataFrame and save
    df = pd.DataFrame(records)
    df.to_parquet(out_path, index=False, engine='pyarrow')
    
    return out_path

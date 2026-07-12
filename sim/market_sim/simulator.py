"""
CookieClickerMarket simulator.

Ports the stock market engine from minigameMarket.js (lines 763-872)
with dragonBoost = 0.

This module exposes the CookieClickerMarket class which manages the price
dynamics of one or more stocks across six hidden regimes.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


class CookieClickerMarket:
    """
    Cookie Clicker stock market simulator.
    
    Six hidden regimes: Stable (0), Bullish (1), Bearish (2), 
    Strong Bull (3), Strong Bear (4), Chaotic (5).
    
    Regime is not observable to traders. Price dynamics follow
    minigameMarket.js lines 813-872 verbatim with dragonBoost = 0.
    """
    
    MODE_NAMES = ["Stable", "Bullish", "Bearish", "Strong Bull", "Strong Bear", "Chaotic"]
    MODE_WEIGHTS = [0, 1, 1, 2, 2, 3, 4, 5]  # JS line 780, 871
    
    def __init__(
        self,
        n_stocks: int = 1,
        bank_level: int = 1,
        seed: Optional[int] = None,
        seconds_per_tick: int = 60,
    ) -> None:
        """
        Initialize the market.
        
        Args:
            n_stocks: Number of independent stocks. Default 1.
            bank_level: Replaces Game.Objects['Bank'].level from JS. Default 1.
            seed: RNG seed. If None, random.
            seconds_per_tick: Typically 60 (JS default). Exposed for time compression.
        """
        self.n_stocks = n_stocks
        self.bank_level = bank_level
        self.seconds_per_tick = seconds_per_tick
        self.rng = np.random.RandomState(seed)
        
        # Per-stock state: each is a dict with keys:
        # stock_id, price, mode, dur, d, vals (price history)
        self.stocks = []
        self.tick_count = 0
        self.history_records = []
        
        self.reset()
    
    @staticmethod
    def _choose(rng: np.random.RandomState, weights: list) -> int:
        """
        Randomly choose an item from the weights list.
        Implements JS choose([0,1,1,2,2,3,4,5]) - picks uniformly from the array.
        """
        # Simply return a random choice from the list values
        return rng.choice(weights)
    
    def _get_resting_val(self, stock_id: int) -> float:
        """
        Get the resting (equilibrium) price for a stock.
        JS: M.getRestingVal(me.id) 
        Returns: 10 + 10*stock_id + (bank_level - 1)
        """
        return 10 + 10 * stock_id + (self.bank_level - 1)
    
    def reset(self) -> None:
        """
        Initialize all stocks and run 15 ticks (JS line 792-795).
        
        Each stock gets:
        - mode = choose([0, 1, 1, 2, 2, 3, 4, 5])
        - dur = floor(10 + rand*690)
        - price = resting_val
        - d = rand*0.2 - 0.1
        
        Matches minigameMarket.js lines 763-796 (with dragonBoost=0).
        """
        self.stocks = []
        self.tick_count = 0
        self.history_records = []
        
        for stock_id in range(self.n_stocks):
            stock = {
                'stock_id': stock_id,
                'price': self._get_resting_val(stock_id),
                'mode': self._choose(self.rng, self.MODE_WEIGHTS),
                'dur': int(10 + self.rng.uniform(0, 690)),
                'd': self.rng.uniform(0, 0.2) - 0.1,
                'vals': [],  # Price history (up to 65 entries, JS line 862)
            }
            stock['vals'].append(stock['price'])
            self.stocks.append(stock)
        
        # Run 15 ticks (JS line 792-795) to warm up
        for _ in range(15):
            self.tick()
    
    def tick(self) -> None:
        """
        Advance the market by one tick.
        
        Updates price, mode, dur, d for every stock following
        minigameMarket.js lines 813-872 verbatim with dragonBoost = 0.
        """
        # dragonBoost = 0 (per CONTRACT.md)
        dragon_boost = 0.0
        glob_d = 0.0
        glob_p = self.rng.uniform(0, 1)
        
        # JS line 807: global shock only triggers when dragonBoost > 0
        if dragon_boost > 0 and self.rng.uniform(0, 1) < 0.1 + 0.1 * dragon_boost:
            glob_d = (self.rng.uniform(0, 1) - 0.5) * 2
        
        for stock in self.stocks:
            stock['last'] = 0  # JS line 811
            
            # JS line 813: decay d
            stock['d'] *= 0.97 + 0.01 * dragon_boost
            
            # JS lines 815-820: mode-specific dynamics
            if stock['mode'] == 0:  # Stable
                stock['d'] *= 0.95
                stock['d'] += 0.05 * (self.rng.uniform(0, 1) - 0.5)
            elif stock['mode'] == 1:  # Bullish
                stock['d'] *= 0.99
                stock['d'] += 0.05 * (self.rng.uniform(0, 1) - 0.1)
            elif stock['mode'] == 2:  # Bearish
                stock['d'] *= 0.99
                stock['d'] -= 0.05 * (self.rng.uniform(0, 1) - 0.1)
            elif stock['mode'] == 3:  # Strong Bull
                stock['d'] += 0.15 * (self.rng.uniform(0, 1) - 0.1)
                stock['price'] += self.rng.uniform(0, 5)
            elif stock['mode'] == 4:  # Strong Bear
                stock['d'] -= 0.15 * (self.rng.uniform(0, 1) - 0.1)
                stock['price'] -= self.rng.uniform(0, 5)
            elif stock['mode'] == 5:  # Chaotic
                stock['d'] += 0.3 * (self.rng.uniform(0, 1) - 0.5)
            
            # JS line 822: mean reversion
            resting_val = self._get_resting_val(stock['stock_id'])
            stock['price'] += (resting_val - stock['price']) * 0.01
            
            # JS line 824: global shock (dragonBoost=0, so skipped)
            if glob_d != 0 and self.rng.uniform(0, 1) < glob_p:
                shock = (1 + stock['d'] * np.power(self.rng.uniform(0, 1), 3) * 7) * glob_d
                stock['price'] -= shock
                stock['price'] -= glob_d * (1 + np.power(self.rng.uniform(0, 1), 3) * 7)
                stock['d'] += glob_d * (1 + self.rng.uniform(0, 1) * 4)
                stock['dur'] = 0
            
            # JS line 826: extreme noise
            stock['price'] += np.power((self.rng.uniform(0, 1) - 0.5) * 2, 11) * 3
            
            # JS line 827: d noise
            stock['d'] += 0.1 * (self.rng.uniform(0, 1) - 0.5)
            
            # JS line 828: 15% chance of medium noise
            if self.rng.uniform(0, 1) < 0.15:
                stock['price'] += (self.rng.uniform(0, 1) - 0.5) * 3
            
            # JS line 829: 3% chance of large noise
            if self.rng.uniform(0, 1) < 0.03:
                stock['price'] += (self.rng.uniform(0, 1) - 0.5) * (10 + 10 * dragon_boost)
            
            # JS line 830: 10% chance of d noise
            if self.rng.uniform(0, 1) < 0.1:
                stock['d'] += (self.rng.uniform(0, 1) - 0.5) * (0.3 + 0.2 * dragon_boost)
            
            # JS line 831-835: Chaotic mode extra volatility
            if stock['mode'] == 5:
                if self.rng.uniform(0, 1) < 0.5:
                    stock['price'] += (self.rng.uniform(0, 1) - 0.5) * 10
                if self.rng.uniform(0, 1) < 0.2:
                    stock['d'] = (self.rng.uniform(0, 1) - 0.5) * (2 + 6 * dragon_boost)
            
            # JS line 836-838: Strong Bull/Bear extra moves
            if stock['mode'] == 3 and self.rng.uniform(0, 1) < 0.3:
                stock['d'] += (self.rng.uniform(0, 1) - 0.5) * 0.1
                stock['price'] += (self.rng.uniform(0, 1) - 0.7) * 10
            
            if stock['mode'] == 3 and self.rng.uniform(0, 1) < 0.03:
                stock['mode'] = 4
            
            if stock['mode'] == 4 and self.rng.uniform(0, 1) < 0.3:
                stock['d'] += (self.rng.uniform(0, 1) - 0.5) * 0.1
                stock['price'] += (self.rng.uniform(0, 1) - 0.3) * 10
            
            # JS line 840: cap d if price is high and d is positive
            if stock['price'] > (100 + (self.bank_level - 1) * 3) and stock['d'] > 0:
                stock['d'] *= 0.9
            
            # JS line 842: apply d to price
            stock['price'] += stock['d']
            
            # JS lines 857-859: floor price at 1 with soft cap at 5
            if stock['price'] < 5:
                stock['price'] += (5 - stock['price']) * 0.5
            if stock['price'] < 5 and stock['d'] < 0:
                stock['d'] *= 0.95
            stock['price'] = max(stock['price'], 1.0)
            
            # JS line 861-862: maintain price history (up to 65 entries)
            stock['vals'].insert(0, stock['price'])
            if len(stock['vals']) > 65:
                stock['vals'].pop()
            
            # JS line 864-872: mode transition
            stock['dur'] -= 1
            if stock['dur'] <= 0:
                stock['dur'] = int(10 + self.rng.uniform(0, 690 - 200 * dragon_boost))
                if self.rng.uniform(0, 1) < dragon_boost and self.rng.uniform(0, 1) < 0.5:
                    stock['mode'] = 5
                elif self.rng.uniform(0, 1) < 0.7 and (stock['mode'] == 3 or stock['mode'] == 4):
                    stock['mode'] = 5
                else:
                    stock['mode'] = self._choose(self.rng, self.MODE_WEIGHTS)
            
            # Record for history
            self.history_records.append({
                'tick': self.tick_count,
                'stock_id': stock['stock_id'],
                'price': stock['price'],
                'mode': stock['mode'],
                'd': stock['d'],
                'dur': stock['dur'],
            })
        
        self.tick_count += 1
    
    def history(self) -> pd.DataFrame:
        """
        Return the full market history.
        
        Returns:
            DataFrame with columns: [tick, stock_id, price, mode, d, dur]
            mode is always included (history is for offline analysis only).
        """
        if not self.history_records:
            return pd.DataFrame()
        return pd.DataFrame(self.history_records)
    
    def get_observation(self, *, reveal: bool = False) -> np.ndarray:
        """
        Get observable features for all stocks.
        
        Args:
            reveal: If True, append one-hot mode vector (6 features).
                   If False, only observable features.
        
        Returns:
            numpy array of shape (n_stocks, n_features).
            Default features: [price, return_1, return_5, return_20,
                              rolling_mean_5, rolling_mean_20, rolling_std_20, momentum_5_20]
            If reveal=True, append 6-dimensional one-hot mode vector.
        """
        features = []
        
        for stock in self.stocks:
            stock_features = []
            price = stock['price']
            vals = stock['vals']  # vals[0] is current, vals[1] is 1-tick ago, etc.
            
            # Current price
            stock_features.append(price)
            
            # Returns over different windows
            # return_1: (price - price_1_tick_ago) / price_1_tick_ago
            ret_1 = (vals[0] - vals[1]) / vals[1] if len(vals) > 1 else 0.0
            stock_features.append(ret_1)
            
            ret_5 = (vals[0] - vals[5]) / vals[5] if len(vals) > 5 else 0.0
            stock_features.append(ret_5)

            ret_20 = (vals[0] - vals[20]) / vals[20] if len(vals) > 20 else 0.0
            stock_features.append(ret_20)
            
            # Rolling mean over 5 and 20 ticks
            mean_5 = np.mean(vals[:5]) if len(vals) >= 5 else np.mean(vals)
            stock_features.append(mean_5)
            
            mean_20 = np.mean(vals[:20]) if len(vals) >= 20 else np.mean(vals)
            stock_features.append(mean_20)
            
            # Rolling std over 20 ticks
            std_20 = np.std(vals[:20]) if len(vals) >= 20 else np.std(vals)
            stock_features.append(std_20)
            
            # Momentum: return_5_20 = (5-tick return) - (20-tick return)
            momentum = ret_5 - ret_20
            stock_features.append(momentum)
            
            # If reveal=True, append one-hot mode vector
            if reveal:
                mode_onehot = np.zeros(6)
                mode_onehot[stock['mode']] = 1.0
                stock_features.extend(mode_onehot)
            
            features.append(stock_features)
        
        return np.array(features, dtype=np.float32)

#!/usr/bin/env python
"""Quick sanity check for CookieClickerMarket."""

import sys
sys.path.insert(0, r'C:\Users\DELL\MyDrive\Desktop\Reinforcement-Learning-in-a-Simulated-Stock-Market-Environment')

from sim.market_sim import CookieClickerMarket
import numpy as np

print("=" * 60)
print("T-001 Sanity Check: CookieClickerMarket Reset & Observation")
print("=" * 60)

# Test 1: Initialize
print("\n[Test 1] Initialize with 2 stocks, seed=42")
market = CookieClickerMarket(n_stocks=2, seed=42)
print(f"✓ Created market with {len(market.stocks)} stocks")

# Test 2: Check reset created valid state
print("\n[Test 2] Check reset() created valid state")
for i, stock in enumerate(market.stocks):
    assert stock['stock_id'] == i, "Stock ID mismatch"
    assert stock['price'] >= 1.0, f"Price too low: {stock['price']}"
    assert stock['mode'] in range(6), f"Invalid mode: {stock['mode']}"
    assert 10 <= stock['dur'] <= 700, f"Invalid duration: {stock['dur']}"
    assert len(stock['vals']) >= 16, f"Price history too short: {len(stock['vals'])}"
    print(f"  Stock {i}: price={stock['price']:.2f}, mode={stock['mode']}, dur={stock['dur']}")
print("✓ All stocks have valid initial state")

# Test 3: Check history
print("\n[Test 3] Check history() DataFrame")
history = market.history()
print(f"✓ History has {len(history)} records")
print(f"  Columns: {list(history.columns)}")
assert 'tick' in history.columns, "Missing 'tick' column"
assert 'stock_id' in history.columns, "Missing 'stock_id' column"
assert 'price' in history.columns, "Missing 'price' column"
assert 'mode' in history.columns, "Missing 'mode' column"
print("✓ History DataFrame has all required columns")

# Test 4: Check observation
print("\n[Test 4] Check get_observation()")
obs_no_reveal = market.get_observation(reveal=False)
obs_reveal = market.get_observation(reveal=True)
print(f"✓ Observation (reveal=False) shape: {obs_no_reveal.shape}")
print(f"✓ Observation (reveal=True) shape: {obs_reveal.shape}")
assert obs_no_reveal.shape == (2, 8), f"Wrong shape: {obs_no_reveal.shape}"
assert obs_reveal.shape == (2, 14), f"Wrong shape: {obs_reveal.shape}"
assert obs_no_reveal.dtype == np.float32, f"Wrong dtype: {obs_no_reveal.dtype}"
print("✓ Observation shapes and dtypes correct")

# Test 5: Check determinism
print("\n[Test 5] Check determinism with same seed")
market2 = CookieClickerMarket(n_stocks=2, seed=42)
prices_match = np.allclose(
    [s['price'] for s in market.stocks],
    [s['price'] for s in market2.stocks]
)
print(f"✓ Same seed produces same prices: {prices_match}")
assert prices_match, "Determinism check failed"

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED - T-001 Implementation Valid")
print("=" * 60)

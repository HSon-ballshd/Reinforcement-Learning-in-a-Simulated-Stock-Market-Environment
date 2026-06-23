"""Pytest invariants for the market simulator (task T-005)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sim.market_sim import CookieClickerMarket, generate_regime_dataset
from sim.market_sim.simulator import (
 BEARISH,
 BULLISH,
 CHAOTIC,
 N_MODES,
 STABLE,
 STRONG_BEAR,
 STRONG_BULL,
)


def test_reset_initializes_one_stock_per_id() -> None:
 m = CookieClickerMarket(n_stocks=3, seed=0)
 assert len(m._stocks) == 3
 for i, s in enumerate(m._stocks):
  assert s.stock_id == i
  assert s.mode in range(N_MODES)


def test_observation_shape_hides_mode_by_default() -> None:
 m = CookieClickerMarket(seed=0)
 m.tick()
 obs = m.get_observation(reveal=False)
 assert obs.shape == (1, 8), obs.shape
 assert obs.dtype == np.float64
 assert np.isfinite(obs).all()


def test_observation_reveal_appends_one_hot_mode() -> None:
 m = CookieClickerMarket(seed=0)
 m.tick()
 obs = m.get_observation(reveal=True)
 assert obs.shape == (1, 14)
 last_six = obs[0, -6:]
 assert last_six.sum() == pytest.approx(1.0)
 assert set(last_six.tolist()).issubset({0.0, 1.0})


def test_price_never_below_one() -> None:
 m = CookieClickerMarket(seed=123)
 for _ in range(5000):
  m.tick()
 prices = m.current_prices
 assert (prices >= 1.0).all(), prices


def test_dur_counts_down_per_tick() -> None:
 m = CookieClickerMarket(seed=7)
 start = [s.dur for s in m._stocks]
 m.tick()
 after = [s.dur for s in m._stocks]
 assert after[0] == start[0] - 1


def test_mode_distribution_matches_weights() -> None:
 """Over a long run, mode frequencies follow [0,1,1,2,2,3,4,5]."""
 m = CookieClickerMarket(seed=42)
 counts = np.zeros(N_MODES)
 for _ in range(50_000):
  m.tick()
  counts += np.bincount(m.current_modes, minlength=N_MODES)
 freqs = counts / counts.sum()
 expected = np.array([1, 2, 2, 1, 1, 1]) / 8.0
 assert np.allclose(freqs, expected, atol=0.05), (freqs, expected)


def test_seed_reproducibility() -> None:
 a = CookieClickerMarket(seed=2024)
 b = CookieClickerMarket(seed=2024)
 for _ in range(100):
  a.tick()
  b.tick()
 assert np.allclose(a.current_prices, b.current_prices)
 assert (a.current_modes == b.current_modes).all()


def test_history_has_expected_columns() -> None:
 m = CookieClickerMarket(seed=0)
 for _ in range(5):
  m.tick()
 h = m.history()
 assert list(h.columns) == ["tick", "stock_id", "price", "mode", "d", "dur"]
 assert len(h) == 5
 assert h["tick"].tolist() == [1, 2, 3, 4, 5]


def test_generate_regime_dataset_writes_parquet(tmp_path: Path) -> None:
 out = generate_regime_dataset(
  n_ticks=200, n_stocks=1, seed=0, out_path=tmp_path / "ds.parquet"
 )
 assert out.exists()
 df = pd.read_parquet(out)
 expected_cols = [
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
 assert list(df.columns) == expected_cols
 assert len(df) == 200
 assert df["mode"].between(0, N_MODES - 1).all()
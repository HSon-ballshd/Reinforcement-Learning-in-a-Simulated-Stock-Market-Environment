"""
Tests for TradingEnv (eval/env/trading_env.py).

Contract checks:
- reset() returns an observation of shape (8,)
- step() returns (obs, reward, done, info) with correct types
- BUY increases holdings; SELL decreases holdings
- HOLD is a no-op on holdings
- done is True after max_steps
- portfolio value never NaN
"""

import pytest
import numpy as np
from sim.market_sim import CookieClickerMarket
from eval.env.trading_env import TradingEnv


class TestTradingEnvReset:
    def test_reset_returns_observation(self):
        market = CookieClickerMarket(n_stocks=1, seed=42)
        env = TradingEnv(market, seed=42)
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (8,)
        assert np.all(np.isfinite(obs))

    def test_reset_deterministic_with_seed(self):
        market1 = CookieClickerMarket(n_stocks=1, seed=99)
        market2 = CookieClickerMarket(n_stocks=1, seed=99)
        env1 = TradingEnv(market1, seed=99)
        env2 = TradingEnv(market2, seed=99)
        np.testing.assert_array_equal(env1.reset(), env2.reset())

    def test_reset_initial_cash(self):
        market = CookieClickerMarket(n_stocks=1, seed=42)
        env = TradingEnv(market, initial_cash=5000.0, seed=42)
        env.reset()
        assert env.cash == 5000.0
        assert env.holdings == 0.0


class TestTradingEnvStep:
    def test_hold_is_noop(self):
        market = CookieClickerMarket(n_stocks=1, seed=42)
        env = TradingEnv(market, seed=42)
        env.reset()
        holdings_before = env.holdings
        cash_before     = env.cash
        env.step(TradingEnv.HOLD)
        assert env.holdings == holdings_before
        assert env.cash     == cash_before

    def test_buy_increases_holdings(self):
        market = CookieClickerMarket(n_stocks=1, seed=42)
        env = TradingEnv(market, seed=42)
        env.reset()
        price = env.market.stocks[0]['price']
        holdings_before = env.holdings
        env.step(TradingEnv.BUY)
        assert env.holdings > holdings_before
        assert env.holdings > 0

    def test_sell_decreases_holdings(self):
        market = CookieClickerMarket(n_stocks=1, seed=42)
        env = TradingEnv(market, seed=42)
        env.reset()
        # First buy something
        env.step(TradingEnv.BUY)
        holdings_before = env.holdings
        assert holdings_before > 0
        # Then sell
        env.step(TradingEnv.SELL)
        assert env.holdings < holdings_before

    def test_step_returns_correct_tuple(self):
        market = CookieClickerMarket(n_stocks=1, seed=42)
        env = TradingEnv(market, seed=42)
        env.reset()
        result = env.step(TradingEnv.HOLD)
        assert isinstance(result, tuple) and len(result) == 4
        obs, reward, done, info = result
        assert isinstance(obs, np.ndarray) and obs.shape == (8,)
        assert isinstance(reward, (int, float, np.floating))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_done_after_max_steps(self):
        market = CookieClickerMarket(n_stocks=1, seed=42)
        env = TradingEnv(market, max_steps=5, seed=42)
        env.reset()
        for i in range(5):
            _, _, done, _ = env.step(TradingEnv.HOLD)
        assert done is True

    def test_portfolio_value_never_nan(self):
        market = CookieClickerMarket(n_stocks=1, seed=42)
        env = TradingEnv(market, seed=42)
        env.reset()
        for _ in range(200):
            _, _, done, _ = env.step(np.random.randint(0, 3))
            assert np.isfinite(env._portfolio_value())
            if done:
                break

    def test_reward_scale_is_reasonable(self):
        """Reward per step should be small (fraction of portfolio)."""
        market = CookieClickerMarket(n_stocks=1, seed=42)
        env = TradingEnv(market, seed=42)
        env.reset()
        _, reward, _, _ = env.step(TradingEnv.HOLD)
        assert abs(reward) < 1.0, f"Reward {reward} seems too large"


class TestTradingEnvRender:
    def test_render_does_not_raise(self):
        market = CookieClickerMarket(n_stocks=1, seed=42)
        env = TradingEnv(market, seed=42)
        env.reset()
        env.render()   # Must not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

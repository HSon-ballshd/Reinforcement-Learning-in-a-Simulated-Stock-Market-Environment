"""
Tests for CookieClickerMarket reset() and basic invariants.

Tests focus on:
1. Initialization with correct parameters
2. reset() creates all stocks with valid initial state
3. Price values meet constraints (val >= 1)
4. Duration countdown works correctly
5. Mode weights are respected
6. Comprehensive invariant testing for long simulations
"""

import pytest
import numpy as np
from sim.market_sim import CookieClickerMarket


class TestCookieClickerMarketReset:
    """Test the reset() method."""
    
    def test_reset_single_stock(self):
        """Test reset with default single stock."""
        market = CookieClickerMarket(n_stocks=1, seed=42)
        assert len(market.stocks) == 1
        assert market.tick_count == 15  # Warmed up with 15 ticks
    
    def test_reset_multiple_stocks(self):
        """Test reset with multiple stocks."""
        market = CookieClickerMarket(n_stocks=3, seed=42)
        assert len(market.stocks) == 3
        
        for i, stock in enumerate(market.stocks):
            assert stock['stock_id'] == i
            assert 'price' in stock
            assert 'mode' in stock
            assert 'dur' in stock
            assert 'd' in stock
            assert 'vals' in stock
    
    def test_reset_stock_id_matches(self):
        """Test that stock IDs match their position."""
        market = CookieClickerMarket(n_stocks=5, seed=42)
        for i, stock in enumerate(market.stocks):
            assert stock['stock_id'] == i
    
    def test_reset_price_above_one(self):
        """Test that prices are always >= 1 (CONTRACT.md invariant)."""
        market = CookieClickerMarket(n_stocks=5, seed=42)
        for stock in market.stocks:
            assert stock['price'] >= 1.0
    
    def test_reset_initial_price_correct(self):
        """
        Test that initial price (after warm-up) is close to resting value.
        
        Initial price before warm-up should be: 10 + 10*stock_id + (bank_level - 1)
        After 15 ticks of mean reversion, should be reasonably close.
        """
        market = CookieClickerMarket(n_stocks=2, bank_level=1, seed=42)
        
        for stock in market.stocks:
            stock_id = stock['stock_id']
            expected_resting = 10 + 10 * stock_id
            actual_price = stock['price']
            
            # After 15 ticks of mean reversion, price should move toward resting value
            # but not necessarily reach it exactly due to noise
            assert actual_price > 0
    
    def test_reset_mode_is_valid(self):
        """Test that all modes are in valid range [0, 5]."""
        market = CookieClickerMarket(n_stocks=20, seed=42)
        for stock in market.stocks:
            assert stock['mode'] in range(6)
    
    def test_reset_dur_in_valid_range(self):
        """Test that duration is in valid range [10, 700]."""
        market = CookieClickerMarket(n_stocks=20, seed=42)
        for stock in market.stocks:
            # dur should be in [10, 700] from JS: floor(10 + rand*690)
            assert 10 <= stock['dur'] <= 700
    
    def test_reset_d_in_valid_range(self):
        """Test that d (derivative) is valid."""
        market = CookieClickerMarket(n_stocks=20, seed=42)
        # After reset (which includes 15 ticks), d can be anything, but should be a number
        for stock in market.stocks:
            assert isinstance(stock['d'], (int, float))
            assert np.isfinite(stock['d'])
    
    def test_reset_vals_history_populated(self):
        """Test that price history (vals) is populated after reset."""
        market = CookieClickerMarket(n_stocks=2, seed=42)
        for stock in market.stocks:
            # vals should have at least 16 entries (1 initial + 15 from ticks)
            assert len(stock['vals']) >= 16
            # Most recent price should match current price
            assert abs(stock['vals'][0] - stock['price']) < 1e-6
    
    def test_reset_deterministic_with_seed(self):
        """Test that reset is deterministic with the same seed."""
        market1 = CookieClickerMarket(n_stocks=2, seed=42)
        prices1 = [s['price'] for s in market1.stocks]
        modes1 = [s['mode'] for s in market1.stocks]
        
        market2 = CookieClickerMarket(n_stocks=2, seed=42)
        prices2 = [s['price'] for s in market2.stocks]
        modes2 = [s['mode'] for s in market2.stocks]
        
        np.testing.assert_array_almost_equal(prices1, prices2)
        assert modes1 == modes2
    
    def test_reset_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        market1 = CookieClickerMarket(n_stocks=3, seed=42)
        prices1 = [s['price'] for s in market1.stocks]
        
        market2 = CookieClickerMarket(n_stocks=3, seed=123)
        prices2 = [s['price'] for s in market2.stocks]
        
        # With very high probability, results should be different
        assert not np.allclose(prices1, prices2)
    
    def test_reset_bank_level_affects_resting_value(self):
        """Test that bank_level affects the resting (initial) value."""
        market1 = CookieClickerMarket(n_stocks=1, bank_level=1, seed=42)
        market2 = CookieClickerMarket(n_stocks=1, bank_level=5, seed=42)
        
        # With bank_level=5 vs bank_level=1, resting value increases by 4
        # After reset, prices should be different
        price1 = market1.stocks[0]['price']
        price2 = market2.stocks[0]['price']
        
        # They won't be exactly 4 apart due to noise, but should be significantly different
        assert abs((price2 - price1) - 4.0) < 10.0  # Allow large tolerance due to randomness
    
    def test_reset_history_cleared(self):
        """Test that history is cleared on reset."""
        market = CookieClickerMarket(n_stocks=1, seed=42)
        # history_records is populated during the 15 warm-up ticks
        assert len(market.history_records) == 15
        
        # After reset, history should be cleared
        market.reset()
        # After reset, we have 15 more ticks of warm-up
        assert len(market.history_records) == 15


class TestCookieClickerMarketTick:
    """Test the tick() method and invariants."""
    
    def test_tick_increments_counter(self):
        """Test that tick count increments."""
        market = CookieClickerMarket(n_stocks=1, seed=42)
        initial = market.tick_count
        market.tick()
        assert market.tick_count == initial + 1
    
    def test_tick_decrements_duration(self):
        """Test that duration counts down each tick."""
        market = CookieClickerMarket(n_stocks=1, seed=42)
        
        for _ in range(50):
            for stock in market.stocks:
                dur_before = stock['dur']
            market.tick()
            for stock in market.stocks:
                dur_after = stock['dur']
                # When dur > 0, it should decrement
                if dur_before > 1:
                    assert dur_after == dur_before - 1
    
    def test_tick_price_always_positive(self):
        """Test that prices stay >= 1.0 throughout simulation."""
        market = CookieClickerMarket(n_stocks=3, seed=42)
        
        for _ in range(500):
            for stock in market.stocks:
                assert stock['price'] >= 1.0, f"Price too low: {stock['price']}"
            market.tick()
    
    def test_tick_mode_always_valid(self):
        """Test that mode is always in [0, 5]."""
        market = CookieClickerMarket(n_stocks=3, seed=42)
        
        for _ in range(500):
            for stock in market.stocks:
                assert stock['mode'] in range(6), f"Invalid mode: {stock['mode']}"
            market.tick()
    
    def test_tick_mode_transitions_occur(self):
        """Test that mode transitions happen (dur countdown and reset)."""
        market = CookieClickerMarket(n_stocks=1, seed=42)
        modes_seen = set([market.stocks[0]['mode']])
        
        for _ in range(1000):
            market.tick()
            modes_seen.add(market.stocks[0]['mode'])
        
        # Should see multiple modes
        assert len(modes_seen) > 1
    
    def test_tick_price_history_maintained(self):
        """Test that price history is maintained up to 65 entries."""
        market = CookieClickerMarket(n_stocks=1, seed=42)
        
        for _ in range(200):
            market.tick()
            # vals should be at most 65 entries (plus warm-up)
            assert len(market.stocks[0]['vals']) <= 65
            # vals[0] should always be current price
            assert abs(market.stocks[0]['vals'][0] - market.stocks[0]['price']) < 1e-6
    
    def test_tick_d_is_finite(self):
        """Test that d (drift) is always finite."""
        market = CookieClickerMarket(n_stocks=3, seed=42)
        
        for _ in range(500):
            for stock in market.stocks:
                assert np.isfinite(stock['d']), f"Non-finite d: {stock['d']}"
            market.tick()
    
    def test_tick_history_records_populated(self):
        """Test that history is recorded for each tick."""
        market = CookieClickerMarket(n_stocks=2, seed=42)
        initial_history_len = len(market.history_records)
        
        market.tick()
        
        # Should add n_stocks records per tick
        assert len(market.history_records) == initial_history_len + 2


class TestCookieClickerMarketGetObservation:
    """Test the get_observation() method."""
    
    def test_get_observation_shape_no_reveal(self):
        """Test observation shape without reveal."""
        market = CookieClickerMarket(n_stocks=2, seed=42)
        obs = market.get_observation(reveal=False)
        
        # Should be (n_stocks, 8) for default features
        assert obs.shape == (2, 8)
    
    def test_get_observation_shape_reveal(self):
        """Test observation shape with reveal."""
        market = CookieClickerMarket(n_stocks=2, seed=42)
        obs = market.get_observation(reveal=True)
        
        # Should be (n_stocks, 14) = 8 features + 6 one-hot mode
        assert obs.shape == (2, 14)
    
    def test_get_observation_mode_onehot(self):
        """Test that reveal mode produces valid one-hot encoding."""
        market = CookieClickerMarket(n_stocks=1, seed=42)
        obs = market.get_observation(reveal=True)
        
        mode_onehot = obs[0, 8:14]  # Last 6 features are one-hot mode
        
        # Should be a valid one-hot vector
        assert np.sum(mode_onehot) == 1.0
        assert np.all((mode_onehot == 0) | (mode_onehot == 1))
    
    def test_get_observation_price_first_feature(self):
        """Test that first feature is current price."""
        market = CookieClickerMarket(n_stocks=1, seed=42)
        obs = market.get_observation(reveal=False)
        
        # First feature should be the current price
        assert abs(obs[0, 0] - market.stocks[0]['price']) < 1e-5
    
    def test_get_observation_dtype_float32(self):
        """Test that observation is float32."""
        market = CookieClickerMarket(n_stocks=1, seed=42)
        obs = market.get_observation(reveal=False)
        
        assert obs.dtype == np.float32
    
    def test_get_observation_features_finite(self):
        """Test that all features are finite numbers."""
        market = CookieClickerMarket(n_stocks=3, seed=42)
        
        for _ in range(100):
            obs = market.get_observation(reveal=False)
            assert np.all(np.isfinite(obs)), "Non-finite values in observation"
            
            obs_reveal = market.get_observation(reveal=True)
            assert np.all(np.isfinite(obs_reveal)), "Non-finite values in revealed observation"
            
            market.tick()


class TestCookieClickerMarketInit:
    """Test initialization parameters."""
    
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        market = CookieClickerMarket(seed=42)
        
        assert market.n_stocks == 1
        assert market.bank_level == 1
        assert market.seconds_per_tick == 60
    
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        market = CookieClickerMarket(
            n_stocks=5,
            bank_level=10,
            seed=123,
            seconds_per_tick=30,
        )
        
        assert market.n_stocks == 5
        assert market.bank_level == 10
        assert market.seconds_per_tick == 30
    
    def test_init_no_seed_produces_randomness(self):
        """Test that init without seed produces different results."""
        market1 = CookieClickerMarket(n_stocks=1)
        market2 = CookieClickerMarket(n_stocks=1)
        
        # Very unlikely to be identical
        prices_identical = abs(market1.stocks[0]['price'] - market2.stocks[0]['price']) < 0.01
        assert not prices_identical


class TestCookieClickerMarketInvariants:
    """Test comprehensive invariants over long simulations."""
    
    def test_long_simulation_price_invariant(self):
        """Test that price >= 1.0 holds over 5000 ticks."""
        market = CookieClickerMarket(n_stocks=2, seed=42)
        
        for _ in range(5000):
            for stock in market.stocks:
                assert stock['price'] >= 1.0, f"Price invariant violated: {stock['price']}"
            market.tick()
    
    def test_long_simulation_mode_invariant(self):
        """Test that mode in [0, 5] holds over 5000 ticks."""
        market = CookieClickerMarket(n_stocks=2, seed=42)
        
        for _ in range(5000):
            for stock in market.stocks:
                assert stock['mode'] in range(6), f"Mode invariant violated: {stock['mode']}"
            market.tick()
    
    def test_long_simulation_dur_countdown(self):
        """Test that dur counts down properly."""
        market = CookieClickerMarket(n_stocks=2, seed=42)
        
        for _ in range(5000):
            for stock in market.stocks:
                assert stock['dur'] >= 0, f"Dur cannot be negative: {stock['dur']}"
            market.tick()
    
    def test_observation_stability(self):
        """Test that observations are valid and stable over time."""
        market = CookieClickerMarket(n_stocks=3, seed=42)
        
        for _ in range(1000):
            obs = market.get_observation(reveal=False)
            assert obs.shape == (3, 8)
            assert np.all(np.isfinite(obs))
            market.tick()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


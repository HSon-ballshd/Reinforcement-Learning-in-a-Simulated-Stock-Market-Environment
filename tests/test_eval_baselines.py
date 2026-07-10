"""
Tests for eval/agents/baselines.py.
"""

import pytest
import numpy as np

from eval.agents.baselines import (
    BaseAgent,
    RandomAgent,
    BuyAndHoldAgent,
    MeanReversionAgent,
    evaluate_agent,
    compare_baselines,
)
from eval.env.trading_env import TradingEnv


class TestRandomAgent:
    def test_select_action_returns_valid(self):
        agent = RandomAgent(seed=0)
        obs   = np.zeros(8)
        for _ in range(100):
            a = agent.select_action(obs, {})
            assert a in (0, 1, 2)

    def test_deterministic_with_seed(self):
        agent1 = RandomAgent(seed=99)
        agent2 = RandomAgent(seed=99)
        obs = np.zeros(8)
        for _ in range(20):
            assert agent1.select_action(obs, {}) == agent2.select_action(obs, {})

    def test_different_seeds_different_actions(self):
        agent1 = RandomAgent(seed=0)
        agent2 = RandomAgent(seed=1)
        obs = np.zeros(8)
        diff = sum(
            agent1.select_action(obs, {}) != agent2.select_action(obs, {})
            for _ in range(100)
        )
        assert diff > 0


class TestBuyAndHoldAgent:
    def test_buys_on_first_step(self):
        agent = BuyAndHoldAgent()
        obs   = np.zeros(8)
        assert agent.select_action(obs, {}) == TradingEnv.BUY

    def test_holds_after_first_step(self):
        agent = BuyAndHoldAgent()
        obs   = np.zeros(8)
        agent.select_action(obs, {})          # first step → BUY
        for _ in range(20):
            assert agent.select_action(obs, {}) == TradingEnv.HOLD

    def test_reset_allows_new_buy(self):
        agent = BuyAndHoldAgent()
        obs   = np.zeros(8)
        assert agent.select_action(obs, {}) == TradingEnv.BUY
        agent.reset()
        assert agent.select_action(obs, {}) == TradingEnv.BUY


class TestMeanReversionAgent:
    def test_buy_when_below_mean(self):
        """Price=9, rolling_mean=10 → below threshold → BUY."""
        agent = MeanReversionAgent(threshold=0.1)
        obs   = np.array([9.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
        assert agent.select_action(obs, {}) == TradingEnv.BUY

    def test_sell_when_above_mean(self):
        """Price=11, rolling_mean=10 → above threshold → SELL."""
        agent = MeanReversionAgent(threshold=0.1)
        obs   = np.array([11.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
        info  = {'holdings': 1.0}   # agent has something to sell
        assert agent.select_action(obs, info) == TradingEnv.SELL

    def test_hold_in_between(self):
        """Price=10.05, rolling_mean=10, threshold=0.01 → in band → HOLD."""
        agent = MeanReversionAgent(threshold=0.01)
        obs   = np.array([10.05, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
        assert agent.select_action(obs, {}) == TradingEnv.HOLD

    def test_no_sell_when_no_holdings(self):
        """Above threshold but no holdings → still HOLD (cannot sell short)."""
        agent = MeanReversionAgent(threshold=0.1)
        obs   = np.array([11.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
        info  = {'holdings': 0.0}
        assert agent.select_action(obs, info) == TradingEnv.HOLD

    def test_hold_on_nan(self):
        agent = MeanReversionAgent()
        obs = np.array([np.nan, 0.0, 0.0, 0.0, np.nan, 0.0, 0.0, 0.0])
        assert agent.select_action(obs, {}) == TradingEnv.HOLD

    def test_hold_on_zero_rolling_mean(self):
        agent = MeanReversionAgent()
        obs   = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert agent.select_action(obs, {}) == TradingEnv.HOLD


class TestEvaluateAgent:
    def test_returns_required_keys(self):
        agent = RandomAgent(seed=0)
        result = evaluate_agent(agent, market_seed=42, n_steps=50)
        assert all(k in result for k in
                   ('total_return_pct', 'final_value', 'n_buys', 'n_sells', 'n_holds'))

    def test_actions_sum_to_steps(self):
        agent = RandomAgent(seed=0)
        result = evaluate_agent(agent, market_seed=42, n_steps=50)
        assert result['n_buys'] + result['n_sells'] + result['n_holds'] == 50

    def test_buy_and_hold_exactly_one_buy(self):
        agent = BuyAndHoldAgent()
        result = evaluate_agent(agent, market_seed=42, n_steps=50)
        assert result['n_buys']  == 1
        assert result['n_sells'] == 0


class TestCompareBaselines:
    def test_all_three_present(self):
        summary = compare_baselines(seeds=[42], n_steps=20)
        assert set(summary.keys()) == {'Random', 'BuyAndHold', 'MeanReversion'}

    def test_mean_return_is_float(self):
        summary = compare_baselines(seeds=[42], n_steps=20)
        for stats in summary.values():
            assert isinstance(stats['mean_return_pct'], float)

    def test_multiple_seeds_produces_std(self):
        summary = compare_baselines(seeds=[42, 123, 456], n_steps=20)
        for stats in summary.values():
            assert 'std_return_pct' in stats
            assert stats['n_runs'] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

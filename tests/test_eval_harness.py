"""
Tests for eval/eval_harness.py.

Tests cover structure and config — not full H1/H2/H3 runs (which
involve training DQN for thousands of steps).
"""

import pytest
import numpy as np
from dataclasses import fields

from eval.eval_harness import (
    EvalConfig,
    _train_ra,
)


class TestEvalConfigDefaults:
    def test_has_required_fields(self):
        config = EvalConfig()
        names = {f.name for f in fields(config)}
        assert names >= {
            "n_episodes", "episode_steps", "seeds",
            "initial_cash", "dataset_n_ticks",
            "dqn_n_steps", "dqn_eval_every", "output_dir",
        }

    def test_episode_steps_positive(self):
        config = EvalConfig(episode_steps=500)
        assert config.episode_steps == 500

    def test_seeds_is_list(self):
        config = EvalConfig(seeds=[10, 20, 30])
        assert isinstance(config.seeds, list)
        assert len(config.seeds) == 3


class TestTrainRAIntegration:
    """Integration test: _train_ra performs gradient steps without crashing."""

    def test_train_ra_runs_without_error(self, tmp_path):
        from eval.agents.dqn_regime import RegimeAwareDQNAgent

        agent = RegimeAwareDQNAgent(
            min_replay_size=10,
            batch_size=4,
            replay_capacity=500,
            seed=0,
        )
        agent.set_classifier(lambda obs: 0)   # always regime 0

        result = _train_ra(
            agent,
            market_seed=42,
            n_steps=50,          # very short — just smoke test
            eval_every=100,      # won't trigger during 50 steps
            max_episode_steps=50,
            eval_steps=20,
            initial_cash=10_000.0,
        )

        assert "logs" in result
        assert "loss" in result["logs"]
        # No crashes — training ran
        assert isinstance(result["best_eval"], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

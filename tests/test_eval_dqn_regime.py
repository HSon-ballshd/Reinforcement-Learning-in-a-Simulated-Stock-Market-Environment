"""
Tests for eval/agents/dqn_regime.py.
"""

import numpy as np
import pytest
import torch

from eval.agents.dqn_regime import (
    RegimeAwareQNetwork,
    RegimeAwareReplayBuffer,
    RegimeAwareDQNAgent,
)


class TestRegimeAwareQNetwork:
    def test_input_dim_is_obs_plus_6(self):
        net = RegimeAwareQNetwork(obs_dim=8)
        x   = torch.randn(4, 14)   # 8 obs + 6 one-hot
        out = net(x)
        assert out.shape == (4, 3)

    def test_wrong_dim_raises(self):
        net = RegimeAwareQNetwork(obs_dim=8)
        x   = torch.randn(4, 8)
        with pytest.raises(RuntimeError):
            net(x)


class TestRegimeAwareReplayBuffer:
    def test_stores_regime(self):
        buf = RegimeAwareReplayBuffer(capacity=10)
        buf.push(np.zeros(8), 3, 1, 0.5, np.ones(8), False)
        t = buf.buffer[0]
        assert t.regime == 3
        assert t.action == 1


class TestRegimeAwareDQNAgentInit:
    def test_default_params(self):
        agent = RegimeAwareDQNAgent(seed=0)
        assert agent.obs_dim    == 8
        assert agent.n_actions  == 3
        assert agent.epsilon    == 1.0

    def test_network_input_dim(self):
        """Q-net should accept 14-dim input (8 obs + 6 regime)."""
        agent = RegimeAwareDQNAgent(seed=0)
        x = torch.randn(2, 14, device=agent.device)
        with torch.no_grad():
            out = agent.q_net(x)
        assert out.shape == (2, 3)


class TestRegimeAwareDQNAgentClassifier:
    def test_no_classifier_defaults_to_zero(self):
        agent = RegimeAwareDQNAgent(seed=0)
        obs = np.zeros(8, dtype=np.float32)
        assert agent._infer_regime(obs) == 0

    def test_set_classifier(self):
        agent = RegimeAwareDQNAgent(seed=0)
        agent.set_classifier(lambda obs: 4)
        obs = np.zeros(8, dtype=np.float32)
        assert agent._infer_regime(obs) == 4

    def test_regime_onehot_correct(self):
        agent = RegimeAwareDQNAgent()
        vec = agent._regime_onehot(2)
        assert vec.shape == (6,)
        assert vec[2] == 1.0
        assert vec.sum() == 1.0

    def test_build_state_concatenates(self):
        agent = RegimeAwareDQNAgent()
        obs = np.array([1.0] * 8, dtype=np.float32)
        state = agent._build_state(obs, 3)
        assert state.shape == (14,)
        assert state[8:14].argmax() == 3


class TestRegimeAwareDQNAgentSelectAction:
    def test_returns_valid_action(self):
        agent = RegimeAwareDQNAgent(seed=0)
        obs  = np.zeros(8, dtype=np.float32)
        for _ in range(50):
            a = agent.select_action(obs, {})
            assert a in (0, 1, 2)

    def test_greedy_deterministic(self):
        """With epsilon=0, _greedy is deterministic."""
        agent = RegimeAwareDQNAgent(seed=0)
        agent.epsilon = 0.0
        obs  = np.zeros(8, dtype=np.float32)
        regime = 0
        for _ in range(10):
            assert agent._greedy(obs, regime) == agent._greedy(obs, regime)


class TestRegimeAwareDQNAgentTrain:
    def test_store_and_retrieve_regime(self):
        agent = RegimeAwareDQNAgent(seed=0, min_replay_size=1)
        obs = np.zeros(8, dtype=np.float32)
        agent.store(obs, 1, 0.1, obs, False, regime=2)
        assert agent.replay.buffer[0].regime == 2

    def test_train_step_returns_loss(self):
        agent = RegimeAwareDQNAgent(
            min_replay_size=5, batch_size=4, seed=0)
        for _ in range(20):
            agent.store(
                np.random.randn(8).astype(np.float32),
                1, 0.1,
                np.random.randn(8).astype(np.float32),
                False,
                regime=3,
            )
        loss = agent.train_step()
        assert loss is not None
        assert isinstance(loss, float)


class TestRegimeAwareDQNAgentSaveLoad:
    def test_load_restores_weights(self, tmp_path):
        agent = RegimeAwareDQNAgent(seed=42)
        obs   = np.zeros(8, dtype=np.float32)
        agent._greedy(obs, 0)   # init weights
        ckpt  = tmp_path / "ra_dqn.pkl"
        agent.save(ckpt)

        loaded = RegimeAwareDQNAgent.load(ckpt)
        x = torch.randn(2, 14, device=loaded.device)
        with torch.no_grad():
            assert torch.allclose(agent.q_net(x), loaded.q_net(x))

    def test_load_with_classifier(self, tmp_path):
        agent = RegimeAwareDQNAgent(seed=0)
        agent.set_classifier(lambda obs: 5)
        ckpt = tmp_path / "ra_dqn.pkl"
        agent.save(ckpt)

        fake_classifier = lambda obs: 5
        loaded = RegimeAwareDQNAgent.load(ckpt, classifier_fn=fake_classifier)
        assert loaded._classifier_fn is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

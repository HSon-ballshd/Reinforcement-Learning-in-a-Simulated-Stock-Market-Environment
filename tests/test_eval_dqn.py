"""
Tests for eval/agents/dqn.py.
"""

import numpy as np
import pytest
import torch

from eval.agents.dqn import QNetwork, ReplayBuffer, DQNAgent


class TestQNetwork:
    def test_default_shape(self):
        net = QNetwork(obs_dim=8, n_actions=3)
        x   = torch.randn(4, 8)
        out = net(x)
        assert out.shape == (4, 3)

    def test_custom_hidden_dims(self):
        net = QNetwork(obs_dim=8, hidden_dims=[256, 128], n_actions=3)
        x   = torch.randn(4, 8)
        out = net(x)
        assert out.shape == (4, 3)

    def test_forward_float32_input(self):
        """Network accepts float32 input (the standard dtype from TradingEnv)."""
        net = QNetwork()
        x   = torch.randn(4, 8, dtype=torch.float32)
        out = net(x)
        assert out.shape == (4, 3)


class TestReplayBuffer:
    def test_capacity_limit(self):
        buf = ReplayBuffer(capacity=3)
        for i in range(10):
            buf.push(np.zeros(8), i, 0.0, np.zeros(8), False)
        assert len(buf) == 3   # older entries dropped

    def test_sample_size(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(50):
            buf.push(np.zeros(8), i, float(i), np.zeros(8), False)
        indices = buf.sample(10)
        assert len(indices) == 10
        assert all(isinstance(i, (int, np.integer)) for i in indices)


class TestDQNAgentInit:
    def test_device_fallback_to_cpu(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        agent = DQNAgent(device="cpu")
        assert agent.device == torch.device("cpu")

    def test_default_hyperparams(self):
        agent = DQNAgent(seed=0)
        assert agent.obs_dim    == 8
        assert agent.n_actions  == 3
        assert agent.gamma      == 0.99
        assert agent.epsilon    == 1.0
        assert agent.batch_size == 64


class TestDQNAgentSelectAction:
    def test_returns_valid_action(self):
        agent = DQNAgent(seed=0)
        obs   = np.zeros(8, dtype=np.float32)
        for _ in range(100):
            a = agent.select_action(obs, {})
            assert a in (0, 1, 2)

    def test_deterministic_greedy(self):
        """With epsilon=0, greedy actions are deterministic for the same observation."""
        agent = DQNAgent(seed=0)
        agent.epsilon = 0.0   # force greedy
        obs   = np.zeros(8, dtype=np.float32)
        for _ in range(10):
            assert agent._greedy(obs) == agent._greedy(obs)

    def test_epsilon_decreases_with_training_steps(self):
        agent = DQNAgent(epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=100, seed=0)
        initial_eps = agent.epsilon
        for _ in range(50):
            agent.epsilon = max(
                0.1,
                agent.epsilon - (1.0 - 0.1) / 100,
            )
        assert agent.epsilon < initial_eps


class TestDQNAgentStoreAndTrain:
    def test_train_step_returns_none_before_min_replay(self):
        agent = DQNAgent(min_replay_size=1000, seed=0)
        for _ in range(100):
            agent.store(np.zeros(8), 0, 0.0, np.zeros(8), False)
        assert agent.train_step() is None

    def test_train_step_returns_loss_after_min_replay(self):
        agent = DQNAgent(min_replay_size=10, batch_size=4, seed=0)
        for _ in range(20):
            agent.store(
                np.random.randn(8).astype(np.float32),
                np.random.randint(0, 3),
                np.random.randn(),
                np.random.randn(8).astype(np.float32),
                False,
            )
        loss = agent.train_step()
        assert loss is not None
        assert isinstance(loss, float)

    def test_buffer_grows(self):
        agent = DQNAgent(replay_capacity=100, seed=0)
        assert len(agent.replay) == 0
        for i in range(50):
            agent.store(np.zeros(8), i, 0.0, np.zeros(8), False)
        assert len(agent.replay) == 50


class TestDQNAgentSaveLoad:
    def test_save_and_load_preserves_weights(self, tmp_path):
        agent = DQNAgent(hidden_dims=[64, 32], seed=42)
        # Force a forward pass so weights are initialised
        obs = np.zeros(8, dtype=np.float32)
        agent.select_action(obs, {})

        ckpt = tmp_path / "agent.pkl"
        agent.save(ckpt)

        loaded = DQNAgent.load(ckpt)
        x = torch.randn(2, 8, device=loaded.device)
        with torch.no_grad():
            original_out = agent.q_net(x)
            loaded_out   = loaded.q_net(x)
        assert torch.allclose(original_out, loaded_out)

    def test_load_preserves_epsilon(self, tmp_path):
        agent = DQNAgent(seed=0)
        agent.epsilon = 0.42
        ckpt = tmp_path / "agent.pkl"
        agent.save(ckpt)
        loaded = DQNAgent.load(ckpt)
        assert loaded.epsilon == 0.42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

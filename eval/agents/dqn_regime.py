"""
Regime-aware DQN agent.

Hypothesis H3: regime-aware DQN beats plain DQN by conditioning Q(s,r|a)
on an inferred regime embedding.

Architecture change vs plain DQN:
    obs (8-dim) + regime_onehot (4-dim) → 12-dim state
    Q(s,r) network: 12 → 128 → 64 → 3

The classifier is injected via set_classifier(), so this module
does NOT import RegimeClassifierPipeline (avoids circular deps).
"""

from __future__ import annotations

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from collections import namedtuple
from typing import Optional

from eval.agents.dqn import ReplayBuffer

Transition = namedtuple("Transition", ["obs", "regime", "action", "reward", "next_obs", "next_regime", "done"])


class RegimeAwareQNetwork(nn.Module):
    """
    Q-network that takes observation + regime one-hot as input.
    Identical MLP shape to QNetwork but with obs_dim + n_regimes inputs.
    """

    def __init__(self, obs_dim: int = 8, hidden_dims: list[int] | None = None, n_actions: int = 3, n_regimes: int = 4):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        layers = []
        prev = obs_dim + n_regimes   # obs + one-hot regime
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RegimeAwareReplayBuffer(ReplayBuffer):
    """Replay buffer that stores regime alongside transitions."""

    def push(self, obs, regime: int, action, reward, next_obs, next_regime: int, done) -> None:
        self.buffer.append(Transition(obs, regime, action, reward, next_obs, next_regime, done))


class RegimeAwareDQNAgent:
    """
    DQN agent that conditions its Q-function on the inferred regime.

    set_classifier(classifier_fn) must be called before use.
    classifier_fn should match RegimeClassifierPipeline.predict: (obs) → int
    """

    def __init__(
        self,
        obs_dim: int = 8,
        n_actions: int = 3,
        n_regimes: int = 4,
        hidden_dims: list[int] | None = None,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 10_000,
        replay_capacity: int = 100_000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        min_replay_size: int = 1000,
        gradient_clip: float = 1.0,
        device: str | None = None,
        seed: int | None = None,
    ) -> None:
        self.obs_dim   = obs_dim
        self.n_actions = n_actions
        self.n_regimes = n_regimes
        self.hidden_dims = hidden_dims or [128, 64]
        self.gamma             = gamma
        self.epsilon_start     = epsilon_start
        self.epsilon_end       = epsilon_end
        self.epsilon_decay     = epsilon_decay
        self.batch_size        = batch_size
        self.target_update_freq = target_update_freq
        self.min_replay_size  = min_replay_size
        self.gradient_clip    = gradient_clip

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self._rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)

        # Networks — 14-dim input (8 obs + 6 regime)
        self.n_regimes = n_regimes
        self.q_net     = RegimeAwareQNetwork(obs_dim, hidden_dims, n_actions, n_regimes).to(self.device)
        self.target_net = RegimeAwareQNetwork(obs_dim, hidden_dims, n_actions, n_regimes).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.replay: RegimeAwareReplayBuffer = RegimeAwareReplayBuffer(replay_capacity)

        self.global_step = 0
        self.epsilon     = epsilon_start

        # Classifier (injected)
        self._classifier_fn: Optional[callable] = None

    # ------------------------------------------------------------------
    # Classifier injection
    # ------------------------------------------------------------------
    def set_classifier(self, classifier_fn: callable) -> None:
        """
        Inject a regime classifier function.

        Args:
            classifier_fn: callable that takes a (18,) feature array
                           and returns an int in [0, 3] (Stable/Bull/Bear/Chaotic).
        """
        self._classifier_fn = classifier_fn

    def set_env(self, env) -> None:
        """Store reference to the current TradingEnv for feature extraction."""
        self._env = env

    # ------------------------------------------------------------------
    # BaseAgent-like interface
    # ------------------------------------------------------------------
    def select_action(self, observation: np.ndarray, info: dict) -> int:
        """Select action using inferred regime (greedy, no exploration)."""
        return self._epsilon_greedy(observation, training=False)

    def reset(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _regime_onehot(self, regime: int) -> np.ndarray:
        vec = np.zeros(self.n_regimes, dtype=np.float32)
        vec[regime] = 1.0
        return vec

    def _build_state(self, obs: np.ndarray, regime: int) -> np.ndarray:
        """Concatenate observation with regime one-hot."""
        obs_f = np.asarray(obs, dtype=np.float32)
        reg_f = self._regime_onehot(regime)
        return np.concatenate([obs_f, reg_f])

    def _infer_regime(self, obs: np.ndarray) -> int:
        """Use injected classifier, default to 0 (Stable) if not set."""
        if self._classifier_fn is None:
            return 0
        return self._classifier_fn(obs)

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def _greedy(self, obs: np.ndarray, regime: int) -> int:
        state = self._build_state(obs, regime)
        x = self._to_tensor(state).unsqueeze(0)
        with torch.no_grad():
            q = self.q_net(x)
        return int(q.argmax(dim=1).item())

    def _epsilon_greedy(self, obs: np.ndarray, training: bool = True) -> int:
        regime = self._infer_regime(obs)
        eps    = self.epsilon if training else self.epsilon_end
        if self._rng.random() < eps:
            return int(self._rng.integers(0, self.n_actions))
        return self._greedy(obs, regime)

    # ------------------------------------------------------------------
    # Store and train
    # ------------------------------------------------------------------
    def store(self, obs, action, reward, next_obs, done, regime: int | None = None, next_regime: int | None = None) -> None:
        if regime is None:
            regime = self._infer_regime(obs)
        if next_regime is None:
            next_regime = self._infer_regime(next_obs)
        self.replay.push(obs, regime, action, reward, next_obs, next_regime, done)

    def train_step(self) -> Optional[float]:
        if len(self.replay) < self.min_replay_size:
            return None

        indices = self.replay.sample(self.batch_size)
        batch   = [self.replay.buffer[i] for i in indices]

        obs_batch      = np.array([t.obs     for t in batch], dtype=np.float32)
        regime_batch   = np.array([t.regime  for t in batch], dtype=np.int64)
        action_batch   = torch.as_tensor(
            [t.action for t in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch   = torch.as_tensor(
            [t.reward for t in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_obs_batch = np.array([t.next_obs for t in batch], dtype=np.float32)
        done_batch     = torch.as_tensor(
            [float(t.done) for t in batch], dtype=torch.float32, device=self.device).unsqueeze(1)

        # Build state tensors
        def state_from_batch(obs_arr, regimes):
            states = np.concatenate([obs_arr, np.eye(self.n_regimes)[regimes]], axis=1)
            return self._to_tensor(states)

        state_batch     = state_from_batch(obs_batch, regime_batch)
        next_regimes_stored = np.array([t.next_regime for t in batch], dtype=np.int64)
        next_state_batch = state_from_batch(next_obs_batch, next_regimes_stored)

        # Q(s, a)
        q_values = self.q_net(state_batch).gather(dim=1, index=action_batch)

        # Target
        with torch.no_grad():
            next_q   = self.target_net(next_state_batch).max(dim=1, keepdim=True)[0]
            target   = reward_batch + self.gamma * next_q * (1.0 - done_batch)

        loss = nn.functional.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.gradient_clip)
        self.optimizer.step()

        # Epsilon decay
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - (self.epsilon_start - self.epsilon_end) / self.epsilon_decay,
        )
        self.global_step += 1

        if self.global_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "q_net_state":      self.q_net.state_dict(),
            "target_net_state": self.target_net.state_dict(),
            "optimizer_state":   self.optimizer.state_dict(),
            "epsilon":           self.epsilon,
            "global_step":       self.global_step,
            "obs_dim":           self.obs_dim,
            "n_actions":         self.n_actions,
            "n_regimes":         self.n_regimes,
            "hidden_dims":       self.hidden_dims,
        }
        pickle.dump(state, open(path, "wb"))

    @classmethod
    def load(cls, path: str | Path, device: str | None = None,
             classifier_fn: callable | None = None) -> "RegimeAwareDQNAgent":
        state = pickle.load(open(Path(path), "rb"))
        agent = cls(
            obs_dim=state["obs_dim"],
            n_actions=state["n_actions"],
            n_regimes=state["n_regimes"],
            hidden_dims=state["hidden_dims"],
            device=device,
        )
        agent.q_net.load_state_dict(state["q_net_state"])
        agent.target_net.load_state_dict(state["target_net_state"])
        agent.optimizer.load_state_dict(state["optimizer_state"])
        agent.epsilon      = state["epsilon"]
        agent.global_step  = state["global_step"]
        if classifier_fn is not None:
            agent.set_classifier(classifier_fn)
        return agent

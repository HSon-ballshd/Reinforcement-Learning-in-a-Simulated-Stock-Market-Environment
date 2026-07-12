"""
DQN agent for Cookie Clicker trading.

Architecture:
    Q(s) network:  8-dim input → 128 → 64 → 3 (HOLD/BUY/SELL)
    Target network: copy of Q, updated every target_update_freq steps.
    Replay buffer: uniform sample of (s, a, r, s', done).

Training follows Mnih et al. 2015 "Human-level control through deep
reinforcement learning" with:
    - Fixed target network
    - Experience replay
    - Gradient clipping
    - Epsilon-greedy exploration (linear decay)
"""

from __future__ import annotations

import pickle
import csv
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from collections import deque, namedtuple
from typing import Optional

from eval.agents.baselines import BaseAgent

# ------------------------------------------------------------------
# Torch module: Q-network
# ------------------------------------------------------------------
class QNetwork(nn.Module):
    """Simple 2-hidden-layer MLP for Q(s, a)."""

    def __init__(self, obs_dim: int = 8, hidden_dims: list[int] | None = None, n_actions: int = 3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        layers = []
        prev = obs_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------------------------------------------
# Replay buffer
# ------------------------------------------------------------------
Transition = namedtuple("Transition", ["obs", "action", "reward", "next_obs", "done"])


class ReplayBuffer:
    """Circular buffer of fixed capacity."""

    def __init__(self, capacity: int = 100_000) -> None:
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, *args) -> None:
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> list[Transition]:
        return list(np.random.choice(len(self.buffer), batch_size, replace=False))

    def __len__(self) -> int:
        return len(self.buffer)


# ------------------------------------------------------------------
# DQNAgent
# ------------------------------------------------------------------
class DQNAgent:
    """
    DQN agent compatible with the BaseAgent interface (can be used
    in evaluate_agent() and the evaluation harness).
    """

    def __init__(
        self,
        obs_dim: int = 8,
        n_actions: int = 3,
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
        # Network params
        self.obs_dim   = obs_dim
        self.n_actions = n_actions
        self.hidden_dims = hidden_dims or [128, 64]
        self.gamma           = gamma
        self.epsilon_start   = epsilon_start
        self.epsilon_end     = epsilon_end
        self.epsilon_decay   = epsilon_decay
        self.batch_size      = batch_size
        self.target_update_freq = target_update_freq
        self.min_replay_size = min_replay_size
        self.gradient_clip   = gradient_clip

        # Device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Save seed for reproducibility
        self.seed = seed

        # Force deterministic on GPU (cuDNN can be non-deterministic)
        if self.device.type == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # RNG — save/restore RNG state in save/load for full reproducibility
        self._rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        # Networks
        self.q_net      = QNetwork(obs_dim, self.hidden_dims, n_actions).to(self.device)
        self.target_net  = QNetwork(obs_dim, self.hidden_dims, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer  = optim.Adam(self.q_net.parameters(), lr=lr)

        # Replay
        self.replay = ReplayBuffer(replay_capacity)

        # Training state
        self.global_step = 0
        self.epsilon      = epsilon_start

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------
    def select_action(self, observation: np.ndarray, info: dict) -> int:
        """Epsilon-greedy.  Exposed so evaluate_agent() can call it."""
        return self._epsilon_greedy(observation, training=False)

    def reset(self) -> None:
        """Called at the start of each episode."""
        pass

    # ------------------------------------------------------------------
    # Training loop helpers
    # ------------------------------------------------------------------
    def _epsilon_greedy(self, obs: np.ndarray, training: bool = True) -> int:
        epsilon = self.epsilon if training else self.epsilon_end
        if self._rng.random() < epsilon:
            return int(self._rng.integers(0, self.n_actions))
        return self._greedy(obs)

    def _greedy(self, obs: np.ndarray) -> int:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.q_net(x)
        return int(q.argmax(dim=1).item())

    def store(self, obs, action, reward, next_obs, done) -> None:
        self.replay.push(obs, action, reward, next_obs, done)

    def train_step(self) -> Optional[float]:
        """
        Perform one gradient update if buffer has enough samples.

        Returns the TD loss (scalar) or None if not enough samples.
        """
        if len(self.replay) < self.min_replay_size:
            return None

        # Sample batch
        indices = self.replay.sample(self.batch_size)
        batch = [self.replay.buffer[i] for i in indices]

        obs_batch     = torch.as_tensor(
            np.array([t.obs for t in batch]), dtype=torch.float32, device=self.device)
        action_batch  = torch.as_tensor(
            [t.action for t in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch  = torch.as_tensor(
            [t.reward for t in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_obs_batch = torch.as_tensor(
            np.array([t.next_obs for t in batch]), dtype=torch.float32, device=self.device)
        done_batch    = torch.as_tensor(
            [float(t.done) for t in batch], dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s, a)
        q_values  = self.q_net(obs_batch).gather(dim=1, index=action_batch)

        # Target: r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q     = self.target_net(next_obs_batch).max(dim=1, keepdim=True)[0]
            target     = reward_batch + self.gamma * next_q * (1.0 - done_batch)

        # MSE loss
        loss = nn.functional.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.gradient_clip)
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - (self.epsilon_start - self.epsilon_end) / self.epsilon_decay,
        )
        self.global_step += 1

        # Hard-copy target network
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
            "target_net_state":  self.target_net.state_dict(),
            "optimizer_state":  self.optimizer.state_dict(),
            "epsilon":           self.epsilon,
            "global_step":      self.global_step,
            "obs_dim":          self.obs_dim,
            "n_actions":        self.n_actions,
            "hidden_dims":      self.hidden_dims,
            "seed":             self.seed,
            "rng_state":        self._rng.bit_generator.state,
            "torch_rng_state":  torch.get_rng_state(),
            "cuda_rng_state":   torch.cuda.get_rng_state() if self.device.type == "cuda" else None,
        }
        pickle.dump(state, open(path, "wb"))

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "DQNAgent":
        state = pickle.load(open(Path(path), "rb"))
        agent = cls(
            obs_dim=state["obs_dim"],
            n_actions=state["n_actions"],
            hidden_dims=state["hidden_dims"],
            device=device,
            seed=state["seed"],
        )
        agent.q_net.load_state_dict(state["q_net_state"])
        agent.target_net.load_state_dict(state["target_net_state"])
        agent.optimizer.load_state_dict(state["optimizer_state"])
        agent.epsilon     = state["epsilon"]
        agent.global_step = state["global_step"]
        # Restore RNG state for deterministic replay
        agent._rng.bit_generator.state = state["rng_state"]
        torch.set_rng_state(state["torch_rng_state"])
        if state["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state(state["cuda_rng_state"], device=agent.device)
        return agent


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------
def train_dqn(
    agent: DQNAgent,
    market_seed: int = 0,
    n_steps: int = 100_000,
    eval_every: int = 10_000,
    max_episode_steps: int | None = None,
    eval_steps: int = 1000,
    initial_cash: float = 10_000.0,
    train_seeds: list[int] | None = None,
    verbose: bool = True,
    log_path: str | Path | None = None,
) -> dict:
    """
    Run DQN training against the Cookie Clicker market.

    Logs are written incrementally to a CSV file (one row per eval checkpoint)
    so partial progress is preserved if training is interrupted.

    Args:
        train_seeds: seeds to use for eval during training (must NOT include
                     the training seed to avoid leakage).
        log_path:    if provided, append eval checkpoints to this CSV file.
    """
    from sim.market_sim import CookieClickerMarket
    from eval.env.trading_env import TradingEnv

    logs = {
        "loss":     [],
        "epsilon":  [],
        "episode_return": [],
        "eval_returns":   [],
    }

    best_eval = -np.inf
    market = CookieClickerMarket(n_stocks=1, seed=market_seed)
    env    = TradingEnv(
        market,
        initial_cash=initial_cash,
        max_steps=max_episode_steps,
        seed=market_seed,
    )

    episode_return = 0.0
    agent.reset()
    obs = env.reset()

    # Default eval pool — exclude training seed
    eval_seed_pool = [42, 123, 456, 789, 1024]
    if train_seeds:
        eval_seed_pool = [s for s in eval_seed_pool if s not in train_seeds]

    # Open CSV log for incremental writes
    csv_file = None
    csv_writer = None
    if log_path:
        csv_file = open(log_path, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=["step", "loss", "epsilon", "episode_return", "eval_return_pct"])
        csv_writer.writeheader()

    iterator = tqdm(range(n_steps), desc="DQN", unit="step", disable=not verbose)

    try:
        for step in iterator:
            # Select and execute action
            action = agent._epsilon_greedy(obs, training=True)
            next_obs, reward, done, info = env.step(action)
            episode_return += reward

            agent.store(obs, action, reward, next_obs, done)
            obs = next_obs

            # Train
            loss = agent.train_step()
            if loss is not None:
                logs["loss"].append(float(loss))

            logs["epsilon"].append(agent.epsilon)

            if done:
                logs["episode_return"].append(float(episode_return))
                episode_return = 0.0
                env.reset()
                agent.reset()
                obs = env.reset()

            # Periodic evaluation on held-out seeds
            if (step + 1) % eval_every == 0:
                returns = _eval_agent(agent, eval_seed_pool[:3], eval_steps, initial_cash)
                mean_ret = float(np.mean(returns))
                logs["eval_returns"].append(mean_ret)
                if mean_ret > best_eval:
                    best_eval = mean_ret
                iterator.set_postfix(epsilon=f"{agent.epsilon:.3f}", eval_ret=f"{mean_ret:.2f}%")

                # Incrementally write to CSV
                if csv_writer is not None:
                    last_loss = logs["loss"][-1] if logs["loss"] else ""
                    last_ep  = logs["episode_return"][-1] if logs["episode_return"] else ""
                    csv_writer.writerow({
                        "step": step + 1,
                        "loss": last_loss,
                        "epsilon": agent.epsilon,
                        "episode_return": last_ep,
                        "eval_return_pct": mean_ret,
                    })
                    csv_file.flush()

    finally:
        if csv_file is not None:
            csv_file.close()

    return {
        "logs":      logs,
        "best_eval": float(best_eval),
        "total_steps": n_steps,
    }


def _eval_agent(
    agent: DQNAgent,
    seeds: list[int],
    n_steps: int,
    initial_cash: float,
) -> list[float]:
    """Run agent greedily (no exploration) and return portfolio return %.

    Uses actual final portfolio value for a fair comparison with baselines.
    """
    from sim.market_sim import CookieClickerMarket
    from eval.env.trading_env import TradingEnv

    returns = []
    for seed in seeds:
        market = CookieClickerMarket(n_stocks=1, seed=seed)
        env    = TradingEnv(
            market,
            initial_cash=initial_cash,
            max_steps=n_steps,
            seed=seed,
        )
        agent.reset()
        if hasattr(agent, '_env') or hasattr(agent, 'set_env'):
            agent._env = env
        obs = env.reset()
        done = False
        info = {
            'portfolio_value': env._portfolio_value(),
            'cash': env.cash,
            'holdings': env.holdings,
            'price': env.market.stocks[0]['price'],
            'step': 0,
        }
        while not done:
            action = agent.select_action(obs, info)
            obs, _, done, info = env.step(action)
        # Compute actual portfolio return %
        final_value   = env._portfolio_value()
        total_return  = (final_value - initial_cash) / initial_cash * 100.0
        returns.append(total_return)
    return returns

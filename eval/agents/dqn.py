"""DQN trader with two variants: baseline (market+portfolio) and regime (adds 6 probs)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _q_network(input_dim: int, n_actions: int = 3, hidden: int = 64) -> nn.Module:
 return nn.Sequential(
  nn.Linear(input_dim, hidden),
  nn.ReLU(),
  nn.Linear(hidden, hidden),
  nn.ReLU(),
  nn.Linear(hidden, n_actions),
 )


@dataclass
class DQNConfig:
 input_dim: int = 11 # 8 market + 3 portfolio
 n_actions: int = 3
 hidden: int = 64
 gamma: float = 0.99
 lr: float = 1e-3
 batch_size: int = 64
 buffer_size: int = 10_000
 warmup: int = 200
 eps_start: float = 1.0
 eps_end: float = 0.05
 eps_decay_steps: int = 5_000
 target_update_every: int = 500
 device: str = "cpu"


class ReplayBuffer:
 def __init__(self, capacity: int) -> None:
  self.buf: deque = deque(maxlen=capacity)

 def push(self, s, a, r, s2, d) -> None:
  self.buf.append((s, a, r, s2, d))

 def sample(self, batch: int) -> tuple:
  idx = np.random.choice(len(self.buf), size=batch, replace=False)
  s, a, r, s2, d = zip(*[self.buf[i] for i in idx])
  return (
   np.asarray(s, dtype=np.float32),
   np.asarray(a, dtype=np.int64),
   np.asarray(r, dtype=np.float32),
   np.asarray(s2, dtype=np.float32),
   np.asarray(d, dtype=np.float32),
  )

 def __len__(self) -> int:
  return len(self.buf)


class DQNAgent:
 def __init__(self, config: DQNConfig, use_regime: bool = False) -> None:
  self.cfg = config
  self.use_regime = use_regime
  self.device = torch.device(config.device)
  self.online = _q_network(config.input_dim, config.n_actions, config.hidden).to(self.device)
  self.target = _q_network(config.input_dim, config.n_actions, config.hidden).to(self.device)
  self.target.load_state_dict(self.online.state_dict())
  self.opt = torch.optim.Adam(self.online.parameters(), lr=config.lr)
  self.buffer = ReplayBuffer(config.buffer_size)
  self._steps = 0
  self._losses: list = []

 @property
 def epsilon(self) -> float:
  cfg = self.cfg
  frac = min(1.0, self._steps / max(1, cfg.eps_decay_steps))
  return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)

 def reset(self) -> None:
  pass

 def act(self, obs: np.ndarray, *, greedy: bool = False) -> int:
  if not greedy and np.random.random() < self.epsilon:
   return int(np.random.randint(self.cfg.n_actions))
  with torch.no_grad():
   q = self.online(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
   return int(q.argmax(dim=1).item())

 def push(self, s, a, r, s2, d) -> None:
  self.buffer.push(s, a, r, s2, d)

 def train_step(self) -> Optional[float]:
  if len(self.buffer) < max(self.cfg.warmup, self.cfg.batch_size):
   return None
  s, a, r, s2, d = self.buffer.sample(self.cfg.batch_size)
  s_t = torch.as_tensor(s, device=self.device)
  a_t = torch.as_tensor(a, device=self.device)
  r_t = torch.as_tensor(r, device=self.device)
  s2_t = torch.as_tensor(s2, device=self.device)
  d_t = torch.as_tensor(d, device=self.device)
  q = self.online(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
  with torch.no_grad():
   q_next = self.target(s2_t).max(dim=1)[0]
   target = r_t + (1.0 - d_t) * self.cfg.gamma * q_next
  loss = F.smooth_l1_loss(q, target)
  self.opt.zero_grad()
  loss.backward()
  nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
  self.opt.step()
  self._steps += 1
  if self._steps % self.cfg.target_update_every == 0:
   self.target.load_state_dict(self.online.state_dict())
  self._losses.append(float(loss.item()))
  return float(loss.item())

 @property
 def mean_recent_loss(self) -> float:
  return float(np.mean(self._losses[-100:])) if self._losses else 0.0
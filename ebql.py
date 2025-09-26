# ebql.py
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Deque, Tuple, Optional, List
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import ale_py

import csv
import os 


# =========================
# Small helpers & modules
# =========================

def to_t(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)

@dataclass
class EBQLConfig:
    K: int = 5                          # ensemble size (#heads)
    gamma: float = 0.99
    lr: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 150_000
    learning_starts: int = 20_000
    train_freq: int = 1
    target_update_interval: int = 10_000
    tau: float = 1.0                    # 1.0 = hard update; <1.0 = soft polyak
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 750_000
    bootstrap_prob: float = 0.5         # P(mask=1) per head per transition
    hidden_sizes: Tuple[int, int] = (256, 256)
    clip_grad_norm: Optional[float] = 10.0     # ← NEW (set None to disable)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0


class MLPHeads(nn.Module):
    """
    Shared encoder for RAM obs (shape 128), K independent heads over actions.
    Outputs shape: [B, K, A].
    """
    def __init__(self, obs_dim: int, n_actions: int, K: int, hidden=(256, 256)):
        super().__init__()
        layers: List[nn.Module] = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.encoder = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(last, n_actions) for _ in range(K)])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.encoder(obs)  # [B, H]
        qs = [head(z) for head in self.heads]  # list of [B, A]
        return torch.stack(qs, dim=1)          # [B, K, A]


class ReplayBuffer:
    """Replay with per-head bootstrap masks."""
    def __init__(self, capacity: int, obs_dim: int, K: int, bootstrap_prob: float, seed: int, device: torch.device):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        # masks[t, h] in {0,1}
        self.masks = np.zeros((capacity, K), dtype=np.float32)
        self.bootstrap_prob = bootstrap_prob
        self.K = K
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.size = 0
        self.ptr = 0

    def add(self, o, a, r, o2, d):
        self.obs[self.ptr] = o
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_obs[self.ptr] = o2
        self.dones[self.ptr] = d
        self.masks[self.ptr] = self.rng.binomial(1, self.bootstrap_prob, size=self.K)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self): return self.size

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=batch_size)
        batch = dict(
            obs      = to_t(self.obs[idx], self.device),
            actions  = torch.as_tensor(self.actions[idx], dtype=torch.long, device=self.device),
            rewards  = to_t(self.rewards[idx], self.device),
            next_obs = to_t(self.next_obs[idx], self.device),
            dones    = to_t(self.dones[idx], self.device),
            masks    = to_t(self.masks[idx], self.device)  # [B, K]
        )
        return batch


# =========================
# EBQL Agent
# =========================

class EBQLAgent:
    """
    Ensemble Bootstrapped Q-Learning (EBQL), matching Algorithm 1 in the paper.

    - Behavior: ε-greedy on ensemble-mean Q(s, a)
    - Update (one SGD step):
        * sample one head k to update
        * select a* = argmax_a Q_k_online(s', a)
        * evaluate target with mean of OTHER heads' target nets at a*
        * update only head k using bootstrap mask for that head
    """
    def __init__(self, env: gym.Env, cfg: EBQLConfig):
        assert hasattr(env.action_space, "n"), "Discrete action space required."
        self.env = env
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.rng = np.random.default_rng(cfg.seed)
        random.seed(cfg.seed); torch.manual_seed(cfg.seed)

        obs_example, _ = env.reset(seed=cfg.seed)
        self.obs_dim = int(np.prod(obs_example.shape))  # RAM -> 128
        self.n_actions = env.action_space.n
        self.K = cfg.K

        # Networks
        self.q = MLPHeads(self.obs_dim, self.n_actions, self.K, cfg.hidden_sizes).to(self.device)
        self.q_targ = MLPHeads(self.obs_dim, self.n_actions, self.K, cfg.hidden_sizes).to(self.device)
        self.q_targ.load_state_dict(self.q.state_dict()); self.q_targ.eval()
        self.optim = optim.Adam(self.q.parameters(), lr=cfg.lr)

        # Replay
        self.replay = ReplayBuffer(cfg.buffer_size, self.obs_dim, self.K, cfg.bootstrap_prob, cfg.seed, self.device)

        # Schedules
        self.step_count = 0
        self.eps = cfg.epsilon_start

        # Bookkeeping
        self._last_obs = obs_example.astype(np.float32)

    # ---------- acting ----------
    @torch.no_grad()
    def _select_action(self, obs: np.ndarray, exploit_only: bool = False) -> int:
        o = to_t(obs[None, ...], self.device)                    # [1, obs_dim]
        q_all = self.q(o)                                        # [1, K, A]
        q_mean = q_all.mean(dim=1)                               # [1, A] (ensemble mean)
        if (not exploit_only) and (self.rng.random() < self.eps):
            return int(self.rng.integers(0, self.n_actions))
        return int(torch.argmax(q_mean, dim=-1).item())

    def _update_epsilon(self):
        t = min(self.step_count, self.cfg.epsilon_decay_steps)
        frac = 1.0 - (t / float(self.cfg.epsilon_decay_steps))
        self.eps = self.cfg.epsilon_end + (self.cfg.epsilon_start - self.cfg.epsilon_end) * max(0.0, frac)

    # ---------- training ----------
    def train(self, total_steps: int, eval_env: Optional[gym.Env] = None, eval_every: int = 0, eval_episodes: int = 0):
        o = self._last_obs
        while self.step_count < total_steps:
            # --- interact ---
            a = self._select_action(o, exploit_only=False)
            o2, r, terminated, truncated, _ = self.env.step(a)
            d = float(terminated or truncated)
            self.replay.add(o, a, r, o2, d)
            o = o2
            self.step_count += 1
            self._update_epsilon()

            if self.step_count % 100 == 0:
                print(f"Step: {self.step_count}/{total_steps}: {r}")


            if terminated or truncated:
                o, _ = self.env.reset()

            # --- learn ---
            if self.step_count >= self.cfg.learning_starts and self.step_count % self.cfg.train_freq == 0:
                self._sgd_step()

            # --- target update ---
            if self.step_count % self.cfg.target_update_interval == 0:
                self._update_target()

            # --- optional eval ---
            if eval_env is not None and eval_every > 0 and self.step_count % eval_every == 0:
                avg_ret = self.evaluate(eval_env, eval_episodes or 10)
                print(f"[eval @ {self.step_count}] reward={avg_ret:.1f}")

        self._last_obs = o  # store for potential continuation

    def _update_target(self):
        if self.cfg.tau >= 1.0:
            self.q_targ.load_state_dict(self.q.state_dict())
        else:
            with torch.no_grad():
                for p, tp in zip(self.q.parameters(), self.q_targ.parameters()):
                    tp.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p)

    def _sgd_step(self):
        batch = self.replay.sample(self.cfg.batch_size)
        obs, actions, rewards, next_obs, dones, masks = (
            batch["obs"], batch["actions"], batch["rewards"],
            batch["next_obs"], batch["dones"], batch["masks"]
        )  # masks: [B, K]

        # --- choose a head k to update; ensure it has nonzero mask if possible ---
        K = self.K
        tries = 0
        k = int(self.rng.integers(0, K))
        while masks[:, k].sum() == 0 and tries < 4:
            k = int(self.rng.integers(0, K)); tries += 1

        # Online Q for head k
        q_all = self.q(obs)                 # [B, K, A]
        qk = q_all[:, k, :]                 # [B, A]
        q_sa = qk.gather(1, actions.view(-1, 1)).squeeze(1)  # [B]

        with torch.no_grad():
            # Selection with ONLINE head k
            q_next_online_k = self.q(next_obs)[:, k, :]      # [B, A]
            a_star = torch.argmax(q_next_online_k, dim=1)    # [B]

            # Evaluation with TARGET mean over OTHER heads
            q_next_targ_all = self.q_targ(next_obs)          # [B, K, A]
            if K == 1:
                # fallback to DQN target if K=1 (not EBQL proper)
                q_eval = q_next_targ_all[:, 0, :]
            else:
                idx = [j for j in range(K) if j != k]
                q_others = q_next_targ_all[:, idx, :]        # [B, K-1, A]
                q_eval = q_others.mean(dim=1)                # [B, A]

            q_next = q_eval.gather(1, a_star.view(-1, 1)).squeeze(1)  # [B]
            target = rewards + (1.0 - dones) * self.cfg.gamma * q_next  # [B]

        # Masked MSE loss for head k only
        mk = masks[:, k]                                      # [B]
        loss = ((q_sa - target) ** 2 * mk).sum() / (mk.sum() + 1e-8)

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.clip_grad_norm)
        self.optim.step()

    # ---------- evaluation ----------
    @torch.no_grad()
    def evaluate(self, eval_env: gym.Env, episodes: int = 10) -> float:
        """Deterministic eval: ε=0 and greedy wrt ensemble-mean Q."""
        total = 0.0
        for _ in range(episodes):
            o, _ = eval_env.reset()
            done = False
            ep_ret = 0.0
            while not done:
                a = self._select_action(o.astype(np.float32), exploit_only=True)
                o, r, term, trunc, _ = eval_env.step(a)
                ep_ret += float(r)
                done = term or trunc
            total += ep_ret
        return total / episodes


# =========================
# Minimal usage example
# =========================
if __name__ == "__main__":
    # IMPORTANT: create the env with RAM observations
    env = gym.make("ALE/SpaceInvaders-v5", obs_type="ram")
    eval_env = gym.make("ALE/SpaceInvaders-v5", obs_type="ram")

    cfg = EBQLConfig(K=5)  # try K=1 (DQN-ish), K=2 (Double Q), K=5/10/20
    agent = EBQLAgent(env, cfg)

    agent.train(
        total_steps=2_000_000,
        eval_env=eval_env,
        eval_every=100_000,
        eval_episodes=20,
    )

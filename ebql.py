# ebql.py
from __future__ import annotations

import argparse
import random
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing


import csv
import os 


# =========================
# Small helpers & modules
# =========================

def to_t(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


class FrameStack(gym.Wrapper):
    """Simplified frame stack wrapper producing channel-first observations."""

    def __init__(self, env: gym.Env, k: int):
        super().__init__(env)
        assert k >= 1, "frame_stack must be >= 1"
        self.k = k
        self.frames: deque[np.ndarray] = deque(maxlen=k)

        assert isinstance(env.observation_space, gym.spaces.Box), "FrameStack only supports Box spaces"
        obs_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(k,) + obs_shape,
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = np.asarray(obs, dtype=np.uint8)
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_observation(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = np.asarray(obs, dtype=np.uint8)
        self.frames.append(obs)
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        assert len(self.frames) == self.k
        return np.stack(self.frames, axis=0)


def make_spaceinvaders_env(seed: int, frame_stack: int) -> gym.Env:
    env = gym.make("ALE/SpaceInvaders-v5", frameskip=1)
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=True,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,
    )
    env = FrameStack(env, frame_stack)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env

@dataclass
class EBQLConfig:
    K: int = 5                       # ensemble size (#heads)
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
    hidden_sizes: Tuple[int, ...] = (512,)
    conv_channels: Tuple[int, int, int] = (32, 64, 64)
    clip_grad_norm: Optional[float] = 10.0     # ← NEW (set None to disable)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    train_csv_path: Optional[str] = "train_rewards.csv"
    eval_csv_path: Optional[str] = "eval_rewards.csv"
    frame_stack: int = 4
    model_path: Optional[str] = "model.pt"


class EnsembleConvHeads(nn.Module):
    """Shared convolutional encoder with independent linear heads."""

    def __init__(self, obs_shape: Tuple[int, ...], n_actions: int, K: int, hidden_sizes: Tuple[int, ...], conv_channels: Tuple[int, int, int]):
        super().__init__()
        assert len(obs_shape) == 3, "Expected CHW observation shape"
        in_channels = obs_shape[0]

        c1, c2, c3 = conv_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(c2, c3, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            conv_out = self.conv(dummy)
            flattened = conv_out.view(1, -1).size(1)

        layers = []
        last_dim = flattened
        for h in hidden_sizes:
            layers.extend([nn.Linear(last_dim, h), nn.ReLU()])
            last_dim = h
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.heads = nn.ModuleList([nn.Linear(last_dim, n_actions) for _ in range(K)])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.conv(obs)
        z = z.view(z.size(0), -1)
        z = self.mlp(z)
        qs = [head(z) for head in self.heads]
        return torch.stack(qs, dim=1)


class ReplayBuffer:
    """Replay with per-head bootstrap masks."""
    def __init__(self, capacity: int, obs_shape: Tuple[int, ...], K: int, bootstrap_prob: float, seed: int, device: torch.device):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
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
            obs      = to_t(self.obs[idx], self.device) / 255.0,
            actions  = torch.as_tensor(self.actions[idx], dtype=torch.long, device=self.device),
            rewards  = to_t(self.rewards[idx], self.device),
            next_obs = to_t(self.next_obs[idx], self.device) / 255.0,
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

    - Behavior: ε-greedy on a per-episode head (Thompson-style exploration)
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
        obs_example = self._convert_obs(obs_example)
        self.obs_shape = obs_example.shape
        self.n_actions = env.action_space.n
        self.K = cfg.K

        # Networks
        self.q = EnsembleConvHeads(self.obs_shape, self.n_actions, self.K, cfg.hidden_sizes, cfg.conv_channels).to(self.device)
        self.q_targ = EnsembleConvHeads(self.obs_shape, self.n_actions, self.K, cfg.hidden_sizes, cfg.conv_channels).to(self.device)
        self.q_targ.load_state_dict(self.q.state_dict()); self.q_targ.eval()
        self.optim = optim.Adam(self.q.parameters(), lr=cfg.lr)

        # Replay
        self.replay = ReplayBuffer(cfg.buffer_size, self.obs_shape, self.K, cfg.bootstrap_prob, cfg.seed, self.device)

        # Schedules
        self.step_count = 0
        self.eps = cfg.epsilon_start

        # Bookkeeping
        self._last_obs = obs_example
        self._episode_idx = 0
        self._ep_return = 0.0
        self._ep_length = 0
        self._active_head = int(self.rng.integers(0, self.K)) if self.K > 0 else 0
        self.train_csv_path = cfg.train_csv_path
        self.eval_csv_path = cfg.eval_csv_path
        self.model_path = cfg.model_path
        if self.model_path:
            Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)

    # ---------- acting ----------
    def _convert_obs(self, obs) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.uint8)
        if arr.ndim == 3 and arr.shape[0] != self.cfg.frame_stack:
            arr = np.transpose(arr, (2, 0, 1))
        return np.ascontiguousarray(arr)

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        return to_t(obs[None, ...] / 255.0, self.device)

    @torch.no_grad()
    def _select_action(self, obs: np.ndarray, exploit_only: bool = False) -> int:
        o = self._obs_to_tensor(obs)                             # [1, C, H, W]
        q_all = self.q(o)                                        # [1, K, A]

        if exploit_only or self._active_head is None:
            q_vals = q_all.mean(dim=1)                           # [1, A]
        else:
            q_vals = q_all[:, self._active_head, :]              # [1, A]

        if (not exploit_only) and (self.rng.random() < self.eps):
            return int(self.rng.integers(0, self.n_actions))
        return int(torch.argmax(q_vals, dim=-1).item())

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
            o2_raw, r, terminated, truncated, _ = self.env.step(a)
            o2 = self._convert_obs(o2_raw)
            d = float(terminated or truncated)
            self.replay.add(o, a, r, o2, d)
            o = o2
            self._ep_return += float(r)
            self._ep_length += 1
            self.step_count += 1
            self._update_epsilon()

            if self.step_count % 100 == 0:
                print(f"Step: {self.step_count}/{total_steps}: {r}")


            if terminated or truncated:
                self._episode_idx += 1
                self._log_train_episode(
                    episode=self._episode_idx,
                    total_steps=self.step_count,
                    episode_steps=self._ep_length,
                    reward=self._ep_return,
                )
                self._ep_return = 0.0
                self._ep_length = 0
                o, _ = self.env.reset()
                o = self._convert_obs(o)
                self._active_head = int(self.rng.integers(0, self.K)) if self.K > 0 else 0

            # --- learn ---
            if self.step_count >= self.cfg.learning_starts and self.step_count % self.cfg.train_freq == 0:
                self._sgd_step()

            # --- target update ---
            if self.step_count % self.cfg.target_update_interval == 0:
                self._update_target()

            # --- optional eval ---
            if eval_env is not None and eval_every > 0 and self.step_count % eval_every == 0:
                avg_ret = self.evaluate(eval_env, eval_episodes or 10)
                self._log_eval(total_steps=self.step_count, avg_reward=avg_ret)
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
        if self.cfg.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.clip_grad_norm)
        self.optim.step()

    # ---------- logging ----------
    def _append_csv(self, path: Optional[str], fieldnames: List[str], row: dict) -> None:
        if not path:
            return
        file_exists = os.path.exists(path)
        with open(path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _log_train_episode(self, episode: int, total_steps: int, episode_steps: int, reward: float) -> None:
        self._append_csv(
            self.train_csv_path,
            ["episode", "total_steps", "episode_steps", "reward"],
            {
                "episode": episode,
                "total_steps": total_steps,
                "episode_steps": episode_steps,
                "reward": reward,
            },
        )

    def _log_eval(self, total_steps: int, avg_reward: float) -> None:
        self._append_csv(
            self.eval_csv_path,
            ["total_steps", "avg_reward"],
            {
                "total_steps": total_steps,
                "avg_reward": avg_reward,
            },
        )

    # ---------- persistence ----------
    def save_model(self, path: Optional[str] = None) -> None:
        target_path = path or self.model_path
        if not target_path:
            return
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.cfg),
            "step_count": self.step_count,
            "episode_idx": self._episode_idx,
            "epsilon": self.eps,
            "model": self.q.state_dict(),
            "target_model": self.q_targ.state_dict(),
            "optimizer": self.optim.state_dict(),
        }
        torch.save(payload, target_path)

    def load_model(self, path: Optional[str] = None, strict: bool = True) -> None:
        source_path = path or self.model_path
        if not source_path:
            raise ValueError("No model path provided for loading.")
        payload = torch.load(source_path, map_location=self.device)
        self.q.load_state_dict(payload["model"], strict=strict)
        self.q_targ.load_state_dict(payload.get("target_model", payload["model"]), strict=strict)
        if "optimizer" in payload:
            self.optim.load_state_dict(payload["optimizer"])
        self.step_count = int(payload.get("step_count", 0))
        self._episode_idx = int(payload.get("episode_idx", 0))
        self.eps = float(payload.get("epsilon", self.cfg.epsilon_start))

    # ---------- evaluation ----------
    @torch.no_grad()
    def evaluate(self, eval_env: gym.Env, episodes: int = 10) -> float:
        """Deterministic eval: ε=0 and greedy wrt ensemble-mean Q."""
        total = 0.0
        for _ in range(episodes):
            o, _ = eval_env.reset()
            o = self._convert_obs(o)
            done = False
            ep_ret = 0.0
            while not done:
                a = self._select_action(o, exploit_only=True)
                o_next, r, term, trunc, _ = eval_env.step(a)
                o = self._convert_obs(o_next)
                ep_ret += float(r)
                done = term or trunc
            total += ep_ret
        return total / episodes


# =========================
# Minimal usage example
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EBQL on Space Invaders across multiple seeds")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Seeds to run")
    parser.add_argument("--run-name", type=str, default="spaceinvaders_k5", help="Name for the run directory under ./runs")
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--eval-every", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--K", type=int, default=5, help="Ensemble size override")
    parser.add_argument("--proof-of-concept", action="store_true", help="Run a short, cheap proof-of-concept training session")
    args = parser.parse_args()

    base_dir = Path("runs") / args.run_name
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"[EBQL] Starting run '{args.run_name}' for seeds: {args.seeds}")

    total_steps = args.total_steps
    eval_every = args.eval_every
    eval_episodes = args.eval_episodes

    cfg_overrides = {}
    if args.proof_of_concept:
        print("[EBQL] Proof-of-concept mode: using lighter config for a quick sanity check")
        total_steps = min(total_steps, 120_000)
        eval_every = min(eval_every, 10_000)
        eval_episodes = max(eval_episodes, 5)
        cfg_overrides.update(
            buffer_size=75_000,
            batch_size=128,
            learning_starts=2_000,
            epsilon_start=0.6,
            epsilon_end=0.05,
            epsilon_decay_steps=90_000,
            target_update_interval=5_000,
        )

    for seed in args.seeds:
        run_dir = base_dir / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        train_csv = run_dir / "train_rewards.csv"
        eval_csv = run_dir / "eval_rewards.csv"
        model_path = run_dir / "final_model.pt"

        cfg = EBQLConfig(
            K=args.K,
            seed=seed,
            train_csv_path=str(train_csv),
            eval_csv_path=str(eval_csv),
            model_path=str(model_path),
            **cfg_overrides,
        )

        env = make_spaceinvaders_env(seed=seed, frame_stack=cfg.frame_stack)
        eval_env = make_spaceinvaders_env(seed=seed + 10_000, frame_stack=cfg.frame_stack)

        agent = EBQLAgent(env, cfg)

        print(f"[EBQL] Seed {seed}: training for {total_steps:,} steps")
        agent.train(
            total_steps=total_steps,
            eval_env=eval_env,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
        )

        agent.save_model()
        env.close()
        eval_env.close()
        print(f"[EBQL] Seed {seed}: complete. Logs in {run_dir}")

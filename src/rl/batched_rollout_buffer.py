# Path: src/rl/batched_rollout_buffer.py
"""Batched rollout storage for fast DDP RIFT training."""

from __future__ import annotations

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


class BatchedRolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.logps = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, state, action, logp, reward, value, done):
        if not _HAS_TORCH:
            raise RuntimeError("torch required")

        self.states.append(_detach_state(state))
        self.actions.append(action.detach().long())
        self.logps.append(logp.detach().float())
        self.rewards.append(reward.detach().float())
        self.values.append(value.detach().float())
        self.dones.append(done.detach().bool())

    def __len__(self):
        return len(self.actions)

    def rewards_tensor(self, device=None):
        x = torch.stack(self.rewards, dim=0)
        return x.to(device) if device is not None else x

    def dones_tensor(self, device=None):
        x = torch.stack(self.dones, dim=0)
        return x.to(device) if device is not None else x

    def values_tensor(self, device=None):
        x = torch.stack(self.values, dim=0)
        return x.to(device) if device is not None else x

    def returns_tensor(self, gamma: float = 0.99, device=None):
        rewards = self.rewards_tensor(device=device)
        dones = self.dones_tensor(device=device)

        R = torch.zeros_like(rewards[-1])
        out = []

        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + float(gamma) * R * (~d).float()
            out.append(R)

        return torch.stack(list(reversed(out)), dim=0)

    def total_reward_mean(self) -> float:
        if not self.rewards:
            return 0.0
        return float(self.rewards_tensor().sum(dim=0).mean().item())


def _detach_state(state):
    out = {}

    for k, v in state.items():
        if _HAS_TORCH and torch.is_tensor(v):
            out[k] = v.detach()
        elif isinstance(v, (list, tuple)):
            out[k] = list(v)
        else:
            out[k] = v

    return out

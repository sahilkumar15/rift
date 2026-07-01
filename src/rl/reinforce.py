# Path: src/rl/reinforce.py
# Status: MODIFIED
"""REINFORCE with baseline."""
from __future__ import annotations

from .rollout_buffer import RolloutBuffer

try:
    import torch
    _HAS = True
except Exception:
    _HAS = False


class Reinforce:
    def __init__(self, policy, lr=3e-4, gamma=0.99, entropy_coef=0.01):
        assert _HAS, "torch required."

        self.policy = policy
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)

    def update(self, buffer: RolloutBuffer):
        device = next(self.policy.parameters()).device

        returns = torch.tensor(buffer.returns(self.gamma), device=device).float()
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

        policy_loss = torch.zeros((), device=device)
        ent = torch.zeros((), device=device)

        for s, a, R in zip(buffer.states, buffer.actions, returns):
            logits, value = self.policy(s)

            logp = torch.log_softmax(logits, -1)[0, a]
            adv = R - value.squeeze().detach()

            policy_loss = policy_loss - logp * adv

            p = torch.softmax(logits, -1)
            ent = ent - (p * torch.log(p + 1e-8)).sum()

        loss = policy_loss - self.entropy_coef * ent

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt.step()

        return {
            "policy_loss": float(policy_loss.detach().item()),
            "entropy": float(ent.detach().item()),
        }
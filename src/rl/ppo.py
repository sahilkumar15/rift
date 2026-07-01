# Path: src/rl/ppo.py
"""PPO-clip with batched rollout support."""

from __future__ import annotations

from .rollout_buffer import RolloutBuffer

try:
    import torch
    _HAS = True
except Exception:
    _HAS = False


class PPO:
    def __init__(
        self,
        policy,
        lr=3e-4,
        gamma=0.99,
        clip=0.2,
        epochs=4,
        entropy_coef=0.01,
        value_coef=0.5,
        lagrangian=False,
        constraint_budget=0.0,
        dual_lr=1e-2,
    ):
        assert _HAS, "torch required."

        self.policy = policy
        self.gamma = gamma
        self.clip = clip
        self.epochs = int(epochs)
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.0)
        self.lagrangian = lagrangian
        self.budget = constraint_budget
        self.dual_lr = dual_lr
        self.lmbda = 0.0

    def update(self, buffer: RolloutBuffer, constraint_costs=None):
        if len(buffer) == 0:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "lambda": self.lmbda,
            }

        device = next(self.policy.parameters()).device

        actions = _stack_2d(buffer.actions, device=device, dtype=torch.long)
        old_logps = _stack_2d(buffer.logps, device=device, dtype=torch.float32)

        if hasattr(buffer, "returns_tensor"):
            returns = buffer.returns_tensor(self.gamma, device=device).float()
            old_values = buffer.values_tensor(device=device).float()
        else:
            returns = torch.tensor(
                buffer.returns(self.gamma),
                device=device,
                dtype=torch.float32,
            ).view(-1, 1)
            old_values = _stack_2d(
                getattr(buffer, "values", [0.0] * len(buffer)),
                device=device,
                dtype=torch.float32,
            )

        adv = returns - old_values
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        logs = {}
        T = int(actions.shape[0])

        for _ in range(max(1, self.epochs)):
            ploss = torch.zeros((), device=device)
            vloss = torch.zeros((), device=device)
            ent = torch.zeros((), device=device)

            for t, state in enumerate(buffer.states):
                logits, value = self.policy(state)
                value = value.squeeze(-1)

                a = actions[t].view(-1)
                old_lp = old_logps[t].view(-1)
                adv_t = adv[t].view(-1)
                ret_t = returns[t].view(-1)

                logp_all = torch.log_softmax(logits, dim=-1)
                logp = logp_all.gather(1, a.unsqueeze(1)).squeeze(1)

                ratio = torch.exp(logp - old_lp)

                surr1 = ratio * adv_t
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv_t

                ploss = ploss - torch.min(surr1, surr2).mean()
                vloss = vloss + (ret_t - value).pow(2).mean()

                p = torch.softmax(logits, dim=-1)
                ent = ent + (-(p * torch.log(p + 1e-8)).sum(dim=-1)).mean()

            ploss = ploss / T
            vloss = vloss / T
            ent = ent / T

            loss = ploss + self.value_coef * vloss - self.entropy_coef * ent

            if self.lagrangian and constraint_costs is not None:
                viol = float(sum(constraint_costs) / max(len(constraint_costs), 1)) - self.budget
                loss = loss + self.lmbda * viol
                self.lmbda = max(0.0, self.lmbda + self.dual_lr * viol)

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.opt.step()

            logs = {
                "policy_loss": float(ploss.detach().item()),
                "value_loss": float(vloss.detach().item()),
                "entropy": float(ent.detach().item()),
                "lambda": self.lmbda,
            }

        return logs


def _stack_2d(seq, *, device, dtype):
    xs = []

    for x in seq:
        if torch.is_tensor(x):
            t = x.detach().to(device=device, dtype=dtype).view(-1)
        else:
            t = torch.tensor([x], device=device, dtype=dtype)

        xs.append(t)

    max_b = max(int(t.numel()) for t in xs)
    out = []

    for t in xs:
        if t.numel() == max_b:
            out.append(t)
        elif t.numel() == 1:
            out.append(t.repeat(max_b))
        else:
            reps = max_b // t.numel() + 1
            out.append(t.repeat(reps)[:max_b])

    return torch.stack(out, dim=0)

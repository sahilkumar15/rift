# Path: src/rl/reinforce.py
# Status: NEW
"""REINFORCE with baseline (the simple, robust RL trainer). torch-guarded."""
from __future__ import annotations
from .rollout_buffer import RolloutBuffer
try: import torch; _HAS=True
except Exception: _HAS=False
class Reinforce:
    def __init__(self, policy, lr=3e-4, gamma=0.99, entropy_coef=0.01):
        assert _HAS, "torch required (Katz)."
        self.policy=policy; self.gamma=gamma; self.entropy_coef=entropy_coef
        self.opt=torch.optim.Adam(policy.parameters(), lr=lr)
    def update(self, buffer: RolloutBuffer):
        returns=torch.tensor(buffer.returns(self.gamma)).float()
        returns=(returns-returns.mean())/(returns.std()+1e-8)
        policy_loss=0.0; ent=0.0
        for s,a,R in zip(buffer.states, buffer.actions, returns):
            logits,value=self.policy(s)
            logp=torch.log_softmax(logits,-1)[0,a]
            adv=R - value.squeeze().detach()
            policy_loss = policy_loss - logp*adv
            p=torch.softmax(logits,-1); ent = ent - (p*torch.log(p+1e-8)).sum()
        loss=policy_loss - self.entropy_coef*ent
        self.opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt.step()
        return {"policy_loss": float(policy_loss), "entropy": float(ent)}

# Path: iganer/rift/rl/ppo.py
# Status: NEW
"""PPO-clip with optional Lagrangian constraint dual. torch-guarded."""
from __future__ import annotations
from .rollout_buffer import RolloutBuffer
try: import torch; _HAS=True
except Exception: _HAS=False
class PPO:
    def __init__(self, policy, lr=3e-4, gamma=0.99, clip=0.2, epochs=4,
                 entropy_coef=0.01, value_coef=0.5,
                 lagrangian=False, constraint_budget=0.0, dual_lr=1e-2):
        assert _HAS, "torch required (Katz)."
        self.policy=policy; self.gamma=gamma; self.clip=clip; self.epochs=epochs
        self.entropy_coef=entropy_coef; self.value_coef=value_coef
        self.opt=torch.optim.Adam(policy.parameters(), lr=lr)
        self.lagrangian=lagrangian; self.budget=constraint_budget
        self.dual_lr=dual_lr; self.lmbda=0.0
    def update(self, buffer: RolloutBuffer, constraint_costs=None):
        returns=torch.tensor(buffer.returns(self.gamma)).float()
        adv=returns-returns.mean(); adv=(adv)/(adv.std()+1e-8)
        old_logps=torch.tensor(buffer.logps).float()
        logs={}
        for _ in range(self.epochs):
            ploss=0.0; vloss=0.0; ent=0.0
            for i,(s,a) in enumerate(zip(buffer.states, buffer.actions)):
                logits,value=self.policy(s)
                logp=torch.log_softmax(logits,-1)[0,a]
                ratio=torch.exp(logp-old_logps[i])
                s1=ratio*adv[i]; s2=torch.clamp(ratio,1-self.clip,1+self.clip)*adv[i]
                ploss=ploss - torch.min(s1,s2)
                vloss=vloss + (returns[i]-value.squeeze())**2
                p=torch.softmax(logits,-1); ent=ent-(p*torch.log(p+1e-8)).sum()
            loss=ploss + self.value_coef*vloss - self.entropy_coef*ent
            if self.lagrangian and constraint_costs is not None:
                viol=float(sum(constraint_costs)/len(constraint_costs))-self.budget
                loss=loss + self.lmbda*viol
                self.lmbda=max(0.0, self.lmbda + self.dual_lr*viol)
            self.opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(),1.0)
            self.opt.step()
            logs={"policy_loss":float(ploss),"value_loss":float(vloss),
                  "entropy":float(ent),"lambda":self.lmbda}
        return logs

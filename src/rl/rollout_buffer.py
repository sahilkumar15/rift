# Path: src/rl/rollout_buffer.py
# Status: NEW
"""Stores (state, action, logp, reward, value, done) for an episode/batch."""
class RolloutBuffer:
    def __init__(self): self.clear()
    def clear(self):
        self.states=[]; self.actions=[]; self.logps=[]
        self.rewards=[]; self.values=[]; self.dones=[]
    def add(self, s,a,lp,r,v,d):
        self.states.append(s); self.actions.append(a); self.logps.append(lp)
        self.rewards.append(r); self.values.append(v); self.dones.append(d)
    def __len__(self): return len(self.actions)
    def returns(self, gamma=0.99):
        R=0; out=[]
        for r,d in zip(reversed(self.rewards), reversed(self.dones)):
            R = r + gamma*R*(0.0 if d else 1.0)
            out.append(R)
        return list(reversed(out))

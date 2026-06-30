# Path: iganer/rift/rl/policy.py
# Status: NEW
"""Grid policy network: state -> action logits + value. Small CNN+MLP. torch-guarded."""
from __future__ import annotations
try:
    import torch; import torch.nn as nn; _HAS=True
except Exception: _HAS=False
if _HAS:
    class GridPolicy(nn.Module):
        def __init__(self, feat_dim=256, grid=8, n_actions=None, hidden=256):
            super().__init__()
            self.grid=grid
            self.n_actions=n_actions or grid*grid+1
            # state encoder consumes [pooled feat, mask-grid flat, scalars]
            self.mask_enc=nn.Sequential(nn.Flatten(), nn.Linear(grid*grid, hidden), nn.ReLU())
            self.scalar_enc=nn.Sequential(nn.Linear(3, hidden//4), nn.ReLU())
            self.feat_proj=nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                         nn.LazyLinear(hidden), nn.ReLU())
            self.trunk=nn.Sequential(nn.LazyLinear(hidden), nn.ReLU())
            self.pi=nn.Linear(hidden, self.n_actions)
            self.v=nn.Linear(hidden, 1)
        def _encode(self, state):
            parts=[]
            if state.get("feat") is not None:
                parts.append(self.feat_proj(state["feat"]))
            parts.append(self.mask_enc(state["current_mask"]))
            sc=torch.tensor([[state["confidence"], float(state["step_idx"]),
                              float(state["last_action"])]],
                            device=state["current_mask"].device).float()
            sc=sc.repeat(state["current_mask"].shape[0],1)
            parts.append(self.scalar_enc(sc))
            h=torch.cat(parts, dim=-1)
            return self.trunk(h)
        def forward(self, state):
            h=self._encode(state); return self.pi(h), self.v(h)
        @torch.no_grad()
        def act(self, state, deterministic=False):
            logits,_=self.forward(state)
            if deterministic: return int(logits.argmax(-1)[0].item())
            probs=torch.softmax(logits,-1)
            return int(torch.multinomial(probs,1)[0,0].item())
else:
    class GridPolicy: 
        def __init__(self,*a,**k): raise RuntimeError("torch required (Katz).")

# Path: iganer/rift/adapters/identity_adapter.py
# Status: NEW
"""Optional ArcFace-style identity embedding for identity_preservation constraint.
If unavailable, identity_preservation falls back to 1.0 (no penalty) with a warning."""
from __future__ import annotations
import warnings
class IdentityAdapter:
    def __init__(self, model=None, device="cuda"):
        self.model = model; self.device = device; self._warned=False
    def embed(self, x):
        if self.model is None:
            if not self._warned:
                warnings.warn("IdentityAdapter has no model; identity_preservation=1.0 stub.",
                              RuntimeWarning); self._warned=True
            return None
        import torch
        with torch.no_grad(): return self.model(x)
    def cosine_preservation(self, x_orig, x_pert):
        e0 = self.embed(x_orig); e1 = self.embed(x_pert)
        if e0 is None or e1 is None: return 1.0
        import torch.nn.functional as F
        return float(F.cosine_similarity(e0, e1, dim=-1).mean().item())

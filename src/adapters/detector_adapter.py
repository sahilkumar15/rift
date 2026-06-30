# Path: src/adapters/detector_adapter.py
# Status: NEW
"""Generic detector interface so RIFT can audit non-CIFT models (Xception/SBI/etc.).
Wrap any classifier exposing forward->logits and a feature hook."""
from __future__ import annotations
class DetectorAdapter:
    def __init__(self, model=None, feature_layer=None, device="cuda"):
        self.model = model; self.feature_layer = feature_layer
        self.device = device; self._feat = None
    def load(self): 
        if self.model is None:
            raise NotImplementedError("Pass a constructed model or subclass load().")
        self.model.eval()
        if self.feature_layer is not None:
            self.feature_layer.register_forward_hook(
                lambda m,i,o: setattr(self, "_feat", o))
        return self
    def predict_logits(self, x):
        import torch
        with torch.no_grad(): return self.model(x)
    def extract_features(self, x):
        import torch
        with torch.no_grad(): self.model(x)
        if self._feat is None: raise RuntimeError("No feature_layer hooked.")
        return self._feat

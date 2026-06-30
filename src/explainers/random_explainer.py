# Path: src/explainers/random_explainer.py
# Status: NEW
"""Random-mask control. Critical baseline: a faithful method must beat random."""
from .base_explainer import BaseExplainer
class RandomExplainer(BaseExplainer):
    name = "random"
    def explain(self, image, adapter, **kw):
        import torch
        B,_,H,W = image.shape
        return torch.rand(B,1,H,W, device=image.device)

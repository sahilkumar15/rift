# Path: iganer/rift/explainers/external_map_explainer.py
# Status: NEW
"""Wrap a precomputed external saliency map (e.g., from a VLM or artifact detector)."""
from .base_explainer import BaseExplainer
class ExternalMapExplainer(BaseExplainer):
    name = "external"
    def __init__(self, map_fn): self.map_fn = map_fn
    def explain(self, image, adapter, **kw): return self.map_fn(image)

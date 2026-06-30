# Path: src/explainers/cift_gap_explainer.py
# Status: NEW
"""CIFT's own identity-gap attribution map (the mechanism's self-explanation).
Uses adapter.identity_gap_map() if present, else explain_identity_gap() fallback."""
from .base_explainer import BaseExplainer
class CIFTGapExplainer(BaseExplainer):
    name = "cift_gap"
    def explain(self, image, adapter, **kw):
        m = adapter.identity_gap_map(image)
        if m is not None: return m
        return adapter.explain_identity_gap(image)

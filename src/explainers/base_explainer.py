# Path: iganer/rift/explainers/base_explainer.py
# Status: NEW
"""Explainer contract: image+adapter -> soft saliency mask (B,1,H,W) in [0,1]."""
from __future__ import annotations
from abc import ABC, abstractmethod
class BaseExplainer(ABC):
    name = "base"
    @abstractmethod
    def explain(self, image, adapter, **kw): ...

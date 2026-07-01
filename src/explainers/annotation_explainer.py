# Path: src/explainers/annotation_explainer.py
"""Annotation-mask explainer.

This is used to test the MARE/plausibility baseline:
a human/annotation mask can have high IoU but still fail causal faithfulness.
"""

from __future__ import annotations

from .base_explainer import BaseExplainer


class AnnotationExplainer(BaseExplainer):
    name = "annotation"

    def __init__(self, gt_mask):
        self.gt_mask = gt_mask

    def explain(self, image, adapter, **kw):
        if self.gt_mask is None:
            raise RuntimeError("AnnotationExplainer requires gt_mask/mask_path.")
        return self.gt_mask

# Path: src/explainers/cift_gap_explainer.py
"""
CIFT identity-gap attribution explainer.

Uses donor-grounded identity-gap gradients when donor is available.
Falls back safely if CIFT internals detach some branch.
"""

from .base_explainer import BaseExplainer


class CIFTGapExplainer(BaseExplainer):
    name = "cift_gap"

    def explain(self, image, adapter, **kw):
        m = adapter.identity_gap_map(
            image,
            donor=kw.get("donor"),
            source_id=kw.get("source_id"),
            target_id=kw.get("target_id"),
        )

        if m is not None:
            return m

        return adapter.explain_identity_gap(
            image,
            donor=kw.get("donor"),
            source_id=kw.get("source_id"),
            target_id=kw.get("target_id"),
        )

# Path: src/faithfulness/__init__.py
"""Faithfulness metrics for RIFT.

This package is intentionally model-free. It only consumes evidence scalars:
E0, E_nec, E_suf.
"""

from .faithfulness_score import (
    FaithfulnessComponents,
    compute_rift_score,
    necessity,
    sufficiency,
    harmonic,
)
from .identity_gap_metrics import (
    necessity_delta_drop,
    sufficiency_delta_retained,
    necessity_logit_drop,
    sufficiency_logit_retained,
)
from .plausibility import plausibility_iou

__all__ = [
    "FaithfulnessComponents",
    "compute_rift_score",
    "necessity",
    "sufficiency",
    "harmonic",
    "necessity_delta_drop",
    "sufficiency_delta_retained",
    "necessity_logit_drop",
    "sufficiency_logit_retained",
    "plausibility_iou",
]

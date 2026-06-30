# Path: src/audit/correlation_runner.py
# Status: NEW
"""Compute per-checkpoint faithfulness + AUCs, then correlate vs zero-shot AUC.
This drives the headline 'faithfulness predicts generalization' result."""
from __future__ import annotations
from typing import List, Dict
from ..metrics.correlation_metrics import correlate_predictors
def run_correlation(rows: List[Dict], min_n=5):
    """rows: one dict per checkpoint with keys faithfulness,in_domain_auc,
    plausibility,zero_shot_auc."""
    results=correlate_predictors(rows, min_n=min_n)
    summary={r.predictor: {"spearman":r.spearman,"pearson":r.pearson,
                           "ci":r.spearman_ci,"n":r.n,"reportable":r.reportable,
                           "note":r.note} for r in results}
    return summary

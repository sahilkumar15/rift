# Path: iganer/rift/metrics/robustness_metrics.py
# Status: NEW
"""Robustness gap + attacked-vs-clean helpers."""
def robustness_gap(clean_auc, attacked_auc): return float(clean_auc - attacked_auc)
def budget_auc_curve(budgets, aucs):
    return [{"budget": b, "auc": a} for b, a in zip(budgets, aucs)]

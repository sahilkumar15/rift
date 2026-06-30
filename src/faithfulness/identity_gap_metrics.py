# Path: iganer/rift/faithfulness/identity_gap_metrics.py
# Status: NEW
"""Δ-channel deltas: drop on necessity, retained on sufficiency."""
def necessity_delta_drop(e0, e_nec): return float(e0 - e_nec)
def sufficiency_delta_retained(e0, e_suf): return float(e_suf)
def necessity_logit_drop(l0, l_nec): return float(l0 - l_nec)
def sufficiency_logit_retained(l0, l_suf): return float(l_suf)

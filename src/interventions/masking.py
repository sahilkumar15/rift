# Path: iganer/rift/interventions/masking.py
# Status: NEW
"""Mask utilities (spec-named). Real impl in interventions.py."""
from .interventions import _to_binary, mask_area
def topk_binary_mask(soft_mask, topk_frac=0.1):
    return _to_binary(soft_mask, topk_frac)

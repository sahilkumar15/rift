# Path: src/rl/constraints.py
# Status: NEW
"""Constraint cost helpers for the Lagrangian dual (identity / perceptual / sparsity)."""
def identity_cost(identity_preservation, budget=0.95):
    return max(0.0, budget - identity_preservation)
def perceptual_cost(perceptual_distance, budget=0.1):
    return max(0.0, perceptual_distance - budget)
def sparsity_cost(mask_area, budget=0.2):
    return max(0.0, mask_area - budget)

# Path: src/rl/reward.py
"""Reward presets for RIFT Table 1/2/3 ablations.

The reward computation itself lives in src/faithfulness/faithfulness_score.py.
This file only defines named objective/weight presets so ablations cannot
silently rewrite scorer code.
"""
from __future__ import annotations

_BASE_GUARDS = {
    # back-compat + variable-mask guards
    "min_selected_cells": 1.0,
    "w_min_cells": 0.10,
    "empty_mask_penalty": 0.25,
    # evidence/sparsity options
    "min_evidence": 0.0,
    "sparsity_mode": "linear",
    "area_lo": 0.02,
    "area_hi": 0.35,
    # Dense shaping must be explicit. The paper/eval RIFT score uses only
    # harmonic necessity-sufficiency plus penalties; keeping dense terms
    # silently on makes train-val scores look good while final eval is weak.
    "w_dense_delta": 0.0,
    "w_dense_logit": 0.0,
    "w_empty": 0.25,
}

REWARD_PRESETS = {
    "acc_only": {
        "w_delta": 0.0, "w_logit": 0.0, "w_sparsity": 0.0,
        "w_identity": 0.0, "w_perceptual": 0.0, "w_plausibility": 0.0,
        "objective": "none", **_BASE_GUARDS,
    },
    "plausibility": {
        "w_delta": 0.0, "w_logit": 0.0, "w_sparsity": 0.1,
        "w_identity": 0.0, "w_perceptual": 0.0, "w_plausibility": 1.0,
        "objective": "harmonic", **_BASE_GUARDS,
    },
    "generic_logit": {
        "w_delta": 0.0, "w_logit": 1.0, "w_sparsity": 0.3,
        "w_identity": 0.0, "w_perceptual": 0.0, "w_plausibility": 0.0,
        "objective": "harmonic", **_BASE_GUARDS,
    },
    "delta_no_interv": {
        "w_delta": 1.0, "w_logit": 0.0, "w_sparsity": 0.3,
        "w_identity": 0.0, "w_perceptual": 0.0, "w_plausibility": 0.0,
        "objective": "harmonic", **_BASE_GUARDS,
    },
    "full_rift": {
        "w_delta": 1.0, "w_logit": 0.5, "w_sparsity": 0.3,
        "w_identity": 0.0, "w_perceptual": 0.0, "w_plausibility": 0.0,
        "objective": "harmonic", **_BASE_GUARDS,
    },
    # Optional exploration/stability preset. Do not use as the headline paper
    # row unless dense terms are reported separately.
    "full_rift_shaped": {
        "w_delta": 1.0, "w_logit": 0.5, "w_sparsity": 0.3,
        "w_identity": 0.0, "w_perceptual": 0.0, "w_plausibility": 0.0,
        "objective": "harmonic",
        "w_dense_delta": 0.10,
        "w_dense_logit": 0.05,
        **_BASE_GUARDS,
    },
    "necessity_only": {
        "w_delta": 1.0, "w_logit": 0.5, "w_sparsity": 0.3,
        "w_identity": 0.0, "w_perceptual": 0.0, "w_plausibility": 0.0,
        "objective": "necessity", **_BASE_GUARDS,
    },
    "sufficiency_only": {
        "w_delta": 1.0, "w_logit": 0.5, "w_sparsity": 0.3,
        "w_identity": 0.0, "w_perceptual": 0.0, "w_plausibility": 0.0,
        "objective": "sufficiency", **_BASE_GUARDS,
    },
    "no_sparsity": {
        "w_delta": 1.0, "w_logit": 0.5, "w_sparsity": 0.0,
        "w_identity": 0.0, "w_perceptual": 0.0, "w_plausibility": 0.0,
        "objective": "harmonic", **_BASE_GUARDS,
    },
}


def get_reward_weights(name: str):
    if name not in REWARD_PRESETS:
        raise KeyError(f"unknown reward preset {name}; have {list(REWARD_PRESETS)}")
    return dict(REWARD_PRESETS[name])

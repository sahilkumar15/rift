"""Reward presets for RIFT Table 1/2/3 ablations."""
from __future__ import annotations

REWARD_PRESETS = {
    "acc_only": {
        "w_delta": 0.0, "w_logit": 0.0, "w_sparsity": 0.0,
        "w_identity": 0.0, "w_perceptual": 0.0, "w_plausibility": 0.0,
        "objective": "none",
    },
    "plausibility": {
        "w_delta": 0.0, "w_logit": 0.0, "w_sparsity": 0.1,
        "w_identity": 0.0, "w_perceptual": 0.0, "w_plausibility": 1.0,
        "objective": "harmonic",
    },
    "generic_logit": {
        "w_delta": 0.0, "w_logit": 1.0, "w_sparsity": 0.3,
        "w_identity": 0.3, "w_perceptual": 0.2, "w_plausibility": 0.0,
        "objective": "harmonic",
    },
    "delta_no_interv": {
        "w_delta": 1.0, "w_logit": 0.0, "w_sparsity": 0.3,
        "w_identity": 0.3, "w_perceptual": 0.2, "w_plausibility": 0.0,
        "objective": "harmonic",
    },
    "full_rift": {
        "w_delta": 1.0, "w_logit": 0.5, "w_sparsity": 0.3,
        "w_identity": 0.3, "w_perceptual": 0.2, "w_plausibility": 0.0,
        "objective": "harmonic",
    },
    "necessity_only": {
        "w_delta": 1.0, "w_logit": 0.5, "w_sparsity": 0.3,
        "w_identity": 0.3, "w_perceptual": 0.2, "w_plausibility": 0.0,
        "objective": "necessity",
    },
    "sufficiency_only": {
        "w_delta": 1.0, "w_logit": 0.5, "w_sparsity": 0.3,
        "w_identity": 0.3, "w_perceptual": 0.2, "w_plausibility": 0.0,
        "objective": "sufficiency",
    },
    "no_sparsity": {
        "w_delta": 1.0, "w_logit": 0.5, "w_sparsity": 0.0,
        "w_identity": 0.3, "w_perceptual": 0.2, "w_plausibility": 0.0,
        "objective": "harmonic",
    },
}


def get_reward_weights(name: str):
    if name not in REWARD_PRESETS:
        raise KeyError(f"unknown reward preset {name}; have {list(REWARD_PRESETS)}")
    return dict(REWARD_PRESETS[name])

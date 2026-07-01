"""Reward presets for RIFT ablations.

The reward math lives in faithfulness_score.py and batched_rift_env.py.
This file only defines clean presets for training each paper-table row.
"""

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

    # Table 1 row 4: RL from logit evidence only.
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

    # Main RIFT.
    "full_rift": {
        "w_delta": 1.0, "w_logit": 0.5, "w_sparsity": 0.3,
        "w_identity": 0.3, "w_perceptual": 0.2, "w_plausibility": 0.0,
        "objective": "harmonic",
    },

    # Table 2 objective ablations.
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


def get_reward_weights(name):
    if name not in REWARD_PRESETS:
        raise KeyError(f"unknown reward preset {name}; have {list(REWARD_PRESETS)}")
    return dict(REWARD_PRESETS[name])

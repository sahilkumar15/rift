# Path: iganer/rift/audit/ablation_runner.py
# Status: NEW
"""Drives the 3 ablation blocks by toggling reward preset + intervention/grounding."""
from __future__ import annotations
from ..rl.reward import get_reward_weights
ABLATION_ROWS = {
    # name -> (reward_preset, use_intervention, ground_in_delta)
    "acc_only":            ("acc_only",        False, False),
    "plausibility_only":   ("plausibility",    False, False),
    "generic_logit_interv":("generic_logit",   True,  False),
    "delta_reward_no_int": ("delta_no_interv", False, True),
    "delta_int_no_rl":     ("full_rift",       True,  True),   # audit-only, no policy update
    "full_rift_rl":        ("full_rift",       True,  True),
}
def resolve_ablation(name):
    if name not in ABLATION_ROWS: raise KeyError(name)
    preset, use_int, ground = ABLATION_ROWS[name]
    return {"reward_weights":get_reward_weights(preset),
            "use_intervention":use_int, "ground_in_delta":ground}

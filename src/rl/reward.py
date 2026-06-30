# Path: src/rl/reward.py
# Status: NEW
"""Reward weight presets used by ablation toggles. The reward MATH lives in
faithfulness.faithfulness_score.compute_rift_score; this just supplies weights."""
REWARD_PRESETS = {
    # ablation row -> weight dict passed to compute_rift_score
    "acc_only":        {"w_delta":0.0,"w_logit":0.0,"w_sparsity":0.0,"w_identity":0.0,"w_perceptual":0.0,"w_plausibility":0.0},
    "plausibility":    {"w_delta":0.0,"w_logit":0.0,"w_sparsity":0.1,"w_identity":0.0,"w_perceptual":0.0,"w_plausibility":1.0},
    "generic_logit":   {"w_delta":0.0,"w_logit":1.0,"w_sparsity":0.3,"w_identity":0.3,"w_perceptual":0.2,"w_plausibility":0.0},
    "delta_no_interv": {"w_delta":1.0,"w_logit":0.0,"w_sparsity":0.3,"w_identity":0.3,"w_perceptual":0.2,"w_plausibility":0.0},
    "full_rift":       {"w_delta":1.0,"w_logit":0.5,"w_sparsity":0.3,"w_identity":0.3,"w_perceptual":0.2,"w_plausibility":0.0},
}
def get_reward_weights(name): 
    if name not in REWARD_PRESETS:
        raise KeyError(f"unknown reward preset {name}; have {list(REWARD_PRESETS)}")
    return dict(REWARD_PRESETS[name])

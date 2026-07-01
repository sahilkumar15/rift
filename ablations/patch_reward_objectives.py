from pathlib import Path

# ---------------------------------------------------------------------
# reward.py: add presets used by Tables 1/2/3
# ---------------------------------------------------------------------
reward_path = Path("src/rl/reward.py")
reward_path.write_text(
'''"""Reward presets for RIFT ablations.

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
'''
)

# ---------------------------------------------------------------------
# faithfulness_score.py: support objective = harmonic / necessity / sufficiency
# ---------------------------------------------------------------------
faith_path = Path("src/faithfulness/faithfulness_score.py")
s = faith_path.read_text()

old = '''    nec_d = necessity(e0_delta, e_nec_delta, delta_floor)
    suf_d = sufficiency(e0_delta, e_suf_delta, delta_floor)
    faith_d = harmonic(nec_d, suf_d)

    nec_l = necessity(e0_logit, e_nec_logit, logit_floor)
    suf_l = sufficiency(e0_logit, e_suf_logit, logit_floor)
    faith_l = harmonic(nec_l, suf_l)

    # HONESTY GUARD: proxy mode cannot claim mechanism faithfulness.
'''

new = '''    nec_d = necessity(e0_delta, e_nec_delta, delta_floor)
    suf_d = sufficiency(e0_delta, e_suf_delta, delta_floor)

    nec_l = necessity(e0_logit, e_nec_logit, logit_floor)
    suf_l = sufficiency(e0_logit, e_suf_logit, logit_floor)

    objective = str(w.get("objective", "harmonic")).lower()
    if objective in ("necessity", "necessity_only", "nec"):
        faith_d = nec_d
        faith_l = nec_l
    elif objective in ("sufficiency", "sufficiency_only", "suf"):
        faith_d = suf_d
        faith_l = suf_l
    elif objective in ("none", "off"):
        faith_d = 0.0
        faith_l = 0.0
    else:
        faith_d = harmonic(nec_d, suf_d)
        faith_l = harmonic(nec_l, suf_l)

    # HONESTY GUARD: proxy mode cannot claim mechanism faithfulness.
'''

if old in s:
    faith_path.write_text(s.replace(old, new))
elif 'objective = str(w.get("objective"' not in s:
    raise RuntimeError("Could not patch src/faithfulness/faithfulness_score.py; pattern changed.")

# ---------------------------------------------------------------------
# batched_rift_env.py: support objective during training and sigmoid logits
# ---------------------------------------------------------------------
env_path = Path("src/rl/batched_rift_env.py")
s = env_path.read_text()

s = s.replace(
'''            logit = self.adapter.predict_logits(self.image)
            self.e0_logit = _fix_vec(logit, self.B, self.image.device)
''',
'''            logit = self.adapter.predict_logits(self.image)
            self.e0_logit = torch.sigmoid(_fix_vec(logit, self.B, self.image.device))
'''
)

s = s.replace(
'''            l_nec = _fix_vec(self.adapter.predict_logits(nec_img), self.B, self.image.device)
            l_suf = _fix_vec(self.adapter.predict_logits(suf_img), self.B, self.image.device)
''',
'''            l_nec = torch.sigmoid(_fix_vec(self.adapter.predict_logits(nec_img), self.B, self.image.device))
            l_suf = torch.sigmoid(_fix_vec(self.adapter.predict_logits(suf_img), self.B, self.image.device))
'''
)

old = '''    nec_d = _nec(e0_delta, e_nec_delta)
    suf_d = _suf(e0_delta, e_suf_delta)
    faith_d = _harmonic(nec_d, suf_d)

    nec_l = _nec(e0_logit, e_nec_logit)
    suf_l = _suf(e0_logit, e_suf_logit)
    faith_l = _harmonic(nec_l, suf_l)

    w_delta = float(w["w_delta"]) if str(identity_gap_mode) == "true" else 0.0
'''

new = '''    nec_d = _nec(e0_delta, e_nec_delta)
    suf_d = _suf(e0_delta, e_suf_delta)

    nec_l = _nec(e0_logit, e_nec_logit)
    suf_l = _suf(e0_logit, e_suf_logit)

    objective = str(w.get("objective", "harmonic")).lower()
    if objective in ("necessity", "necessity_only", "nec"):
        faith_d = nec_d
        faith_l = nec_l
    elif objective in ("sufficiency", "sufficiency_only", "suf"):
        faith_d = suf_d
        faith_l = suf_l
    elif objective in ("none", "off"):
        faith_d = torch.zeros_like(nec_d)
        faith_l = torch.zeros_like(nec_l)
    else:
        faith_d = _harmonic(nec_d, suf_d)
        faith_l = _harmonic(nec_l, suf_l)

    w_delta = float(w["w_delta"]) if str(identity_gap_mode) == "true" else 0.0
'''

if old in s:
    env_path.write_text(s.replace(old, new))
elif 'objective = str(w.get("objective"' not in s:
    raise RuntimeError("Could not patch src/rl/batched_rift_env.py; pattern changed.")

print("[ok] reward/objective patch applied")

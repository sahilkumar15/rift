from __future__ import annotations

from pathlib import Path


def write_reward_presets() -> None:
    Path("src/rl").mkdir(parents=True, exist_ok=True)

    Path("src/rl/reward.py").write_text(
'''"""Reward presets for RIFT Table 1/2/3 ablations."""
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
'''
    )

    print("[write] src/rl/reward.py")


def replace_from_def(path: Path, def_name: str, new_code: str) -> None:
    s = path.read_text()
    marker = f"def {def_name}("
    idx = s.find(marker)

    if idx < 0:
        raise RuntimeError(f"{path}: could not find {marker}")

    # These two target functions are the final functions in their files in this project.
    s = s[:idx] + new_code.rstrip() + "\n"
    path.write_text(s)


def patch_faithfulness_score() -> None:
    p = Path("src/faithfulness/faithfulness_score.py")

    new_func = r'''
def compute_rift_score(
    *,
    e0_delta: float, e_nec_delta: float, e_suf_delta: float, delta_floor: float = 0.0,
    e0_logit: float, e_nec_logit: float, e_suf_logit: float, logit_floor: float = 0.0,
    mask_area: float,
    identity_preservation: float = 1.0,
    perceptual_distance: float = 0.0,
    plausibility_iou: Optional[float] = None,
    identity_gap_mode: str = "proxy",
    weights: Optional[Dict[str, float]] = None,
) -> FaithfulnessComponents:
    """
    Compute RIFT causal-faithfulness score.

    objective:
      harmonic     = harmonic mean of necessity and sufficiency
      necessity    = necessity-only ablation
      sufficiency  = sufficiency-only ablation
      none/off     = no faithfulness objective
    """
    w = {
        "w_delta": 1.0,
        "w_logit": 0.5,
        "w_sparsity": 0.3,
        "w_identity": 0.3,
        "w_perceptual": 0.2,
        "w_plausibility": 0.0,
        "objective": "harmonic",
    }

    if weights:
        w.update(weights)

    nec_d = necessity(e0_delta, e_nec_delta, delta_floor)
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
    if identity_gap_mode != "true":
        w["w_delta"] = 0.0

    reward = float(w["w_delta"]) * faith_d + float(w["w_logit"]) * faith_l
    reward -= float(w["w_sparsity"]) * mask_area
    reward -= float(w["w_identity"]) * (1.0 - identity_preservation)
    reward -= float(w["w_perceptual"]) * perceptual_distance

    if plausibility_iou is not None:
        reward += float(w["w_plausibility"]) * plausibility_iou

    return FaithfulnessComponents(
        necessity_delta=nec_d,
        sufficiency_delta=suf_d,
        faithfulness_delta=faith_d,
        necessity_logit=nec_l,
        sufficiency_logit=suf_l,
        faithfulness_logit=faith_l,
        mask_area=mask_area,
        identity_preservation=identity_preservation,
        perceptual_distance=perceptual_distance,
        plausibility_iou=plausibility_iou,
        rift_score=float(reward),
        identity_gap_mode=identity_gap_mode,
    )
'''

    replace_from_def(p, "compute_rift_score", new_func)
    print("[write] src/faithfulness/faithfulness_score.py compute_rift_score")


def patch_batched_env() -> None:
    p = Path("src/rl/batched_rift_env.py")
    s = p.read_text()

    old_init = '''        self._reset_state()

    def _reset_state(self):
'''

    new_init = '''        # Cheap initial state only. reset() performs the expensive CIFT forward.
        # This avoids a duplicate frozen-CIFT forward at env construction time.
        self.step_idx = 0
        self.mask = torch.zeros(self.B, 1, self.grid, self.grid, device=self.image.device)
        self.last_action = torch.full((self.B,), -1, device=self.image.device, dtype=torch.long)
        self.done = torch.zeros(self.B, device=self.image.device, dtype=torch.bool)
        self.e0_gap = torch.zeros(self.B, device=self.image.device)
        self.e0_logit = torch.zeros(self.B, device=self.image.device)
        self.identity_gap_mode = "proxy"
        self.feat0 = None

    def _reset_state(self):
'''

    if old_init in s:
        s = s.replace(old_init, new_init)

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

    p.write_text(s)

    new_tensor_func = r'''
def _compute_rift_score_tensor(
    *,
    e0_delta,
    e_nec_delta,
    e_suf_delta,
    e0_logit,
    e_nec_logit,
    e_suf_logit,
    mask_area,
    identity_gap_mode: str,
    weights=None,
):
    w = {
        "w_delta": 1.0,
        "w_logit": 0.5,
        "w_sparsity": 0.3,
        "w_identity": 0.3,
        "w_perceptual": 0.2,
        "w_plausibility": 0.0,
        "objective": "harmonic",
    }

    if weights:
        w.update(weights)

    nec_d = _nec(e0_delta, e_nec_delta)
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

    reward = w_delta * faith_d + float(w["w_logit"]) * faith_l
    reward = reward - float(w["w_sparsity"]) * mask_area

    comps = {
        "rift_score": reward,
        "faithfulness_delta": faith_d,
        "faithfulness_logit": faith_l,
        "necessity_delta": nec_d,
        "sufficiency_delta": suf_d,
        "necessity_logit": nec_l,
        "sufficiency_logit": suf_l,
        "mask_area": mask_area,
    }

    return reward, comps
'''

    replace_from_def(p, "_compute_rift_score_tensor", new_tensor_func)
    print("[write] src/rl/batched_rift_env.py _compute_rift_score_tensor")


if __name__ == "__main__":
    print("=== Applying RIFT Table123 patches ===")
    write_reward_presets()
    patch_faithfulness_score()
    patch_batched_env()
    print("=== Patch complete ===")

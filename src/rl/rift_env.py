# Path: src/rl/rift_env.py
# Status: NEW
"""
RIFTEnv — a REAL multi-step MDP for explanation-evidence selection.

NOT a contextual bandit: the mask is built incrementally over `horizon` steps,
the state carries step index + history + current mask, and each action's value
depends on what was already selected (selecting a cell already covered wastes a
step; the optimal next cell depends on remaining Δ-evidence after prior masking).

The H=1-vs-H>1 ablation (configs/ablations_rift.yaml) is what *proves* this isn't
a bandit. If H=1 matches H>1 on your data, RL is demoted to a repair note — keep
that result honest.

State (dict): feat, gap_map, current_mask, confidence, step_idx, last_action, history
Action: index into a (grid_h*grid_w) cell grid, OR the STOP action (last index).
Transition: add/raise the chosen grid cell's weight in the running mask.
Reward: mostly TERMINAL — the necessity/sufficiency faithfulness score after the
        full mask is built; small per-step shaping only for constraint violations.
"""
from __future__ import annotations
from typing import Optional, Dict, Any
try:
    import torch; import torch.nn.functional as F; _HAS=True
except Exception: _HAS=False

class RIFTEnv:
    def __init__(self, image, adapter, *, grid=8, horizon=4,
                 intervention_mode="blur", topk_frac=0.12,
                 reward_fn=None, donor=None, source_id=None, target_id=None,
                 allow_stop=True):
        assert _HAS, "RIFTEnv requires torch (run on Katz)."
        self.image=image; self.adapter=adapter
        self.grid=grid; self.horizon=horizon
        self.intervention_mode=intervention_mode; self.topk_frac=topk_frac
        self.reward_fn=reward_fn; self.allow_stop=allow_stop
        self.donor=donor; self.source_id=source_id; self.target_id=target_id
        B,_,H,W = image.shape
        self.B,self.H,self.W = B,H,W
        self.n_cells = grid*grid
        self.n_actions = self.n_cells + (1 if allow_stop else 0)
        self.stop_action = self.n_cells if allow_stop else None
        self._reset_state()

    def _reset_state(self):
        self.step_idx=0
        self.mask=torch.zeros(self.B,1,self.grid,self.grid, device=self.image.device)
        self.history=[]; self.done=False
        with torch.no_grad():
            self.e0_gap=float(self.adapter.identity_gap(
                self.image, donor=self.donor,
                source_id=self.source_id, target_id=self.target_id).value)
            logit=self.adapter.predict_logits(self.image)
            self.e0_logit=float(logit.mean().item() if hasattr(logit,"mean") else logit)

    def reset(self): self._reset_state(); return self._state()

    def current_mask(self):
        """Upsample grid mask -> pixel mask (B,1,H,W)."""
        return F.interpolate(self.mask, size=(self.H,self.W), mode="nearest")

    def _state(self) -> Dict[str, Any]:
        with torch.no_grad():
            feat=None
            try: feat=self.adapter.extract_features(self.image)
            except Exception: pass
        return {
            "feat": feat,
            "current_mask": self.mask.clone(),
            "confidence": self.e0_logit,
            "step_idx": self.step_idx,
            "last_action": self.history[-1] if self.history else -1,
            "history": list(self.history),
            "e0_gap": self.e0_gap,
        }

    def step(self, action: int):
        assert not self.done, "step() after done"
        self.step_idx += 1
        stop = self.allow_stop and action == self.stop_action
        if not stop:
            r = action // self.grid; c = action % self.grid
            # raising an already-set cell is a wasted step -> history-dependent value
            self.mask[..., r, c] = torch.clamp(self.mask[..., r, c] + 0.5, 0, 1)
        self.history.append(int(action))
        self.done = stop or (self.step_idx >= self.horizon)
        reward = 0.0; info={}
        if self.done:
            reward, info = self._terminal_reward()
        else:
            # tiny shaping: discourage over-large masks early
            area = self.current_mask().flatten(1).mean().item()
            reward = -0.01*max(0.0, area-0.5)
        return self._state(), reward, self.done, info

    def _terminal_reward(self):
        from ..interventions.interventions import apply_necessity, apply_sufficiency, mask_area
        from ..faithfulness.faithfulness_score import compute_rift_score
        pm = self.current_mask()
        with torch.no_grad():
            nec_img=apply_necessity(self.image, pm, self.intervention_mode, self.topk_frac)
            suf_img=apply_sufficiency(self.image, pm, self.intervention_mode, self.topk_frac)
            gap_nec=self.adapter.identity_gap(nec_img, donor=self.donor,
                      source_id=self.source_id, target_id=self.target_id)
            gap_suf=self.adapter.identity_gap(suf_img, donor=self.donor,
                      source_id=self.source_id, target_id=self.target_id)
            l_nec=float(self.adapter.predict_logits(nec_img).mean().item())
            l_suf=float(self.adapter.predict_logits(suf_img).mean().item())
        comp=compute_rift_score(
            e0_delta=self.e0_gap, e_nec_delta=gap_nec.value, e_suf_delta=gap_suf.value,
            e0_logit=self.e0_logit, e_nec_logit=l_nec, e_suf_logit=l_suf,
            mask_area=mask_area(pm, self.topk_frac),
            identity_gap_mode=gap_nec.mode.value,
            weights=(self.reward_fn or {}),
        )
        return comp.rift_score, comp.to_dict()

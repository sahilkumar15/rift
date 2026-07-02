# Path: src/rl/batched_rift_env.py
"""Vectorized RIFT environment for fast multi-GPU PPO.

Reward-collapse fix:
  * STOP/no-op is not allowed by default.
  * Repeated cells are repaired to the first empty cell.
  * Raw detector logits are converted to positive evidence by softplus.
  * Empty masks receive an explicit penalty.
  * Dense necessity/sufficiency shaping gives PPO signal before the harmonic
    faithfulness score becomes non-zero.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


EPS = 1e-8


class BatchedRIFTEnv:
    def __init__(
        self,
        image,
        adapter,
        *,
        grid: int = 8,
        horizon: int = 4,
        intervention_mode: str = "blur",
        topk_frac: float = 0.12,
        reward_fn: Optional[Dict[str, float]] = None,
        donor=None,
        allow_stop_as_noop: bool = False,
        forbid_revisit: bool = True,
        cache_features: bool = True,
    ):
        assert _HAS_TORCH, "BatchedRIFTEnv requires torch."

        if image.dim() != 4:
            raise RuntimeError(f"image must be BCHW, got shape={tuple(image.shape)}")

        self.image = image
        self.adapter = adapter
        self.grid = int(grid)
        self.horizon = int(horizon)
        self.intervention_mode = intervention_mode
        self.topk_frac = float(topk_frac)
        self.reward_fn = dict(reward_fn or {})
        self.donor = donor
        self.allow_stop_as_noop = bool(allow_stop_as_noop)
        self.forbid_revisit = bool(forbid_revisit)
        self.cache_features = bool(cache_features)

        self.B, _, self.H, self.W = image.shape
        self.n_cells = self.grid * self.grid
        self.n_actions = self.n_cells + 1
        self.stop_action = self.n_cells

        # Cheap initial state only. reset() performs the expensive CIFT forward.
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
        self.step_idx = 0
        self.mask = torch.zeros(self.B, 1, self.grid, self.grid, device=self.image.device)
        self.last_action = torch.full((self.B,), -1, device=self.image.device, dtype=torch.long)
        self.done = torch.zeros(self.B, device=self.image.device, dtype=torch.bool)

        with torch.no_grad():
            self.e0_gap, self.identity_gap_mode = _identity_gap_tensor(
                self.adapter,
                self.image,
                donor=self.donor,
            )
            self.e0_gap = _fix_vec(self.e0_gap, self.B, self.image.device)

            logit = self.adapter.predict_logits(self.image)
            self.e0_logit = _logit_to_evidence(_fix_vec(logit, self.B, self.image.device))

            self.feat0 = None
            if self.cache_features:
                try:
                    self.feat0 = self.adapter.extract_features(self.image).detach()
                except Exception:
                    self.feat0 = None

    def reset(self):
        self._reset_state()
        return self._state()

    def current_mask(self):
        return F.interpolate(self.mask, size=(self.H, self.W), mode="nearest")

    def _state(self) -> Dict[str, Any]:
        return {
            "feat": self.feat0,
            "current_mask": self.mask.detach().clone(),
            "confidence": self.e0_logit.detach().clone(),
            "step_idx": torch.full((self.B,), float(self.step_idx), device=self.image.device),
            "last_action": self.last_action.detach().clone().float(),
            "e0_gap": self.e0_gap.detach().clone(),
        }

    def _first_empty_cell(self, rows):
        if rows.numel() == 0:
            return torch.empty(0, device=self.image.device, dtype=torch.long)

        flat = self.mask[rows, 0].flatten(1)
        empty = flat <= 0
        return empty.float().argmax(dim=1).long()

    def _repair_actions(self, actions):
        actions = actions.clamp(0, self.stop_action).clone()

        if not self.allow_stop_as_noop:
            stop_rows = torch.nonzero(actions == self.stop_action, as_tuple=False).flatten()
            if stop_rows.numel() > 0:
                actions[stop_rows] = self._first_empty_cell(stop_rows)

        if self.forbid_revisit:
            cell_rows = torch.nonzero(actions != self.stop_action, as_tuple=False).flatten()
            if cell_rows.numel() > 0:
                cell_actions = actions[cell_rows]
                rr = torch.div(cell_actions, self.grid, rounding_mode="floor")
                cc = cell_actions % self.grid
                already = self.mask[cell_rows, 0, rr, cc] > 0
                bad_rows = cell_rows[already]
                if bad_rows.numel() > 0:
                    actions[bad_rows] = self._first_empty_cell(bad_rows)

        return actions

    def step(self, actions):
        actions = actions.detach().to(self.image.device).long().view(-1)

        if actions.numel() != self.B:
            raise RuntimeError(f"actions must have B={self.B} elements, got {actions.numel()}")

        self.step_idx += 1
        actions = self._repair_actions(actions)
        self.last_action = actions

        cell_mask = actions != self.stop_action

        if cell_mask.any():
            idx = torch.nonzero(cell_mask, as_tuple=False).flatten()
            rr = torch.div(actions[idx], self.grid, rounding_mode="floor")
            cc = actions[idx] % self.grid
            self.mask[idx, 0, rr, cc] = torch.clamp(
                self.mask[idx, 0, rr, cc] + 1.0,
                0.0,
                1.0,
            )

        done_now = self.step_idx >= self.horizon
        self.done = torch.full((self.B,), bool(done_now), device=self.image.device, dtype=torch.bool)

        if done_now:
            reward, info = self._terminal_reward()
        else:
            area = _mask_area_per_sample(self.current_mask(), self.topk_frac)
            reward = -0.01 * torch.clamp(area - 0.5, min=0.0)
            info = {"mask_area": float(area.mean().item())}

        return self._state(), reward.detach(), self.done.detach().clone(), info

    def _terminal_reward(self):
        from ..interventions.interventions import apply_necessity, apply_sufficiency

        pm = self.current_mask()

        with torch.no_grad():
            nec_img = apply_necessity(self.image, pm, self.intervention_mode, self.topk_frac)
            suf_img = apply_sufficiency(self.image, pm, self.intervention_mode, self.topk_frac)

            gap_nec, _ = _identity_gap_tensor(self.adapter, nec_img, donor=self.donor)
            gap_suf, _ = _identity_gap_tensor(self.adapter, suf_img, donor=self.donor)

            gap_nec = _fix_vec(gap_nec, self.B, self.image.device)
            gap_suf = _fix_vec(gap_suf, self.B, self.image.device)

            l_nec = _logit_to_evidence(
                _fix_vec(self.adapter.predict_logits(nec_img), self.B, self.image.device)
            )
            l_suf = _logit_to_evidence(
                _fix_vec(self.adapter.predict_logits(suf_img), self.B, self.image.device)
            )

        area = _mask_area_per_sample(pm, self.topk_frac)

        reward, comps = _compute_rift_score_tensor(
            e0_delta=self.e0_gap,
            e_nec_delta=gap_nec,
            e_suf_delta=gap_suf,
            e0_logit=self.e0_logit,
            e_nec_logit=l_nec,
            e_suf_logit=l_suf,
            mask_area=area,
            identity_gap_mode=self.identity_gap_mode,
            weights=self.reward_fn,
        )

        info = {
            k: float(v.mean().item()) if torch.is_tensor(v) else v
            for k, v in comps.items()
        }
        info["identity_gap_mode"] = str(self.identity_gap_mode)

        return reward, info


def _fix_vec(x, B: int, device):
    if not torch.is_tensor(x):
        x = torch.tensor([float(x)], device=device)

    x = x.detach().float().to(device).view(-1)

    if x.numel() == B:
        return x

    if x.numel() == 1:
        return x.repeat(B)

    if x.numel() > B:
        return x[:B].contiguous()

    reps = B // x.numel() + 1
    return x.repeat(reps)[:B].contiguous()


def _identity_gap_tensor(adapter, image, donor=None):
    if hasattr(adapter, "identity_gap_tensor"):
        return adapter.identity_gap_tensor(image, donor=donor)

    res = adapter.identity_gap(image, donor=donor)
    mode = getattr(getattr(res, "mode", "proxy"), "value", getattr(res, "mode", "proxy"))

    return torch.full(
        (image.shape[0],),
        float(res.value),
        device=image.device,
    ), str(mode)


def _logit_to_evidence(x):
    return F.softplus(x.float())


def _topk_binary(mask, topk_frac: float):
    B = mask.shape[0]
    flat = mask.flatten(1)
    k = max(1, int(float(topk_frac) * flat.shape[1]))
    thresh = flat.topk(k, dim=1).values[:, -1:].clamp(min=1e-6)
    return (flat >= thresh).view_as(mask).float()


def _mask_area_per_sample(mask, topk_frac: float):
    return _topk_binary(mask, topk_frac).flatten(1).mean(dim=1)


def _nec(e0, e1, floor=0.0):
    denom = (e0 - float(floor)) + EPS
    raw = (e0 - e1) / denom.clamp_min(EPS)
    return torch.where(denom <= EPS, torch.zeros_like(raw), raw.clamp(0.0, 1.0))


def _suf(e0, e1, floor=0.0):
    denom = (e0 - float(floor)) + EPS
    raw = (e1 - float(floor)) / denom.clamp_min(EPS)
    return torch.where(denom <= EPS, torch.zeros_like(raw), raw.clamp(0.0, 1.0))


def _harmonic(a, b):
    h = 2.0 * a * b / (a + b + EPS)
    return torch.where((a > 0) & (b > 0), h, torch.zeros_like(h))




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

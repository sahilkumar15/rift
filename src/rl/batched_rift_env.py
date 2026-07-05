# Path: src/rl/batched_rift_env.py
"""Vectorized RIFT environment for fast DDP/PPO training.

Fixes included:
  - STOP/no-op is disabled by default.
  - Repeated grid cells are repaired to the first empty cell.
  - Action mask is exposed in the state for policy/PPO masking.
  - Raw logits are converted to positive evidence with softplus.
  - Objective switch supports harmonic, necessity-only, sufficiency-only.
  - Dense reward shaping and empty-mask penalty prevent reward collapse.
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
        min_cells: int = 1,
        max_cells: Optional[int] = None,
        force_min_cells: bool = True,
        fast_reward: bool = True,
        skip_unused_interventions: bool = True,
        **unused_kwargs,
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

        self.min_cells = max(0, int(min_cells or 0))
        if max_cells is None:
            max_cells = self.horizon
        self.max_cells = max(1, int(max_cells or self.horizon))
        self.force_min_cells = bool(force_min_cells)
        self.fast_reward = bool(fast_reward)
        self.skip_unused_interventions = bool(skip_unused_interventions)

        self.B, _, self.H, self.W = image.shape
        self.n_cells = self.grid * self.grid
        self.n_actions = self.n_cells + 1
        self.stop_action = self.n_cells

        self.min_cells = min(self.n_cells, max(0, self.min_cells))
        self.max_cells = min(self.n_cells, max(self.min_cells, self.max_cells))

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

        need_delta = abs(float(self.reward_fn.get("w_delta", 1.0))) > 0.0

        with torch.no_grad():
            if need_delta:
                self.e0_gap, self.identity_gap_mode = _identity_gap_tensor(
                    self.adapter,
                    self.image,
                    donor=self.donor,
                )
                self.e0_gap = _fix_vec(self.e0_gap, self.B, self.image.device)
            else:
                self.e0_gap = torch.zeros(self.B, device=self.image.device)
                self.identity_gap_mode = "proxy"

            logit = self.adapter.predict_logits(self.image)
            self.e0_logit = _logit_to_evidence(_fix_vec(logit, self.B, self.image.device))

            self.feat0 = None
            if self.cache_features:
                try:
                    # Fast after CIFTAdapter patch: this reuses the feature captured
                    # during predict_logits(self.image), avoiding another CIFT forward.
                    self.feat0 = self.adapter.extract_features(self.image).detach()
                except Exception:
                    self.feat0 = None

    def reset(self):
        self._reset_state()
        return self._state()

    def current_mask(self):
        return F.interpolate(self.mask, size=(self.H, self.W), mode="nearest")

    def action_mask(self):
        valid = torch.ones(self.B, self.n_actions, device=self.image.device, dtype=torch.bool)

        selected = self.mask[:, 0].flatten(1).sum(dim=1)

        # STOP/no-op is disabled by default. If it is enabled, still prevent it
        # before min_cells has been selected.
        if not self.allow_stop_as_noop:
            valid[:, self.stop_action] = False
        elif self.force_min_cells:
            valid[selected < float(self.min_cells), self.stop_action] = False

        if self.forbid_revisit:
            filled = self.mask[:, 0].flatten(1) > 0
            valid[:, : self.n_cells] = ~filled

        # Do not allow adding more cells after max_cells.
        reached_max = selected >= float(self.max_cells)
        if reached_max.any():
            valid[reached_max, : self.n_cells] = False
            if self.allow_stop_as_noop:
                valid[reached_max, self.stop_action] = True

        # Safety fallback: if every action became invalid, allow STOP if possible;
        # otherwise allow the first cell. This avoids NaN logits in PPO.
        dead = ~valid.any(dim=1)
        if dead.any():
            valid[dead, :] = False
            if self.allow_stop_as_noop:
                valid[dead, self.stop_action] = True
            else:
                valid[dead, 0] = True

        return valid

    def _state(self) -> Dict[str, Any]:
        return {
            "feat": self.feat0,
            "current_mask": self.mask.detach().clone(),
            "action_mask": self.action_mask().detach().clone(),
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
        has_empty = empty.any(dim=1)
        first = empty.float().argmax(dim=1).long()
        first = torch.where(has_empty, first, torch.zeros_like(first))
        return first

    def _repair_actions(self, actions):
        actions = actions.clamp(0, self.stop_action).clone()

        selected = self.mask[:, 0].flatten(1).sum(dim=1)

        # If a row already reached max_cells, do not add more cells.
        reached_max_rows = torch.nonzero(selected >= float(self.max_cells), as_tuple=False).flatten()
        if reached_max_rows.numel() > 0:
            actions[reached_max_rows] = self.stop_action

        # If STOP is disabled or min_cells is not reached, repair STOP to first empty cell.
        stop_rows = torch.nonzero(actions == self.stop_action, as_tuple=False).flatten()
        if stop_rows.numel() > 0:
            must_fill = (not self.allow_stop_as_noop)
            if self.allow_stop_as_noop and self.force_min_cells:
                must_fill_rows = stop_rows[selected[stop_rows] < float(self.min_cells)]
            elif must_fill:
                must_fill_rows = stop_rows
            else:
                must_fill_rows = torch.empty(0, device=self.image.device, dtype=torch.long)

            if must_fill_rows.numel() > 0:
                actions[must_fill_rows] = self._first_empty_cell(must_fill_rows)

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

        objective = str(self.reward_fn.get("objective", "harmonic")).strip().lower()
        w_delta = abs(float(self.reward_fn.get("w_delta", 1.0)))
        w_logit = abs(float(self.reward_fn.get("w_logit", 0.5)))
        w_dense_delta = abs(float(self.reward_fn.get("w_dense_delta", 0.0)))
        w_dense_logit = abs(float(self.reward_fn.get("w_dense_logit", 0.0)))

        need_delta = (w_delta > 0.0 or w_dense_delta > 0.0) and _is_true_gap_mode(self.identity_gap_mode)
        need_logit = w_logit > 0.0 or w_dense_logit > 0.0

        fast_skip = bool(self.fast_reward and self.skip_unused_interventions)

        if fast_skip and objective in ("sufficiency", "sufficiency_only", "suf"):
            need_nec = False
            need_suf = True
        elif fast_skip and objective in ("necessity", "necessity_only", "nec"):
            need_nec = True
            need_suf = False
        else:
            need_nec = True
            need_suf = True

        # Defaults for skipped branches. These branches are not used by the
        # selected objective during fast ablation training.
        gap_nec = self.e0_gap
        gap_suf = self.e0_gap
        l_nec = self.e0_logit
        l_suf = self.e0_logit

        with torch.no_grad():
            if need_nec:
                nec_img = apply_necessity(self.image, pm, self.intervention_mode, self.topk_frac)

                if need_delta:
                    gap_nec, _ = _identity_gap_tensor(self.adapter, nec_img, donor=self.donor)
                    gap_nec = _fix_vec(gap_nec, self.B, self.image.device)

                if need_logit:
                    l_nec = _logit_to_evidence(
                        _fix_vec(self.adapter.predict_logits(nec_img), self.B, self.image.device)
                    )

            if need_suf:
                suf_img = apply_sufficiency(self.image, pm, self.intervention_mode, self.topk_frac)

                if need_delta:
                    gap_suf, _ = _identity_gap_tensor(self.adapter, suf_img, donor=self.donor)
                    gap_suf = _fix_vec(gap_suf, self.B, self.image.device)

                if need_logit:
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

        selected_cells = self.mask[:, 0].flatten(1).sum(dim=1)
        info["selected_cells"] = float(selected_cells.float().mean().item())
        info["selected_frac"] = float((selected_cells.float() / float(self.n_cells)).mean().item())
        info["identity_gap_mode"] = str(self.identity_gap_mode)
        info["fast_reward"] = float(bool(self.fast_reward))
        info["skipped_nec"] = float(not need_nec)
        info["skipped_suf"] = float(not need_suf)

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


def _is_true_gap_mode(mode) -> bool:
    m = str(mode).strip().lower()
    return m == "true" or m.endswith(".true") or "true" in m


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
    """Vectorized RIFT reward with ablation objective support.

    objective:
      harmonic     -> full RIFT: balanced necessity + sufficiency
      necessity    -> necessity-only ablation
      sufficiency  -> sufficiency-only ablation
      none/off     -> no faithfulness objective
    """

    w = {
        "w_delta": 1.0,
        "w_logit": 0.5,
        "w_sparsity": 0.3,
        "w_identity": 0.3,
        "w_perceptual": 0.2,
        "w_plausibility": 0.0,
        "w_dense_delta": 0.25,
        "w_dense_logit": 0.10,
        "w_empty": 0.25,
        "objective": "harmonic",
    }

    if weights:
        w.update(weights)

    nec_d = _nec(e0_delta, e_nec_delta)
    suf_d = _suf(e0_delta, e_suf_delta)

    nec_l = _nec(e0_logit, e_nec_logit)
    suf_l = _suf(e0_logit, e_suf_logit)

    objective = str(w.get("objective", "harmonic")).strip().lower()

    if objective in ("necessity", "necessity_only", "nec"):
        faith_d = nec_d
        faith_l = nec_l
    elif objective in ("sufficiency", "sufficiency_only", "suf"):
        faith_d = suf_d
        faith_l = suf_l
    elif objective in ("none", "off", "zero"):
        faith_d = torch.zeros_like(nec_d)
        faith_l = torch.zeros_like(nec_l)
    else:
        faith_d = _harmonic(nec_d, suf_d)
        faith_l = _harmonic(nec_l, suf_l)

    w_delta = float(w["w_delta"]) if _is_true_gap_mode(identity_gap_mode) else 0.0
    w_logit = float(w["w_logit"])
    w_sparsity = float(w["w_sparsity"])

    dense_d = 0.5 * (nec_d + suf_d)
    dense_l = 0.5 * (nec_l + suf_l)

    reward_delta_component = w_delta * faith_d
    reward_logit_component = w_logit * faith_l
    dense_delta_component = w_delta * float(w.get("w_dense_delta", 0.0)) * dense_d
    dense_logit_component = float(w.get("w_dense_logit", 0.0)) * dense_l
    sparsity_penalty = w_sparsity * mask_area
    empty_penalty = float(w.get("w_empty", 0.0)) * (mask_area <= EPS).float()

    reward = (
        reward_delta_component
        + reward_logit_component
        + dense_delta_component
        + dense_logit_component
        - sparsity_penalty
        - empty_penalty
    )

    comps = {
        "rift_score": reward,
        "faithfulness_delta": faith_d,
        "faithfulness_logit": faith_l,
        "necessity_delta": nec_d,
        "sufficiency_delta": suf_d,
        "necessity_logit": nec_l,
        "sufficiency_logit": suf_l,
        "dense_delta": dense_d,
        "dense_logit": dense_l,
        "reward_delta_component": reward_delta_component,
        "reward_logit_component": reward_logit_component,
        "dense_delta_component": dense_delta_component,
        "dense_logit_component": dense_logit_component,
        "sparsity_penalty": sparsity_penalty,
        "empty_penalty": empty_penalty,
        "mask_area": mask_area,
        "objective": objective,
    }

    return reward, comps

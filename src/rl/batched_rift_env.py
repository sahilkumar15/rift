# Path: src/rl/batched_rift_env.py
"""Vectorized RIFT environment for fast multi-GPU PPO.

Important interpretation:

* Default fixed-budget mode is allow_stop_as_noop=False and forbid_revisit=True.
  Then every episode selects exactly horizon distinct cells. Therefore
  selected_cells, selected_frac, mask_area, and the linear sparsity penalty are
  flat by construction. They are not learning diagnostics.

* Learning/collapse must be judged from reward/evidence metrics plus action and
  mask-location diagnostics: action_entropy, action_top1_frac,
  mask_cell_entropy, active_cell_frac, unique_mask_frac, etc.

* Variable-size masks are supported with allow_stop_as_noop=True and min_cells.
  STOP freezes that sample but the batched rollout still runs to horizon for a
  stable PPO buffer shape.

* Reward math is delegated to src/faithfulness/faithfulness_score.py.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

from ..faithfulness.faithfulness_score import compute_rift_score_tensor, logit_to_evidence

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
        self.min_cells = max(0, int(min_cells))

        self.B, _, self.H, self.W = image.shape
        self.n_cells = self.grid * self.grid
        self.n_actions = self.n_cells + 1
        self.stop_action = self.n_cells

        self.step_idx = 0
        self.mask = torch.zeros(self.B, 1, self.grid, self.grid, device=self.image.device)
        self.last_action = torch.full((self.B,), -1, device=self.image.device, dtype=torch.long)
        self.done = torch.zeros(self.B, device=self.image.device, dtype=torch.bool)
        self.stopped = torch.zeros(self.B, device=self.image.device, dtype=torch.bool)
        self.e0_gap = torch.zeros(self.B, device=self.image.device)
        self.e0_logit = torch.zeros(self.B, device=self.image.device)
        self.identity_gap_mode = "proxy"
        self.feat0 = None

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def _reset_state(self):
        self.step_idx = 0
        self.mask.zero_()
        self.last_action.fill_(-1)
        self.done.zero_()
        self.stopped.zero_()

        with torch.no_grad():
            self.e0_gap, self.identity_gap_mode = _identity_gap_tensor(self.adapter, self.image, donor=self.donor)
            self.e0_gap = _fix_vec(self.e0_gap, self.B, self.image.device)

            raw_logit = self.adapter.predict_logits(self.image)
            self.e0_logit = logit_to_evidence(_fix_vec(raw_logit, self.B, self.image.device))

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

    def selected_cells_per_sample(self):
        return self.mask[:, 0].flatten(1).sum(dim=1)

    def selected_frac_per_sample(self):
        return self.selected_cells_per_sample().float() / float(max(1, self.n_cells))

    def action_mask(self):
        valid = torch.ones(self.B, self.n_actions, device=self.image.device, dtype=torch.bool)

        if self.allow_stop_as_noop:
            if self.min_cells > 0:
                enough = self.selected_cells_per_sample() >= float(self.min_cells)
                valid[:, self.stop_action] = enough
        else:
            valid[:, self.stop_action] = False

        if self.forbid_revisit:
            filled = self.mask[:, 0].flatten(1) > 0
            valid[:, : self.n_cells] = ~filled

        # Frozen/stopped samples can only emit STOP. The transition ignores them.
        if self.stopped.any():
            valid[self.stopped] = False
            valid[self.stopped, self.stop_action] = True

        dead = ~valid.any(dim=1)
        if dead.any():
            valid = valid.clone()
            valid[dead, : self.n_cells] = True
            if not self.allow_stop_as_noop:
                valid[dead, self.stop_action] = False

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
            "selected_frac": self.selected_frac_per_sample().detach().clone(),
        }

    # ------------------------------------------------------------------
    # transitions
    # ------------------------------------------------------------------

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

        if self.stopped.any():
            actions[self.stopped] = self.stop_action

        active = ~self.stopped

        stop_rows = torch.nonzero(active & (actions == self.stop_action), as_tuple=False).flatten()
        if stop_rows.numel() > 0:
            if self.allow_stop_as_noop:
                enough = self.selected_cells_per_sample()[stop_rows] >= float(self.min_cells)
                bad = stop_rows[~enough]
            else:
                bad = stop_rows
            if bad.numel() > 0:
                actions[bad] = self._first_empty_cell(bad)

        if self.forbid_revisit:
            cell_rows = torch.nonzero(active & (actions != self.stop_action), as_tuple=False).flatten()
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

        cell_mask = (actions != self.stop_action) & (~self.stopped)
        if cell_mask.any():
            idx = torch.nonzero(cell_mask, as_tuple=False).flatten()
            rr = torch.div(actions[idx], self.grid, rounding_mode="floor")
            cc = actions[idx] % self.grid
            self.mask[idx, 0, rr, cc] = 1.0

        if self.allow_stop_as_noop:
            legal_stop = (actions == self.stop_action) & (~self.stopped)
            if legal_stop.any():
                self.stopped = self.stopped | legal_stop

        done_now = self.step_idx >= self.horizon
        self.done = torch.full((self.B,), bool(done_now), device=self.image.device, dtype=torch.bool)

        if done_now:
            reward, info = self._terminal_reward()
        else:
            area = self.selected_frac_per_sample()
            reward = -0.01 * torch.clamp(area - 0.5, min=0.0)
            info = {"mask_area": float(area.mean().item())}

        return self._state(), reward.detach(), self.done.detach().clone(), info

    # ------------------------------------------------------------------
    # terminal reward
    # ------------------------------------------------------------------

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

            l_nec = logit_to_evidence(_fix_vec(self.adapter.predict_logits(nec_img), self.B, self.image.device))
            l_suf = logit_to_evidence(_fix_vec(self.adapter.predict_logits(suf_img), self.B, self.image.device))

        selected_cells = self.selected_cells_per_sample().float()
        area = selected_cells / float(max(1, self.n_cells))

        reward, comps = compute_rift_score_tensor(
            e0_delta=self.e0_gap,
            e_nec_delta=gap_nec,
            e_suf_delta=gap_suf,
            e0_logit=self.e0_logit,
            e_nec_logit=l_nec,
            e_suf_logit=l_suf,
            mask_area=area,
            selected_cells=selected_cells,
            identity_gap_mode=str(self.identity_gap_mode),
            weights=self.reward_fn,
        )

        info = {k: float(v.mean().item()) if torch.is_tensor(v) else v for k, v in comps.items()}
        info.update(_mask_diagnostics(self.mask))
        info["selected_cells"] = float(selected_cells.mean().item())
        info["selected_cells_std"] = float(selected_cells.std(unbiased=False).item())
        info["selected_cells_min"] = float(selected_cells.min().item())
        info["selected_cells_max"] = float(selected_cells.max().item())
        info["selected_frac"] = float(area.mean().item())
        info["stopped_frac"] = float(self.stopped.float().mean().item())
        info["identity_gap_mode"] = str(self.identity_gap_mode)
        info["effective_w_delta"] = float(self.reward_fn.get("w_delta", 1.0)) if str(self.identity_gap_mode) == "true" else 0.0
        return reward, info


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


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


def _normalize_mode(mode) -> str:
    return str(getattr(mode, "value", mode))


def _identity_gap_tensor(adapter, image, donor=None):
    if hasattr(adapter, "identity_gap_tensor"):
        gap, mode = adapter.identity_gap_tensor(image, donor=donor)
        return gap, _normalize_mode(mode)

    res = adapter.identity_gap(image, donor=donor)
    mode = _normalize_mode(getattr(res, "mode", "proxy"))
    return torch.full((image.shape[0],), float(res.value), device=image.device), mode


def _mask_diagnostics(mask):
    """Batch-level mask-location diagnostics that reveal spatial collapse."""
    if mask.numel() == 0:
        return {}

    B = max(1, int(mask.shape[0]))
    flat = (mask[:, 0].flatten(1) > 0).float()
    per_cell = flat.mean(dim=0)
    active = per_cell > 0

    total_mass = per_cell.sum()
    if float(total_mass.item()) > 0:
        q = per_cell / total_mass.clamp_min(EPS)
        entropy = -(q * torch.log(q + EPS)).sum() / math.log(max(2, flat.shape[1]))
        max_frac = per_cell.max()
    else:
        entropy = torch.zeros((), device=mask.device)
        max_frac = torch.zeros((), device=mask.device)

    rows = torch.arange(mask.shape[-2], device=mask.device).float().view(1, -1, 1)
    cols = torch.arange(mask.shape[-1], device=mask.device).float().view(1, 1, -1)
    mass = mask[:, 0].sum(dim=(1, 2)).clamp_min(EPS)
    center_r = (mask[:, 0] * rows).sum(dim=(1, 2)) / mass
    center_c = (mask[:, 0] * cols).sum(dim=(1, 2)) / mass

    try:
        uniq = torch.unique(flat.to(torch.uint8), dim=0).shape[0] / float(B)
    except Exception:
        uniq = 0.0

    return {
        "mask_cell_entropy": float(entropy.item()),
        "mask_cell_max_frac": float(max_frac.item()),
        "active_cell_frac": float(active.float().mean().item()),
        "unique_mask_frac": float(uniq),
        "mask_center_row": float(center_r.mean().item()),
        "mask_center_col": float(center_c.mean().item()),
    }

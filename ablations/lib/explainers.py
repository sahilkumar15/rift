from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _to_float(x) -> float:
    try:
        import torch
        if torch.is_tensor(x):
            return float(x.detach().float().mean().item())
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return 0.0


def logit_to_evidence(x):
    import torch
    import torch.nn.functional as F
    if torch.is_tensor(x):
        return F.softplus(x.detach().float())
    return float(F.softplus(torch.tensor(float(x))).item())


def sigmoid_mean(x) -> float:
    """Backward-compatible name. RIFT uses softplus evidence, not sigmoid."""
    ev = logit_to_evidence(x)
    try:
        return float(ev.mean().item())
    except Exception:
        return float(ev)


def gap_value(res) -> float:
    if hasattr(res, "value"):
        return _to_float(res.value)
    return _to_float(res)


def _gap_mode(res) -> str:
    m = getattr(res, "mode", "proxy")
    return str(getattr(m, "value", m))


def _tensor_key(x) -> Optional[Tuple[int, Tuple[int, ...], str, str]]:
    try:
        import torch
        if not torch.is_tensor(x):
            return None
        return int(x.data_ptr()), tuple(x.shape), str(x.device), str(x.dtype)
    except Exception:
        return None


def _repeat_interleave_batch(x, repeats: int):
    if x is None:
        return None
    shape = x.shape
    return x[:, None].expand(shape[0], repeats, *shape[1:]).reshape(shape[0] * repeats, *shape[1:])


def _mode_from_donor(donor) -> str:
    return "true" if donor is not None else "proxy"


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text and ("cuda" in text or "cudnn" in text)


def _direct_evidence_once(adapter, image, donor=None, *, return_features: bool = False):
    """Evaluate CIFT evidence with a verified source-free-logit fast path.

    A donor-grounded CIFT forward always supplies the identity gap. Depending on
    the external CIFT implementation, its returned validation logit may either be
    the deployment/source-free logit or a donor-conditioned training fusion. On
    the first donor batch we compare it against predict_logits(image). If they
    match, subsequent calls reuse the combined one-pass path. If not, we retain
    the donor forward for gap but run the separate source-free logit path. This
    makes the optimization fast without silently changing the audited evidence.
    """
    import warnings
    import torch

    image = image.to(adapter.device).float()
    donor = donor.to(adapter.device).float() if donor is not None else None
    features = None

    with torch.inference_mode():
        if hasattr(adapter, "_forward"):
            logits, gap = adapter._forward(image, donor=donor)
            logits = logits.detach().float().view(-1)
            gap = gap.detach().float().view(-1)

            combined_ok = getattr(adapter, "_rift_combined_logits_verified", None)
            if donor is not None and combined_ok is None:
                reference = adapter.predict_logits(image).detach().float().view(-1)
                combined_ok = bool(
                    torch.allclose(
                        logits,
                        reference,
                        rtol=1e-4,
                        atol=1e-5,
                    )
                )
                adapter._rift_combined_logits_verified = combined_ok
                adapter._rift_combined_logits_max_abs_diff = float(
                    (logits - reference).abs().max().item()
                )
                if not combined_ok:
                    warnings.warn(
                        "CIFT donor-forward logits differ from source-free logits; "
                        "using the safe two-pass logit+gap path.",
                        RuntimeWarning,
                    )
                logits = reference
            elif donor is not None and combined_ok is False:
                logits = adapter.predict_logits(image).detach().float().view(-1)

            if return_features:
                feat = getattr(adapter, "_spatial_feat", None)
                if torch.is_tensor(feat):
                    if hasattr(adapter, "_crop_spatial_feature"):
                        features = adapter._crop_spatial_feature(feat, int(image.shape[0]))
                    elif feat.shape[0] >= image.shape[0]:
                        features = feat[-image.shape[0]:].detach().contiguous()
        else:
            logits = adapter.predict_logits(image).detach().float().view(-1)
            if hasattr(adapter, "identity_gap_tensor"):
                gap, mode = adapter.identity_gap_tensor(image, donor=donor)
                gap = gap.detach().float().view(-1)
            else:
                res = adapter.identity_gap(image, donor=donor)
                mode = _gap_mode(res)
                gap = torch.full((image.shape[0],), gap_value(res), device=image.device)

            if return_features:
                try:
                    features = adapter.extract_features(image).detach()
                except Exception:
                    features = None

    batch = int(image.shape[0])
    if logits.numel() == 1 and batch > 1:
        logits = logits.repeat(batch)
    if gap.numel() == 1 and batch > 1:
        gap = gap.repeat(batch)
    logits = logits[:batch].contiguous()
    gap = gap[:batch].contiguous()

    mode = _mode_from_donor(donor)
    return logits, gap, mode, features


def predict_evidence(
    adapter,
    image,
    donor=None,
    *,
    max_batch: int = 32,
    return_features: bool = False,
):
    """Chunked, OOM-adaptive CIFT evidence evaluation.

    Returns raw logits [B], identity gaps [B], mode string, and optional spatial
    features. If a requested chunk does not fit, it is split recursively until it
    fits, so a fast setting remains safe across different GPUs.
    """
    import torch

    image = image.to(adapter.device).float()
    donor = donor.to(adapter.device).float() if donor is not None else None
    total = int(image.shape[0])
    max_batch = max(1, int(max_batch or total))

    logits_parts = []
    gap_parts = []
    feat_parts = []
    modes = []

    def run_slice(start: int, end: int):
        try:
            d = donor[start:end] if donor is not None else None
            out = _direct_evidence_once(
                adapter,
                image[start:end],
                donor=d,
                return_features=return_features,
            )
            logits_parts.append((start, out[0]))
            gap_parts.append((start, out[1]))
            modes.append(out[2])
            if return_features and out[3] is not None:
                feat_parts.append((start, out[3]))
        except RuntimeError as exc:
            if not _is_cuda_oom(exc) or end - start <= 1:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mid = start + (end - start) // 2
            run_slice(start, mid)
            run_slice(mid, end)

    for start in range(0, total, max_batch):
        run_slice(start, min(total, start + max_batch))

    logits_parts.sort(key=lambda z: z[0])
    gap_parts.sort(key=lambda z: z[0])
    logits = torch.cat([x for _, x in logits_parts], dim=0)
    gaps = torch.cat([x for _, x in gap_parts], dim=0)

    features = None
    if return_features and len(feat_parts) == len(logits_parts):
        feat_parts.sort(key=lambda z: z[0])
        features = torch.cat([x for _, x in feat_parts], dim=0)

    mode = "true" if donor is not None else (modes[0] if modes else "proxy")
    return logits, gaps, mode, features


class CausalSelectExplainer:
    """Exact greedy causal cell selection with batched candidate trials.

    For pool=16 and horizon=4, the old code performed 58 candidate trials and
    recomputed original, necessity, and sufficiency evidence separately for both
    channels, causing roughly 348 CIFT forwards per image. This implementation
    preserves the same greedy objective but evaluates every remaining candidate
    of a step in one batched CIFT call. A donor-grounded CIFT forward already
    returns both the deployed logit and identity gap, so no duplicate channel
    passes are needed. The final selected candidate's evidence is cached and
    reused by the outer evaluator.
    """

    name = "causal_select"

    def __init__(
        self,
        base,
        *,
        channel: str = "delta",
        grid: int = 8,
        horizon: int = 4,
        candidate_pool: int = 16,
        intervention_mode: str = "blur",
        topk_frac: float = 0.12,
        forward_batch_size: int = 32,
    ):
        self.base = base
        self.channel = str(channel)
        self.grid = int(grid)
        self.horizon = int(horizon)
        self.candidate_pool = int(candidate_pool)
        self.intervention_mode = str(intervention_mode)
        self.topk_frac = float(topk_frac)
        self.forward_batch_size = int(forward_batch_size)
        self._cache_key = None
        self._cache = None

    def cached_original_evidence(self, image):
        if self._cache_key == _tensor_key(image):
            return self._cache
        return None

    def explain(self, image, adapter, **kw):
        import torch
        import torch.nn.functional as F

        from src.faithfulness.faithfulness_score import compute_rift_score_tensor
        from src.interventions.interventions import apply_necessity, apply_sufficiency
        from src.rl.reward import get_reward_weights

        donor = kw.get("donor")
        image = image.to(adapter.device).float()
        donor = donor.to(adapter.device).float() if donor is not None else None
        batch = int(image.shape[0])

        base_mask = self.base.explain(image, adapter, donor=donor).detach()
        saliency = F.adaptive_avg_pool2d(base_mask, (self.grid, self.grid)).flatten(1)
        k = min(max(self.candidate_pool, self.horizon), saliency.shape[1])
        candidates = saliency.topk(k, dim=1).indices

        raw0, gap0, mode, _ = predict_evidence(
            adapter,
            image,
            donor,
            max_batch=self.forward_batch_size,
            return_features=False,
        )
        logit0 = logit_to_evidence(raw0)

        grid_mask = torch.zeros(
            batch,
            1,
            self.grid,
            self.grid,
            device=image.device,
            dtype=image.dtype,
        )
        chosen = torch.zeros(batch, k, device=image.device, dtype=torch.bool)

        final_gap_nec = gap0.clone()
        final_gap_suf = gap0.clone()
        final_logit_nec = logit0.clone()
        final_logit_suf = logit0.clone()

        steps = min(self.horizon, k)
        weights = get_reward_weights("full_rift")

        for step in range(steps):
            pairs = (~chosen).nonzero(as_tuple=False)
            if pairs.numel() == 0:
                break

            batch_idx = pairs[:, 0]
            slot_idx = pairs[:, 1]
            cell_idx = candidates[batch_idx, slot_idx]

            trial_grid = grid_mask[batch_idx].clone()
            rr = torch.div(cell_idx, self.grid, rounding_mode="floor")
            cc = cell_idx % self.grid
            rows = torch.arange(pairs.shape[0], device=image.device)
            trial_grid[rows, 0, rr, cc] = 1.0
            trial_mask = F.interpolate(
                trial_grid,
                size=image.shape[-2:],
                mode="nearest",
            )

            image_trials = image[batch_idx]
            donor_trials = donor[batch_idx] if donor is not None else None
            nec = apply_necessity(
                image_trials,
                trial_mask,
                self.intervention_mode,
                self.topk_frac,
            )
            suf = apply_sufficiency(
                image_trials,
                trial_mask,
                self.intervention_mode,
                self.topk_frac,
            )

            all_images = torch.cat([nec, suf], dim=0)
            all_donors = None
            if donor_trials is not None:
                all_donors = torch.cat([donor_trials, donor_trials], dim=0)

            raw, gaps, _, _ = predict_evidence(
                adapter,
                all_images,
                all_donors,
                max_batch=self.forward_batch_size,
                return_features=False,
            )
            logit_ev = logit_to_evidence(raw)
            n_trials = int(pairs.shape[0])

            gap_nec = gaps[:n_trials]
            gap_suf = gaps[n_trials:]
            logit_nec = logit_ev[:n_trials]
            logit_suf = logit_ev[n_trials:]

            area = torch.full(
                (n_trials,),
                float(step + 1) / float(self.grid * self.grid),
                device=image.device,
            )
            selected_cells = torch.full_like(area, float(step + 1))

            reward, comps = compute_rift_score_tensor(
                e0_delta=gap0[batch_idx],
                e_nec_delta=gap_nec,
                e_suf_delta=gap_suf,
                e0_logit=logit0[batch_idx],
                e_nec_logit=logit_nec,
                e_suf_logit=logit_suf,
                mask_area=area,
                selected_cells=selected_cells,
                identity_gap_mode=mode,
                weights=weights,
            )

            channel = self.channel.strip().lower()
            if channel == "delta":
                trial_score = comps["faithfulness_delta"]
            elif channel == "logit":
                trial_score = comps["faithfulness_logit"]
            else:
                trial_score = reward

            score_matrix = torch.full(
                (batch, k),
                -torch.inf,
                device=image.device,
            )
            score_matrix[batch_idx, slot_idx] = trial_score
            best_slot = score_matrix.argmax(dim=1)
            best_cell = candidates.gather(1, best_slot[:, None]).squeeze(1)

            out_rows = torch.arange(batch, device=image.device)
            best_rr = torch.div(best_cell, self.grid, rounding_mode="floor")
            best_cc = best_cell % self.grid
            grid_mask[out_rows, 0, best_rr, best_cc] = 1.0
            chosen[out_rows, best_slot] = True

            # Locate each chosen pair in the flattened trial list and retain its
            # final evidence so the outer evaluator does not recompute it.
            selected_pair = (batch_idx[:, None] == out_rows[None, :]) & (
                slot_idx[:, None] == best_slot[None, :]
            )
            pair_position = selected_pair.float().argmax(dim=0).long()
            final_gap_nec = gap_nec[pair_position]
            final_gap_suf = gap_suf[pair_position]
            final_logit_nec = logit_nec[pair_position]
            final_logit_suf = logit_suf[pair_position]

        self._cache_key = _tensor_key(image)
        self._cache = {
            "gap": gap0.detach(),
            "logit": logit0.detach(),
            "gap_nec": final_gap_nec.detach(),
            "gap_suf": final_gap_suf.detach(),
            "logit_nec": final_logit_nec.detach(),
            "logit_suf": final_logit_suf.detach(),
            "mode": mode,
            "complete": True,
        }

        return F.interpolate(
            grid_mask,
            size=image.shape[-2:],
            mode="nearest",
        ).detach()


class PolicyExplainer:
    """Load a GridPolicy checkpoint and perform deterministic evaluation.

    Supports both fixed-budget policies and variable-budget policies with a
    permanent STOP action. Once a sample stops, its mask remains unchanged for
    all remaining rollout steps.
    """

    name = "rift_policy"

    def __init__(
        self,
        ckpt_path: str,
        *,
        grid: int = 8,
        hidden: int = 256,
        feat_dim: int = 1024,
        horizon: int = 4,
        reward_preset: str = "full_rift",
        intervention_mode: str = "blur",
        topk_frac: float = 0.12,
        device: str = "cuda:0",
        forward_batch_size: int = 32,
        allow_stop: bool = False,
        min_cells: int = 1,
        max_cells: Optional[int] = None,
        force_min_cells: bool = True,
        forbid_revisit: bool = True,
        state_blind: bool = False,
    ):
        self.ckpt_path = str(ckpt_path)
        self.grid = int(grid)
        self.hidden = int(hidden)
        self.feat_dim = int(feat_dim)
        self.horizon = int(horizon)
        self.reward_preset = str(reward_preset)
        self.intervention_mode = str(intervention_mode)
        self.topk_frac = float(topk_frac)
        self.device = str(device)
        self.forward_batch_size = max(1, int(forward_batch_size))

        self.allow_stop = bool(allow_stop)
        self.min_cells = max(0, int(min_cells))
        self.force_min_cells = bool(force_min_cells)
        self.forbid_revisit = bool(forbid_revisit)
        self.state_blind = bool(state_blind)

        if max_cells is None:
            max_cells = self.horizon

        self.max_cells = min(
            self.grid * self.grid,
            max(self.min_cells, int(max_cells)),
        )

        self._policy = None
        self._cache_key = None
        self._cache = None

    def _load_policy(self):
        import torch
        from src.rl.policy import GridPolicy

        if self._policy is not None:
            return self._policy

        path = Path(self.ckpt_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Policy checkpoint not found: {path}"
            )

        try:
            state = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            state = torch.load(path, map_location="cpu")

        state = state.get("policy", state)

        cleaned = {
            (
                str(key)[7:]
                if str(key).startswith("module.")
                else str(key)
            ): value
            for key, value in state.items()
        }

        policy = GridPolicy(
            grid=self.grid,
            n_actions=self.grid * self.grid + 1,
            hidden=self.hidden,
            feat_dim=self.feat_dim,
            state_blind=self.state_blind,
        ).to(self.device)

        policy.load_state_dict(cleaned, strict=True)
        policy.eval()

        self._policy = policy
        return policy

    def cached_original_evidence(self, image):
        if self._cache_key == _tensor_key(image):
            return self._cache
        return None

    def explain(self, image, adapter, **kw):
        import torch
        import torch.nn.functional as F

        policy = self._load_policy()

        image = image.to(self.device).float()

        donor = kw.get("donor")
        if donor is not None:
            donor = donor.to(self.device).float()

        batch = int(image.shape[0])
        n_cells = self.grid * self.grid
        stop_action = n_cells

        raw_logits, gaps, mode, features = predict_evidence(
            adapter,
            image,
            donor,
            max_batch=self.forward_batch_size,
            return_features=True,
        )

        confidence = logit_to_evidence(raw_logits)

        mask = torch.zeros(
            batch,
            1,
            self.grid,
            self.grid,
            device=image.device,
            dtype=image.dtype,
        )

        last_action = torch.full(
            (batch,),
            -1.0,
            device=image.device,
        )

        stopped = torch.zeros(
            batch,
            device=image.device,
            dtype=torch.bool,
        )

        with torch.inference_mode():
            for step in range(self.horizon):
                selected = (
                    mask[:, 0]
                    .flatten(1)
                    .sum(dim=1)
                )

                filled = mask[:, 0].flatten(1) > 0

                action_mask = torch.ones(
                    batch,
                    n_cells + 1,
                    device=image.device,
                    dtype=torch.bool,
                )

                if self.forbid_revisit:
                    action_mask[:, :n_cells] = ~filled

                if not self.allow_stop:
                    action_mask[:, stop_action] = False
                elif self.force_min_cells:
                    before_minimum = selected < float(self.min_cells)
                    action_mask[
                        before_minimum,
                        stop_action,
                    ] = False

                reached_maximum = selected >= float(self.max_cells)

                if reached_maximum.any():
                    action_mask[
                        reached_maximum,
                        :n_cells,
                    ] = False
                    action_mask[
                        reached_maximum,
                        stop_action,
                    ] = True

                # A stopped sample remains permanently stopped.
                if stopped.any():
                    action_mask[stopped, :] = False
                    action_mask[stopped, stop_action] = True

                dead = ~action_mask.any(dim=1)

                if dead.any():
                    action_mask[dead, :] = False

                    if self.allow_stop:
                        action_mask[dead, stop_action] = True
                    else:
                        action_mask[dead, 0] = True

                state = {
                    "feat": features,
                    "current_mask": mask,
                    "action_mask": action_mask,
                    "confidence": confidence,
                    "step_idx": torch.full(
                        (batch,),
                        float(step),
                        device=image.device,
                    ),
                    "last_action": last_action,
                    "selected_frac": (
                        selected.float() / float(n_cells)
                    ),
                    "stopped": stopped,
                    "e0_gap": gaps,
                }

                logits, _ = policy(state)
                logits = logits.masked_fill(
                    ~action_mask,
                    -1e9,
                )

                action = logits.argmax(dim=-1)

                active_before = ~stopped

                newly_stopped = (
                    active_before
                    & (action == stop_action)
                    & self.allow_stop
                )

                stopped = stopped | newly_stopped

                choose_cell = (
                    active_before
                    & (~newly_stopped)
                    & (action != stop_action)
                )

                if choose_cell.any():
                    rows = torch.nonzero(
                        choose_cell,
                        as_tuple=False,
                    ).flatten()

                    rr = torch.div(
                        action[rows],
                        self.grid,
                        rounding_mode="floor",
                    )
                    cc = action[rows] % self.grid

                    mask[rows, 0, rr, cc] = 1.0

                selected_after = (
                    mask[:, 0]
                    .flatten(1)
                    .sum(dim=1)
                )

                if self.allow_stop:
                    stopped = stopped | (
                        selected_after >= float(self.max_cells)
                    )

                last_action = action.float()

                if stopped.any():
                    last_action = torch.where(
                        stopped,
                        torch.full_like(
                            last_action,
                            float(stop_action),
                        ),
                        last_action,
                    )

                if (
                    self.allow_stop
                    and bool(stopped.all().item())
                ):
                    break

        selected_final = (
            mask[:, 0]
            .flatten(1)
            .sum(dim=1)
            .float()
        )

        self._cache_key = _tensor_key(image)
        self._cache = {
            "gap": gaps.detach(),
            "logit": confidence.detach(),
            "mode": mode,
            "selected_cells": selected_final.detach(),
            "stopped": stopped.detach(),
        }

        return F.interpolate(
            mask,
            size=image.shape[-2:],
            mode="nearest",
        ).detach()


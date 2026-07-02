from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


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


def sigmoid_mean(x) -> float:
    import torch
    import torch.nn.functional as F
    if not torch.is_tensor(x):
        return float(x)
    return float(F.softplus(x.detach().float()).mean().item())


def gap_value(res) -> float:
    if hasattr(res, "value"):
        return _to_float(res.value)
    return _to_float(res)


def _gap_mode(res) -> str:
    m = getattr(res, "mode", "proxy")
    return str(getattr(m, "value", m))


class CausalSelectExplainer:
    """Greedy causal cell selection on top of a base saliency map.

    This is an evaluation-only baseline for Tables 1/2/3. It does not train a
    policy; it greedily chooses grid cells from a candidate saliency pool that
    maximize the current necessity/sufficiency RIFT component.
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
    ):
        self.base = base
        self.channel = str(channel)
        self.grid = int(grid)
        self.horizon = int(horizon)
        self.candidate_pool = int(candidate_pool)
        self.intervention_mode = intervention_mode
        self.topk_frac = float(topk_frac)

    def _score(self, image, donor, adapter, grid_mask) -> float:
        import torch
        import torch.nn.functional as F
        from src.faithfulness.faithfulness_score import compute_rift_score
        from src.interventions.interventions import apply_necessity, apply_sufficiency, mask_area
        from src.rl.reward import get_reward_weights

        mask = F.interpolate(grid_mask, size=image.shape[-2:], mode="nearest")
        weights = get_reward_weights("full_rift")

        with torch.no_grad():
            g0 = adapter.identity_gap(image, donor=donor)
            l0 = sigmoid_mean(adapter.predict_logits(image))
            nec = apply_necessity(image, mask, self.intervention_mode, self.topk_frac)
            suf = apply_sufficiency(image, mask, self.intervention_mode, self.topk_frac)
            gn = adapter.identity_gap(nec, donor=donor)
            gs = adapter.identity_gap(suf, donor=donor)
            ln = sigmoid_mean(adapter.predict_logits(nec))
            ls = sigmoid_mean(adapter.predict_logits(suf))

            comp = compute_rift_score(
                e0_delta=gap_value(g0),
                e_nec_delta=gap_value(gn),
                e_suf_delta=gap_value(gs),
                e0_logit=l0,
                e_nec_logit=ln,
                e_suf_logit=ls,
                mask_area=mask_area(mask, self.topk_frac),
                identity_gap_mode=_gap_mode(g0),
                weights=weights,
            )

        if self.channel == "delta":
            return float(comp.faithfulness_delta)
        if self.channel == "logit":
            return float(comp.faithfulness_logit)
        return float(comp.rift_score)

    def explain(self, image, adapter, **kw):
        import torch
        import torch.nn.functional as F

        donor = kw.get("donor")
        base_mask = self.base.explain(image, adapter, **kw).detach()
        sal = F.adaptive_avg_pool2d(base_mask, (self.grid, self.grid)).flatten(1)
        k = min(max(self.candidate_pool, self.horizon), sal.shape[1])
        candidates = sal.topk(k, dim=1).indices[0].tolist()

        grid_mask = torch.zeros(image.shape[0], 1, self.grid, self.grid, device=image.device)
        chosen = set()

        for _ in range(min(self.horizon, len(candidates))):
            best_cell = None
            best_score = None

            for cell in candidates:
                if cell in chosen:
                    continue
                trial = grid_mask.clone()
                rr, cc = divmod(int(cell), self.grid)
                trial[:, 0, rr, cc] = 1.0
                score = self._score(image, donor, adapter, trial)
                if best_score is None or score > best_score:
                    best_score = score
                    best_cell = int(cell)

            if best_cell is None:
                break
            chosen.add(best_cell)
            rr, cc = divmod(best_cell, self.grid)
            grid_mask[:, 0, rr, cc] = 1.0

        return F.interpolate(grid_mask, size=image.shape[-2:], mode="nearest").detach()


class PolicyExplainer:
    """Load a trained GridPolicy checkpoint and roll it out greedily."""

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
        self._policy = None

    def _load_policy(self):
        import torch
        from src.rl.policy import GridPolicy

        if self._policy is not None:
            return self._policy

        p = Path(self.ckpt_path)
        if not p.exists():
            raise FileNotFoundError(f"Policy checkpoint not found: {p}")

        try:
            st = torch.load(p, map_location="cpu", weights_only=False)
        except TypeError:
            st = torch.load(p, map_location="cpu")

        state = st.get("policy", st)
        cleaned = {}
        for k, v in state.items():
            cleaned[k[7:] if str(k).startswith("module.") else k] = v

        policy = GridPolicy(
            grid=self.grid,
            n_actions=self.grid * self.grid + 1,
            hidden=self.hidden,
            feat_dim=self.feat_dim,
        ).to(self.device)
        policy.load_state_dict(cleaned, strict=True)
        policy.eval()
        self._policy = policy
        return self._policy

    def explain(self, image, adapter, **kw):
        import torch
        from src.rl.batched_rift_env import BatchedRIFTEnv
        from src.rl.reward import get_reward_weights

        policy = self._load_policy()
        img = image.to(self.device)
        donor = kw.get("donor")
        donor = donor.to(self.device) if donor is not None else None

        env = BatchedRIFTEnv(
            img,
            adapter,
            grid=self.grid,
            horizon=self.horizon,
            intervention_mode=self.intervention_mode,
            topk_frac=self.topk_frac,
            reward_fn=get_reward_weights(self.reward_preset),
            donor=donor,
            allow_stop_as_noop=False,
            forbid_revisit=True,
            cache_features=True,
        )

        state = env.reset()
        with torch.no_grad():
            for _ in range(self.horizon):
                logits, _ = policy(state)
                mask = state.get("action_mask")
                if torch.is_tensor(mask):
                    logits = logits.masked_fill(~mask.to(logits.device).bool(), -1e9)
                action = logits.argmax(dim=-1)
                state, _, done, _ = env.step(action)
                if bool(done.all().item()):
                    break

        return env.current_mask().detach()

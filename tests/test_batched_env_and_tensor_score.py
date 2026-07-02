# Path: tests/test_batched_env_and_tensor_score.py
"""Regression tests for the exact failure modes seen in run RIFT_ffpp_h4_full_3
and the fixes: single scorer, evidence transform, variable-size masks."""
import torch
import pytest

from src.faithfulness.faithfulness_score import (
    compute_rift_score, compute_rift_score_tensor, logit_to_evidence)
from src.rl.batched_rift_env import BatchedRIFTEnv


class DummyAdapter:
    """Detector whose fake-evidence lives in the top-left image quadrant."""
    def predict_logits(self, img):
        B, _, H, W = img.shape
        return img[:, :, : H // 2, : W // 2].mean(dim=(1, 2, 3)) * 10.0 - 2.0

    def identity_gap_tensor(self, img, donor=None):
        B, _, H, W = img.shape
        gap = img[:, :, : H // 2, : W // 2].abs().mean(dim=(1, 2, 3))
        return gap, "true" if donor is not None else "proxy"

    def extract_features(self, img):
        return torch.randn(img.shape[0], 1024)


def _env(B=6, horizon=4, allow_stop=False, min_cells=1, weights=None):
    torch.manual_seed(0)
    img = torch.rand(B, 3, 32, 32)
    return BatchedRIFTEnv(
        img, DummyAdapter(), grid=8, horizon=horizon,
        intervention_mode="zero", topk_frac=0.12,
        reward_fn=weights or {}, donor=img.clone(),
        allow_stop_as_noop=allow_stop, min_cells=min_cells,
        cache_features=False,
    )


def test_negative_logits_do_not_zero_faithfulness():
    # real-image regime: raw logit <= 0 must not collapse nec/suf to 0
    e0 = logit_to_evidence(torch.tensor([-2.0, -0.5]))
    en = logit_to_evidence(torch.tensor([-4.0, -3.0]))
    es = logit_to_evidence(torch.tensor([-2.2, -0.8]))
    assert (e0 > 0).all()
    r, c = compute_rift_score_tensor(
        e0_delta=e0, e_nec_delta=en, e_suf_delta=es,
        e0_logit=e0, e_nec_logit=en, e_suf_logit=es,
        mask_area=torch.full((2,), 0.1), identity_gap_mode="true")
    assert (c["necessity_logit"] > 0).all()
    assert (c["sufficiency_logit"] > 0).all()


def test_scalar_and_tensor_scorers_agree():
    kw = dict(e0_delta=0.8, e_nec_delta=0.2, e_suf_delta=0.6,
              e0_logit=1.5, e_nec_logit=0.4, e_suf_logit=1.1,
              mask_area=0.1, identity_gap_mode="true")
    scalar = compute_rift_score(**kw).rift_score
    tkw = {k: (torch.tensor([v]) if isinstance(v, float) else v) for k, v in kw.items()}
    tkw["identity_gap_mode"] = "true"
    tensor, _ = compute_rift_score_tensor(**tkw)
    assert abs(scalar - float(tensor[0])) < 1e-5


def test_proxy_mode_strips_delta_credit_tensor():
    kw = dict(e0_delta=torch.tensor([1.0]), e_nec_delta=torch.tensor([0.0]),
              e_suf_delta=torch.tensor([1.0]), e0_logit=torch.tensor([2.0]),
              e_nec_logit=torch.tensor([0.0]), e_suf_logit=torch.tensor([2.0]),
              mask_area=torch.tensor([0.1]))
    rt, _ = compute_rift_score_tensor(identity_gap_mode="true", **kw)
    rp, _ = compute_rift_score_tensor(identity_gap_mode="proxy", **kw)
    assert float(rp[0]) < float(rt[0])


def test_hinge_sparsity_zero_inside_band_penalizes_outside():
    base = dict(e0_delta=torch.ones(3), e_nec_delta=torch.zeros(3),
                e_suf_delta=torch.ones(3), e0_logit=torch.ones(3),
                e_nec_logit=torch.zeros(3), e_suf_logit=torch.ones(3),
                identity_gap_mode="true",
                weights={"sparsity_mode": "hinge", "area_lo": 0.02, "area_hi": 0.35})
    _, c = compute_rift_score_tensor(mask_area=torch.tensor([0.1, 0.005, 0.6]), **base)
    pen = c["sparsity_penalty"]
    assert float(pen[0]) == 0.0          # inside band
    assert float(pen[1]) > 0.0           # too small
    assert float(pen[2]) > 0.0           # too large


def test_empty_mask_is_penalized():
    kw = dict(e0_delta=torch.ones(1), e_nec_delta=torch.ones(1),
              e_suf_delta=torch.zeros(1), e0_logit=torch.ones(1),
              e_nec_logit=torch.ones(1), e_suf_logit=torch.zeros(1),
              identity_gap_mode="true")
    r_empty, _ = compute_rift_score_tensor(mask_area=torch.zeros(1), **kw)
    r_tiny, _ = compute_rift_score_tensor(mask_area=torch.full((1,), 1 / 64), **kw)
    assert float(r_empty[0]) < float(r_tiny[0])


def test_min_evidence_gates_out_no_evidence_samples():
    kw = dict(e0_delta=torch.tensor([1.0, 0.01]), e_nec_delta=torch.zeros(2),
              e_suf_delta=torch.tensor([1.0, 0.01]),
              e0_logit=torch.tensor([1.0, 0.01]), e_nec_logit=torch.zeros(2),
              e_suf_logit=torch.tensor([1.0, 0.01]),
              mask_area=torch.full((2,), 0.1), identity_gap_mode="true",
              weights={"min_evidence": 0.05})
    _, c = compute_rift_score_tensor(**kw)
    assert float(c["faithfulness_logit"][1]) == 0.0
    assert abs(float(c["valid_frac_logit"]) - 0.5) < 1e-6


def test_fixed_budget_selects_exactly_horizon_cells():
    # documents WHY selected_cells flatlines at horizon in the default config
    env = _env(horizon=4, allow_stop=False)
    env.reset()
    done = torch.zeros(6, dtype=torch.bool)
    while not bool(done.all()):
        a = torch.randint(0, env.n_actions, (6,))
        _, _, done, info = env.step(a)
    assert info["selected_cells"] == pytest.approx(4.0)
    assert info["selected_cells_std"] == pytest.approx(0.0)
    assert info["mask_area"] == pytest.approx(4 / 64)


def test_allow_stop_yields_variable_mask_sizes():
    env = _env(B=32, horizon=8, allow_stop=True, min_cells=2)
    env.reset()
    done = torch.zeros(32, dtype=torch.bool)
    torch.manual_seed(1)
    while not bool(done.all()):
        a = torch.randint(0, env.n_actions, (32,))
        _, _, done, info = env.step(a)
    assert 2.0 <= info["selected_cells"] <= 8.0
    assert info["selected_cells_std"] > 0.0     # sizes actually vary
    # min_cells floor respected per sample
    per_sample = env.mask[:, 0].flatten(1).sum(dim=1)
    assert (per_sample >= 2).all()


def test_stopped_samples_are_frozen():
    env = _env(B=4, horizon=6, allow_stop=True, min_cells=1)
    env.reset()
    # step 1: everyone picks cell 0
    env.step(torch.zeros(4, dtype=torch.long))
    # step 2: samples 0,1 stop; 2,3 pick cell 1
    a = torch.tensor([env.stop_action, env.stop_action, 1, 1])
    env.step(a)
    # step 3: stopped samples try to pick cell 2 — must be ignored
    env.step(torch.full((4,), 2, dtype=torch.long))
    counts = env.mask[:, 0].flatten(1).sum(dim=1)
    assert counts[0] == 1 and counts[1] == 1
    assert counts[2] == 3 and counts[3] == 3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml


TICK = "✓"
CROSS = "✗"


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / max(1, len(xs))


def fmt(v):
    if isinstance(v, float):
        return f"{v:.6f}"
    if v is None:
        return ""
    return str(v)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return

    cols = []
    for r in rows:
        for k in r.keys():
            if k not in cols:
                cols.append(k)

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k)) for k in cols})


def cfg_get(cfg, dotted: str, default=None):
    return cfg.get_dotted(dotted, default) if hasattr(cfg, "get_dotted") else default


def policy_ckpt(manifest: Dict[str, Any], key: str) -> str:
    p = manifest["policies"][key]
    ckpt = str(p.get("ckpt", "auto"))
    if ckpt != "auto":
        return ckpt
    return str(Path(manifest["root_dir"]) / p["run_name"] / "ckpt" / "latest.pth")


def full_score_weights():
    from src.rl.reward import get_reward_weights
    return get_reward_weights("full_rift")


def sigmoid_mean(logits) -> float:
    import torch
    if not torch.is_tensor(logits):
        return float(logits)
    return float(torch.sigmoid(logits.float()).mean().item())


def gap_value(res) -> float:
    return float(getattr(res, "value", res))


def gap_mode(res) -> str:
    m = getattr(res, "mode", "proxy")
    return str(getattr(m, "value", m))


class CausalSelectExplainer:
    def __init__(
        self,
        base_explainer,
        *,
        channel: str,
        grid: int,
        horizon: int,
        candidate_pool: int,
        intervention_mode: str,
        topk_frac: float,
    ):
        self.base_explainer = base_explainer
        self.channel = channel
        self.grid = int(grid)
        self.horizon = int(horizon)
        self.candidate_pool = int(candidate_pool)
        self.intervention_mode = intervention_mode
        self.topk_frac = float(topk_frac)
        self.name = f"causal_select_{channel}_{getattr(base_explainer, 'name', 'base')}"

    def explain(self, image, adapter, **kw):
        import torch
        import torch.nn.functional as F

        B, _, H, W = image.shape
        if B != 1:
            raise RuntimeError("CausalSelectExplainer expects B=1 during audit.")

        base = self.base_explainer.explain(image, adapter, **kw)
        if base.shape[-2:] != (H, W):
            base = F.interpolate(base, size=(H, W), mode="bilinear", align_corners=False)

        with torch.no_grad():
            cell_scores = F.adaptive_avg_pool2d(base.float(), (self.grid, self.grid)).flatten()
            k = min(self.candidate_pool, cell_scores.numel())
            candidates = [int(i) for i in torch.topk(cell_scores, k=k).indices.detach().cpu().tolist()]

        selected = []
        for _ in range(self.horizon):
            best_idx = None
            best_score = -1e9

            for idx in candidates:
                if idx in selected:
                    continue
                score = self._score_cells(image, adapter, selected + [idx], **kw)
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is None:
                break
            selected.append(best_idx)

        grid_mask = torch.zeros(1, 1, self.grid, self.grid, device=image.device)
        for idx in selected:
            grid_mask[:, :, idx // self.grid, idx % self.grid] = 1.0

        return F.interpolate(grid_mask, size=(H, W), mode="nearest")

    def _score_cells(self, image, adapter, cells, **kw) -> float:
        import torch
        import torch.nn.functional as F
        from src.interventions.interventions import apply_necessity, apply_sufficiency
        from src.faithfulness.faithfulness_score import necessity, sufficiency, harmonic

        grid_mask = torch.zeros(1, 1, self.grid, self.grid, device=image.device)
        for idx in cells:
            grid_mask[:, :, idx // self.grid, idx % self.grid] = 1.0

        mask = F.interpolate(grid_mask, size=image.shape[-2:], mode="nearest")
        donor = kw.get("donor")

        with torch.no_grad():
            nec_img = apply_necessity(image, mask, self.intervention_mode, self.topk_frac)
            suf_img = apply_sufficiency(image, mask, self.intervention_mode, self.topk_frac)

            if self.channel == "delta":
                e0 = gap_value(adapter.identity_gap(image, donor=donor))
                en = gap_value(adapter.identity_gap(nec_img, donor=donor))
                es = gap_value(adapter.identity_gap(suf_img, donor=donor))
            else:
                e0 = sigmoid_mean(adapter.predict_logits(image))
                en = sigmoid_mean(adapter.predict_logits(nec_img))
                es = sigmoid_mean(adapter.predict_logits(suf_img))

        n = necessity(e0, en)
        s = sufficiency(e0, es)
        return harmonic(n, s) - 0.01 * len(cells) / float(self.grid * self.grid)


class PolicyExplainer:
    def __init__(
        self,
        ckpt_path: str,
        *,
        grid: int,
        hidden: int,
        feat_dim: int,
        horizon: int,
        reward_preset: str,
        intervention_mode: str,
        topk_frac: float,
        device: str,
    ):
        self.name = f"rift_policy_h{horizon}_{reward_preset}"
        self.ckpt_path = ckpt_path
        self.grid = int(grid)
        self.hidden = int(hidden)
        self.feat_dim = int(feat_dim)
        self.horizon = int(horizon)
        self.reward_preset = reward_preset
        self.intervention_mode = intervention_mode
        self.topk_frac = float(topk_frac)
        self.device = device
        self.policy = None

    def _load_policy(self):
        if self.policy is not None:
            return self.policy

        import torch
        from src.rl.policy import GridPolicy

        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"Missing policy checkpoint: {self.ckpt_path}")

        raw = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        sd = raw.get("policy", raw)
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

        policy = GridPolicy(
            feat_dim=self.feat_dim,
            grid=self.grid,
            n_actions=self.grid * self.grid + 1,
            hidden=self.hidden,
        ).to(self.device)

        policy.load_state_dict(sd, strict=True)
        policy.eval()
        self.policy = policy
        return policy

    def explain(self, image, adapter, **kw):
        import torch
        from src.rl.batched_rift_env import BatchedRIFTEnv
        from src.rl.reward import get_reward_weights

        policy = self._load_policy()

        donor = kw.get("donor")
        if donor is not None:
            donor = donor.to(self.device)

        env = BatchedRIFTEnv(
            image.to(self.device),
            adapter,
            grid=self.grid,
            horizon=self.horizon,
            intervention_mode=self.intervention_mode,
            topk_frac=self.topk_frac,
            reward_fn=get_reward_weights(self.reward_preset),
            donor=donor,
            cache_features=True,
        )

        state = env.reset()

        with torch.no_grad():
            for _ in range(self.horizon):
                logits, _ = policy(state)
                action = logits.argmax(dim=-1)
                state, _, done, _ = env.step(action)
                if bool(done.all().item()):
                    break

        return env.current_mask().detach()


def make_explainer(row: Dict[str, Any], manifest: Dict[str, Any], device: str):
    from src.explainers.random_explainer import RandomExplainer
    from src.explainers.gradcam_explainer import GradCAMExplainer
    from src.explainers.cift_gap_explainer import CIFTGapExplainer

    kind = row["kind"]
    ev = manifest["eval"]
    pd = manifest["policy_defaults"]

    if kind == "random":
        return RandomExplainer()

    if kind == "gradcam":
        return GradCAMExplainer(target_class=1)

    if kind == "cift_delta":
        return CIFTGapExplainer()

    if kind == "causal_select":
        base_name = row.get("base")
        if base_name == "gradcam":
            base = GradCAMExplainer(target_class=1)
        elif base_name == "cift_delta":
            base = CIFTGapExplainer()
        else:
            raise RuntimeError(f"unknown causal_select base={base_name}")

        return CausalSelectExplainer(
            base,
            channel=row.get("channel", "delta"),
            grid=int(pd.get("grid", 8)),
            horizon=int(manifest["policies"].get("full_h4", {}).get("horizon", 4)),
            candidate_pool=int(ev.get("candidate_pool", 16)),
            intervention_mode=str(ev.get("intervention_mode", "blur")),
            topk_frac=float(ev.get("topk_frac", 0.12)),
        )

    if kind == "policy":
        key = row["policy"]
        p = manifest["policies"][key]
        return PolicyExplainer(
            policy_ckpt(manifest, key),
            grid=int(pd.get("grid", 8)),
            hidden=int(pd.get("hidden", 256)),
            feat_dim=int(pd.get("feat_dim", 1024)),
            horizon=int(p["horizon"]),
            reward_preset=str(p["reward_preset"]),
            intervention_mode=str(ev.get("intervention_mode", "blur")),
            topk_frac=float(ev.get("topk_frac", 0.12)),
            device=device,
        )

    raise RuntimeError(f"unknown explainer kind={kind}")


def audit_explainer(cfg, adapter, explainer, *, device: str, max_items: int, intervention_mode: str, topk_frac: float):
    import torch
    from src.audit.ablation_runner import iter_audit_samples
    from src.interventions.interventions import apply_necessity, apply_sufficiency, mask_area
    from src.faithfulness.faithfulness_score import compute_rift_score

    rows = []
    weights = full_score_weights()

    for img, donor, gt in iter_audit_samples(cfg, device=device, n=max_items):
        mask = explainer.explain(img, adapter, donor=donor)

        with torch.no_grad():
            g0 = adapter.identity_gap(img, donor=donor)
            l0 = sigmoid_mean(adapter.predict_logits(img))

            nec_img = apply_necessity(img, mask, intervention_mode, topk_frac)
            suf_img = apply_sufficiency(img, mask, intervention_mode, topk_frac)

            gn = adapter.identity_gap(nec_img, donor=donor)
            gs = adapter.identity_gap(suf_img, donor=donor)

            ln = sigmoid_mean(adapter.predict_logits(nec_img))
            ls = sigmoid_mean(adapter.predict_logits(suf_img))

        comp = compute_rift_score(
            e0_delta=gap_value(g0),
            e_nec_delta=gap_value(gn),
            e_suf_delta=gap_value(gs),
            e0_logit=l0,
            e_nec_logit=ln,
            e_suf_logit=ls,
            mask_area=mask_area(mask, topk_frac),
            identity_gap_mode=gap_mode(g0),
            weights=weights,
        )

        rows.append(comp.to_dict())

    def avg(key):
        return mean([r.get(key) for r in rows])

    return {
        "n": len(rows),
        "Nec Δ ↑": avg("necessity_delta"),
        "Suf Δ ↑": avg("sufficiency_delta"),
        "Faith Δ ↑": avg("faithfulness_delta"),
        "Faith logit ↑": avg("faithfulness_logit"),
        "Mask area ↓": avg("mask_area"),
        "RIFT score ↑": avg("rift_score"),
    }


def tick(v):
    return TICK if bool(v) else CROSS


def run_one_table(table_name: str, table_spec: Dict[str, Any], manifest: Dict[str, Any], cfg, adapter, device: str):
    rows_out = []
    ev = manifest["eval"]

    for row in table_spec["rows"]:
        print(f"[eval] {table_name} id={row['id']} variant={row['variant']}", flush=True)

        out = {
            "ID": row["id"],
            "Variant": row["variant"],
        }

        if "mask_source" in row:
            out["Mask source"] = row["mask_source"]

        if "delta_g" in row:
            out["ΔG"] = tick(row.get("delta_g"))
            out["NS"] = tick(row.get("ns"))
            out["RP"] = tick(row.get("rp"))

        if "necessity" in row:
            out["Necessity"] = tick(row.get("necessity"))
            out["Sufficiency"] = tick(row.get("sufficiency"))
            out["Sparsity"] = tick(row.get("sparsity"))

        if "horizon" in row:
            out["Horizon"] = row.get("horizon")

        try:
            ex = make_explainer(row, manifest, device)
            metrics = audit_explainer(
                cfg,
                adapter,
                ex,
                device=device,
                max_items=int(ev.get("max_items", 512)),
                intervention_mode=str(ev.get("intervention_mode", "blur")),
                topk_frac=float(ev.get("topk_frac", 0.12)),
            )
            out.update(metrics)
            out["status"] = "ok"
            out["error"] = ""
        except Exception as e:
            out["n"] = 0
            out["status"] = "failed"
            out["error"] = f"{type(e).__name__}: {e}"
            print(f"[WARN] {table_name} row failed: {out['error']}", flush=True)

        rows_out.append(out)

    return rows_out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation-config", default="ablations/configs/table123_rift.yaml")
    ap.add_argument("--tables", default="table1_component,table2_objective,table3_horizon")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-items", type=int, default=None)
    args = ap.parse_args()

    manifest = load_yaml(args.ablation_config)

    if args.max_items is not None:
        manifest["eval"]["max_items"] = int(args.max_items)

    device = args.device or "cuda"
    if device == "cuda":
        device = "cuda:0"

    from src.utils.config import load_config, merge_overrides
    from src.adapters.cift_adapter import CIFTAdapter

    cfg = load_config(manifest["base_config"])
    cfg = merge_overrides(
        cfg,
        {
            "device": device,
            "detector.cift_root": manifest["cift"]["root"],
            "detector.cift_ckpt": manifest["cift"]["ckpt"],
            "detector.strict_identity_gap": True,
            "dataset.split_csv": manifest["data"]["eval_csv"],
            "dataset.max_items": int(manifest["eval"].get("max_items", 512)),
            "intervention.mode": manifest["eval"].get("intervention_mode", "blur"),
            "intervention.topk_frac": float(manifest["eval"].get("topk_frac", 0.12)),
        },
    )

    adapter = CIFTAdapter(
        ckpt_path=cfg_get(cfg, "detector.cift_ckpt"),
        device=device,
        backbone=cfg_get(cfg, "detector.backbone", "convnextv2_base"),
        strict_identity_gap=True,
        cift_root=cfg_get(cfg, "detector.cift_root"),
        config_path=cfg_get(cfg, "detector.cift_config", "configs/diffusionfake_mixed.yaml"),
    ).load_detector()

    output_dir = manifest["eval"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    wanted = [x.strip() for x in args.tables.split(",") if x.strip()]
    combined = []

    for t in wanted:
        spec = manifest["tables"][t]
        rows = run_one_table(t, spec, manifest, cfg, adapter, device)
        out_path = os.path.join(output_dir, spec["filename"])
        write_csv(out_path, rows)
        print(f"[wrote] {out_path}", flush=True)

        for r in rows:
            rr = {"table": t}
            rr.update(r)
            combined.append(rr)

    combined_path = os.path.join(output_dir, "combined_tables_1_2_3.csv")
    write_csv(combined_path, combined)
    print(f"[done] combined CSV: {combined_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

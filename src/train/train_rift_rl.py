# Path: src/train/train_rift_rl.py
# Status: MODIFIED
"""RIFT-RL training loop.

Freezes detector, rolls out policy in RIFTEnv, and updates with REINFORCE/PPO.

Main fix:
  donor/reference tensors from the CSV are carried into RIFTEnv, so strict
  donor-grounded identity-gap mode can run instead of crashing.
"""
from __future__ import annotations

from ..utils.logging import get_logger
from ..utils.seed import seed_everything, get_rng_states
from ..utils.checkpoint_manager import CheckpointManager
from ..utils.wandb_logger import WandbLogger

from ..rl.rift_env import RIFTEnv
from ..rl.policy import GridPolicy
from ..rl.reinforce import Reinforce
from ..rl.ppo import PPO
from ..rl.rollout_buffer import RolloutBuffer
from ..rl.reward import get_reward_weights

log = get_logger("train")


def _device_item(item, key, device):
    x = item.get(key)
    if x is None:
        return None
    return x.unsqueeze(0).to(device)


def train(cfg, adapter, dataloaders):
    import torch

    seed_everything(cfg.get("seed", 42))

    train_dl, val_dl, id_mode = dataloaders
    device = cfg.get("device", "cuda")

    log.info(f"identity_gap_mode={id_mode}")

    if id_mode == "proxy" and getattr(adapter, "strict_identity_gap", False):
        raise RuntimeError(
            "Training data has no donor/reference tensors but detector.strict_identity_gap=True. "
            "Add donor_path/source_ref_path to the CSV, or set detector.strict_identity_gap=false "
            "for proxy/logit-only debugging."
        )

    grid = cfg.get("grid", 8)
    horizon = cfg.get("horizon", 4)

    policy = GridPolicy(
        grid=grid,
        n_actions=grid * grid + 1,
        hidden=cfg.get("hidden", 256),
    ).to(device)

    algo_name = cfg.get("algo", "reinforce")

    if algo_name == "ppo":
        algo = PPO(
            policy,
            lr=cfg.get("lr", 3e-4),
            clip=cfg.get("clip", 0.2),
            epochs=cfg.get("ppo_epochs", 4),
            entropy_coef=cfg.get("entropy_coef", 0.01),
            value_coef=cfg.get("value_coef", 0.5),
            lagrangian=cfg.get("lagrangian", False),
            constraint_budget=cfg.get("constraint_budget", 0.0),
        )
    else:
        algo = Reinforce(
            policy,
            lr=cfg.get("lr", 3e-4),
            entropy_coef=cfg.get("entropy_coef", 0.01),
        )

    weights = get_reward_weights(cfg.get("reward_preset", "full_rift"))

    ckpt = CheckpointManager(
        cfg.get("out_dir", "outputs/rift_rl"),
        monitor=cfg.get("monitor", "rift_score"),
        mode=cfg.get("mode", "max"),
        top_k=cfg.get("top_k", 3),
        interval=cfg.get("interval", 10),
    )

    wb = WandbLogger(
        cfg.get("wandb_project"),
        cfg.get("exp_name"),
        cfg,
        enabled=cfg.get("wandb", False),
    )

    resume = ckpt.resume(cfg.get("resume", "auto"))

    start_epoch = 0
    gstep = 0

    if resume:
        st = ckpt.load(resume)
        policy.load_state_dict(st["policy"])
        start_epoch = st["epoch"] + 1
        gstep = int(st.get("global_step", 0))
        log.info(f"resumed from {resume} @ epoch {start_epoch}")

    epochs = cfg.get("epochs", 50)

    for epoch in range(start_epoch, epochs):
        policy.train()

        for batch in train_dl:
            for item in batch:
                img = _device_item(item, "image", device)
                donor = _device_item(item, "donor", device)
                s = item.get("sample")

                env = RIFTEnv(
                    img,
                    adapter,
                    grid=grid,
                    horizon=horizon,
                    intervention_mode=cfg.get("intervention_mode", "blur"),
                    topk_frac=cfg.get("topk_frac", 0.12),
                    reward_fn=weights,
                    donor=donor,
                    source_id=getattr(s, "source_id", None) if s else None,
                    target_id=getattr(s, "target_id", None) if s else None,
                )

                buf = RolloutBuffer()
                state = env.reset()
                done = False

                while not done:
                    logits, value = policy(state)
                    probs = torch.softmax(logits, -1)

                    a = int(torch.multinomial(probs, 1)[0, 0].item())
                    logp = float(torch.log_softmax(logits, -1)[0, a].detach().item())

                    nstate, r, done, info = env.step(a)

                    buf.add(
                        state,
                        a,
                        logp,
                        r,
                        float(value.detach().item()),
                        done,
                    )

                    state = nstate

                logs = algo.update(buf)
                gstep += 1

                wb.log(
                    {f"train/{k}": v for k, v in logs.items()}
                    | {
                        "train/reward_total": sum(buf.rewards),
                        "epoch": epoch,
                    },
                    step=gstep,
                )

        metrics = validate(cfg, adapter, policy, val_dl, weights, grid, horizon)
        metrics["epoch_string"] = f"Epoch {epoch + 1}/{epochs}"

        wb.log(
            {f"val/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))},
            step=gstep,
        )

        paths = ckpt.save(
            {
                "policy": policy.state_dict(),
                "epoch": epoch,
                "global_step": gstep,
                "config": dict(cfg),
                "rng": get_rng_states(),
            },
            epoch,
            {cfg.get("monitor", "rift_score"): metrics.get("rift_score", 0.0)},
        )

        log.info(
            f"{metrics['epoch_string']} "
            f"rift_score={metrics.get('rift_score'):.4f} "
            f"ckpt={paths.get('best')}"
        )

    wb.finish()
    return policy


def validate(cfg, adapter, policy, val_dl, weights, grid, horizon):
    import torch

    from ..audit.audit_runner import audit_one, aggregate
    from ..explainers.rift_policy_explainer import RIFTPolicyExplainer

    rows = []
    device = cfg.get("device", "cuda")

    expl = RIFTPolicyExplainer(
        policy,
        lambda img, ad, **kw: RIFTEnv(
            img,
            ad,
            grid=grid,
            horizon=horizon,
            intervention_mode=cfg.get("intervention_mode", "blur"),
            topk_frac=cfg.get("topk_frac", 0.12),
            reward_fn=weights,
            donor=kw.get("donor"),
            source_id=kw.get("source_id"),
            target_id=kw.get("target_id"),
        ),
        horizon,
    )

    policy.eval()

    with torch.no_grad():
        for batch in val_dl:
            for item in batch:
                img = _device_item(item, "image", device)
                donor = _device_item(item, "donor", device)
                s = item.get("sample")

                row, _, _, _ = audit_one(
                    img,
                    adapter,
                    expl,
                    intervention_mode=cfg.get("intervention_mode", "blur"),
                    topk_frac=cfg.get("topk_frac", 0.12),
                    donor=donor,
                    source_id=getattr(s, "source_id", None) if s else None,
                    target_id=getattr(s, "target_id", None) if s else None,
                    reward_weights=weights,
                )

                rows.append(row)

    agg = aggregate(rows)

    return {
        "rift_score": agg.get("rift_score", 0.0),
        "faithfulness_ns_delta": agg.get("faithfulness_delta", 0.0),
        "faithfulness_ns_logit": agg.get("faithfulness_logit", 0.0),
        "necessity_delta_drop": agg.get("necessity_delta", 0.0),
        "sufficiency_delta_retained": agg.get("sufficiency_delta", 0.0),
        "mask_area": agg.get("mask_area", 0.0),
    }
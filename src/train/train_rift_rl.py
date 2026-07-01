# Path: src/train/train_rift_rl.py
"""RIFT-RL training loop with optional DDP.

DDP mode:
  torchrun launches N processes.
  Each process owns one GPU and a shard of the dataset.
  Policy gradients are synchronized through DistributedDataParallel.
  Rank 0 saves checkpoints.
"""

from __future__ import annotations

import os

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


def _is_ddp():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _rank():
    return int(os.environ.get("RANK", "0"))


def _world():
    return int(os.environ.get("WORLD_SIZE", "1"))


def _local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


def _is_main():
    return _rank() == 0


def _ddp_setup():
    import torch
    import torch.distributed as dist

    if not _is_ddp():
        return False

    torch.cuda.set_device(_local_rank())

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    return True


def _ddp_cleanup():
    if not _is_ddp():
        return

    try:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    except Exception:
        pass


def _ddp_barrier():
    if not _is_ddp():
        return

    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()


def _policy_state(policy):
    return policy.module.state_dict() if hasattr(policy, "module") else policy.state_dict()


def _base_policy(policy):
    return policy.module if hasattr(policy, "module") else policy


def _device_item(item, key, device):
    x = item.get(key)

    if x is None:
        return None

    return x.unsqueeze(0).to(device, non_blocking=True)


def _set_sampler_epoch(dl, epoch):
    sampler = getattr(dl, "sampler", None)

    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def _reduce_metrics(metrics):
    """Weighted average validation metrics across ranks."""
    if not _is_ddp():
        return metrics

    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        return metrics

    keys = [
        "rift_score",
        "faithfulness_ns_delta",
        "faithfulness_ns_logit",
        "necessity_delta_drop",
        "sufficiency_delta_retained",
        "mask_area",
    ]

    n = float(metrics.get("n", 0.0))

    vals = []

    for k in keys:
        vals.append(float(metrics.get(k, 0.0)) * n)

    vals.append(n)

    t = torch.tensor(vals, device=f"cuda:{_local_rank()}", dtype=torch.float64)

    dist.all_reduce(t, op=dist.ReduceOp.SUM)

    total_n = max(float(t[-1].item()), 1.0)

    out = dict(metrics)

    for i, k in enumerate(keys):
        out[k] = float(t[i].item() / total_n)

    out["n"] = int(total_n)

    return out


def train(cfg, adapter, dataloaders):
    import torch

    ddp = _ddp_setup()

    # Important:
    # all ranks use same seed base, but offset by rank for RL trajectory diversity.
    seed_everything(int(cfg.get("seed", 42)) + _rank())

    train_dl, val_dl, id_mode = dataloaders

    if ddp:
        device = f"cuda:{_local_rank()}"
    else:
        device = cfg.get("device", "cuda")

    log.info(f"identity_gap_mode={id_mode}")

    if _is_main():
        if ddp:
            print(f"[ddp train] world={_world()} local_rank={_local_rank()} device={device}")
        print(
            f"[train] device={device} algo={cfg.get('algo', 'ppo')} "
            f"epochs={cfg.get('epochs', 50)} horizon={cfg.get('horizon', 4)} "
            f"grid={cfg.get('grid', 8)}"
        )

    if id_mode == "proxy" and getattr(adapter, "strict_identity_gap", False):
        raise RuntimeError(
            "Training data has no donor/reference tensors but detector.strict_identity_gap=True. "
            "Add donor_path/source_ref_path to the CSV, or set detector.strict_identity_gap=false."
        )

    grid = cfg.get("grid", 8)
    horizon = cfg.get("horizon", 4)

    policy = GridPolicy(
        grid=grid,
        n_actions=grid * grid + 1,
        hidden=cfg.get("hidden", 256),
    ).to(device)

    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP

        policy = DDP(
            policy,
            device_ids=[_local_rank()],
            output_device=_local_rank(),
            find_unused_parameters=False,
        )

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

    ckpt = None
    wb = None

    if _is_main():
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

    resume = None

    if _is_main() and ckpt is not None:
        resume = ckpt.resume(cfg.get("resume", "none"))

    # Broadcast resume state by just loading on rank0 for now.
    # For clean distributed full runs, use resume=none.
    if resume and _is_main():
        st = ckpt.load(resume)
        _base_policy(policy).load_state_dict(st["policy"])
        start_epoch = st["epoch"] + 1
        gstep = int(st.get("global_step", 0))
        log.info(f"resumed from {resume} @ epoch {start_epoch}")
    else:
        start_epoch = 0
        gstep = 0

    # Sync initial model params from rank0 to all ranks.
    _ddp_barrier()

    epochs = int(cfg.get("epochs", 50))

    for epoch in range(start_epoch, epochs):
        _set_sampler_epoch(train_dl, epoch)

        policy.train()

        if _is_main():
            try:
                from tqdm import tqdm

                pbar = tqdm(
                    total=len(train_dl),
                    desc=f"train epoch {epoch + 1}/{epochs}",
                    dynamic_ncols=True,
                    leave=True,
                )
            except Exception:
                pbar = None
        else:
            pbar = None

        local_items = 0
        local_reward_sum = 0.0

        for batch_idx, batch in enumerate(train_dl):
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
                local_items += 1
                local_reward_sum += float(sum(buf.rewards))

                if _is_main() and wb is not None:
                    wb.log(
                        {f"train/{k}": v for k, v in logs.items()}
                        | {
                            "train/reward_total": sum(buf.rewards),
                            "epoch": epoch,
                            "rank0/local_items": local_items,
                        },
                        step=gstep,
                    )

            if pbar is not None:
                pbar.set_postfix(
                    {
                        "batch": batch_idx + 1,
                        "rank0_items": local_items,
                        "reward": f"{local_reward_sum / max(local_items, 1):.3f}",
                    }
                )
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        _ddp_barrier()

        metrics = validate(cfg, adapter, _base_policy(policy), val_dl, weights, grid, horizon)
        metrics = _reduce_metrics(metrics)
        metrics["epoch_string"] = f"Epoch {epoch + 1}/{epochs}"

        if _is_main():
            if wb is not None:
                wb.log(
                    {f"val/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))},
                    step=gstep,
                )

            paths = ckpt.save(
                {
                    "policy": _policy_state(policy),
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
                f"faith_delta={metrics.get('faithfulness_ns_delta'):.4f} "
                f"n={metrics.get('n', 0)} "
                f"ckpt={paths.get('best')}"
            )

            print(
                f"[val] {metrics['epoch_string']} "
                f"rift_score={metrics.get('rift_score'):.4f} "
                f"faith_delta={metrics.get('faithfulness_ns_delta'):.4f} "
                f"n={metrics.get('n', 0)} "
                f"ckpt={paths.get('best')}"
            )

        _ddp_barrier()

    if _is_main() and wb is not None:
        wb.finish()

    _ddp_cleanup()

    return _base_policy(policy)


def validate(cfg, adapter, policy, val_dl, weights, grid, horizon):
    import torch

    from ..audit.audit_runner import audit_one, aggregate
    from ..explainers.rift_policy_explainer import RIFTPolicyExplainer

    rows = []

    if _is_ddp():
        device = f"cuda:{_local_rank()}"
    else:
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
        "n": agg.get("n", 0),
        "rift_score": agg.get("rift_score", 0.0),
        "faithfulness_ns_delta": agg.get("faithfulness_delta", 0.0),
        "faithfulness_ns_logit": agg.get("faithfulness_logit", 0.0),
        "necessity_delta_drop": agg.get("necessity_delta", 0.0),
        "sufficiency_delta_retained": agg.get("sufficiency_delta", 0.0),
        "mask_area": agg.get("mask_area", 0.0),
    }

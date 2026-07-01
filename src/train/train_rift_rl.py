# Path: src/train/train_rift_rl.py
"""RIFT-RL training loop with robust DDP resume.

This version fixes:
  - DDP resume across all ranks.
  - Stable resume from latest.pth.
  - Optimizer state restore.
  - PPO lambda restore.
  - Validation frequency control.
  - Checkpoint saving with optimizer/algo state.
"""

from __future__ import annotations

import os

from ..utils.checkpoint_manager import CheckpointManager
from ..utils.logging import get_logger
from ..utils.seed import get_rng_states, seed_everything
from ..utils.wandb_logger import WandbLogger

from ..rl.policy import GridPolicy
from ..rl.ppo import PPO
from ..rl.reinforce import Reinforce
from ..rl.reward import get_reward_weights
from ..rl.rift_env import RIFTEnv
from ..rl.rollout_buffer import RolloutBuffer

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
            dist.barrier(device_ids=[_local_rank()])
            dist.destroy_process_group()
    except Exception:
        pass


def _ddp_barrier():
    if not _is_ddp():
        return

    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier(device_ids=[_local_rank()])


def _policy_state(policy):
    return policy.module.state_dict() if hasattr(policy, "module") else policy.state_dict()


def _base_policy(policy):
    return policy.module if hasattr(policy, "module") else policy


def _load_checkpoint_cpu(path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _move_optimizer_state_to_device(algo, device):
    if not hasattr(algo, "opt"):
        return

    import torch

    for state in algo.opt.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device)


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
    """Weighted average validation metrics across DDP ranks."""
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
    vals = [float(metrics.get(k, 0.0)) * n for k in keys]
    vals.append(n)

    t = torch.tensor(vals, device=f"cuda:{_local_rank()}", dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)

    total_n = max(float(t[-1].item()), 1.0)
    out = dict(metrics)

    for i, k in enumerate(keys):
        out[k] = float(t[i].item() / total_n)

    out["n"] = int(total_n)

    return out


def _make_policy(cfg, device, grid):
    return GridPolicy(
        grid=grid,
        n_actions=grid * grid + 1,
        hidden=cfg.get("hidden", 256),
        feat_dim=cfg.get("feat_dim", 1024),
    ).to(device)


def _make_algo(cfg, policy):
    algo_name = cfg.get("algo", "ppo")

    if algo_name == "ppo":
        return PPO(
            policy,
            lr=cfg.get("lr", 3e-4),
            clip=cfg.get("clip", 0.2),
            epochs=cfg.get("ppo_epochs", 4),
            entropy_coef=cfg.get("entropy_coef", 0.01),
            value_coef=cfg.get("value_coef", 0.5),
            lagrangian=cfg.get("lagrangian", False),
            constraint_budget=cfg.get("constraint_budget", 0.0),
        )

    return Reinforce(
        policy,
        lr=cfg.get("lr", 3e-4),
        entropy_coef=cfg.get("entropy_coef", 0.01),
    )


def train(cfg, adapter, dataloaders):
    import torch

    ddp = _ddp_setup()

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

    grid = int(cfg.get("grid", 8))
    horizon = int(cfg.get("horizon", 4))

    policy = _make_policy(cfg, device, grid)

    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP

        policy = DDP(
            policy,
            device_ids=[_local_rank()],
            output_device=_local_rank(),
            find_unused_parameters=False,
        )

    algo = _make_algo(cfg, policy)
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

    # ---------------------------------------------------------------------
    # Robust resume:
    #   - rank 0 resolves latest.pth / explicit path
    #   - path is broadcast to all ranks
    #   - every rank loads same policy and optimizer state
    # ---------------------------------------------------------------------
    resume_path = None

    if _is_main() and ckpt is not None:
        resume_path = ckpt.resume(cfg.get("resume", "auto"))

    if _is_ddp():
        import torch.distributed as dist

        obj = [resume_path]
        dist.broadcast_object_list(obj, src=0)
        resume_path = obj[0]

    if resume_path:
        st = _load_checkpoint_cpu(resume_path)

        _base_policy(policy).load_state_dict(st["policy"])

        if st.get("optimizer") is not None and hasattr(algo, "opt"):
            try:
                algo.opt.load_state_dict(st["optimizer"])
                _move_optimizer_state_to_device(algo, device)
            except Exception as e:
                if _is_main():
                    print(f"[resume][WARN] optimizer state not restored: {e}")

        algo_state = st.get("algo_state") or {}
        if hasattr(algo, "lmbda") and "lambda" in algo_state:
            algo.lmbda = float(algo_state["lambda"])

        start_epoch = int(st.get("epoch", -1)) + 1
        gstep = int(st.get("global_step", 0))

        if _is_main():
            print(f"[resume] loaded {resume_path} start_epoch={start_epoch} global_step={gstep}")
            log.info(f"resumed from {resume_path} @ epoch {start_epoch}")
    else:
        start_epoch = 0
        gstep = 0

        if _is_main():
            print("[resume] no checkpoint found, starting fresh")

    if _is_ddp():
        import torch.distributed as dist

        obj = [start_epoch, gstep]
        dist.broadcast_object_list(obj, src=0)
        start_epoch = int(obj[0])
        gstep = int(obj[1])

        # Safety sync.
        for param in _base_policy(policy).parameters():
            dist.broadcast(param.data, src=0)

    _ddp_barrier()

    epochs = int(cfg.get("epochs", 50))
    val_every = int(cfg.get("val_every", 1) or 1)

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
            # Current stable path keeps per-item RIFTEnv behavior.
            # DDP still shards data across GPUs. For max speed, use batched env patch separately.
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

                    nstate, r, done, _info = env.step(a)

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

        do_val = ((epoch + 1) % val_every == 0) or (epoch + 1 == epochs)

        if do_val:
            metrics = validate(cfg, adapter, _base_policy(policy), val_dl, weights, grid, horizon)
            metrics = _reduce_metrics(metrics)
        else:
            metrics = {
                "n": 0,
                "rift_score": 0.0,
                "faithfulness_ns_delta": 0.0,
                "faithfulness_ns_logit": 0.0,
                "necessity_delta_drop": 0.0,
                "sufficiency_delta_retained": 0.0,
                "mask_area": 0.0,
            }

        metrics["epoch_string"] = f"Epoch {epoch + 1}/{epochs}"

        if _is_main():
            if wb is not None and do_val:
                wb.log(
                    {f"val/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))},
                    step=gstep,
                )

            paths = ckpt.save(
                {
                    "policy": _policy_state(policy),
                    "optimizer": algo.opt.state_dict() if hasattr(algo, "opt") else None,
                    "algo_state": {"lambda": getattr(algo, "lmbda", 0.0)},
                    "epoch": epoch,
                    "global_step": gstep,
                    "config": dict(cfg),
                    "rng": get_rng_states(),
                },
                epoch,
                {cfg.get("monitor", "rift_score"): metrics.get("rift_score", 0.0)},
            )

            if do_val:
                msg = (
                    f"[val] {metrics['epoch_string']} "
                    f"rift_score={metrics.get('rift_score'):.4f} "
                    f"faith_delta={metrics.get('faithfulness_ns_delta'):.4f} "
                    f"n={metrics.get('n', 0)} "
                    f"ckpt={paths.get('best')}"
                )
            else:
                msg = (
                    f"[ckpt] {metrics['epoch_string']} "
                    f"validation skipped val_every={val_every} "
                    f"latest={paths.get('latest')}"
                )

            log.info(msg)
            print(msg)

        _ddp_barrier()

    if _is_main() and wb is not None:
        wb.finish()

    _ddp_cleanup()

    return _base_policy(policy)


def validate(cfg, adapter, policy, val_dl, weights, grid, horizon):
    import torch

    from ..audit.audit_runner import aggregate, audit_one
    from ..explainers.rift_policy_explainer import RIFTPolicyExplainer

    rows = []

    if _is_ddp():
        device = f"cuda:{_local_rank()}"
    else:
        device = cfg.get("device", "cuda")

    max_batches = int(cfg.get("val_max_batches", 0) or 0)

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
        for bidx, batch in enumerate(val_dl):
            if max_batches > 0 and bidx >= max_batches:
                break

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

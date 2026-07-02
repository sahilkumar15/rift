# Path: src/train/train_rift_rl.py
"""Fast RIFT-RL training loop.

Main speed fix:
  Old path: for each batch, loop over every item and create one RIFTEnv per image.
  New path: create one BatchedRIFTEnv per mini-batch and update PPO once per batch.
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
from ..rl.batched_rift_env import BatchedRIFTEnv
from ..rl.batched_rollout_buffer import BatchedRolloutBuffer

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


def _set_sampler_epoch(dl, epoch):
    sampler = getattr(dl, "sampler", None)

    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def _as_bool(v, default=False):
    if v is None:
        return default

    if isinstance(v, bool):
        return v

    s = str(v).strip().lower()

    if s in ("1", "true", "yes", "on"):
        return True

    if s in ("0", "false", "no", "off", "none", "null", "disabled", ""):
        return False

    return default


def _metric_value(metrics, monitor):
    key = str(monitor or "rift_score")

    if key in metrics:
        return metrics[key]

    if "/" in key:
        short = key.split("/", 1)[-1]
        if short in metrics:
            return metrics[short]

    return None


def _is_better(value, best, mode, min_delta=0.0):
    if value is None:
        return False

    if best is None:
        return True

    value = float(value)
    best = float(best)
    min_delta = float(min_delta or 0.0)

    if str(mode).lower() == "min":
        return value < best - min_delta

    return value > best + min_delta


def _stack_tensor(xs, device):
    import torch

    return torch.stack([x for x in xs], dim=0).to(device, non_blocking=True)


def _batch_to_device(batch, device):
    """Convert DataLoader list[dict] into batched BCHW tensors."""
    images = _stack_tensor([item["image"] for item in batch], device)

    donor_values = [item.get("donor") for item in batch]
    has_all_donors = all(d is not None for d in donor_values)
    donor = _stack_tensor(donor_values, device) if has_all_donors else None

    samples = [item.get("sample") for item in batch]

    return images, donor, samples


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
        "necessity_logit_drop",
        "sufficiency_logit_retained",
        "dense_delta",
        "dense_logit",
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


def _make_env(images, donor, adapter, cfg, weights, grid, horizon):
    return BatchedRIFTEnv(
        images,
        adapter,
        grid=grid,
        horizon=horizon,
        intervention_mode=cfg.get("intervention_mode", "blur"),
        topk_frac=cfg.get("topk_frac", 0.12),
        reward_fn=weights,
        donor=donor,
        cache_features=_as_bool(cfg.get("cache_features", True), default=True),
        allow_stop_as_noop=_as_bool(cfg.get("allow_stop", False), default=False),
        forbid_revisit=_as_bool(cfg.get("forbid_revisit", True), default=True),
    )


def _mask_invalid_action_logits(logits, state, env, *, allow_stop=False, forbid_revisit=True):
    import torch

    logits = logits.clone()

    # IMPORTANT:
    # Do NOT use:
    #   state.get("action_mask") or state.get("valid_actions")
    # because action_mask is a multi-value torch.Tensor, and Python cannot
    # convert that tensor to a single True/False value.
    action_mask = state.get("action_mask", None)
    if action_mask is None:
        action_mask = state.get("valid_actions", None)

    if torch.is_tensor(action_mask):
        mask = action_mask.to(device=logits.device, dtype=torch.bool)

        if mask.dim() == 1:
            mask = mask.unsqueeze(0)

        if mask.shape[0] == 1 and logits.shape[0] > 1:
            mask = mask.repeat(logits.shape[0], 1)

        if mask.shape == logits.shape:
            dead = ~mask.any(dim=1)
            if dead.any():
                mask = mask.clone()
                mask[dead, :] = True
            return logits.masked_fill(~mask, -1e9)

    if not allow_stop and hasattr(env, "stop_action") and env.stop_action < logits.shape[1]:
        logits[:, env.stop_action] = -1e9

    if forbid_revisit:
        m = state.get("current_mask", None)
        if torch.is_tensor(m):
            if m.dim() == 4:
                filled = m[:, 0].flatten(1) > 0
            elif m.dim() == 3:
                filled = m.flatten(1) > 0
            else:
                filled = None

            if filled is not None and hasattr(env, "n_cells") and filled.shape[1] == env.n_cells:
                all_filled = filled.all(dim=1)

                if all_filled.any():
                    filled = filled.clone()
                    filled[all_filled] = False

                logits[:, : env.n_cells] = logits[:, : env.n_cells].masked_fill(filled, -1e9)

    return logits


def _collect_batched_rollout(policy, env, *, deterministic=False, allow_stop=False, forbid_revisit=True):
    import torch

    buf = BatchedRolloutBuffer()
    state = env.reset()
    info = {}

    for _ in range(env.horizon):
        logits, value = policy(state)
        logits = _mask_invalid_action_logits(
            logits,
            state,
            env,
            allow_stop=allow_stop,
            forbid_revisit=forbid_revisit,
        )
        logp_all = torch.log_softmax(logits, dim=-1)

        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            actions = torch.multinomial(probs, 1).squeeze(1)

        logp = logp_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_state, reward, done, info = env.step(actions)

        buf.add(
            state,
            actions,
            logp,
            reward,
            value.squeeze(-1),
            done,
        )

        state = next_state

        if bool(done.all().item()):
            break

    return buf, info

def train(cfg, adapter, dataloaders):
    import torch

    ddp = _ddp_setup()

    try:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

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
            f"grid={cfg.get('grid', 8)} mode=batched"
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
            cfg.get("out_dir", "experiments/RIFT_rl/ckpt"),
            monitor=cfg.get("monitor", "val/rift_score"),
            mode=cfg.get("mode", "max"),
            top_k=cfg.get("top_k", 3),
            interval=cfg.get("interval", 1),
            save_last=cfg.get("save_last", True),
            best_filename=cfg.get("best_filename", "rift-best-score-epoch={epoch:02d}"),
            every_filename=cfg.get("every_filename", "rift-epoch={epoch:02d}"),
            save_epochs=cfg.get("save_epochs", []),
        )

        wb = WandbLogger(
            project=cfg.get("wandb_project"),
            name=cfg.get("wandb_name") or cfg.get("exp_name"),
            config=cfg,
            enabled=cfg.get("wandb", False),
            entity=cfg.get("wandb_entity"),
            group=cfg.get("wandb_group"),
            tags=cfg.get("wandb_tags", []),
            notes=cfg.get("wandb_notes"),
            mode=cfg.get("wandb_mode", "online"),
            save_code=cfg.get("wandb_save_code", False),
            log_model=cfg.get("wandb_log_model", False),
        )

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

        for param in _base_policy(policy).parameters():
            dist.broadcast(param.data, src=0)

    _ddp_barrier()

    epochs = int(cfg.get("epochs", 50))
    val_every = int(cfg.get("val_every", 1) or 1)
    log_every = int(cfg.get("train_log_every", 10) or 10)

    es_monitor = cfg.get("early_stopping_monitor", cfg.get("monitor", "val/rift_score"))
    es_mode = cfg.get("early_stopping_mode", cfg.get("mode", "max"))
    es_patience = cfg.get("early_stopping_patience", None)
    es_min_delta = float(cfg.get("early_stopping_min_delta", 0.0) or 0.0)
    es_enabled = es_patience is not None and int(es_patience) > 0
    es_best = None
    es_bad_epochs = 0

    # Rank-0 only:
    #   epoch_pbar = overall training progress
    #   pbar       = current epoch batch progress
    if _is_main():
        try:
            from tqdm import tqdm

            epoch_pbar = tqdm(
                total=epochs,
                initial=start_epoch,
                desc="overall training",
                dynamic_ncols=True,
                leave=True,
                position=0,
            )
        except Exception:
            epoch_pbar = None
    else:
        epoch_pbar = None

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
                    leave=False,
                    position=1,
                )
            except Exception:
                pbar = None
        else:
            pbar = None

        local_items = 0
        local_reward_sum = 0.0

        for batch_idx, batch in enumerate(train_dl):
            images, donor, _samples = _batch_to_device(batch, device)
            B = int(images.shape[0])

            env = _make_env(images, donor, adapter, cfg, weights, grid, horizon)
            buf, info = _collect_batched_rollout(
                policy, env, deterministic=False,
                allow_stop=_as_bool(cfg.get("allow_stop", False), default=False),
                forbid_revisit=_as_bool(cfg.get("forbid_revisit", True), default=True),
            )

            logs = algo.update(buf)
            gstep += 1

            batch_reward = buf.total_reward_mean()
            local_items += B
            local_reward_sum += batch_reward * B

            if _is_main() and wb is not None and (gstep % log_every == 0):
                wb.log(
                    {f"train/{k}": v for k, v in logs.items()}
                    | {
                        "train/reward_total": batch_reward,
                        "train/batch_size": B,
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
                "necessity_logit_drop": 0.0,
                "sufficiency_logit_retained": 0.0,
                "dense_delta": 0.0,
                "dense_logit": 0.0,
                "mask_area": 0.0,
            }

        metrics["epoch_string"] = f"Epoch {epoch + 1}/{epochs}"

        should_stop = False

        if _is_main():
            if wb is not None and do_val:
                wb.log(
                    {f"val/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))},
                    step=gstep,
                )

            monitor_key = cfg.get("monitor", "val/rift_score")
            monitor_value = _metric_value(metrics, monitor_key)

            save_metrics = dict(metrics)
            if monitor_value is not None:
                save_metrics[monitor_key] = monitor_value

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
                save_metrics,
            )

            if do_val and es_enabled:
                es_value = _metric_value(metrics, es_monitor)

                if _is_better(es_value, es_best, es_mode, es_min_delta):
                    es_best = float(es_value)
                    es_bad_epochs = 0
                else:
                    es_bad_epochs += 1

                if es_bad_epochs >= int(es_patience):
                    should_stop = True

            if do_val:
                es_txt = ""
                if es_enabled:
                    es_txt = f" early_stop={es_bad_epochs}/{int(es_patience)} best={es_best}"

                msg = (
                    f"[val] {metrics['epoch_string']} "
                    f"rift_score={metrics.get('rift_score'):.4f} "
                    f"faith_delta={metrics.get('faithfulness_ns_delta'):.4f} "
                    f"faith_logit={metrics.get('faithfulness_ns_logit'):.4f} "
                    f"mask_area={metrics.get('mask_area'):.4f} "
                    f"n={metrics.get('n', 0)} "
                    f"ckpt={paths.get('best')}"
                    f"{es_txt}"
                )
            else:
                msg = (
                    f"[ckpt] {metrics['epoch_string']} "
                    f"validation skipped val_every={val_every} "
                    f"latest={paths.get('latest')}"
                )

            log.info(msg)
            print(msg)

        if _is_ddp():
            import torch.distributed as dist

            obj = [bool(should_stop)]
            dist.broadcast_object_list(obj, src=0)
            should_stop = bool(obj[0])

        _ddp_barrier()

        if epoch_pbar is not None:
            epoch_pbar.set_postfix(
                {
                    "epoch": f"{epoch + 1}/{epochs}",
                    "rift": f"{metrics.get('rift_score', 0.0):.4f}",
                    "reward": f"{local_reward_sum / max(local_items, 1):.4f}",
                }
            )
            epoch_pbar.update(1)

        if should_stop:
            if _is_main():
                stop_msg = (
                    f"[early_stop] monitor={es_monitor} mode={es_mode} "
                    f"patience={int(es_patience)} min_delta={es_min_delta} epoch={epoch + 1}"
                )
                log.info(stop_msg)
                print(stop_msg)
            break

    if epoch_pbar is not None:
        epoch_pbar.close()

    if _is_main() and wb is not None:
        wb.finish()

    _ddp_cleanup()

    return _base_policy(policy)


def validate(cfg, adapter, policy, val_dl, weights, grid, horizon):
    import torch

    if _is_ddp():
        device = f"cuda:{_local_rank()}"
    else:
        device = cfg.get("device", "cuda")

    max_batches = int(cfg.get("val_max_batches", 0) or 0)

    sums = {
        "rift_score": 0.0,
        "faithfulness_ns_delta": 0.0,
        "faithfulness_ns_logit": 0.0,
        "necessity_delta_drop": 0.0,
        "sufficiency_delta_retained": 0.0,
        "necessity_logit_drop": 0.0,
        "sufficiency_logit_retained": 0.0,
        "dense_delta": 0.0,
        "dense_logit": 0.0,
        "mask_area": 0.0,
    }
    n = 0

    policy.eval()

    with torch.no_grad():
        for bidx, batch in enumerate(val_dl):
            if max_batches > 0 and bidx >= max_batches:
                break

            images, donor, _samples = _batch_to_device(batch, device)
            B = int(images.shape[0])

            env = _make_env(images, donor, adapter, cfg, weights, grid, horizon)
            buf, info = _collect_batched_rollout(
                policy, env, deterministic=True,
                allow_stop=_as_bool(cfg.get("allow_stop", False), default=False),
                forbid_revisit=_as_bool(cfg.get("forbid_revisit", True), default=True),
            )

            reward = buf.rewards_tensor(device=device).sum(dim=0)

            vals = {
                "rift_score": float(reward.mean().item()),
                "faithfulness_ns_delta": float(info.get("faithfulness_delta", 0.0)),
                "faithfulness_ns_logit": float(info.get("faithfulness_logit", 0.0)),
                "necessity_delta_drop": float(info.get("necessity_delta", 0.0)),
                "sufficiency_delta_retained": float(info.get("sufficiency_delta", 0.0)),
                "necessity_logit_drop": float(info.get("necessity_logit", 0.0)),
                "sufficiency_logit_retained": float(info.get("sufficiency_logit", 0.0)),
                "dense_delta": float(info.get("dense_delta", 0.0)),
                "dense_logit": float(info.get("dense_logit", 0.0)),
                "mask_area": float(info.get("mask_area", 0.0)),
            }

            for k, v in vals.items():
                sums[k] += v * B

            n += B

    denom = max(1, n)

    return {
        "n": n,
        **{k: v / denom for k, v in sums.items()},
    }

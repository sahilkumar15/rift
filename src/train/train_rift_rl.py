# Path: iganer/rift/train/train_rift_rl.py
# Status: NEW
"""RIFT-RL training loop (importable). Freezes detector, rolls out policy in RIFTEnv,
updates with REINFORCE/PPO. Detector + Δ come from a wired CIFTAdapter."""
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
log=get_logger("train")

def train(cfg, adapter, dataloaders):
    import torch
    seed_everything(cfg.get("seed",42))
    train_dl, val_dl, id_mode = dataloaders
    log.info(f"identity_gap_mode={id_mode}")
    grid=cfg.get("grid",8); horizon=cfg.get("horizon",4)
    policy=GridPolicy(grid=grid, n_actions=grid*grid+1).to(cfg.get("device","cuda"))
    algo_name=cfg.get("algo","reinforce")
    algo=(PPO(policy, lr=cfg.get("lr",3e-4),
              lagrangian=cfg.get("lagrangian",False),
              constraint_budget=cfg.get("constraint_budget",0.0))
          if algo_name=="ppo" else Reinforce(policy, lr=cfg.get("lr",3e-4)))
    weights=get_reward_weights(cfg.get("reward_preset","full_rift"))
    ckpt=CheckpointManager(cfg.get("out_dir","outputs/rift_rl"),
                           monitor=cfg.get("monitor","val/rift_score"),
                           mode="max", top_k=cfg.get("top_k",3),
                           interval=cfg.get("interval",10))
    wb=WandbLogger(cfg.get("wandb_project"), cfg.get("exp_name"),
                   cfg, enabled=cfg.get("wandb",False))
    resume=ckpt.resume(cfg.get("resume","auto"))
    start_epoch=0; gstep=0
    if resume:
        st=ckpt.load(resume); policy.load_state_dict(st["policy"]); start_epoch=st["epoch"]+1
        log.info(f"resumed from {resume} @ epoch {start_epoch}")
    epochs=cfg.get("epochs",50)
    for epoch in range(start_epoch, epochs):
        policy.train()
        for batch in train_dl:
            for item in batch:
                img=item["image"].unsqueeze(0).to(cfg.get("device","cuda"))
                s=item.get("sample")
                env=RIFTEnv(img, adapter, grid=grid, horizon=horizon,
                            reward_fn=weights,
                            source_id=getattr(s,"source_id",None) if s else None,
                            target_id=getattr(s,"target_id",None) if s else None)
                buf=RolloutBuffer(); state=env.reset(); done=False
                while not done:
                    logits,value=policy(state)
                    probs=torch.softmax(logits,-1)
                    a=int(torch.multinomial(probs,1)[0,0].item())
                    logp=float(torch.log_softmax(logits,-1)[0,a].item())
                    nstate,r,done,info=env.step(a)
                    buf.add(state,a,logp,r,float(value.item()),done)
                    state=nstate
                logs=algo.update(buf); gstep+=1
                wb.log({f"train/{k}":v for k,v in logs.items()}|
                       {"train/reward_total":sum(buf.rewards),
                        "epoch":epoch}, step=gstep)
        metrics=validate(cfg, adapter, policy, val_dl, weights, grid, horizon)
        metrics["epoch_string"]=f"Epoch {epoch+1}/{epochs}"
        wb.log({f"val/{k}":v for k,v in metrics.items() if isinstance(v,(int,float))}, step=gstep)
        paths=ckpt.save({"policy":policy.state_dict(),"epoch":epoch,
                         "global_step":gstep,"config":dict(cfg),
                         "rng":get_rng_states()}, epoch, {cfg.get("monitor","val/rift_score"):metrics.get("rift_score",0.0)})
        log.info(f"{metrics['epoch_string']} rift_score={metrics.get('rift_score'):.4f} ckpt={paths.get('best')}")
    wb.finish(); return policy

def validate(cfg, adapter, policy, val_dl, weights, grid, horizon):
    import torch
    from ..audit.audit_runner import audit_one, aggregate
    from ..explainers.rift_policy_explainer import RIFTPolicyExplainer
    rows=[]
    expl=RIFTPolicyExplainer(policy, lambda img,ad: RIFTEnv(img,ad,grid=grid,horizon=horizon,reward_fn=weights), horizon)
    policy.eval()
    with torch.no_grad():
        for batch in val_dl:
            for item in batch:
                img=item["image"].unsqueeze(0).to(cfg.get("device","cuda"))
                row,_,_,_=audit_one(img, adapter, expl, reward_weights=weights)
                rows.append(row)
    agg=aggregate(rows)
    return {"rift_score":agg.get("rift_score",0.0),
            "faithfulness_ns_delta":agg.get("faithfulness_delta",0.0),
            "faithfulness_ns_logit":agg.get("faithfulness_logit",0.0),
            "necessity_delta_drop":agg.get("necessity_delta",0.0),
            "sufficiency_delta_retained":agg.get("sufficiency_delta",0.0),
            "mask_area":agg.get("mask_area",0.0)}

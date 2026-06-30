#!/usr/bin/env bash
set -e
for row in acc_only plausibility_only generic_logit_interv delta_reward_no_int delta_int_no_rl full_rift_rl; do
  echo "=== ablation: $row ==="
  python train_rift_rl.py --config configs/train_rift_rl.yaml \
    # override reward_preset via your config loader / env as needed
done

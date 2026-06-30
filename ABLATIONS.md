# RIFT — Ablation Study (design + how to run)

Every block defends one paper claim. ✓ = factor on, ✗ = off, — = undefined/not-applicable.
Cells are defined in `configs/ablations_rift.yaml`; run them via `scripts/run_rift.sh`.

Print the whole plan without touching the model:
```bash
bash scripts/run_rift.sh --mode ablations --dry-run
```

---

## Block 0 — Intervention validity (precondition)

**Claim defended:** masking the cited region moves Δ *specifically* (not "masking anything
moves it"). If this fails, every downstream number is confounded → pivot.

Sweep the masking operator/area; PASS if `mean(cited_drop) − mean(random_drop) ≥ 0.15`.

| cell        | mask_op | topk_frac |
|-------------|---------|-----------|
| blur_topk10 | blur    | 0.10      |
| blur_topk12 | blur    | 0.12      |
| mean_topk12 | mean    | 0.12      |
| zero_topk12 | zero    | 0.12      |

```bash
bash scripts/run_rift.sh --mode gates --csv data/slices/example_ffpp_forged.csv --ckpt $CKPT
```

---

## Block 1 — Method factorial (the MAIN table)

**Claim defended:** Δ-grounding beats a generic-logit (DeFacto-style) intervention, and
plausibility ≠ faithfulness (MARE-style exposed).

Factors: **I** = intervene (necessity+sufficiency), **G** = ground evidence in Δ (vs detector
logit), **P** = plausibility (annotation IoU), **K** = constraints (sparsity/identity/perceptual).
**G is "—" whenever I = ✗** — the runner refuses to credit Δ-faithfulness to those cells.

| cell                | I | G | P | K | ≈ prior work | isolates                         |
|---------------------|---|---|---|---|--------------|----------------------------------|
| acc_only            | ✗ | — | ✗ | ✗ | floor        | lower bound                      |
| plausibility_only   | ✗ | — | ✓ | ✗ | MARE         | plausibility without causation   |
| generic_logit       | ✓ | ✗ | ✗ | ✓ | DeFacto      | intervention without Δ-grounding |
| delta_reward_no_int | ✗ | — | ✗ | ✓ | degenerate   | Δ reward but no falsification     |
| delta_grounded      | ✓ | ✓ | ✗ | ✓ | **RIFT core**| **+ Δ-grounding** (vs generic)   |
| full_rift           | ✓ | ✓ | ✓ | ✓ | **RIFT**     | + plausibility reporting         |

Auto-computed contrasts (`outputs/cells/block1_contrasts.json`):
- `delta_grounding_gain = faithfulness(delta_grounded) − faithfulness(generic_logit)` — the novelty.
- `mare_plausibility` high & `mare_faithfulness` low — the exposé.
- `no_int_sanity` — `delta_reward_no_int` must NOT earn Δ-faithfulness.

```bash
bash scripts/run_rift.sh --mode ablations --block 1 --seeds 0,1,2 --ckpt $CKPT
# decisive pair only:
bash scripts/run_rift.sh --mode ablations --block 1 --only generic_logit,delta_grounded --ckpt $CKPT
```

---

## Block 2 — Audit leaderboard (the exposé)

**Claim defended:** the leaderboard surfaces plausible-but-unfaithful explanations,
including CIFT's own Δ regions. A row is flagged `exposed` if plausibility ≥ 0.60 and
faithfulness ≤ 0.35.

| explainer     | spatial | uses Δ | external | expect plausible | expect faithful |
|---------------|---------|--------|----------|------------------|-----------------|
| random        | ✓       | ✗      | ✗        | ✗                | ✗ (floor)       |
| gradcam_logit | ✓       | ✗      | ✗        | ~                | ~ (logit)       |
| cift_gap      | ✓       | ✓      | ✗        | ~                | ✓ (if Gate 1)   |
| annotation    | ✓       | ✗      | human    | ✓                | ✗  ← exposé     |
| vlm_external  | ✓       | ✗      | VLM      | ✓                | ?  (measured)   |
| rift_policy   | ✓       | ✓      | ✗        | ~                | ✓ (repair)      |

```bash
bash scripts/run_rift.sh --mode audit --ckpt $CKPT       # -> outputs/leaderboard.csv
```

---

## Block 3 — Predictive correlation (the HEADLINE)

**Claim defended:** faithfulness predicts zero-shot AUC; in-domain accuracy does not.
Includes a **saturation control** (re-run on a matched in-domain-AUC band) so the
dissociation is not just "in-domain AUC is saturated, so it correlates with nothing."

| predictor     | expected | role             |
|---------------|----------|------------------|
| faithfulness  | + strong | headline         |
| in_domain_auc | ~ 0      | the dissociation |
| plausibility  | weak     | control          |

n < 15 → reported as a **trend**, not a headline.

```bash
# build checkpoints_metrics.csv: faithfulness,in_domain_auc,plausibility,zero_shot_auc (one row/ckpt)
bash scripts/run_rift.sh --mode correlation --corr-csv checkpoints_metrics.csv
```

---

## Block 4 — RL sensitivity (is it a bandit?)

**Claim defended:** the repair policy is a real sequential policy (H>1 beats H=1).
If `seq_h4` ties `bandit` within the tie margin, **demote RL to a repair note** (state honestly).

| cell       | H | game | protect | tests                       |
|------------|---|------|---------|-----------------------------|
| static     | 1 | ✗    | ✗       | no policy (floor)           |
| bandit     | 1 | ✓    | ✓       | single-step policy          |
| seq_h2     | 2 | ✓    | ✓       | does multi-step help?       |
| seq_h4     | 4 | ✓    | ✓       | RIFT default                |
| seq_h8     | 8 | ✓    | ✓       | diminishing returns?        |
| no_protect | 4 | ✓    | ✗       | reward-hacking check        |

```bash
# train one horizon per call (Block-4 trains a policy per cell):
bash scripts/run_rift.sh --mode train --horizon 1 --ckpt $CKPT
bash scripts/run_rift.sh --mode train --horizon 4 --ckpt $CKPT
```

---

## Outputs

- `outputs/table_rift.csv` — combined table across all cells/blocks.
- `outputs/cells/*.csv|json` — per-cell rows + `block1_contrasts.json`.
- `outputs/leaderboard.csv` — Block-2 leaderboard with the `exposed` flag.
- `outputs/correlation.json` — Block-3 results incl. the saturation-controlled re-run.

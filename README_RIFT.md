# RIFT — Reinforced Identity-Gap Faithfulness Testing

Auditing and reinforcing deepfake **explanation faithfulness** via **identity-gap
interventions**, built on the CIFT donor-grounded identity-gap detector.

RIFT is **not** a new detector and **not** "CIFT + RL". It is a *measurement*:
it tests whether a detector's stated explanation is the actual cause of its forensic
signal, by intervening on the cited region and checking whether the donor-grounded
identity gap **Δ** collapses (necessity) and survives when only that region is kept
(sufficiency). RL is a repair mechanism, not the core novelty.

The three contributions:
1. explanation-as-a-falsifiable-mechanistic-claim (necessity/sufficiency on Δ),
2. a faithfulness leaderboard that exposes plausible-but-unfaithful explanations
   (including CIFT's own Δ regions and MARE-style annotation matching), and
3. the finding that **faithfulness predicts cross-dataset generalization while
   in-domain accuracy does not** — *conditional on the Phase-0 gates passing.*

---

## 0. The one thing to read first (honesty)

CIFT's deployed detector is **source-free**: its decision logit comes from the
Global Head + Spatial-Mamba branches on the analyzed image alone. **Δ is an
analysis-time diagnostic**, computed by re-enabling the training-only XID-Mamba
branch and pairing the sample with a **retrieved same-identity reference** (CIFT
paper §C.6, Table 10). Two consequences baked into this codebase:

- RIFT audits **two distinct evidence channels** and keeps them separate:
  the **logit** channel (the deployed decision) and the **Δ** channel (CIFT's
  validated forensic signal — the thing CIFT shows is what *generalizes*).
  Do **not** claim necessity/sufficiency-on-Δ tests the deployed *decision*.
- **TRUE Δ needs a donor stream.** A *retrieved same-identity reference* counts
  (that is how CIFT itself measures test-time Δ). With no donor, Δ is a tagged
  **proxy** and `w_delta` is forced to 0 — proxy numbers can never earn Δ credit.

**Nothing downstream of Phase 0 is valid until Gate 1 passes.** Run the gates first.

---

## 1. Project structure

```
RIFT/
├── pyproject.toml              # installable; defines rift-gate1/2/3 console scripts
├── conftest.py                 # makes `src` importable for pytest in-place
├── requirements_rift.txt
├── README_RIFT.md              # this file
├── ABLATIONS.md                # the tick/cross ablation design (read alongside this file)
├── src/                        # the package root (import as src.xxx)
│   ├── adapters/
│   │   ├── cift_adapter.py            # the seam, fully WIRED to real CIFT
│   │   ├── identity_gap_contract.py   # honesty guard (true/proxy/error)
│   │   └── detector_adapter.py        # generic detector wrapper (Xception/SBI/...)
│   ├── gates/                         # PHASE 0 decision scripts (run these first)
│   │   ├── gate1_validity.py          #   intervention validity (precondition)
│   │   ├── gate2_separation.py        #   novelty isolation (delta vs logit vs MARE)
│   │   ├── gate3_correlation.py       #   the headline correlation test
│   │   └── _io.py                     #   tiny image/mask loaders
│   ├── interventions/                 # necessity/sufficiency masking + Gate-1 probe
│   ├── faithfulness/                  # causal-faithfulness math (no model deps)
│   ├── explainers/                    # random / gradcam / cift_gap / rift_policy / ...
│   ├── audit/                         # leaderboard + ablation_runner (config-driven planner)
│   ├── metrics/                       # binary / robustness / correlation stats
│   ├── rl/                            # PPO/REINFORCE repair policy + env + reward
│   ├── data/                          # CSV split dataset + transforms + datamodule
│   ├── eval/                          # eval + correlation entrypoint logic
│   └── utils/                         # config, seed, logging, checkpointing, wandb
├── configs/
│   ├── rift_general.yaml       # master config (CIFT-schema dataset/model + RIFT blocks)
│   ├── ablations_rift.yaml     # the tick/cross cell spec (read by ablate_rift.py)
│   └── *.yaml                  # audit / correlation / train / eval configs
├── scripts/
│   └── run_rift.sh             # generalized launcher (gates|audit|correlation|ablations|train)
├── ablate_rift.py              # ablation/audit dispatcher (pure-logic --dry-run + torch execution)
├── train_rift_rl.py            # RL repair-policy entrypoint (Block 4)
├── eval_rift.py  audit_rift.py  correlate_rift.py
├── tests/                      # pure-logic unit tests (run without torch)
└── data/slices/                # example split CSV (format reference)
```

The package name is `src````

The package name is `src` (matches every file header and the test imports).

---

## 2. Install

The pure-logic core and **all gate decision logic run WITHOUT torch** — you can
validate the whole pipeline offline before touching a GPU.

**Offline (laptop / login node), enough to run tests + gate `--selftest`:**
```bash
pip install -e .
```

**On Katz, for the real model runs (adds torch + model stack):**
```bash
pip install -e ".[runtime,test]"
# CIFT itself (cldm/share/datasets) is imported from your existing checkout via
# --cift-root; you do NOT install it. Make sure that checkout's own requirements
# (timm, omegaconf, einops, the LDM stack, mamba deps) are already satisfied there.
```

---

## 3. Verify the install (do this now, offline)

```bash
pytest tests/ -q                 # expect: 13 passed
rift-gate1 --selftest            # decision logic for Gate 1
rift-gate2 --selftest            # decision logic for Gate 2
rift-gate3 --selftest            # decision logic for Gate 3 (shows n>=15 vs n<15)
```
If those pass, your wiring is sound and only the model-touching numbers remain.

---

## 4. The split CSV format

Every dataset is a CSV with this header (see `data/slices/example_ffpp_forged.csv`):

```
image_path,label,source_id,target_id,manipulation_type,mask_path,donor_path,metadata_json
```

- `image_path`  — the analyzed image (the candidate forgery).
- `label`       — 1=forged, 0=genuine.
- `mask_path`   — the **cited region** to test (the explanation under audit). For
                  Gate 1 use the detector's own cited map; for the MARE-style row
                  use a human annotation mask.
- `donor_path`  — **required for TRUE Δ.** Path to the donor face OR a retrieved
                  same-identity reference. Absent → proxy mode (Δ credit disabled).
- `source_id/target_id/manipulation_type/metadata_json` — optional; used by the
  datamodule for retrieval/forgery-type routing. Never crash on missing fields.

---

## 5. Phase-gated workflow (run in this order)

> Set once:
> ```bash
> export CIFT_ROOT=/scratch/sahil/projects/img_deepfake/code/ImageDifussionFake
> export CKPT=/path/to/your_cift.ckpt
> ```

### PHASE 0 — Validity gate (single-GPU; nothing else until it passes)

**Gate 1 — intervention validity (the precondition):**
```bash
rift-gate1 \
  --csv data/slices/example_ffpp_forged.csv \
  --cift-root "$CIFT_ROOT" --ckpt "$CKPT" --backbone convnextv2_base \
  --device cuda --n 50 --min-sep 0.15
```
Read the `separation=` line. **PASS (≥0.15) → GO** to Gate 2. **FAIL → STOP and pivot.**

**Gate 2 — novelty isolation (Δ-grounded vs generic-logit vs MARE-style):**
```bash
rift-gate2 \
  --csv data/slices/example_ffpp_forged.csv \
  --cift-root "$CIFT_ROOT" --ckpt "$CKPT" --device cuda --margin 0.10
```
**GO** iff Δ-grounded faithfulness beats generic-logit by the margin **and** the
MARE-style row is high-plausibility/low-faithfulness. **NO-GO → fall back to
"RIFT = audit protocol"** (the script prints the exact honest framing).

**Gate 3 — the headline correlation test:**
First build `checkpoints_metrics.csv` (one row per checkpoint, columns
`faithfulness,in_domain_auc,plausibility,zero_shot_auc`) from the audit + CIFT eval,
then:
```bash
rift-gate3 --csv checkpoints_metrics.csv
```
**HEADLINE** requires n≥15 and the faithfulness-yes / in-domain-no dissociation;
n<15 → reported as a **trend**, not a headline.

### PHASE 1 — CIFT adapter (already wired)

`src/adapters/cift_adapter.py` is complete (WIRE 1–5), verified against the
real CIFT source (`cldm/diffusionfake.py`, `cldm/mamba_modules.py`,
`cift_eval_complete.py`). On your **first** Katz run confirm three conventions
(one smoke batch is enough):
1. images reach the adapter as BCHW float in **[-1,1]** (the `_io` loaders do this);
2. CIFT output exposes `v/logits` or `v/probs` (handled either way);
3. `control_model._gap` is non-zero on a donor-paired forged sample (dual path active).
If `--strict-identity-gap` raises "proxy", you forgot `donor_path`.

### PHASE 2 — Generalization + modularity (**only after Gate 1 PASS**)

Config-driven 5-block ablation, multi-seed, CSV/JSON output, all through the one
generalized launcher (see `ABLATIONS.md` for the full ✓/✗ cell tables):
```bash
bash scripts/run_rift.sh --mode ablations --dry-run                       # print the ✓/✗ plan first
bash scripts/run_rift.sh --mode ablations --seeds 0,1,2 --ckpt "$CKPT" \
    --cift-root "$CIFT_ROOT"                                              # Blocks 0/1/2/4
bash scripts/run_rift.sh --mode audit --ckpt "$CKPT" --cift-root "$CIFT_ROOT"          # Block 2 only
bash scripts/run_rift.sh --mode correlation --corr-csv checkpoints_metrics.csv         # Block 3 only
```
Honesty is enforced in code: proxy mode earns no Δ credit; n<5 correlations flagged;
H=1-ties-H>1 is reported, not hidden (demotes RL to a repair note). The legacy
single-purpose scripts (`run_rift_audit.sh`, `run_rift_correlation.sh`,
`run_rift_ablations.sh`) are kept in `scripts/` for reference but `run_rift.sh` is
the supported entrypoint going forward.

### PHASE 3 — Training hardening (**only after Phase 0 GO and the method is fixed**)

Only now add scale (HF Accelerate / multi-GPU / bf16 / safe resume / rank-0 W&B):
```bash
bash scripts/train_rift_rl.sh
bash scripts/resume_rift_rl.sh
```
Do not introduce distributed complexity earlier — it obscures bugs in the science.

---

## 6. Honesty guarantees built into the code

- `identity_gap()` returns a mode tag (`true`/`proxy`/`error`); **proxy can never
  earn Δ-faithfulness credit** (`w_delta` forced to 0 in `compute_rift_score`).
- Missing donor → warn + tag `proxy`, never a silent fake Δ.
- `strict_identity_gap=True` → raises instead of proxying.
- `assert_mechanism_valid()` refuses to let a proxy number masquerade as the mechanism.
- Correlation flags small-n as non-reportable; Gate 3 flags n<15 as trend-only.

---

## 7. Troubleshooting

- `ModuleNotFoundError: cldm/share/datasets` → `--cift-root` is wrong or that
  checkout isn't on `PYTHONPATH`; the adapter prepends `cift_root` to `sys.path`.
- Gate 1 prints "all Δ ≈ 0" → the dual-identity path didn't activate: check that
  `donor_path` is set and that the CIFT ablation flag `use_dimf` is on.
- `pytest` can't import `src` → run from the repo root, or `pip install -e .`.
- Out-of-memory on the 80GB A100 in Phase 3 only → drop batch size, set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (Phase 0–2 are single-GPU,
  small-slice, and should not OOM).
```
```

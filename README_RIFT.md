# RIFT — Reinforced Identity-Gap Faithfulness Testing

Modular implementation scaffold for the RIFT paper. **Pure-logic core is unit-tested
and runs without torch; model-touching code is one wired adapter.**

## Quickstart (on Katz)
```bash
pip install -r requirements_rift.txt
# 1) WIRE your CIFT model: edit iganer/rift/adapters/cift_adapter.py (5 spots: # === WIRE ===)
# 2) GATE-1 first (precondition for the whole paper):
python -c "from iganer.rift.interventions.interventions import gate1_intervention_validity"
#    feed ~50 images + cited masks + evidence_fn=adapter.identity_gap(...).value
#    -> if Gate1Report.passed()==False : STOP, pivot. Don't train.
# 3) Audit leaderboard:           bash scripts/run_rift_audit.sh
# 4) Correlation (4.5/5 result):  bash scripts/run_rift_correlation.sh
# 5) Train RL repair (gated):     bash scripts/train_rift_rl.sh
# 6) Ablations:                   bash scripts/run_rift_ablations.sh
```

## Run tests (no torch needed)
```bash
pip install pytest pyyaml numpy
pytest tests/ -q
```

## Honesty guarantees built into the code
- `identity_gap()` returns a mode tag (true/proxy/error). Proxy mode **cannot** earn
  Δ-faithfulness credit (`w_delta` forced to 0).
- Missing donor metadata -> warns + tags `identity_gap_mode=proxy`, never silent fake Δ.
- `strict_identity_gap=true` -> raises instead of proxying.
- Correlation flags n<5 as non-reportable.

## The one file you complete
`adapters/cift_adapter.py` — 5 spots marked `# === WIRE ===`. Everything else consumes it.
WIRE 4 is critical: confirm you return true Δ=‖g_s−g_t‖₂ (gap readout), not a norm proxy.

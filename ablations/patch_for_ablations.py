from __future__ import annotations

"""Ablation compatibility verifier.

Older versions of this file rewrote src/rl/reward.py, src/faithfulness/
faithfulness_score.py and src/rl/batched_rift_env.py at launch time. That was
unsafe: running an ablation could silently overwrite the training reward used by
main RIFT. The fixed version only verifies that the required objective presets
and reward hooks are present.
"""

from pathlib import Path

REQUIRED_PRESETS = [
    "acc_only",
    "plausibility",
    "generic_logit",
    "delta_no_interv",
    "full_rift",
    "full_rift_shaped",
    "necessity_only",
    "sufficiency_only",
    "no_sparsity",
]


def _read(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return p.read_text()


def verify() -> None:
    reward = _read("src/rl/reward.py")
    faith = _read("src/faithfulness/faithfulness_score.py")
    env = _read("src/rl/batched_rift_env.py")

    missing = [name for name in REQUIRED_PRESETS if f'"{name}"' not in reward]
    if missing:
        raise RuntimeError(f"reward presets missing: {missing}")

    for token in ["objective", "necessity", "sufficiency"]:
        if token not in faith:
            raise RuntimeError(f"faithfulness_score.py missing token: {token}")
        if token not in env:
            raise RuntimeError(f"batched_rift_env.py missing token: {token}")

    if "logit_to_evidence" not in env:
        raise RuntimeError("batched_rift_env.py should use logit_to_evidence/softplus evidence, not raw signed logits")

    print("[ok] RIFT ablation reward/objective code verified; no files overwritten")


if __name__ == "__main__":
    verify()

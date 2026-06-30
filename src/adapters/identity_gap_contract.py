# Path: src/adapters/identity_gap_contract.py
# Status: NEW
"""
identity_gap_contract.py  —  THE most important file in RIFT.

Every mechanism-level RIFT claim (necessity/sufficiency on Δ, the audit
leaderboard, the faithfulness-vs-generalization correlation) is only valid if
the number returned by `identity_gap()` is the *true donor-grounded* Δ from the
CIFT formulation:  Δ = || g_s - g_t ||_2  between a donor (source) stream and a
target stream, with donor metadata actually present.

CIFT's own paper is explicit that for genuine frames x_s = x so Δ = 0 by
construction, and that NO test benchmark ships donor references. That means: on
every zero-shot dataset, true Δ is UNAVAILABLE and any "Δ" you compute is a
proxy. RIFT must never present a proxy as if it were the real mechanism.

This module does not compute Δ. It defines the contract + a guard so that the
rest of RIFT cannot accidentally run mechanism-level claims on proxy numbers.
Wire your real CIFT model into CIFTAdapter.identity_gap() (adapters/cift_adapter.py);
this file enforces honesty around whatever that returns.
"""
from __future__ import annotations

import enum
import warnings
from dataclasses import dataclass
from typing import Optional


class IdentityGapMode(str, enum.Enum):
    TRUE = "true"      # donor-grounded Δ = ||g_s - g_t||_2, real donor metadata present
    PROXY = "proxy"    # feature/embedding-distance stand-in; donor metadata absent
    ERROR = "error"    # metadata missing AND strict mode on -> caller must raise


@dataclass
class IdentityGapResult:
    """What identity_gap() returns. `value` is meaningless unless you check `mode`."""
    value: float
    mode: IdentityGapMode
    has_donor_metadata: bool
    detail: str = ""

    @property
    def is_true_delta(self) -> bool:
        return self.mode == IdentityGapMode.TRUE

    def assert_mechanism_valid(self, what: str = "mechanism-level RIFT claim") -> None:
        """Call this before ANY result that the paper interprets as Δ-grounded."""
        if self.mode != IdentityGapMode.TRUE:
            raise MechanismValidityError(
                f"{what} requires true donor-grounded Δ, but identity_gap() ran in "
                f"mode='{self.mode.value}' (has_donor_metadata={self.has_donor_metadata}). "
                f"Detail: {self.detail or 'n/a'}. "
                f"Either supply source_id/target_id donor metadata, or label these "
                f"results identity_gap_mode=proxy and do NOT claim they test the CIFT Δ mechanism."
            )


class MechanismValidityError(RuntimeError):
    """Raised when a Δ-mechanism claim is attempted on proxy/missing data."""


def resolve_mode(has_donor_metadata: bool, strict: bool) -> IdentityGapMode:
    """
    Single decision point used by every adapter so behaviour is uniform:

      donor metadata present            -> TRUE
      absent, strict_identity_gap=True  -> ERROR (caller raises)
      absent, strict_identity_gap=False -> PROXY (warn once, mark outputs)
    """
    if has_donor_metadata:
        return IdentityGapMode.TRUE
    if strict:
        return IdentityGapMode.ERROR
    warnings.warn(
        "RIFT: donor (source/target) identity metadata is ABSENT -> falling back to "
        "PROXY identity-gap. Proxy numbers are a feature-distance stand-in, NOT the "
        "CIFT donor-grounded Δ. All outputs will be tagged identity_gap_mode=proxy and "
        "must NOT be reported as a test of the identity-gap mechanism.",
        RuntimeWarning,
        stacklevel=2,
    )
    return IdentityGapMode.PROXY

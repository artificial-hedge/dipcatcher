"""Online multiple-testing — alpha-investing over the receipt stream.

Batch BH (``corpus_inference``) answers "which committed claims survive"
at one fixed corpus. But receipts accumulate over time: each new artifact
is another hypothesis. Pool-and-rerun after every arrival silently
multiplies the family size; the honest stream procedure is
alpha-investing (Foster & Stine 2008; Javanmard & Montanari 2018 LORD):
a bounded wealth budget W is allocated per test, spent on failures, and
replenished by rejections, keeping mFDR <= alpha at every arrival time.

Rule implemented (Foster–Stine alpha-investing):

- ``alpha_t = gamma_{j_t} * W_t`` where ``gamma`` is a summable sequence
  (``gamma_j ∝ 1/(j+1)^2``, normalized to sum 1) and ``j_t`` counts tests
  since the last rejection — the longer a dry streak, the less wealth
  each new test may wager.
- Rejecting ``p_t <= alpha_t`` pays ``omega`` into wealth; a non-rejection
  forfeits ``psi_t = alpha_t / (1 - alpha_t)`` (the Foster–Stine cost
  function — paying only alpha_t would let an adversary burn wealth
  sublinearly on near-boundary losses).
- Wealth floors at zero (alpha_t = 0 ⇒ nothing can be wagered — a
  dead lane stays dead rather than borrowing future wealth).

Deliberately conservative: p-values are validated per update, the
gamma index restarts only on true rejections, and the verdict object
records every wager so a stream can be replayed for audit.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quant_fund.research.fleet_eval import _atomic_write_text
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes


def _gamma(j: int) -> float:
    """Summable spending sequence: gamma_j ∝ 1/(j+1)^2, Σ = 1."""
    zeta2 = math.pi**2 / 6.0
    return (1.0 / (j + 1) ** 2) / zeta2


@dataclass(frozen=True)
class OnlineTestState:
    """State after one stream update."""

    index: int
    p_value: float
    alpha_t: float
    rejected: bool
    wealth: float


@dataclass
class OnlineFDR:
    """Foster–Stine alpha-investing controller over a p-value stream.

    ``level`` is the target mFDR bound; ``initial_wealth`` and ``payout``
    default to the standard split ``w0·α`` / ``(1−w0)·α`` with w0=0.5.
    """

    level: float = 0.05
    initial_wealth: float | None = None
    payout: float | None = None
    _w0_fraction: float = 0.5
    _wealth: float = field(init=False)
    _payout_resolved: float = field(init=False)
    _since_rejection: int = field(default=0, init=False)
    _states: list[OnlineTestState] = field(default_factory=list, init=False)
    _n_rejections: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not (0.0 < self.level < 1.0):
            raise ValueError("level must lie in (0, 1)")
        if self.initial_wealth is None:
            self.initial_wealth = self._w0_fraction * self.level
        if self.payout is None:
            self.payout = (1.0 - self._w0_fraction) * self.level
        if not (0.0 < self.initial_wealth <= self.level):
            raise ValueError("initial_wealth must lie in (0, level]")
        if not (self.payout > 0.0):
            raise ValueError("payout must be positive")
        self._wealth = float(self.initial_wealth)
        self._payout_resolved = float(self.payout)

    def update(self, p_value: float) -> OnlineTestState:
        """Test one new p-value; returns the post-update state."""
        p = float(p_value)
        if not math.isfinite(p) or not (0.0 <= p <= 1.0):
            raise ValueError("p_value must lie in [0, 1]")
        j = self._since_rejection
        alpha_t = min(_gamma(j) * self._wealth, self._wealth)
        rejected = p <= alpha_t
        if rejected:
            self._wealth = self._wealth + self._payout_resolved
            self._n_rejections += 1
            self._since_rejection = 0
        else:
            self._wealth = max(
                0.0, self._wealth - alpha_t / (1.0 - alpha_t) if alpha_t < 1.0 else 0.0
            )
            self._since_rejection += 1
        state = OnlineTestState(
            index=len(self._states),
            p_value=p,
            alpha_t=alpha_t,
            rejected=rejected,
            wealth=self._wealth,
        )
        self._states.append(state)
        return state

    @property
    def wealth(self) -> float:
        return self._wealth

    @property
    def rejections(self) -> list[int]:
        return [s.index for s in self._states if s.rejected]

    def stream_report(
        self,
        *,
        inputs_sha256: str | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Streaming summary; with ``inputs_sha256`` and ``params``
        supplied it is also a complete ``online_fdr.v1`` receipt body for
        :func:`write_online_fdr_receipt` (the sealer's contract requires
        them — the digest and run params belong to the caller's dataset)."""
        report: dict[str, Any] = {
            "kind": "online_fdr.v1",
            "schema": "online_fdr.v1",
            "research_only": True,
            "live_pnl_claim": False,
            "n_tests": len(self._states),
            "n_rejections": self._n_rejections,
            "rejection_indices": self.rejections,
            "final_wealth": self._wealth,
            "level": self.level,
            "evidence": ["foster_stine_alpha_investing", "mfdr_bounded", "summable_gamma"],
        }
        if inputs_sha256 is not None:
            report["inputs_sha256"] = inputs_sha256
        if params is not None:
            report["params"] = dict(params)
        return report


def _is_sha256_str(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def _online_fdr_body_errors(receipt: Mapping[str, Any]) -> list[str]:
    """Contract errors for an ``online_fdr.v1`` receipt body — count fields
    re-derived from the stream summary, never trusted verbatim."""
    errors: list[str] = []
    if receipt.get("kind") != "online_fdr.v1" or receipt.get("schema") != "online_fdr.v1":
        errors.append("kind_schema_mismatch")
    if receipt.get("research_only") is not True:
        errors.append("research_only_not_true")
    if receipt.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    if not _is_sha256_str(receipt.get("inputs_sha256")):
        errors.append("inputs_sha256_not_hex")
    if not isinstance(receipt.get("params"), Mapping):
        errors.append("params_not_mapping")
    level = receipt.get("level")
    try:
        level_ok = (
            isinstance(level, (int, float))
            and not isinstance(level, bool)
            and 0.0 < float(level) < 1.0
        )
    except (TypeError, ValueError, OverflowError):
        level_ok = False
    if not level_ok:
        errors.append("level_out_of_unit_interval")
    n_tests = receipt.get("n_tests")
    if not isinstance(n_tests, int) or isinstance(n_tests, bool) or n_tests < 0:
        errors.append("n_tests_not_nonnegative_int")
        n_tests = None
    n_rejections = receipt.get("n_rejections")
    if not isinstance(n_rejections, int) or isinstance(n_rejections, bool) or n_rejections < 0:
        errors.append("n_rejections_not_nonnegative_int")
    elif n_tests is not None and n_rejections > n_tests:
        errors.append("n_rejections_exceeds_n_tests")
    indices = receipt.get("rejection_indices")
    if not isinstance(indices, list) or not all(
        isinstance(i, int) and not isinstance(i, bool) for i in indices
    ):
        errors.append("rejection_indices_not_int_list")
    elif n_tests is not None:
        if any(not (0 <= i < n_tests) for i in indices):
            errors.append("rejection_index_out_of_range")
        if len(set(indices)) != len(indices):
            errors.append("rejection_indices_not_unique")
        if isinstance(n_rejections, int) and n_rejections != len(indices):
            errors.append("n_rejections_mismatches_indices")
    wealth = receipt.get("final_wealth")
    try:
        wealth_ok = (
            isinstance(wealth, (int, float))
            and not isinstance(wealth, bool)
            and math.isfinite(float(wealth))
            and float(wealth) >= 0.0
        )
    except (TypeError, ValueError, OverflowError):
        wealth_ok = False
    if not wealth_ok:
        errors.append("final_wealth_not_finite_nonneg")
    evidence = receipt.get("evidence")
    if (
        not isinstance(evidence, list)
        or not evidence
        or not all(isinstance(e, str) for e in evidence)
    ):
        errors.append("evidence_not_nonempty_str_list")
    return errors


def write_online_fdr_receipt(
    receipt: Mapping[str, Any],
    receipts_dir: Path | str = Path("receipts"),
    *,
    receipt_version: int = 1,
) -> Path:
    """Seal an online_fdr receipt and write ``online_fdr_<hash>.json``.

    Filename digest = ``inputs_sha256`` (v1) or the canonical
    ``receipt_sha256`` (v2). Atomic, fail-closed on a malformed receipt.
    ``receipt_version=2`` wraps the same body in the unified ``receipt.v2``
    envelope instead.
    """
    from quant_fund.research.receipt_v2 import seal_receipt, wrap_receipt_v2

    body_errors = _online_fdr_body_errors(receipt)
    if body_errors:
        raise ValueError("online_fdr receipt violates its contract: " + ",".join(body_errors))
    if receipt_version == 1:
        canonical = json.loads(canonical_json_bytes(dict(receipt)))
        digest = hash_bytes(canonical_json_bytes(canonical))
        payload = {**canonical, "receipt_sha256": digest}
        name_digest = str(receipt["inputs_sha256"])[:16]
    elif receipt_version == 2:
        payload = seal_receipt(
            wrap_receipt_v2(
                receipt,
                code_files=(Path(__file__),),
                verdict="pass",
            )
        )
        name_digest = str(payload["receipt_sha256"])[:16]
    else:
        raise ValueError(f"receipt_version must be 1 or 2, got {receipt_version!r}")
    path = Path(receipts_dir) / f"online_fdr_{name_digest}.json"
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path

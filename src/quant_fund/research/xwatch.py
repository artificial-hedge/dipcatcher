"""Cross-head lead audit — an e-process over paired residual signs.

Sibling lanes audit one stream at a time: coverage, calibration, tail depth,
serial dependence. This lane audits *pairs* of heads. The hypothesis is a
Granger-style lead at the residual level: if head A's per-origin proper-loss
residual sign predicts head B's sign ``k`` origins ahead, then B's signal is
a lagging copy of A's — ensemble-design evidence that B is redundant on the
information A already supplies (or that B is fit to stale features).

Inputs are per-origin *difference* streams: ``leader_diff`` and
``follower_diff`` are either loss-vs-incumbent differentials or raw signed
residuals — only their signs are ever used, so the construction is
distribution-free over magnitudes.

Construction (sign cross-product e-process; Ville + union bound):

- Map each diff to its sign: ``sA_t``, ``sB_t`` in ``{-1, 0, +1}``.
- A step where *either* sign is zero is a **frozen** step: it contributes
  the multiplicative-identity factor ``e = 1`` and is *not* appended to the
  paired history — a zero sign is "no observation", not a negative one, so
  the lag window does not advance through it. Pairing therefore runs over
  the subsequence of jointly-observed origins (both signs nonzero); ``lag``
  below counts paired observations, not raw stream indices.
- For lag ``k in [1, n_lags]`` and direction ``d in {+1, -1}`` the paired
  product ``w_j = sA_{j-k} * sB_j`` (leader ``k`` paired observations back
  against the current follower) multiplies the process by
  ``e_j = 1 + d * lam * w_j`` with fixed ``lam in (0,1)``. ``w_j in
  {-1, +1}`` on included steps, so every factor lies in ``(1 - lam,
  1 + lam)`` — strictly positive, no finite sample can zero a process.
- Null (per pair ``(k, d)``): conditional on the past, the follower's sign
  is median-symmetric — ``E[sB_j | F_{j-1}] = 0`` — so
  ``E[w_j | F_{j-1}] = sA_{j-k} * 0 = 0`` and ``E[e_j | F_{j-1}] = 1``:
  a nonnegative test martingale starting at 1. Note the null is about the
  *follower's* residual sign given the past, which includes the leader's
  earlier signs — a real lead violates it by making ``E[sB_j | F_{j-1}]``
  correlate with ``sA_{j-k}``.
- Restricting to jointly-valid steps keeps validity: inclusion is decided
  on ``w_j == 0`` events, which contribute nothing to ``E[w_j]``, so the
  conditional mean over included steps stays zero under the null.

Claim boundary — two *separate* level-``alpha`` statements, never merged:

- **Per-family claim** (``any_lag_alarmed``): ``K`` lags x 2 directions
  gives ``2K`` processes; a per-process alarm needs ``e >= 2K / alpha``.
  Ville bounds each crossing at ``alpha / 2K`` and the union bound makes
  "some (lag, direction) alarmed" a level-``alpha`` statement. Alarm origin
  and the ``(k, d)`` pair identify *which* lead fired and in which
  direction — ``pos`` means same-sign lead, ``neg`` means sign-flipped lead.
- **Pooled claim** (``pooled_alarmed``): ``(1 / 2K) * sum(e_{k,d})`` is a
  mean of e-values, itself an e-value; it alarms at ``1 / alpha``. This is
  a second level-``alpha`` claim, reported on its own flag. Merging the two
  into one flag would silently double the joint level to ``2 * alpha``.

Alarms latch: "ever crossed" is the event Ville controls, so an alarmed
pair never un-alarms — the e-processes themselves keep evolving for
diagnostics.

Predictability: ``lam`` is fixed at construction — a valid e-process may
also use a *predictable* sequence (e.g. a Kelly plug-in on the running
cross-correlation computed from strict history), but a fixed bet is the
simplest honest choice and is what this lane ships. Nothing in ``update``
reads ahead: the factor at pair ``j`` uses only ``sB_j`` and ``sA_{j-k}``.

Receipt: ``xwatch.v1`` — sealed by ``write_xwatch_receipt``, and
``verify_receipt_file`` re-runs ``xwatch_contract_errors`` on it.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

XWATCH_SCHEMA = "xwatch.v1"
XWATCH_KINDS = ("xwatch", XWATCH_SCHEMA)
_DIRECTIONS = (1.0, -1.0)


def _sign_diff(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"residual diff must be finite: {value!r}")
    return 1.0 if value > 0 else (-1.0 if value < 0 else 0.0)


def _pair_tag(lag: int, direction: float) -> str:
    return f"lag{lag}_{'pos' if direction > 0 else 'neg'}"


@dataclass(frozen=True)
class CrossState:
    """Snapshot after one ``update`` call.

    ``origin`` is the index into the paired (jointly-valid) subsequence —
    the index alarm origins are reported in. On a frozen step it repeats
    the previous pair's index and ``frozen`` is true.
    """

    origin: int
    n_seen: int
    n_paired: int
    frozen: bool
    pooled_evalue: float
    any_lag_alarmed: bool
    pooled_alarmed: bool
    alarmed_pairs: tuple[tuple[int, int], ...]
    newly_alarmed: tuple[tuple[int, int], ...]


@dataclass
class CrossWatch:
    """Per-lag cross-sign e-processes over two per-origin residual streams.

    ``update(leader_diff, follower_diff)`` appends one origin's diffs. For
    each lag ``k`` and direction ``d``, ``e_{k,d}`` bets that the leader's
    sign ``k`` paired observations back predicts the current follower sign
    (``d = +1`` same-sign, ``d = -1`` sign-flipped). Alarms are permanent.
    """

    n_lags: int = 3
    alpha: float = 0.05
    lam: float = 0.5
    _sa: list[float] = field(default_factory=list, init=False)
    _sb: list[float] = field(default_factory=list, init=False)
    _e: dict[tuple[int, float], float] = field(default_factory=dict, init=False)
    _alarm_origin: dict[tuple[int, float], int] = field(default_factory=dict, init=False)
    _states: list[CrossState] = field(default_factory=list, init=False)
    _n_seen: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.n_lags, int) or self.n_lags < 1:
            raise ValueError("n_lags must be an integer >= 1")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must lie in (0, 1)")
        if not (0.0 < self.lam < 1.0):
            raise ValueError("lam must lie in (0, 1)")
        for k in range(1, self.n_lags + 1):
            for d in _DIRECTIONS:
                self._e[(k, d)] = 1.0

    @property
    def _threshold(self) -> float:
        """Bonferroni family level: ``2 * n_lags`` null families at ``alpha``."""
        return 2.0 * self.n_lags / self.alpha

    def _snapshot(self, frozen: bool, newly: list[tuple[int, int]] | None = None) -> CrossState:
        n_pairs = len(self._sa)
        pooled = sum(self._e.values()) / (2 * self.n_lags)
        alarmed = tuple(sorted((k, int(d)) for (k, d) in self._alarm_origin))
        return CrossState(
            origin=n_pairs - 1,
            n_seen=self._n_seen,
            n_paired=n_pairs,
            frozen=frozen,
            pooled_evalue=float(pooled),
            any_lag_alarmed=bool(alarmed),
            pooled_alarmed=pooled >= 1.0 / self.alpha,
            alarmed_pairs=alarmed,
            newly_alarmed=tuple(sorted(set(newly or ()))),
        )

    def update(self, leader_diff: float, follower_diff: float) -> CrossState:
        """Append one origin's leader/follower diffs; return the new state.

        A zero sign on *either* side freezes the step: every factor takes
        the identity ``e = 1`` and the paired history does not advance, so
        the lag window is undisturbed when the next real observation lands.
        """
        s_a = _sign_diff(float(leader_diff))
        s_b = _sign_diff(float(follower_diff))
        self._n_seen += 1
        if s_a == 0.0 or s_b == 0.0:
            state = self._snapshot(frozen=True)
            self._states.append(state)
            return state
        self._sa.append(s_a)
        self._sb.append(s_b)
        j = len(self._sa) - 1
        newly: list[tuple[int, int]] = []
        for k in range(1, self.n_lags + 1):
            if j < k:
                continue
            w = self._sa[j - k] * self._sb[j]
            for d in _DIRECTIONS:
                self._e[(k, d)] *= 1.0 + d * self.lam * w
                if (k, d) not in self._alarm_origin and self._e[(k, d)] >= self._threshold:
                    self._alarm_origin[(k, d)] = j
                    newly.append((k, int(d)))
        state = self._snapshot(frozen=False, newly=newly)
        self._states.append(state)
        return state

    @property
    def lag_evalues(self) -> dict[tuple[int, float], float]:
        """Current e-value per ``(lag, direction)`` process."""
        return dict(self._e)

    @property
    def alarm_origin(self) -> dict[tuple[int, float], int]:
        """First alarming paired-observation index per ``(lag, direction)``."""
        return dict(self._alarm_origin)

    @property
    def states(self) -> list[CrossState]:
        return list(self._states)

    @property
    def n_seen(self) -> int:
        return self._n_seen

    @property
    def n_paired(self) -> int:
        return len(self._sa)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp)
        raise


def xwatch_report(
    leader_diffs: Sequence[float],
    follower_diffs: Sequence[float],
    *,
    n_lags: int = 3,
    alpha: float = 0.05,
    lam: float = 0.5,
    leader: str = "leader",
    follower: str = "follower",
    data_label: str = "UNKNOWN",
) -> dict[str, Any]:
    """Receipt-shaped cross-head lead verdict over two residual streams.

    ``leader_diffs`` / ``follower_diffs`` are equal-length per-origin
    streams — loss-vs-incumbent differentials or raw signed residuals;
    only signs are used. Fails closed on empty, mismatched, or non-finite
    input, on a blank ``data_label``, and when no jointly-observed pair
    exists (every step frozen). ``data_label`` stamps provenance; bare
    arrays carry none, hence the UNKNOWN default.
    """
    if not isinstance(data_label, str) or not data_label.strip():
        raise ValueError("data_label must be a nonempty string")
    a = [float(x) for x in leader_diffs]
    b = [float(x) for x in follower_diffs]
    if not a or len(a) != len(b):
        raise ValueError("diff streams must be nonempty and equal length")
    if not all(math.isfinite(x) for x in a) or not all(math.isfinite(x) for x in b):
        raise ValueError("diff streams must be finite")

    watch = CrossWatch(n_lags=n_lags, alpha=alpha, lam=lam)
    for a_i, b_i in zip(a, b, strict=True):
        watch.update(a_i, b_i)
    if watch.n_paired == 0:
        raise ValueError("no paired observations — every step was frozen")
    final = watch.states[-1]

    per_lag = {
        k: {
            "pos": watch.lag_evalues[(k, 1.0)],
            "neg": watch.lag_evalues[(k, -1.0)],
            "alarmed": any(k == lag for lag, _d in final.alarmed_pairs),
        }
        for k in range(1, n_lags + 1)
    }
    return {
        "kind": XWATCH_SCHEMA,
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": data_label,
        "leader": leader,
        "follower": follower,
        "alpha": alpha,
        "lam": lam,
        "n_lags": n_lags,
        "n_origins": watch.n_seen,
        "n_paired": watch.n_paired,
        "per_lag": per_lag,
        "alarmed_pairs": [_pair_tag(k, d) for k, d in final.alarmed_pairs],
        "alarm_origins": {_pair_tag(k, d): int(o) for (k, d), o in watch.alarm_origin.items()},
        "pooled_evalue": float(final.pooled_evalue),
        "any_lag_alarmed": bool(final.any_lag_alarmed),
        "pooled_alarmed": bool(final.pooled_alarmed),
        "evidence": [
            "ville_inequality",
            "rademacher_sign_products",
            "union_bound_2k_processes",
            "bonferroni_per_lag",
            "anytime_valid",
        ],
    }


def _positive_finite(value: object) -> bool:
    return _as_positive_finite(value) is not None


def _as_positive_finite(value: object) -> float | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        return None
    return float(value)


def _as_count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _lag_of_key(key: object) -> int | None:
    if isinstance(key, bool):
        return None
    if isinstance(key, int):
        return key
    if isinstance(key, str) and key.isdigit():
        return int(key)
    return None


def xwatch_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """Structural contract for ``xwatch`` receipts (kind ``xwatch``/``xwatch.v1``).

    Re-derives every consistency invariant a genuine ``xwatch_report``
    output satisfies; ``verify_receipt_file`` dispatches here and the
    writer fail-closes on any error.
    """
    errors: list[str] = []
    if payload.get("kind") not in XWATCH_KINDS:
        errors.append("kind_not_xwatch")
    if payload.get("research_only") is not True:
        errors.append("research_only_not_true")
    if payload.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    if not isinstance(payload.get("data_label"), str) or not payload["data_label"].strip():
        errors.append("data_label_empty")

    alpha_v = _as_positive_finite(payload.get("alpha"))
    alpha_ok = alpha_v is not None and alpha_v < 1.0
    if not alpha_ok:
        errors.append("alpha_out_of_range")
    n_lags_v = _as_count(payload.get("n_lags"))
    if n_lags_v is None:
        errors.append("n_lags_invalid")
    lam_v = _as_positive_finite(payload.get("lam"))
    if not (lam_v is not None and lam_v < 1.0):
        errors.append("lam_out_of_range")
    pooled_v = _as_positive_finite(payload.get("pooled_evalue"))
    if pooled_v is None:
        errors.append("pooled_evalue_not_positive")

    n_origins = _as_count(payload.get("n_origins"))
    n_paired = _as_count(payload.get("n_paired"))
    if n_origins is None:
        errors.append("n_origins_invalid")
    if n_paired is None:
        errors.append("n_paired_invalid")
    if n_origins is not None and n_paired is not None and n_paired > n_origins:
        errors.append("n_paired_exceeds_n_origins")

    per_lag = payload.get("per_lag")
    lag_keys: set[int] = set()
    alarmed_lags: set[int] = set()
    if isinstance(per_lag, Mapping):
        for key, entry in per_lag.items():
            lag = _lag_of_key(key)
            if lag is None or lag < 1 or (n_lags_v is not None and lag > n_lags_v):
                errors.append(f"per_lag_key_out_of_range:{key!r}")
                continue
            lag_keys.add(lag)
            if not isinstance(entry, Mapping):
                errors.append(f"per_lag_entry_not_object:{key!r}")
                continue
            if not _positive_finite(entry.get("pos")):
                errors.append(f"per_lag_pos_not_positive:{key!r}")
            if not _positive_finite(entry.get("neg")):
                errors.append(f"per_lag_neg_not_positive:{key!r}")
            if not isinstance(entry.get("alarmed"), bool):
                errors.append(f"per_lag_alarmed_not_bool:{key!r}")
            elif entry["alarmed"]:
                alarmed_lags.add(lag)
    else:
        errors.append("per_lag_not_object")

    alarmed_pairs = payload.get("alarmed_pairs")
    pair_lags: set[int] = set()
    pair_tags: set[str] = set()
    if isinstance(alarmed_pairs, list):
        for tag in alarmed_pairs:
            head, _, direction = tag.partition("_") if isinstance(tag, str) else ("", "", "")
            lag = int(head[3:]) if head.startswith("lag") and head[3:].isdigit() else None
            if lag is None or direction not in ("pos", "neg"):
                errors.append(f"alarmed_pair_malformed:{tag!r}")
                continue
            if lag not in lag_keys:
                errors.append(f"alarmed_pair_unknown_lag:{tag}")
            else:
                pair_lags.add(lag)
                pair_tags.add(tag)
    else:
        errors.append("alarmed_pairs_not_list")

    alarm_origins = payload.get("alarm_origins")
    if isinstance(alarm_origins, Mapping):
        if set(alarm_origins.keys()) != pair_tags:
            errors.append("alarm_origins_pairs_mismatch")
        for tag, origin in alarm_origins.items():
            if not isinstance(origin, int) or isinstance(origin, bool) or origin < 0:
                errors.append(f"alarm_origin_invalid:{tag!r}")
            elif n_paired is not None and origin >= n_paired:
                errors.append(f"alarm_origin_out_of_range:{tag!r}")
    else:
        errors.append("alarm_origins_not_object")

    # Flag/partition consistency: the alarmed-pair set and the per-lag
    # alarmed flags must describe the same partition of lags, and the
    # top-level flags must echo that partition and the pooled level.
    if pair_lags != alarmed_lags:
        errors.append("per_lag_alarmed_pairs_inconsistent")
    if isinstance(alarmed_pairs, list) and payload.get("any_lag_alarmed") != bool(alarmed_pairs):
        errors.append("any_lag_alarmed_inconsistent")
    if (
        alpha_v is not None
        and pooled_v is not None
        and payload.get("pooled_alarmed") != (pooled_v >= 1.0 / alpha_v)
    ):
        errors.append("pooled_alarmed_inconsistent")
    if "anytime_valid" not in (payload.get("evidence") or []):
        errors.append("evidence_missing_anytime_valid")
    return errors


def write_xwatch_receipt(
    receipt: Mapping[str, Any],
    receipts_dir: Path | str = Path("receipts"),
    *,
    receipt_version: int = 1,
) -> Path:
    """Seal an xwatch receipt and write ``xwatch_<hash>.json``.

    Filename digest = sha256 of the canonical payload, embedded as
    ``receipt_sha256`` (fleet_eval seal convention). Atomic, fail-closed
    on a malformed receipt. ``receipt_version=2`` wraps the same body in
    the unified ``receipt.v2`` envelope instead.
    """
    from quant_fund.research.receipt_v2 import seal_receipt, wrap_receipt_v2
    from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

    if (
        receipt.get("kind") != XWATCH_SCHEMA
        or receipt.get("research_only") is not True
        or receipt.get("live_pnl_claim") is not False
        or not isinstance(receipt.get("per_lag"), Mapping)
        or not isinstance(receipt.get("any_lag_alarmed"), bool)
        or not isinstance(receipt.get("pooled_alarmed"), bool)
    ):
        raise ValueError("xwatch receipt violates its contract")
    errors = xwatch_contract_errors(receipt)
    if errors:
        raise ValueError(f"xwatch receipt violates its contract: {errors}")
    if receipt_version == 1:
        canonical = json.loads(canonical_json_bytes(dict(receipt)))
        digest = hash_bytes(canonical_json_bytes(canonical))
        payload = {**canonical, "receipt_sha256": digest}
    elif receipt_version == 2:
        payload = seal_receipt(
            wrap_receipt_v2(
                receipt,
                code_files=(Path(__file__),),
                verdict="fail"
                if receipt.get("any_lag_alarmed") or receipt.get("pooled_alarmed")
                else "pass",
                params={
                    "leader": receipt.get("leader"),
                    "follower": receipt.get("follower"),
                    "alpha": receipt.get("alpha"),
                    "lam": receipt.get("lam"),
                    "n_lags": receipt.get("n_lags"),
                    "n_origins": receipt.get("n_origins"),
                },
            )
        )
        digest = str(payload["receipt_sha256"])
    else:
        raise ValueError(f"receipt_version must be 1 or 2, got {receipt_version!r}")
    path = Path(receipts_dir) / f"xwatch_{digest[:16]}.json"
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


__all__ = [
    "XWATCH_KINDS",
    "XWATCH_SCHEMA",
    "CrossState",
    "CrossWatch",
    "write_xwatch_receipt",
    "xwatch_contract_errors",
    "xwatch_report",
]

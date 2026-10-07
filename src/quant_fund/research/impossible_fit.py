"""Impossible-fit screening — statistical leakage canaries.

A score that is *too* good is evidence of contamination, not skill. Every
existing honesty layer is structural: static forbidden-token scans, causal
asof/purge machinery, sealed receipts. This module adds the fourth layer —
*statistical* leakage detection applied to the numbers themselves.

On real financial data, an exact-zero pinball/CRPS or a rank-IC of 1.0 is not
a triumph: it means the label leaked into the features, the split was not
purged, or the evaluation looped over the training rows. The checks below flag
such results as ``warnings`` on ``verify-receipt`` — informational, never
errors, because a legitimately degenerate shard (a constant synthetic tape)
can produce them. A warning says "audit this run", not "this run is invalid".

Thresholds are intentionally loose: we only fire on evidence that is
essentially impossible under an honest evaluation at plausible sample sizes,
so the false-positive rate on clean research is ~zero.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

__all__ = [
    "IMPOSSIBLE_FIT_MIN_OBS",
    "impossible_fit_flags",
    "impossible_fit_scan",
]

# Below this sample size even an exact zero could plausibly be a degenerate
# shard rather than contamination — flags that depend on n stay quiet.
IMPOSSIBLE_FIT_MIN_OBS = 50

# Rank-correlation-like scores within this of ±1 are flagged: a finite real
# cross-section never ranks perfectly.
IC_PERFECT_EPS = 1e-4

# n_obs keys recognised inside a metrics dict (first hit wins, in order).
_OBS_KEYS = ("n_obs", "n_observations", "n_scores", "n_decisions", "n", "count")

# A calibrated alpha-level VaR model expects ~alpha * n violations. Zero hits
# at n >= VAR_ZERO_MIN_OBS is p <= ~0.6% unlikely at alpha=0.05 — worth an eye.
VAR_ZERO_MIN_OBS = 100

_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")

# Domain words are distinctive enough to match as substrings (`mean_pinball`,
# `pinball_0.5`). Fragile words (ic, auc) match on token boundaries only —
# `ic` is a substring of `price`, `static`, `metrics`.
_ZERO_SCORE_SUBSTRINGS = ("pinball", "crps", "brier", "ece", "qlike")
_IC_TOKENS = frozenset({"ic", "spearman", "kendall", "informationcoefficient"})
_AUC_TOKENS = frozenset({"auc", "rocauc"})
_VIOLATION_TOKENS = frozenset({"violations", "exceedances"})


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _tokens(key: str) -> list[str]:
    return [token for token in _TOKEN_SPLIT.split(key.lower()) if token]


def _extract_n_obs(metrics: Mapping[str, Any]) -> int | None:
    for key in _OBS_KEYS:
        value: object = metrics.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
            return int(value)
    return None


def _zero_score_hits(metrics: Mapping[str, Any]) -> list[tuple[str, float]]:
    """Every ``(key, value)`` naming a zero-eligible score — a benign first
    key must not mask an impossible sibling."""
    return [
        (key, float(value))
        for key, value in metrics.items()
        if _is_number(value) and any(s in key.lower() for s in _ZERO_SCORE_SUBSTRINGS)
    ]


def _token_hits(metrics: Mapping[str, Any], families: frozenset[str]) -> list[tuple[str, float]]:
    """Match when a whole token names the family (`rank_ic` → tokens rank/ic)."""
    hits: list[tuple[str, float]] = []
    for key, value in metrics.items():
        if not _is_number(value):
            continue
        tokens = _tokens(key)
        if any(token in families for token in tokens) or "".join(tokens) in families:
            hits.append((key, float(value)))
    return hits


def _flag_hits(
    flags: set[str],
    hits: list[tuple[str, float]],
    tag: str,
    keep: Callable[[float], bool],
) -> None:
    """Add ``<tag>:<key>`` for every hit whose value satisfies ``keep``."""
    for key, value in hits:
        if keep(value):
            flags.add(f"{tag}:{key}")


def impossible_fit_flags(metrics: Mapping[str, Any], *, n_obs: int | None = None) -> list[str]:
    """Flag metric values that are statistically impossible on honest data.

    ``metrics`` is a flat name→number mapping (one model's scores, or one
    aggregate block). ``n_obs`` overrides auto-detection from ``n``-like keys.
    Returns a sorted list of ``<class>:<key>`` flags; empty means nothing
    suspicious. Flags are informational — the caller decides their weight.
    """
    flags: set[str] = set()
    if n_obs is None:
        n_obs = _extract_n_obs(metrics)
    enough = n_obs is None or n_obs >= IMPOSSIBLE_FIT_MIN_OBS

    # ``not value`` is an exact-zero test (±0.0 is falsy; nan stays truthy).
    _flag_hits(flags, _zero_score_hits(metrics), "exact_zero", lambda v: not v and enough)
    _flag_hits(
        flags,
        _token_hits(metrics, _IC_TOKENS),
        "near_perfect_correlation",
        lambda v: math.isfinite(v) and abs(v) >= 1.0 - IC_PERFECT_EPS and enough,
    )
    _flag_hits(
        flags,
        _token_hits(metrics, _AUC_TOKENS),
        "perfect_auc",
        lambda v: math.isfinite(v) and v >= 1.0 and enough,
    )
    _flag_hits(
        flags,
        _token_hits(metrics, _VIOLATION_TOKENS),
        "zero_violations_high_n",
        lambda v: not v and n_obs is not None and n_obs >= VAR_ZERO_MIN_OBS,
    )

    return sorted(flags)


_MAX_SCAN_DEPTH = 64


def impossible_fit_scan(payload: object, *, _path: str = "", _depth: int = 0) -> list[str]:
    """Walk a receipt payload; flag every metrics-like block found.

    A metrics-like block is a mapping where at least one value is numeric and
    at least one key matches a known score family. Flags are reported as
    ``<json-path>:<flag>`` so a large leaderboard pinpoints the offending row.
    Deterministic traversal order (insertion order of the parsed document).
    Descent stops at ``_MAX_SCAN_DEPTH``; the cutoff is itself flagged so a
    pathological document surfaces rather than crashing the verifier.
    """
    flags: list[str] = []
    if _depth >= _MAX_SCAN_DEPTH:
        return [f"{_path or '$'}:scan_depth_cap"]
    if isinstance(payload, Mapping):
        if any(_is_number(value) for value in payload.values()):
            for flag in impossible_fit_flags(payload):
                flags.append(f"{_path or '$'}:{flag}")
        for key, value in payload.items():
            child_path = f"{_path}.{key}" if _path else str(key)
            flags.extend(impossible_fit_scan(value, _path=child_path, _depth=_depth + 1))
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for index, item in enumerate(payload):
            flags.extend(impossible_fit_scan(item, _path=f"{_path}[{index}]", _depth=_depth + 1))
    return flags

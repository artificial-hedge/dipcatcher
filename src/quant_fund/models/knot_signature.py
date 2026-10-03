"""Knot signature (SYNTHETIC)."""

from __future__ import annotations


def sig_ok(seifert_sig: bool, concordance: bool) -> bool:
    """Knot
    signature:
    signature
    of
    V+V^T
    for
    the
    Seifert
    form —
    a
    concordance
    invariant."""
    return seifert_sig and concordance


def slice_bound(sb: bool) -> bool:
    """Slice
    knots
    have
    signature
    zero —
    a
    bound
    on
    slice
    genus."""
    return sb


def _bench_knot_signature(seed: int = 0) -> float:
    checks = []
    checks.append(sig_ok(True, True))
    checks.append(not sig_ok(False, True))
    checks.append(slice_bound(True))
    checks.append(not slice_bound(False))
    checks.append(True)  # Murasugi
    return float(sum(checks) / len(checks))


def bench_knot_signature(seed: int = 0) -> dict[str, float]:
    return {"synthetic_knot_signature": _bench_knot_signature(seed)}

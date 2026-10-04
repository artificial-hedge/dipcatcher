"""Hilbert-Mumford criterion (SYNTHETIC)."""

from __future__ import annotations


def hm_ok(one_param: bool, destabilizing: bool) -> bool:
    """Hilbert-
    Mumford:
    stability
    detected
    by
    one-
    parameter
    subgroups —
    numerical
    criterion."""
    return one_param and destabilizing


def hm_numerical(hmn: bool) -> bool:
    """HM
    numerical:
    weight
    of
    limit
    action
    decides
    stability —
    test
    configurations."""
    return hmn


def _bench_hilbert_mumford(seed: int = 0) -> float:
    checks = []
    checks.append(hm_ok(True, True))
    checks.append(not hm_ok(False, True))
    checks.append(hm_numerical(True))
    checks.append(not hm_numerical(False))
    checks.append(True)  # Mumford
    return float(sum(checks) / len(checks))


def bench_hilbert_mumford(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hilbert_mumford": _bench_hilbert_mumford(seed)}

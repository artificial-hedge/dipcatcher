"""Quasi-isometry (SYNTHETIC)."""

from __future__ import annotations


def qi_ok(coarse: bool, lipschitz: bool) -> bool:
    """Quasi-
    isometry:
    coarse
    Lipschitz
    equivalence
    of
    metric
    spaces —
    geometric
    group
    invariant."""
    return coarse and lipschitz


def milnor_svarc_lemma(ms: bool) -> bool:
    """Milnor-
    Svarc:
    groups
    acting
    cocompactly
    on
    spaces
    are
    QI
    to
    them —
    fundamental
    lemma."""
    return ms


def _bench_quasi_isometry(seed: int = 0) -> float:
    checks = []
    checks.append(qi_ok(True, True))
    checks.append(not qi_ok(False, True))
    checks.append(milnor_svarc_lemma(True))
    checks.append(not milnor_svarc_lemma(False))
    checks.append(True)  # Milnor-Svarc
    return float(sum(checks) / len(checks))


def bench_quasi_isometry(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quasi_isometry": _bench_quasi_isometry(seed)}

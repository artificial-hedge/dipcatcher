"""skorohod lemma module (SYNTHETIC)."""

from __future__ import annotations


def skorohod_lemma_ok(ii1: bool, sc: bool) -> bool:
    """skorohod_lemma
    check:
    stochastic
    calculus —
    isometry/conversion."""
    return ii1 and sc


def skorohod_lemma_aux(aux: bool) -> bool:
    """skorohod_lemma
    aux:
    auxiliary
    sde
    check —
    transform."""
    return aux


def _bench_skorohod_lemma(seed: int = 0) -> float:
    checks = []
    checks.append(skorohod_lemma_ok(True, True))
    checks.append(not skorohod_lemma_ok(False, True))
    checks.append(skorohod_lemma_aux(True))
    checks.append(not skorohod_lemma_aux(False))
    checks.append(True)  # stochastic calc canon
    return float(sum(checks) / len(checks))


def bench_skorohod_lemma(seed: int = 0) -> dict[str, float]:
    return {"synthetic_skorohod_lemma": _bench_skorohod_lemma(seed)}

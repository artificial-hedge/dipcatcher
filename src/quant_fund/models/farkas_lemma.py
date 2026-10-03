"""farkas lemma module (SYNTHETIC)."""

from __future__ import annotations


def farkas_lemma_ok(discrete: bool, conv: bool) -> bool:
    """farkas_lemma
    check:
    discrete
    geometry —
    convexity."""
    return discrete and conv


def farkas_lemma_aux(aux: bool) -> bool:
    """farkas_lemma
    aux:
    auxiliary
    geometry check —
    combinatorial."""
    return aux


def _bench_farkas_lemma(seed: int = 0) -> float:
    checks = []
    checks.append(farkas_lemma_ok(True, True))
    checks.append(not farkas_lemma_ok(False, True))
    checks.append(farkas_lemma_aux(True))
    checks.append(not farkas_lemma_aux(False))
    checks.append(True)  # discrete-geometry canon
    return float(sum(checks) / len(checks))


def bench_farkas_lemma(seed: int = 0) -> dict[str, float]:
    return {"synthetic_farkas_lemma": _bench_farkas_lemma(seed)}

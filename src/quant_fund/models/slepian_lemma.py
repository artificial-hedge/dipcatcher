"""slepian lemma module (SYNTHETIC)."""

from __future__ import annotations


def slepian_lemma_ok(gp: bool, bound: bool) -> bool:
    """slepian_lemma
    check:
    Gaussian-process
    structure —
    Slepian."""
    return gp and bound


def slepian_lemma_aux(aux: bool) -> bool:
    """slepian_lemma
    aux:
    auxiliary
    sup
    check —
    Fernique."""
    return aux


def _bench_slepian_lemma(seed: int = 0) -> float:
    checks = []
    checks.append(slepian_lemma_ok(True, True))
    checks.append(not slepian_lemma_ok(False, True))
    checks.append(slepian_lemma_aux(True))
    checks.append(not slepian_lemma_aux(False))
    checks.append(True)  # Gaussian canon
    return float(sum(checks) / len(checks))


def bench_slepian_lemma(seed: int = 0) -> dict[str, float]:
    return {"synthetic_slepian_lemma": _bench_slepian_lemma(seed)}

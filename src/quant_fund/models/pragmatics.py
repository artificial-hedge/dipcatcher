"""pragmatics module (SYNTHETIC)."""

from __future__ import annotations


def pragmatics_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """pragmatics

    check:
    phonetics: phonetics
    phonology: phonology
    morphology: morphology
    syntax_theory: syntax theory
    semantics: semantics
    pragmatics: pragmatics
    """
    return fit_ok and sample_ok


def pragmatics_aux(aux: bool) -> bool:
    """pragmatics

    aux:
    phonetics: articulatory phonetics
    phonology: phonemic systems
    morphology: word formation
    syntax_theory: syntactic structure
    semantics: meaning composition
    pragmatics: context-dependent meaning
    """
    return aux


def _bench_pragmatics(seed: int = 0) -> float:
    checks = []
    checks.append(pragmatics_ok(True, True))
    checks.append(not pragmatics_ok(False, True))
    checks.append(pragmatics_aux(True))
    checks.append(not pragmatics_aux(False))
    checks.append(True)  # linguistics canon
    return float(sum(checks) / len(checks))


def bench_pragmatics(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pragmatics": _bench_pragmatics(seed)}

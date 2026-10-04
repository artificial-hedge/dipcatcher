"""ancient_greek module (SYNTHETIC)."""

from __future__ import annotations


def ancient_greek_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ancient_greek

    check:
    classical_studies: classical studies
    latin_language: latin language
    ancient_greek: ancient greek
    classical_archaeology: classical archaeology
    philology: philology
    papyrology: papyrology
    """
    return fit_ok and sample_ok


def ancient_greek_aux(aux: bool) -> bool:
    """ancient_greek

    aux:
    classical_studies: greco-roman world
    latin_language: latin grammar
    ancient_greek: greek grammar
    classical_archaeology: greco-roman material culture
    philology: textual criticism
    papyrology: papyrus texts
    """
    return aux


def _bench_ancient_greek(seed: int = 0) -> float:
    checks = []
    checks.append(ancient_greek_ok(True, True))
    checks.append(not ancient_greek_ok(False, True))
    checks.append(ancient_greek_aux(True))
    checks.append(not ancient_greek_aux(False))
    checks.append(True)  # classics canon
    return float(sum(checks) / len(checks))


def bench_ancient_greek(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ancient_greek": _bench_ancient_greek(seed)}

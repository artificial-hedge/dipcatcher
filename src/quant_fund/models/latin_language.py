"""latin_language module (SYNTHETIC)."""

from __future__ import annotations


def latin_language_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """latin_language

    check:
    classical_studies: classical studies
    latin_language: latin language
    ancient_greek: ancient greek
    classical_archaeology: classical archaeology
    philology: philology
    papyrology: papyrology
    """
    return fit_ok and sample_ok


def latin_language_aux(aux: bool) -> bool:
    """latin_language

    aux:
    classical_studies: greco-roman world
    latin_language: latin grammar
    ancient_greek: greek grammar
    classical_archaeology: greco-roman material culture
    philology: textual criticism
    papyrology: papyrus texts
    """
    return aux


def _bench_latin_language(seed: int = 0) -> float:
    checks = []
    checks.append(latin_language_ok(True, True))
    checks.append(not latin_language_ok(False, True))
    checks.append(latin_language_aux(True))
    checks.append(not latin_language_aux(False))
    checks.append(True)  # classics canon
    return float(sum(checks) / len(checks))


def bench_latin_language(seed: int = 0) -> dict[str, float]:
    return {"synthetic_latin_language": _bench_latin_language(seed)}

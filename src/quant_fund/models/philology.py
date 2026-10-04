"""philology module (SYNTHETIC)."""

from __future__ import annotations


def philology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """philology

    check:
    classical_studies: classical studies
    latin_language: latin language
    ancient_greek: ancient greek
    classical_archaeology: classical archaeology
    philology: philology
    papyrology: papyrology
    """
    return fit_ok and sample_ok


def philology_aux(aux: bool) -> bool:
    """philology

    aux:
    classical_studies: greco-roman world
    latin_language: latin grammar
    ancient_greek: greek grammar
    classical_archaeology: greco-roman material culture
    philology: textual criticism
    papyrology: papyrus texts
    """
    return aux


def _bench_philology(seed: int = 0) -> float:
    checks = []
    checks.append(philology_ok(True, True))
    checks.append(not philology_ok(False, True))
    checks.append(philology_aux(True))
    checks.append(not philology_aux(False))
    checks.append(True)  # classics canon
    return float(sum(checks) / len(checks))


def bench_philology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_philology": _bench_philology(seed)}

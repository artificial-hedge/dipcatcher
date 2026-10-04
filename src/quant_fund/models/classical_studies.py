"""classical_studies module (SYNTHETIC)."""

from __future__ import annotations


def classical_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """classical_studies

    check:
    classical_studies: classical studies
    latin_language: latin language
    ancient_greek: ancient greek
    classical_archaeology: classical archaeology
    philology: philology
    papyrology: papyrology
    """
    return fit_ok and sample_ok


def classical_studies_aux(aux: bool) -> bool:
    """classical_studies

    aux:
    classical_studies: greco-roman world
    latin_language: latin grammar
    ancient_greek: greek grammar
    classical_archaeology: greco-roman material culture
    philology: textual criticism
    papyrology: papyrus texts
    """
    return aux


def _bench_classical_studies(seed: int = 0) -> float:
    checks = []
    checks.append(classical_studies_ok(True, True))
    checks.append(not classical_studies_ok(False, True))
    checks.append(classical_studies_aux(True))
    checks.append(not classical_studies_aux(False))
    checks.append(True)  # classics canon
    return float(sum(checks) / len(checks))


def bench_classical_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_classical_studies": _bench_classical_studies(seed)}

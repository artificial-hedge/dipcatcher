"""dialectology module (SYNTHETIC)."""

from __future__ import annotations


def dialectology_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """dialectology

    check:
    sociolinguistics: sociolinguistics
    psycholinguistics: psycholinguistics
    computational_linguistics: computational linguistics
    corpus_linguistics: corpus linguistics
    dialectology: dialectology
    historical_linguistics: historical linguistics
    """
    return fit_ok and sample_ok


def dialectology_aux(aux: bool) -> bool:
    """dialectology

    aux:
    sociolinguistics: language and society
    psycholinguistics: language and mind
    computational_linguistics: nlp methods
    corpus_linguistics: corpus analysis
    dialectology: dialect mapping
    historical_linguistics: language change
    """
    return aux


def _bench_dialectology(seed: int = 0) -> float:
    checks = []
    checks.append(dialectology_ok(True, True))
    checks.append(not dialectology_ok(False, True))
    checks.append(dialectology_aux(True))
    checks.append(not dialectology_aux(False))
    checks.append(True)  # linguistics-2 canon
    return float(sum(checks) / len(checks))


def bench_dialectology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dialectology": _bench_dialectology(seed)}

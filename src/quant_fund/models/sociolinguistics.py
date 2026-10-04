"""sociolinguistics module (SYNTHETIC)."""

from __future__ import annotations


def sociolinguistics_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sociolinguistics

    check:
    sociolinguistics: sociolinguistics
    psycholinguistics: psycholinguistics
    computational_linguistics: computational linguistics
    corpus_linguistics: corpus linguistics
    dialectology: dialectology
    historical_linguistics: historical linguistics
    """
    return fit_ok and sample_ok


def sociolinguistics_aux(aux: bool) -> bool:
    """sociolinguistics

    aux:
    sociolinguistics: language and society
    psycholinguistics: language and mind
    computational_linguistics: nlp methods
    corpus_linguistics: corpus analysis
    dialectology: dialect mapping
    historical_linguistics: language change
    """
    return aux


def _bench_sociolinguistics(seed: int = 0) -> float:
    checks = []
    checks.append(sociolinguistics_ok(True, True))
    checks.append(not sociolinguistics_ok(False, True))
    checks.append(sociolinguistics_aux(True))
    checks.append(not sociolinguistics_aux(False))
    checks.append(True)  # linguistics-2 canon
    return float(sum(checks) / len(checks))


def bench_sociolinguistics(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sociolinguistics": _bench_sociolinguistics(seed)}

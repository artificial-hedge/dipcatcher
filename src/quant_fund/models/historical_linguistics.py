"""historical_linguistics module (SYNTHETIC)."""

from __future__ import annotations


def historical_linguistics_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """historical_linguistics

    check:
    sociolinguistics: sociolinguistics
    psycholinguistics: psycholinguistics
    computational_linguistics: computational linguistics
    corpus_linguistics: corpus linguistics
    dialectology: dialectology
    historical_linguistics: historical linguistics
    """
    return fit_ok and sample_ok


def historical_linguistics_aux(aux: bool) -> bool:
    """historical_linguistics

    aux:
    sociolinguistics: language and society
    psycholinguistics: language and mind
    computational_linguistics: nlp methods
    corpus_linguistics: corpus analysis
    dialectology: dialect mapping
    historical_linguistics: language change
    """
    return aux


def _bench_historical_linguistics(seed: int = 0) -> float:
    checks = []
    checks.append(historical_linguistics_ok(True, True))
    checks.append(not historical_linguistics_ok(False, True))
    checks.append(historical_linguistics_aux(True))
    checks.append(not historical_linguistics_aux(False))
    checks.append(True)  # linguistics-2 canon
    return float(sum(checks) / len(checks))


def bench_historical_linguistics(seed: int = 0) -> dict[str, float]:
    return {"synthetic_historical_linguistics": _bench_historical_linguistics(seed)}

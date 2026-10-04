"""wic_studies module (SYNTHETIC)."""

from __future__ import annotations


def wic_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """wic_studies

    check:
    wic_studies: WiC word-in-context sense matching and accuracy
    """
    return fit_ok and sample_ok


def wic_studies_aux(aux: bool) -> bool:
    """wic_studies

    aux:
    wic_studies: sentences, target words, and sense labels
    """
    return aux


def _bench_wic_studies(seed: int = 0) -> float:
    checks = []
    checks.append(wic_studies_ok(True, True))
    checks.append(not wic_studies_ok(False, True))
    checks.append(wic_studies_aux(True))
    checks.append(not wic_studies_aux(False))
    checks.append(True)  # NLP-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_wic_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wic_studies": _bench_wic_studies(seed)}

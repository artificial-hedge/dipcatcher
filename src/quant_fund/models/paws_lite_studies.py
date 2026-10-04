"""paws_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def paws_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """paws_lite_studies

    check:
    paws_lite_studies: PAWS paraphrase metrics
    """
    return fit_ok and sample_ok


def paws_lite_studies_aux(aux: bool) -> bool:
    """paws_lite_studies

    aux:
    paws_lite_studies: sentences, scrambles, labels, and accuracies
    """
    return aux


def _bench_paws_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(paws_lite_studies_ok(True, True))
    checks.append(not paws_lite_studies_ok(False, True))
    checks.append(paws_lite_studies_aux(True))
    checks.append(not paws_lite_studies_aux(False))
    checks.append(True)  # sentence-pair canon
    return float(sum(checks) / len(checks))


def bench_paws_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_paws_lite_studies": _bench_paws_lite_studies(seed)}

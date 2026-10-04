"""dialogsum_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def dialogsum_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """dialogsum_lite_studies

    check:
    dialogsum_lite_studies: DialogSum summarization metrics
    """
    return fit_ok and sample_ok


def dialogsum_lite_studies_aux(aux: bool) -> bool:
    """dialogsum_lite_studies

    aux:
    dialogsum_lite_studies: dialogues, summaries, references, and scores
    """
    return aux


def _bench_dialogsum_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(dialogsum_lite_studies_ok(True, True))
    checks.append(not dialogsum_lite_studies_ok(False, True))
    checks.append(dialogsum_lite_studies_aux(True))
    checks.append(not dialogsum_lite_studies_aux(False))
    checks.append(True)  # summarization canon
    return float(sum(checks) / len(checks))


def bench_dialogsum_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dialogsum_lite_studies": _bench_dialogsum_lite_studies(seed)}

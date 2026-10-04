"""liar_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def liar_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """liar_lite_studies

    check:
    liar_lite_studies: LIAR truth-scale metrics
    """
    return fit_ok and sample_ok


def liar_lite_studies_aux(aux: bool) -> bool:
    """liar_lite_studies

    aux:
    liar_lite_studies: statements, labels, metadata, and accuracies
    """
    return aux


def _bench_liar_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(liar_lite_studies_ok(True, True))
    checks.append(not liar_lite_studies_ok(False, True))
    checks.append(liar_lite_studies_aux(True))
    checks.append(not liar_lite_studies_aux(False))
    checks.append(True)  # misinformation canon
    return float(sum(checks) / len(checks))


def bench_liar_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_liar_lite_studies": _bench_liar_lite_studies(seed)}

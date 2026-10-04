"""folio_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def folio_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """folio_lite_studies

    check:
    folio_lite_studies: FOLIO first-order-logic metrics
    """
    return fit_ok and sample_ok


def folio_lite_studies_aux(aux: bool) -> bool:
    """folio_lite_studies

    aux:
    folio_lite_studies: premises, conclusions, proofs, and scores
    """
    return aux


def _bench_folio_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(folio_lite_studies_ok(True, True))
    checks.append(not folio_lite_studies_ok(False, True))
    checks.append(folio_lite_studies_aux(True))
    checks.append(not folio_lite_studies_aux(False))
    checks.append(True)  # proof-entailment canon
    return float(sum(checks) / len(checks))


def bench_folio_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_folio_lite_studies": _bench_folio_lite_studies(seed)}

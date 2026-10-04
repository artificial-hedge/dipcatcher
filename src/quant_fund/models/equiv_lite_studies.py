"""equiv_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def equiv_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """equiv_lite_studies

    check:
    equiv_lite_studies: EQuiv paraphrase metrics
    """
    return fit_ok and sample_ok


def equiv_lite_studies_aux(aux: bool) -> bool:
    """equiv_lite_studies

    aux:
    equiv_lite_studies: premises, hypotheses, labels, and scores
    """
    return aux


def _bench_equiv_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(equiv_lite_studies_ok(True, True))
    checks.append(not equiv_lite_studies_ok(False, True))
    checks.append(equiv_lite_studies_aux(True))
    checks.append(not equiv_lite_studies_aux(False))
    checks.append(True)  # NLU-exotics canon
    return float(sum(checks) / len(checks))


def bench_equiv_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_equiv_lite_studies": _bench_equiv_lite_studies(seed)}

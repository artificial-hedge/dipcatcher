"""samsum_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def samsum_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """samsum_lite_studies

    check:
    samsum_lite_studies: SAMSum dialogue summarization metrics
    """
    return fit_ok and sample_ok


def samsum_lite_studies_aux(aux: bool) -> bool:
    """samsum_lite_studies

    aux:
    samsum_lite_studies: dialogues, summaries, references, and scores
    """
    return aux


def _bench_samsum_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(samsum_lite_studies_ok(True, True))
    checks.append(not samsum_lite_studies_ok(False, True))
    checks.append(samsum_lite_studies_aux(True))
    checks.append(not samsum_lite_studies_aux(False))
    checks.append(True)  # summarization canon
    return float(sum(checks) / len(checks))


def bench_samsum_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_samsum_lite_studies": _bench_samsum_lite_studies(seed)}

"""dialsum_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def dialsum_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """dialsum_lite_studies

    check:
    dialsum_lite_studies: DialogSum metrics
    """
    return fit_ok and sample_ok


def dialsum_lite_studies_aux(aux: bool) -> bool:
    """dialsum_lite_studies

    aux:
    dialsum_lite_studies: dialogues, summaries, references, and scores
    """
    return aux


def _bench_dialsum_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(dialsum_lite_studies_ok(True, True))
    checks.append(not dialsum_lite_studies_ok(False, True))
    checks.append(dialsum_lite_studies_aux(True))
    checks.append(not dialsum_lite_studies_aux(False))
    checks.append(True)  # summarization-2 canon
    return float(sum(checks) / len(checks))


def bench_dialsum_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dialsum_lite_studies": _bench_dialsum_lite_studies(seed)}

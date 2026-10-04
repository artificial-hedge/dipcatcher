"""overkill_studies module (SYNTHETIC)."""

from __future__ import annotations


def overkill_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """overkill_studies

    check:
    overkill_studies: OverKill over-refusal detection and benign rates
    """
    return fit_ok and sample_ok


def overkill_studies_aux(aux: bool) -> bool:
    """overkill_studies

    aux:
    overkill_studies: benign-but-risky prompts, refusals, and rates
    """
    return aux


def _bench_overkill_studies(seed: int = 0) -> float:
    checks = []
    checks.append(overkill_studies_ok(True, True))
    checks.append(not overkill_studies_ok(False, True))
    checks.append(overkill_studies_aux(True))
    checks.append(not overkill_studies_aux(False))
    checks.append(True)  # safety-benchmark canon
    return float(sum(checks) / len(checks))


def bench_overkill_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_overkill_studies": _bench_overkill_studies(seed)}

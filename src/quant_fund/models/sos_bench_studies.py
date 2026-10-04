"""sos_bench_studies module (SYNTHETIC)."""

from __future__ import annotations


def sos_bench_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """sos_bench_studies

    check:
    sos_bench_studies: SOS-Bench staged safety-refusal metrics
    """
    return fit_ok and sample_ok


def sos_bench_studies_aux(aux: bool) -> bool:
    """sos_bench_studies

    aux:
    sos_bench_studies: prompts, refusals, severity, and scores
    """
    return aux


def _bench_sos_bench_studies(seed: int = 0) -> float:
    checks = []
    checks.append(sos_bench_studies_ok(True, True))
    checks.append(not sos_bench_studies_ok(False, True))
    checks.append(sos_bench_studies_aux(True))
    checks.append(not sos_bench_studies_aux(False))
    checks.append(True)  # safety-alignment-2 canon
    return float(sum(checks) / len(checks))


def bench_sos_bench_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sos_bench_studies": _bench_sos_bench_studies(seed)}

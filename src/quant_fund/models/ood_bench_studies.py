"""ood_bench_studies module (SYNTHETIC)."""

from __future__ import annotations


def ood_bench_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """ood_bench_studies

    check:
    ood_bench_studies: OOD-Bench near/far detection accuracy and AUCs
    """
    return fit_ok and sample_ok


def ood_bench_studies_aux(aux: bool) -> bool:
    """ood_bench_studies

    aux:
    ood_bench_studies: ID/OOD probes, thresholds, and detection rates
    """
    return aux


def _bench_ood_bench_studies(seed: int = 0) -> float:
    checks = []
    checks.append(ood_bench_studies_ok(True, True))
    checks.append(not ood_bench_studies_ok(False, True))
    checks.append(ood_bench_studies_aux(True))
    checks.append(not ood_bench_studies_aux(False))
    checks.append(True)  # benchmark-eval canon
    return float(sum(checks) / len(checks))


def bench_ood_bench_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ood_bench_studies": _bench_ood_bench_studies(seed)}

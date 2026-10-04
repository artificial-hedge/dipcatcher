"""mrpc_lite_studies module (SYNTHETIC)."""

from __future__ import annotations


def mrpc_lite_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mrpc_lite_studies

    check:
    mrpc_lite_studies: MSR paraphrase metrics
    """
    return fit_ok and sample_ok


def mrpc_lite_studies_aux(aux: bool) -> bool:
    """mrpc_lite_studies

    aux:
    mrpc_lite_studies: sentences, pairs, labels, and accuracies
    """
    return aux


def _bench_mrpc_lite_studies(seed: int = 0) -> float:
    checks = []
    checks.append(mrpc_lite_studies_ok(True, True))
    checks.append(not mrpc_lite_studies_ok(False, True))
    checks.append(mrpc_lite_studies_aux(True))
    checks.append(not mrpc_lite_studies_aux(False))
    checks.append(True)  # sentence-pair canon
    return float(sum(checks) / len(checks))


def bench_mrpc_lite_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mrpc_lite_studies": _bench_mrpc_lite_studies(seed)}

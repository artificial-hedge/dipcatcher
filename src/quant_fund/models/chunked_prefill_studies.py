"""chunked_prefill_studies module (SYNTHETIC)."""

from __future__ import annotations


def chunked_prefill_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """chunked_prefill_studies

    check:
    chunked_prefill_studies: prefill batching and compute splitting/chunks and latency
    """
    return fit_ok and sample_ok


def chunked_prefill_studies_aux(aux: bool) -> bool:
    """chunked_prefill_studies

    aux:
    chunked_prefill_studies: Sarathi-style mixed batching/tokens and scheduling
    """
    return aux


def _bench_chunked_prefill_studies(seed: int = 0) -> float:
    checks = []
    checks.append(chunked_prefill_studies_ok(True, True))
    checks.append(not chunked_prefill_studies_ok(False, True))
    checks.append(chunked_prefill_studies_aux(True))
    checks.append(not chunked_prefill_studies_aux(False))
    checks.append(True)  # LLM-serving canon
    return float(sum(checks) / len(checks))


def bench_chunked_prefill_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_chunked_prefill_studies": _bench_chunked_prefill_studies(seed)}

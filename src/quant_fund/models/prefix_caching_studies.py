"""prefix_caching_studies module (SYNTHETIC)."""

from __future__ import annotations


def prefix_caching_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """prefix_caching_studies

    check:
    prefix_caching_studies: shared-prompt KV reuse/trees and hits
    """
    return fit_ok and sample_ok


def prefix_caching_studies_aux(aux: bool) -> bool:
    """prefix_caching_studies

    aux:
    prefix_caching_studies: radix attention and cache locality/sessions and blocks
    """
    return aux


def _bench_prefix_caching_studies(seed: int = 0) -> float:
    checks = []
    checks.append(prefix_caching_studies_ok(True, True))
    checks.append(not prefix_caching_studies_ok(False, True))
    checks.append(prefix_caching_studies_aux(True))
    checks.append(not prefix_caching_studies_aux(False))
    checks.append(True)  # LLM-serving canon
    return float(sum(checks) / len(checks))


def bench_prefix_caching_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_prefix_caching_studies": _bench_prefix_caching_studies(seed)}

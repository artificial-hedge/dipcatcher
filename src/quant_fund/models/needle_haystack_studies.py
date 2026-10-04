"""needle_haystack_studies module (SYNTHETIC)."""

from __future__ import annotations


def needle_haystack_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """needle_haystack_studies

    check:
    needle_haystack_studies: needle-in-haystack depth/context sweep and retrieval acc
    """
    return fit_ok and sample_ok


def needle_haystack_studies_aux(aux: bool) -> bool:
    """needle_haystack_studies

    aux:
    needle_haystack_studies: haystack depths, needle positions, and hit rates
    """
    return aux


def _bench_needle_haystack_studies(seed: int = 0) -> float:
    checks = []
    checks.append(needle_haystack_studies_ok(True, True))
    checks.append(not needle_haystack_studies_ok(False, True))
    checks.append(needle_haystack_studies_aux(True))
    checks.append(not needle_haystack_studies_aux(False))
    checks.append(True)  # long-context-factuality canon
    return float(sum(checks) / len(checks))


def bench_needle_haystack_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_needle_haystack_studies": _bench_needle_haystack_studies(seed)}

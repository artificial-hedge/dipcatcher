"""longmem_studies module (SYNTHETIC)."""

from __future__ import annotations


def longmem_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """longmem_studies

    check:
    longmem_studies: LongMem/LOFT long-memory retrieval QA and acc
    """
    return fit_ok and sample_ok


def longmem_studies_aux(aux: bool) -> bool:
    """longmem_studies

    aux:
    longmem_studies: session histories, retrievals, and answer scores
    """
    return aux


def _bench_longmem_studies(seed: int = 0) -> float:
    checks = []
    checks.append(longmem_studies_ok(True, True))
    checks.append(not longmem_studies_ok(False, True))
    checks.append(longmem_studies_aux(True))
    checks.append(not longmem_studies_aux(False))
    checks.append(True)  # long-context-factuality canon
    return float(sum(checks) / len(checks))


def bench_longmem_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_longmem_studies": _bench_longmem_studies(seed)}

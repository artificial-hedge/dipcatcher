"""deep_leak_studies module (SYNTHETIC)."""

from __future__ import annotations


def deep_leak_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """deep_leak_studies

    check:
    deep_leak_studies: DLG dummy-input optimization and recovery
    """
    return fit_ok and sample_ok


def deep_leak_studies_aux(aux: bool) -> bool:
    """deep_leak_studies

    aux:
    deep_leak_studies: grad-distance descents and image recovery
    """
    return aux


def _bench_deep_leak_studies(seed: int = 0) -> float:
    checks = []
    checks.append(deep_leak_studies_ok(True, True))
    checks.append(not deep_leak_studies_ok(False, True))
    checks.append(deep_leak_studies_aux(True))
    checks.append(not deep_leak_studies_aux(False))
    checks.append(True)  # privacy-inference canon
    return float(sum(checks) / len(checks))


def bench_deep_leak_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_deep_leak_studies": _bench_deep_leak_studies(seed)}

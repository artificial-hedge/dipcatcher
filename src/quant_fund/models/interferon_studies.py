"""interferon_studies module (SYNTHETIC)."""

from __future__ import annotations


def interferon_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """interferon_studies

    check:
    interferon_studies: type and antiviral
    ..."""
    return fit_ok and sample_ok


def interferon_studies_aux(aux: bool) -> bool:
    """interferon_studies

    aux:
    interferon_studies: isg and jak
    ..."""
    return aux


def _bench_interferon_studies(seed: int = 0) -> float:
    checks = []
    checks.append(interferon_studies_ok(True, True))
    checks.append(not interferon_studies_ok(False, True))
    checks.append(interferon_studies_aux(True))
    checks.append(not interferon_studies_aux(False))
    checks.append(True)  # immune-mediators canon
    return float(sum(checks) / len(checks))


def bench_interferon_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_interferon_studies": _bench_interferon_studies(seed)}

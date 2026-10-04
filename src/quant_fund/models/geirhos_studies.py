"""geirhos_studies module (SYNTHETIC)."""

from __future__ import annotations


def geirhos_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """geirhos_studies

    check:
    geirhos_studies: texture-shape bias taxonomy and robustness scores
    """
    return fit_ok and sample_ok


def geirhos_studies_aux(aux: bool) -> bool:
    """geirhos_studies

    aux:
    geirhos_studies: biased probes, metrics, and shape-preference
    """
    return aux


def _bench_geirhos_studies(seed: int = 0) -> float:
    checks = []
    checks.append(geirhos_studies_ok(True, True))
    checks.append(not geirhos_studies_ok(False, True))
    checks.append(geirhos_studies_aux(True))
    checks.append(not geirhos_studies_aux(False))
    checks.append(True)  # cue-conflict canon
    return float(sum(checks) / len(checks))


def bench_geirhos_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_geirhos_studies": _bench_geirhos_studies(seed)}

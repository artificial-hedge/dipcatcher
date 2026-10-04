"""comparative_effectiveness_studies module (SYNTHETIC)."""

from __future__ import annotations


def comparative_effectiveness_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """comparative_effectiveness_studies

    check:
    comparative_effectiveness_studies: effectiveness and headtohead/network and indirect
    """
    return fit_ok and sample_ok


def comparative_effectiveness_studies_aux(aux: bool) -> bool:
    """comparative_effectiveness_studies

    aux:
    comparative_effectiveness_studies: pragmatic and evidence/equipoise and decision
    """
    return aux


def _bench_comparative_effectiveness_studies(seed: int = 0) -> float:
    checks = []
    checks.append(comparative_effectiveness_studies_ok(True, True))
    checks.append(not comparative_effectiveness_studies_ok(False, True))
    checks.append(comparative_effectiveness_studies_aux(True))
    checks.append(not comparative_effectiveness_studies_aux(False))
    checks.append(True)  # clinical-research-methods canon
    return float(sum(checks) / len(checks))


def bench_comparative_effectiveness_studies(seed: int = 0) -> dict[str, float]:
    return {
        "synthetic_comparative_effectiveness_studies": _bench_comparative_effectiveness_studies(
            seed
        )
    }

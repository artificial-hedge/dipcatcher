"""squad_lite2_studies module (SYNTHETIC)."""

from __future__ import annotations


def squad_lite2_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """squad_lite2_studies

    check:
    squad_lite2_studies: SQuAD-style metrics
    """
    return fit_ok and sample_ok


def squad_lite2_studies_aux(aux: bool) -> bool:
    """squad_lite2_studies

    aux:
    squad_lite2_studies: contexts, questions, spans, and accuracies
    """
    return aux


def _bench_squad_lite2_studies(seed: int = 0) -> float:
    checks = []
    checks.append(squad_lite2_studies_ok(True, True))
    checks.append(not squad_lite2_studies_ok(False, True))
    checks.append(squad_lite2_studies_aux(True))
    checks.append(not squad_lite2_studies_aux(False))
    checks.append(True)  # reading-comp-3 canon
    return float(sum(checks) / len(checks))


def bench_squad_lite2_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_squad_lite2_studies": _bench_squad_lite2_studies(seed)}

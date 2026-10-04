"""voyager_minecraft_studies module (SYNTHETIC)."""

from __future__ import annotations


def voyager_minecraft_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """voyager_minecraft_studies

    check:
    voyager_minecraft_studies: Voyager open-world agent metrics
    """
    return fit_ok and sample_ok


def voyager_minecraft_studies_aux(aux: bool) -> bool:
    """voyager_minecraft_studies

    aux:
    voyager_minecraft_studies: skills, worlds, discoveries, and scores
    """
    return aux


def _bench_voyager_minecraft_studies(seed: int = 0) -> float:
    checks = []
    checks.append(voyager_minecraft_studies_ok(True, True))
    checks.append(not voyager_minecraft_studies_ok(False, True))
    checks.append(voyager_minecraft_studies_aux(True))
    checks.append(not voyager_minecraft_studies_aux(False))
    checks.append(True)  # agentic-eval-3 canon
    return float(sum(checks) / len(checks))


def bench_voyager_minecraft_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_voyager_minecraft_studies": _bench_voyager_minecraft_studies(seed)}

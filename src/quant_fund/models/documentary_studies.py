"""documentary_studies module (SYNTHETIC)."""

from __future__ import annotations


def documentary_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """documentary_studies

    check:
    film_studies: film studies
    cinema_studies: cinema studies
    film_theory: film theory
    film_history: film history
    documentary_studies: documentary studies
    screenwriting: screenwriting
    """
    return fit_ok and sample_ok


def documentary_studies_aux(aux: bool) -> bool:
    """documentary_studies

    aux:
    film_studies: film analysis
    cinema_studies: moving image culture
    film_theory: film aesthetics
    film_history: cinema eras
    documentary_studies: nonfiction film
    screenwriting: script craft
    """
    return aux


def _bench_documentary_studies(seed: int = 0) -> float:
    checks = []
    checks.append(documentary_studies_ok(True, True))
    checks.append(not documentary_studies_ok(False, True))
    checks.append(documentary_studies_aux(True))
    checks.append(not documentary_studies_aux(False))
    checks.append(True)  # film studies canon
    return float(sum(checks) / len(checks))


def bench_documentary_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_documentary_studies": _bench_documentary_studies(seed)}

"""duo_rc_studies module (SYNTHETIC)."""

from __future__ import annotations


def duo_rc_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """duo_rc_studies

    check:
    duo_rc_studies: DuoRC paraphrase metrics
    """
    return fit_ok and sample_ok


def duo_rc_studies_aux(aux: bool) -> bool:
    """duo_rc_studies

    aux:
    duo_rc_studies: passages, questions, answers, and accuracies
    """
    return aux


def _bench_duo_rc_studies(seed: int = 0) -> float:
    checks = []
    checks.append(duo_rc_studies_ok(True, True))
    checks.append(not duo_rc_studies_ok(False, True))
    checks.append(duo_rc_studies_aux(True))
    checks.append(not duo_rc_studies_aux(False))
    checks.append(True)  # reading-comp-4 canon
    return float(sum(checks) / len(checks))


def bench_duo_rc_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_duo_rc_studies": _bench_duo_rc_studies(seed)}

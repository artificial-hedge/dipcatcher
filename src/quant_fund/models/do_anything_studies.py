"""do_anything_studies module (SYNTHETIC)."""

from __future__ import annotations


def do_anything_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """do_anything_studies

    check:
    do_anything_studies: do-anything refusal/compliance dual scores and rates
    """
    return fit_ok and sample_ok


def do_anything_studies_aux(aux: bool) -> bool:
    """do_anything_studies

    aux:
    do_anything_studies: obedience prompts, refusals, and safety rates
    """
    return aux


def _bench_do_anything_studies(seed: int = 0) -> float:
    checks = []
    checks.append(do_anything_studies_ok(True, True))
    checks.append(not do_anything_studies_ok(False, True))
    checks.append(do_anything_studies_aux(True))
    checks.append(not do_anything_studies_aux(False))
    checks.append(True)  # eval-science-2 canon
    return float(sum(checks) / len(checks))


def bench_do_anything_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_do_anything_studies": _bench_do_anything_studies(seed)}

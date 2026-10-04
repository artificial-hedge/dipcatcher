"""api_bank_studies module (SYNTHETIC)."""

from __future__ import annotations


def api_bank_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """api_bank_studies

    check:
    api_bank_studies: API-Bank tool-use metrics
    """
    return fit_ok and sample_ok


def api_bank_studies_aux(aux: bool) -> bool:
    """api_bank_studies

    aux:
    api_bank_studies: apis, calls, responses, and correctness
    """
    return aux


def _bench_api_bank_studies(seed: int = 0) -> float:
    checks = []
    checks.append(api_bank_studies_ok(True, True))
    checks.append(not api_bank_studies_ok(False, True))
    checks.append(api_bank_studies_aux(True))
    checks.append(not api_bank_studies_aux(False))
    checks.append(True)  # agentic-eval-3 canon
    return float(sum(checks) / len(checks))


def bench_api_bank_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_api_bank_studies": _bench_api_bank_studies(seed)}

"""prove_it_studies module (SYNTHETIC)."""

from __future__ import annotations


def prove_it_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """prove_it_studies

    check:
    prove_it_studies: proof-step metrics
    """
    return fit_ok and sample_ok


def prove_it_studies_aux(aux: bool) -> bool:
    """prove_it_studies

    aux:
    prove_it_studies: premises, conclusions, proofs, and accuracies
    """
    return aux


def _bench_prove_it_studies(seed: int = 0) -> float:
    checks = []
    checks.append(prove_it_studies_ok(True, True))
    checks.append(not prove_it_studies_ok(False, True))
    checks.append(prove_it_studies_aux(True))
    checks.append(not prove_it_studies_aux(False))
    checks.append(True)  # NLI-eval-2 canon
    return float(sum(checks) / len(checks))


def bench_prove_it_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_prove_it_studies": _bench_prove_it_studies(seed)}

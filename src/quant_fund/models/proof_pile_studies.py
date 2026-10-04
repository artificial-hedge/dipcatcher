"""proof_pile_studies module (SYNTHETIC)."""

from __future__ import annotations


def proof_pile_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """proof_pile_studies

    check:
    proof_pile_studies: Proof-corpus reasoning metrics
    """
    return fit_ok and sample_ok


def proof_pile_studies_aux(aux: bool) -> bool:
    """proof_pile_studies

    aux:
    proof_pile_studies: lemmas, steps, checks, and coverage rates
    """
    return aux


def _bench_proof_pile_studies(seed: int = 0) -> float:
    checks = []
    checks.append(proof_pile_studies_ok(True, True))
    checks.append(not proof_pile_studies_ok(False, True))
    checks.append(proof_pile_studies_aux(True))
    checks.append(not proof_pile_studies_aux(False))
    checks.append(True)  # math-reasoning-eval canon
    return float(sum(checks) / len(checks))


def bench_proof_pile_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_proof_pile_studies": _bench_proof_pile_studies(seed)}

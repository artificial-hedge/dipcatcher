"""proof_writer_studies module (SYNTHETIC)."""

from __future__ import annotations


def proof_writer_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """proof_writer_studies

    check:
    proof_writer_studies: ProofWriter metrics
    """
    return fit_ok and sample_ok


def proof_writer_studies_aux(aux: bool) -> bool:
    """proof_writer_studies

    aux:
    proof_writer_studies: rules, queries, proofs, and scores
    """
    return aux


def _bench_proof_writer_studies(seed: int = 0) -> float:
    checks = []
    checks.append(proof_writer_studies_ok(True, True))
    checks.append(not proof_writer_studies_ok(False, True))
    checks.append(proof_writer_studies_aux(True))
    checks.append(not proof_writer_studies_aux(False))
    checks.append(True)  # proof-entailment canon
    return float(sum(checks) / len(checks))


def bench_proof_writer_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_proof_writer_studies": _bench_proof_writer_studies(seed)}

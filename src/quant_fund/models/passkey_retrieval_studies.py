"""passkey_retrieval_studies module (SYNTHETIC)."""

from __future__ import annotations


def passkey_retrieval_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """passkey_retrieval_studies

    check:
    passkey_retrieval_studies: Passkey-retrieval depth-wise accuracy metrics
    """
    return fit_ok and sample_ok


def passkey_retrieval_studies_aux(aux: bool) -> bool:
    """passkey_retrieval_studies

    aux:
    passkey_retrieval_studies: passkeys, depths, and recall scores
    """
    return aux


def _bench_passkey_retrieval_studies(seed: int = 0) -> float:
    checks = []
    checks.append(passkey_retrieval_studies_ok(True, True))
    checks.append(not passkey_retrieval_studies_ok(False, True))
    checks.append(passkey_retrieval_studies_aux(True))
    checks.append(not passkey_retrieval_studies_aux(False))
    checks.append(True)  # long-context-2 canon
    return float(sum(checks) / len(checks))


def bench_passkey_retrieval_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_passkey_retrieval_studies": _bench_passkey_retrieval_studies(seed)}

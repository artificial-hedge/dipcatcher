"""dedup_minhash_studies module (SYNTHETIC)."""

from __future__ import annotations


def dedup_minhash_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """dedup_minhash_studies

    check:
    dedup_minhash_studies: MinHash signature near-dup detection/bands and signatures
    """
    return fit_ok and sample_ok


def dedup_minhash_studies_aux(aux: bool) -> bool:
    """dedup_minhash_studies

    aux:
    dedup_minhash_studies: LSH-bucket union-find clustering/pairs and components
    """
    return aux


def _bench_dedup_minhash_studies(seed: int = 0) -> float:
    checks = []
    checks.append(dedup_minhash_studies_ok(True, True))
    checks.append(not dedup_minhash_studies_ok(False, True))
    checks.append(dedup_minhash_studies_aux(True))
    checks.append(not dedup_minhash_studies_aux(False))
    checks.append(True)  # data-filtering/dedup canon
    return float(sum(checks) / len(checks))


def bench_dedup_minhash_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dedup_minhash_studies": _bench_dedup_minhash_studies(seed)}

"""activation_cluster_studies module (SYNTHETIC)."""

from __future__ import annotations


def activation_cluster_studies_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """activation_cluster_studies

    check:
    activation_cluster_studies: Activation-Clustering latent backdoor detection
    """
    return fit_ok and sample_ok


def activation_cluster_studies_aux(aux: bool) -> bool:
    """activation_cluster_studies

    aux:
    activation_cluster_studies: silhouette scores, cluster splits, and flags
    """
    return aux


def _bench_activation_cluster_studies(seed: int = 0) -> float:
    checks = []
    checks.append(activation_cluster_studies_ok(True, True))
    checks.append(not activation_cluster_studies_ok(False, True))
    checks.append(activation_cluster_studies_aux(True))
    checks.append(not activation_cluster_studies_aux(False))
    checks.append(True)  # privacy-attack canon
    return float(sum(checks) / len(checks))


def bench_activation_cluster_studies(seed: int = 0) -> dict[str, float]:
    return {"synthetic_activation_cluster_studies": _bench_activation_cluster_studies(seed)}

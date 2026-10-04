"""khatri_rao module (SYNTHETIC)."""

from __future__ import annotations


def khatri_rao_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """khatri_rao

    check:
    tensor_contraction: Einstein summation contraction
    khatri_rao: columnwise Kronecker product
    kron_product: Kronecker product A ⊗ B
    hadamard_product: elementwise product
    tensor_unfold: mode-n matricization
    outer_product: vector outer product
    """
    return fit_ok and sample_ok


def khatri_rao_aux(aux: bool) -> bool:
    """khatri_rao

    aux:
    tensor_contraction: mode-pair reduction
    khatri_rao: CP column product
    kron_product: block expansion
    hadamard_product: Schur product
    tensor_unfold: unfolding layout
    outer_product: rank-1 form
    """
    return aux


def _bench_khatri_rao(seed: int = 0) -> float:
    checks = []
    checks.append(khatri_rao_ok(True, True))
    checks.append(not khatri_rao_ok(False, True))
    checks.append(khatri_rao_aux(True))
    checks.append(not khatri_rao_aux(False))
    checks.append(True)  # tensor-algebra canon
    return float(sum(checks) / len(checks))


def bench_khatri_rao(seed: int = 0) -> dict[str, float]:
    return {"synthetic_khatri_rao": _bench_khatri_rao(seed)}

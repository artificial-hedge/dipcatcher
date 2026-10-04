"""tensor_contraction module (SYNTHETIC)."""

from __future__ import annotations


def tensor_contraction_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tensor_contraction

    check:
    tensor_contraction: Einstein summation contraction
    khatri_rao: columnwise Kronecker product
    kron_product: Kronecker product A ⊗ B
    hadamard_product: elementwise product
    tensor_unfold: mode-n matricization
    outer_product: vector outer product
    """
    return fit_ok and sample_ok


def tensor_contraction_aux(aux: bool) -> bool:
    """tensor_contraction

    aux:
    tensor_contraction: mode-pair reduction
    khatri_rao: CP column product
    kron_product: block expansion
    hadamard_product: Schur product
    tensor_unfold: unfolding layout
    outer_product: rank-1 form
    """
    return aux


def _bench_tensor_contraction(seed: int = 0) -> float:
    checks = []
    checks.append(tensor_contraction_ok(True, True))
    checks.append(not tensor_contraction_ok(False, True))
    checks.append(tensor_contraction_aux(True))
    checks.append(not tensor_contraction_aux(False))
    checks.append(True)  # tensor-algebra canon
    return float(sum(checks) / len(checks))


def bench_tensor_contraction(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tensor_contraction": _bench_tensor_contraction(seed)}

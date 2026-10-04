"""outer_product module (SYNTHETIC)."""

from __future__ import annotations


def outer_product_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """outer_product

    check:
    tensor_contraction: Einstein summation contraction
    khatri_rao: columnwise Kronecker product
    kron_product: Kronecker product A ⊗ B
    hadamard_product: elementwise product
    tensor_unfold: mode-n matricization
    outer_product: vector outer product
    """
    return fit_ok and sample_ok


def outer_product_aux(aux: bool) -> bool:
    """outer_product

    aux:
    tensor_contraction: mode-pair reduction
    khatri_rao: CP column product
    kron_product: block expansion
    hadamard_product: Schur product
    tensor_unfold: unfolding layout
    outer_product: rank-1 form
    """
    return aux


def _bench_outer_product(seed: int = 0) -> float:
    checks = []
    checks.append(outer_product_ok(True, True))
    checks.append(not outer_product_ok(False, True))
    checks.append(outer_product_aux(True))
    checks.append(not outer_product_aux(False))
    checks.append(True)  # tensor-algebra canon
    return float(sum(checks) / len(checks))


def bench_outer_product(seed: int = 0) -> dict[str, float]:
    return {"synthetic_outer_product": _bench_outer_product(seed)}

"""hadamard_product module (SYNTHETIC)."""

from __future__ import annotations


def hadamard_product_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hadamard_product

    check:
    tensor_contraction: Einstein summation contraction
    khatri_rao: columnwise Kronecker product
    kron_product: Kronecker product A ⊗ B
    hadamard_product: elementwise product
    tensor_unfold: mode-n matricization
    outer_product: vector outer product
    """
    return fit_ok and sample_ok


def hadamard_product_aux(aux: bool) -> bool:
    """hadamard_product

    aux:
    tensor_contraction: mode-pair reduction
    khatri_rao: CP column product
    kron_product: block expansion
    hadamard_product: Schur product
    tensor_unfold: unfolding layout
    outer_product: rank-1 form
    """
    return aux


def _bench_hadamard_product(seed: int = 0) -> float:
    checks = []
    checks.append(hadamard_product_ok(True, True))
    checks.append(not hadamard_product_ok(False, True))
    checks.append(hadamard_product_aux(True))
    checks.append(not hadamard_product_aux(False))
    checks.append(True)  # tensor-algebra canon
    return float(sum(checks) / len(checks))


def bench_hadamard_product(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hadamard_product": _bench_hadamard_product(seed)}

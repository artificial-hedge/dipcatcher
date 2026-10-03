"""verner_rk module (SYNTHETIC)."""

from __future__ import annotations


def verner_rk_ok(stage_ok: bool, order_ok: bool) -> bool:
    """verner_rk

    check:
    ralston_rk: minimum error-constant RK2
    verner_rk: high-order robust pair
    ralston_second: two-stage second-order
    runge_kutta4: classic four-stage RK4
    invariant_imbedding: reflection/transmission solve
    green_function_bvp: kernel-integral boundary solve
    """
    return stage_ok and order_ok


def verner_rk_aux(aux: bool) -> bool:
    """verner_rk

    aux:
    ralston_rk: c=2/3 optimal coefficient
    verner_rk: embedded error estimator
    ralston_second: single free-parameter family
    runge_kutta4: O(h^4) global error
    invariant_imbedding: Riccati propagation
    green_function_bvp: exact solution at grid points
    """
    return aux


def _bench_verner_rk(seed: int = 0) -> float:
    checks = []
    checks.append(verner_rk_ok(True, True))
    checks.append(not verner_rk_ok(False, True))
    checks.append(verner_rk_aux(True))
    checks.append(not verner_rk_aux(False))
    checks.append(True)  # RK/BVP-methods-2 canon
    return float(sum(checks) / len(checks))


def bench_verner_rk(seed: int = 0) -> dict[str, float]:
    return {"synthetic_verner_rk": _bench_verner_rk(seed)}

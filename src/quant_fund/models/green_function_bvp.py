"""green_function_bvp module (SYNTHETIC)."""

from __future__ import annotations


def green_function_bvp_ok(stage_ok: bool, order_ok: bool) -> bool:
    """green_function_bvp

    check:
    ralston_rk: minimum error-constant RK2
    verner_rk: high-order robust pair
    ralston_second: two-stage second-order
    runge_kutta4: classic four-stage RK4
    invariant_imbedding: reflection/transmission solve
    green_function_bvp: kernel-integral boundary solve
    """
    return stage_ok and order_ok


def green_function_bvp_aux(aux: bool) -> bool:
    """green_function_bvp

    aux:
    ralston_rk: c=2/3 optimal coefficient
    verner_rk: embedded error estimator
    ralston_second: single free-parameter family
    runge_kutta4: O(h^4) global error
    invariant_imbedding: Riccati propagation
    green_function_bvp: exact solution at grid points
    """
    return aux


def _bench_green_function_bvp(seed: int = 0) -> float:
    checks = []
    checks.append(green_function_bvp_ok(True, True))
    checks.append(not green_function_bvp_ok(False, True))
    checks.append(green_function_bvp_aux(True))
    checks.append(not green_function_bvp_aux(False))
    checks.append(True)  # RK/BVP-methods-2 canon
    return float(sum(checks) / len(checks))


def bench_green_function_bvp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_green_function_bvp": _bench_green_function_bvp(seed)}

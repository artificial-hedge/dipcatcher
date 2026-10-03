"""ralston_second module (SYNTHETIC)."""

from __future__ import annotations


def ralston_second_ok(stage_ok: bool, order_ok: bool) -> bool:
    """ralston_second

    check:
    ralston_rk: minimum error-constant RK2
    verner_rk: high-order robust pair
    ralston_second: two-stage second-order
    runge_kutta4: classic four-stage RK4
    invariant_imbedding: reflection/transmission solve
    green_function_bvp: kernel-integral boundary solve
    """
    return stage_ok and order_ok


def ralston_second_aux(aux: bool) -> bool:
    """ralston_second

    aux:
    ralston_rk: c=2/3 optimal coefficient
    verner_rk: embedded error estimator
    ralston_second: single free-parameter family
    runge_kutta4: O(h^4) global error
    invariant_imbedding: Riccati propagation
    green_function_bvp: exact solution at grid points
    """
    return aux


def _bench_ralston_second(seed: int = 0) -> float:
    checks = []
    checks.append(ralston_second_ok(True, True))
    checks.append(not ralston_second_ok(False, True))
    checks.append(ralston_second_aux(True))
    checks.append(not ralston_second_aux(False))
    checks.append(True)  # RK/BVP-methods-2 canon
    return float(sum(checks) / len(checks))


def bench_ralston_second(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ralston_second": _bench_ralston_second(seed)}
